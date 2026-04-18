---
title: Python으로 MLP 바닥부터 구현하기
categories:
  - Pytorch
tags: [torch]
---

> 코딩테스트로 Python으로만 MLP를 구현하는 문제가 나왔던 적이 있습니다. 당시에 역전파 구현을 하지 못해 코딩테스트에서 떨어졌었고 완전히 바닥에서부터 구현해보고자 시작한 프로젝트입니다.

## Multi-Layer Perceptron
Multi-Layer Perceptron(MLP)은 퍼셉트론으로 이루어진 층(layer)들이 쌓여 신경망을 이루는 모델입니다. 구현이 간단하기 때문에 딥러닝을 바닥부터 구현하는 프로젝트를 시작하는데 좋은 모델입니다.
저는 MNIST를 데이터셋으로 하여 모델을 훈련시키고 classification task를 수행해볼 것입니다.

## Linear
Linear 레이어는 affine linear transformation을 수행하는 레이어입니다. Linear 레이어를 fully-connected layer, dense layer 등으로 부르기도 합니다.
Linear 레이어는 weight와 bias를 파라미터로 들고 있습니다. weight는 입력한 데이터가 출력 데이터에 얼마나 영향을 미치는지 결정하는 값이고 bias는 함수의 이동을 도와 선형조합만으로 표현할 수 없는 것을 학습합니다.

### Forward
Linear의 수식은 간단합니다.

$y = xW + b$

```python
import numpy as np
from typing_extensions import Tuple
from numpy.typing import NDArray

from hcrot.layers import Module

class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.zeros((self.in_features, self.out_features))
        self.bias = np.zeros((1, self.out_features))
        self.X = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        sqrt_k = np.sqrt(1 / self.in_features)
        for key in self.param_names:
            setattr(self, key, np.random.uniform(-sqrt_k, sqrt_k, getattr(self,key).shape))

    def forward(self, x: NDArray) -> NDArray:
        # x: (bsz, ..., in_features)
        # weight: (in_features, out_features)
        # bias: (1, out_features)
        self.X = x
        mat = np.matmul(x, self.weight) # (bsz, ..., out_features)
        return mat + self.bias # (bsz, ..., out_features)
```

### Backward
먼저, forward 수식에 대해 편미분을 하여 다음의 수식들을 얻을 수 있습니다.

$\frac{\partial y}{\partial x} = W$

$\frac{\partial y}{\partial W} = x$

$\frac{\partial y}{\partial b} = 1$

다음, chain rule에 의해 x, W, b에 대한 기울기를 구할 수 있습니다.

$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}$

$\frac{\partial L}{\partial b} = \sum \frac{\partial L}{\partial y}$

$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}$

출력 레이어로부터 오는 upstream gradient를 $dz$라 가정하여 수식을 정리합니다.

$\frac{\partial L}{\partial W} = x^\top dz$

$\frac{\partial L}{\partial b} = \sum dz$

$\frac{\partial L}{\partial x} = dzW^\top$

```python
def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    dw = np.matmul(self.X.swapaxes(-1,-2), dz)
    if dw.ndim == 3:
        dw = np.sum(dw, axis=0)
    db = np.sum(dz, axis=tuple(np.arange(len(dz.shape))[:-1]))
    dx = np.matmul(dz, self.weight.swapaxes(-1,-2))
    return dx, dw, db
```

## Sigmoid
Sigmoid는 입력값을 0과 1사이의 값으로 변환하는 활성화 함수입니다. 매우 큰 값은 1로 근사하고 매우 작은 값은 0으로 근사하는 특징을 가집니다.

Sigmoid와 같은 활성화함수들은 비선형성(Non-linear)을 가지고 있어 딥러닝 모델에 비선형성을 추가하여 복잡한 데이터를 학습하는데 도와줄 수 있습니다. Sigmoid의 또다른 특징은 element-wise한 점입니다. 입력 각각에 sigmoid 연산이 적용되어 각 연산들이 independent합니다. 따라서, 역전파 시 통과되는 연산도 independent하게 됩니다.

### Forward
Sigmoid의 수식은 다음과 같습니다.

$\sigma(x) = \frac{1}{1 + e^{-x}}$

```python
class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: NDArray) -> NDArray:
        self.X = x
        return 1/(1+np.exp(-x))
```

### Backward
Sigmoid를 미분하게 되면 다음과 같습니다.

$\frac{\partial \sigma}{\partial x} = \sigma(x)(1 - \sigma(x))$

위에서 설명한대로 backward시 element-wise 연산이 필요하므로 numpy.multiply를 사용해야 합니다.

```python
def backward(self, dz: NDArray) -> NDArray:
    x = self.forward(self.X)
    return x * (1 - x) * dz
```

## 결과
MNIST 5000장을 훈련데이터로 사용하고 1000장을 테스트데이터로 사용했습니다.

```python
class Model(layers.Module):
    def __init__(self, input_len=28*28, hidden=512, num_classes=10):
        super().__init__()
        self.net1 = layers.Sequential(
            layers.Linear(in_features=input_len, out_features=hidden),
            layers.Sigmoid()
            )
        self.dropout = layers.Dropout(p=0.3)
        self.net2 = layers.Sequential(
            layers.Linear(in_features=hidden, out_features=hidden),
            layers.Sigmoid()
            )
        self.fc = layers.Linear(in_features=hidden, out_features=num_classes)
        
    def forward(self, x):
        o = self.dropout(self.net1(x))
        return self.fc(self.net2(o))
```

모델은 Linear -> Sigmoid -> Dropout(0.3) -> Linear -> Sigmoid -> Linear으로 이어지도록 구현했습니다.

<figure class="half">
  <a href="https://onedrive.live.com/embed?resid=502FD124B305BA80%213207&authkey=%21AHbDw6fwQLOIC2Y&width=608&height=604" data-lightbox="gallery">
    <img src="https://onedrive.live.com/embed?resid=502FD124B305BA80%213207&authkey=%21AHbDw6fwQLOIC2Y&width=608&height=604" alt="01">
  </a>
  <a href="https://onedrive.live.com/embed?resid=502FD124B305BA80%213206&authkey=%21ANIblIieb6OZcuE&width=601&height=604" data-lightbox="gallery">
    <img src="https://onedrive.live.com/embed?resid=502FD124B305BA80%213206&authkey=%21ANIblIieb6OZcuE&width=601&height=604" alt="02">
  </a>
</figure>

10 에포크에도 loss가 잘 떨어지고 Accuracy도 잘 증가하는 것을 볼 수 있습니다. 

## 코드
- [https://github.com/emeraldgoose/hcrot](https://github.com/emeraldgoose/hcrot)
- [mlp.ipynb](https://github.com/emeraldgoose/hcrot/blob/master/notebooks/mlp.ipynb)

## Reference
- [Sigmoid 함수 미분 정리](http://taewan.kim/post/sigmoid_diff/)
- [Softmax-with-Loss 계층](https://ratsgo.github.io/deep%20learning/2017/10/02/softmax/)
- [Linear - Pytorch Docs](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [Softmax-with-Loss 계층 계산그래프](https://velog.io/@gjtang/Softmax-with-Loss-%EA%B3%84%EC%B8%B5-%EA%B3%84%EC%82%B0%EA%B7%B8%EB%9E%98%ED%94%84)
- [Activation Functions and Their Gradients](https://aew61.github.io/blog/artificial_neural_networks/1_background/1.b_activation_functions_and_derivatives.html)