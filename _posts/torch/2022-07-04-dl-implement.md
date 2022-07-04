---
title: 최대한 기본 라이브러리로만 딥러닝 구현하기
categories:
  - Pytorch
tags: [torch]
---

## Motivation
> 모기업 코딩테스트에 파이썬 기본 라이브러리로만 MLP를 구현하는 문제가 나왔던 적이 있습니다. 당시에 학습이 되지 않아 코딩테스트에서 떨어졌었고 구현하지 못했던 것이 계속 생각났었습니다.
> 그리고 numpy로 구현한 코드는 많았지만 numpy도 사용하지 않고 구현한 코드는 많이 없었습니다. 그래서 도전해봤습니다.

## 계획
데이터셋을 MNIST로 잡고 MLP를 구현하고자 했습니다. 코딩테스트때도 입력으로 MNIST와 비슷한 값이 들어왔었기 때문입니다.  

레이어는 총 3개로 input -> (Linear -> Activation) -> (Linear -> Activation) -> (Linear -> Softmax) -> output 으로 생각하고 각 모듈 구현을 시작했습니다.

처음 계획은 notebook 파일로 각 모듈을 만들고 마지막에 코드를 돌려보는 식으로 구상했다가 차라리 패키지로 만들어서 torch처럼 모듈을 import 하는것이 더 깔끔해보였습니다.

그렇게 데이터셋을 불러오는 dataset.py, 레이어를 불러오는 layers.py, 옵티마지어를 불러오는 optim.py, 각종 계산에 필요한 함수를 불러오는 utils.py로 나누게 되었습니다.

## 계산
모델이 학습을 하기 위해서는 역전파(backpropagation)가 진행되어야 합니다. 각 모듈들의 미분값을 출력하고 chain rule에 의해 값들을 곱해가면서 Linear 레이어의 가중치와 바이어스를 업데이트해야 합니다.

먼저, input -> (linear, sigmoid) -> (linear, softmax) -> output으로 구성된 모델이 있다고 가정하고 순전파(Feedforward)때 계산되는 과정을 살펴봐야 합니다.
- $y_1 = \sigma_{1}(z_1) = \sigma_1(w_1x + b_1), \sigma_{1} = \text{sigmoid}$
- $\hat{y} = \sigma_{2}(z_2) = \sigma_{2}(w_2y_1+b_2), \sigma_{2} = \text{softmax}$
- $Loss_{MSE} = \sum(\hat{y}-{y})^2$

이제, 위 식을 거꾸로 돌려가면서 역전파를 진행합니다.

$W_2$에 대해 편미분된 값을 먼저 구하면 다음과 같이 진행됩니다.
- $\frac{dL}{dw_2}=\frac{dL}{d\sigma_{2}} \cdot \frac{d\sigma_{2}}{dz_2} \cdot \frac{dz_2}{dw_2}$
- $\frac{dL}{dw_2} = \frac{1}{m}(\hat{y}-y) \cdot d\sigma_{2}(z_2) \cdot y_1$

$W_1$에 대해 편미분된 값을 구하면 다음과 같이 진행됩니다.
- $\frac{dL}{dw_1}=\frac{dL}{d\sigma_{2}} \cdot \frac{d\sigma_{2}}{dz_2} \cdot \frac{dz_2}{dy_1} \cdot \frac{dy_1}{dz_1} \cdot \frac{dz_1}{dw_1}$
- $\frac{dL}{dw_1}= \frac{1}{m}(\hat{y}-y) \cdot d\sigma_{2}(z_2) \cdot w_2 \cdot d\sigma(z_1) \cdot x$

두 weight의 공통된 점은 마지막 곱에는 **입력값**에 대해 dot product하고 입력 레이어로 움직일 때마다 **이전 레이어의 가중치**를 dot product한다는 특징이 있습니다.

또한, sigmoid와 softmax를 통과한 출력값들은 역전파때 **element-wise product**를 진행해야 합니다. 활성화함수는 입력값 각각에 대해 함수를 통과시키므로 역전파때도 똑같이 진행되어야 하기 때문입니다.

이 과정을 구현하기 위해 레이어마다 backward()함수를 추가하여 편미분을 계산하도록 코드를 작성했습니다.

## 결과
MNIST 5000장을 훈련데이터로 사용하고 1000장을 테스트데이터로 사용했습니다.

![](https://drive.google.com/uc?export=view&id=1k18xXPI4qMx31qgSTajBwQ6NwjycTpkr){:width="400"}
![](https://drive.google.com/uc?export=view&id=1Pzta5dtXVxduFsIgHGaSqsHOtKsh6jSh){:width="400"}  

10 에포크에도 loss가 잘 떨어지고 Accuracy도 잘 증가하는 것을 볼 수 있습니다. 

벡터 계산이나 다른 수식 계산에 도움이 되는 numpy 없이 구현하려고 하니 코드에서 실수를 많이 했습니다. 계산 결과를 확인하기 위해 torch나 numpy에 있는 똑같은 함수를 불러오고 저의 코드를 불러와 계산결과가 맞는지 계속 확인했습니다.  

torch로 모델을 학습하는 방법과 최대한 유사하게 작성할 수 있도록 구현했습니다. torch에서는 autograd 기능과 텐서를 사용할 수 있어 사용자가 쉽게 모델을 학습할 수 있었습니다. 하지만 직접 구현하려면 역전파를 위해 레이어마다 미분을 진행해줘야 하는 과정이 추가되어 생각보다 구현이 어려웠습니다. (딥러닝 라이브러리 개발자분들에게 정말 감사합니다.)

### 문제점
가장 큰 문제점은 e^x 함수의 Overflow 현상입니다. 입력값이 음수이면서 큰 수일 때 softmax와 sigmoid 계산에서 overflow 현상이 일어났습니다. 이를 방지하기 위해 round 함수로 소수점 아래 4자리까지만 사용하도록 했지만 가끔씩 overflow가 터지는 현상이 있습니다.

다음은, softmax 함수의 편미분 식을 계산하지 못했습니다. 원래는 Loss에 대한 편미분 * softmax에 대한 편미분으로 계산하는 것이 맞지만 현재는 softmax와 CrossEntropy를 사용하면서 옵티마이저에서 softmax 편미분 코드를 빼고 Loss함수에서 간단하게 도출된 식을 사용하고 있습니다. 나중에 모듈들이 추가될 경우 이 부분에 대해 수정이 필요하다고 생각합니다.

## Reference
- [http://taewan.kim/post/sigmoid_diff/](http://taewan.kim/post/sigmoid_diff/)
- [https://ratsgo.github.io/deep%20learning/2017/10/02/softmax/](https://ratsgo.github.io/deep%20learning/2017/10/02/softmax/)
- [https://pytorch.org/docs/stable/generated/torch.nn.Linear.html](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [https://velog.io/@gjtang/Softmax-with-Loss-%EA%B3%84%EC%B8%B5-%EA%B3%84%EC%82%B0%EA%B7%B8%EB%9E%98%ED%94%84](https://velog.io/@gjtang/Softmax-with-Loss-%EA%B3%84%EC%B8%B5-%EA%B3%84%EC%82%B0%EA%B7%B8%EB%9E%98%ED%94%84)