---
title: 최대한 기본 라이브러리로만 CNN 구현하기
categories:
  - Pytorch
tags: [torch]
---

## Related
> 이전 포스트에서 MLP를 구현했고 이번에는 CNN을 구현하는 삽질을 진행했습니다.  
> 여기서는 Conv2d의 구현에 대해서만 정리하려고 합니다. 밑바닥부터 구현하실때 도움이 되었으면 좋겠습니다.

## CNN
CNN은 [Conv2d + Pooling + (Activation)] 레이어가 수직으로 쌓여있는 뉴럴넷을 말합니다. 
구현해보려는 CNN의 구조는 다음과 같습니다.  
- Layer1 : Conv2d(1, 5, 5) -> ReLU -> MaxPool2d(2, 2)
- Layer2 : Conv2d(5, 7, 5) -> ReLU -> MaxPool2d(2, 2)
- Flatten Layer
- Linear Layer

## Conv2d
### Convolution
단순하게 for문 중첩으로 Convolution을 구현하면 연산속도가 너무 느려지기 때문에 다른 방법들을 찾아봤습니다.
- Numpy를 이용하는 방법
- FFT를 이용하는 방법  

FFT(Fast Fourier Transform)를 사용하는 방법은 수식과 구현방법이 어려워서 포기하고 첫 번째 방법인 Numpy를 사용하는 방법을 선택했습니다. 코드는 스택 오버플로우에 있는 코드를 가져와서 사용했습니다.
```python
def convolve2d(a, f):
    a,f = np.array(a), np.array(f)
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)
```
동작방법은 다음과 같습니다.  
- 적용할 이미지를 커널 크기의 맞게 잘라서 저장합니다.  
- 잘려진 이미지들을 하나씩 꺼내 커널과 element-wise product를 진행하여 더한 값들을 리턴합니다.  
- for문으로 커널을 움직일 필요 없이 곱셈과 합 연산만 진행하므로 속도가 빠릅니다.

### Forward
Conv2d 레이어의 forward를 먼저 보겠습니다.  
(3,3)인 X, (2,2)인 W를 convolution해서 (2,2)인 O를 계산한다고 가정합니다.  
![](https://drive.google.com/uc?export=view&id=12LftVBInOBxYeZkTQ0gI7hVsN-ibOWTH)
- $o_{11} = k_{11}x_{11} + k_{12}x_{12} + k_{21}x_{21} + k_{22}x_{22}$
- $o_{12} = k_{11}x_{12} + k_{12}x_{13} + k_{21}x_{22} + k_{22}x_{23}$
- $o_{21} = k_{11}x_{21} + k_{12}x_{22} + k_{21}x_{31} + k_{22}x_{32}$
- $o_{22} = k_{11}x_{22} + k_{12}x_{23} + k_{21}x_{32} + k_{22}x_{33}$

### Backward
이제 Conv2d 레이어의 backward를 계산해보겠습니다. dout은 뒤의 레이어에서 들어오는 gradient를 의미합니다.  
Backward 연산부터는 Foward의 그림과 같이 보면서 이해하시는 것을 추천드립니다.  

가장 먼저, forward에서 계산한 모든 식을 weight에 대해 편미분을 시도합니다.  
- $\frac{do_{11}}{dk_{11}}=x_{11} \quad \frac{do_{11}}{dk_{12}}=x_{12} \quad \frac{do_{11}}{dk_{21}}=x_{21} \quad \frac{do_{11}}{dk_{22}}=x_{22}$
- $\frac{do_{12}}{dk_{11}}=x_{12} \quad \frac{do_{12}}{dk_{12}}=x_{13} \quad \frac{do_{12}}{dk_{21}}=x_{22} \quad \frac{do_{12}}{dk_{22}}=x_{23}$
- $\frac{do_{21}}{dk_{11}}=x_{21} \quad \frac{do_{21}}{dk_{12}}=x_{22} \quad \frac{do_{21}}{dk_{21}}=x_{31} \quad \frac{do_{21}}{dk_{22}}=x_{32}$
- $\frac{do_{22}}{dk_{11}}=x_{22} \quad \frac{do_{22}}{dk_{12}}=x_{23} \quad \frac{do_{22}}{dk_{21}}=x_{32} \quad \frac{do_{22}}{dk_{22}}=x_{33}$

이제 weight의 gradient 하나만 계산해보면 다음과 같습니다.  
- $\frac{dL}{dk_{11}} = \frac{dL}{do_{11}} \cdot \frac{do_{11}}{dk_{11}} + \frac{dL}{do_{12}} \cdot \frac{do_{12}}{dk_{11}} + \frac{dL}{do_{21}} \cdot \frac{do_{21}}{dk_{11}} + \frac{dL}{do_{22}} \cdot \frac{do_{22}}{dk_{11}}  \quad \text{by the chain rule}$
- $\frac{dL}{dk_{11}} = d_{11} \cdot \frac{do_{11}}{dk_{11}} + d_{12} \cdot \frac{do_{12}}{dk_{11}} + d_{21} \cdot \frac{do_{21}}{dk_{11}} + d_{22} \cdot \frac{do_{22}}{dk_{11}} $

위에서 계산한 값을 대입하면 다음과 같습니다.
- $\frac{dL}{dk_{11}} = d_{11} \cdot x_{11} + d_{12} \cdot x_{12} + d_{21} \cdot x_{21} + d_{22} \cdot x_{22} $

따라서, **weight의 gradient는 dout과 입력 X와의 convolution 연산과 같습니다.**  
bias는 forward때 덧셈으로 계산되므로 편미분 값이 1입니다. 그래서 bias의 gradient는 dout의 합으로 계산할 수 있습니다.

이제 conv layer에서 나오는 gradient를 입력 레이어 방향으로 전달하기 위한 계산을 진행하겠습니다.  

이번에는 O를 계산하는 forward 식에서 x에 대해 편미분을 계산해두겠습니다.
- $\frac{do_{11}}{dx_{11}}=k_{11} \quad \frac{do_{11}}{dx_{12}}=k_{12} \quad \frac{do_{11}}{dx_{21}}=k_{21} \quad \frac{do_{11}}{dx_{22}}=k_{22}$
- $\frac{do_{12}}{dx_{12}}=k_{11} \quad \frac{do_{12}}{dx_{13}}=k_{12} \quad \frac{do_{12}}{dx_{22}}=k_{21} \quad \frac{do_{12}}{dx_{23}}=k_{22}$
- $\frac{do_{21}}{dx_{21}}=k_{11} \quad \frac{do_{21}}{dx_{22}}=k_{12} \quad \frac{do_{21}}{dx_{31}}=k_{21} \quad \frac{do_{21}}{dx_{32}}=k_{22}$
- $\frac{do_{22}}{dx_{22}}=k_{11} \quad \frac{do_{22}}{dx_{23}}=k_{12} \quad \frac{do_{22}}{dx_{32}}=k_{21} \quad \frac{do_{22}}{dx_{33}}=k_{22}$

다음 입력값 각각에 대한 gradient인 $\frac{dL}{dX}$값을 다음과 같이 유도할 수 있습니다.
- $\frac{dL}{dx_{11}} = \frac{dL}{do_{11}} \cdot \frac{do_{11}}{dx_{11}}$
- $\frac{dL}{dx_{12}} = \frac{dL}{do_{12}} \cdot \frac{do_{12}}{dx_{12}} + \frac{dL}{do_{11}} \cdot \frac{do_{11}}{dx_{12}}$
- $\frac{dL}{dx_{13}} = \frac{dL}{do_{12}} \cdot \frac{do_{12}}{dx_{13}}$
- $\frac{dL}{dx_{21}} = \frac{dL}{do_{21}} \cdot \frac{do_{21}}{dx_{21}} + \frac{dL}{do_{11}} \cdot \frac{do_{11}}{dx_{21}}$
- $\frac{dL}{dx_{22}} = \frac{dL}{do_{22}} \cdot \frac{do_{22}}{dx_{22}} + \frac{dL}{do_{21}} \cdot \frac{do_{21}}{dx_{22}} + \frac{dL}{do_{12}} \cdot \frac{do_{12}}{dx_{22}} + \frac{dL}{do_{11}} \cdot \frac{do_{11}}{dx_{22}}$
- $\frac{dL}{dx_{23}} = \frac{dL}{do_{22}} \cdot \frac{do_{22}}{dx_{23}} + \frac{dL}{do_{12}} \cdot \frac{do_{12}}{dx_{23}}$
- $\frac{dL}{dx_{31}} = \frac{dL}{do_{21}} \cdot \frac{do_{21}}{dx_{31}}$
- $\frac{dL}{dx_{32}} = \frac{dL}{do_{22}} \cdot \frac{do_{22}}{dx_{32}} + \frac{dL}{do_{21}} \cdot \frac{do_{21}}{dx_{32}}$
- $\frac{dL}{dx_{33}} = \frac{dL}{do_{22}} \cdot \frac{do_{22}}{dx_{33}}$

위의 식에 다시 정리해둔 값을 대입하면 다음과 같습니다.
- $\frac{dL}{dx_{11}} = d_{11} \cdot k_{11}$
- $\frac{dL}{dx_{12}} = d_{12} \cdot k_{11} + d_{11} \cdot k_{12}$
- $\frac{dL}{dx_{13}} = d_{12} \cdot k_{12}$
- $\frac{dL}{dx_{21}} = d_{21} \cdot k_{11} + d_{11} \cdot k_{21}$
- $\frac{dL}{dx_{22}} = d_{22} \cdot k_{11} + d_{21} \cdot k_{12} + d_{12} \cdot k_{21} + d_{11} \cdot k_{22}$
- $\frac{dL}{dx_{23}} = d_{22} \cdot k_{12} + d_{12} \cdot k_{22}$
- $\frac{dL}{dx_{31}} = d_{21} \cdot k_{21}$
- $\frac{dL}{dx_{32}} = d_{22} \cdot k_{21} + d_{21} \cdot k_{22}$
- $\frac{dL}{dx_{33}} = d_{22} \cdot k_{22}$

식만 보고서는 어떤식으로 계산되어야 하는지 감이 잘 안옵니다. 이것을 그림으로 표현했을 때 정말 쉽게 계산이 가능합니다. 파란색 테두리가 weight, 빨간색 테두리가 dout, 노란색 테두리는 계산에 참여하는 cell입니다.  
- $\frac{dL}{dx_{11}} = d_{11} \cdot k_{11}$
![](https://drive.google.com/uc?export=view&id=18KIBdo2AbiexxzKaL15siJAFc5KtVpbW){:width="300"}

- $\frac{dL}{dx_{12}} = d_{12} \cdot k_{11} + d_{11} \cdot k_{12}$
![](https://drive.google.com/uc?export=view&id=1_5_BHHzLyqCyAJaaincxoM0fSg0jVJ2V){:width="200"}

- $\frac{dL}{dx_{13}} = d_{12} \cdot k_{12}$
![](https://drive.google.com/uc?export=view&id=1RdI042LLWwvWM6g64K8RBDiYfLhGfl04){:width="300"}

파란색 테두리인 weight를 보시면 아시겠지만 왼쪽 상단이 k21로 시작합니다. 즉, weight를 뒤집은 형태로 convolution 연산을 진행합니다.  
따라서, **dout을 적절하게 padding하고 weight를 뒤집어서 convolution을 진행한 결과가 입력에 대한 gradient입니다.**  

## CrossEntropyLoss
이전에 구현했던 CrossEntropyLoss는 log를 취하는 과정에서 입력값이 음수가 들어오는 경우 에러가 일어납니다.

Pytorch의 CrossEntropyLoss는 Softmax와 NLLLoss(Negative Log Likelihood)로 구성되어 있습니다. Softmax와 NLLLoss로 구현하게 되면 음수 입력에 대해서도 cross entropy 값을 구할 수 있습니다.

## Result
저번 포스팅과 똑같이 MNIST 5000장을 훈련 데이터로 사용하고 1000장을 테스트 데이터로 사용했습니다.  
![](https://drive.google.com/uc?export=view&id=1vBR0h4xl5xUitvcgjCtPi8VpBcrbUwVD){:width="500"}  
![](https://drive.google.com/uc?export=view&id=1DNukdv1AXX7Xfv7ODVPQU9xiKRQemhrj){:width="500"}  

이전 MLP보다 사이즈가 작아서 그런지 같은 10 epoch에도 Accuracy 90%를 넘지 못했습니다. 그래도 loss도 잘 떨어지고 ACC도 잘 증가하는 형태를 보였습니다.

이번 구현에서 가장 어려웠던 점은 Conv2d의 backward 구현이었습니다. 정리된 내용은 2차원 배열을 가지고 계산한 내용인데 직접 구현한 Conv2d Layer는 channel까지 포함시킨 3차원(Channel, Height, Width) 입출력을 계산해야 합니다.

수식을 다시 계산하고 구현된 코드를 보면서 어디에 값이 저장되어야 하는지 새롭게 정리해서 다시 구현을 진행할 수 있었습니다. 그리고 numpy 없이 구현하기 위해 numpy의 함수들도 새롭게 작성했습니다. 만약 구현을 한다면 Numpy 사용을 적극 권장합니다...


### Code
- [https://github.com/emeraldgoose/hcrot](https://github.com/emeraldgoose/hcrot)

## Reference
- [Convolve2d(StackOverflow)](https://stackoverflow.com/a/43087771)
- [https://ratsgo.github.io/deep%20learning/2017/04/05/CNNbackprop/](https://ratsgo.github.io/deep%20learning/2017/04/05/CNNbackprop/)
- [https://velog.io/@changdaeoh/backpropagationincnn](https://velog.io/@changdaeoh/backpropagationincnn)
- [https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509](https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509)
- [https://towardsdatascience.com/forward-and-backward-propagation-of-pooling-layers-in-convolutional-neural-networks-11e36d169bec](https://towardsdatascience.com/forward-and-backward-propagation-of-pooling-layers-in-convolutional-neural-networks-11e36d169bec)