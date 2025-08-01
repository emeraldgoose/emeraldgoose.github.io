---
title: Python으로 CNN 바닥부터 구현하기
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
```
CNN(
  (layer1): Sequential(
    (0): Conv2d(1, 5, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
  )
  (layer2): Sequential(
    (0): Conv2d(5, 7, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
  )
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc): Linear(in_features=112, out_features=10, bias=True)
)
```

## Convolution
> CNN 모델에서의 Convolution 연산은 입력 신호를 커널을 이용하여 증폭하거나 감소시켜 특징을 추출하거나 필터링하는 역할을 합니다.

단순하게 for문 중첩으로 Convolution을 구현하면 연산속도가 너무 느려지기 때문에 다른 방법들을 찾아봤습니다.
- Numpy를 이용하는 방법
- FFT를 이용하는 방법  

FFT(Fast Fourier Transform)를 사용하는 방법은 수식과 구현방법이 어려워서 포기하고 첫 번째 방법인 Numpy를 사용하는 방법을 선택했습니다. 코드는 스택 오버플로우에 있는 코드를 가져와서 사용했습니다.
<script src="https://gist.github.com/emeraldgoose/edd919bed30ae5ea2a1b021caaea33af.js"></script>

동작방법은 다음과 같습니다.  
- 적용할 이미지를 커널 크기의 맞게 잘라서 저장합니다.  
- 잘려진 이미지들을 하나씩 꺼내 커널과 곱하고 더한 값을 리턴합니다.  
- for문으로 커널을 움직일 필요 없이 곱셈과 합 연산만 진행하므로 속도가 빠릅니다.

## Forward
Conv2d 레이어의 forward를 먼저 보겠습니다.  
(3,3)인 입력 X, (2,2)인 가중치 W를 convolution해서 (2,2)인 출력 O를 계산한다고 가정합니다.  
![](https://onedrive.live.com/embed?resid=502FD124B305BA80%213270&authkey=%21AA6lg4k19kN2HHU&width=1273&height=491)
$o_{11} = k_{11}x_{11} + k_{12}x_{12} + k_{21}x_{21} + k_{22}x_{22}$

$o_{12} = k_{11}x_{12} + k_{12}x_{13} + k_{21}x_{22} + k_{22}x_{23}$

$o_{21} = k_{11}x_{21} + k_{12}x_{22} + k_{21}x_{31} + k_{22}x_{32}$

$o_{22} = k_{11}x_{22} + k_{12}x_{23} + k_{21}x_{32} + k_{22}x_{33}$
<script src="https://gist.github.com/emeraldgoose/d816ef2fddecef83236316f9316dcde0.js"></script>

## Backward
이제 Conv2d 레이어의 backward를 계산해보겠습니다. dout은 뒤의 레이어에서 들어오는 gradient를 의미합니다.  
Backward 연산부터는 Foward의 그림과 같이 보면서 이해하시는 것을 추천드립니다.  

가장 먼저, forward에서 계산한 모든 식을 weight에 대해 편미분을 시도합니다.

$\frac{do_{11}}{dk_{11}}=x_{11} \quad \frac{do_{11}}{dk_{12}}=x_{12} \quad \frac{do_{11}}{dk_{21}}=x_{21} \quad \frac{do_{11}}{dk_{22}}=x_{22}$

$\frac{do_{12}}{dk_{11}}=x_{12} \quad \frac{do_{12}}{dk_{12}}=x_{13} \quad \frac{do_{12}}{dk_{21}}=x_{22} \quad \frac{do_{12}}{dk_{22}}=x_{23}$

$\frac{do_{21}}{dk_{11}}=x_{21} \quad \frac{do_{21}}{dk_{12}}=x_{22} \quad \frac{do_{21}}{dk_{21}}=x_{31} \quad \frac{do_{21}}{dk_{22}}=x_{32}$

$\frac{do_{22}}{dk_{11}}=x_{22} \quad \frac{do_{22}}{dk_{12}}=x_{23} \quad \frac{do_{22}}{dk_{21}}=x_{32} \quad \frac{do_{22}}{dk_{22}}=x_{33}$


이제 weight의 gradient 하나만 계산해보면 다음과 같습니다.  

$\frac{dL}{dk_{11}} = \frac{dL}{do_{11}} \cdot \frac{do_{11}}{dk_{11}} + \frac{dL}{do_{12}} \cdot \frac{do_{12}}{dk_{11}} + \frac{dL}{do_{21}} \cdot \frac{do_{21}}{dk_{11}} + \frac{dL}{do_{22}} \cdot \frac{do_{22}}{dk_{11}}$

$\frac{dL}{dk_{11}} = d_{11} \cdot \frac{do_{11}}{dk_{11}} + d_{12} \cdot \frac{do_{12}}{dk_{11}} + d_{21} \cdot \frac{do_{21}}{dk_{11}} + d_{22} \cdot \frac{do_{22}}{dk_{11}} $

위에서 계산한 값을 대입하면 다음과 같습니다.

$\frac{dL}{dk_{11}} = d_{11} \cdot x_{11} + d_{12} \cdot x_{12} + d_{21} \cdot x_{21} + d_{22} \cdot x_{22} $

따라서, **weight의 gradient는 dout과 입력 X와의 convolution 연산과 같습니다.**  
bias는 forward때 덧셈으로 계산되므로 편미분 값이 1입니다. 그래서 bias의 gradient는 dout의 합으로 계산할 수 있습니다.

이제 conv layer에서 나오는 gradient를 입력 레이어 방향으로 전달하기 위한 계산을 진행하겠습니다.  

먼저, 출력 O를 계산하는 forward 식에서 입력 x에 대해 편미분을 계산해두겠습니다.

$\frac{do_{11}}{dx_{11}}=k_{11} \quad \frac{do_{11}}{dx_{12}}=k_{12} \quad \frac{do_{11}}{dx_{21}}=k_{21} \quad \frac{do_{11}}{dx_{22}}=k_{22}$

$\frac{do_{12}}{dx_{12}}=k_{11} \quad \frac{do_{12}}{dx_{13}}=k_{12} \quad \frac{do_{12}}{dx_{22}}=k_{21} \quad \frac{do_{12}}{dx_{23}}=k_{22}$

$\frac{do_{21}}{dx_{21}}=k_{11} \quad \frac{do_{21}}{dx_{22}}=k_{12} \quad \frac{do_{21}}{dx_{31}}=k_{21} \quad \frac{do_{21}}{dx_{32}}=k_{22}$

$\frac{do_{22}}{dx_{22}}=k_{11} \quad \frac{do_{22}}{dx_{23}}=k_{12} \quad \frac{do_{22}}{dx_{32}}=k_{21} \quad \frac{do_{22}}{dx_{33}}=k_{22}$

다음 입력값 각각에 대한 gradient인 $\frac{dL}{dX}$값을 다음과 같이 계산할 수 있습니다.

$\frac{dL}{dx_{11}} = \frac{dL}{do_{11}} \cdot \frac{do_{11}}{dx_{11}}$

$\frac{dL}{dx_{12}} = \frac{dL}{do_{12}} \cdot \frac{do_{12}}{dx_{12}} + \frac{dL}{do_{11}} \cdot \frac{do_{11}}{dx_{12}}$

$\frac{dL}{dx_{13}} = \frac{dL}{do_{12}} \cdot \frac{do_{12}}{dx_{13}}$

$\frac{dL}{dx_{21}} = \frac{dL}{do_{21}} \cdot \frac{do_{21}}{dx_{21}} + \frac{dL}{do_{11}} \cdot \frac{do_{11}}{dx_{21}}$

$\frac{dL}{dx_{22}} = \frac{dL}{do_{22}} \cdot \frac{do_{22}}{dx_{22}} + \frac{dL}{do_{21}} \cdot \frac{do_{21}}{dx_{22}} + \frac{dL}{do_{12}} \cdot \frac{do_{12}}{dx_{22}} + \frac{dL}{do_{11}} \cdot \frac{do_{11}}{dx_{22}}$

$\frac{dL}{dx_{23}} = \frac{dL}{do_{22}} \cdot \frac{do_{22}}{dx_{23}} + \frac{dL}{do_{12}} \cdot \frac{do_{12}}{dx_{23}}$

$\frac{dL}{dx_{31}} = \frac{dL}{do_{21}} \cdot \frac{do_{21}}{dx_{31}}$

$\frac{dL}{dx_{32}} = \frac{dL}{do_{22}} \cdot \frac{do_{22}}{dx_{32}} + \frac{dL}{do_{21}} \cdot \frac{do_{21}}{dx_{32}}$

$\frac{dL}{dx_{33}} = \frac{dL}{do_{22}} \cdot \frac{do_{22}}{dx_{33}}$

위의 식에 다시 정리해둔 값을 대입하면 다음과 같습니다.

$\frac{dL}{dx_{11}} = d_{11} \cdot k_{11}$

$\frac{dL}{dx_{12}} = d_{12} \cdot k_{11} + d_{11} \cdot k_{12}$

$\frac{dL}{dx_{13}} = d_{12} \cdot k_{12}$

$\frac{dL}{dx_{21}} = d_{21} \cdot k_{11} + d_{11} \cdot k_{21}$

$\frac{dL}{dx_{22}} = d_{22} \cdot k_{11} + d_{21} \cdot k_{12} + d_{12} \cdot k_{21} + d_{11} \cdot k_{22}$

$\frac{dL}{dx_{23}} = d_{22} \cdot k_{12} + d_{12} \cdot k_{22}$

$\frac{dL}{dx_{31}} = d_{21} \cdot k_{21}$

$\frac{dL}{dx_{32}} = d_{22} \cdot k_{21} + d_{21} \cdot k_{22}$

$\frac{dL}{dx_{33}} = d_{22} \cdot k_{22}$

위의 수식들을 그림으로 표현했을 때 쉽게 이해할 수 있습니다. 파란색 테두리가 weight, 빨간색 테두리가 dout, 노란색 테두리는 계산에 참여하는 cell입니다.  

$\frac{dL}{dx_{11}} = d_{11} \cdot k_{11}$

![](https://onedrive.live.com/embed?resid=502FD124B305BA80%213266&authkey=%21AEjhrJQGGw89pMw&width=553&height=519){:width="300"}

$\frac{dL}{dx_{12}} = d_{12} \cdot k_{11} + d_{11} \cdot k_{12}$

![](https://onedrive.live.com/embed?resid=502FD124B305BA80%213273&authkey=%21ALz5nrzbqpVk2Qg&width=391&height=559){:width="200"}

$\frac{dL}{dx_{13}} = d_{12} \cdot k_{12}$

![](https://onedrive.live.com/embed?resid=502FD124B305BA80%213272&authkey=%21AMaGT5MenuAUQT8&width=566&height=537){:width="300"}

파란색 테두리인 weight를 보시면 아시겠지만 왼쪽 상단이 k22로 시작합니다. 즉, weight를 뒤집은 형태로 convolution 연산을 진행합니다.  
따라서, **dout을 적절하게 padding하고 weight를 뒤집어서 convolution을 진행한 결과가 입력에 대한 gradient입니다.**  
<script src="https://gist.github.com/emeraldgoose/3458dac08e743c36beec99446fc3231f.js"></script>

## CrossEntropyLoss
Pytorch의 CrossEntropyLoss는 Softmax와 NLLLoss(Negative Log Likelihood)로 구성되어 있습니다. Softmax와 NLLLoss로 구현하게 되면 음수 입력에 대해서도 cross entropy 값을 구할 수 있습니다.

모델의 출력값에 softmax를 취한 값을 NLLLoss에 넣으면 CrossEntropyLoss 수식과 같아집니다. CrossEntropyLoss는 모델이 정답에 가까워지게 cross entropy가 낮아지는 방향으로 학습하도록 합니다.  
$\text{CrossEntropyLoss } L = -\sum^C_i t_i log(P(x)), P(x) = \text{softmax}(x)$  
- $P(x)$는 확률분포, $t$는 정답 라벨

> NLLLoss(Negative Log Likelihood Loss)의 함수는 $L(y) = -log(y)$입니다. -log(정답 라벨에 대해 모델이 예측한 확률값) 값의 합이 Loss이고 이를 줄이는 방향이 모델이 정답을 강하게 예측할 수 있도록 합니다. 예를들면, pred = [0.5, 0.4, 0.1], label = [1] 이라면 loss = -log(0.4) = [0.92]이 됩니다.

<script src="https://gist.github.com/emeraldgoose/74d9ec6df399b257ae15348bea5299d7.js"></script>

## Result
저번 포스팅과 똑같이 MNIST 5000장을 훈련 데이터로 사용하고 1000장을 테스트 데이터로 사용했습니다. 다음과 같이 (conv2d, relu, pooling) 구조를 2번 거치고 fc 레이어로 logits을 출력할 수 있도록 하는 모델을 사용했습니다.

<script src="https://gist.github.com/emeraldgoose/d5198773511fc9da28517861b7df4160.js"></script>

MaxPool2d를 사용한 결과입니다.  

<figure class="half">
  <a href="https://onedrive.live.com/embed?resid=502FD124B305BA80%213267&authkey=%21AGS5LwJF6asYL2A&width=835&height=855" data-lightbox="gallery">
    <img src="https://onedrive.live.com/embed?resid=502FD124B305BA80%213267&authkey=%21AGS5LwJF6asYL2A&width=835&height=855" alt="01">
  </a>
  <a href="https://onedrive.live.com/embed?resid=502FD124B305BA80%213268&authkey=%21ADJCVOwK9-BfcI8&width=853&height=855" data-lightbox="gallery">
    <img src="https://onedrive.live.com/embed?resid=502FD124B305BA80%213268&authkey=%21ADJCVOwK9-BfcI8&width=853&height=855" alt="02">
  </a>
</figure>

AvgPool2d를 사용한 결과입니다.  

<figure class="half">
  <a href="https://onedrive.live.com/embed?resid=502FD124B305BA80%213271&authkey=%21AGzFHbahJdrYi-Y&width=835&height=855" data-lightbox="gallery">
    <img src="https://onedrive.live.com/embed?resid=502FD124B305BA80%213271&authkey=%21AGzFHbahJdrYi-Y&width=835&height=855" alt="03">
  </a>
  <a href="https://onedrive.live.com/embed?resid=502FD124B305BA80%213269&authkey=%21ABIFF7Ga1quAIRc&width=844&height=855" data-lightbox="gallery">
    <img src="https://onedrive.live.com/embed?resid=502FD124B305BA80%213269&authkey=%21ABIFF7Ga1quAIRc&width=844&height=855" alt="04">
  </a>
</figure>

결과만 봐서는 loss의 경우 MaxPool2d와 AvgPool2d의 결과가 비슷했고 accuracy의 경우 MaxPool2d가 조금 더 좋았습니다.

### Code
- [https://github.com/emeraldgoose/hcrot](https://github.com/emeraldgoose/hcrot)
- [cnn.ipynb](https://github.com/emeraldgoose/hcrot/blob/master/notebooks/cnn.ipynb)

## Reference
- [Convolve2d(StackOverflow)](https://stackoverflow.com/a/43087771)
- [CNN의 역전파(backpropagation)](https://ratsgo.github.io/deep%20learning/2017/04/05/CNNbackprop/)
- [Forward and Backward propagation of Max Pooling Layer in Convolutional Neural Networks](https://towardsdatascience.com/forward-and-backward-propagation-of-pooling-layers-in-convolutional-neural-networks-11e36d169bec)
