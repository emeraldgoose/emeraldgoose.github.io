---
title: Python으로 MLP 구현하기
categories:
  - Pytorch
tags: [torch]
---

> 코딩테스트로 Python으로만 MLP를 구현하는 문제가 나왔던 적이 있습니다. 당시에 역전파 구현을 하지 못해 코딩테스트에서 떨어졌었고 완전히 바닥에서부터 구현해보고자 시작한 프로젝트입니다.

## Multi-Layer Perceptron
Multi-Layer Perceptron(MLP)은 퍼셉트론으로 이루어진 층(layer)들이 쌓여 신경망을 이루는 모델입니다. 구현이 간단하기 때문에 딥러닝을 바닥부터 구현하는 프로젝트를 시작하는데 좋은 모델입니다.
저는 MNIST를 데이터셋으로 하여 모델을 훈련시키고 classification task를 수행해볼 것입니다.

구현하려고 하는 모델의 구조는 다음과 같습니다.
```text
Model(
  (net1): Sequential(
    (0): Linear(in_features=784, out_features=28, bias=True)
    (1): Sigmoid()
  )
  (net2): Sequential(
    (0): Linear(in_features=28, out_features=28, bias=True)
    (1): Sigmoid()
  )
  (fc): Linear(in_features=28, out_features=10, bias=True)
)
```


## Forward Propagation
먼저 위의 모델의 순전파를 구현하기 위해 수식으로 표현합니다.

$y_1 = \sigma(z_1) = \sigma(w_1x + b_1), \sigma = \text{sigmoid}$

$y_2 = \sigma(z_2) = \sigma(w_2y_1 + b_2)$

$\hat{y} = w_3y_2 + b_3$

$L_{\text{MSE}} = \sum(\hat{y}-{y})^2$

## Backward Propagation
먼저, $w_3$에 대해 편미분된 값은 쉽게 구할 수 있습니다.

$\frac{\partial L}{ \partial w_3}=\frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{dw_3}$

다음, $w_2$에 대해 편미분된 값을 구하면 다음과 같이 진행됩니다.

$\frac{\partial L}{ \partial w_2}=\frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial y_2} \cdot \frac{\partial y_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial w_2}$

${\partial y_2 \over \partial z_2} = \sigma_2(y_2)(1 - \sigma_2(y_2)) = y_2(1 - y_2)$

$\frac{\partial L}{\partial w_2} = \frac{2}{m}(\hat{y}-y) \cdot w_3 \cdot y_2(1 - y_2) \cdot y_1, \\ \text{m = batch size}$

마지막으로, $w_1$에 대해 편미분된 값을 구하면 다음과 같이 진행됩니다.

$\frac{\partial L}{\partial w_1}=\frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial y_2} \cdot \frac{\partial y_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial y_1} \cdot \frac{\partial y_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$

$\frac{\partial L}{\partial w_1}= \frac{2}{m}(\hat{y}-y) \cdot w_3 \cdot y_2(1 - y_2) \cdot w_2 \cdot y_1(1 - y_1) \cdot x$

gradient의 계산에서 마지막 곱에는 입력값에 대해 dot product하고 입력 레이어 방향으로 이전 레이어의 weight를 dot product해야 합니다. 따라서 Linear 레이어는 입력값을 저장해야 backward 계산에서 사용할 수 있습니다. 
<script src="https://gist.github.com/emeraldgoose/5bbdab6c658bc73da63bbc694bcf5f2a.js"></script>

또한, sigmoid를 통과한 출력값들은 역전파때 element-wise product를 진행해야 합니다. 활성화함수는 입력값 각각에 대해 함수를 통과시키므로 역전파때도 똑같이 진행되어야 하기 때문입니다. 참고로 softmax는 element-wise independent하지 않아 element-wise product를 수행해서는 안됩니다. 
<script src="https://gist.github.com/emeraldgoose/d6e424241f020b7f80e1a1c6441e1c6f.js"></script>

gradient를 업데이트하는 간단한 Optimizer를 이용하여 weight와 bias를 학습할 수 있습니다.  
<script src="https://gist.github.com/emeraldgoose/f256205e7bed257c9b1c5ecbcfc409e5.js"></script>


## 결과
MNIST 5000장을 훈련데이터로 사용하고 1000장을 테스트데이터로 사용했습니다.

![](https://lh3.google.com/u/0/d/1k18xXPI4qMx31qgSTajBwQ6NwjycTpkr){:width="400"}
![](https://lh3.google.com/u/0/d/1Pzta5dtXVxduFsIgHGaSqsHOtKsh6jSh){:width="400"}  

10 에포크에도 loss가 잘 떨어지고 Accuracy도 잘 증가하는 것을 볼 수 있습니다. 

## 코드
[https://github.com/emeraldgoose/hcrot](https://github.com/emeraldgoose/hcrot)

## Reference
- [http://taewan.kim/post/sigmoid_diff/](http://taewan.kim/post/sigmoid_diff/)
- [https://ratsgo.github.io/deep%20learning/2017/10/02/softmax/](https://ratsgo.github.io/deep%20learning/2017/10/02/softmax/)
- [https://pytorch.org/docs/stable/generated/torch.nn.Linear.html](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [https://velog.io/@gjtang/Softmax-with-Loss-%EA%B3%84%EC%B8%B5-%EA%B3%84%EC%82%B0%EA%B7%B8%EB%9E%98%ED%94%84](https://velog.io/@gjtang/Softmax-with-Loss-%EA%B3%84%EC%B8%B5-%EA%B3%84%EC%82%B0%EA%B7%B8%EB%9E%98%ED%94%84)
- [https://aew61.github.io/blog/artificial_neural_networks/1_background/1.b_activation_functions_and_derivatives.html](https://aew61.github.io/blog/artificial_neural_networks/1_background/1.b_activation_functions_and_derivatives.html)