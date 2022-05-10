---
title: Convolution 정리
categories:
  - BoostCamp
tags: [convolution]
---
## Convolution
- 신호처리에서 정말 많이 등장하는 컨볼루션 연산입니다. 기계학습에서도 컨볼루션을 사용하여 학습을 하게 됩니다.
- 아래 식은 각 뉴련들이 선형모델과 활성함수로 모두 연결된(fully connected) 구조를 가졌습니다.
    - $h_i = \sigma(\sum_{j=1}^{p}W_{i,j}x_j)$, $\sigma$는 활성함수, $W_{i,j}$는 가중치 행렬입니다.
    - 각 성분 $h_i$에 대응하는 가중치 행 $W_i$이 필요합니다.
- 컨볼루션 연산은 커널(kernel)을 입력벡터 상에서 움직여가면서 선형모델과 합성함수가 적용되는 구조입니다.
    - $h_i = \sigma(\sum_{j=1}^{k}V_jx_{i+j-1})$, $\sigma$는 활성함수, $V_j$는 가중치 행렬입니다.
    - 모든 $i$에 대해 적용되는 커널은 $V$로 같고 커널의 사이즈만큼 $x$상에서 이동하면서 적용됩니다.
	- 또한, 활성화 함수를 제외한 컨볼루션 연산도 선형변환에 속합니다.
- 컨볼루션 연산의 수학적인 의미는 신호(signal)를 커널을 이용해 국소적으로 증폭 또는 감소시켠서 정보의 추출을 필터링하는 것입니다.
    - Continuous : $\[f\*g\](x) = \int\_{R^d}f(z)g(x-z)dz = \int\_{R^d}f(x-z)g(z)dz = \[g\*f\](x)$
    - discrete : $\[f\*g\](i) = \sum\_{a \in Z^d}f(a)g(i-a) = \sum\_{a \in Z^d}f(i-a)g(a) = \[g\*f\](i)$
    - CNN에서 사용하는 연산은 Convolution이 아니고 cross-correlatoin이라 한다.
- 커널은 정의역 내에서 움직여도 변하지 않고(translation invariant) 주어진 신호에 국소적(local)으로 적용한다.
- 또한, Convolution 연산은 1차원 뿐만 아니라 다양한 차원에서 계산 가능하다.
    - 데이터의 성격에 따라 사용하는 커널이 달라진다.
    - 2D-conv $\[f*g\](i,j) = \sum_{p,q}f(p,q)g(i+p,j+q)$
    - 3D-conv $\[f*g\](i,j,k) = \sum_{p,q,r}f(p,q,r)g(i+p,j+q,k+r)$
    
## 2D Convolution
- 2D-Conv 연산은 커널(kernel)을 입력벡터 상에서 움직여가면서 선형모델과 합성함수가 적용되는 구조이다.
- 입력크기를 (H,W), 커널 크기를 ($K_H,K_W$), 출력크기를 ($O_H,O_W$)라 하면 출력 크기는 다음과 같이 계산한다.
    - $O_H = H-K_H+1$
    - $O_W = W-K_W+1$
- 채널이 여러개인 2차원 입력의 경우 커널의 채널 수와 입력의 채널 수가 같아야 한다.
    - 커널($K_H,K_W,C$) * 입력($H,W,C$) -> 출력($O_H,O_W,1$)
    - 만약 커널을 $O_C$개 사용하면 출력도 $O_C$개가 된다. -> 출력이 텐서가 된다.
    
## Convolution연산의 역전파
- Conv 연산은 커널이 모든 입력데이터에 공통으로 적용되기 때문에 역전파를 계산할 때도 conv 연산이 나오게 된다.
    - $\frac{\partial}{\partial x}\[f\*g\](x) = \frac{\partial}{\partial x}\int_{R^d}f(y)g(x-y)dy = \int_{R^d}f(y)\frac{\partial g}{\partial x}(x-y)dy = \[f\*g'\](x)$
    - Discrete 일 때도 성립한다.    