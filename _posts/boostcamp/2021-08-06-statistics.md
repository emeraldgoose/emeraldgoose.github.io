---
title: Statistics 정리
categories:
 - BoostCamp
tags: [statistics]
---
## 통계적 모델링
- 적절한 가정 위에서 확률분포를 추정(inference)하는 것이 목표이며, 기계학습과 통계학이 공통적으로 추구하는 목표입니다.
- 그러나 유한개의 데이터만 관찰해서 모집단의 분포를 정확하게 알아내는 것은 불가능하므로, 근사적으로 확률분포를 추정해야 합니다.
    - 예측모형의 목적은 분포를 정확하게 맞추는 것보다 데이터의 추정 방법의 불확실성을 고려해서 위험(risk)을 최소화하는 것입니다.
- 데이터가 특정 확률분포를 따른다고 가정한 뒤 그 분포를 결정하는 모수(parameter)를 추정하는 방법을 모수적(parametric) 방법론이라 합니다.
    - 예를들어, 정규분포에서 평균 $\mu$과 분산 $\sigma^2$이 모수라고 할 수 있습니다.
- 반면에, 특정 확률분포를 가정하지 않고 데이터에 따라 모델의 구조와 모수의 개수가 유연하게 바뀌면 비모수(nonparametric) 방법론이라 합니다.
    - 비모수 방법론은 모수가 너무 많거나 데이터에 따라 모수가 자주 바뀌는 경우를 말하기 때문에 모수가 없다고 할 수 없습니다.

## 확률분포 가정
- 확률분포를 가정하는 방법으로 아래와 같은 예제들이 존재합니다.
    - 데이터가 2개의 값(0, 1)만 가지는 경우 : 베르누이 분포
    - 데이터가 n개의 이산적인 값만 가지는 경우 : 카테고리 분포
    - 데이터가 [0,1] 사이에서 값을 가지는 경우 : 베타 분포
    - 데이터가 0이상의 값을 가지는 경우 : 감마분포, 로그정규분포 등
    - 데이터가 R 전체에서 값을 가지는 경우 -> 정규 분포, 라플라스 분포 등
- 그러나, 예외가 있을 수 있으므로 데이터가 생성되는 원리를 먼저 고려하여 확률분포를 가정해야 합니다.
    - 예를들어, 항상 양수인 데이터인 경우 정규분포로 모형화가 가능하다면 정규분포를 사용할 수 있습니다.
- 확률분포를 가정했다면 이제 데이터로부터 해당 확률분포의 모수의 값을 구해야 합니다.

## 모수 추정 방법론
- 모수의 값으로 가장 가능성이 높은 하나의 숫자를 찾아내는 작업을 모수 추정(parameter estimation)이라 합니다.
- 모수를 추정할 수 있는 방법은 다음과 같습니다.
    - 모멘트 방법(Method of Moment, MM)
    - 최대가능도 추정법(Maximum likelihood estimation, MLE)
    - 베이즈 추정법(Bayesian estimation)

## 모멘트 방법
- 모멘트 방법은 표본자료에 대하 표본모멘트가 확률분포의 이론적 모멘트와 같다고 가정하여 모수를 구하는 방법입니다.
- 1차 모멘트(평균) : $\mu = E[\bar{x}] = \frac{1}{N}\sum_{i=1}^{N}X_i$
- 2차 모멘트(분산) : $\sigma^2 = E[(X - \mu)^2] = \bar{s}^2=\frac{1}{N-1}\sum_{i=1}^{N}(X_i-\bar{X})^2$
    - 평균과 달리 (N-1)로 나누는 이유는 불편추정량(unbiased estimate)을 구하기 위해서입니다.
    - 불편추정량이란, 추정량의 기댓값과 실제 모수와의 차이가 없는 경우를 말합니다.
    - 베셀 보정(Bassel's Correction)을 검색하시면 관련 내용을 보실 수 있습니다.

## 표집분포? 표본분포?
- 표집분포(sample distribution)
    - 모집단의 속성을 알기 위해 모집단을 대표할 수 있게 추출된 대상군의 분포입니다.
    - 표본의 속성은 표본평균과 표준편차로 표기하며 통계치 혹은 추정치라 합니다.
    - 항상 정규분포를 따르지 않습니다.
- 표본분포(sampling distribution)
    - 추리통계의 의사결정을 위한 이론적 분포
    - 이론적으로 표본의 크기가 n인 표본을 무한히 반복 추출한 후 무한개의 표본들의 평균을 가지고 그린 분포로 추정치의 분포라고도 함
    - n이 커질수록 정규분포를 따른다.
- 중심극한정리(central limit theorem)
    - 표집분포의 평균은 모집합의 평균이다.
    - 표집분포의 분산은 모집단의 분산을 표본의 크기로 나눈 것과 같다.
    - 표본의 크기가 충분히 클 때(N>30) 모집단의 분포와 상관없이 정규분포가 된다.

## 최대가능도 추정법
- 표본평균이나 표본분산은 확률분포마다 사용하는 모수가 다르므로 적절한 통계량이 달라지게 된다.
- 이론적으로 가장 가능성이 높은 모수를 추정하는 방법 중 하나가 최대가능도 추정법이다.
- 먼저, 확률과 가능도의 차이를 알아보자.
    - 10개의 동전을 던져 3번 앞면이 나왔다고 가정하자.
    - 확률 : 앞면이 나올 확률은 0.3이다. $\rightarrow$ 확률실험의 결과를 집계한 것
    - 가능도 : 10개의 동전을 던져 앞면이 나올 확률에 따라 앞면이 3번 나올 가능성을 말한다.
- 확률분포 $X$에 대한 확률밀도함수 또는 확률질량함수를 다음과 같이 대표해서 사용한다
    - $p(x;\theta)$
        - $x$는 확률분포가 가질 수 있는 실수값이며 $x$는 스칼라값일 수도 벡터값일 수도 있다.
        - $\theta$는 확률밀도함수의 모수를 표시하는 대표기호이며 $\theta$도 스칼라일 수도 있고 벡터일 수도  있다.
        - 확률밀도함수에서는 모수 $\theta$가 이미 알고 있는 상수계수고 $x$가 변수이다.
- 모수 추정 문제에서는 $x$값은 알고 있지만 모수 $\theta$를 모르기 때문에 반대로 x를 이미 알고있는 상수계수로 놓고 $\theta$를 변수로 생각한다.
    - 이 함수를 가능도함수(likelihood function)이라 하고 $L(\theta;x)$ 기호로 표기하고 다음과 같이 정의한다.
        - $L(\theta;x) = P(x \vert  \theta)$
        - 가능도 값을 계산하는 함수를 가능도 함수라 한다.
    - 최대가능도 추정법은 주어진 표본에 대해 가능도를 가장 크게 하는 모수 $\theta$를 찾는 방법
        - $\hat{\theta}\_\text{MLE} = \text{argmax}\_{\theta} L(\theta;x) = \text{argmax}\_\theta p(x \vert \theta)$
- 가능도 함수와 확률밀도함수의 수식은 같다. 그러나 가능도 함수는 확률밀도함수가 아니다.
    - $\int_{-\infty}^{\infty} p(x;\theta)dx = 1$
    - $\int_{-\infty}^{\infty}L(\theta;x)d\theta = \int_{-\infty}^{\infty}p(x;\theta)d\theta ≠ 1$ (적분하면 전체 면적이 1이 아닐 수 있다)
    
### 복수의 표본 데이터가 있는 경우

- 일반적으로 확률변수 표본의 수가 여러개 {$x_1, x_2, ... , x_N$}이므로 가능도 함수에서 복수 표본값에 대한 결합확률밀도 $p_{x_1x_2x_3...x_N}(x_1,x_2,x_3,...,x_N;\theta)$가 된다
- $x_1, x_2, ..., x_N$은 같은 확률분포에서 나온 독립적인 값들이므로 결합밀도함수는 다음과 같이 곱으로 표현된다.
    - $L(\theta; x_1, x_2, ..., x_N) = p(x_1, x_2, ..., x_N \vert  \theta) = \prod_{i=1}^{N} p(x_i\vert \theta)$
- 일반적으로 최대가능도 추정법을 사용하여 최대가 되는 $\theta$를 찾기 위해 최적화를 진행해야한다.
    - 보통 로그가능도함수 $LL = log\space L$을 사용하는 경우가 많다.
        - 로그 변환에 의해서는 최댓값의 위치가 변하지 않는다.
        - 복수 표본 데이터의 경우, 결합확률밀도함수가 동일한 함수의 곱으로 나타나게 되는데 로그 변환에 의해 곱셈이 덧셈이 되어 계산이 편해진다.
    - $log\space L(\theta; X) = \sum_{i=1}^{N} log\space P(x_i\vert \theta)$

### 최대가능도 예제: 정규분포

- 정규분포를 따른 확률변수 $X$로부터 독립적인 표본 $n$개를 얻었을 때, 최대가능도 추정법을 이용하여 모수를 추정하면?
    - 정규분포 $p(x) = \frac{1}{\sqrt(2\pi\sigma^2)}e^{-\frac{\vert x_i-\mu\vert ^2}{2\sigma^2}}$
    - $\begin{matrix} log L(\theta; X) &=& \sum_{i=1}^{n} log\space P(x_i\vert \theta) \\ &=& \sum_{i=1}^{n}\frac{1}{\sqrt(2\pi\sigma^2)}e^{-\frac{\vert x_i-\mu\vert ^2}{2\sigma^2}} \\ &=& -\frac{n}{2}log2\pi\sigma^2 - \sum_{i=1}^{n}\frac{\vert x_i-\mu\vert ^2}{2\sigma^2} \end{matrix}$
    - $log\space L$에 대해 $\mu$와 $\theta$로 미분하여 0이 되는 지점을 찾으면 가능도를 최대화할 수 있다.
        - $0 = \frac{\partial log\space L}{\partial \mu}=-\sum\_{i=1}^{n}\frac{x\_i-\mu}{\sigma^2} \Rightarrow \hat{\mu}\_{MLE}=\frac{1}{n}\sum\_{i=1}^{n}x\_i$
        - $0 = \frac{\partial log\space L}{\partial \sigma}=-\frac{n}{\sigma}+\frac{1}{\sigma^3}\sum\_{i=1}^{n}\vert x\_i-\mu\vert ^2 \Rightarrow \hat{\sigma}\_{MLE}^2 = \frac{1}{n}\sum\_{i=1}^{n}(x\_i-\mu)^2$

- 정규분포로부터 표본값 {1, 0, -3}을 얻었다고 가정하자.
    - $log\space L(\theta;x_1,x_2,x_3) = log\space (\frac{1}{(2\pi\sigma^2)^\frac{3}{2}})-\frac{3\mu^2+4\mu+10}{2\sigma^2} = log\space (\frac{1}{(2\pi\sigma^2)^\frac{3}{2}})-\frac{3(\mu+\frac{2}{3}) +\frac{26}{3}}{2\sigma^2}$
    - 따라서 최댓값의 위치 $\mu = -\frac{2}{3}$를 찾을 수 있다.

### 최대가능도 예제: 카테고리분포

- 카테고리 분포 Multinoulli$(x;p_1,...p_d)$를 따르는 확률변수 $X$로부터 독립적인 표본 {$x_1, ...,x_n$}을 얻었을 때, 최대가능도 추정법을 이용하여 모수를 추정하면?
- 모수가 $\mu=(\mu_1,...,\mu_k)$인 카테고리분포의 확률질량함수는 다음과 같다.
    - $p(x; \mu_1,...,\mu_k)=Cat(x;\mu_1,...,\mu_k)=\prod_{i=1}^{k}\mu_i^{x_i}, \space (\sum_{i=1}^{k}\mu_i=1)$
    - 이 식에서 $x$는 모두 $k$개의 원소를 가지는 원핫인코딩(one-hot-encoding) 벡터이다.
    - 표본데이터가 $x_1,x_2,...,x_n$이 있을 경우 모두 독립이므로 전체 확률밀도함수는 각각의 확률질량함수의 곱과 같다.
        - $L(\mu_1, ..., \mu_k;x_1,...x_k) = \prod_{i=1}^{n}\prod_{k=1}^{d}\mu_k^{x_{i,k}}$
- 로그 변환을 한 로그가능도를 구하면 다음과 같다.
    - $\hat{\theta}\_{\text{MLE}} = \text{argmax}\_{p1,...,p_d} log(\prod\_{i=1}^{n}\prod\_{k=1}^{d}p\_k^{x\_{i,k}})$
    - $log(\prod_{i=1}^{n}\prod_{k=1}^{d}p_k^{x_{i,k}})=\sum_{k=1}^d(\sum_{i=1}^nx_{i,k})log\space p_k, \space n_k=\sum_{i=1}^n x_{i,k}$
        - $n_k$는 주어진 각 데이터들에 대해서 $k$값이 1인 개수를 카운팅하는 값으로 대체할 수 있다.
    - $\sum_{k=1}^d n_k log\space p_k$와 $\sum_{k=1}^dp_k=1$의 식을 모두 만족하면서 최대가능도를 추정해야한다.
    - 라그랑주 승수법을 통해 새로운 목적식을 만들 수 있다.
        - $\sum_{k=1}^{d}n_klog\space p_k+\lambda(1-\sum_k p_k)$
        - $0=\frac{\partial L}{\partial p_k}=\frac{n_k}{p_k}-\lambda, \space 0=\frac{\partial L}{\partial \lambda}=1-\sum_{k=1}^{d}p_k$
        - $p_k=\frac{n_k}{\sum_{k=1}^dn_k}$

## 딥러닝에서 최대가능도 추정법

- 딥러닝 모델의 가중치를 $\theta = (W^{(1)},...,W^{(L)})$라 표기했을 때 분류문제에서 소프트맥스 벡터는 카테고리분포의 모수 $(p_1,...,p_k)$를 모델링한다.
- 원핫벡터로 표현한 정답레이블 $y=(y_1,...,y_k)$을 관찰데이터로 이용해 확률분포인 소프트맥스 벡터의 로그가능도를 최적화할 수 있다.
    - $ \hat{\theta}\_{\text{MLE}}=\text{argmax}\_{\theta}\frac{1}{n}\sum\_{i=1}^n\sum\_{k=1}^{K}y\_{i,k}log(\text{MLP}\_{\theta}(x\_i)\_k) $

## 확률분포의 거리

- 기계학습에서 사용되는 손실함수들은 모델이 학습하는 확률분포와 데이터에서 관찰되는 확률분포의 거리를 통해 유도한다
- 데이터공간에 두 개의 확률분포 P(x), Q(x)가 있을 경우 두 확률분포 사이의 거리(distant)를 계산할 때 다음과 같은 함수를 이용
    - 총변동 거리(Total Variation Distance, TV)
    - 쿨백-라이블러 발산(Kullback-Leibler Divergence, KL)
    - 바슈타인 거리(Wasserstein Distance)

### 쿨백-라이블러 발산

- 쿨백-라이블러 발산은 다음과 같이 정의한다.
    - 이산확률변수 : $KL(P\Vert Q)=\sum_{x\in\chi}P(x)log(\frac{P(x)}{Q(x)})$
    - 연속확률변수 : $KL(P\Vert Q)=\int_{\chi}P(x)log(\frac{P(x)}{Q(x)})dx$
- 쿨백-라이블러 발산을 다음과 같이 분해할 수 있다.
    - $KL(P\Vert Q)=-E_{x\sim P(x)}[logQ(x)]+E_{x\sim P(x)}[logP(x)]=$ 크로스 엔트로피 + 엔트로피
- 분류 문제에서 정답레이블을 $P$, 모델 예측을 $Q$라 두면 최대가능도 추정법은 쿨백-라이블러 발산을 최소화하는 것과 같다.
- 쿨백-라이블러 발산에서 p와 q를 바꾼 값과 원래의 값이 다르다. 즉, 비대칭(asymmetric)하다.
    - $KL(p\Vert q)=H(p,q)-H(p) ≠ H(q,p)-H(q)=KL(q\Vert p)$ $\Rightarrow$ $KL(p\Vert q) ≠ KL(q\Vert p)$
    - 만약, 두 확률분포 사이의 거리라면 p에서 q사이의 거리나 q에서 p사이의 거리나 같아야 한다. 따라서 KL-divergence는 거리 개념(distance metric)이 아니라고 한다.
    - 하지만 거리 개념처럼 사용할 수 있는 방법이 존재하는데, 바로 Jesen-Shannon divergence이다.
        - $JSD(p\Vert q) = \frac{1}{2}KL(p\Vert M)+\frac{1}{2}KL(q\Vert M) \space where, M=\frac{1}{2}(p+q)$
        - 간편하게 쓸 수 있지만 KL-divergence 처럼 자주 쓰이지 않는다.
    - 관련내용 : [링크](https://hyunw.kim/blog/2017/10/27/KL_divergence.html)

## 베이즈 추정법
- 베이즈 정리는 조건부확률을 이용하여 정보를 갱신하는 방법
    - $P(\theta\vert D)=P(\theta)\frac{P(D\vert \theta)}{P(D)}$
    - $D$는 새로 관찰하는 데이터, $\theta$는 모수
        - $P(\theta\vert D)$ : 사후확률(posterior)
        - $P(\theta)$ : 사전확률(prior)
        - $P(D\vert \theta)$ : 가능도(likelihood)
        - $P(D)$ : Evidence, 어떠한 데이터들을 관찰하려고 할때 데이터 자체의 분포
    - 가능도와 Evidence를 가지고 사전확률을 사후확률로 갱신할 수 있다.
    
### 베이즈 정리의 확장1
- 만약, 사건 $A_1=A$와 $A_2=A^C$가 있다고 가정하자.(Binary 분류문제)
- $\begin{matrix}P(A\vert B) &=& \frac{P(B\vert A)P(A)}{P(B)} \\\\&=&\frac{P(B\vert A)P(A)}{P(B,A)+P(B,A^C)}\\\\ &=& \frac{P(B\vert A)P(A)}{P(B\vert A)P(A)+P(B\vert A^C)P(A^C)} \\\\ &=& \frac{P(B\vert A)P(A)}{P(B\vert A)P(A)+P(B\vert A^C)(1-P(A))}\end{matrix}$

### 베이즈 정리의 확장2

- 사건 A의 확률이 사건 B에 의해 갱신(Update)된 확률을 계산할 때, 이 상태에서 또 추가적인 사건 C가 
발생했다면 베이즈정리는 다음과 같이 쓸 수 있다.
   - $P(A\vert B,C) = \frac{P(C\vert A,B)P(A\vert B)}{P(C\vert B)}$
   - $P(A\vert B,C)$는 $B$와 $C$가 조건인 $A$인 확률이고 $P(A\vert (B\cap C))$을 뜻한다.

- 관련내용 : [링크](https://datascienceschool.net/02%20mathematics/06.06%20%EB%B2%A0%EC%9D%B4%EC%A6%88%20%EC%A0%95%EB%A6%AC.html)

## 조건부확률 → 인과관계?

- 조건부확률은 인과관계(causality)를 추론할 때 함부로 사용해서는 안된다. 데이터가 많아도 인과관계를 추론하는 것은 불가능하다.
- 또한, 중첩요인(confounding factor)의 효과를 제거하고 원인에 해당하는 변수만 인과관계를 계산해야 한다.
    - 제거하지 않을 시 가짜 연관성(spurious correlation)이 나온다
