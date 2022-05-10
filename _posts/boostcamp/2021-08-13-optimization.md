---
title: Optimization 정리
categories:
  - BoostCamp
tags: [optimization]
---
## Intorduction

### Gradient Descent
- 1차 반복적인 최적화 알고리즘은 미분 가능한 함수의 극소값을 찾기 위한 알고리즘이다.

## Important Concepts in Optimization

### Generalization
- 일반화 성능을 높이는 것이 목표
- 학습을 진행하게 되면 학습 데이터에 대한 에러가 줄어든다. 그러나 테스트는 일정 에러 이후에 오히려 늘어난다. 이 차이를 Generalization gap이라 한다.
- "좋은 Generalization performance를 가지고 있다"라는 의미는 이 네트워크 성능이 학습데이터와 비슷한 성능을 보여준다는 것을 의미한다.
- 학습 데이터에 대한 성능이 안좋으면 좋은 generalization performance라고 해도 안좋기 때문에 이런 단어들에 대해 유의해야 한다.

### Underfitting vs. Overfitting
- Underfitting
    - 학습의 양이 부족하거나 해서 학습 데이터에도 잘 동작하지 않는 현상
- Balanced
    - Underfitting과 Overfitting의 중간으로 적당한 성능이 나와주는 것
- Overfitting
    - 학습데이터에 대해 잘 동작하지만 테스트데이터에 대해 잘 동작하지 않는 현상

### Cross-validation
- 일반적으로 train 데이터와 test 데이터(validation 데이터)로 나눠서 학습을 시킨다.
- 학습한 모델이 학습에 사용하지 않은 validation 데이터로 얼마나 잘 나오는지 확인하는 것
- Cross-validation은 k개의 fold로 나눠 k-1개로 학습을 시킨 후 나머지 한개로 테스트하는 것.
    - 예를들어, 학습 데이터를 5개의 fold로 나누고 (2-5)번으로 학습한 후 1번으로 validation (1,3-5)번으로 학습한 후 2번으로 validation을 하는 과정이다.
- 뉴럴 네트워크에서 최적의 하이퍼 파라미터를 찾아야하는데 사용될 수 있다.
    - 테스트 데이터는 절대 학습에 사용해서는 안된다.(사실상 치팅이다.)

### Bias and Variance
- Variance란, 입력을 넣었을 때 출력이 얼마나 일관적인지를 알려준다.
    - Variance가 크면 같은 입력을 넣었을 때 출력이 많이 달라지는 것을 의미하므로 Overfitting이 발생할 가능성이 높다
- Bias란, 출력의 평균적인 위치가 True target에 접근하게 될 때 Bias가 낮다고 한다. 높다는 것은 bias가 target의 mean에 많이 벗어났음을 의미한다.

#### Bias and Variance Tradeoff
- Given $D = {(x_i,t_i)}_{i=1}^N, \space where t = f(x) + \epsilon \space and \space \epsilon \sim N(0,\sigma^2)$
- 내 학습데이터에 noise가 껴있다고 가정할 때 노이즈가 껴있는 타겟 데이터를 minimizing 하는 것은 세 가지로 나눌 수 있다. : bias, variance, noise
    - 내가 한 가지를 낮추고자 할 때 다른 한 가지고 높아질 수 있다(Tradeoff)
    - $E\[(t-\hat{f})^2\](cost) = E\[(t-f+f-\hat{f})^2\] = ... = E\[(f-E[\hat{f}]^2)^2\](bias^2) + E\[(E[\hat{f}]-\hat{f})^2\](variance)+E\[\epsilon\](noise)$
        - $t$ : noise가 껴있는 true target
        - noise가 껴있는 경우에는 bias와 variance를 둘 다 줄일 수 있는 것은 사실상 얻기가 어렵다.

### Bootstrapping
- 부트스트래핑은 학습데이터가 고정되어 있을 때 서브샘플링을 여러개 만들고 여러 모델 혹은 metric을 만들어서 학습을 하겠다는 의미

#### Bagging vs. Boosting
- Single Classifier(Single Iteration)
    - 하나의 모델을 만드는 방법
- Bagging(Bootstrapping aggregating)
    - 여러 모델을 랜덤 서브샘플링을 통해서 만들고 모델들의 아웃풋을 가지고 어떤식으로 평균을 내는 것
    - 앙상블 학습이라고도 함
- Boosting
    - 일부 데이터에 대해 간단한 모델을 만들고 아웃풋이 남은 데이터에 대해 잘 동작하지 않으면 남은 데이터에 대해 잘 동작하는 모델을 만든다.
    - 이런방식으로 여러개의 모델(weak learner)을 만들고 이 모델들을 합치는 방법(하나의 strong model을 만든다)
    - 각각의 weak learner들의 weight를 찾는 방법으로 정보를 취합하여 boosting한다.