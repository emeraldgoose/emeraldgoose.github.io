---
title: 경사하강법 정리
categories:
  - BoostCamp
tags: [gradient_descent]
---
## 미분
- 함수 f의 주어진 점(x,f(x))에서의 미분을 통해 접선의 기울기를 구할 수 있다. 접선의 기울기는 어느 방향으로 점을 움직여야 함수값이 증가하는지/감소하는지 알 수 있습니다.
- 미분값을 더하면 경사상승법(gradient ascent)라 하며 극댓값의 위치를 구할 때 사용합니다.
- 미분값을 빼면 경사하강법(gradient descent)라 하며 극소값의 위치를 구할 때 사용합니다.
- 경사상승법과 경사하강법 모두 극값에 도착하면 움직임을 멈추게 됩니다. (미분값이 0이므로)

## 경사하강법
- 경사하강법(Gradient Descent)는 함수값이 낮아지는 방향으로 독립변수 값을 변형시켜가며 최종적으로 극값에 이를때까지 반복하는 방법입니다.
- 최적화 할 함수 $f(x)$에 대하여 먼저 시작점 $x_0$를 정합니다. 현재 위치 $x_i$에서 다음 위치인 $x_{i+1}$은 다음과 같이 계산됩니다.
    - $x_{i+1} = x_i - \gamma\_{i} \nabla f(x_i)$
    - $\gamma\_i $는 이동할 거리를 조절하는 매개변수입니다.

### 경사하강법: 알고리즘

```python
"""
input: gradient(미분을 계산하는 함수), init(시작점), lr(학습률), eps(종료조건 위한 상수) 
output: var
"""
var = init
grad = gradient(var)
while(abs(grad) > eps): # 컴퓨터가 계산할 때 미분이 정확히 0이 되는 것은 불가능하므로 eps보다 작을 때 종료한다.
  var = var - lr * grad # 다음 위치 = 현재 위치 - 이동거리 * 기울기
  grad = gradient(var) # 움직인 현재 위치에서 다시 미분값을 계산한다.
```

## 변수가 벡터인 경우
- 벡터가 입력인 경우 편미분(partial differentiataion)을 사용합니다.
- 각 변수별로 편미분을 계산한 그레디언트 벡터(gradient vector, $\nabla f$)를 이용하여 경사하강법/경사상승법에 사용할 수 있습니다.
    - $\nabla f = (\partial x_1f, \partial x_2f,...,\partial x_df)$

### 경사하강법: 알고리즘

```python
"""
input: gradient, init, lr, eps
output: var
"""
var = init
grad = gradient(var)
while(norm(grad) > eps): # 벡터는 절댓값 대신 노름을 계산해서 종료조건을 설정한다
  var = var - lr * grad
  grad = gradient(var)
```

## 경사하강법으로 선형회귀 계수 구하기
- 데이터를 선형모델(linear model)로 해석하면 $X\beta=\hat{y} \approx y$이므로 $\beta$에 대해서 다음과 같이 정리할 수 있습니다.
    - $\beta$를 최소화하는 $min\|\|y-\hat{y}\|\|_2$ 식을 얻을 수 있습니다.
- 따라서 선형회귀의 목적식은 $min\|\|y-\hat{y}\|\|_2$이고 이를 최소화하는 $\beta$를 찾아야 하므로 다음과 같은 그레디언트 벡터를 구해야 합니다.
    - $\nabla_\beta \|\|y-X\beta\|\_2 = (\partial\_{\beta_1}\|\|y-X\beta\|\|\_2,...,\partial_{\beta_d}\|\|y-X\beta\|\|\_2)$
    - $\|\|y-X\beta\|\|_2$말고 $\|\|y-X\beta\|\|_2^2$를 최적화 해도 좋습니다.
- $\beta$에 대해 미분하게 되면 $\partial_{\beta_k}\|\|y-X\beta\|\|_2 = -\frac{X^T(y-X\beta)}{n\|\|y-X\beta\|\|_2}$인 식을 얻을 수 있습니다.
- 목적식을 최소화하는 $\beta$를 구하는 경사하강법 식은 다음과 같습니다.
    - $\beta^{(t+1)} \leftarrow \beta^{(t)}-\lambda \nabla_\beta \|\|y-X\beta^{(t)}\|\|_2$
    - 위의 식을 정리하면 $\beta^{(t+1)} \leftarrow \beta^{(t)}+\frac{2\lambda}{n}X^T(y-X\beta^{(t)})$인 식을 얻을 수 있습니다.
    
### 경사하강법 기반 선형회귀 알고리즘

```python
"""
input: X, y, lr, T(학습횟수)
Output: beta
"""
for t in range(T):
  error = y - X @ beta # (y - Xb)
  grad = - transpose(X) @ error 
  beta = beta - lr * grad
```

### 코드로 구현

```python
import numpy as np
# print('numpy version : {\%\s}'%(np.__version__))

train_x = (np.random.rand(1000) - 0.5) * 10 # 1000개의 랜덤 x값
train_y = np.zeros_like(train_x) # y값의 초기화

# f(x) = 5x+3으로 가정
for i in range(1000):
  train_y[i] = 5 * train_x[i] + 3

# parameter
w, b = 0.0, 0.0

lr_rate = 1e-2 # learning rate
errors = []

for i in range(100):
  pred_y = w * train_x + b # prediction

  # gradient
  grad_w = np.sum((pred_y - train_y) * train_x) / len(train_x) # (1)
  grad_b = np.sum(pred_y - train_y) / len(train_x) # (2)

  # update
  w -= lr_rate * grad_w
  b -= lr_rate * grad_b

  # error
  error = np.sum((pred_y - train_y)**2) / len(train_x)
  errors.append(error)

print('w: {}, b: {}'.format(w,b)) # w: 5.008559005692836, b: 1.9170264334235934
```

- grad_w, grad_b를 구하는 방법
  - error = $\frac{1}{n}(\hat{y}-y)^2$인데 이것을 $w$,$b$에 대해 편미분하면 다음과 같다.
  - $\frac{\partial err}{\partial w}=\frac{2}{n}(\hat{y}-y)X$
  - $\frac{\partial err}{\partial w}=\frac{2}{n}(\hat{y}-y)$
  - 따라서 $dw=((\hat{y}-y)X).mean$, $db=(\hat{y}-y).mean$이 된다.
- 에러를 그래프로 출력하면 다음과 같습니다.
![](https://drive.google.com/uc?export=view&id=1WEEOKyJg8Tv43sd0ZoqQ9WRhjeCRNQoz)

## 경사하강법은 만능?
- 이론적으로 경사하강법은 미분가능하고 볼록(convex)한 함수에 대해선 적절한 학습률과 학습횟수를 선택했을 때 수렴이 보장됩니다.
-  그러나 비선형회귀 문제의 경우 목적식이 볼록하지 않기 때문에 수렴이 항상 보장되지 않습니다.
    - 극소값의 위치를 찾았어도 최솟값의 위치가 아닐 수 있습니다.
- 이를 위해 확률적 경사하강법(stochastic gradient descent)를 사용하게 됩니다.

## 확률적 경사하강법
- 확률적 경사하강법은 데이터 한 개 또는 일부 활용하여 업데이트 합니다.
- 볼록이 아닌(non-convex) 목적식도 SGD를 통해 최적화할 수 있습니다.
- $\theta^{(t+1)} \leftarrow \theta^{(t)}-\hat{\nabla_\theta L(\theta^{(t)})}, \space E[\hat{\nabla_\theta L}] \approx \nabla_\theta L$
- 데이터의 일부를 가지고 패러미터를 업데이트하기 때문에 연산자원을 좀 더 효율적으로 활용할 수 있습니다.

### 확률적 경사하강법의 원리: 미니배치 연산
- 경사하강법은 전체데이터 $D = (X,y)$를 가지고 목적식의 그레디언트 벡터를 계산합니다.
- SGD는 미니배치 $D_{(b)}=(X_{(b)},y_{(b)}) \subset D$를 가지고 그레디언트 벡터를 계산합니다.
- 매번 다른 미니배치를 사용하기 때문에 목적식 모양(곡선 모양)이 바뀌게 됩니다.
    - 경사하강법의 경우 error가 줄어들 때 곡선형태로 줄어들지만, SGD는 요동치면서 줄어드는 것을 볼 수 있습니다.
- 또한, SGD는 볼록이 아닌 목적식에서도 사용가능하므로 경사하강법보다 머신러닝 학습에 더 효율적입니다.

### 코드로 구현

```python
import numpy as np
# print('numpy version : {\%\s}'%(np.__version__))

train_x = (np.random.rand(1000) - 0.5) * 10
train_y = np.zeros_like(train_x)

# f(x) = 5x+3으로 가정
for i in range(1000):
  train_y[i] = 5 * train_x[i] + 3

# parameter
w, b = 0.0, 0.0

lr_rate = 1e-2 # learning rate
errors = []

for i in range(100):
  # mini batch
  # 1000개 중 10개를 랜덤하게 뽑아 미니배치 생성
  idx = np.random.choice(1000,10,replace=False)
  mini_x = train_x[idx]
  mini_y = train_y[idx]

  pred_y = w * mini_x + b # prediction

  # gradient
  grad_w = np.sum((pred_y - mini_y) * mini_x) / len(mini_x)
  grad_b = np.sum(pred_y - mini_y) / len(mini_x)

  # update
  w -= lr_rate * grad_w
  b -= lr_rate * grad_b

  # error
  error = np.sum((pred_y - mini_y)**2) / len(mini_x)
  errors.append(error)

print('w: {}, b: {}'.format(w,b)) # w: 5.008246440224214, b: 1.9408158614474957
```

- 에러를 그래프로 출력하면 다음과 같다.
![](https://drive.google.com/uc?export=view&id=1oTfDOO8oFBbyWZim8zhkXazqs24DU70d)