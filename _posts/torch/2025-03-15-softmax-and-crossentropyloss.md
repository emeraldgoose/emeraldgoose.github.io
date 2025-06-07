---
title: Python으로 Softmax와 CrossEntropyLoss 바닥부터 구현하기
categories:
  - Pytorch
tags: [torch]
---
## Softmax
Softmax는 입력받은 값을 확률로 변환하는 함수입니다. 입력 값을 0과 1사이의 확률값으로 변환하고 총합은 항상 1이 되는 특징을 가집니다. 주로 딥러닝에서 마지막 출력층의 활성화함수로 사용되어 각 클래스에 속할 확률을 계산하는데 사용합니다. 그리고 지수함수를 사용하기 때문에 큰 값에 대해 오버플로우가 발생할 수 있습니다.

### Forward
Softmax의 수식은 다음과 같습니다.

$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$

이 수식을 구현한 코드는 다음과 같습니다.

<script src="https://gist.github.com/emeraldgoose/16326706b8cc37c31eb8da0ae27e97b1.js"></script>

위와 같이 구현한 이유는 안정성때문입니다. x값이 너무 크면 오버플로우가 발생할 수 있기 때문에 최대값을 0으로(`np.exp(x) = 1`) 보정하는 작업을 수행합니다. `np.exp(x)`를 `np.exp(x - np.max(x))`로 구현하더라도 변화량은 같기 때문에 최종 소프트맥스 결과값은 같습니다.

### Backward
Softmax 함수를 미분하는 과정은 다음과 같습니다.

$\frac{\partial S(x_i)}{\partial x_k} = \frac{\partial}{\partial x_k} \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{(\frac{\partial}{\partial x_k}e^{x_i})\sum_j e^{x_j} - e^{x_i}(\frac{\partial}{\partial x_k}\sum_j e^{x_j})}{(\sum_j e^{x_j})^2}$

이때, 다음 두 가지를 생각해볼 수 있습니다.

1) $k = i$

$\frac{\partial}{\partial x_i} \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i} \sum_j e^{x_j} \\ - \\ e^{x_i} e^{x_i}} {(\sum_j e^{x_j})^2} = \frac{e^{x_i}(\sum_j e^{x_j} - e^{x_i})}{(\sum_j e^{x_j})^2} = \frac{e^{x_i}}{\sum_j e^{x_j}} (1 - \frac{e^{x_i}}{\sum_j e^{x_j}}) = S_{x_i}(1-S_{x_i})$

2) $k \neq i$

$\frac{\partial}{\partial x_i} \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{0 \\ - \\ e^{x_i} \\ \frac{\partial}{\partial x_k}\sum_j e^{x_j}}{(\sum_j e^{x_j})^2} = \frac{- e^{x_i} e^{x_k}}{(\sum_j e^{x_j})^2} = - \frac{e^{x_i}}{\sum_j e^{x_j}} \\ \frac{e^{x_k}}{\sum_j e^{x_j}} = -S_{x_i} \\ S_{x_k}$

따라서 Jacobian 행렬이 생성되고 이 행렬이 Softmax의 기울기가 됩니다.

$Jacobian = \begin{cases} 
S_{x_i}(1-S_{x_i}) & i = k \\\\ 
-S_{x_i} S_{x_k} & i \neq k 
\end{cases}$

Softmax 함수의 입력에 대한 기울기를 구하는 코드는 다음과 같습니다.

<script src="https://gist.github.com/emeraldgoose/a1bb6f44b227ca37a451612f68213223.js"></script>

**축 변경**  
아래처럼 계산을 용이하게 하기 위해 축을 변경합니다. 적용하고자 하는 축을 마지막 축과 바꿔줍니다. 

이때 `np.reshape`를 사용하는 것은 추천하지 않습니다. `np.reshape`함수는 배열의 원소 순서를 고려하지 않고 모양만 변경합니다. 물리적 연속성을 보장하지 않기 때문에 저장한 정보가 뒤틀릴 수 있습니다. 그래서 `np.transpose`를 사용해 물리적인 연속성을 보존하면서 축만 바꿔주게 합니다.

```python
transposed_axes = list(range(dz.ndim))
transposed_axes[self.dim], transposed_axes[-1] = transposed_axes[-1], transposed_axes[self.dim]
transposed_dout = np.transpose(dz, transposed_axes)
transposed_softmax = np.transpose(self.output, transposed_axes)
transposed_dx = np.transpose(dx, transposed_axes)
```

**Jacobian 행렬과 dz의 행렬곱**  
다음, softmax 값을 이용해 대각행렬(`np.diagflat`)을 구하고 $S^2$ 값을 빼서 Jacobian 행렬을 만든 뒤 축이 변환된 dz와 곱해줍니다. 
마지막은 축을 원상태로 돌려줍니다.
```python
for idx in np.ndindex(batch_size):
    s = transposed_softmax[idx].reshape(-1, 1)
    jacobian = np.diagflat(s) - np.dot(s, s.T)
    transposed_dx[idx] = np.dot(jacobian, transposed_dout[idx])

dx = np.transpose(transposed_dx, transposed_axes)
```

## CrossEntropyLoss
CrossEntropyLoss는 모델이 예측한 확률분포와 데이터의 확률분포간 차이를 줄이도록 유도하는 손실함수입니다. 

교차 엔트로피의 수식은 $H(p,q) = -\sum p(x) log(q(x))$이고 $p(x)$는 실제 사건의 확률 분포, $q(x)$는 모델이 예측한 확률분포를 말합니다.

$p(x)log(q(x))$는 실제 확률 $p(x)$가 높은 사건 $x$에 대해 모델이 예측한 확률 $q(x)$가 낮게 예측된다면, log에 의해 음수방향으로 값이 커지게 됩니다. 
여기에 음수를 취했기 때문에 모델이 잘못 예측할 수록 교차 엔트로피가 커져 실제 확률분포와 예측 확률분포의 차이가 크다는 것을 의미하게 됩니다.

### Forward
CrossEntropyLoss의 수식은 다음과 같습니다.

$L = - \sum_{i=1}^c y_i log(p(x_i))$

$y_i$는 실제 데이터의 One-hot encoding이고 $p(x_i)$는 $\text{Softmax}(x_i)$를 의미합니다.

<script src="https://gist.github.com/emeraldgoose/8917e4f3ab587bb59e53828cc8004b81.js"></script>

### Backward
$y$는 정답 레이블인 경우 1, 아니면 0의 값을 가지고 있습니다. 

따라서 k가 정답 레이블이라고 가정하면 $L = - \sum_{i=1}^c y_i log(p(x_i)) = -log(p(x_k))$로 유도할 수 있습니다.

이때, 기울기를 다음과 같이 유도할 수 있습니다.

$\frac{\partial L}{\partial x_i} = \frac{\partial}{\partial x_i}(- log p(x_k)) = - \frac{1}{p(x_k)}\frac{\partial p(x_k)}{\partial x_i}$

1) 정답인 경우, $i = k$

$\frac{\partial p(x_k)}{\partial x_k} = p(x_k)(1 - p(x_k)) \rightarrow -\frac{1}{p(x_k)} p(x_k)(1 - p(x_k)) = -(1 - p(x_k)) = p(x_k) - 1 = p(x_i) - 1$

2) 정답이 아닌 경우, $i \neq k$

$\frac{\partial p(x_k)}{\partial x_k} = -p(x_k)p(x_i) \rightarrow -\frac{1}{p(x_k)} -p(x_k)p(x_i) = p(x_i) = p(x_i) - 0$

유도된 식을 살펴보면 $p(x_i)$에 정답인 경우 1, 정답이 아닌 경우 0을 빼주고 있으므로 $y$로 치환할 수 있습니다. 따라서, CrossEntropyLoss의 기울기는 $p(x_i) - y_i$입니다.

<script src="https://gist.github.com/emeraldgoose/139b6199df3edfa26a078bfb20712645.js"></script>

## Code
- [Softmax Class](https://github.com/emeraldgoose/hcrot/blob/master/hcrot/layers/activation.py#L10)
- [CrossEntropyLoss](https://github.com/emeraldgoose/hcrot/blob/master/hcrot/layers/loss.py#L32)