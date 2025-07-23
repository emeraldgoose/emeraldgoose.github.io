---
title: Python으로 Transformer 바닥부터 구현하기[1] (MultiHead-Attention, LayerNorm, GELU)
categories:
  - Pytorch
tags: [torch]
---
# Transformer
트랜스포머(Transformer)는 2017년에 등장한 모델입니다. [Attention is all you need](https://arxiv.org/abs/1706.03762) 논문은 트랜스포머 모델에 대해 설명하고 있으며 이후 BERT, GPT라는 새로운 모델을 탄생시키는 배경이 됩니다.

# Scaled Dot-Product Attention
트랜스포머 모델은 셀프 어텐션(Self-Attention) 매커니즘을 사용하고 있습니다. 셀프 어텐션 매커니즘은 자기 자신에 대해 어텐션 밸류를 구하는 방법으로 문장 내 단어간의 관계를 학습할 수 있고 병렬로 처리가능한 점이 장점입니다. 트랜스포머 등장 이전 모델의 경우 단어가 나타나는 시간 흐름에 따라 정보를 파악해야 하는 한계가 있었지만 어텐션을 통해 시간 흐름에 상관없이 관계를 학습할 수 있는 방법이 되었습니다.

Self Attention 매커니즘의 핵심은 Scaled Dot Product Attention입니다. Dot-product한 값을 Scale Down하여 어텐션을 구하는 매커니즘으로 하나의 단어마다 계산하는 것이 아닌 문장 전체 단어에 대해 어텐션을 계산할 수 있습니다.

## Forward
SDPA의 forward 수식을 살펴보겠습니다.

$Attention(Q,K,V) = \text{softmax}(\frac{QK^{\top}}{\sqrt{d_k}})V$

Query는 현재 단어의 정보를 말합니다. 문장 내 임의의 단어들의 정보가 Query로 사용되면 다른 단어의 정보를 담은 Key와의 유사도(Similarity) 연산을 Dot-product로 수행하여 Softmax 연산을 통해 Attention Score를 구하게 됩니다. 스코어와 Value를 곱하여 Attention Value를 계산하게되는데 이것은 스코어를 이용해 Value에서 얼마나 정보를 가져올 것인지 결정하는 것입니다.  

### Scaling
Query와 Key와의 유사도 연산 후에 Scale Down을 하는 이유는 Dot-product 후 값이 너무 커지는 것을 방지하기 위함입니다. $d_k$가 길어질 수록 $QK^{\top}$의 값도 같이 커져 기울기가 0이 되어 gradient vanishing 문제가 발생할 수 있기 때문에 스케일링이 필요합니다.

다음의 간단한 코드와 그래프로 스케일링 전후의 차이를 쉽게 이해할 수 있습니다.

<script src="https://gist.github.com/emeraldgoose/0e868296f2d457d433b4b7cfbae6d45c.js"></script>

간단하게 랜덤한 Query와 Key가 주어질 때 dot-product를 수행하고 softmax를 실행하면 다음의 왼쪽 그래프를 얻을 수 있습니다. 그러나 위 코드에서 `softmax(np.dot(Q, k.T) / np.sqrt(d_k))`로 수정하게 되면 다음의 오른쪽 그래프를 얻을 수 있습니다.

<figure class="half">
  <a href="https://1drv.ms/i/s!AoC6BbMk0S9Qm0ld7x9iBmT1MwWP?embed=1&width=826&height=813" data-lightbox="gallery"><img src="https://1drv.ms/i/s!AoC6BbMk0S9Qm0ld7x9iBmT1MwWP?embed=1&width=826&height=813" alt="01"></a>
  <a href="https://1drv.ms/i/s!AoC6BbMk0S9Qm0tbtiMJQyPGzMUW?embed=1&width=826&height=813" data-lightbox="gallery"><img src="https://1drv.ms/i/s!AoC6BbMk0S9Qm0tbtiMJQyPGzMUW?embed=1&width=826&height=813" alt="02"></a>
</figure>

왼쪽 그래프에서 9번 확률이 1.0이지만 오른쪽 그래프에서는 0.4x로 줄고 나머지 번호들의 정보들이 드러나는 것을 볼 수 있습니다. 즉, softmax로 정보가 사라지는 문제를 스케일링으로 해결할 수 있습니다.

또한, $d_k$를 사용하는 이유는 다음과 같습니다.

Query와 Key가 가우시안 분포를 따른다고 가정하면 평균은 0, 분산은 1을 가지게 됩니다. $QK^{\top} = \sum_{i}^{d_k}Q_iK_i$이고 Query와 Key는 independent하므로 평균은 0, 분산은 $d_k$를 가지게 됩니다. 따라서 $\sqrt{d_k}$로 스케일링하여 분산을 1로 줄일 수 있게 됩니다.

<script src="https://gist.github.com/emeraldgoose/a23756410bc023d518cabaef8e5e1945.js"></script>

attention padding mask 처럼 어텐션 스코어에 마스킹을 위한 masked_fill 함수는 `numpy.ma.array`를 이용하여 구현했습니다.

## Backward
Backward 함수를 구현하기 위해 Attention 계산에 참여하는 Query, Key, Value 세 가지 입력에 대한 기울기가 필요합니다.

먼저, Attention forward 수식을 나눠서 생각해보겠습니다.

$1. \\ A = \frac{QK^{\top}}{\sqrt{d_k}}$  
$2. \\ W = \text{softmax}(A)$  
$3. \\ Z = WV$

1번 수식으로부터 $\frac{\partial A}{\partial Q}$와 $\frac{\partial A}{\partial K}$를 구할 수 있습니다.

$\frac{\partial A}{\partial Q} = \frac{K}{\sqrt{d_k}}$

$\frac{\partial A}{\partial K} = \frac{Q}{\sqrt{d_k}}$

2번 수식에서 얻을 수 있는 $\frac{\partial W}{\partial A}$는 softmax의 미분결과이므로 W(1 - W)로 가정해보겠습니다.(실제 계산은 W(1 - W)가 아닙니다. 저는 미리 구현해둔 softmax 클래스를 이용할 것입니다.)

또한, 3번 수식으로부터 $\frac{\partial Z}{\partial W} = V^{\top}$도 알 수 있습니다.

다음, Value에 대한 기울기부터 계산합니다. upstream gradient를 $dZ$라 가정하겠습니다.  

$\frac{\partial L}{\partial V} = \frac{\partial L}{\partial Z} \frac{\partial Z}{\partial V} = W^{\top} dZ$

다음, Query와 Key에 대한 기울기를 계산합니다.  

$\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial Z} \frac{\partial Z}{\partial W} \frac{\partial W}{\partial A} \frac{\partial A}{\partial Q} = dZ \cdot W(1-W) \cdot \frac{K}{\sqrt{d_k}}$

$\frac{\partial L}{\partial K} = \frac{\partial L}{\partial Z} \frac{\partial Z}{\partial W} \frac{\partial W}{\partial A} \frac{\partial A}{\partial K} = dZ \cdot W(1-W) \cdot \frac{Q}{\sqrt{d_k}}$

<script src="https://gist.github.com/emeraldgoose/4ff45c20339d57fef96c3206d66583b2.js"></script>

dK를 계산할 때는 사이즈를 맞추기 위해 einsum을 사용했습니다.

# MultiHead Attention
MultiHead Attention은 Attention을 여러개의 head가 동시에 수행하는 모듈입니다. 하나의 문장에 대해 관계에 집중하는 어텐션 혹은 단어에 집중하는 있는 어텐션 등을 학습할 수 있고 이러한 어텐션을 병렬로 처리하기 위해 구현되었습니다.

pytorch의 경우 MultiHead Attention의 파라미터는 self-attention인 경우와 아닌 경우 다르게 파라미터를 선언하도록 구현되었습니다. self-attention인 경우 $W_Q$, $W_K$, $W_V$가 concat된 하나의 weight(`in_proj_weight`, `in_proj_bias`)로 선언하고 최종 계산된 값을 3개의 청크로 나눠 리턴하도록 합니다.

하지만, 트랜스포머를 구현하면서 모든 어텐션을 셀프 어텐션으로 사용하지 않으므로 구현을 간단히 하기 위해 q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight로 따로 선언하여 구현했습니다.

## Forward
MultiHead Attention의 계산 수식은 다음과 같습니다.

$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1,...,head_h)W^O$, $head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

MultiHead Attention 모듈의 `embed_dim`은 모델의 전체 차원을 말합니다. 이것을 head의 개수(`num_heads`)로 나눈 값을 각 head에서 사용될 차원 `head_dim`으로 사용하게 됩니다. 따라서, `embed_dim`은 `num_heads`의 배수로 설정해야 합니다.

<script src="https://gist.github.com/emeraldgoose/b8d3081f5885ead2f962bdcd337fbd47.js"></script>

`attn_output_transpose`는 backward에서 shape가 필요하기 때문에 저장했고 `attn_output_reshaped`는 backward에서 `d_out_proj_weight`를 계산하기 위해 저장했습니다. 차원을 맞춰주는 것에 주의하면서 구현해야 했습니다.

## Backward
backward 구현은 forward의 역순으로 구성하면 됩니다. 역시 차원을 맞춰주는 것이 가장 중요하기 때문에 이 부분만 주의하면 됩니다.

MultiHeadAttention 모듈은 입력값으로 Query, Key, Value가 들어가므로 당연히 Query에 대한 기울기, Key에 대한 기울기, Value에 대한 기울기을 리턴해야 합니다. 만약 Self-attention이라면 세 리턴값을 더한 값이 입력에 대한 기울기가 됩니다.

<script src="https://gist.github.com/emeraldgoose/ad345617843049ba9dee9fdfcb5536dc.js"></script>

# LayerNorm
Normalization은 입력값에 대해 정규화하기 위한 과정입니다. 학습 데이터를 배치 단위로 학습할 때, 랜덤하게 샘플링된 배치들의 데이터 분포가 다를 수 있습니다. 이러한 배치를 그대로 학습하게 되면 그에 맞게 네트워크는 가중치를 조절하게 되는데 이로 인해 학습에 들이는 시간이 증가될 수 있습니다. 따라서, 정규화를 통해 데이터 분포를 조정하여 안정적이고 학습이 빠르게 이루어지도록 하는 기법이 Normalization입니다.

배치 단위로 정규화를 진행하는 Batch Normalization(BN), 데이터 feature 단위로 정규화를 진행하는 Layer Normalization(LN), BatchNorm과 LayerNorm의 특징을 섞은 Group Normalization(GN)이 있습니다. 여기서는 Layer Normalization을 구현합니다.

Layer Normalization은 각 샘플마다 모든 feature들에 대해 평균을 0, 분산을 1로 조정하여 배치 사이즈에 의존적이지 않게 됩니다.

또한, RNN 계열의 모델은 BN을 사용하기 어렵습니다. 왜냐하면, Sequential한 모델은 각 타임스텝에 따라 다른 데이터가 들어와야 하고 길이가 다른 경우 padding 처리가 필요했습니다. 그래서 배치 정규화를 적용하기 어려웠기 때문에 LN을 적용합니다. 

## Forward
LN의 수식은 다음과 같습니다.

$y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta$

$\gamma$와 $\beta$는 학습가능한 파라미터입니다. $\gamma$와 $\beta$를 사용하는 이유는 Normalization으로 인해 데이터 특성이 사라지는데 이것이 학습하는데 그리 좋지 못합니다. 그래서 파라미터를 추가하여 네트워크가 학습하기 좋도록 스케일링, 시프트 작업을 수행하기 위해 $\gamma$
와 $\beta$가 추가되었습니다.

<script src="https://gist.github.com/emeraldgoose/819c446042184f37fd2a7086286a1061.js"></script>

## Backward
먼저, 수식을 다음과 같이 분리하여 생각해보겠습니다.

1. $\mu = \frac{1}{N}\sum x$

2. $\sigma^2 = \frac{1}{N}\sum(x - \mu)^2$

3. $\hat{x} = \frac{x-\mu}{\sqrt{\sigma^2 + \epsilon}}$

4. $y = \gamma \hat{x} + \beta$

먼저, $\hat{x}$에 대한 기울기를 계산하겠습니다.

$\frac{\partial L}{\partial \hat{x}} = \gamma$

다음, 분산에 대한 기울기를 계산하겠습니다.

$\frac{\partial L}{\partial \sigma^2} = \frac{\partial L}{\partial \hat{x}} \cdot \frac{\partial \hat{x}}{\partial \sigma^2}$

$\frac{\partial \hat{x}}{\partial \sigma^2} = -\frac{1}{2}(x - \mu)(\sigma^2 + \epsilon)^{-\frac{3}{2}}$

$\frac{\partial L}{\partial \sigma^2} = \frac{\partial L}{\partial \hat{x}} \cdot -\frac{1}{2}(x - \mu)(\sigma^2 + \epsilon)^{-\frac{3}{2}}$

다음, 평균에 대한 기울기를 계산하겠습니다. 2번에서 분산을 계산할 때 평균이 참여하므로 분산을 평균으로 미분하여 얻은 기울기도 추가되어야 합니다.

$\frac{\partial L}{\partial \mu} = \frac{\partial L}{\partial \hat{x}} \cdot \frac{\partial \hat{x}}{\partial \mu} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\partial \sigma^2}{\partial \mu}$

$\frac{\partial \sigma^2}{\partial \mu} = -\frac{2}{N}(x - \mu)$

$\frac{\partial L}{\partial \mu} = \frac{\partial L}{\partial \hat{x}} \cdot -\frac{1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial L}{\partial \sigma^2} \cdot -\frac{2}{N}(x-\mu)$

마지막으로, x에 대한 기울기를 계산하겠습니다.

$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial \hat{x}} \cdot \frac{\partial \hat{x}}{\partial x} + \frac{\partial L}{\partial \mu} \cdot \frac{\partial \mu}{\partial x} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\partial \sigma^2}{\partial x}$

$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial \hat{x}} \cdot \frac{1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{2}{N}(x - \mu) + \frac{\partial L}{\partial \mu} \cdot \frac{1}{N}$

<script src="https://gist.github.com/emeraldgoose/7d25c8808ddbbe96eab4e348ed8ef020.js"></script>

코드로 구현할 때는 epsilon 값을 0으로 가정하여 구현되었습니다.

# GELU
GELU는 ReLU의 단점을 보완한 활성화 함수입니다.

ReLU는 다음의 그래프와 같이 입력값이 음수일 경우 0을 출력하여 뉴런이 죽는 현상인 Dying ReLU 문제가 발생했습니다. 입력된 값이 음수가 한번이라도 등장하게 되면 활성화 함수의 기울기가 0이므로 그 뉴런과 연결된 다른 뉴런들 또한 기울기가 0이 됩니다. 대안으로 음수 입력에도 기울기를 제공하는 leaky relu가 있지만 음수에 대한 boundary가 없기 때문에 어떠한 음수값에 대해서도 기울기를 제공해버립니다. 그래서 swish나 GELU와 같이 음수 boundary를 제공하는 활성화 함수가 제안되었습니다.

<figure class="half">
  <a href="https://1drv.ms/i/s!AoC6BbMk0S9Qm0o_jVY2iJmV4LNq?embed=1&width=640&height=480" data-lightbox="gallery"><img src="https://1drv.ms/i/s!AoC6BbMk0S9Qm0o_jVY2iJmV4LNq?embed=1&width=640&height=480" alt="03"></a>
  <a href="https://1drv.ms/i/s!AoC6BbMk0S9Qm0yGxsrvrjyLkgip?embed=1&width=640&height=480" data-lightbox="gallery"><img src="https://1drv.ms/i/s!AoC6BbMk0S9Qm0yGxsrvrjyLkgip?embed=1&width=640&height=480" alt="04"></a>
  <figcaption>Refs: https://pytorch.org</figcaption>
</figure>

하지만, NLP 태스크에서 꼭 GELU여야 하는 이유는 잘 모르겠습니다. 다른 블로그에서는 "ReLU나 ELU보다 GELU가 성능이 좋았다"정도로만 다들 기록하고 있습니다.  
가우시안 분포의 연속성이 임베딩와 같은 벡터 학습에 유리하기 때문일 수도 있지만 정확한 이유를 찾기 어려웠습니다.

## Forward
GELU는 가우시안 분포의 누적 분포 함수(CDF)와 입력값을 곱하는 활성화 함수입니다.

$\text{GELU}(x) = x \cdot \Phi(x)$

제가 구현할 때는 tanh로 근사한 수식을 사용했습니다.

$\text{GELU}(x) = 0.5 * x * (1 + \text{tanh}(\sqrt{2/\pi} * (x + 0.044715 * x^3)))$

<script src="https://gist.github.com/emeraldgoose/400eedea60daf0c9ecfd7854b0c1e3db.js"></script>

## Backward
미분을 하기 위해 수식을 두개로 나누어 생각해보겠습니다.

1. $\text{GELU}(x) = 0.5x(1 + \text{tanh}(y))$

2. $y = \sqrt{\frac{2}{\pi}}(x + 0.044715x^3)$

먼저, 1번 수식에서 $\hat{x}$에 대한 기울기를 계산합니다.

$\frac{\partial \text{GELU}}{\partial x} = 0.5(1 + \text{tanh}(y)) + 0.5x\frac{\partial \text{tanh}(y)}{\partial x}$

$\frac{\partial \text{tanh}(y)}{\partial x} = (1 - \text{tanh}^2(y))\frac{\partial y}{\partial x}$

다음, 2번 수식에서 x에 대한 기울기를 계산합니다.

$\frac{\partial y}{\partial x} = \sqrt{\frac{2}{\pi}}(1 + 0.044715 * 3 * x^2)$

마지막으로, 정리합니다.

$\frac{\partial \text{GELU}}{\partial x} = 0.5(1 + \text{tanh}(y)) + 0.5x(1 - \text{tanh}^2(y))\sqrt{\frac{2}{\pi}}(1 + 0.044715 * 3 * x^2)$

<script src="https://gist.github.com/emeraldgoose/aaa360c241799e50ea55448c9026a9dc.js"></script>

# Reference
- [torch.nn.functional.scaled_dot_product_attention.html](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention)
- [why-does-this-multiplication-of-q-and-k-have-a-variance-of-d-k-in-scaled](https://ai.stackexchange.com/questions/21237/why-does-this-multiplication-of-q-and-k-have-a-variance-of-d-k-in-scaled)
- [Understanding LSTM Networks](https://www.blossominkyung.com/deeplearning/transformer-mha)
- [MultiheadAttention - Pytorch Docs](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [LayerNorm - Pytorch Docs](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
- [Batch Normalization을 제대로 이해해보자](https://lifeignite.tistory.com/47)
- [GELU - Pytorch Docs](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)
- [#1 Dying ReLU](https://brunch.co.kr/@kdh7575070/27)