---
title: Python으로 Transformer 바닥부터 구현하기[2] (Transformer)
categories:
  - Pytorch
tags: [torch]
---
# Objective
앞에서 구현한 LayerNorm, MultiHeadAttention, GELU를 사용하고 이전에 구현해둔 Linear, Dropout, Softmax 클래스를 사용하여 Transformer 클래스를 구현하여 테스트해봅니다.
> [Python으로 Transformer 바닥부터 구현하기[1] (MultiHead-Attention, LayerNorm, GELU)](https://emeraldgoose.github.io/pytorch/transformer-scratch-implementation-1/)

가장 바깥쪽에 위치한 Transformer부터 시작해서 EncoderLayer, DecoderLayer 순으로 설명하고자 합니다.

# Transformer
Transformer 클래스의 구조는 TransformerEncoder와 TransformerDeocder로 구성됩니다. Transformer로 들어오는 입력은 인코더를 통해 인코딩되어 디코더의 입력으로 사용됩니다.

<figure style="text-align:center;">
    <a href="https://1drv.ms/i/s!AoC6BbMk0S9Qm07_7PVAiXxoixjr?embed=1&width=798&height=1100" data-lightbox="gallery"><img style="width:50%;" src="https://1drv.ms/i/s!AoC6BbMk0S9Qm07_7PVAiXxoixjr?embed=1&width=798&height=1100" alt="01"></a>
    <figcaption>Transformer Architecture</figcaption>
</figure>

## Forward
Transformer 클래스를 구현하기 위해 TransformerEncoder와 TransformerDecoder에서 사용할 encoder_layer, encoder_norm, decoder_layer, decoder_norm을 선언합니다.

<script src="https://gist.github.com/emeraldgoose/ba48acb781e5437b6585d18d57ecd83e.js"></script>

## Backward
Backward에서는 Encoder와 Decoder의 backward 함수를 호출하고 리턴되는 기울기들을 저장합니다.

<script src="https://gist.github.com/emeraldgoose/a9b4402ce637590185e56919604d6efe.js"></script>

# TransformerEncoder
Transformer의 Encoder는 EncoderLayer들이 스택되어 있는 구조로 구현됩니다.

## Forward
Pytorch의 TransformerEncoder 클래스는 인코더 레이어를 `num_layers`만큼 복사하여 ModuleList로 구성합니다. Transformer 클래스에서 선언된 EncoderLayer를 `_get_clone` 함수에서 `copy.deepcopy()`로 복사하기 때문에 스택되어 있는 인코더 레이어들은 같은 초기 파라미터를 가지고 다른 기울기를 가지게 됩니다.

<script src="https://gist.github.com/emeraldgoose/c3997f35f4a3e76a6da5d697109de5ae.js"></script>

## Backward
Forward에서 반복문을 통해 순서대로 계산하고 있으므로 그 역순으로 Backward 함수를 불러 계산하고 각 레이어의 기울기를 저장합니다.

<script src="https://gist.github.com/emeraldgoose/734b8145dc245a8199e3f3a271081ffe.js"></script>

# TransformerDecoder
Transformer의 Decoder는 DecoderLayer들이 스택되어 있는 구조로 구현됩니다. 다음 그림의 오른쪽 처럼 Decoder는 Output 임베딩과 인코딩 정보를 입력으로 받아 출력값을 계산합니다.

## Forward
forward 함수의 argument로 `tgt`와 `memory`가 있습니다. `tgt`는 output 임베딩을 말하고 `memory`는 인코더 출력을 말합니다. Encoder 구현과 마찬가지로 Transformer 클래스에서 선언된 DecoderLayer를 복사하여 ModuleList로 구성하고 반복문을 통해 호출하여 계산합니다.

<script src="https://gist.github.com/emeraldgoose/23d416bbbe0732af2d080e7c1aa4f1eb.js"></script>

## Backward
Foward에서 반복문을 통해 순서대로 계산하고 있으므로 그 역순으로 Backward 함수를 불러 계산하고 각 레이어의 기울기를 저장합니다.

<script src="https://gist.github.com/emeraldgoose/52a3b5ed4af3d7a7bbb7606455d9e39c.js"></script>

# TransformerEncoderLayer
Transformer의 인코딩을 담당하는 레이어입니다. TransformerEncoder로 들어온 입력은 EncoderLayer의 순서대로 처리되며 최종 출력은 Decoder에서 사용됩니다.

## Forward
<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQREv7BEX40uQLTR1Sy8KDZXAWUehzE-LH83Rw4WIW0oQm0?width=1024" data-lightbox="gallery" style="width:120%;"><img style="width:30%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQREv7BEX40uQLTR1Sy8KDZXAWUehzE-LH83Rw4WIW0oQm0?width=1024" alt="01"></a>
    <figcaption>Encoder Layer forward</figcaption>
</figure>

계산 순서는 들어온 입력이 먼저 MultiheadAttention을 거치고 FeedForward 연산을 통해 인코딩됩니다. 각 결과는 Residual Connection 구조를 사용하여 입력과 더해준 후 Layer Normalization을 수행합니다.
<script src="https://gist.github.com/emeraldgoose/511db6e7427152cf747b55f2982e3570.js"></script>

## Backward
<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQRVtS1PfLIFTJ0K8RzrSNlwAc9q9HElWPxwrEcgZJQ-LzY?width=1024" data-lightbox="gallery" style="width:120%;"><img style="width:30%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQRVtS1PfLIFTJ0K8RzrSNlwAc9q9HElWPxwrEcgZJQ-LzY?width=1024" alt="01"></a>
    <figcaption>Encoder Layer backward</figcaption>
</figure>

Backward 연산은 Forward의 역순으로 진행되며 Forward에서 사용된 Residual Connection은 Backward에서는 upstream gradient와 더해지게 됩니다.
<script src="https://gist.github.com/emeraldgoose/5ce3f68358a1d4c2b1a0b9151981f1e7.js"></script>

# TransformerDecoderLayer
Transformer의 디코딩을 담당하는 레이어입니다. 인코더의 출력을 디코더에서 사용하여 output 시퀀스 이후에 나올 토큰을 예측하게 됩니다.

## Forward
<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQSOmRpxl0G5RINabXS-mKZIAVyiQITuF0imLAiB-pNmHFM?width=1024" data-lightbox="gallery" style="width:120%;"><img style="width:30%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQSOmRpxl0G5RINabXS-mKZIAVyiQITuF0imLAiB-pNmHFM?width=1024" alt="01"></a>
    <figcaption>Encoder Layer backward</figcaption>
</figure>
forward의 argument로 `tgt`와 `memory`가 있습니다. `tgt`는 output 임베딩 입력을 담당하고 `memory`는 인코더의 출력을 의미합니다. EncoderLayer와 마찬가지로 각 단계마다 Residual Connection 구조를 사용합니다.

<script src="https://gist.github.com/emeraldgoose/d55c7cf1dda2b952a0e3ac3610f2f84d.js"></script>

## Backward
<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQSAbNENQ2jXTJqRZ7lQDx60AZf6AFmNJIuMybiJdB0bh60?width=1024" data-lightbox="gallery" style="width:120%;"><img style="width:30%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQSAbNENQ2jXTJqRZ7lQDx60AZf6AFmNJIuMybiJdB0bh60?width=1024" alt="01"></a>
    <figcaption>Encoder Layer backward</figcaption>
</figure>
Backward 연산은 Forward의 역순으로 진행되며 Residual Connection은 upstream gradient와 더해지게 됩니다.

<script src="https://gist.github.com/emeraldgoose/be7a478bcd7b067516a81ca37e4d2239.js"></script>

# Test
## MNIST Classification
이전 테스트와 마찬가지로 MNIST 5000장과 테스트 1000장으로 실험했습니다. hidden_size는 32, learning_rate는 1e-3, 10 epoch로 학습을 진행했습니다.

다음은 학습에 사용한 모델을 정의한 코드입니다.

<script src="https://gist.github.com/emeraldgoose/b998ee81096e78ccc7694291df5f242e.js"></script>

MNIST 이미지에 순서 정보를 주기 위해 positional encoding 정보를 추가했습니다. 그리고 Transformer의 출력값이 (batch_size, 28, embed_size)이므로 Linear 레이어로 통과시키게 되면 (batch_size, 28, 10)이 되어버리기 때문에 Flatten 레이어를 통해 (batch_size, 28 * embed_size)로 바꿔준 후 Linear 레이어를 통해 (batch_size, 10) 크기를 가진 logits 값으로 출력하도록 모델을 구성했습니다.

아래 그래프들은 학습시킨 결과입니다. 왼쪽 그래프는 loss, 오른쪽 그래프는 accuracy를 기록한 것입니다.

<figure class="half">
  <a href="https://1drv.ms/i/s!AoC6BbMk0S9Qm08fTVKlHgRMYqb4?embed=1&width=846&height=833" data-lightbox="gallery"><img src="https://1drv.ms/i/s!AoC6BbMk0S9Qm08fTVKlHgRMYqb4?embed=1&width=846&height=833" alt="02"></a>
  <a href="https://1drv.ms/i/s!AoC6BbMk0S9Qm03PeZKru6RF6coJ?embed=1&width=841&height=833" data-lightbox="gallery"><img src="https://1drv.ms/i/s!AoC6BbMk0S9Qm03PeZKru6RF6coJ?embed=1&width=841&height=833" alt="03"></a>
</figure>

hidden size가 크지 않았지만 잘 학습되는 것을 볼 수 있습니다. hidden size를 256으로 올리고 학습을 돌려보면 accuracy가 0.95 이상으로 올라가기도 합니다.

## GPT
디코더만을 이용해 다음 토큰을 예측하도록 모델을 구성할 수 있습니다.

<script src="https://gist.github.com/emeraldgoose/e780545e68eac50bcf2b21000ccdd829.js"></script>

사용된 파라미터와 학습 문장은 다음과 같습니다.

```
embed_size=256, num_heads=4, num_layers=3, learning_rate=1e-4, epochs=50
```
```
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be that is the question",
    "All that glitters is not gold but it is very valuable",
    "Knowledge is power but enthusiasm pulls the switch",
    "The only thing we have to fear is fear itself",
    "In the end we will remember not the words of our enemies",
    "Life is what happens when you’re busy making other plans",
    "To succeed in life you need two things ignorance and confidence",
    "The future belongs to those who believe in the beauty of their dreams"
]
```

다음 토큰을 예측할 때는 간단하게 구현하기 위해 logits의 가장 높은 확률을 가진 토큰을 선택했고 다음의 코드를 사용하여 start_sequence 이후의 문장을 생성해봤습니다.

<script src="https://gist.github.com/emeraldgoose/02072ada41a9bcc83c962c98693ca3f1.js"></script>

결과는 다음과 같습니다.

```
Input : The
Output: The future belongs to those who believe in the <eos>

Input : The quick
Output : The quick brown <eos>

Input : To
Output: To over over over over over over over over over over over over over over quick over of our of our of our jumps over our enemies <eos>

Input : To be
Output : To be not to be or not to be or not <eos>
```

Output을 보면 학습 문장과 관련있는 토큰을 생성하는 것을 볼 수 있습니다. 시작 문장이 "To"인 경우 over는 관련이 없지만 그 이후 등장하는 토큰들이 over에 의해 첫 번째 학습 문장과 관련있는 것을 볼 수 있습니다.

여러번 생성할 때마다 다른 문장들이 등장하는 랜덤성과 Transformer 모델 특징답게 토큰이 반복적으로 등장하는 문제도 볼 수 있습니다. 이 결과를 통해 LLM이 생성하는 토큰을 선택하는 전략이 필요함을 알 수 있습니다.

## BERT
인코더만을 이용해 [MASK]토큰을 예측하도록 구성할 수 있습니다.

사용된 파라미터는 다음과 같습니다.
```
embed_size=128, num_heads=8, num_layers=2, learning_rate=1e-3, epochs=100
```
<script src="https://gist.github.com/emeraldgoose/0309e135c7e6235a0798ffd9d01e794b.js"></script>

학습과정은 MLM(Masked Language Model) 태스크만 수행하고 문장마다 랜덤하게 30%의 마스킹을 하여 예측하도록 구성했습니다.

<script src="https://gist.github.com/emeraldgoose/e3a7a8c2285066e62c116b1d211e09b3.js"></script>

토큰을 예측할 때는 logits의 가장 높은 확률을 가진 토큰을 선택했습니다.

<script src="https://gist.github.com/emeraldgoose/c67f7fe6ff72e87faadb594ec8970ce7.js"></script>

마스킹된 토큰을 예측하고 어떤 토큰이 top-5에 속하는지 확인해보고 싶어 이렇게 코드를 작성했습니다.

결과는 다음과 같습니다.

```
Orignal: Great minds think alike, but they also think differently

Input: Great minds [MASK] alike, but they also think differently

Output: <sos> Great minds think alike, but they also think differently <pad> <pad> <pad> <pad> <pad> <pad> <eos>

Token(Prob): think(0.31) | alike,(0.08) | best(0.05) | differently(0.02) | but(0.02)
```
신기하게 [MASK] 토큰 위치에 들어갈 토큰 후보들이 주로 학습한 문장 내에서 나오는 것을 볼 수 있습니다.

> GPT와 BERT 실험 코드는 아래 simple_gpt.ipynb에 작성되어 있습니다.

# Code
- [https://github.com/emeraldgoose/hcrot](https://github.com/emeraldgoose/hcrot)
- [simple_LM.ipynb](https://github.com/emeraldgoose/hcrot/blob/master/notebooks/simple_LM.ipynb)
