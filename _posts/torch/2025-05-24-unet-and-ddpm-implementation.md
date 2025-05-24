---
title: Python으로 Diffusion 바닥부터 구현하기[2] (im2col, UNet, DDPM)
categories:
  - Pytorch
tags: [torch]
---
## Objective
이전 글에서 UNet의 구성요소인 ResidualBlock, AttentionBlock, UpsampleBlock을 구현했습니다.
> [Python으로 Diffusion 바닥부터 구현하기[1] (ResidualBlock, AttentionBlock, UpsampleBlock)](https://emeraldgoose.github.io/pytorch/text-to-image-implementation/)

이 글에선 Conv2d의 개선, UNet, DDPMScheduler의 구현을 마무리하고 테스트 결과를 작성하겠습니다.

## Conv2d
기존 Conv2d의 컨볼루션 연산은 다음과 같았습니다.

<script src="https://gist.github.com/emeraldgoose/d816ef2fddecef83236316f9316dcde0.js"></script>

이 코드는 나이브하게 구현한 컨볼루션 연산보단 빠르지만 UNet 구현 후 실제 테스트했을 때의 속도는 절망적이었습니다. 적어도 3개의 Conv2d 레이어가 포함되는 ResidualBlock이 UNet의 Down, Mid, Up 레이어마다 쌓여있기 때문에 매우 느린 연산 속도를 보여주었고 개선이 필요하다는 판단을 하게 되었습니다.

컨볼루션 연산을 행렬곱으로 해결할 수 있는 Image to Colum(im2col), Column to Image(col2im)을 구현하게 되었고 연산 속도가 약 30% 향상되었습니다.
> CNN 모델로 5000장의 MNIST 학습 시 기존 방식 1분에서 im2col을 도입했을 때 40초로 감소

### im2col, col2im
Image to Column과 Column to Image의 아이디어는 컨볼루션 연산을 행렬곱으로 변환하는 것입니다. 컨볼루션 연산은 커널이 슬라이딩하면서 연산을 수행하는데 커널은 고정이고 입력이 바뀌기 때문에 커널을 똑같이 늘리면 행렬곱으로 만들 수 있다는 아이디어입니다.

먼저, MNIST와 같은 1채널 짜리를 생각해보겠습니다. 2 $\times$ 2 Filter와 3 $\times$ 3 Input을 컨볼루션해야 한다고 가정하겠습니다.

<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQR9OkzSKXsGSovGmf0gM29PAa2H2fqg9C8zJTANo-MbIxY?width=1024" data-lightbox="gallery" style="width:120%;">
      <img style="width:80%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQR9OkzSKXsGSovGmf0gM29PAa2H2fqg9C8zJTANo-MbIxY?width=1024" alt="01">
    </a>
    <figcaption>1 channel im2col</figcaption>
</figure>

그림과 같이 Filter를 1 $\times$ 4 배열로 변환하고 Input을 연산이 수행될 4 $\times$ 1 배열짜리 4개를 이어서 4 $\times$ 4 행렬로 구성합니다. 이후 행렬곱을 수행하면 1 $\times$ 4 배열이 생성되고 다시 원래의 2 $\times$ 2 배열로 변환하면 기존의 컨볼루션 연산과 동일한 계산이 이루어지게 됩니다.

실제로 코드로 구현하게 되면 다음과 같습니다.
```python
import numpy as np

filter = np.array([[0.1, 0.2], [0.3, 0.4]])
input_image = np.array([[1,2,3],[4,5,6],[7,8,9]])
```
위와 같은 filter와 input_iamge로 가정하겠습니다.

```python
out = np.zeros((2,2))

for i in range(input_image.shape[0]-filter.shape[0]+1):
    for j in range(input_image.shape[1]-filter.shape[1]+1):
        out[i,j] = np.sum(input_image[i:i+filter.shape[0], j:j+filter.shape[1]] * filter)

print(out)
"""
[[3.7 4.7]
 [6.7 7.7]]
"""
```
위 코드는 컨볼루션 연산을 나이브하게 구현할 때의 결과입니다.

```python
filter_col = filter.reshape((1,4))
print(filter_col)
"""
[[0.1 0.2 0.3 0.4]]
"""

out_h = input_image.shape[0] - filter.shape[0] + 1
out_w = input_image.shape[1] - filter.shape[1] + 1

cols = []

for i in range(out_h):
    for j in range(out_w):
        patch = []
        for y in range(filter.shape[0]):
            for x in range(filter.shape[1]):
                val = input_image[i + y, j + x]
                patch.append(val)
        cols.append(patch)
x_col = np.array(cols)
print(x_col)
"""
[[1 2 4 5]
 [2 3 5 6]
 [4 5 7 8]
 [5 6 8 9]]
"""

out_col = filter_col @ x_col
print(out_col)
"""
[[3.7 4.7 6.7 7.7]]
"""

out = out_col.reshape(2,2)
print(out)
"""
[[3.7 4.7]
 [6.7 7.7]]
"""
```
위 코드는 im2col로 행렬로 변환한 뒤 행렬곱을 통해 컨볼루션 연산을 구현한 것입니다.

colab을 통해 `%%timeit` 키워드로 100000번 반복 측정해보면 아래 결과처럼 연산 속도 차이를 보실 수 있습니다.
```python
%%timeit -n 100000
out = np.zeros((2,2))

for i in range(input_image.shape[0]-filter.shape[0]+1):
    for j in range(input_image.shape[1]-filter.shape[1]+1):
        out[i,j] = np.sum(input_image[i:i+filter.shape[0], j:j+filter.shape[1]] * filter)

"""
34.2 µs ± 5.15 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
"""
```
```python
%%timeit -n 100000
filter_col = filter.reshape((1,4))

out_h = input_image.shape[0] - filter.shape[0] + 1
out_w = input_image.shape[1] - filter.shape[1] + 1

cols = []

for i in range(out_h):
    for j in range(out_w):
        patch = []
        for y in range(filter.shape[0]):
            for x in range(filter.shape[1]):
                val = input_image[i + y, j + x]
                patch.append(val)
        cols.append(patch)

x_col = np.array(cols)

out_col = filter_col @ x_col

out = out_col.reshape(2,2)
"""
15.1 µs ± 3.46 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
"""
```
작은 필터와 이미지에서 나이브한 컨볼루션 연산은 평균 34 마이크로 초가 걸리는 반면 im2col을 이용한 컨볼루션 연산은 평균 15 마이크로 초로 약 50% 속도 향상이 이루어졌습니다. 

연산 속도가 빨라지는 장점이 있지만 그 만큼 요구하는 메모리가 증가하는 단점도 존재합니다. 충분한 메모리를 사용할 수 있다면 im2col, col2im는 최적화에 좋은 대안이 될 수 있습니다.

### Forward

<script src="https://gist.github.com/emeraldgoose/58d5e066ac61d6366f84dece2903f8ce.js"></script>
위 코드는 im2col과 col2im을 구현한 코드입니다. im2col은 이미지를 컬럼으로 변환하는 함수이며 col2im은 컬럼을 이미지로 변환하는 함수입니다. im2col은 forward에서 사용되며 col2im은 역전파에서 사용됩니다.

위에서 설명한 대로 이미지는 im2col 함수를 이용해 행렬로 구성하고 필터는 reshape함수를 이용해 변환합니다. 이 둘의 행렬곱을 수행한 결과를 다시 원래의 형태로 되돌리면 컨볼루션 연산이 종료됩니다.

<script src="https://gist.github.com/emeraldgoose/e3463d6306720e0e66edb122844993fc.js"></script>

### Backward
Backward 연산도 기존과 동일하고 컨볼루션 연산만 행렬곱으로 대체됩니다.

<script src="https://gist.github.com/emeraldgoose/1d30ca03f7a64c3b4207b78eb5dc1379.js"></script>

## UNet2dModel
UNet의 아키텍처 구조는 이미지를 Downsample 과정에서 해상도를 낮추고 Upsample 과정에서 해상도를 복원하면서 어떤 타임스텝의 노이즈를 예측하게 됩니다.

<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQTXWSwqQ5GqRLDBN-Dl4TQHAaA5cd9_segjuDlKoyrwxBo?width=1024" data-lightbox="gallery" style="width:120%;">
      <img style="width:80%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQTXWSwqQ5GqRLDBN-Dl4TQHAaA5cd9_segjuDlKoyrwxBo?width=1024" alt="01">
    </a>
    <figcaption>UNet architecture (https://huggingface.co/learn/diffusion-course/unit1/2#step-4-define-the-model)</figcaption>
</figure>

제가 구현한 UNet은 diffusers 라이브러리의 UNet2dModel 구조를 따라서 구현했고 DownBlock, MidBlock, UpBlock 모두 2개, 2개, 3개의 레이어로 구성됩니다. 또한, skip connection이 있기 때문에 이를 위한 저장 공간도 필요합니다.

### Init
UNetModel을 이루는 구성요소 중 `down_blocks`, `mid_blocks`는 2개의 레이어를 사용하고 `up_blocks`는 3개의 레이어로 구성했습니다.
> `UNet2dModel`의 `layers_per_blocks=2`인 것과 같음

`down_blocks`의 각 블럭의 마지막 연산은 stride가 적용되어 있어 해상도를 낮추고 마지막 블럭은 해상도를 낮추지 않도록 Identity 레이어를 사용했습니다. 반대로 `up_blocks`의 각 블럭의 마지막 연산은 Upsample 레이어를 사용하여 해상도를 복원하고 마지막 블럭은 Identity 레이어를 사용합니다.
> Identity 레이어의 forward는 입력을 그대로 출력하고 backward도 upstream gradient를 그대로 전달합니다.

<script src="https://gist.github.com/emeraldgoose/35c1b236d4a625477a8eef5fd1c3b0a5.js"></script>

### Forward
`class_labels`를 입력받게 되면 임베딩되어 time_embedding과 합치게 됩니다. 이는 시간 정보와 조건 정보를 합쳐 모든 영역에 함께 주기 위함이라고 생각합니다.

이후, `sample` 이미지는 `down_blocks`, `mid_blocks`, `up_blocks`를 차례로 지나가게 됩니다. UNet의 아키텍처에서 볼 수 있듯이 Down Layer의 출력을 Up Layer에서 사용하도록 Skip connection이 사용되었습니다.

diffusers의 `UNet2dModel`에서는 각 블럭의 출력이 아닌 각 블럭 내 레이어마다 출력을 저장하고 똑같이 Up Layer에서도 각 블럭의 입력에 사용되도록 구현되었습니다. 따라서, `down_block_res_sample`이라는 변수를 두어 Up Layer에서 사용할 수 있도록 출력값을 저장합니다.

Up Layer에서 `res_sample_channels`라는 리스트를 볼 수 있는데 이는 Backward에서 concat되어 있는 기울기를 두 기울기로 나누기 위해서입니다. 여기서 나눠진 `sample`과 관련된 기울기와 `res_sample`과 관련된 기울기는 각각 Down Layer에서 사용됩니다.

<script src="https://gist.github.com/emeraldgoose/56130c10419897cb664328982c1d87cb.js"></script>

### Backward
Backward의 연산 순서는 Forward의 역순입니다. 주의할 점은 Forward에서 concat된 변수들을 올바른 크기로 나누는 작업이 필요하기 때문에 Forward에서 저장했던 `res_sample_channels`를 이용해 `res_sample`의 채널 크기를 알아낼 수 있습니다.

Forward와는 반대로 Backward에서는 Up Layer에서 출력된 기울기가 Down Layer에 사용되어야 하므로 `up_block_dres_sample`이라는 변수를 두었습니다. 이것을 이용해 Down Layer의 각 블록마다 입력에 대한 기울기인 `dz_sample`과 `res_sample`에 대한 기울기인 dres_sample을 더한 기울기로 backward를 계산합니다.
<script src="https://gist.github.com/emeraldgoose/cd76f5b9801763071ff78ab5e36c8788.js"></script>

## DDPMScheduler
DDPM(Denoising Diffusion Probabilistic Models)은 노이즈를 추가하는 forward process와 노이즈를 걷어내는 reverse process로 구성해서 고품질의 이미지 생성을 가능하게 한 방법입니다. DDPMScheduler는 이미지에 노이즈를 추가하고 걷어내는 역할을 수행합니다.
> DDPM 스케줄러는 이해하는 것이 어려워서 diffusers 라이브러리의 DDPMScheduler 코드를 따라가면서 작성했습니다.

노이즈를 추가하는 수식과 코드는 다음과 같습니다.

$x_t = \sqrt{\bar{\alpha_t}} \cdot x_0 + \sqrt{1-\bar{\alpha_t}}\cdot \epsilon$

원본 이미지 $x_0$인 `sample`에 노이즈 $\epsilon$인 `noise`를 원본이 얼마나 유지되어야 하는지 결정하는 $\sqrt{\bar{\alpha_t}}$인 `sqrt_alpha_prod`와 얼마나 노이즈를 섞어야 하는지 결정하는 $\sqrt{1-\bar{\alpha_t}}$인 `sqrt_one_minus_alpha_prod` 의해 forward process가 수행됩니다.

<script src="https://gist.github.com/emeraldgoose/f0f3df0a62a28d9cac1ab2f37fb1f4f8.js"></script>

다음 코드는 노이즈를 걷어내는 수식과 함수입니다.

이 과정에서는 noisy sample $x_t$와 예측한 노이즈 $\epsilon_\theta$를 이용해 이전 스텝의 이미지 $x_{t-1}$을 추정하는 것이 목적입니다.

다음 수식으로 먼저 모델이 예측한 원본 이미지 `pred_original_sample`을 계산합니다.

$\hat{x_0} = \frac{1}{\sqrt{\bar{\alpha_t}}}(x_t - \sqrt{1-\bar{\alpha_t}}\cdot \epsilon_\theta)$

모델이 예측한 원본 이미지와 현재 noisy 이미지를 이용해 평균 $\tilde{\mu}_t$를 구해야 합니다.

$\tilde{\mu}\_t = \frac{\bar{\alpha_{t-1}}\beta\_t}{1-\bar{\alpha\_t}}x\_0 + \frac{\sqrt{\alpha\_t}\cdot(1-\bar{\alpha\_{t-1}})}{1-\bar{\alpha\_t}}x\_t$

이미지에 곱하는 가중치 값은 `pred_original_sample_coeff`와 `current_sample_coeff`를 사용합니다.

이제 `prev_sample`인 $x_{t-1}$을 추정하기 위해 다음 수식을 사용합니다.

$x_{t-1} = \tilde{\mu}_t + \sigma_t \cdot z$

무작위 노이즈인 variance를 추가하는 과정을 통해 이미지에 무작위성을 추가하여 다양한 결과를 생성할 수 있도록 합니다.

<script src="https://gist.github.com/emeraldgoose/373603dfd8054726ef9d689e82b0a15b.js"></script>

MNIST 이미지 중 하나에 노이즈를 섞게 되면 다음과 같이 기존의 모양을 유지할 수 없게 됩니다.
<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQRNYSt_XlUtR7M1dvHGSHQOAVVVPhNZmXGqpZbPLkcn0aI?width=1024" data-lightbox="gallery" style="width:120%;">
      <img style="width:80%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQRNYSt_XlUtR7M1dvHGSHQOAVVVPhNZmXGqpZbPLkcn0aI?width=1024" alt="01">
    </a>
    <figcaption>add noise 1000 steps</figcaption>
</figure>

## Train
MNIST train 이미지 모두 학습하는 것이 너무 오래걸리기 때문에 5120장만 학습에 사용했습니다. MNIST의 이미지 크기는 28 $\times$ 28인데 14 $\times$ 14로 줄여 학습 시간을 좀 더 줄였습니다.

<script src="https://gist.github.com/emeraldgoose/b0b46612aab31c13ea8e8e2675be786a.js"></script>

`lr_rate=1e-3, 25 epochs` 학습에 약 7시간정보 걸렸습니다. 여기에 `lr_rate=1e-4, 16 epochs`로 더 학습시켜 총 학습시간 11시간 ~ 12시간정도 걸렸던 것 같습니다.

## Inference
이미지 생성 시 시작 이미지 `latents`는 노이즈입니다. 또한, 추론 시간을 빠르게 하기 위해 `num_train_timesteps`를 `num_inference_steps`크기만큼 줄여 사용했습니다. 예를 들어, 학습이 10 timesteps라면 추론은 [0, 1, 3, 5, 7, 9] 총 6번의 timesteps로 노이즈를 걷어내는 것을 말합니다.

<script src="https://gist.github.com/emeraldgoose/4a4360b279711304bcdf1db2b80740f3.js"></script>

다음은 숫자 5를 생성해본 이미지입니다. 노이즈로 시작해서 점차 숫자 모양을 잡아가는 모습을 볼 수 있습니다.
<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQR2NytmlqyyRJIO-ohkQyGqAd467SprGDz8XyeXjijv9zI?width=1024" data-lightbox="gallery" style="width:120%;">
      <img style="width:80%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQR2NytmlqyyRJIO-ohkQyGqAd467SprGDz8XyeXjijv9zI?width=1024" alt="01">
    </a>
    <figcaption>Label: 5</figcaption>
</figure>

하지만, 학습량의 부족으로 인해 이상한 모양이나 다른 숫자가 튀어나오기도 합니다. 위의 이미지도 여러번 생성시켜 나온 이미지입니다.

최종 0부터 9까지의 생성되는 MNIST 이미지는 다음과 같습니다.
<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQRNoZxNDRhzQ4Une7IovQuLARaUbNcik3NJk-7fod-zUKs?width=1024" data-lightbox="gallery" style="width:120%;">
      <img style="width:80%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQRNoZxNDRhzQ4Une7IovQuLARaUbNcik3NJk-7fod-zUKs?width=1024" alt="01">
    </a>
    <figcaption>Create MNIST images</figcaption>
</figure>

## Code
[hcrot](https://github.com/emeraldgoose/hcrot)

## Reference
- [Introduction to 🤗 Diffusers](https://huggingface.co/learn/diffusion-course/unit1/2)
- [Diffusers](https://github.com/huggingface/diffusers/tree/main/src/diffusers)
- [[개념 정리] Diffusion Model 과 DDPM 수식 유도 과정](https://xoft.tistory.com/33)