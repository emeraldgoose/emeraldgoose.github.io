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

```python
def convolve2d(a: NDArray, f: NDArray) -> NDArray:
    # Ref: https://stackoverflow.com/a/43087771
    a,f = np.array(a), np.array(f)
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)

def forward(self, x: np.ndarray):
  self.X = x
  image, kernel, B = x[0][0], self.weight[0][0], len(x)
  hin, win = image.shape
  hout = np.floor((hin + 2 * self.padding[0] - 1 * (len(kernel)-1) - 1) / self.stride[0] + 1).astype(int)
  wout = np.floor((win + 2 * self.padding[1] - 1 * (len(kernel[0])-1) - 1) / self.stride[1] + 1).astype(int)
  ret = np.zeros((B, self.out_channel, hout, wout))

  for b in range(B):
    for cout in range(self.out_channel):
      for cin in range(self.in_channel):
        ret[b][cout] += convolve2d(x[b][cin], self.weight[cout][cin])[::self.stride[0], ::self.stride[1]]
      ret[b][cout] += self.bias[cout]
  
  return ret
```

이 코드는 나이브하게 구현한 컨볼루션 연산보단 빠르지만 UNet 구현 후 실제 테스트했을 때의 속도는 절망적이었습니다. 적어도 3개의 Conv2d 레이어가 포함되는 ResidualBlock이 UNet의 Down, Mid, Up 레이어마다 쌓여있기 때문에 매우 느린 연산 속도를 보여주었고 개선이 필요하다는 판단을 하게 되었습니다.

컨볼루션 연산을 행렬곱으로 해결할 수 있는 Image to Column(im2col), Column to Image(col2im)을 구현하게 되었고 연산 속도가 약 30% 향상되었습니다.
> CNN 모델로 5000장의 MNIST 학습 시 기존 방식 1분에서 im2col을 도입했을 때 40초로 감소

### im2col, col2im
Image to Column과 Column to Image의 아이디어는 컨볼루션 연산을 행렬곱으로 변환하는 것입니다. 컨볼루션 연산은 커널이 슬라이딩하면서 연산을 수행하는데 커널은 고정이고 입력이 바뀌기 때문에 커널과 입력 배열을 재구성하면 행렬곱으로 똑같은 결과를 얻을 수 있다는 아이디어입니다.

먼저, MNIST와 같은 1채널 짜리를 생각해보겠습니다. 2 $\times$ 2 Filter와 3 $\times$ 3 Input을 컨볼루션해야 한다고 가정하겠습니다.

<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQR9OkzSKXsGSovGmf0gM29PAa2H2fqg9C8zJTANo-MbIxY?width=1024" data-lightbox="gallery" style="width:120%;">
      <img style="width:80%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQR9OkzSKXsGSovGmf0gM29PAa2H2fqg9C8zJTANo-MbIxY?width=1024" alt="01">
    </a>
    <figcaption>1 channel im2col</figcaption>
</figure>

그림과 같이 Filter를 1 $\times$ 4 배열로 변환하고 Input을 컨볼루션 연산에 참여하는 원소들을 4 $\times$ 1 배열로 구성한 뒤, 4개를 이어서 4 $\times$ 4 행렬로 구성합니다. 이후 행렬곱을 수행하면 1 $\times$ 4 배열이 생성되고 다시 원래의 2 $\times$ 2 배열로 변환하면 기존의 컨볼루션 연산과 동일한 계산이 이루어지게 됩니다.

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

```python
def im2col(input_data, filter_h, filter_w, stride=1, padding=0):
    """Image to Column"""
    B, C, H_in, W_in = input_data.shape
    H_out = (H_in + 2*padding[0] - filter_h) // stride[0] + 1
    W_out = (W_in + 2*padding[1] - filter_w) // stride[1] + 1

    img = np.pad(input_data, [(0,0), (0,0), (padding[0], padding[0]), (padding[1], padding[1])], 'constant')
    col = np.zeros((B, C, filter_h, filter_w, H_out, W_out))

    for y in range(filter_h):
        y_max = y + stride[0]*H_out
        for x in range(filter_w):
            x_max = x + stride[1]*W_out
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride[1], x:x_max:stride[1]]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(B * H_out * W_out, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, padding=0):
    """Column to Image"""
    B, C, H_in, W_in = input_shape
    H_out = (H_in + 2 * padding[0] - filter_h) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - filter_w) // stride[1] + 1
    col = col.reshape(B, H_out, W_out, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((B, C, H_in + 2 * padding[0] + stride[0] - 1, W_in + 2 * padding[1] + stride[1] - 1))

    for y in range(filter_h):
        y_max = y + stride[0]*H_out
        for x in range(filter_w):
            x_max = x + stride[1]*W_out
            img[:, :, y:y_max:stride[0], x:x_max:stride[1]] += col[:, :, y, x, :, :]

    return img[:, :, padding[0]:H_in+padding[0], padding[1]:W_in+padding[1]]
```
위 코드는 im2col과 col2im을 구현한 코드입니다. im2col은 이미지를 컬럼으로 변환하는 함수이며 col2im은 컬럼을 이미지로 변환하는 함수입니다. im2col은 forward에서 사용되며 col2im은 역전파에서 사용됩니다.

위에서 설명한 대로 이미지는 im2col 함수를 이용해 행렬로 구성하고 필터는 reshape함수를 이용해 변환합니다. 이 둘의 행렬곱을 수행한 결과를 다시 원래의 형태로 되돌리면 컨볼루션 연산이 종료됩니다.

```python
def forward(self, x):
    self.x = x
    B, _, H_in, W_in = x.shape

    self.col = im2col(x, self.kernel_size[0], self.kernel_size[1], self.stride, self.padding) # (B * H_out * W_out, in_channels * kernel_height * kernel_width)
    self.col_W = self.weight.reshape(self.out_channel, -1).T # (in_channels * kernel_height * kernel_width, out_channels)
    out = self.col @ self.col_W + self.bias.T # (B * H_out * W_out, out_channels)
    H_out = (H_in + 2*self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
    W_out = (W_in + 2*self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

    out = out.reshape(B, H_out, W_out, -1).transpose(0, 3, 1, 2) # (B, out_channels, H_out, W_out)
    return out
```

### Backward
Backward 연산도 기존과 동일하고 컨볼루션 연산만 행렬곱으로 대체됩니다.

```python
def backward(self, dz):
    dz_reshaped = dz.transpose(0, 2, 3, 1).reshape(-1, self.out_channel) # (B * H_out * W_out, out_channels)

    dw = self.col.T @ dz_reshaped # (in_channels * kernel_height * kernel_width, out_channels)
    dw = dw.transpose(1, 0).reshape(self.weight.shape) # (out_channels, in_channels, kernel_height, kernel_width)

    db = np.sum(dz_reshaped, axis=0).reshape(self.bias.shape)

    dcol = dz_reshaped @ self.col_W.T
    dx = col2im(dcol, self.x.shape, *self.kernel_size, self.stride, self.padding)

    return dx, dw, db
```

## UNet2dModel
UNet의 아키텍처 구조는 이미지를 Downsample 과정에서 해상도를 낮추고 Upsample 과정에서 해상도를 복원하면서 어떤 타임스텝의 노이즈 $\epsilon$를 예측하게 됩니다.

<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQTXWSwqQ5GqRLDBN-Dl4TQHAaA5cd9_segjuDlKoyrwxBo?width=1024" data-lightbox="gallery" style="width:120%;">
      <img style="width:80%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQTXWSwqQ5GqRLDBN-Dl4TQHAaA5cd9_segjuDlKoyrwxBo?width=1024" alt="01">
    </a>
    <figcaption>UNet architecture (https://huggingface.co/learn/diffusion-course/unit1/2#step-4-define-the-model)</figcaption>
</figure>

제가 구현한 UNet은 diffusers 라이브러리의 UNet2dModel 구조를 따라서 구현했습니다. 또한, skip connection이 있기 때문에 이를 위한 저장 공간도 필요합니다.

### Init
UNetModel을 이루는 구성요소 중 `down_blocks`, `mid_blocks`는 2개의 레이어를 사용하고 `up_blocks`는 3개의 레이어로 구성했습니다.
> `UNet2dModel`의 `layers_per_blocks=2`인 것과 같음

`down_blocks`의 각 블럭의 마지막 연산인 Conv2d 레이어에서 stride가 적용되어 있기 때문에 해상도를 낮추고 마지막 블럭은 해상도를 낮추지 않도록 Identity 레이어를 사용했습니다. 반대로 `up_blocks`의 각 블럭의 마지막 연산은 Upsample 레이어를 사용하여 해상도를 복원하고 마지막 블럭은 Identity 레이어를 사용합니다.
> Identity 레이어의 forward는 입력을 그대로 출력하고 backward도 upstream gradient를 그대로 전달합니다.

```python
class UNetModel(Module):
    def __init__(
            self,
            sample_size: int = 28,
            in_channels: int = 3,
            out_channels: int = 3,
            time_embed_dim: Optional[int] = None,
            block_out_channels: Tuple[int] = (32,64,128),
            norm_num_groups: int = 32,
            attention_head_dim: Optional[int] = 8,
            freq_shift: int = 0,
            num_class_embeds: int = None,
        ):
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.freq_shift = freq_shift
        self.num_class_embeds = num_class_embeds
        
        timestep_input_dim = block_out_channels[0]
        self.time_embed_dim = time_embed_dim or block_out_channels[0] * 4

        self.time_embedding = Sequential(
            Linear(timestep_input_dim, self.time_embed_dim),
            SiLU(),
            Linear(self.time_embed_dim, self.time_embed_dim)
        )

        self.class_embedding = Embedding(
            num_embeddings=num_class_embeds, embedding_dim=self.time_embed_dim
        )

        self.conv_in = Conv2d(
            in_channel=in_channels,
            out_channel=block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # down
        down_blocks = []
        in_channels = block_out_channels[0]
        for i, out_channels in enumerate(block_out_channels):
            is_last = i == len(block_out_channels) - 1
            down_blocks.append(
                ModuleList([
                    ResidualBlock(in_channels, out_channels, self.time_embed_dim, groups=norm_num_groups),
                    ResidualBlock(out_channels, out_channels, self.time_embed_dim, groups=norm_num_groups),
                    Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) if not is_last else Identity()
                ])
            )
            in_channels = out_channels
        self.down_blocks = ModuleList(down_blocks)

        # mid
        mid_channels = block_out_channels[-1]
        self.mid_block1 = ResidualBlock(mid_channels, mid_channels, self.time_embed_dim, groups=norm_num_groups)
        self.mid_attn = Attention(query_dim=mid_channels, dim_head=attention_head_dim)
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels, self.time_embed_dim, groups=norm_num_groups)

        # up
        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, out_channels in enumerate(reversed_block_out_channels):
            prev_output_channel = output_channel
            in_channels = reversed_block_out_channels[min(i+1, len(block_out_channels) - 1)]
            is_last = i == len(reversed_block_out_channels) - 1
            up_blocks.append(
                ModuleList([
                    ResidualBlock(prev_output_channel + out_channels, out_channels, self.time_embed_dim, groups=norm_num_groups),
                    ResidualBlock(out_channels + out_channels, out_channels, self.time_embed_dim, groups=norm_num_groups),
                    ResidualBlock(out_channels + in_channels, out_channels, self.time_embed_dim, groups=norm_num_groups),
                    Upsample(out_channels, out_channels=out_channels) if not is_last else Identity()
                ])
            )
            output_channel = out_channels
            
        self.up_blocks = ModuleList(up_blocks)

        self.conv_norm_out = GroupNorm(num_groups=norm_num_groups, num_channels=block_out_channels[0])
        self.conv_act = SiLU()
        self.conv_out = Conv2d(block_out_channels[0], out_channel=self.out_channels, kernel_size=3, padding=1)
```

### Forward
`class_labels`를 입력받게 되면 임베딩되어 time_embedding과 합치게 됩니다. 이는 시간 정보와 조건 정보를 합쳐 모든 영역에 함께 주기 위함입니다.

이후, `sample` 이미지는 `down_blocks`, `mid_blocks`, `up_blocks`를 차례로 지나가게 됩니다. UNet의 아키텍처 그림에서 볼 수 있듯이 Downsample의 출력을 Upsample에서 사용하도록 Skip connection이 사용되었습니다.

diffusers의 `UNet2dModel`에서는 Down Layers의 레이어의 출력이 아닌 레이어를 구성하는 블럭의 출력을 저장하여 Up Layers의 레이어를 구성하는 블럭의 입력으로 사용되도록 구현되었습니다. 따라서, `down_block_res_sample`이라는 변수를 두어 Up Layers에서 사용할 수 있도록 출력값을 저장합니다.

Up Layers에서 `res_sample_channels`라는 변수를 볼 수 있는데 이 공간에는 skip connection으로 들어오는 `res_sample`의 채널이 저장되어 있습니다.
이는 Backward에서 concat되어 있는 기울기를 `sample`과 관련된 기울기와 `res_sample`과 관련된 기울기로 나누기 위해서입니다.

```python
def forward(self, sample: Union[int, NDArray], timesteps: NDArray, class_labels: Optional[NDArray] = None) -> NDArray:
    class_embeds = None
    if self.num_class_embeds is not None:
        class_embeds = self.class_embedding(class_labels)

    # time
    if isinstance(timesteps, int):
        timesteps = np.array([timesteps], dtype=np.int64)

    timesteps = timesteps * np.ones(sample.shape[0], dtype=sample.dtype)
    temb = sinusoidal_embedding(timesteps, self.block_out_channels[0], self.freq_shift)
    temb = self.time_embedding(temb)

    if class_embeds is not None:
        emb = temb + class_embeds

    # pre-process
    sample = self.conv_in(sample)

    # down
    down_block_res_samples = [sample,]
    for block1, block2, downsample in self.down_blocks:
        sample = block1(sample, emb)
        down_block_res_samples.append(sample)

        sample = block2(sample, emb)
        down_block_res_samples.append(sample)

        sample = downsample(sample)
        if not isinstance(downsample, Identity):
            down_block_res_samples.append(sample)

    # mid
    sample = self.mid_block1(sample, emb)
    sample = self.mid_attn(sample)
    sample = self.mid_block2(sample, emb)

    # up
    self.res_samples_channels = []
    for block1, block2, block3, upsample in self.up_blocks:
        res_sample = down_block_res_samples.pop()
        self.res_samples_channels.append(res_sample.shape[1])
        sample = np.concatenate((sample, res_sample), axis=1)
        sample = block1(sample, emb)

        res_sample = down_block_res_samples.pop()
        self.res_samples_channels.append(res_sample.shape[1])
        sample = np.concatenate((sample, res_sample), axis=1)
        sample = block2(sample, emb)

        res_sample = down_block_res_samples.pop()
        self.res_samples_channels.append(res_sample.shape[1])
        sample = np.concatenate((sample, res_sample), axis=1)
        sample = block3(sample, emb)

        if isinstance(upsample, Upsample):
            upsample_size = down_block_res_samples[-1].shape[-2]
            sample = upsample(sample, upsample_size)
        else:
            sample = upsample(sample)

    # post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    return sample
```

### Backward
주의할 점은 Forward에서 concat된 변수들을 올바른 크기로 나누는 작업이 필요하기 때문에 Forward에서 저장했던 `res_sample_channels`를 이용해 `res_sample`의 채널 크기를 알아낼 수 있습니다.

Forward와는 반대로 Backward에서는 Up Layers에서 출력된 기울기가 Down Layers에 사용되어야 하므로 `up_block_dres_sample`이라는 변수를 두었습니다. 이것을 이용해 Down Layers의 각 블록마다 입력에 대한 기울기인 `dz_sample`과 `res_sample`에 대한 기울기인 `dres_sample`을 더한 기울기로 backward를 계산합니다.

```python
def backward(self, dz: NDarray) -> Tuple[NDArray, NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
    dx, dw, db = np.zeros_like(dz), {}, {}

    # post-process
    dz, dw_conv_out, db_conv_out = self.conv_out.backward(dz)
    dw['conv_out.weight'] = dw_conv_out
    db['conv_out.bias'] = db_conv_out

    dz = self.conv_act.backward(dz)

    dz_sample, dw_conv_norm_out, db_conv_norm_out = self.conv_norm_out.backward(dz)
    dw['conv_norm_out.weight'] = dw_conv_norm_out
    db['conv_norm_out.bias'] = db_conv_norm_out

    # up
    up_block_dres_samples = []
    demb = None
    for i, (block1, block2, block3, upsample) in zip(range(len(self.up_blocks)-1,-1,-1),reversed(self.up_blocks)):
        if isinstance(upsample, Upsample):
            dz_sample, dw_upsample, db_upsample = upsample.backward(dz_sample)
            for k, v in dw_upsample.items():
                param = f'up_blocks.{i}.2.{k}'
                if param in self.parameters.keys():
                    dw[param] = v

            for k, v in db_upsample.items():
                param = f'up_blocks.{i}.2.{k}'
                if param in self.parameters.keys():
                    db[param] = v
        else:
            dz_sample = upsample.backward(dz_sample)

        dz_sample, demb_upblock3, dw_block3, db_block3 = block3.backward(dz_sample)
        demb = demb_upblock3 if demb is None else demb + demb_upblock3
        for k, v in dw_block3.items():
            param = f'up_blocks.{i}.2.{k}'
            if param in self.parameters.keys():
                dw[param] = v

        for k, v in db_block3.items():
            param = f'up_blocks.{i}.2.{k}'
            if param in self.parameters.keys():
                db[param] = v

        _channel = self.res_samples_channels.pop()
        dz_sample, dres_sample = dz_sample[:,:-_channel,:,:], dz_sample[:,-_channel:,:,:]
        up_block_dres_samples.append(dres_sample)

        dz_sample, demb_upblock2, dw_block2, db_block2 = block2.backward(dz_sample)
        demb += demb_upblock2
        for k, v in dw_block2.items():
            param = f'up_blocks.{i}.1.{k}'
            if param in self.parameters.keys():
                dw[param] = v

        for k, v in db_block2.items():
            param = f'up_blocks.{i}.1.{k}'
            if param in self.parameters.keys():
                db[param] = v

        _channel = self.res_samples_channels.pop()
        dz_sample, dres_sample = dz_sample[:,:-_channel,:,:], dz_sample[:,-_channel:,:,:]
        up_block_dres_samples.append(dres_sample)

        dz_sample, demb_upblock1, dw_block1, db_block1 = block1.backward(dz_sample)
        demb += demb_upblock1
        for k, v in dw_block1.items():
            param = f'up_blocks.{i}.0.{k}'
            if param in self.parameters.keys():
                dw[param] = v

        for k, v in db_block1.items():
            param = f'up_blocks.{i}.0.{k}'
            if param in self.parameters.keys():
                db[param] = v

        _channel = self.res_samples_channels.pop()
        dz_sample, dres_sample = dz_sample[:,:-_channel,:,:], dz_sample[:,-_channel:,:,:]
        up_block_dres_samples.append(dres_sample)

    # mid
    dz_sample, demb_mid_block2, dw_mid_block2, db_mid_block2 = self.mid_block2.backward(dz_sample)
    demb += demb_mid_block2
    for k, v in dw_mid_block2.items():
        param = f'mid_block2.{k}'
        if param in self.parameters.keys():
            dw[param] = v

    for k, v in db_mid_block2.items():
        param = f'mid_block2.{k}'
        if param in self.parameters.keys():
            db[param] = v

    dz_sample, dw_mid_attn, db_mid_attn = self.mid_attn.backward(dz_sample)
    for k, v in dw_mid_attn.items():
        param = f'mid_attn.{k}'
        if param in self.parameters.keys():
            dw[param] = v

    for k, v in db_mid_attn.items():
        param = f'mid_attn.{k}'
        if param in self.parameters.keys():
            db[param] = v

    dz_sample, demb_mid_block1, dw_mid_block1, db_mid_block1 = self.mid_block1.backward(dz_sample)
    demb += demb_mid_block1
    for k, v in dw_mid_block1.items():
        param = f'mid_block1.{k}'
        if param in self.parameters.keys():
            dw[param] = v

    for k, v in db_mid_block1.items():
        param = f'mid_block1.{k}'
        if param in self.parameters.keys():
            db[param] = v

    # down
    for i, (block1, block2, downsample) in zip(range(len(self.down_blocks)-1,-1,-1),reversed(self.down_blocks)):
        if not isinstance(downsample, Identity):
            dres_sample = up_block_dres_samples.pop()
            dz_sample, dw_downsample, db_downsample = downsample.backward(dz_sample + dres_sample)
            dw[f'down_blocks.{i}.2.weight'] = dw_downsample
            db[f'down_blocks.{i}.2.bias'] = db_downsample
        else:
            dz_sample = downsample.backward(dz_sample)

        dres_sample = up_block_dres_samples.pop()
        dz_sample, demb_downblock2, dw_block2, db_block2 = block2.backward(dz_sample + dres_sample)
        demb += demb_downblock2
        for k, v in dw_block2.items():
            param = f'down_blocks.{i}.1.{k}'
            if param in self.parameters.keys():
                dw[param] = v

        for k, v in db_block2.items():
            param = f'down_blocks.{i}.1.{k}'
            if param in self.parameters.keys():
                db[param] = v

        dres_sample = up_block_dres_samples.pop()
        dz_sample, demb_downblock1, dw_block1, db_block1 = block1.backward(dz_sample + dres_sample)
        demb += demb_downblock1
        for k, v in dw_block1.items():
            param = f'down_blocks.{i}.0.{k}'
            if param in self.parameters.keys():
                dw[param] = v

        for k, v in db_block1.items():
            param = f'down_blocks.{i}.0.{k}'
            if param in self.parameters.keys():
                db[param] = v

    # pre-process
    dx, dw_conv_in, db_conv_in = self.conv_in.backward(dz_sample)
    dw['conv_in.weight'] = dw_conv_in
    db['conv_in.bias'] = db_conv_in

    # class embedding
    if self.num_class_embeds is not None:
        _, dw_class_emb = self.class_embedding.backward(demb)
        dw['class_embedding.weight'] = dw_class_emb

    # time
    demb, dw_temb2, db_temb2 = self.time_embedding[2].backward(demb)
    dw['time_embedding.2.weight'] = dw_temb2
    db['time_embedding.2.bias'] = db_temb2

    demb = self.time_embedding[1].backward(demb)

    demb, dw_temb1, db_temb1 = self.time_embedding[0].backward(demb)
    dw['time_embedding.0.weight'] = dw_temb1
    db['time_embedding.0.bias'] = db_temb1

    return dx, demb, dw, db
```

## DDPMScheduler
DDPM(Denoising Diffusion Probabilistic Models)은 노이즈를 추가하는 forward process와 노이즈를 걷어내는 reverse process로 구성해서 고품질의 이미지 생성을 가능하게 한 방법입니다. DDPMScheduler는 이미지에 노이즈를 추가하고 걷어내는 역할을 수행합니다.
> DDPM 스케줄러는 이해하는 것이 어려워서 diffusers 라이브러리의 DDPMScheduler 코드와 필요한 수식만 따라가면서 작성했습니다.

노이즈를 추가하는 수식과 코드는 다음과 같습니다.

$x_t = \sqrt{\bar{\alpha_t}} \cdot x_0 + \sqrt{1-\bar{\alpha_t}}\cdot \epsilon$

원본 이미지 $x_0$인 `sample`에 노이즈 $\epsilon$인 `noise`를 원본이 얼마나 유지되어야 하는지 결정하는 $\sqrt{\bar{\alpha_t}}$인 `sqrt_alpha_prod`와 얼마나 노이즈를 섞어야 하는지 결정하는 $\sqrt{1-\bar{\alpha_t}}$인 `sqrt_one_minus_alpha_prod` 의해 forward process가 수행됩니다.

```python
def add_noise(
      self,
      original_samples: NDArray,
      noise: NDArray,
      timesteps: NDArray,
  ) -> NDArray:
  sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
  sqrt_alpha_prod = sqrt_alpha_prod.flatten()
  while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
      sqrt_alpha_prod = sqrt_alpha_prod[..., np.newaxis]

  sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
  sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
  while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
      sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[..., np.newaxis]

  noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
  return noisy_samples
```

MNIST 이미지 중 하나에 노이즈를 섞게 되면 다음과 같이 기존의 모양을 유지할 수 없게 됩니다.
<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQRNYSt_XlUtR7M1dvHGSHQOAVVVPhNZmXGqpZbPLkcn0aI?width=1024" data-lightbox="gallery" style="width:120%;">
      <img style="width:80%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQRNYSt_XlUtR7M1dvHGSHQOAVVVPhNZmXGqpZbPLkcn0aI?width=1024" alt="01">
    </a>
    <figcaption>add noise 1000 steps</figcaption>
</figure>

다음은 노이즈를 걷어내는 수식과 코드입니다.

이 과정에서는 noisy sample $x_t$와 예측한 노이즈 $\epsilon_\theta$를 이용해 이전 스텝의 이미지 $x_{t-1}$을 추정하는 것이 목적입니다.

다음 수식으로 먼저 모델이 예측한 원본 이미지 `pred_original_sample`을 계산합니다.

$\hat{x_0} = \frac{1}{\sqrt{\bar{\alpha_t}}}(x_t - \sqrt{1-\bar{\alpha_t}}\cdot \epsilon_\theta)$

모델이 예측한 원본 이미지와 현재 noisy 이미지를 이용해 평균 $\tilde{\mu}_t$를 구해야 합니다.

$\tilde{\mu}\_t = \frac{\bar{\alpha_{t-1}}\beta\_t}{1-\bar{\alpha\_t}}x\_0 + \frac{\sqrt{\alpha\_t}\cdot(1-\bar{\alpha\_{t-1}})}{1-\bar{\alpha\_t}}x\_t$

이미지 $\hat{x_0}, x_t$ 에 곱하는 가중치 값은 각각 `pred_original_sample_coeff`와 `current_sample_coeff`를 사용합니다.

이제 `prev_sample`인 $x_{t-1}$을 추정하기 위해 다음 수식을 사용합니다.

$x_{t-1} = \tilde{\mu}_t + \sigma_t \cdot z$

무작위 노이즈인 variance를 추가하는 과정을 통해 이미지에 무작위성을 추가하여 다양한 결과를 생성할 수 있도록 합니다.

```python
def step(
      self,
      model_output: NDArray,
      timestep: NDArray,
      sample: NDArray
  ) -> DDPMSchedulerOutput:
  t = timestep
  prev_t = self.previous_timestep(t)

  # 1. compute alphas, betas
  alpha_prod_t = self.alphas_cumprod[t]
  alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
  beta_prod_t = 1 - alpha_prod_t
  beta_prod_t_prev = 1 - alpha_prod_t_prev
  current_alpha_t = alpha_prod_t / alpha_prod_t_prev
  current_beta_t = 1 - current_alpha_t

  # 2. compute predicted original sample from predicted noise also called
  # prediction_type = epsilon
  pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

  # 3. Clip prediction x_0
  pred_original_sample = pred_original_sample.clip(-self.clip_sample_range, self.clip_sample_range)

  # 4. Compute coefficients for pred_original_sample x_9 and current sample x_t
  pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
  current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

  # 5. Compute predicted previous sample u_t
  pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

  # 6. Add noise
  # variance_type: fixed_small
  variance = 0
  if t > 0:
      variance_noise = np.random.randn(*model_output.shape).astype(model_output.dtype)
      variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
      variance = (np.clip(variance, a_min=1e-20, a_max=None) ** 0.5) * variance_noise

  pred_prev_sample = pred_prev_sample + variance

  return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
```

## Train
MNIST train 이미지 모두 학습하는 것이 너무 오래걸리기 때문에 5120장만 학습에 사용했습니다. MNIST의 이미지 크기는 28 $\times$ 28인데 `average_pooling` 함수를 이용해 14 $\times$ 14로 줄여 학습 시간을 좀 더 줄였습니다.

```python
batch_size = 512
num_epochs = 10
timesteps = 1000
lr = 1e-4

def average_pooling(img, pool_size=2):
    B, C, H, W = img.shape
    new_H, new_W = H // pool_size, W // pool_size
    img = img[:,:,:new_H * pool_size, :new_W * pool_size]
    img_reshaped = img.reshape(B, C, new_H, pool_size, new_W, pool_size)
    downsampled = img_reshaped.mean(axis=(3, 5))
    return downsampled

df = pd.read_csv('./datasets/mnist_test.csv')
label = df['7'].to_numpy()
df = df.drop('7',axis=1)
dat = df.to_numpy()

mnist = dat[:batch_size * 10]
train_label = label[:batch_size * 10]
mnist = mnist.reshape(-1,1,28,28).astype(np.float32)
mnist = (mnist / 255.) * 2. - 1.
mnist = average_pooling(mnist, 2) # resize
dataloader = hcrot.dataset.Dataloader(mnist, train_label, batch_size=batch_size, shuffle=True)

class Model(layers.Module):
    def __init__(self):
        super().__init__()
        self.unet = layers.UNetModel(
            sample_size=14,
            in_channels=1,
            out_channels=1,
            block_out_channels=(32,64,32),
            num_class_embeds=10,
        )
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, *kwargs)

    def forward(self, x_noisy, t, labels):
        noise_pred = self.unet(x_noisy, t, labels)
        return noise_pred

# Model, Optimizer, Loss
model = Model()
# model.load_state_dict(hcrot.utils.load('artifact.pkl'))
optimizer = hcrot.optim.AdamW(model, lr_rate=lr)
criterion = layers.MSELoss()

noise_scheduler = layers.DDPMScheduler(num_train_timesteps=timesteps, beta_schedule="squaredcos_cap_v2")

pbar = trange(num_epochs)
for epoch in pbar:
    total_loss = 0
    for i, (x, label) in enumerate(dataloader):
        timestep = np.random.randint(0, timesteps, (x.shape[0],))
        # x_noisy, noise = q_sample(x, t)

        noise = np.random.randn(*x.shape)
        noisy_x = noise_scheduler.add_noise(x, noise, timestep)

        noise_pred = model(noisy_x, timestep, label)
        loss = criterion(noise_pred, noise)
        total_loss += loss.item()
        
        dz = criterion.backward()
        optimizer.update(dz)
    
    pbar.set_postfix(loss=total_loss/(i+1))

hcrot.utils.save(model.state_dict(), 'artifact.pkl')
```

`lr_rate=1e-3, 25 epochs` 학습에 약 7시간 정도 걸렸습니다. 여기에 `lr_rate=1e-4, 16 epochs`로 더 학습시켜 총 학습시간 11시간 ~ 12시간정도 걸렸던 것 같습니다.

## Inference
이미지 생성 시 시작 이미지 `latents`는 노이즈입니다. 또한, 추론 시간을 빠르게 하기 위해 `num_train_timesteps`를 `num_inference_steps`크기만큼 줄여 사용했습니다. 예를 들어, 학습이 10 timesteps라면 추론은 [0, 1, 3, 5, 7, 9] 총 6번의 timestep으로 노이즈를 걷어내는 것을 말합니다.

```python
class Model(layers.Module):
    def __init__(self):
        super().__init__()
        self.unet = layers.UNetModel(
            sample_size=14,
            in_channels=1,
            out_channels=1,
            block_out_channels=(32,64,32),
            num_class_embeds=10,
        )
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, *kwargs)

    def forward(self, x_noisy, t, labels):
        noise_pred = self.unet(x_noisy, t, labels)
        return noise_pred

model = Model()
model.load_state_dict(hcrot.utils.load('artifact.pkl'))

scheulder = layers.DDPMScheduler(beta_schedule="squaredcos_cap_v2", num_train_timesteps=1000)
scheulder.set_timesteps(num_inference_steps=800)

all_images = []
for i in range(10):
    latents = np.random.randn(1,1,14,14)
    label = np.array([i])

    for t in tqdm(scheulder.timesteps, desc=f"label_{i}"):
        latent_model_input = scheulder.scale_model_input(latents, t)
        noise_pred = model(latent_model_input, t, label)
        latents = scheulder.step(noise_pred, t, latents).prev_sample

    image = (latents / 2 + 0.5).clip(0, 1)
    image = np.squeeze(image)
    image = (image * 255).round().astype("uint8")
    all_images.append(image)

fig, axs = plt.subplots(1, 10, figsize=(10,10))
for i, img in enumerate(all_images):
    img = img.reshape(14,14,1)
    axs[i].imshow(img, cmap='gray')
    axs[i].axis('off')
plt.tight_layout()
plt.show()
```

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
- [hcrot](https://github.com/emeraldgoose/hcrot)
- [diffusion.ipynb](https://github.com/emeraldgoose/hcrot/blob/master/notebooks/diffusion.ipynb)

## Reference
- [Introduction to 🤗 Diffusers](https://huggingface.co/learn/diffusion-course/unit1/2)
- [Diffusers](https://github.com/huggingface/diffusers/tree/main/src/diffusers)
- [[개념 정리] Diffusion Model 과 DDPM 수식 유도 과정](https://xoft.tistory.com/33)