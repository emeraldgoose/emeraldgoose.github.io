---
title: Python으로 Diffusion 바닥부터 구현하기[1] (ResidualBlock, AttentionBlock, UpsampleBlock)
categories:
  - Pytorch
tags: [torch]
---
## Text-to-Image
최근 Stable Diffusion이라는 모델이 발전하면서 많은 사람들이 이미지 생성 AI를 사용하고 있습니다. Diffusion 모델은 노이즈를 기존 이미지에 더한 뒤에 노이즈를 예측하고 지워가며 복원하는 학습과정을 거치게 됩니다.

저는 Diffusion 모델을 구현하여 MNIST를 학습시키고 label이라는 condition을 주어 생성해보려고 합니다. 

Diffusion 모델 중 Huggingface의 [UNet2DModel](https://huggingface.co/docs/diffusers/api/models/unet2d)을 선택했는데 UNet2DModel 내부에 class_embedding 레이어가 있어 class condition을 넣을 수 있도록 제공하고 있습니다.

이 페이지에선 ResidualBlock, AttentionBlock, UpsampleBlock을 구현하고 UNet의 구현은 다음 글에서 이어서 작성하겠습니다.

## ResidualBlock
ResidualBlock은 UNet에서 Down, Mid, Up 레이어 모두에서 사용됩니다. ResidualBlock은 컨볼루션 레이어를 이용해 이미지의 특징을 추출하거나 채널을 변경할 수 있습니다. 또한, 시간 임베딩(Time Embedding) 정보도 입력으로 받아 통합하는 역할을 수행합니다. 

### Forward
아래는 ResidualBlock의 순전파를 도식화한 것입니다.

<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQR7nqBZCWA-S7k68T1w2uMBAa1NDGFx9LCkv8ji7VUL4K4?width=643&height=808" data-lightbox="gallery" style="width:120%;">
      <img style="width:50%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQR7nqBZCWA-S7k68T1w2uMBAa1NDGFx9LCkv8ji7VUL4K4?width=643&height=808" alt="01">
    </a>
    <figcaption>ResidualBlock Forward Flow</figcaption>
</figure>

먼저, 입력 X를 연산하는 (Conv2d, Norm, Actv) 과정을 두 번 반복하여 `out_channels`를 가진 이미지가 됩니다. 중간에 time을 인코딩한 time embedding 벡터를 이미지에 더해주어 ResidualBlock을 사용하는 모든 연산들에 대해 timestep 정보를 알도록 합니다.

그리고 `in_channels` 채널을 가진 이미지를 kernel size 1인 Conv2d를 이용해 `out_channels` 채널을 가진 이미지로 변환하여 skip connection을 수행합니다.

```python
class ResidualBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            groups: int = 8,
            eps: float = 1e-5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_emb_proj = Linear(temb_channels, out_channels)

        self.residual_conv = Conv2d(
            in_channel=in_channels,
            out_channel=out_channels,
            kernel_size=1
        )
        
        self.conv1 = Conv2d(
            in_channel=in_channels,
            out_channel=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.conv2 = Conv2d(
            in_channel=out_channels,
            out_channel=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.norm1 = GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps)
        self.norm2 = GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps)
        
        self.nonlinearity1 = SiLU()
        self.nonlinearity2 = SiLU()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: NDArray, temb: NDArray) -> NDArray:
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlinearity1(x)
        
        temb = self.time_emb_proj(temb)
        x += temb[:, :, np.newaxis, np.newaxis]

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.nonlinearity2(x)
        
        return x + residual
```

### Backward

아래는 ResidualBlock의 역전파를 도식화한 것입니다.

<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQQh0BcbN5ZHT4s1BufBP33VAfD8NDBrAcwUveakdjtUXWQ?width=598&height=785" data-lightbox="gallery" style="width:120%;">
      <img style="width:50%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQQh0BcbN5ZHT4s1BufBP33VAfD8NDBrAcwUveakdjtUXWQ?width=598&height=785" alt="02">
    </a>
    <figcaption>ResidualBlock Backward Flow</figcaption>
</figure>

Upstream gradient `dz`는 (Conv2d, Norm, Actv) 레이어와 kernel size 1인 Conv2d레이어에 동일하게 전파됩니다. 마찬가지로 time embedding 또한 동일하게 전파됩니다.
Forward에서 입력 X가 두 Conv2d에 동일하게 전파되었으므로 Backward에선 두 Conv2d의 기울기가 더해져 `dx`가 됩니다.

```python
def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, Optional[Dict[str,NDArray]], Optional[Dict[str,NDArray]]]:
    dw, db = {}, {}

    dz_ = self.nonlinearity2.backward(dz)

    dz_, dw_norm2, db_norm2 = self.norm2.backward(dz_)
    dw['norm2.weight'], db['norm2.bias'] = dw_norm2, db_norm2

    dz_, dw_conv2, db_conv2 = self.conv2.backward(dz_)
    dw['conv2.weight'], db['conv2.bias'] = dw_conv2, db_conv2

    dtemb = np.sum(dz_, axis=(2,3))

    dtemb, dw_time_emb_linear, db_time_emb_linear = self.time_emb_proj.backward(dtemb)
    dw['time_emb_proj.1.weight'], db['time_emb_proj.1.bias'] = dw_time_emb_linear, db_time_emb_linear

    dz_ = self.nonlinearity1.backward(dz_)

    dz_, dw_norm1, db_norm1 = self.norm1.backward(dz_)
    dw['norm1.weight'], db['norm1.bias'] = dw_norm1, db_norm1

    dz_, dw_conv1, db_conv1 = self.conv1.backward(dz_)
    dw['conv1.weight'], db['conv1.bias'] = dw_conv1, db_conv1

    dz, dw_residual_conv, db_residual_conv = self.residual_conv.backward(dz)
    dw['residual_conv.weight'], db['residual_conv.bias'] = dw_residual_conv, db_residual_conv

    return dz_ + dz, dtemb, dw, db
```

## AttentionBlock
AttentionBlock은 노이즈가 섞인 이미지로부터 정보의 중요도를 재조정하고, 관계를 파악하는데 집중합니다. 이를 이용해 추론단계에서 노이즈를 예측하는데 중요한 역할을 담당합니다.

### Forward

아래는 AttentionBlock의 순전파를 도식화한 것입니다.

<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQQf3pJ3nmNISI-XqKVt098wAc--Gzj57c1pILtBEcTlmVA?width=1024" data-lightbox="gallery" style="width:120%;">
      <img style="width:30%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQQf3pJ3nmNISI-XqKVt098wAc--Gzj57c1pILtBEcTlmVA?width=1024" alt="03">
    </a>
    <figcaption>AttentionBlock Forward Flow</figcaption>
</figure>

입력 X는 정규화를 진행한 뒤에 Query, Key, Value로 만드는 Linear 레이어를 통과하게 됩니다. 
이 Query, Key, Value는 다시 Scaled-dot Product Attention 연산에 의해 `hidden_states`가 생성되고 이 벡터를 `to_out`레이어를 거쳐 AttentionBlock의 출력으로 사용됩니다.

> Scaled Dot Product Attention 함수의 forward와 backward는 [Python으로 Transformer 바닥부터 구현하기[1] (MultiHead-Attention, LayerNorm, GELU)](https://emeraldgoose.github.io/pytorch/transformer-scratch-implementation-1/#scaled-dot-product-attention) 글을 참고해주세요.

연산 중간마다 reshape을 통해 변환을 수행하고 있는데 주석으로 어떤 형태가 되어야 하는지 추가로 적어두었습니다.

```python
class Attention(Module):
    def __init__(
            self,
            query_dim: int,
            corss_attention_dim: Optional[int] = None,
            heads: int = 4,
            kv_heads: Optional[int] = None,
            dim_head: int = 32,
            norm_num_groups: Optional[int] = None,
            eps: bool = 1e-5,
            scale_qk: bool = True,
            out_dim: Optional[int] = None,
            rescale_output_factor: float = 1.0,
            only_cross_attention: bool = False,
            dropout: bool = 0.0,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.heads = heads
        self.only_cross_attention = only_cross_attention
        self.rescale_output_factor = rescale_output_factor
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.corss_attention_dim = corss_attention_dim if corss_attention_dim is not None else query_dim
        self.scale = dim_head**-0.5 if scale_qk else 1.0
        self.softmax = Softmax()

        if norm_num_groups is not None:
            self.group_norm = GroupNorm(
                num_channels=query_dim,
                num_groups=norm_num_groups,
                eps=eps,
                affine=True
            )
        else:
            self.group_norm = None

        self.to_q = Linear(query_dim, self.inner_dim)
        if not self.only_cross_attention:
            self.to_k = Linear(self.corss_attention_dim, self.inner_kv_dim)
            self.to_v = Linear(self.corss_attention_dim, self.inner_kv_dim)
        else:
            self.to_k = None
            self.to_v = None

        self.to_out = ModuleList(
            [
                Linear(self.inner_dim, self.out_dim),
                Dropout(dropout)
            ]
        )
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, x: NDArray):
        input_dim = x.ndim
        assert input_dim == 4, 'x.shape must be (batch_size, channel, height, width).'

        if input_dim == 4:
            batch_size, channel, height, width = x.shape
            x = x.reshape(batch_size, channel, height * width)
            x = np.swapaxes(x, 1, 2) # (batch_size, height * width, channel)
        
        if self.group_norm is not None:
            x = np.swapaxes(x, 1, 2)
            x = self.group_norm(x) # (batch_size, channel, height * width)
            x = np.swapaxes(x, 1, 2) # (batch_size, height * width, channel)

        query = self.to_q(x) # (batch_size, height * width, head_dim * heads)
        key = self.to_k(x) # (batch_size, height * width, head_dim * heads)
        value = self.to_v(x) # (batch_size, height * width, head_dim * heads)

        inner_dim = key.shape[-1] # head_dim * heads
        head_dim = inner_dim // self.heads # head_dim = dim_head

        query = query.reshape(batch_size, -1, self.heads, head_dim)
        self.q = np.swapaxes(query, 1, 2) # (batch_size, heads, height * width, head_dim)

        key = key.reshape(batch_size, -1, self.heads, head_dim)
        self.k = np.swapaxes(key, 1, 2) # (batch_size, heads, height * width, head_dim)

        value = value.reshape(batch_size, -1, self.heads, head_dim)
        self.v = np.swapaxes(value, 1, 2) # (batch_size, heads, height * width, head_dim)

        hidden_states = MultiHeadAttention.scaled_dot_product_attention(
            self=self,
            query=self.q,
            key=self.k,
            value=self.v
        ) # (batch_size, heads, height * width, head_dim)
        hidden_states = np.swapaxes(hidden_states, 1, 2) # (batch_size, height * width, heads, head_dim)
        hidden_states = hidden_states.reshape(batch_size, -1, self.heads * head_dim) # (batch_size, height * width, head_dim * heads)
        
        # linear proj
        hidden_states = self.to_out[0](hidden_states) # (batch_size, height * width, channel)
        # dropout
        hidden_states = self.to_out[1](hidden_states) # (batch_size, height * width, channel)

        if input_dim == 4:
            hidden_states = np.swapaxes(hidden_states, -1, -2) # (batch_size, channel, height * width)
            hidden_states = hidden_states.reshape(batch_size, channel, height, width) # (batch_size, channel, height, width)

        hidden_states /= self.rescale_output_factor

        return hidden_states
```

### Backward

아래는 AttentionBlock의 역전파를 도식화한 것입니다.

<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQQ9-xJ_j0LiT72Lb2Wg-E7gAfHopBmSAf1dM--nAzu0NAc?width=382&height=654" data-lightbox="gallery" style="width:120%;">
      <img style="width:30%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQQ9-xJ_j0LiT72Lb2Wg-E7gAfHopBmSAf1dM--nAzu0NAc?width=382&height=654" alt="04">
    </a>
    <figcaption>AttentionBlock Backward Flow</figcaption>
</figure>

Upstream gradient는 `to_out`을 거쳐 Scaled-dot Product Attention에 전파됩니다. 입력을 `to_q`, `to_k`, `to_v`에 똑같이 들어갔기 때문에 SDPA의 역전파 결과인 dQ, dK, dV를 `to_q`, `to_k`, `to_v`로 전파된 후 합쳐서 전파됩니다.

```python
def backward(self, dz) -> Tuple[NDArray, Optional[Dict[str, NDArray]], Optional[Dict[str, NDArray]]]:
    dw, db = {}, {}
    batch_size, channel, height, width = dz.shape
    input_dim = dz.ndim
    head_dim = self.inner_dim // self.heads

    dz /= self.rescale_output_factor

    if input_dim == 4:
        dz = dz.reshape(batch_size, channel, height * width) # (batch_size, channel, height * width)
        dz = np.swapaxes(dz, -1, -2) # (batch_size, height * width, channel)

    dz = self.to_out[1].backward(dz) # (batch_size, height * width, channel)
    dz, dw_to_out_linear, db_to_out_linear = self.to_out[0].backward(dz) # (batch_size, height * width, head_dim * heads)
    dw['to_out.0.weight'], db['to_out.0.bias'] = dw_to_out_linear, db_to_out_linear

    dz = dz.reshape((batch_size, -1, self.heads, head_dim)) # (batch_size, height * width, heads, head_dim)
    dz = np.swapaxes(dz, 1, 2) # (batch_size, heads, height * width, head_dim)

    dq, dk, dv = MultiHeadAttention.scaled_dot_product_attention_backward(self, dz) # (batch_size, heads, height * width, head_dim) * 3

    dq = np.swapaxes(dq, 1, 2) # (batch_size, height * width, heads, head_dim)
    dq = dq.reshape(batch_size, -1, head_dim * self.heads) # (batch_size, height * width, head_dim * heads)

    dk = np.swapaxes(dk, 1, 2) # (batch_size, height * width, heads, head_dim)
    dk = dk.reshape(batch_size, -1, head_dim * self.heads) # (batch_size, height * width, head_dim * heads)

    dv = np.swapaxes(dv, 1, 2) # (batch_size, height * width, heads, head_dim)
    dv = dv.reshape(batch_size, -1, head_dim * self.heads) # (batch_size, height * width, head_dim * heads)

    dx_q, dw_to_q, db_to_q = self.to_q.backward(dq) # (batch_size, height * width, channel)
    dw['to_q.weight'], db['to_q.bias'] = dw_to_q, db_to_q

    dx_k, dw_to_k, db_to_k = self.to_k.backward(dk) # (batch_size, height * width, channel)
    dw['to_k.weight'], db['to_k.bias'] = dw_to_k, db_to_k

    dx_v, dw_to_v, db_to_v = self.to_v.backward(dv) # (batch_size, height * width, channel)
    dw['to_v.weight'], db['to_v.bias'] = dw_to_v, db_to_v

    dx = dx_q + dx_k + dx_v # (batch_size, height * width, channel)

    if self.group_norm is not None:
        dx = np.swapaxes(dx, 1, 2) # (batch_size, channel, height * width)

        dx, dw_group_norm, db_group_norm = self.group_norm.backward(dx) # (batch_size, channel, height * width)
        dw['group_norm.weight'], db['group_norm.bias'] = dw_group_norm, db_group_norm

        dx = np.swapaxes(dx, 1, 2) # (batch_size, height * width, channel)

    if input_dim == 4:
        dx = np.swapaxes(dx, 1, 2) # (batch_size, channel, height * width)
        dx = dx.reshape(batch_size, channel, height, width) # (batch_size, channel, height, width)

    return dx, dw, db
```

## UpsampleBlock
UpsampleBlock은 UNet의 Up layer에서 사용되어 Down layer에서 낮아진 resolution을 복원하는 역할을 합니다. 사용자 정의에 따라 interpolate + Conv2d 또는 ConvTranspose2d 레이어를 선택할 수 있습니다.

### Forward
Forward 과정은 간단하게 ConvTranspose2d 레이어를 사용하여 해상도와 채널을 이전 상태로 복원하거나 interpolate로 해상도 복원, Conv2d로 채널 복원을 시도하는 과정을 수행합니다.

```python
class Upsample(Module):
    def __init__(
            self,
            channels: int,
            out_channels: Optional[int] = None,
            use_conv_transpose: bool = False,
            kernel_size: Optional[int] = None,
            padding: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv_transpose = use_conv_transpose

        self.conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            self.conv = ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding
            )
        else:
            if kernel_size is None:
                kernel_size = 3
            self.conv = Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, hidden_states: NDArray, output_size: Optional[int] = None) -> NDArray:
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)
        
        self.x = hidden_states
        self.output_size = None
        if output_size is None:
            hidden_states = utils.interpolate(hidden_states, scale_factor=2.0)
        else:
            self.output_size = output_size
            hidden_states = utils.interpolate(hidden_states, size=output_size)
        
        hidden_states = self.conv(hidden_states)
        return hidden_states
```

interpolate 함수는 다음과 같이 `nearest` 모드를 기준으로 작성했습니다. `nearest` 모드는 추가해야 할 공간에 주변 픽셀을 복사하는 방법으로 수행됩니다.

예를 들어, [0.1, 0.2, 0.3]인 (1,3) 배열을 (1,5)로 보간한다면 [0.1, 0.1, 0.2, 0.2, 0.3]의 결과를 받을 수 있습니다.

```python
def interpolate(inputs: NDArray, scale_factor: Optional[int] = None, size: Optional[int] = None, mode: str = "nearest"):
    if mode not in ("nearest"):
        raise ValueError(f"Not supported mode: {mode}")

    B, C, H_in, W_in = inputs.shape
    H_out = np.floor(H_in * scale_factor).astype(np.int32) if scale_factor else size
    W_out = np.floor(W_in * scale_factor).astype(np.int32) if scale_factor else size
    out = np.zeros((B, C, H_out, W_out), dtype=inputs.dtype)

    row_allocs = np.linspace(0, H_in, H_out, endpoint=False)
    row_indices = np.floor(row_allocs).astype(int)

    col_allocs = np.linspace(0, W_in, W_out, endpoint=False)
    col_indices = np.floor(col_allocs).astype(int)

    row_indices = np.clip(row_indices, 0, H_in - 1)
    col_indices = np.clip(col_indices, 0, W_in - 1)

    for n in range(B):
        for c in range(C):
            out[n,c] = inputs[n,c][row_indices[:, None], col_indices[None, :]]

    return out
```

### Backward
Backward의 경우도 Forward의 역순으로 수행합니다.

```python
def backward(self, dz: NDArray) -> Tuple[NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
    dx, dw, db = None, {}, {}
    if self.use_conv_transpose:
        dx, dw_conv_transpose, db_conv_transpose = self.conv.backward(dz)
        dw['conv.weight'] = dw_conv_transpose
        db['conv.bias'] = db_conv_transpose
        return dx, dw, db

    dx, dw_conv, db_conv = self.conv.backward(dz)
    dw['conv.weight'] = dw_conv
    db['conv.bias'] = db_conv

    dx = utils.interpolate_backward(dx, origin_x=self.x, mode="nearest")

    return dx, dw, db
```

interpolate의 backward 함수는 Upstream gradient를 원래의 위치에 누적해야 합니다.

예를 들어, [0.1, 0.2, 0.3]인 (1,3) 배열을 보간하여 [0.1, 0.1, 0.2, 0.2, 0.3]이 되었고, upstream gradient [1,2,3,4,5]를 역전파시킨다면 입력에 대한 기울기는 [1+2,3+4,5] = [3,7,5]가 됩니다.

```python
def interpolate_backward(dz: NDArray, origin_x: NDArray, mode: str = "nearest"):
    if mode not in ("nearest"):
        raise ValueError(f"Not supported mode: {mode}")
    
    B, C, H_in, W_in = origin_x.shape
    H_out, W_out = dz.shape[2:]
    dx = np.zeros_like(origin_x)

    row_allocs = np.linspace(0, H_in, H_out, endpoint=False)
    row_indices = np.floor(row_allocs).astype(int)

    col_allocs = np.linspace(0, W_in, W_out, endpoint=False)
    col_indices = np.floor(col_allocs).astype(int)

    row_indices = np.clip(row_indices, 0, H_in - 1)
    col_indices = np.clip(col_indices, 0, W_in - 1)

    (x, y) = np.meshgrid(row_indices, col_indices, indexing='ij')

    for n in range(B):
        for c in range(C):
            np.add.at(dx[n,c], (x,y), dz[n,c])
    
    return dx
```

## Reference
- [UNet2DModel](https://huggingface.co/docs/diffusers/api/models/unet2d)
- [[Annotated Diffusion] DDPM-(3): 코드 세부 구현하기](https://velog.io/@yeomjinseop/DDPM-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B03)