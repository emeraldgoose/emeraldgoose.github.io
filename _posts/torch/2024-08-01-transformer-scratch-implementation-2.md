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

```python
class Transformer(Module):
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 bias: bool = True
        ) -> None:
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, layer_norm_eps, batch_first)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, layer_norm_eps, batch_first, bias)
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
    
    def forward(self,
                src: NDArray,
                tgt: NDArray,
                src_mask: Optional[NDArray] = None,
                tgt_mask: Optional[NDArray] = None,
                memory_mask: Optional[NDArray] = None,
        ) -> NDArray:
        is_batched = (src.ndim == 3)
        if not self.batch_first and (src.shape[1] != tgt.shape[1]) and is_batched:
            raise RuntimeError('the batch number of src and tgt must be equal')
        elif self.batch_first and (src.shape[0] != tgt.shape[0]) and is_batched:
            raise RuntimeError('the batch number of src and tgt must be equal')
        
        if src.shape[-1] != self.d_model or tgt.shape[-1] != self.d_model:
            raise RuntimeError('the feature number of src and tgt must be equal to d_model')
        
        memory = self.encoder(src, mask=src_mask)

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        return output
```

## Backward
Backward에서는 Encoder와 Decoder의 backward 함수를 호출하고 리턴되는 기울기들을 저장합니다.

```python
def backward(self, dz: NDArray) -> Tuple[NDArray, Mapping[str, NDArray], Mapping[str, NDArray]]:
    dx, dw, db = dz, {}, {}

    dtgt, dmem, dw_, db_ = self.decoder.backward(dx)
    for k,v in dw_.items():
        for param in self.parameters.keys():
            if k in param:
                dw[param] = v

    for k,v in db_.items():
        for param in self.parameters.keys():
            if k in param:
                db[param] = v

    dsrc, dw_, db_ = self.encoder.backward(dmem)
    for k,v in dw_.items():
        for param in self.parameters.keys():
            if k in param:
                dw[param] = v

    for k,v in db_.items():
        for param in self.parameters.keys():
            if k in param:
                db[param] = v

    # 원래는 dsrc, dmem를 리턴하는 것이 맞으나 직접 구현한 optimizer 구조때문에 dsrc만 리턴합니다.
    return dsrc, dw, db
```

# TransformerEncoder
Transformer의 Encoder는 EncoderLayer들이 스택되어 있는 구조로 구현됩니다.

## Forward
Pytorch의 TransformerEncoder 클래스는 인코더 레이어를 `num_layers`만큼 복사하여 ModuleList로 구성합니다. Transformer 클래스에서 선언된 EncoderLayer를 `_get_clone` 함수에서 `copy.deepcopy()`로 복사하기 때문에 스택되어 있는 인코더 레이어들은 같은 초기 파라미터를 가지고 다른 기울기를 가지게 됩니다.

```python
class TransformerEncoder(Module):
    def __init__(self,
                 encoder_layer: TransformerEncoderLayer,
                 num_layers: int,
                 norm: Optional[Module] = None,
        ) -> None:
        super().__init__()
        self.layers = _get_clone(encoder_layer, num_layers)
        self.norm = norm
        self.num_layers = num_layers
    
    def forward(self, src: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output
```

## Backward
Forward에서 반복문을 통해 순서대로 계산하고 있으므로 그 역순으로 Backward 함수를 불러 계산하고 각 레이어의 기울기를 저장합니다.

```python
def backward(self, dz: NDArray) -> Tuple[NDArray, Mapping[str, NDArray], Mapping[str, NDArray]]:
    dx, dw, db = dz, {}, {}

    if self.norm is not None:
        dz, dw_norm, db_norm = self.norm.backward(dz)
        dw['norm.weight'], db['norm.bias'] = dw_norm, db_norm

    for i, mod in reversed(list(enumerate(self.layers))):
        dx, dw_, db_ = mod.backward(dx)
        for k,v in dw_.items():
            for param in self.parameters.keys():
                if f'{i}.{k}' in param:
                    dw[param] = v

        for k,v in db_.items():
            for param in self.parameters.keys():
                if f'{i}.{k}' in param:
                    db[param] = v

    return dx, dw, db
```

# TransformerDecoder
Transformer의 Decoder는 DecoderLayer들이 스택되어 있는 구조로 구현됩니다. 다음 그림의 오른쪽 처럼 Decoder는 Output 임베딩과 인코딩 정보를 입력으로 받아 출력값을 계산합니다.

## Forward
forward 함수의 argument로 `tgt`와 `memory`가 있습니다. `tgt`는 output 임베딩을 말하고 `memory`는 인코더 출력을 말합니다. Encoder 구현과 마찬가지로 Transformer 클래스에서 선언된 DecoderLayer를 복사하여 ModuleList로 구성하고 반복문을 통해 호출하여 계산합니다.

```python
class TransformerDecoder(Module):
    def __init__(self,
                 decoder_layer: TransformerDecoderLayer,
                 num_layers: int,
                 norm: Optional[Module] = None,
        ) -> None:
        super().__init__()
        self.layers = _get_clone(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self,
                tgt: NDArray, 
                memory: NDArray, 
                tgt_mask: Optional[NDArray] = None, 
                memory_mask: Optional[NDArray] = None
        ) -> NDArray:
        self.memory_shape = memory.shape
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask, memory_mask)
            
        if self.norm is not None:
            output = self.norm(output)
        
        return output
```

## Backward
Foward에서 반복문을 통해 순서대로 계산하고 있으므로 그 역순으로 Backward 함수를 불러 계산하고 각 레이어의 기울기를 저장합니다.

```python
def backward(self, dz: NDArray) -> Tuple[NDArray, Mapping[str, NDArray], Mapping[str, NDArray]]:
    d_tgt, d_memory = dz, np.zeros(self.memory_shape)
    dw, db = {}, {}

    if self.norm is not None:
        dz, dw_norm, db_norm = self.norm.backward(dz)
        dw['norm.weight'], db['norm.bias'] = dw_norm, db_norm

    for i, mod in reversed(list(enumerate(self.layers))):
        d_tgt, d_memory_partial, dw_, db_ = mod.backward(d_tgt)
        d_memory += d_memory_partial
        for k,v in dw_.items():
            for param in self.parameters.keys():
                if f'{i}.{k}' in param:
                    dw[param] = v

        for k,v in db_.items():
            for param in self.parameters.keys():
                if f'{i}.{k}' in param:
                    db[param] = v

    return d_tgt, d_memory, dw, db
```

# TransformerEncoderLayer
Transformer의 인코딩을 담당하는 레이어입니다. TransformerEncoder로 들어온 입력은 EncoderLayer의 순서대로 처리되며 최종 출력은 Decoder에서 사용됩니다.

## Forward
<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQREv7BEX40uQLTR1Sy8KDZXAWUehzE-LH83Rw4WIW0oQm0?width=1024" data-lightbox="gallery" style="width:120%;"><img style="width:30%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQREv7BEX40uQLTR1Sy8KDZXAWUehzE-LH83Rw4WIW0oQm0?width=1024" alt="01"></a>
    <figcaption>Encoder Layer forward</figcaption>
</figure>

계산 순서는 들어온 입력이 먼저 MultiheadAttention을 거치고 FeedForward 연산을 통해 인코딩됩니다. 각 결과는 Residual Connection 구조를 사용하여 입력과 더해준 후 Layer Normalization을 수행합니다.
```python
class TransformerEncoderLayer(Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
        ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = GELU()
    
    def forward(self, src: NDArray, src_mask: Optional[NDArray] = None) -> NDArray:
        x = src
        
        # self Attention
        x = x + self.dropout1(self.self_attn(x, x, x, attn_mask=src_mask))
        x = self.norm1(x)
        
        # Feed Forward
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        x = self.norm2(x)
        
        return x
```

## Backward
<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQRVtS1PfLIFTJ0K8RzrSNlwAc9q9HElWPxwrEcgZJQ-LzY?width=1024" data-lightbox="gallery" style="width:120%;"><img style="width:30%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQRVtS1PfLIFTJ0K8RzrSNlwAc9q9HElWPxwrEcgZJQ-LzY?width=1024" alt="01"></a>
    <figcaption>Encoder Layer backward</figcaption>
</figure>

Backward 연산은 Forward의 역순으로 진행되며 Forward에서 사용된 Residual Connection은 Backward에서는 upstream gradient와 더해지게 됩니다.
```python
def backward(self, dz: NDArray):
    dw, db = {}, {}

    # Feed-Forward backward
    dz, dw_norm2, db_norm2 = self.norm2.backward(dz)
    dw['norm2.weight'], db['norm2.bias'] = dw_norm2, db_norm2

    dx_ = self.dropout2.backward(dz)
    dx_, dw_linear2, db_linear2 = self.linear2.backward(dx_)
    dw['linear2.weight'], db['linear2.bias'] = dw_linear2, db_linear2

    dx_ = self.dropout.backward(dx_)
    dx_ = self.activation.backward(dx_)
    dx_, dw_linear1, db_linear1 = self.linear1.backward(dx_)
    dw['linearr1.weight'], db['linear1.bias'] = dw_linear1, db_linear1

    dx = dz + dx_

    # Multi-Head Attention backward
    dx, dw_norm1, db_norm1 = self.norm1.backward(dx)
    dw['norm1.weight'], db['norm1.bias'] = dw_norm1, db_norm1

    dx_ = self.dropout1.backward(dx)
    (dx_q, dx_k, dx_v), dw_attn, db_attn = self.self_attn.backward(dx_)

    dx = dx_ + dx_q + dx_k + dx_v

    for k, v in dw_attn.items():
        for param in self.parameters.keys():
            if k in param:
                dw[param] = v

    for k, v in db_attn.items():
        for param in self.parameters.keys():
            if k in param:
                db[param] = v

    return dx, dw, db
```

# TransformerDecoderLayer
Transformer의 디코딩을 담당하는 레이어입니다. 인코더의 출력을 디코더에서 사용하여 output 시퀀스 이후에 나올 토큰을 예측하게 됩니다.

## Forward
<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQSOmRpxl0G5RINabXS-mKZIAVyiQITuF0imLAiB-pNmHFM?width=1024" data-lightbox="gallery" style="width:120%;"><img style="width:30%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQSOmRpxl0G5RINabXS-mKZIAVyiQITuF0imLAiB-pNmHFM?width=1024" alt="01"></a>
    <figcaption>Encoder Layer backward</figcaption>
</figure>
forward의 argument로 `tgt`와 `memory`가 있습니다. `tgt`는 output 임베딩 입력을 담당하고 `memory`는 인코더의 출력을 의미합니다. EncoderLayer와 마찬가지로 각 단계마다 Residual Connection 구조를 사용합니다.

```python
class TransformerDecoderLayer(Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 bias: bool = True,
        ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, batch_first=batch_first)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, batch_first=batch_first)
        
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.activation = GELU()
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self,
                tgt: NDArray,
                memory: NDArray,
                tgt_mask: Optional[NDArray] = None,
                memory_mask: Optional[NDArray] = None
        ) -> NDArray:
        x = tgt
        
        # self-attention
        x = x + self.dropout1(self.self_attn(x, x, x, attn_mask=tgt_mask))
        x = self.norm1(x)
        
        # multihead attention
        x = x + self.dropout2(self.multihead_attn(x, memory, memory, attn_mask=memory_mask))
        x = self.norm2(x)
        
        # feed forward
        x = x + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        x = self.norm3(x)
        
        return x
```

## Backward
<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQSAbNENQ2jXTJqRZ7lQDx60AZf6AFmNJIuMybiJdB0bh60?width=1024" data-lightbox="gallery" style="width:120%;"><img style="width:30%;" src="https://1drv.ms/i/c/502fd124b305ba80/IQSAbNENQ2jXTJqRZ7lQDx60AZf6AFmNJIuMybiJdB0bh60?width=1024" alt="01"></a>
    <figcaption>Encoder Layer backward</figcaption>
</figure>
Backward 연산은 Forward의 역순으로 진행되며 Residual Connection은 upstream gradient와 더해지게 됩니다.

```python
def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, Mapping[str, NDArray], Mapping[str, NDArray]]:
    dx, dw, db = dz, {}, {}

    dx, dw_norm3, db_norm3 = self.norm3.backward(dx)
    dw['norm3.weight'], db['norm3.bias'] = dw_norm3, db_norm3
    
    # feed-forward
    dx_ = self.dropout3.backward(dx)
    dx_, dw_linear2, db_linear2 = self.linear2.backward(dx_)
    dw['linear2.weight'], db['linear2.bias'] = dw_linear2, db_linear2

    dx_ = self.dropout.backward(dx_)
    dx_ = self.activation.backward(dx_)
    dx_, dw_linear1, db_linear1 = self.linear1.backward(dx_)
    dw['linear1.weight'], db['linear1.bias'] = dw_linear1, db_linear1

    dx = dx + dx_

    dx, dw_norm2, db_norm2 = self.norm2.backward(dx)
    dw['norm2.weight'], db['norm2.bias'] = dw_norm2, db_norm2

    # multihead_attn
    dx_ = self.dropout2.backward(dx)
    (dx_q, dmem_k, dmem_v), dw_, db_ = self.multihead_attn.backward(dx_)
    for k,v in dw_.items():
        for param in self.parameters.keys():
            if k in param:
                dw[param] = v

    for k,v in db_.items():
        for param in self.parameters.keys():
            if k in param:
                db[param] = v
    dx = dx + dx_q
    dmem = dmem_k + dmem_v

    dx, dw_norm1, db_norm1 = self.norm1.backward(dx)
    dw['norm1.weight'], db['norm1.bias'] = dw_norm1, db_norm1

    # self_attn
    dx_ = self.dropout1.backward(dx)
    (dx_q, dx_k, dx_v), dw_, db_ = self.self_attn.backward(dx_)
    for k,v in dw_.items():
        for param in self.parameters.keys():
            if k in param:
                dw[param] = v

    for k,v in db_.items():
        for param in self.parameters.keys():
            if k in param:
                db[param] = v
    dx = dx + (dx_q + dx_k + dx_v)

    return dx, dmem, dw, db
```

# Test
## MNIST Classification
이전 테스트와 마찬가지로 MNIST 5000장과 테스트 1000장으로 실험했습니다. hidden_size는 32, learning_rate는 1e-3, 10 epoch로 학습을 진행했습니다.

다음은 학습에 사용한 모델을 정의한 코드입니다.

```python
class TransformerClassifier(layers.Module):
    def __init__(self, embed_size=28, num_heads=7, hidden_dim=256, num_layers=2, num_classes=10, seq_length=28):
        super().__init__()
        self.embed_size = embed_size
        self.positional_encoding = np.expand_dims(get_sinusoid_encoding_table(seq_length, embed_size), axis=0)
        self.transformer = layers.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.flatten = layers.Flatten(start_dim=1)
        self.fc = layers.Linear(seq_length * embed_size, num_classes)

    def forward(self, src, tgt):
        src += self.positional_encoding[:, :src.shape[1], :]
        tgt += self.positional_encoding[:, :tgt.shape[1], :]
        
        output = self.transformer(src, tgt)
        
        out = self.flatten(output)
        out = self.fc(out)
        return out
```

MNIST 이미지에 순서 정보를 주기 위해 positional encoding 정보를 추가했습니다. 그리고 Transformer의 출력값이 (batch_size, 28, embed_size)이므로 Linear 레이어로 통과시키게 되면 (batch_size, 28, 10)이 되어버리기 때문에 Flatten 레이어를 통해 (batch_size, 28 * embed_size)로 바꿔준 후 Linear 레이어를 통해 (batch_size, 10) 크기를 가진 logits 값으로 출력하도록 모델을 구성했습니다.

아래 그래프들은 학습시킨 결과입니다. 왼쪽 그래프는 loss, 오른쪽 그래프는 accuracy를 기록한 것입니다.

<figure class="half">
  <a href="https://1drv.ms/i/s!AoC6BbMk0S9Qm08fTVKlHgRMYqb4?embed=1&width=846&height=833" data-lightbox="gallery"><img src="https://1drv.ms/i/s!AoC6BbMk0S9Qm08fTVKlHgRMYqb4?embed=1&width=846&height=833" alt="02"></a>
  <a href="https://1drv.ms/i/s!AoC6BbMk0S9Qm03PeZKru6RF6coJ?embed=1&width=841&height=833" data-lightbox="gallery"><img src="https://1drv.ms/i/s!AoC6BbMk0S9Qm03PeZKru6RF6coJ?embed=1&width=841&height=833" alt="03"></a>
</figure>

hidden size가 크지 않았지만 잘 학습되는 것을 볼 수 있습니다. hidden size를 256으로 올리고 학습을 돌려보면 accuracy가 0.95 이상으로 올라가기도 합니다.

## GPT
디코더만을 이용해 다음 토큰을 예측하도록 모델을 구성할 수 있습니다.

```python
def get_sinusoid_encoding_table(n_seq, d_hidn):
    # refs: https://paul-hyun.github.io/transformer-01/
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table

class GPT(layers.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_len=512):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = layers.Embedding(vocab_size, embed_size)
        self.positional_encoding = np.expand_dims(get_sinusoid_encoding_table(max_len, embed_size), axis=0)
        self.transformer_decoder_layer = layers.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            batch_first=True
        )
        self.transformer_decoder = layers.TransformerDecoder(
            self.transformer_decoder_layer,
            num_layers=num_layers,
        )
        self.fc_out = layers.Linear(embed_size, vocab_size)

    def forward(self, tgt):
        tgt_len = tgt.shape[1]
        tgt_mask = self._generate_square_subsequent_mask(tgt_len)

        tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.shape[1], :]

        output = self.transformer_decoder(tgt_emb, tgt_emb, tgt_mask=tgt_mask)
        output = self.fc_out(output)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = np.triu(np.ones((sz, sz)), 1)
        return mask
```

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

```python
def generate_sentence(model, start_sentence, max_len):
    generated = [vocab[token] for token in start_sentence.split()]
    input_seq = np.expand_dims(np.array(generated),0)
    
    while len(input_seq) < max_len:
        output = model.forward(input_seq)
        next_token_logits = output[-1, -1]
        next_token = np.argmax(next_token_logits).item()
        generated.append(next_token)
        if next_token == vocab['<eos>']:
            break
        input_seq = np.array([generated])
        
    return ' '.join(inverse_vocab[token] for token in generated)

print(generate_sentence(model, 'The', max_len=100))
```

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

```python
class BERT(layers.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_len=17):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = layers.Embedding(vocab_size, embed_size)
        self.positional_encoding = np.expand_dims(get_sinusoid_encoding_table(max_len, embed_size), axis=0)
        self.transformer_encoder_layer = layers.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            batch_first=True
        )
        self.transformer_encoder = layers.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )
        self.fc = layers.Linear(embed_size, vocab_size)

    def forward(self, src):
        src_len = src.shape[1]
        src_emb = self.embedding(src) + self.positional_encoding[:, :src_len, :]
        
        output = self.transformer_encoder(src_emb)
        output = self.fc(output)
        
        return output
```

학습과정은 MLM(Masked Language Model) 태스크만 수행하고 데이터가 적기 때문에 문장마다 랜덤하게 30%의 마스킹을 하고 반복시켜 예측하도록 구성했습니다.

```python
import copy

model = BERT(vocab_size, embed_size, num_heads, num_layers)
criterion = layers.CrossEntropyLoss()
optimizer = optim.Adam(model, lr_rate=1e-3)

targets = padded_data
bsz, seq_len = targets.shape

num_epochs = 100
pbar = tqdm(range(num_epochs))
for epoch in pbar:
    inputs = copy.deepcopy(padded_data)

    # random masking
    # masking ratio: 30%
    for i in range(len(inputs)):
        sentence = inputs[i]
        masking_ids = np.random.randint(1,len(sentence)-1,size=int(len(sentence)*0.3))
        for idx in masking_ids:
            inputs[i][idx] = vocab['[MASK]']
    
    outputs = model.forward(inputs)
    
    outputs = outputs.reshape(-1, vocab_size)
    targets = targets.reshape(-1)
    loss = criterion(outputs, targets)

    dz = criterion.backward()
    dz = dz.reshape(bsz, seq_len, -1)
    optimizer.update(dz)
    
    pbar.set_description(f'Loss: {loss.item():5f}')
```

토큰을 예측할 때는 logits의 가장 높은 확률을 가진 토큰을 선택했습니다.

```python
def bert_predict_mask_token(model, sentence):
    inputs = tokenize(sentence)
    inputs = [[vocab['<sos>']] + inputs + [vocab['<eos>']] + [vocab['<pad>']] * (max_len - len(inputs))]
    mask_idx = inputs[0].index(vocab['[MASK]'])

    inputs = np.array(inputs)
    output = model.forward(inputs)

    logits = softmax(output[0][mask_idx])
    predict = sorted([(inverse_vocab[token],round(logit,2)) for token, logit in enumerate(logits)],key=lambda e: -e[1])[:5]
    
    output = output.argmax(-1)[0]
    return ' '.join(inverse_vocab[token] for token in output), predict
```

마스킹된 토큰을 예측하고 어떤 토큰이 top-5에 속하는지 확인해보고 싶어 이렇게 코드를 작성했습니다.

결과는 다음과 같습니다.

```
Orignal: Great minds think alike, but they also think differently

Input: Great minds [MASK] alike, but they also think differently

Output: <sos> Great minds think alike, but they also think differently <pad> <pad> <pad> <pad> <pad> <pad> <eos>

Token(Prob): think(0.31) | alike,(0.08) | best(0.05) | differently(0.02) | but(0.02)
```
신기하게 [MASK] 토큰 위치에 들어갈 토큰 후보들이 주로 학습한 문장 내에서 나오는 것을 볼 수 있습니다.

> GPT와 BERT 실험 코드는 아래 simple_LM.ipynb에 작성되어 있고 토크나이저까지 구현해서 GPT를 실험하는 코드는 simple_gpt.ipynb에 작성되어 있습니다.

# Code
- [https://github.com/emeraldgoose/hcrot](https://github.com/emeraldgoose/hcrot)
- [simple_LM.ipynb](https://github.com/emeraldgoose/hcrot/blob/master/notebooks/simple_LM.ipynb)
- [simple_gpt.ipynb](https://github.com/emeraldgoose/hcrot/blob/master/notebooks/simple_gpt.ipynb)
