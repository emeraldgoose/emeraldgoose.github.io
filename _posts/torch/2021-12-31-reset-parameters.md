---
title: Pytorch layer 초기화 함수
categories:
  - Pytorch
tags: [torch, initialize]
---

> 모델링을 하게되면 초기화를 신경쓰지 않게 되는데 어떤식으로 이루어지는지 잘 모르고 있었습니다.
> 그래서 Linear layer를 선언했을 때 weight와 bias를 어떻게 초기화하는지 알아보고자 합니다.

## Class Linear

`Linear` 레이어 모듈을 살펴보기 위해 pytorch 코드를 가져왔습니다.

```python
class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, 
                bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(
                torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
```

`Linear layer`를 선언하게 되면 `__init__()`함수에 의해 `self.weight`와 `self.bias`를 생성합니다.

이후 이 파라미터들은 `self.reset_parameters()`에 의해 초기화가 진행됩니다.

## reset parameters()

```python
def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)
```

먼저, `self.weight`는 `kaiming_unifrom`에 의해 초기화가 진행됩니다. kaiming initialize는 검색하시면 자세한 내용이 있기 때문에 넘어가겠습니다.  

그냥 kaiming 초기화 방법을 사용하는구나 정도로 봐주세요.

다음, `self.bias는` weight의 `bound`를 계산한 후 `unform(-bound, bound)`에 의해 초기화가 진행됩니다.

### initialize Bias
논문을 잠깐 읽어봤는데 kaiming 방법은 `bias`를 0으로 초기화하고 진행했습니다. 

그러나 torch 코드에서는 `bias`를 `uniform(-bound, bound)`로 초기화하고 있기 때문에 다시 찾아봤는데 [링크](https://stackoverflow.com/questions/48529625/in-pytorch-how-are-layer-weights-and-biases-initialized-by-default)에서 bias는 **LeCunn 초기화 방법**을 사용한다고 합니다.

[링크](https://yeomko.tistory.com/40)에 따르면 Lecunn 초기화 방법의 아이디어는 확률분포를 `fan_in`으로 조절하고자 하는 것이라고 합니다.

- bound를 계산하기 전에 `_calculate_fan_in_and_fan_out()`이라는 함수를 통해 `fan_in`이라는 값을 계산하는데 input layer의 뉴런 수를 `fan_in`, output layer의 뉴런 수를 `fan_out`이라고 합니다.

lecunn init 논문인 **Efficient BackProp**의 섹션 4.6을 보면 sqrt(1/fan_in)으로 표준편자를 정하고 평균은 0인 uniform하게 초기화합니다.

이렇게 `nn.Linear` 레이어의 weight는 kaiming, bias는 lecunn 초기화가 진행되어 `reset_parameters()`가 끝나게 됩니다.

## 다른 초기화 방법들

초기화 방법들은 여러가지가 있고 torch의 모듈마다 초기화 방법이 달라집니다.

예를들면 embedding 레이어는 normal을 사용합니다.

다른 Xaiver, He 등의 여러가지 초기화 방법들 또한 사용가능합니다.

사용할때는 초기화를 신경쓰면서 사용하는 편은 아니지만 한번 코드 내부를 돌면서 이런식으로 진행하는구나 정도로 봐주시면 될 것 같습니다.

## Reference
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852v1)
- [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
- [갈아먹는 딥러닝 기초 [2] weight initialization](https://yeomko.tistory.com/40)
- [Pytorch](https://github.com/pytorch/pytorch/tree/v0.3.1/torch/nn/modules)
- [StackOverflow](https://stackoverflow.com/questions/48529625/in-pytorch-how-are-layer-weights-and-biases-initialized-by-default)