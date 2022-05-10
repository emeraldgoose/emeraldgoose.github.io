---
title: Sequential Model 정리
categories:
  - BoostCamp
tags: [model]
---
## Recurrent Neural Networks

### Sequential Model
- 시퀀설 데이터를 처리하는 데 가장 어려운 점은 입력의 차원을 알 수 없다는 점이다.
- $p(x_t\|x_{t-1},x_{t-2},...)$에서 $x_{t-1},x_{t-2},...$는 과거에 내가 고려해야 하는 정보량이다.
- $p(x_t\|x_{t-1},...,x_{t-\tau}$에서 $x_{t-\tau}$는 과거에 내가 볼 정보량을 고정하는 것이다.
- Markov model (first-order autogressive model)
    - $p(x_1,...,x_T) = p(x_T\|x_{T-1})p(x_{T-1}\|x_{T-2})...p(x_2\|x_1)p(x_1)=\Pi_{t=1}^Tp(x_t\|x_{t-1})$
    - 이 모델의 장점이 joint distribution을 표현하는 것이 쉬워진다
- Latent autogressive model
    - $\hat{x} = p(x_t\|h_t)$, $h_t$는 hidden state이고 과거의 정보를 요약한다.
    - $h_t = g(h_{t-1},x_{t-1})$
    
### Recurrent Neural Network (RNN)
- RNN은 이전의 입력이 recurrent하게 들어오는 모델이다.
- timespan t에서 보고 있는 정보는 timespan t-1에서 전달된 정보이다.
- Short-term dependency는 RNN의 가장 큰 단점이다.
    - 과거의 정보가 먼 미래에서 살아남기가 매우 힘들다.
    - 이러한 문제를 해결하기 위해 등장한 모델이 Long Short-Term Memory(LSTM)이다
- Long-term dependency 문제도 일어날 수 있다.
- RNN의 학습의 방법은 다음과 같다.
    - $h_1 = \phi(W^T h_0 + U^T x_1)$
    - $h_2 = \phi(W^T\phi(W^T h_0 + U^T x_1) + U^T x_2)$
    - $h_3 = \phi(W^T\phi(W^T\phi(W^T h_0 + U^T x_1) + U^T x_2) + U^T x_3)$
    - $h_4 = \phi(W^T\phi(W^T\phi(W^T\phi(W^T h_0 + U^T x_1) + U^T x_2) + U^T x_3) + U^T x_4)$
        - $\phi(W^T\phi(W^T..)$가 반복되는 곳이 Vanishing / exploding gradient가 일어나는 곳이다.
        - 만약 activation function이 sigmoid나 tanh라면 과거의 정보는 점점 사라진다.
        - 만약 ReLU이고 우연히 가중치 값이 양수라면 과거의 정보는 현재에서 매우 커져버린다.

### Long Short-Term Memory (LSTM)
- LSTM에서 Cell의 input, output은 다음과 같다.
    - input으로 들어오는 x, Previous cell state, Previous hidden state
    - Output으로 나가는 h(hidden state), Next cell state, Next hidden state
        - h와 Next hidden state는 같다
- LSTM은 세 개의 게이트 Forget gate, Input gate, Output gate로 이루어져 있다.
    - 게이트는 어떤 것이 유용하고 어떤 것이 유용하지 않은지에 대한 정보를 가지고 올라온 정보를 잘 조작해서 넘겨주는 역할을 맡는다.
    - Forget gate
        - $f_t = \sigma(W_f [h_{t-1},x_t] + b_f)$
        - 이전에 들어온 히든 스테이트와 현재 입력을 가지고 어떤 정보를 버릴지 결정하는 게이트, sigmoid를 통과하기 때문에 0과 1사이의 값을 가지게 된다.
    - Input gate
        - $i_t = \sigma(W_i[h_{t-1},x_t] + b_i)$
        - $\overset{\sim}{C_t}=tanh(W_C[h_{t-1},x_t] + b_C)$
        - 현재 입력에 어떤 정보를 올릴지 말지 결정하는 게이트
        - $\overset{\sim}{C_t}$는 이전 히든 스테이트와 현재 입력을 가지고 tanh로 인해 -1과 1사이의 값을 가지게 되는데 $\overset{\sim}{C_{t-1}}$과 잘 섞여서 cell state를 업데이트하는데 사용된다.
    - Update cell
        - $i_t = \sigma(W_i[h_{t-1},x_t] + b_i)$
        - $C_t = f_t * C_{t-1} + i_t * \overset{\sim}{C_t}$
        - forget gate에서 나온 $f_t$를 사용해서 버릴건 버리고 $\overset{\sim}{C_t}$를 사용해서 어느만큼 살릴지 결정해서 두 값을 combine하여 세로운 cell state로 업데이트한다.
    - Output Gate
        - $o_t = \sigma(W_o[h_{t-1},x_t] + b_o)$
        - $h_t = o_t * tanh(C_t)$
        - 어떤 값을 밖으로 내보낼지 해당하는 output gate를 만들고 output gate 만큼 곱해서 output과 next hidden state로 흘러가게 된다.

### Gated Recurrent Unit (GRU)
- LSTM과 다르게 사용되는 게이트가 2개이다.
    - reset gate
    - update gate
- 또한, cell state가 없고 hidden state만 존재하게 되면서 output gate가 있을 필요가 없어졌다. 그리고 hidden state가 output이고 다시 $h_{t-1}$로 들어간다
- GRU의 방식은 다음과 같다.
    - $z_t = \sigma(W_z[h_{t-1},x_t])$
    - $r_t = \sigma(W_r[h_{t-1},x_t])$
    - $\overset{\sim}{h_t} = tanh(W[r_t * h_{t-1},x_t])$
    - $h_t = (1-z_t)*h_{t-1} + z_t * \overset{\sim}{h_t}$

