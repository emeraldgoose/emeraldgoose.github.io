---
title: LSTM & GRU 정리
categories:
  - BoostCamp
tags: [nlp]
---
## Long Short-Term Memory (LSTM)

- Original RNN에서의 Long term dependency를 개선한 모델이다.
- 매 타임스텝마다 변화하는 hidden state vector를 단기기억을 담당하는 기억소자로 볼 수 있다.
- 단기기억을 이 시퀀스가 타임스텝별로 진행됨에 따라서 단기기억을 보다 길게 기억할 수 있도록 개선한 모델이다.
- LSTM은 RNN과 달리 이전 state에서 두 가지의 정보가 들어오는데 $C_{t-1}$와 $h_{t-1}$이다.
    - $C_{t-1}$을 Cell state라 부르고 $h_{t-1}$을 hidden state vector라 한다.
- Long short-term memory
    - i : Input gate
    - f : Forget gate
    - o : Output gate
    - g : Gate gate
    - W는 $x$와 $h$를 concat한 크기와 4개의 output의 크기만큼의 사이즈를 갖게된다.
    - 이후에 sigmoid 값을 곱해 일정 비율의 정보만 갖도록 계산해주고 tanh를 통해 현재 타임스텝에서 유의미한 정보로 변환하게 된다.
- Forget gate
    - $f_t=\sigma(W_f\cdot [h_{t-1},x_t] + b_f)$
    - 이전 cell state vector의 값 $C_{t-1}$과 $f_t$를 곱해 일부 정보를 잊게 한다.
- Generate information to be added and cut it by input gate
    - $i_t = \sigma(W_i\cdot [h_{t-1},x_t] + b_i)$
    - $\tilde{C_t}=tanh(W_C \cdot [h_{t-1},x_t] + b_C)$ → -1과 1사이의 값
- Generate new cell state by adding current information to previous cell state
    - $C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C_t}$
    - $\tilde{C}$와 input gate를 곱하는 것은 한번의 선형변환만으로 $C_{t-1}$의 더해줄 정보를 만들기가 어려운 경우 더해주고자 하는 값보다 좀 더 큰 값들로 구성된 $\tilde{C}$ 혹은 Gate gate로 만들어 준 후 그 값에서 dimension별로 특정 비율만큼의 정보를 덜어내서  $C_t$를 만들기 위함이다.
- Generate hidden state by passing cell state to tanh and output gate
- Pass this hidden state to next time step, and output or next layer if needed
    - $o_t = \sigma(W_o[h_{t-1},x_t] + b_o)$
    - $h_t = o_t \cdot tanh(C_t)$
        - $0 ≤ o_t ≤ 1, -1 ≤ tanh(C_t) ≤ 1$
        - hidden state를 만들 때에도 dimension 별로 특정 비율로 작게 만들어서 구성된다.

## Gated Recurrent Unit(GRU)

- GRU는 적은 메모리와 빠른 계산시간이 가능하도록 만든 모델이다.
- LSTM의 cell state vector와 hidden state vector를 일원화해서 hidden state 하나만 사용하는 것이 GRU의 특징
    - $z_t = \sigma(W_z \cdot [h_{t-1},x_t])$
    - $r_t = \sigma(W_r \cdot [h_{t-1},x_t])$
    - $\tilde{h_t} = tanh(W \cdot [r_t \cdot h_{t-1},x_t])$
    - $h_t = (1 - z_t)\cdot h_{t-1} + z_t\cdot \tilde{h_t}$
        - $h_{t-1}$와 $\tilde{h_t}$ 두 정보를 더하는 것이 아닌 두 정보간의 가중평균을 내는 형태로 계산된다.
- Backpropagation in LSTM? GRU
    - Uninterrupted gradient flow!
    - forget gate를 곱하는 것이 아닌 덧셈으로 계산하기 때문에 gradient vanish / explosion problem이 없어진다.
    - 덧셈연산은 backprop을 수행할 때 gradient를 복사해주는 연산이 되고 따라서 항상 동일한 $W_{hh}$가 곱해지는 Original RNN에 비해 멀리있는 타임스텝까지 Gradient를 큰 변형없이 전달할 수 있다.
    - 이를 통해 긴 타임스텝간에 존재하는 Long term dependency문제를 해결할 수 있게 된다.

## Summary on RNN/LSTM/GRU

- RNN은 다양한 길이를 가질 수 있는 시퀀스 데이터에 특화된 유연한 형태의 딥러닝 모델 구조이다.
- Vanila RNN(Original RNN)은 구조가 간단하지만 학습 시에 Gradient Vanishing/Explosion 문제가 있어서 실제로 많이 사용하지 않는다.
- LSTM이나 GRU를 실제로 많이 사용하고 Cell state vector 혹은 hidden state vector에서 업데이트하는 과정이 덧셈이기 때문에 Long term dependency문제를 해결할 수 있다.