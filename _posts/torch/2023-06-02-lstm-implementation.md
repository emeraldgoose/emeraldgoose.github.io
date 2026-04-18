---
title: Python으로 LSTM 바닥부터 구현하기
categories:
  - Pytorch
tags: [torch]
---
## Related
> RNN에 이어서 LSTM을 구현했습니다.

## LSTM
LSTM(Long-Short Term Memory)은 RNN의 long-term dependencies 문제를 해결한 모델입니다. 장기의존성(long-term dependencies) 문제란, 과거의 정보가 먼 미래까지 전달되기 어려운 문제를 말합니다. 

이러한 long-term dependencies는 gradient vanishinig problem과도 관련이 있습니다. gradient vanishing problem이란, 미분 기울기가 0과 1사이의 값을 가지고 여러번 반복적으로 곱하게 되면 기울기가 0으로 수렴되는 문제를 말합니다. 반대로 1보다 큰 기울기를 반복적으로 곱하면 gradient explodinig problem이라 합니다. 먼 과거 정보에 대해 gradient가 소실되어 weight를 업데이트 할 수 없기 때문에 장기의존성 문제가 발생합니다.

LSTM은 이러한 문제를 cell state를 추가하여 해결합니다. LSTM은 cell state에 정보를 저장하거나 삭제할 수 있는데 cell state를 제어하기 위한 Gate들을 가지고 있습니다.

## Forward
pytorch 문서에 있는 수식을 사용했고 선언한 weight는 $W_{ih},\ W_{hh},\ b_{ih},\ b_{hh}$ 입니다.

$W_{ih}$는 각각 (hidden_size, input_size) 크기를 가진 ($W_{ii}, W_{if}, W_{ig}, W_{io}$)로 구성되며 (4 * hidden_size, input_size) 크기를 가집니다. 만약 멀티레이어 LSTM이라면 두 번째 레이어의 $W_{ih}$ 크기는 (4 * hidden_size, hidden_size)입니다.

$W_{hh}$는 각각 (hidden_size, hidden_size) 크기를 가진 ($W_{hi}, W_{hf}, W_{hg}, W_{ho}$)로 구성되며 (4 * hidden_size, hidden_size) 크기를 가집니다.

$b_{ih}$는 ($b_{ii}, b_{if}, b_{ig}, b_{io}$), $b_{hh}$는 ($b_{hi}, b_{hf}, b_{hg}, b_{ho}$)로 구성되고 둘 다 (4 * hidden_size) 크기를 가집니다. 

> $\odot$은 hadamard product(아다마르 곱)의 기호이고 두 행렬의 원소끼리 곱하는 element-wise product 연산입니다. $\sigma$는 sigmoid 연산입니다.

input gate 

- $i_t = \sigma(x_t W_{ii}^\top + b_ii + h_{t-1} W_{hi}^\top + b_hi)$

- $g_t = \tanh(x_t W_{ig}^\top + b_ig + h_{t-1} W_{hg}^\top + b_hg)$

forget gate 

- $f_t = \sigma(x_t W_{if}^\top + b_if + h_{t-1} W_{hf}^\top + b_hf)$

output gate 

- $o_t = \sigma(x_t W_{io}^\top + b_io + h_{t-1} W_{ho}^\top + b_ho)$

cell state 

- $c_t = f_t \odot c_{t-1} + i_t \odot g_t$

hidden state 

- $h_t = o_t \odot \tanh(c_t)$  

먼저, forget gate는 이전 hidden state와 입력을 보고 cell state의 정보를 유지할지 제거할지 결정합니다. sigmoid를 사용하기 때문에 0과 1사이의 범위를 가지고 1이라면 완전히 유지, 0이라면 완전히 제거합니다.

다음, cell state에 새로운 정보를 추가합니다. sigmoid를 사용하여 얼마나 반영할지 결정할 $i_t$와 새로운 후보 값이 들어간 벡터를 생성하는 $g_t$를 곱하여 cell state에 정보를 업데이트 합니다.

마지막으로, output은 입력에 sigmoid로 출력하여 cell state의 어느 부분을 출력으로 내보낼지 결정합니다.

```python
def forward(
        self, x: NDArray, 
        h_0: Optional[NDArray] = None, 
        c_0: Optional[NDArray] = None
    ) -> Tuple[NDArray, NDArray, NDArray]:
    """LSTM forward process"""
    self.X = []
    if self.batch_first:
        x = np.transpose(x, (1,0,2))
    (T, B, _), H = x.shape, self.hidden_size

    if h_0 is None:
        h_0 = np.zeros((self.num_layers, B, H))
    if c_0 is None:
        c_0 = np.zeros((self.num_layers, B, H))

    self.h = [{-1:h_0[i]} for i in range(self.num_layers)] # hidden_state
    self.c = [{-1:c_0[i]} for i in range(self.num_layers)] # cell_state

    self.i = np.zeros((self.num_layers, T, B, H)) # input_gate
    self.f = np.zeros((self.num_layers, T, B, H)) # forget_gate
    self.g = np.zeros((self.num_layers, T, B, H)) # input_gate
    self.o = np.zeros((self.num_layers, T, B, H)) # output_gate
    out = np.zeros((T, B, H))

    for l in range(self.num_layers):
        x = x if l == 0 else out
        self.X.append(x)
        w_ih, w_hh = getattr(self, f'weight_ih_l{l}'), getattr(self, f'weight_hh_l{l}')
        b_ih, b_hh = getattr(self, f'bias_ih_l{l}'), getattr(self, f'bias_hh_l{l}')

        for t in range(T):
            tmp = np.dot(x[t], w_ih.T) + b_ih + np.dot(self.h[l][t-1], w_hh.T) + b_hh
            self.i[l][t] = sigmoid(tmp[:, :H])
            self.f[l][t] = sigmoid(tmp[:, H:H*2])
            self.g[l][t] = np.tanh(tmp[:, H*2:H*3])
            self.o[l][t] = sigmoid(tmp[:, H*3:])
            self.c[l][t] = self.f[l][t] * self.c[l][t-1] + self.i[l][t] * self.g[l][t]
            self.h[l][t] = self.o[l][t] * np.tanh(self.c[l][t])
            out[t] = self.h[l][t]

    if self.batch_first:
        out = np.transpose(out, (1,0,2))

    return out, self.h, self.c
```

구현할때는 weight_ih_l[k]와 weight_hh_l[k]를 분리하지 말고 그대로 곱해도 상관없습니다.

## Backward
가장 먼저 hidden state $h_t$와 cell state $c_t$를 미분해야 합니다. input gate, forget gate, output gate 모두 hidden state, cell state 계산에 참여하기 때문입니다. LSTM의 output으로 들어오는 gradient를 $\Delta_{t}$, cost function을 $J$라 하겠습니다.

RNN과 마찬가지로 LSTM 또한 hidden state가 다음 셀로 전달되고 t에서의 hidden state가 output이 됩니다. 따라서 시간의 역순으로 hidden state의 gradient가 전달되어야 합니다. t 시점에서의 gradient는 $\Delta_{t}$와 t+1 시점에서의 hidden state의 gradient와 합치면 됩니다.

${dJ \over dh_t} = \Delta_{t} + {dJ \over dh_{t+1}}$  

LSTM은 cell state도 다음 셀로 전달되기 때문에 t+1 시점에서의 cell state의 gradient를 더해줘야 합니다.

${dJ \over dc_t} = {dJ \over dh_t} \ {dh_t \over dc_t} + {dJ \over dc_{t+1}}$

input gate 부터 output gate의 미분은 다음과 같습니다. bias는 따로 계산하지 않겠습니다.  

${dJ \over dW_{ii}} = ({dJ \over dc_t} \ {dc_t \over di_t} \ d\sigma)^\top \cdot x_t$

${dJ \over dW_{if}} = ({dJ \over dc_t} \ {dc_t \over df_t} \ d\sigma)^\top \cdot x_t$  

${dJ \over dW_{ig}} = ({dJ \over dc_t} \ {dc_t \over dg_t} \ d\tanh)^\top \cdot x_t$  

${dJ \over dW_{io}} = ({dJ \over dh_t} \ {dh_t \over do_t} \ d\sigma)^\top \cdot x_t$  

${dJ \over dW_{hi}} = ({dJ \over dc_t} \ {dc_t \over di_t} \ d\sigma)^\top \cdot h_{t-1}$  

${dJ \over dW_{hf}} = ({dJ \over dc_t} \ {dc_t \over df_t} \ d\sigma)^\top \cdot h_{t-1}$  

${dJ \over dW_{hg}} = ({dJ \over dc_t} \ {dc_t \over dg_t} \ d\tanh)^\top \cdot h_{t-1}$  

${dJ \over dW_{ho}} = ({dJ \over dh_t} \ {dh_t \over do_t} \ d\sigma)^\top \cdot h_{t-1}$  

$d\sigma$와 $d\tanh$는 다음의 예시로 설명할 수 있습니다.

$i_t = \sigma(Wx+b)$라는 수식이 주어졌다고 가정한다면 $i_t = \sigma(s_t),\ s_t = Wx+b$라는 두 수식으로 분리할 수 있고 chain rule에 따라 ${di_t \over dW} = {di_t \over ds_t} {ds_t \over dW}$로 기울기를 구할 수 있습니다.

sigmoid를 미분한다면 $\sigma \cdot (1 - \sigma)$이므로 ${di_t \over ds_t} = \sigma(s_t) (1 - \sigma(s_t)) = i_t * (1 - i_t)$라는 결과를 얻을 수 있습니다. 

마찬가지로 $\tanh$를 미분한다면 $1 - \tanh^2$이므로 $g_t = \tanh(s_t)$에서 ${dg_t \over dW} = 1 - \tanh(s_t)^2 = 1 - g_t^2$라는 결과를 얻을 수 있습니다.

위에서 구한 값들을 모두 대입하면 다음과 같은 결과를 얻을 수 있습니다. transpose를 취한 이유는 forward에서 weight에 transpose를 취하여 계산했기 때문입니다.

${dJ \over dW_{ii}} = ({dJ \over dc_t} \cdot g_t \cdot i_t \ (1 - i_t))^\top \cdot x_t$

${dJ \over dW_{if}} = ({dJ \over dc_t} \cdot c_{t-1} \cdot f_t \ (1 - f_t))^\top \cdot x_t$  

${dJ \over dW_{ig}} = ({dJ \over dc_t} \cdot i_t \cdot (1 - g_t^2))^\top \cdot x_t$  

${dJ \over dW_{io}} = ({dJ \over dh_t} \cdot \tanh(c_t) \cdot o_t \ (1 - o_t))^\top \cdot x_t$  

${dJ \over dW_{hi}} = ({dJ \over dc_t} \cdot g_t \cdot i_t \ (1 - i_t))^\top  \cdot h_{t-1}$  

${dJ \over dW_{hf}} = ({dJ \over dc_t} \cdot c_{t-1} \cdot f_t \ (1 - f_t))^\top \cdot h_{t-1}$  

${dJ \over dW_{hg}} = ({dJ \over dc_t} \cdot i_t \cdot (1 - g_t^2))^\top \cdot h_{t-1}$  

${dJ \over dW_{ho}} = ({dJ \over dh_t} \cdot \tanh(c_t) \cdot o_t \ (1 - o_t))^\top \cdot h_{t-1}$  

${dJ \over dc_t} = {dJ \over dh_t} \cdot o_t \cdot (1 - \tanh(c_t)^2) + {dJ \over dc_{t+1}}$

다음 셀에 전달할 ${dJ \over dh_{t-1}}$와 ${dJ \over dc_{t-1}}$, 입력에 대한 gradient ${dJ \over dx_t}$는 다음과 같습니다.

${dJ \over dh_{t-1}} = {dJ \over di_t} \ {di_t \over dh_{t-1}} + {dJ \over df_t} \ {df_t \over dh_{t-1}} + {dJ \over dg_t} \ {dg_t \over dh_{t-1}} + {dJ \over do_t} \ {do_t \over dh_{t-1}}$

${dJ \over dc_{t-1}} = {dJ \over dc_t} {dc_t \over dc_{t-1}} = {dJ \over dc_t} \cdot f_t$

${dJ \over dx_t} = {dJ \over di_t} \ {di_t \over dx_t} + {dJ \over df_t} \ {df_t \over dx_t} + {dJ \over dg_t} \ {dg_t \over dx_t} + {dJ \over do_t} \ {do_t \over dx_t}$

구현하기 위해 forward 과정과 마찬가지로 $W_{ih}$와 $W_{hh}$를 $W_{ii}, \ W_{hi}$부터 $W_{io}, \ W_{ho}$로 분리하지 말고 그대로 곱해줍니다. 

```python
def layer_backward(self, layer, dz):
    """
    LSTM layer backward process
    layer : multilayer LSTM인 경우 레이어 번호
    dz : delta, upstream gradient
    """
    T = len(self.X[layer]) # time length
    
    # 마지막 cell에 들어오는 hidden state, cell state의 gradient, 0으로 초기화된 값을 넣어준다
    dhnext = np.zeros_like(self.h[layer][0])
    dcnext = np.zeros_like(self.c[layer][0])

    w_ih = getattr(self, f'weight_ih_l{layer}')
    w_hh = getattr(self, f'weight_hh_l{layer}')
    b_ih = getattr(self, f'bias_ih_l{layer}')
    b_hh = getattr(self, f'bias_hh_l{layer}')

    dwih = np.zeros_like(w_ih)
    dwhh = np.zeros_like(w_hh)
    dbih = np.zeros_like(b_ih)
    dbhh = np.zeros_like(b_hh)
    dx = np.zeros_like(self.X[layer])
    
    for t in reversed(range(T)):
        """LSTM cell backward process"""
        dh = dhnext + dz[t]
        dc = dcnext + dh * self.o[layer][t] * (1 - np.tanh(self.c[layer][t])**2)
        
        di = dc * self.g[layer][t] * self.i[layer][t] * (1 - self.i[layer][t]) # (batch_size, hidden_size)
        df = dc * self.c[layer][t-1] * self.f[layer][t] * (1 - self.f[layer][t]) # (batch_size, hidden_size)
        dg = dc * self.i[layer][t] * (1 - self.g[layer][t]**2) # (batch_size, hidden_size)
        do = dh * np.tanh(self.c[layer][t]) * self.o[layer][t] * (1 - self.o[layer][t]) # (batch_size, hidden_size)
        dgates = np.hstack((di, df, dg, do)) # (batch_size, 4 * hidden_size)

        dx[t] = np.dot(dgates, w_ih) # (batch_size, input_size)
        dwih += np.dot(dgates.T, self.X[layer][t]) # (4 * hidden_size, input_size or hidden_size)
        dwhh += np.dot(dgates.T, self.h[layer][t-1]) # (4 * hidden_size, hidden_size)
        dbih += np.sum(dgates, axis=0) # (4 * hidden_size,)
        dbhh += np.sum(dgates, axis=0) # (4 * hidden_size,)
        dhnext = np.dot(dgates, w_hh) # (batch_size, hidden_size)
        dcnext = dc * self.f[layer][t] # (batch_size, hidden_size)

    return dx, dwih, dwhh, dbih, dbhh
```

## Result
LSTM과 Linear레이어만으로 이루어진 모델을 사용했고 hidden size는 256으로 사용했습니다. LSTM의 마지막 output을 Linear로 출력했고 이전 포스트들과 똑같이 MNIST 5000장을 훈련 데이터로 사용하고 1000장을 테스트로 사용했습니다.

```python
import hcrot

class LSTM(hcrot.layers.Module):
    def __init__(self):
        super().__init__()
        self.lstm = hcrot.layers.LSTM(
            input_size=28,
            hidden_size=256,
            num_layers=1,
            batch_first=False
            )
        self.fc = hcrot.layers.Linear(256, 10)
    
    def forward(self, x):
        x,_,_ = self.lstm(x)
        return self.fc(x[-1])
```

10 Epochs의 Loss와 Accuracy의 그래프는 다음과 같습니다.

<figure class="half">
  <a href="https://lh3.googleusercontent.com/fife/ALs6j_FloVHtNMwJpyCKWsskBPUVJG_FbKbRqVfvhmI_akmPIwh80DWDSvX93prWrCeObbn7mtnkWibjMJA2h40xMaZwrf38woyJB810S3CfHdDe6C4JevyzmCERA0r20Eqxbyf4X9BT3STcj2n7kdTg4vaE_JLBmY1n51VC7yEKBw0MVXS2dYlB3jsve02qD2JP5IHUbJ3K1_-Zdx9yvq1yjMJVWV4vLs99p6T56MeNMTRNLoAP0fGieFb9bSV9Z7hwSF6_IvaNdGn4z6zjtXRTuAz1mtMkQP_E_g16xQNMiBcCn2eD0B-7kM3ENsJ4Nx9ymwswkPT_IZyJmAzxjf6RxlynXG2HTzmm4_v8Q8rNpOuetxyhkw4hJnXk8sNPs6X6mJLxQfrTuMUdUtwZFOkhnm2DOg1t6DLY_-4wl_vQvPCjVgzCJMbKp57kNwRu2RpMcBWl9chv2aBoXvIYxTfaGeStqTaVnzRx9UNEZvLJrKOPgA8XvZ1bzFIHrhtdfrhnv8yE0PVM-EAsoxJoLhEdbMykyNNOhMdG_mgh9NlMUDRfLgMmgOKZI2txKTwV6ZiI7wNhbtD3Jj4l6S4QD5kk3V3xnFXHYzGny_AHvqta9Moiqy6O2wQ2yBXJNMMFdNX-VVpRVAgfKvafhqrASH6YtGgVEBKIO1krJt34Ojn1bPSTllvfmLYWqx9YXRFUhvyNAAtRfJpBDmC1S3-5ZM63MMwZl-_E1c0u7dBrhCK4XuCqh8DmDQPzpDilGMVKoYJG1_A_YhcXeYki-ApJOr-LEnTtBd0sMNxu65skYFP67RZcZKG48ur6tz4tsSy_Civ1KWDgS2NDVhxRVf8-YW-dXFgsfPltVqGabpGMOssaxrmxz09rNRZtt7ScGIf3pDyUxDjxlK60XmnZpnhrmwlDcK558bKgleOEMUhiezzFqQLVhKgjIOSrm4t_vgVhVxE7mqALEUwB1xFAXGi-uEfZK_BaiIJzGPuyQtQURyPqmb_ccdP2kmTvIzYywZOmSsQha34klcufM4gH9t9O_9SCB1ZPnkpX7euGrLn-mmV2xcey1C0dx0ftHgz5t-YFhif-yYsRKZxLKxs71LPiW6zPEFO05zOR_nMFPPw5E5xx7emgcnQ1Bg4yXcIApwzYQQNP28RuWyfozFh5qiD7kbQUMPpRJ8lKg-I6Fm3_urlMRBls63_GT7LPjMg9LPBAgxKug10Oq13ZkYjSomDTwpIBIG0lxwr-pN21bMpDLWFWbxPdfyZjXJqSOdzRHiDGi0m-qDdQe4WqPGveebaqihv6BVpXQfyCVWucsMSKSCn8ejRHEHpIdnblcUcYSub6NpGju8qOFBfcDrAI3c60-dWdLR739QcX6C3xb7BlY6f5aNQrTvK2UQ_lpkahYqEj1jnVo-0rpO0JrkatXs2KMIbB6kn8-tPH7bYL_aNrLWgEhqNIMpgzY7MWFXwWo80zHqik8opdJ4a4j_5iWkll9BEsF3-M_bwAsWsHLa5cDpT1_Vs69OMn5p4iTBjB3ggn0NVeJCzv_U3Yh6PGwMiPgCaBvQh2Fuy339Xq-K9-zcvaAzuMnFxAAkjENU4sNMlTr7k1QJUXkj_SFnwrhxJfQ6rn2rYolDacl0bUOrX0Z99gqFaOobFLBg" data-lightbox="gallery">
    <img src="https://lh3.googleusercontent.com/fife/ALs6j_FloVHtNMwJpyCKWsskBPUVJG_FbKbRqVfvhmI_akmPIwh80DWDSvX93prWrCeObbn7mtnkWibjMJA2h40xMaZwrf38woyJB810S3CfHdDe6C4JevyzmCERA0r20Eqxbyf4X9BT3STcj2n7kdTg4vaE_JLBmY1n51VC7yEKBw0MVXS2dYlB3jsve02qD2JP5IHUbJ3K1_-Zdx9yvq1yjMJVWV4vLs99p6T56MeNMTRNLoAP0fGieFb9bSV9Z7hwSF6_IvaNdGn4z6zjtXRTuAz1mtMkQP_E_g16xQNMiBcCn2eD0B-7kM3ENsJ4Nx9ymwswkPT_IZyJmAzxjf6RxlynXG2HTzmm4_v8Q8rNpOuetxyhkw4hJnXk8sNPs6X6mJLxQfrTuMUdUtwZFOkhnm2DOg1t6DLY_-4wl_vQvPCjVgzCJMbKp57kNwRu2RpMcBWl9chv2aBoXvIYxTfaGeStqTaVnzRx9UNEZvLJrKOPgA8XvZ1bzFIHrhtdfrhnv8yE0PVM-EAsoxJoLhEdbMykyNNOhMdG_mgh9NlMUDRfLgMmgOKZI2txKTwV6ZiI7wNhbtD3Jj4l6S4QD5kk3V3xnFXHYzGny_AHvqta9Moiqy6O2wQ2yBXJNMMFdNX-VVpRVAgfKvafhqrASH6YtGgVEBKIO1krJt34Ojn1bPSTllvfmLYWqx9YXRFUhvyNAAtRfJpBDmC1S3-5ZM63MMwZl-_E1c0u7dBrhCK4XuCqh8DmDQPzpDilGMVKoYJG1_A_YhcXeYki-ApJOr-LEnTtBd0sMNxu65skYFP67RZcZKG48ur6tz4tsSy_Civ1KWDgS2NDVhxRVf8-YW-dXFgsfPltVqGabpGMOssaxrmxz09rNRZtt7ScGIf3pDyUxDjxlK60XmnZpnhrmwlDcK558bKgleOEMUhiezzFqQLVhKgjIOSrm4t_vgVhVxE7mqALEUwB1xFAXGi-uEfZK_BaiIJzGPuyQtQURyPqmb_ccdP2kmTvIzYywZOmSsQha34klcufM4gH9t9O_9SCB1ZPnkpX7euGrLn-mmV2xcey1C0dx0ftHgz5t-YFhif-yYsRKZxLKxs71LPiW6zPEFO05zOR_nMFPPw5E5xx7emgcnQ1Bg4yXcIApwzYQQNP28RuWyfozFh5qiD7kbQUMPpRJ8lKg-I6Fm3_urlMRBls63_GT7LPjMg9LPBAgxKug10Oq13ZkYjSomDTwpIBIG0lxwr-pN21bMpDLWFWbxPdfyZjXJqSOdzRHiDGi0m-qDdQe4WqPGveebaqihv6BVpXQfyCVWucsMSKSCn8ejRHEHpIdnblcUcYSub6NpGju8qOFBfcDrAI3c60-dWdLR739QcX6C3xb7BlY6f5aNQrTvK2UQ_lpkahYqEj1jnVo-0rpO0JrkatXs2KMIbB6kn8-tPH7bYL_aNrLWgEhqNIMpgzY7MWFXwWo80zHqik8opdJ4a4j_5iWkll9BEsF3-M_bwAsWsHLa5cDpT1_Vs69OMn5p4iTBjB3ggn0NVeJCzv_U3Yh6PGwMiPgCaBvQh2Fuy339Xq-K9-zcvaAzuMnFxAAkjENU4sNMlTr7k1QJUXkj_SFnwrhxJfQ6rn2rYolDacl0bUOrX0Z99gqFaOobFLBg" alt="01">
  </a>
  <a href="https://lh3.googleusercontent.com/fife/ALs6j_Ef5Ule5JbX5AVEbamkI6YX3BRuQVnnR4ETBlP13MioXVhA6esG2L2KlsaT5K3xrsfY8FtQ74lUGmjFF5-xTMGkp2LaqPZWbrSovXPvLVHWXeskzpZJ5c-YzCnmZuqcRIEdXHwR3jrlJdAMK2abEAcno1Zlvyrel27kX2gBjS8mkELP_85-8M4Bp9vbgC8eA4n5_z9w5MMuukLHf9UW3w8KJEdu1wKkJ-BTXugz2ti-fGOvoGQDIgU3xpphNefems7pPJQfUKQR9q7892L5Eyl-EZNFZpMadgZzR5fNlQkNUndYRF2BJnUAqNpRiP3Dzj9SiASzzqy2bUOp3e1uribRNcvsFBwVMuqvmEL5IHHIxYQ-TOUu0yiuP7E7Unf8LvwviIkLyWKD_odgo8VjdnoMoOzO_12nbk5BSk5K8X2_u4oMiaMJ5Vvp-o9Rf_ZUZkptnveXgDGJeLeUlMPfwHqLNw-vAyWlfYnGzMS_jxqoM_SYsRD1OQPyJHltj7zlU3GrDA0sLK4eIxicDRzstKpBM0mAQXvsdNwHeCceMPwY1LFLMVqwsrr9q7cdUTej6faVaSL1Y8np_oMREPyADUUHb8kETeQtsYb3iatSmt0SAqm8T9thxSAXUHy3JiphUYQP9IuZVRM90GODeWaMFTm9cgY4mryK_HD6YnwiP6LCelXxoCXVanx1cZP9IzbG15-FyTqu2OAmvdeaeBPiOWyXxNObVJApnunOv-_4UIvBOJLjwZeVGylj96mU5IFvvH2QGh9_ZM_beWqINt2UlVSKd4Xx8bjwGnGaP3wTdULzG3HDzW4vDhRkR_bwD7wcqclufwCywWgIFr8f0ZaQY-HhcXROvCzgW6lQjwD1fcrjSk811sVNGKIrSWLytopzaRIw2sQO2oTgADKIzzAKjN0Ef2ygJfNM6x1VV2RcgXHWX0vic0mggKj9VuxqTzKab3gG28JNXrCgJMuhZx5VUPbfpONShI48zSUQaXNzhe2X_-N7mLfl1S8LVtpSPGuxjUki3R_3e7qyF2CNVmDMm9FhHs5VZt8CaBLR659-3gVWw7AJ1J1bBH6jIPck2TUaqTG6Y5omeMW9Ac838_c0ogb3icHi-8Ro9dJ2RZoqqxRA2ow49f8slTptebXA7kv7CTyomgoadUGyRiTPNARc0qs--j5wfaFSSfoCn0WJBPZwHO0N0UUYFlGHVnXPimFbKxfBEBtgAIzOzPmaZ4bB8kiizP7--q21yQiIEVA6IH-p2mXTGC9YXYlYj3eXekBzB23C4fPZr0_isNJJpZdGmft3oX_9r6hSBag7wPmo-zklPNrLi0dHxtqoyL0RVMQ65X8SKM_eEpU66Q3KFJy4pBAwRgB6fUnP1T1AWCLu0-tnqpWXuRcvp0P5fSuT3s1IGzIvXMrbTwlOLMSR5eJtSdvFRnq23NpDz-tcF2khZsiGL24Y-sNt4Q73jEjrHSyyCjLKPVtbAWfPmKGxcS_TBQ7LaIHPLm-OphalgqrJs4POcHca18xH6nSnLpxE5Nr0bl1Fzb75rUThkZZcjGnYAqX5iJJpvHDxCUCx-8j2h2y5KoADc-8U61iac20Ty8AKAnpgny4AlWZJH0Z8BH5w7XVugaMyw_f70AUDuzRdas6aVafznw" data-lightbox="gallery">
    <img src="https://lh3.googleusercontent.com/fife/ALs6j_Ef5Ule5JbX5AVEbamkI6YX3BRuQVnnR4ETBlP13MioXVhA6esG2L2KlsaT5K3xrsfY8FtQ74lUGmjFF5-xTMGkp2LaqPZWbrSovXPvLVHWXeskzpZJ5c-YzCnmZuqcRIEdXHwR3jrlJdAMK2abEAcno1Zlvyrel27kX2gBjS8mkELP_85-8M4Bp9vbgC8eA4n5_z9w5MMuukLHf9UW3w8KJEdu1wKkJ-BTXugz2ti-fGOvoGQDIgU3xpphNefems7pPJQfUKQR9q7892L5Eyl-EZNFZpMadgZzR5fNlQkNUndYRF2BJnUAqNpRiP3Dzj9SiASzzqy2bUOp3e1uribRNcvsFBwVMuqvmEL5IHHIxYQ-TOUu0yiuP7E7Unf8LvwviIkLyWKD_odgo8VjdnoMoOzO_12nbk5BSk5K8X2_u4oMiaMJ5Vvp-o9Rf_ZUZkptnveXgDGJeLeUlMPfwHqLNw-vAyWlfYnGzMS_jxqoM_SYsRD1OQPyJHltj7zlU3GrDA0sLK4eIxicDRzstKpBM0mAQXvsdNwHeCceMPwY1LFLMVqwsrr9q7cdUTej6faVaSL1Y8np_oMREPyADUUHb8kETeQtsYb3iatSmt0SAqm8T9thxSAXUHy3JiphUYQP9IuZVRM90GODeWaMFTm9cgY4mryK_HD6YnwiP6LCelXxoCXVanx1cZP9IzbG15-FyTqu2OAmvdeaeBPiOWyXxNObVJApnunOv-_4UIvBOJLjwZeVGylj96mU5IFvvH2QGh9_ZM_beWqINt2UlVSKd4Xx8bjwGnGaP3wTdULzG3HDzW4vDhRkR_bwD7wcqclufwCywWgIFr8f0ZaQY-HhcXROvCzgW6lQjwD1fcrjSk811sVNGKIrSWLytopzaRIw2sQO2oTgADKIzzAKjN0Ef2ygJfNM6x1VV2RcgXHWX0vic0mggKj9VuxqTzKab3gG28JNXrCgJMuhZx5VUPbfpONShI48zSUQaXNzhe2X_-N7mLfl1S8LVtpSPGuxjUki3R_3e7qyF2CNVmDMm9FhHs5VZt8CaBLR659-3gVWw7AJ1J1bBH6jIPck2TUaqTG6Y5omeMW9Ac838_c0ogb3icHi-8Ro9dJ2RZoqqxRA2ow49f8slTptebXA7kv7CTyomgoadUGyRiTPNARc0qs--j5wfaFSSfoCn0WJBPZwHO0N0UUYFlGHVnXPimFbKxfBEBtgAIzOzPmaZ4bB8kiizP7--q21yQiIEVA6IH-p2mXTGC9YXYlYj3eXekBzB23C4fPZr0_isNJJpZdGmft3oX_9r6hSBag7wPmo-zklPNrLi0dHxtqoyL0RVMQ65X8SKM_eEpU66Q3KFJy4pBAwRgB6fUnP1T1AWCLu0-tnqpWXuRcvp0P5fSuT3s1IGzIvXMrbTwlOLMSR5eJtSdvFRnq23NpDz-tcF2khZsiGL24Y-sNt4Q73jEjrHSyyCjLKPVtbAWfPmKGxcS_TBQ7LaIHPLm-OphalgqrJs4POcHca18xH6nSnLpxE5Nr0bl1Fzb75rUThkZZcjGnYAqX5iJJpvHDxCUCx-8j2h2y5KoADc-8U61iac20Ty8AKAnpgny4AlWZJH0Z8BH5w7XVugaMyw_f70AUDuzRdas6aVafznw" alt="02">
  </a>
</figure>

놀라운 점은 RNN에 비해 4 epoch만에 90%의 Accuracy를 기록했고 9 epoch에서 95% 이상의 Accuracy를 기록했습니다. 뿐만 아니라 loss도 잘떨어졌습니다. 

### Code
- [https://github.com/emeraldgoose/hcrot](https://github.com/emeraldgoose/hcrot)
- [rnn.ipynb#LSTM](https://github.com/emeraldgoose/hcrot/blob/master/notebooks/rnn.ipynb)

## Reference
- [RNN과 LSTM을 이해해보자!](https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/)
- [LSTM - Pytorch Docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Backpropogating an LSTM: A Numerical Example](https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
