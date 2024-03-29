---
title: Basics of Recurrent Nerual Networks (RNN)
categories:
  - BoostCamp
tags: [nlp]
---
## Types fo RNNs
- 각 타임스텝에서 들어오는 입력벡터 $X_t$와 이전 state에서 계산된 $h_{t-1}$을 입력으로 받아서 현재 타임스텝에서의 $h_t$를 출력으로 내어주는 구조를 가지고 있다.
- 모듈 A가 재귀적으로 호출되는 구조를 가지고 있다. 왼쪽의 그림을 rolled 버전, 오른쪽 그림을 unrolled 버전이라 한다.

## Recurrent Neural Network

- $h_{t-1}$ : old hidden-state vector
- $x_t$ : input vector at some time step
- $h_t$ : new hidden-state vector
- $f_W$ : RNN function with parameters $W$
    - 파라미터 $W$는 모든 타임 스텝에서 공유되는 것이 RNN의 특징
- $y_t$ : output vector at time step $t$
    - 만약 출력해야 하는 y가 품사인 경우 매 타임스텝마다 단어의 품사를 출력해야 하고 만약 문장의 sentiment를 출력해야 하는 경우(positive, negative) 맨 마지막 state에서 출력해야 한다.
- hidden state $h_t = f_W(h_{t-1},x_t)$
    - $h_t=f_W(h_{t-1},x_t)=tanh(W_{hh}h_{t-1}+W_{xh}x_t)$
    - $y_t=W_{hy}h_t$
- 예를들면, $x_t = (x,1)$와 $h_t = (h,1)$이라고 가정하면
    - $W = (h, x+h)$인 크기를 갖게 되고 $W$의 $(h,x)$는 $X_t$와 내적하고 $W$의 나머지 부분인 $(h,h)$는 $h_t$와 내적하게 된다.
    - $W_{xh}$는 $X_t$를 $h_t$로 변환해주는 역할을 하고 $W_{hh}$는 $h_{t-1}$을 $h_t$로 변환해주는 역할을 한다. 이렇게 계산된 값을 비선형 변환 중 하나인 $tanh$을 통과시켜준다.
    - $W_{hy}$는 $h_t$를 $y_t$로 변환해주는 역할을 하는 것으로 이해할 수 있다.
    - 여기에 sigmoid를 씌우게 되면 binary classification을 수행할 수 있고 multi class classification에서는 $y_t$가 class개수만큼의 dimension을 가지는 벡터로 나와서 softmax 레이어를 통과해서 분류하고자 하는 클래스와 동일한 개수만큼의 확률분포를 얻을 수 있게 된다.

- Types of RNNs
    - one to one : Standard Nerual Networks
    - one to many : Image Captioning
        - one to many의 경우 입력이 들어가지 않게 하기 위해 입력과 동일한 크기의 0으로 채워진 입력을 넣어준다.
    - many to one: Sentiment Classification
    - many to many(left) : Machine Translation
    - many to many(right) : Video classification on frame level, POS

## Character-level Language Model

- 문자열이나 단어들의 순서를 바탕으로 다음 단어가 무엇인지 맞추는 모델
- sequence : "hello"
    - Vocabulary : [h, e, l, o] → one-hot vector : [1, 0, 0, 0], [0, 1, 0, 0], ...
    - $h_t = tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$
    - Logit = $W_{hy}h_t + b$
        - 선형변환을 통과하여 사전의 크기와 동일한 output vector가 나오고 multi class classification을 수행하기 위해 softmax layer를 통과하게 된다.
        - "h"의 예측값은 "o"이지만 ground truth는 'e'이므로 softmax loss를 통해 'e'에 가까워지도록 학습하게 된다.
    - 3번째와 4번째는 같은 입력이 들어오지만 다음으로 올 문자는 달라야 한다. 이것이 가능한 이유는 이전 타임스텝에서 넘어오는 정보가 있기 때문이다.
- 문단에 대해서도 학습을 할 수 있는데 이때 문장안에 빈칸이나 특수문자도 Vocab의 dimension을 차지하게 된다.

### Backpropagation through time (BPTT)

- 임의의 타임스텝 $t$에서 $W_{hh}$, $W_{xh}$, $W_{hy}$ 모두 backpropagtion을 수행하게 되는데 이것이 많아질수록 메모리 한계가 올 수 있다. 이를 예방하기 위해 일부를 잘르는 truncation작업을 통해 제한된 길이의 sequence를 학습시키는 방법을 사용하고 있다.

### Searching for Interpretable Cells

- RNN에서 필요로하는 정보를 저장하는 공간은 매 타임스텝마다 업데이트를 수행하는 hidden state vector $h_t$이다.
- 필요한 정보가 hidden state vector의 어디에 저장되었는지 확인하려면 vector의 각각의 차원 하나를 고정?하여 타임스텝가 진행됨에 따라 어떻게 변하는지를 분석함으로써 RNN의 특성을 확인할 수 있다.

### Character-level Language Model

![](https://lh3.googleusercontent.com/fife/ALs6j_FebZol2fq2T7B8tkPNf6hNih7D2NPA2PdQxiAlyb2mWMx7VrIGXXCzXVmh2yiGs64Ly2nBMp3TRALQ3QNZtPQjqEBCbsZ0neosdw-rvDvXagpeoSkhhFyWyCOnp2Ia0rxH3-FKZ3GyhYsy9u1aa5JcCowtp5WLt-S253dUV4M0Aea5AHPPtCiUvkNmEJXdrXrkTsG2aVCbUYvXFivK1-bGEQA_XBvX7B6B_8_aNcXBZmggKMfr-lMWHo5Xotljyfh8VElsbk7OzDppaBE2c6Pa4hkcc2tY94gfpStkZa0Cw5ygdsq8LaqIhDaI0laFGYUkzFfsKzRZg_MpOYPS39FicNOCn3hLt0AlrfNc_ZP6DPCL4BvugEfbxMD235B1XFeP6yWwH-sqVFGRnfHPmu09lmoNHo_s64YnYLTuFTImFX2ncyRQOsn68sbn8R8D-A-l5M6Ediygt-1ODexfxHgCFWK-t6_g3Rl8fOP17Jr-Hpe0HsSCuzCNJ4DTkCzxNXA4YrwuIFjfRMyofu6O12mDIXU4ikedD-evr_y85f3RJ60j7J5yKXuPHz4HRCZ0IrlMqv2-w_25yj4KcWlvFytcs5v5R6UiK9WoCYYkITDyaiR58z5bhYBXw8CsAEKThaLjFWhdNTXlOsY15aRwybfhj7nJFEgOrA16700RpWk0OIg7Fys09Hgbq1Yui0ntlAo1weZDDt-pmorRORcooxhYQoO70muVNHdUwoNCPkbzfi9ArLrlGDxfdPvj6nOEiscf9sQSbrEWLJfcCcEAERts3fEejk1fcWzhxJjVia07sh4p78hed7arA8ubdduou3n803MSG5KY7vlOiQHjfsl0GzJKYkaTX8ukI_xsjj4FdiTC0QC8CfXcZgM0ph3dI2SGoZHqO02NObxu5Sbf5ZFBylS2wURuxv8SxGbG0u8rPBNwFHQjlLtoKceThgkuF2MCe_BGCWnrbcJNCKnwDUXrLA26B3AFM0QBixVMEpn0UhbvzU5QY7EgwF9YM7JnjRQq-HgUHRPHE96C1EZo2KPeNAHatQ9eu0q1cccFwL5aK54PE-kveM7xt9OSHjvfWBfLgGUizZaGe2Vn6GL6lJcAioFapS0qItKAlIWWy5EYI8hi-bTudS5zC7izKIN5HXRiUhe_Gg7kzErXel0k4Zo1Y2gpbqBO1IgSaV9XOwKnMXbgpZT8RFKc5Bs-tP0BOG_cAWQJo4bKgpLQ6lKdmZomAADc8hDmBZLyPJo7vLmb2wwfcl6POG2l4b7Cttuzjcb9zATNRfg55E_jWnBOfN8Qoyvs81xRdD_-0JR4ViCVR1iaWN2yQjDdy0KHFRIeyIJiui7CUTuK4jjogzbsdFURHgKYB_5w2Ba7RMSFlcAAc_Wkh_Io6thNuNn7cDX_8fdx0ooGmTVkJFJW8eUOEu1jO-qyxOlrt-etxusyI3Smy-kRJIFeXlc_0fA03b6NDgm8_L-aA9jEmdjMl17Va74I7Qef8xXpjahTOmT7-0nuz1opp-ATuvk8gAUH7HZgNAEN4DIa1nkU8RpM9RAVoHHOiItUWQmeOJJ1OX6QlxpGRg1WdiZveh0Q9ZzYb-1nXBrqXPeRfEB-hjB3wjhkHeIXC7-NsC0bOQ6N2XEno_LtqPdY)

- 특정한 dimension의 노드 state가 어떻게 변하는지 시각화한 모습이다.

![](https://lh3.googleusercontent.com/fife/ALs6j_GKx6ijvX19ueovyIE6TDIqkcGnequ8eajSB6p7FPQjxTS9KGWonkqooDl3wKnvvBYjgdrJT_ep3RwBzwtg5pQ7_sGONQcFEROv4qHW-FjX0xz7aS-4u5CIPnYMzmJSvy8IZFVXH3cpoSyDgVxEKe4Is8izhXRtf-m4aFcepWk0Wqg59ipbTYeb30i1VUEZt-3XWQZ84NeVuMqxhzHGW8qEz54v8NEIKWxN73dKVBbBVLNopSuy-NuzsVz8nOXNuAKkUipse3S-rY9YfulOKdgcqEuZ0Fq2iYddwKiZJcl7GzxC9rYvxV9u5Vnp5eAI7hi8vLHNzjKZPbpa0zowjQyy68W4NSqsmH4HZY_LZc6Qs-zmzk24BvDJN3algYlYXIuBscshKgB_fTXNMsBw9bheoaMOLarrNPh1n0vH_OH4tECjjYQkvedumLT5w9M0OzVptyF9_M8mDoBCowEFFe3Josdj4QcQ_GouQDOxqVQ2rK-UoUAFJbF_KT_xRbkPyY50ErGkzc1wx5DbvJVT7wHb57uZYhDblS0hMoMDFtFRjpzYm15YU1NO43uC3Xa1Lpna_2HEbGUP3rNP5SxzUOwMt5GlKXO0hjLljdKgUGaKSngLKY9o91I1Nij_3Lqtj7sCxEIn8HtKKIcC0olvBKEv7k7_zE4MspYtmFn-07enzE2jmYUjPXr47xPJWmq9JqFmwaPfEHfPJ-Oa8FtO7ivb4cFw-xYDA6qHr6vI47M5X5iDSoN5ip8IrQv5g9DegLPb7mKtZwV0vKdemZrLoZmjwHYdy4tw3hdiQpNoix2cf8T9w2PNGppBUe0BtC9xjIIuKTBpDSdghDFFmc6nTIqBks29QzU6pQmw3Mh5FER9aKTcRYnyUrS63LNTU_l1rxUVQfDGiYBw30NfN3tmk-wCY-Cf9_Qnq47geIWzJyD3kpRfdPN37x27CJSJXAxxUVHBTN1FK9zkKWATIgspMVdYgB-DBCaDNwl0FJEeqMRQf5NqqijYvLnJ6dIs-_dTyy6F8Mf3UCgxxxDowWvL8WRB6CAkLCGoo28bFw6u3iAO0GV9r7iHpov9A-Ry82H_Q9N6dXq3UKrUN8QbqFEJOlLMpNXrAgitS_bsL1bLe_bEHZ2z36IbRS2e61QxR_-2_4zI3T2M4JjMpSlsXJPUDWuCJ3JMUP7MYFLadwonRqDNVhAArZADapX5MKoZkr80ywYQb9V03B093Nck73SQj4Ef7nqcK54a2OKIWSp8uNN3WhOg_ke_NXFXHUO9P09zsWSuLoMjFy7pO2IUY0uz7j8O3xEgU0Hioc4O0oQyqn1fPcnS8vGHb7_pUIuAX3HGcN6CMw2myFuiG14RSiEGYyModAEjz2LjMelVgXHmrb3KuxYFhdeid6knMuJKau7Jt-qgnlKA3Z7h23FgvCMAzfyoqFo9jSwukuA2RS6PVIeGnVDdq2Df0FIjgctMs7v_eOj9VjbUQkrtUtnPRLWPCeNKujPGEqJTSRjH79PMGqCri3krmFpTExbinvjYDnrAOLbRWKKgTAW-JSU4gDuqYSZEu942GXjuIg36CT2f5Hby81R9svw7f2QDeAEPpLE1SwBpqYpmH09ZAT23LW92fOoHTTO0G8olQDCrAfK1SBI1VP6U)

- "가 열릴때 음수였다가 "가 닫히면서 양수, 다시 열리면 음수가 되는 것을 보면 이 모델은 "의 정보를 계속 기억하고 있음을 볼 수 있다.

- 그림 출처 : [http://karpathy.github.io/2015/05/21/rnn-effectiveness/](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### Vanishing/Exploding Gradient Problem in RNN

- RNN은 동일한 matrix를 매번 곱하게 backpropagation의 결과가 매우 작아지거나 매우 커지는 경우를 볼 수 있다.
- 간단한 예시:
    - $h_t = tahn(W_{xh}x_t + W_{hh}tahn(W_{xh}x_{t-1}+W_{hh}tahn(W_{xh}x_{t-2}+W_{hh}h_{t-2} + b) + b) + b)$
    - $h_t$에서 backpropagation을 위해 편미분을 실행하게 되는데 타임스텝를 거슬러 올라갈수록 점점 증폭된다. 이때 이전 타임스텝의 유의미한 값을 전달해줄 수 없게 된다.
