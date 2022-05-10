---
title: Encoder-Decoder & Attention mechanism 정리
categories:
  - BoostCamp
tags: [attention]
---
## Seq2Seq Model

- seq2seq는 RNN의 구조 중 Many-to-many 형태에 해당한다.

    ![](https://drive.google.com/uc?export=view&id=1HVLlyrlYemel2aojBhxw6Oe8ypoCTdq2)

    - 인코더와 디코더는 서로 공유하지 않는 모듈을 사용한다.
    - 인코더는 마지막까지 읽고 마지막 타임스텝의 hidden state vector는 디코더의 $h_0$ 즉, 이전 타임스텝의 hidden state vector로써 사용된다.
    - 디코더에 첫 번째 단어로 넣어주는 어떤 하나의 특수문자로서 <start>토큰 혹은 <SoS>(Start of Sentence)토큰이라 부른다.
        - 이런 특수한 토큰은 Vocabulary 상에 정의해두고 이를 가장 처음 디코더 타임스텝에 넣어서 실질적으로 생성되는 첫 번째 단어부터 예측을 수행하게 된다.
    - 또한, 문장이 끝나는 시점에도 특수문자인 <EoS>(End of Sentence)토큰이 나오게 되는데 디코더는 EoS 토큰이 나올때까지 구동된다. EoS토큰이 생성되면 다음부터 단어가 생성되지 않는다.
- Seq2Seq Model with Attention
    - 기존 lstm의 경우 문장의 길이와 상관없이 모든 단어의 정보를 맨 마지막 vector에 욱여넣어야 한다. 이때, 멀리 있는 단어의 경우 정보가 소실되거나 줄어버리는 문제(bottleneck problem)가 있기 때문에 Attention을 사용하려고 한다.
    - 단어를 넣을 때마다 생성된 모든 hidden state vector를 디코더에 제공하고 각 타임스텝에서 단어를 생성할 때 그때그때 필요한 인코더 hidden state vector를 선별적으로 가져가서 예측에 도움을 주는 형태로 제공된다.

        ![](https://drive.google.com/uc?export=view&id=1Uxv60oMwaj6_L3wHl9FDlpwez_rj1pZf)

        - 각 단어가 입력되고 계산된 hidden state vector가 있고 마지막 타임스텝에서 생성되어 $h_0$가 디코더로 들어가게 된다.
        - 디코더는 start 토큰이 word embedding으로 주어지고 $h_0$를 디코더의 hidden state vector $h_1^{(d)}$를 계산하게 된다.
        - $h_1^{(d)}$는 다음 단어의 예측에 사용될 뿐만 아니라 이 벡터를 통해 인코더에서 주어진 4개의 hidden state vector 중 어떤것을 필요로 하는지 고를 수 있다.
            - 고르는 과정은 각각의 $h_i^{(e)}$와 $h_1^{(d)}$를 내적을 통해 그 내적값은 인코더 state vector와 디코더 state vector의 유사도라 생각할 수 있다.
            - 각각 내적된 값들을 각각의 인코더 hidden state vector에 대응하는 확률값을 계산해 줄 수 있는 입력벡터 혹은 logit vector로 생각할 수 있다.
            - softmax에 통과시킨 값은 인코더 hidden state vector의 가중치로 사용될 수 있고 각 hidden state vector에 적용하여 가중 평균으로서 나오는 하나의 인코딩 벡터를 구할 수 있다. 여기서 Attention output이 이 인코딩 벡터이고 Attention output을 Context vector로 부르기도 한다.
            - Attention distribution은 softmax layer를 통과한 output인데 합이 1인 형태의 가중치를 Attention vector라고 부른다.
        - 이제 Attention output(혹은 context vector)과 디코더의 hidden state vector가 concat이 되어서 output layer의 입력으로 들어가게 되고 다음에 나올 단어를 예측하게 된다.

            ![](https://drive.google.com/uc?export=view&id=1t6m4xai2cnUGW7TjCVXqmH1e6X8UaCnz)

        - 두 번째도 동일하게 $h_2^{(d)}$와 각 encoder hidden state vector의 내적을 통해 Attention output(context vector)를 구하게 된다.

            ![](https://drive.google.com/uc?export=view&id=1EKG7hU16-ELwPhBgTZ7ZIjgjI5RzqJqw)

        - EoS토큰이 나오기 전까지 반복한다.

            ![](https://drive.google.com/uc?export=view&id=1KW1z7ICYoq1ABBqr1mvk354s2lE4RR17)

        - Backpropagation
            - 디코더의 hidden state vector는 output layer의 입력으로 들어감과 동시에 인코더의 각 word별로 어떤 hidden state vector를 가져와야 할지 결정하는 attention 가중치를 결정해주는 역할을 수행하게 된다.
            - Backpropagation을 수행하게 되면 디코더와 attention 가중치까지 업데이트 하게 된다.
        - 한 가지 중요한 점은 Train 단계에서의 입력과 Inference 단계에서의 입력의 방법이 다르다는 점이다.
            - Train 단계에서는 Ground Truth 단어들을 입력으로 넣어주는데 그 이유는 어떤 단계에서 잘못된 단어가 출력이 되었고 그 출력을 다시 입력으로 넣었을 때 다시 잘못된 방향으로 학습할 수 있기 때문이다. 이러한 입력 방식을 "Teacher forcing"이라한다.
            - 반면에 Inference 단계에서는 출력된 단어를 다시 입력으로 넣어 다음 단어를 예측하도록 한다. 이러한 입력 방식은 실제 모델을 사용했을 때와 가깝다는 특징이 있다. 이러한 방법은 자기회기(autoregressive)하도록 추론을 적용하는 방식이다.
            - Teacher forcing방법만으로 학습하게 되면 test 시 괴리가 있을 수 있는데 초반부터 teacher  forcing 방법으로 학습하다가 후반부에 출력을 입력으로 넣는 방법을 사용하여 괴리를 줄이는 방법이 존재한다.

## Difference Attention Mechanisms

- 앞에서 내적을 통해 hidden state vector간의 유사도를 계산했는데 내적이 아닌 다른 방법으로 유사도를 구하는 attention이 있다.
- $socre(h_t,\bar{h_s})=\begin{cases} h_t^{\top} \bar{h_s}, & dot \cr h_t^{\top} W_a\bar{h_s}, & general \cr v_a^{\top} tanh(W_a[h_t;\bar{h_s}]), &concat \end{cases}$
    - 인코더 hidden state vector $\bar{h_s}$, 디코더 hidden state vector $h_t$
- general dot product
    - $W_a$는 두 hidden state vector 사이에서 행렬곱을 하게 하여 모든 서로 다른 dimension끼리의 곱해진 값들에 각각 부여되는 가중치이면서 학습가능한 파라미터 형태인 행렬이다.
- concat
    - 디코더 hidden state vector와 유사도를 구해야 하는 인코더 hidden state vector가 입력으로 주어졌을 때 유사도(scalar)를 output으로 하는 Multi layer perceptron 혹은 neural net을 구성할 수 있다.
        - Layer를 더 쌓아서 구성할 수도 있다.
    - 위 수식의 경우 $W_a$를 첫 번째 layer의 가중치로 두고 tanh를 non-linear unit으로 적용한다. 마지막에 선형변환에 해당하는 $v_a$를 적용하여 최종 scalar값을 출력하도록 한다. 중간 layer의 결과가 vector이므로 scalar로 계산해야 하므로 $v_a$또한 벡터로 주어져야 한다.
- Attention is Great!
    - NMT(Neural Machine Translation) performance를 상당히 올려주었다.
    - bottleneck problem을 해결하고 vanishing gradient problem에 도움이 된다.
        - 이 모델에서의 bottleneck problem은 맨 마지막 hidden state vector에 모든 단어의 정보가 담겨있지 못하는 문제를 말한다. Attention을 통해 디코더에서 필요한 인코더의 hidden state vector를 사용할 수 있게 해주면서 마지막 인코더 hidden state vector의 부담이 줄어든다.
    - Attention은 interpretablilty(해석 가능성)을 제공해준다. 또한, 언제 어떤 단어를 배워야 하는지 스스로 alignment를 학습한다.

## Beam search

- 자연어 생성 모델에서 Test에서 보다 더 좋은 품질의 생성결과를 얻을 수 있도록 하는 기법이다.
- Greedy decoding
    - 어떤 sequence로서의 전체적인 문장의 어떤 확률값을 보는게 아니라 근시안적으로 현재 타임스텝에서 가장 좋아보이는 단어를 그때그때 선택(aproach)하는 형태를 말한다. (Greedy Aproach)
    - 만약 잘못 생성된 단어가 있을 때는 뒤로 돌아가지 못한다.
- Exhaustive search
    - $P(y\|x)=P(y_1\|x)P(y_2\|y_1,x)P(y_3\|y_2,y_1,x)...P(y_T\|y_1,...,y_{T-1},x)=\Pi_1^TP(y_t\|y_1,...,y_{t-1},x)$
        - $P(y_1\|x)$ : $x$를 입력했을 때 출력문장 $y$에서의 첫번째 단어 $y_1$의 확률
        - $P(y_2\|y_1,x)$ : $x$와 $y_1$이 주어졌을 때 $y_2$의 확률
        - 입력문장 $x$에 대해 출력문장 $y$가 나올 확률을 최대한 높여야 한다. 첫 번째 $P(y_1\|x)$를 조금 낮춰서라도 뒤의 확률을 높일 수 있다면 전체 확률 또한 증가할 것이다.
    - 타임스텝 $t$까지의 가능한 모든 경우를 다 따진다면 그 경우는 매 타임스텝마다 고를 수 있는 단어의 수가 Vocabulary 사이즈가 되고 그것을 $V$라 하자. 그렇다면 $V^t$인 가능한 경우의 수를 구할 수 있다. 그러나 너무 크기 때문에 시간이 너무 오래 걸린다.
- Beam search
    - Greedy aproach와 exhaustive aproach의 중간에 해당하는 기법이다.
    - 디코더의 매 타임스텝마다 우리가 정해놓은 $k$개의 가능한 경우를 고려하고 마지막 까지 진행한 $k$개의 candidate 중에서 가장 확률이 높은 것을 선택하는 방식이다.
    - $k$개의 경우의 수에 해당하는 디코딩의 output을 하나하나의 가설(hypothesis)라 부른다.
    - $k$는 beam size라 부르고 일반적으로 5 ~ 10사이에서 설정한다.
    - joint probability에 log를 취해 덧셈으로 만들어버린다.
        - $score(y_1,...,y_t) = logP_{LM}(y_1,...,y_t\|x) = \sum_{i=1}^t logP_{LM}(y_i\|y_1,...,y_{i-1},x)$
        - 물론 log는 단조증가하는 함수이므로 최댓값은 유지된다.
    - Beam search는 모든 경우의 수를 보는 것은 아니지만 완전탐색하는 것보다 효율적으로 계산할 수 있게 한다.
        - 아래 그림은 $k=2$인 beam search의 예제이다.

            ![](https://drive.google.com/uc?export=view&id=1ii3SARsgu5Hp83eTqC5UzlJtBDEa1kf3)

            - 그림 출처 : [https://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture08-nmt.pdf](https://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture08-nmt.pdf)

            - 각 단계에서 생성되어 확률을 확인하고 그 중 $k$개까지만 다음 단어를 생성할 수 있도록 한다.
    - greedy decoding은 모델이 END 토큰을 생성할 때 끝내도록 되어있다.
    - beam searching decoding은 서로 다른 시점에서 END 토큰을 생성할 수 있다.
        - 어떤 hypothesis가 END토큰을 만든다면 현재 시점에서 그 hypothesis는 완료한다. 완성된 문장은 임시 공간에 넣어둔다.
        - 남은 hypothesis 또한 END토큰을 만들때까지 수행한 후 임시 공간에 넣어둔다.
    - 미리 정한 타임스텝 $T$까지 디코딩하여 빔 서치 과정을 중단하거나 $T$시점에 END토큰을 발생시켜 중단하도록 한다. 또는 미리 정한 $n$개 만큼의 hypothesis를 저장하게 되면 빔 서치를 종료하게 된다.
    - 빔 서치 종료 후 완성된 hypotheses의 리스트를 얻게 되고 이 중 가장 높은 score를 가진 하나를 뽑아야 한다.
        - 그러나 긴 길이의 hypotheses들은 낮은 score를 가진다는 예상을 할 수 있다. 왜냐하면 log때문에 새로운 단어가 생성될 때마다 음수값을 더해줘야 하기 때문이다.
        - 그래서 좀 더 공평하게 비교하기 위해 각 hypotheses별로 Normalize를 수행한다.
            - $score = \frac{1}{t}\sum_{i=1}^t logP_{LM}(y_i\|y_1,...,y_{i-1},x)$

## BLUE score

- 기존 평가방법의 한계
    - 정답 I love you와 예측 oh I love you라는 문장이 주어진 경우 각각의 위치에는 모두 다른 단어이기 때문에 예측률 0%로 잡히게 된다.
    - Precision and Recall
        - Precision = # of correct words / length of prediction(예측)
        - Recall = # of correct words / length of reference(정답)
        - F-measure = (precision $\times$ recall) / (precision + recall) / 2 (조화평균)
            - 조화평균은 Precision과 Recall 중 작은 수에 치중하는 특징을 가지고 있다.
    - 하지만 F-measure가 높다고 해서 그 문장의 문법까지 맞지는 않는 단점이 있다.
- BLEU(BiLingual Evaluation Understudy) score
    - N-gram이라는 연속된 N개의 연속된 단어가 ground truth와 얼마나 겹치는 가를 계산하여 평가에 반영하는 방법이다.
    - Precision 만을 반영하고 Recall은 무시한다. 정답에서 몇 개의 단어가 떨어져도 비슷한 해석이 가능하기 때문이다.
    - BLUE = min(1, length_of_precision / length_of_reference)$(\Pi_{i=1}^4 precision_i)^{\frac{1}{4}}$ (기하평균)
        - min(1, length precision/length reference)는 brevity panelty라 부르는데  길이만을 고려했을 때 ground truth 문장과 짧을 경우 짧은 비율만큼 뒤에서 계산된 precision을 낮추는 역할을 한다.