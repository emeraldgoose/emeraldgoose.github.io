---
title: Word Embedding 정리
categories:
  - BoostCamp
tags: [nlp]
---

> Word Embedding이란, 자연어가 단어들을 정보의 기본 단위로 해서 이런 단어들의 sequence라 볼때, 각 단어들을 특정한 차원으로 이루어진 공간상의 한 점 혹은 그 점의 좌표를 나타내는 벡터로 변환해주는 기법이다.

- 예를들어, cat과 Kitty는 비슷한 단어이므로, 거리가 가깝다. 그러나 cat과 hamburger는 비슷하지 않는 단어이므로 거리가 멀게 표현될 수 있다.
- Word embedding을 통해 두 문장이 유사한 입력임을 알 수 있어서 같은 클래스로 분류할 수 있게 된다.

## Word2Vec

- 같은 문장에서 나타난 인접한 단어들 간에 그 의미가 비슷할 것이라는 가정을 사용한다.
- 단어들의 거리에 따라 확률분포를 예측하게 된다.
    - "the cat purss", "this cat hurts mice"
    - P("the"\|"cat"), P("this"\|"cat"), P("hurts"\|"cat"), ... , P(w\|"cat")
- Word2Vec 모델이 "cat"과 자주 나타나는 단어들의 확률을 학습할 수 있게 된다.
- 학습할 때는 Sliding window를 통해 입력 단어와 출력 단어 쌍을 만들어 데이터를 생성한다.
    - "I study math." → (input, output) = (I, study), (study, I), (study, math), (math, study)
    - Input : "study" → [0, 1, 0]
    - output : "math" → [0, 0, 1]
    - [webi](http://ronxin.github.io/wevi/)라는 사이트에서 word embedding을 시각화하여 보여준다.

        ![](https://drive.google.com/uc?export=view&id=1GmV3Dws2H5edjE53UwWn2uAEc6vh4MLN)

        - "I study math"를 학습시키기 위해 hidden layer는 2, 위의 입력,출력 쌍을 데이터로 넣고 학습을 진행한다.
        - 각 데이터의 입력 단어와 출력 단어 벡터의 내적값이 멀어지도록 학습을 진행하게 되고 500번 진행 결과 다음과 같이 시각화되어 출력된다.

            ![](https://drive.google.com/uc?export=view&id=1BtD_dj_cI-Ey6jFg636n5rdoh-rbc8SC)

            - Weight Matrices를 보면 "I"에 해당하는 가중치 벡터와  "math"에 해당하는 가중치 벡터가 유사한 모습을 보여준다.
            - 빨간색은 양수, 파란색은 음수를 표현한다.
            - 비슷한 단어는 뭉치고 다른 단어는 떨어지는 모습도 볼 수 있다.
        - 통상적으로 Input Vector의 가중치 행렬을 최종 결과로 사용하게 된다.
        - 이 가중치 벡터를 통해 단어들 간의 관계를 표현하게 된다.

          ![](https://drive.google.com/uc?export=view&id=1g4de93TVrpVI2eHMi1Sr1EYSzH6JigPx)

            - vec[queen] - vec[king] = vec[woman] - vec[man]
- Word2Vec을 통해 Word intrusion detection이라는 task가 가능하다.
  - 문장에서 나머지 단어와 가장 상이한 한 단어를 찾는 task
  - embedding vector를 통해 유클리드 거리를 사용하여 평균을 취하여 평균거리가 가장 큰 단어를 찾으면 그 단어가 나머지 단어와 가장 상이한 단어임을 알 수 있다.
- Word2Vec의 시간복잡도
  - CBoW 모델에서 하나의 단어를 처리하는데 드는 계산량
    - C개의 단어를 Projection하는데 C x N
    - Projection Layer에서 Output Layer로 가는 데 N x V
    - 따라서 O(CN+NV)
  - Skip-gram 모델에서 하나의 단어를 처리하는데 드는 계산량
    - 현재 단어를 Projection하는 데 N
    - Output을 계산하는 데 N x V, 테크닉을 사용하면 N x lnV
    - 총 C개의 단어에 대해 진행하므로 x C
    - 따라서 O(C(N + N x lnV))
  - [참고](https://shuuki4.wordpress.com/2016/01/27/word2vec-%EA%B4%80%EB%A0%A8-%EC%9D%B4%EB%A1%A0-%EC%A0%95%EB%A6%AC/)
   - Word2Vec은 몇 개의 단어를 샘플링하냐에 따라서 계산량이 비례해서 올라간다.

## GloVe : Another Word Embedding Model

- 각 입력 및 출력 단어 쌍들에 대해서 학습 데이터에서 두 단어가 한 window 내에서 총 몇번 동시에 등장했는지 사전에 계산을 미리 하고 그 값에 대해 최적화된 loss function을 사용한 모델이다.
    - $J(\theta)=\frac{1}{2}\sum_{i,j=1}^{W}f(P_{ij})(u_i^Tv_j-logP_{ij})^2$
    - 임베딩된 중심 단어 벡터와 주변 단어 벡터의 내적이 두 단어가 동시 등장 확률이 되도록 하는 loss function이다.
        - GloVe의 복잡성은 행렬 P의 0이 아닌 요소의 수에 따라 결정됨
        - Shallow window(Shallow, Local-context-based window method) 기법은 말뭉치 전체 사이즈 C에 고려됨
        - 동시 발생 단어의 분산에 대해 빈도수의 순위가 멱함수를 따른다고 가정
        - 말뭉치를 전체 스캔하는 것에서 통계적 수치를 최적화 함
        - GloVe의 시간 복잡도는 $O(\|C\|^\frac{1}{\alpha})$이다.
        - [참고](https://insighting.tistory.com/10)
    - 이를 통해 중복되는 계산을 줄여 줄 수 있다. → Fast training → 적은 데이터에 대해서도 잘 학습된다.