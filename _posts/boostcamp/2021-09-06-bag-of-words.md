---
title: Bag-of Words 정리
categories:
  - BoostCamp
tags: [nlp]
---

> 현재 NLP는 Transformer가 주도하고 그 이전에는 RNN과 같이 Recurrent한 모델이 주도했지만 그 이전에는 단어 및 문서를 숫자 형태로 나타내는 Bag-of Words 기법을 정리한다.

## Bag-of-Words Representation

- Step 1. Constructing the **vocabulary** containing **unique words** → 사전을 만드는 과정
    - "John really really loves this movie", "Jane really likes this song"
    - Vocabulary : {"John", "really", "loves", "this", "movie", "Jane", "likes", "song"}
- Step 2. Encoding unique words to **one-hot vectors**
    - Vocabulary : {"John", "really", "loves", "this", "movie", "Jane", "likes", "song"}
        - John : [1 0 0 0 0 0 0 0]
        - movie : [0 0 0 0 1 0 0 0]
        - loves : [0 0 1 0 0 0 0 0]
        - ...
    - 임의의 단어들간의 거리는 $\sqrt{2}$ (Euclidean distance)
    - 임의의 단어들간의 cosine similarity는 0
        - 코사인 유사도는 같은 벡터일때 1, 다르면 0
    - Bag-of-Words vector : A sentence / document can be represented as **the sum of one-hot vectors**
        - sentence 1 : "John really really loves this movie"
            - John + really + really + loves + this + movie : [1 2 1 1 1 0 0 0]
        - sentence 2 : "Jane really likes this song"
            - Jane + really + likes + this + song : [0 1 0 1 0 1 1 1]
        - Bag-of-Words vector라 부르는 이유는 문장마다 단어들을 순차적으로 가방에 넣고 가방에 넣은 단어들의 수를 벡터로 표현한 것이기 때문입니다.

## NaiveBayes Classifier for Document Classification

- Bag-of-Words for Document Classification
    - Bag-of-Words vector를 정해진 카테코리 혹은 클래스 중의 하나로 분류할 수 있는 대표적인 방법
- Bayes' Rule Applied to Documents and Classes
    - MAP is "maximum a posterion" → most likely class
    - a document $d$ and a class $c$, the number of class is $C$
    - $C_{MAP} = argmax_{c \in C} \space P(c\|d) = argmax_{c \in C}\space \frac{P(d\|c)P(c)}{P(d)} = argmax_{c \in C} P(d\|c)P(c)$
        - Bayes Rule에 의해 $P(d)$가 필요하지만 argmax operation으로 인해 필요하지 않게 된다.
    - $P(d\|c)P(c) = P(w_1, w_2, ..., w_n\|c)P(c) → P(c)\Pi_{w_i \in W}P(w_i\|c)$
        - by conditional independence assumption
        - 특정 카테코리 $c$가 고정되었을 때 문서 $d$가 나타날 확률 $P(d\|c)$은 문서 안 word $w_1$부터 $w_n$까지 동시 사건으로서 볼 수 있다.
        - 각 단어들의 확률이 $c$가 고정되었을 경우 독립으로 가정할 수 있다면 각 단어들의 확률을 모두 곱한 것으로 나타낼 수 있다.
- 학습 데이터 내에 특정 단어가 전혀 발견되지 않았을 경우 확률 계산 시 그 단어에 대한 확률값은 0이 되면서 그 단어가 문장과 밀접한 관련이 있어도 해당 클래스로 절대 분류되지 않는다는 단점이 있다.