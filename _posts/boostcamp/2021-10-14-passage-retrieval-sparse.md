---
title: Passage Retrieval - Sparse Embedding
categories:
  - BoostCamp
tags: [MRC]
---
## Introduction to Passage Retrieval

### Passage Retrieval

> 질문(query)에 맞는 문서(passage)를 찾는 것.
> 

### Passage Retrieval with MRC

Open-domain Question Answering: 대규모의 문서 중에서 질문에 대한 답을 찾기

- Passage Retrieval과 MRC를 이어서 2-Stage로 만들 수 있음
    
    ![](https://lh3.google.com/u/0/d/14iurFvUnXIiLZj_XEwNbh5M2ze3ZeDxV)
    

### Overview of Passage Retrieval

Query와 Passage를 임베딩한 뒤 유사도로 랭킹을 매기고, 유사도가 가장 높은 Passage를 선택함

![](https://lh3.google.com/u/0/d/1BE1hDFSaRXPx3aTVvuP5BRSe_NU1S210)

질문이 들어왔을 때, 질문을 어떤 vector space로 임베딩하고 마찬가지로 Passage 또한 같은 vector space에 임베딩한다. Passage의 임베딩은 미리 해놓아 효율성을 높인다.  
다음, 쿼리와 문서의 Similarity를 측정하여 Ranking을 매기게 된다. 이때 Score는 nearest neighbor(고차원 space에서 서로의 거리를 측정)와 inner product 방법을 사용한다.

## Passage Embedding and Sparse Embedding

### Passage Embedding

> 구절(Passage)을 벡터로 변환하는 것
>

### Passage Embedding Space

> Passage Embedding의 벡터 공간
벡터화된 Passage를 이용하여 Passage 간 유사도 등을 알고리즘으로 계산할 수 있음
> 

### Sparse Embedding 소개

- Bag-of-Word(BoW)
    - BoW를 구성하는 방법 → n-gram
        - unigram (1-gram)
        - bigram (2-gram)
    - Term value를 결정하는 방법
        - Term이 document에 등장하는지 안하는지 (binary)
        - Term이 몇번 등장하는지 (term frequency), 등. (eg. TF-IDF)

### Sparse Embedding 특징

1. Dimension of embedding vector = number of terms
    1. 등장하는 단어가 많아질수록 증가
    2. N-gram의 n이 커질수록 증가
2. Term overlap을 정확하게 잡아 내야 할 때 유용
3. 반면, 의미(semantic)가 비슷하지만 다른 단어인 경우 비교가 불가

## TF-IDF

### TF-IDF (Term Frequency - Inverse Document Frequency) 소개

- Term Frequency (TF) : 단어의 등장빈도
- Inverse Document Frequency (IDF) : 단어가 제공하는 정보의 양
- ex) It was the best of times
    - It, was, the, of : 자주 등장하지만 제공하는 정보의 양이 적음
    - best, times : 좀 더 많은 정보를 제공, 문서에서 등장하는 정도가 적다 = 중요한 정보

### Term Frequency (TF)

> 해당 문서 내 단어의 등장 빈도

1. Raw count
2. Adjusted for doc length: raw count / num words (TF)
3. Other variants: binary, log normalization, etc.

### Inverse Document Frequency (IDF)

> 단어가 제공하는 정보의 양

- $IDF(t) = log\frac{N}{DF(t)}$
- Document Frequency (DF) = Term $t$가 등장한 document의 개수
- $N$ = 총 document의 개수
- 모든 문서에 등장하는 단어의 경우 N = DF이기 때문에 log를 씌어주면 0이 된다.

### Combine TF & DF

TF-IDF($t$, $d$): TF-IDF for term $t$ in document $d$, $TF(t, d) \times IDF(t)$

- 'a', 'the' 등 관사 → Low TF-IDF
    - TF는 높지만 IDF가 0에 가까울 것 (거의 모든 document에 등장 → log(N/DF) = 0)
- 자주 등장하지 않는 고유명사 (ex. 사람 이름, 지명 등) → High TF-IDF (IDF가 커지면서 전체적인 TF-IDF가 증가)

### BM25

> TF-IDF의 개념을 바탕으로, 문서의 길이까지 고려하여 점수를 매김

- TF 값에 한계를 지정해두어 일정한 범위를 유지하도록 함
- 평균적인 문서의 길이 보다 더 작은 문서에서 단어가 매칭된 경우 그 문서에 대해 가중치 부여
- 실제 검색엔진, 추천 시스템 등에서 아직까지도 많이 사용되는 알고리즘

$Score(D, Q) = \sum_{term \in Q} IDF(term) \cdot \frac{TFIDF(term, D) \cdot (k_1 + 1)}{TFIDF(term, D) + k_1 \cdot (1 - b + b \cdot \frac{\vert D \vert}{avgdl})}$