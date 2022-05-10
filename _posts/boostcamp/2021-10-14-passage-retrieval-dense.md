---
title: Passage Retrieval - Dense Embedding
categories:
  - BoostCamp
tags: [MRC]
---
## Introduction to Dense Embedding

### Limitations of sparse embedding

1. 자주 등장하는 단어의 경우 사실상 0이 되면서 90%이상의 벡터 디멘션들이 0이 되는 경우가 발생한다.
2. 차원의 수가 매우 크다 → compressed format으로 극복가능
3. 가장 큰 단점은 유사성을 고려하지 못한다.

### Dense Embedding

> Complementary to sparse representations by design

- 더 작은 차원의 고밀도 벡터(length = 50 ~ 1000)
- 각 차원 특정 term에 대응되지 않음
- 대부분의 요소가 nonzero값

### Retrieval: Sparse vs. Dense

- Sparse Embedding
    - 중요한 Term들이 정확히 일치해야 하는 경우 성능이 뛰어남
    - 임베딩이 구축되고 나서는 추가적인 학습이 불가능함
- Dense Embedding
    - 단어의 유사성 또는 맥락을 파악해야 하는 경우 성능이 뛰어남
    - 학습을 통해 임베딩을 만들며 추가적인 학습이 가능함
- 최근 사전학습 모델, 검색 기술 등의 발전 등으로 인해 Dense Embedding을 활발히 이용

## Training Dense Encoder

### What can be Dense Encoder?

BERT와 같은 Pre-trained language model (PLM)이 자주 사용되고 그 외 다양한 neural network 구조도 가능하다.  
Question과 Passage를 Encoder에 넣어 인코딩한 후 둘을 dot product하여 유사도 Score를 매긴다. 이 중 가장 큰 점수를 가진 Passage를 선택하여 쿼리에 답을 할 수 있도록 한다.

### Dense Encoder 학습 목표와 학습 데이터

- 학습목표 : 연관된 question과 passage dense embedding 간의 거리를 좁히는 것(또는 inner product를 높이는 것) = higher similarity
- Challenge : 연관된 question과 passage를 어떻게 찾을 것인가?
    - 기존 MRC 데이터셋을 활용
        
        ![](https://drive.google.com/uc?export=view&id=1K-8g0jT3XYO_ZQrt2J-Lte9tzqmTKq5K)
        

### Negative Sampling

- 연관된 question과 passage간의 dense embedding 거리를 좁히는 것 (higher similarity) = Positive
- 연간되지 않는 embedding간의 거리는 멀어야 함 = Negative
- Chosing negative examples:
    - Corpus 내에서 랜덤하게 뽑기
    - 좀 더 헷갈리는 negative 샘플들 뽑기 (ex. 높은 TF-IDF 스코어를 가지지만 답을 포함하지 않는 샘플)

### Objective function

Positive passage에 대한 negative log likelihood (NLL) loss 사용

### Evalution Metric for Dense Encoder

Top-k retrieval accuracy: retrieve된 passage 중에서 답을 포함하는 passage의 비율

## Passage Retrieval with Dense Encoder

### From dense encoding to retrieval

Inference: Passage와 query를 각각 embedding한 후, query로부터 가까운 순서대로 passage의 순위를 매김

### From retrieval to open-domain question answering

Retriever를 통해 찾아낸 Passage을 활용, MRC(Machine Reading Comprehension) 모델로 답을 찾음

### How to make better dense encoding

- 학습 방법 개선 (e.g. DPR)
- 인코더 모델 개선 (BERT보다 큰, 정확학 pretrained 모델)
- 데이터 개선 (더 많은 데이터, 전처리, 등)