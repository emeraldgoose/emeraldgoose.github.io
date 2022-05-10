---
title: QA with Phrase Retrieval
categories:
  - BoostCamp
tags: [MRC]
---
## Phrase Retrieval in Open-Domain Qeustion Answering

### Current limitation of Retriever-Reader approach

1. Error Propagation: 5-10개의 문서만 reader에게 전달
2. Query-dependent encoding: query에 따라 정답이 되는 answer span에 대한 encoding이 달라짐

### How does Document Search work?

사전에 Dense 또는 Sparse vector를 만들고 indexing. MIPS와 같은 방법으로 검색

Retrieve-Read 두 단계 말고 정답을 바로 search 할 수는 없을까?

### One solution: Phrase Indexing

"Barack Obama (1961-present) was the 44th President of the United States."

"**Barack Obama ...**", "... (**1961**-presnet) ...", 등의 모든 phrase들을 enumeration하고 phrase에 대한 임베딩을 맵핑해준다. 이때, 임베딩은 key vector가 되며 query가 들어왔을 때  query vector와 key vector에 대해 Nearest Neighbor Search를 통해 가장 가까운 phrase를 찾게 된다.

또한, 새로운 질문이 들어오더라도 key vector의 업데이트 없이 가장 가까운 벡터로 연결시킬 수 있다.

### Query-Agnostic Decomposition

기존 MRC 방법 : $\hat{a} = \text{argmax}_a F_{\theta}(a,q,d)$

- $F_{\theta}$ : Model
- $a$ : phrase, $q$ : question, $d$ : document

Query-Agnostic Decomposition : $\hat{a} = \text{argmax}_a G_{\theta}(q) \cdot H_{\theta}(a,d)$

- $G_{\theta}$ : Question encoder
- $H_{\theta}$ : Phrase encoder

기존 방식에서 $F$는 $G$와 $H$의 곱으로 나타내어 질 수 있는데 이 가정이 애초에 틀린 것 일수도 있다. $F$는 복잡한 함수이고 이 함수가 $G$와 $H$로 decomposed된다고 보장할 수 없다.

따라서 Key Challenge는 다음과 같다.

- Key Challenge : 어떻게 각 phrase를 vector space 상에 잘 mapping 할 수 있을까?
- solution: Dense와 Sparse 임베딩을 둘 다 사용해보자

## Dense-sparse Representation for Phrase

### Dense vectors vs. Sparse vectors

- Dense vectors : 통사적, 의미적 정보를 담는 데 효과적
- Sparse vectors : 어휘적 정보를 담는 데 효과적

### Phrase and Question Embedding

Dense vector와 sparse vector를 모두 사용하여 phrase (and question) embedding

각각의 phrase embedding을 구할 때 Dense와 Sparse vector를 구하여 하나로 합쳐 inner product나 nearest search를 진행한다.

### Dense representation

Dense vector를 만드는 방법

- Pre-trained LM (e.g. BERT)를 이용
- Start vector와 end vector를 재사용해서 메모리 사용량을 줄임
    
    ![](https://drive.google.com/uc?export=view&id=1TpNKovvDBZbRFCPR6xzFOVumnBGGkRf6)
    

Question embedding

- Question을 임베딩할 때는 [CLS] 토큰 (BERT)을 활용
    
    ![](https://drive.google.com/uc?export=view&id=1SRmmNEyT39a58yVQKZAavbWGmb42lUB3)
    

### Sparse representation

Sparse vector를 만드는 방법

- 문맥화된 임베딩(contextualized embedding)을 활용하여 가장 관련성이 높은 n-gram으로 sparse vector를 구성한다-
    
    ![](https://drive.google.com/uc?export=view&id=1DXYxlr792-eAZxyAbPVnDLXEhMfYeAlf)
    
- 각 Phrase에 주변에 있는 단어들과 유사성을 측정해 유사성을 각 단어에 해당하는 sparse vector상의 dimension에 넣어준다
- TF-IDF와 비슷하지만 phrase마다 weight가 다르게 다이나믹하게 변하는 형태의 벡터로 만들어진다

### Scalability Challenge

Wikipedia는 60 billion개의 phrase가 존재한다. 따라서 Storage, indexing, search의 scalability가 고려되어야 한다.

- Storage : Pointer, filter, scalar quantization 활용
- Search : FAISS를 활용해 dense vector에 대한 search를 먼저 수행 후 sparse vector로 ranking

## Experiment Results & Analysis

### Experiment Results - SQuAD-open

DrQA(Retriever-reader)보다 +3.6% 성능, 68배 빠른 inference speeds

### Limitataion in Phrase Retrieval Approach

1. Large storage required
2. 최신 Retriever-reader 모델 대비 낮은 성능 → Decomposability gap이 하나의 원인

Decomposability gap :

- (기존) Question, passage, answer가 모두 함께 encoding
- (Phrase retrieval) Question과 passage/answer가 각각 encoding → Question과 passage사이 attention 정보 X