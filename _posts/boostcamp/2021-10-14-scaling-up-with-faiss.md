---
title: Scaling Up with FAISS
categories:
  - BoostCamp
tags: [MRC]
---
## Passage Retrieval and Similarity Search

> How to find the passage in real time? ⇒ Similarity Search
> 

### MIPS (Maximum Inner Product Search)

주어진 질문(query) 벡터 q에 대해 Passage 벡터 v들 중 가장 질문과 관련된 벡터를 찾아야함  
여기서 관련성은 내적(inner product)이 가장 큰 것

### MIPS & Challenge

실제로 검색해야 할 데이터는 훨씬 방대함 → 더 이상 모든 문서 임베딩을 일일히 보면서 검색할 수 없음

### Tradeoffs of Similarity search

1. Search speed
    1. 쿼리 당 유사한 벡터를 k개 찾는데 얼마나 걸리는지? → 가지고 있는 벡터량이 클 수록 더 오래 걸림
    2. 해결방법 : Pruning
2. Memory Usage
    1. 벡터를 사용할 때, 어디에서 가져올 것인지? → 모든 임베딩을 RAM에 올릴 수 없다
    2. 해결방법 : Compression
3. Accuracy
    1. brute-force 검색결과와 얼마나 비슷한지? → 속도 증가시 정확도 하락을 피할 수 없음
    2. 해결방법 : Exhaustive search

- 속도(search time)와 재현율(recall)의 관계 : 더 정확한 검색을 하려면 더 오랜 시간이 소모됨

### Increasing search speed by bigger corpus

코퍼스 크기가 커질수록 → 탐색공간이 커지고, 저장해 둘 메모리 공간 또한 커짐. 그리고 sparse embedding의 경우 이러한 문제가 훨씬 커짐

## Approximating Similarity Search

### Compression - Scalar Quantization (SQ)

> Compression: vector를 압축하여, 하나의 vector가 적은 용량을 차지 → 압축량 증가로 메모리 공간 감소하지만 정보손실이 증가한다
> 

### Pruning - Inverted File (IVF)

> Pruning: Search space를 줄여 search 속도 개선 (dataset의 subset만 방문) → Clustering + Inverted file을 활용한 search

- Clustering의 경우 가장 자주 사용하는 방법은 k-means clustering이다.
- 원하는 Cluster만 방문하여 Exhaustive Search 수행
- Inverted File (IVF) : vector의 index = inverted list structure → (각 cluster의 centroid id)와 (해당 cluster의 vector들)이 연결되어 있는 형태

## Introduction to FAISS

## FAISS

> Faiss is a library for efficient similarity search and clustering of dense vectors.

- 인덱싱을 도와주고 인코딩에는 관련없다

### Passage Retrieval with FAISS

1. Train index and map vectors
    1. Scaling을 하기위해 min, max index value를 알아야 하고 무작정 랜덤하게 고를 수는 없기 때문에 이를 학습해야 하는 과정이 필요하다.
    2. train 단계와 add 단계가 있는데 train을 통해 cluster와 SQ8를 정의한 뒤에 add 단계를 통해 cluster에 실제 벡터를 SQ8형태로 투입한다.
    3. 전부를 학습하기에는 비효율적일때 일부를 샘플링해서 학습한다.
2. Search based on FAISS index
    1. *nprobe : 몇 개의 가장 가까운 cluster를 방문하여 search 할 것인지 
    2. query 벡터가 들어오면 nprobe개의 cluster를 SQ8 베이스로 search하고 top-k개를 Search Result로 내보낸다.

## Scaling up with FAISS

### FAISS basics

```python
d = 64 # 벡터의 차원
nb = 100000 # 데이터베이스 크기
nq = 10000 # 쿼리 개수
xb = np.random.random((nb,d)).astype('float32') # 데이터베이스 예시
xq = np.random.random((nq,d)).astype('float32') # 쿼리 예시

index = faiss.IndexFlatL2(d) # 인덱스 빌드
index.add(xb) # 인덱스에 벡터 추가

k = 4 # 가장 가까운 벡터 4개를 찾기
D, I = index.search(xq, k) # 검색하기
# D : 쿼리와의 거리
# I : 검색된 벡터의 인덱스
```

- 이 예제에서는 scaling up quantization을 활용하지 않기 때문에 학습 과정이 필요없다.

### IVF with FAISS

- IVF 인덱스를 만들고 클러스터링을 통해 가까운 클러스터 내 벡터들만 비교함
- 빠른 검색이 가능하며 클러스터 내에서는 여전히 전체 벡터와 거리 비교(Flat)

```python
nlist = 10 # 클러스터 개수
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist) # Inverted File 만들기
index.train(xb) # 클러스터 학습

index.add(xb) # 클러스터에 벡터 투입
D, I = index.search(xq, k) # k개의 벡터 검색
```

### Using GPU with FAISS

- GPU의 빠른 연산 속도를 활용하여 거리 계산을 위한 행렬곱 등에서 유리
- GPU 메모리 제햔이나 메모리 random access 시간이 느린 것 등이 단점

```python
rss = faiss.StandardGpuResources() # 단일 GPU 사용
	
index_flat = faiss.IndexFlatL2(d) # 인덱스 (CPU) 빌드하기

gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat) # GPU 인덱스로 옮기기

gpu_index_flat.add(xb)
D, I = gpu_index_flat.search(xq, k) # 검색하기
```

### Using Multiple GPUs with FAISS

```python
cpu_index = faiss.IndexFlatL2(d)

gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

gpu_index.add(xb)
D, I = gpu_index.search(xq, k)
```