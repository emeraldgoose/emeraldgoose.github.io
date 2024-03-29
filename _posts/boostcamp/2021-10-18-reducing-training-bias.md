---
title: Reducing Training Bias
categories:
  - BoostCamp
tags: [MRC]
---
## Definition of Bias

### Bias의 종류

Bias in learning

- 학습할 때 과적합을 막거나 사전 지식을 주입하기 위해 특정 형태의 함수를 선호하는 것 (inductive bias)

A Biased World

- 현실 세계가 편향되어 있기 때문에 모델에 원치 않는 속성이 학습되는 것 (historical bias)
- 성별과 직업 간 관계 등 표면적인 상관관계 때문에 원치 않는 속성이 학습되는 것 (co-occurrence bias)

Bias in Data Generator

- 입력과 출력을 정의한 방식 때문에 생기는 편향 (specification bias)
- 데이터를 샘플링한 방식 때문에 생기는 편향 (sampling bias)
- 어노테이터의 특성 때문에 생기는 편향 (annotator bias)

### Gender Bias

- 대표적인 bias 예시
- 특정 성별과 행동을 연관시켜서 예측 오류가 발생

### Sampling Bias

[리터러시 다이제스트] 여론조사  

- 포본크기 : 240만명
- 예측 : 루즈벨트 43%, 알프레드 랜던 57% → 실제 : 루즈벨트 62%, 알프레드 랜던 38%
- 설문대상 : 잡지 정기구독자, 자동차 등록명부, 사교클럽 명부 등 → 중산층 이상으로 표본 왜곡


## Bias in Open-domain Question Answering
### Training bias in reader model
만약 reader 모델이 한정된 데이터셋에만 학습이 된다면 reader는 항상 정답이 문서 내에 포함된 데이터쌍만(Positive)을 보게됨  
특히 SQuAD와 같은 (Context, Query, Answer)가 모두 포함된 데이터는 positive가 완전히 고정됨  
 
- Inference 시 만약 데이터 내에서 찾아볼 수 없었던 새로운 문서를 준다면?
- Reader 모델은 문서에 대한 독해 능력이 매우 떨어질 것이고, 결과적으로 정답을 내지 못할 것임

### How to mitigate training bias?

1. Train negative examples
    1. 훈련할 때 잘못된 예시를 보여줘야 retriever이 negative한 내용들을 먼 곳에 배치할 수 있음 → Negative sample도 완전히 다른 negative와 비슷한 negative에 대한 차이 고려가 필요함
2. Add no answer bias
    1. 입력 시퀸스의 길이가 N일시, 시퀸스의 길이 외 1개의 토큰이 더 있다고 생각하기 → 훈련 모델의 마지막 레이어 wight에 훈련 가능한 bias를 하나 더 추가
    2. softmax로 answer prediction을 최종적으로 수행할 때, start end 확률이 해당 bias 위치에 있는 경우가 가장 확률이 높으면 이는 "대답할 수 없다"라고 취급

어떻게 좋은 Negative sample을 만들 수 있을까?  

1. Corpus내에서 랜덤하게 뽑기
2. 좀 더 헷갈리는 negative 샘플들 뽑기
    1. 높은 BM25 / TF-IDF 매칭 스코어를 가지지만, 답을 포함하지 않는 샘플
    2. 같은 문서에서 나온 다른 Passage/Question 선택하기
    

## Annotation Bias from Datasets

### What is annotation bias?
ODQA 학습 시 기존의 MRC 데이터셋 활용 → ODQA 세팅에는 적합하지 않은 bias가 데이터 제작(annotation) 단계에서 발생할 수 있음  
SQuAD 같은 경우, 질문의 답을 아는 사람이 질문과 답을 작성해서 질문의 단어들이 context와 어느정도 겹치고 context에 의존적이다. 그리고 사용하는 문단이 사람들이 자주 보는 문단 500개 정도이기 때문에 어느정도 bias가 존재한다. 그래서 다른 데이터셋과 달리 SQuAD는 BM25의 점수가 DPR보다 점수가 높게 나온다. DPR에 BM25를 추가해서 사용하면 

### Dealing with annotation bias
Annotation 단계에서 발생할 수 있는 bias를 인지하고, 이를 고려하여 데이터를 모아야 함  
ex) ODQA 세팅과 유사한 데이터 수집 방법  
→ Natural Questions: Supporting  evidence가 주어지지 않은, 실제 유저의 question들을 모아서 dataset을 구성

### Another bias in MRC dataset
SQuAD: Passage가 주어지고, 주어진 passage 내에서 질문과 답을 생성 → ODQA에 applicable하지 않는 질문들이 존재