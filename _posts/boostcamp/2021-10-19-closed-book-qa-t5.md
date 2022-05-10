---
title: Closed-book QA with T5
categories:
  - BoostCamp
tags: [MRC]
---
## Closed-book Question Answering

> 모델이 이미 사전학습으로 대량의 지식을 학습했다면, 사전학습 언어모델 자체가 이미 하나의 knowledge storage라고 볼 수 있지 않을까? → 굳이 다른 곳에서 지식을 가져와야할 필요가 있을까?


### Zero-shot QA performance of GPT-2

사전학습 시 전혀 본적없는 Natural Question 데이터셋에도 어느정도 대답이 가능함

### Open-book QA vs. Closed-book QA

- 지식을 찾는 방법
    - Open-book QA : 대량의 지식소스를 특정 문서 단위로 나누어 Dense/Sparse 형태로 표현한 후, qeury가 들어오면 가장 그와 관련된 문서를 search
    - Closed-book QA : 대량의 지식소스를 기반으로 사전학습된 언어모델이 그 지식을 기억하고 있을 것이라 가정하여 Search 과정없이 바로 정답을 생성함
- 문제점
    - Open-book QA : 지식 소스를 저장하기 어려움, 검색하는 시간 소요
    - Closed-book QA : 사전학습된 언어 모델이 얼마나 지식을 잘 기억하고 있는지가 매우 중요함

## Text-to Text Format

### Closed-book QA as Text-to-Text Format

Closed-book QA에 사용되는 방법은 Generation-based MRC와 유사함

- 단, 입력에 지문(Context)가 없이 질문만 들어간다는 것이 차이점
- 사전학습된 언어 모델은 BART와 같은 seq-to-seq 형태의 Transformer 모델을 사용함
- Text-to-Text format에서는 각 입력값(질문)과 출력값(답변)에 대한 설명을 맨 앞에 추가함

### Text-to-Text Format

1. Text-to-text problem : input으로 text를 받아서, output으로 새로운 text를 생성하는 문제
    
    다양한 text processing problem을 Text-to-text 문제로 변형
    
2. Text classification (MNLI)
    
    MNLI: 두 개의 sentence (premise, hypothesis)가 주어지고, 이 둘의 관계를 예측하는 task (neutral, contradiction, entailment)
    

### T5

Text-to-Text format 이라는 형태로 데이터의 입출력을 만들어 거의 모든 자연어 처리 문제를 해결하도록 학습된 seq-to-seq 형태의 Transformer 모델

### Pre-training T5

다양한 모델 구조, 사전학습 목표, 사전학습용 데이터, Fine-tuning 방법 등을 체계적으로 실험할 가장 성능이 좋은 방식을 선택하여 방대한 규모의 모델을 학습시킴

### Using T5 for Closed-book QA

Fine-tuning T5:

- Fine-tuning: MRC 데이터셋 (TriviaQA, WebQuestions, Natural Question)의 QA pair를 활용
- MRC 데이터셋에 제공되는 supporting document는 무시
- input: Task-specific prefix cnrk → "trivia question: [question]"
- Natural Question와 같이 답이 여러개인 경우 target → "answer: [answer1] answer: [answer2]"

### Experiment Results & Analysis

### Experiment Setting

- Dataset : Open-domain QA 데이터셋 또는 MRC 데이터셋에서 지문을 제거하고 질문과 답변만 남긴 데이터셋을 활용
- Salient Span Masking : 고유명사, 날짜 등 의미를 갖는 단위에 속하는 토큰 범위를 마스킹한 뒤 학습, Pre-trained 체크포인트에서 추가로 pre-training 함
- Fine-tuning : Pre-trained T5 체크포인트를 Open-domain QA 학습 데이터셋으로 추가 학습

### Quanitative Examples
대부분의 Open-book 스타일 모델 (문서 검색 후 기계 독해) 보다 뛰어난 성능을 보여줌  
모델의 크기가 커질수록 성능이 증가함  
Salient Span Masking이 성능을 크게 끌어올림  

### False negatives

Exact match 기준으로 오답으로 채점된 결과를 사람이 평가한 결과 오답이 아닌 경우

1. Phrasing Mismatch : 정답에 대한 표현이 다른 경우
2. Incomplete Annotation : 정답이 여러 개일 수 있으나 하나만 정답으로 처리되는 경우
3. Unanswerable : 질문을 한 시간이나 문맥에 따라서 정답이 달라지는 경우  

### Limitations

Closed-book QA의 한계점 및 앞으로의 개선 방향

1. 모델의 크기가 커서 계산량이 많고 속도가 느림 → 더 효율적인 모델 필요
2. 모델이 어떤 데이터로 답을 내는지 알 수 없음 → 결과의 해석 가능성(interpretability)을 높이는 연구 필요
3. 모델이 참조하는 지식을 추가하거나 제거하기 어려움