---
title: Linking MRC and Retrieval
categories:
  - BoostCamp
tags: [MRC]
---
## Introduction to ODQA

### Linking MRC and Retrieval: Open-domain Question Answering (ODQA)

> MRC: 지문이 주어진 상황에서 질의응답
ODQA: 지문이 따로 주어지지 않음. 방대한 World Knowledge에 기반해서 질의응답  

- Modern search engines: 연관문서 뿐만 아니라 질문의 답을 같이 제공

### History of ODQA

Text retrieval conference (TREC) - QA Tracks (1999-2007): 연관문서만 반환하는 information retrieval (IR)에서 더 나아가서, short answer with support 형태가 목표  
Question processing + Passage retrieval + Answering processing  

1. Question processing
    1. Query formulation: 질문으로부터 키워드를 선택 / Answering type selection (ex. LOCATION: country)
2. Passage retrieval
    1. 기존의 IR 방법을 활용해서 연관된 document를 뽑고, passage 단위로 자른 후 선별
    2. Named entity / Passage 내 question 단어의 개수 등과 같은 hand-crafted features 활용
3. Answering processing
    1. Hand-crafted features와 heuristic을 활용한 classifier
    2. 주어진 question과 선별된 passage들 내에서 답을 선택

## Retriever-Reader Approach

### Retriever-Reader 접근방식

- Retriever : 데이터베이스에서 관련있는 문서를 검색(search)함
    - 입력 : 문서셋(document corpus), 질문(query)
    - 출력 : 관련성 높은 문서(document)
- Reader : 검색된 문서에서 질문에 해당하는 답을 찾아냄
    - 입력 : Retrieved된 문서(document), 질문(query)
    - 출력 : 답변(answer)

### 학습단계

- Retriever
    - TF-IDF, BM25 → 학습 없음
    - Dense → 학습 있음
- Reader
    - SQuAD와 같은 MRC 데이터셋으로 학습
    - 학습데이터를 추가하기 위해서 Distant supervision 활용

### Distant supervision

질문-답변만 있는 데이터셋 (CuratedTREC, WebQuestions, WikiMovies)에서 MRC 학습 데이터를 만들기 위해 Supporting document가 필요함

1. 위키피디아에서 Retriever를 이용해 관련성 높은 문서를 검색
2. 너무 짧거나 긴 문서, 질문의 고유명사를 포함하지 않는 등 부적합한 문서 제거
3. answer가 exact match로 들어있지 않은 문서 제거
4. 남은 문서 중에 질문과 연관성이 가장 높은 단락을 supporting evidence로 사용함

### Inference

- Retriever가 질문과 가장 관련성 높은 5개 문서 출력
- Reader는 5개 문서를 읽고 답변 예측
- Reader가 예측한 답변 중 가장 score가 높은 것을 최종 답으로 사용함

## Issue & Recent Approach

### Different granularities of text of indexing time

위키피디아에서 각 Passage의 단위를 문서, 단락, 또는 문장으로 정의할지 정해야 함.

- Article : 5.08 million
- Paragraph : 29.5 million
- Sentence : 75.9 million

Retriever 단계에서 몇개(top-k)의 문서를 넘길지 정해야 함. Granularity에 따라 k가 다를 수 밖에 없음  
(e.g. article → k=5, paragraph → k=29, sentence → k=78)

### Single-passage training vs. Multi-passage training

- Single Passage : 현재 우리는 k개의 passages들을 reader가 각각 확인하고 특정 answer span에 대한 예측점수를 나타냄. 그리고 이 중 가장 높은 점수를 가진 answer span을 고르도록 함
    - 이 경우 각 retrieved passages들에 대한 직접적인 비교라고 볼 수 없음
    - 따로 reader 모델이 보는게 아니라 전체를 한 번에 보면 어떨까?
- Multi Passage : retrieved passages 전체를 하나의 passage로 취급하고 reader 모델이 그 안에서 answer span하나를 찾도록 함
    - 문서가 너무 길어지므로 GPU에 더 많은 메모리를 할당해야 함
    - 처리해야하는 연산량이 많아짐

### Importance of each passage

Retriever 모델에서 추출된 top-k passage들의 retrieved score를 reader모델에 전달  
최근에는 passage의 retrieved score와 reader가 각 passages들을 읽고 내놓은 answer의 score를 합하여 최종 출력을 내고 있다.