---
title: Generation-based MRC
categories:
  - BoostCamp
tags: [MRC]
---
## Generation-based MRC

### Generation-based MRC 문제 정의

MRC 문제를 푸는 방법

1. Extraction-based mrc : 지문(context) 내 답의 위치를 예측 → 분류문제(classification)
2. Generation-based mrc : 주어진 지문과 질의(question)을 보고, 답변을 생성 → 생성문제(generation)
    - generation-based mrc는 지문에 답이 없어도 답을 생성하는 방법이다.
    - 모든 extraction-based mrc문제들은 generation-based mrc 문제로 치환할 수 있다.
    - 반대로 generation-based mrc문제들을 extraction-based mrc문제로 치환할 수 없다.
    

### Generation-based MRC 평가방법

동일한 extractive answer datasets ⇒ Extraction-based MRC와 동일한 평가 방법을 사용 (recap)

1. Exact Match (EM) Score
2. F1 Score

### Generation-based MRC & Extraction-based MRC 비교

1. MRC 모델 구조
    1. Seq-to-seq PLM 구조 (generation) vs. PLM + Classifier 구조 (extraction)
2. Loss 계산을 위한 답의 형태 / Prediction의 형태
    1. Free-form text 형태 (generation) vs. 지문 내 답의 위치 (extraction)
    2. Extraction-based MRC : F1 계산을 위해 text로의 별도 변환 과정이 필요
    

## Pre-processing

### 입력표현 - 토큰화

Tokenization(토큰화) : 텍스트를 의미를 가진 작은 단위로 나눈 것 (형태소)

- Extraction-based MRC와 같이 WordPiece Tokenizer를 사용함
    - WordPiece Tokenizer 사전학습 단계에서 먼저 학습에 사용한 전체 데이터 집합(코퍼스)에 대해서 구축되어 있어야 함
    - 구축 과정에서 미리 각 단어 토큰들에 대해 순서대로 번호(인덱스)를 부여해둠
- Tokenizer은 입력 텍스트를 토큰화한 뒤, 각 토큰을 미리 만들어둔 단어 사전에 따라 인덱스로 변환함

WordPiece Tokenizer 사용 예시

- 질문 : "미국 군대 내 두 번째로 높은 직위는 무엇인가?"
- 토큰화된 질문 : ['미국', '군대', '내', '두 번째', '##로', '높은', '직', '##위는', '무엇인가', '?']
- 인덱스로 바뀐 질문 : [101, 23545, 8910, 14423, 8996, 9102, 48506, 11261, 55600, 9707, 19855, 11018, 9294, 119137, 12030, 11287, 136, 102]
- 인덱스로 바뀐 질문을 보통 input_ids(또는 input_token_ids)로 부름
- 모델의 기본 입력은 input_ids만 필요하나, 그 외 추가적인 정보가 필요함

### 입력표현 - Special Token

학습 시에만 사용되며 단어 자체의 의미는 가지지 않는 특별한 토큰

SOS(Start Of Sentence), EOS(End Of Sentence), CLS, SEP, PAD, UNK, ... 등등

- Extraction-based MRC 에선 CLS, SEP, PAD 토큰을 사용
- Generation-based MRC에서도 PAD 토큰은 사용됨. CLS, SEP 토큰 또한 사용할 수 있으나, 대신 자연어를 이용하여 정해진 텍스트 포맷(format)으로 데이터를 생성
    - question: question_text context: context_text

### 입력표현 - additional Information

Attention mask

- Extraction-based MRC와 똑같이 어텐션 연산을 수행할 지 결정하는 어텐션 마스크 존재

Token type ids

- BERT와 달ㄹ리 BART에서는 입력시퀀스에 대한 구분이 없어 token_type_ids가 존재하지 않음
- 따라서 Extraction-based MRC와 달리 입력에 token_type_ids가 들어가지 않음

### 출력표현 - 정답 출력

Sequence of token ids

- Extraction-based MRC에선 텍스트를 생성해내는 대신 시작/끝 토큰의 위치를 출력하는 것이 모델의 최종 목표였음
- Generation-based MRC는 그보다 조금 더 어려운 실제 텍스트를 생성하는 과제를 수행
- 전체 시퀀스의 각 위치 마다 모델이 아는 모든 단어들 중 하나의 단어를 맞추는 classification 문제

## Model

### BART

> 기계 독해, 기계 번역, 요약, 대화 등 sequence to sequence 문제의 pre-training을 위한 denoising autoencoder

### BART Encoder & Decoder

- BART의 인코더는 BERT처럼 bi-directional
- BART의 디코더는 GPT처럼 uni-directional(autoregressive)

### Pre-training BART

BART는 텍스트에 노이즈를 주고 원래 텍스트를 복구하는 문제를 푸는 것으로 pre-training함

## Post-processing

### Searching

1. Greedy Search
    - 현재 단어에서 다음 단어 중 가장 확률이 높은 단어가 오는 방법이다.
    - 빠르지만 현재 선택한 단어가 좋지 않은 선택인 경우 아웃풋이 좋지 않게 된다.
2. Exhaustive Search
    - 모든 가능성을 보는 방법이다.
    - 가능한 가짓수가 타임스텝에 비례해서 exponential하게 늘어나기 때문에 문장의 길이가 커지거나 vocabulary의 사이즈가 커져도 사실상 불가능한 방법이다.
3. Beam Search
    - 각 타임스텝마다 가장 확률이 높은 Top-K만을 선택하는 방법이다. 나머지는 모두 버린다.
    - 일반적으로 선택하는 방법