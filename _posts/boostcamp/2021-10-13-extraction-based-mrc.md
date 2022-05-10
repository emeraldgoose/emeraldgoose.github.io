---
title: Extraction-based MRC
categories:
  - BoostCamp
tags: [MRC]
---

## Extraction-based MRC

### Extraction-based MRC 문제 정의

> 질문(question)의 답변(answer)이 항상 주어진 지문(context)내에 span으로 존재.

### Extraction-based MRC 평가 방법

- Exact Match (EM) Score
    - 예측값과 정답이 캐릭터 단위로 완전히 똑같은 경우에만 1점 부여. 하나라도 다르면 0점
- F1 Score
    - 예측값과 정답의 overlap을 비율로 계산. 0점과 1점 사이의 부분점수를 받을 수 있음

## Pre-processing

### Tokenization

> 텍스트를 작은 단위(Token)으로 나누는 것  

- 띄어쓰기 기준, 형태소, subword 등 여러 단위 토큰 기준이 사용됨
- 최근에 Out-of-Vocabulary (OOV) 문제를 해결해주고 정보학적으로 이점을 가진 Byte Pair Encoding(BPE)을 주로 사용함

### Special Tokens

[CLS], [SEP], ...

### Attention Mask

> 입력 시퀸스 중에서 attention을 연산할 때 무시할 토큰을 표시한다. 보통 [PAD]와 같은 의미가 없는 특수토큰을 무시하기 위해 사용

### Token Type IDs

> 입력이 2개이상의 시퀸스일 때, 각각에게 ID를 부여하여 모델이 구분해서 해석하도록 유도한다.

### 모델 출력값

- 정답은 문서내 존재하는 연속된 단어토큰 (span)이므로, span의 시작과 끝 위치를 알면 정답을 맞출 수 있음
- Extraction-based에선 답안을 생성하기 보다, 시작위치와 끝위치를 예측하도록 학습함. 즉, Token Classification문제로 치환한다.

## Fine-tuning

- 지문내에서 정답에 해당되는 임베딩을 linear transformation을 통해서 각 단어마다 하나의 숫자로 나올 수 있도록 바꿔주고 bert 지문 토큰마다 하나의 숫자가 아웃풋으로 나오게 되고 그 숫자는 정수가 된다. (높을 수록 정답일 확률 증가).
- 시작 포지션과 엔드 포지션을 위와 같은 방법으로 찾는다.
- 이 수치를 softmax 적용후 negative log likelihood로 학습한다.
- ???

## Post-processing

### 불가능한 답 제거하기

다음과 같은 경우 candidate list에서 제거  

- End position이 start position보다 앞에 있는 경우 (ex. start=90, end=80)
- 예측한 위치가 context를 벗어난 경우(question 위치쪽에 답이 나온 경우)
- 미리 설정한 max_answer_length보다 길이가 더 긴 경우

### 최적의 답안 찾기

1. Start/End position prediction에서 score(logits)가 가장 높은 N개를 각각 찾는다.
2. 불가능한 start/end 조합을 제거한다.
3. 가능한 조합들을 score의 합이 큰 순서대로 정렬한다.
4. Score가 가장 큰 조합을 최종 예측으로 선정한다.
5. Top-k 가 필요한 경우 차례대로 내보낸다.