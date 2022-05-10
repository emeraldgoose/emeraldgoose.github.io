---
title: Introduction to MRC
categories:
  - BoostCamp
tags: [MRC]
---

## Introduction to MRC

### Machine Reading Comprehension(MRC)의 개념

> 기계독해(Machine Reading Comprehension)
: 주어진 지문(Context)를 이해하고, 주어진 질의(Query/Question)의 답변을 추론하는 문제

### MRC의 종류

1. Extraction Answer Datasets
: 질의(question)에 대한 답이 항상 주어진 지문(context)의 segment (or span)으로 존재
    - SQuAD, KorQuAD, NewsQQ, Natural Questions, etc
2. Descriptive/Narrative Answer Datasets
: 답이 지문 내에서 추출한 span이 아니라, 질의를 보고 생성 된 sentence (or free-form)의 형태
    - MS MARCO, Narrative QA
3. Multiple-choice Datasets
: 질의에 대한 답을 여러 개의 answer candidates 중 하나로 고르는 형태
    - MCTest, RACE, ARC, tec

### Chanllenges in MRC

- 단어들의 구성이 유사하지는 않지만 동일한 문장을 이해 (DuoRC (paraphrased paragraph) / QuoRef (coreference resolution))
- Unanswerable questions → 'No Answer' (SQuAD 2.0)
- Multi-hop reasoning
: 여러 개의 document에서 질의에 대한 supporting fact를 찾아야지만 답을 찾을 수 있음 (HotpotQA, QAnagroo)

### MRC의 평가방법

1. Exact Match / F1 Score : For **extractive** answer and **multiple-choice** answer datasets
    - Exact Match (EM) or Accuracy : 예측한 답과 ground-truth가 정확히 일치하는 샘플의 비율
    - F1 Score : 예측한 답과 ground-truth 사이의 token overlap을 F1으로 계산
2. ROUGE-L / BLEU : For descriptive answer datasets → Ground-truth과 예측한 답 사이의 overlap을 계산
    - ROUGE-L Score : 예측한 값과 ground-truth 사이의 overlap recall (ROUGE-L ⇒ LCS(Longest common subsequence))
    - BLEU : 예측한 답과 ground-truth 사이의 precision (BLEU-n ⇒ uniform n-gram weight)
    

## Unicode & Tokenization

### Unicode란?

> 전 세계의 모든 문자를 일관되게 표현하고 다룰 수 있도록 만들어진 문자셋. 각 문자마다 숫자 하나에 매핑한다. 

### 인코딩 & UTF-8

> 인코딩이란, 문자를 컴퓨터에서 저장 및 처리할 수 있게 이진수로 바꾸는 것을 말한다.

> UTF-8 (Unicode Transformation Format)는 현재 가장 많이 쓰이는 인코딩 방식이며 문자 타입에 따라다른 길이의 바이트를 할당한다.

- 1 byte : Standard ASCII
- 2 bytes : Arabic, Hebrew, most European scripts
- 3 bytes : BMP(Basic Multilingual Plane) - 대부분의 현대 글자 (한글 포함)
- 4 bytes : All Unicode characters - 이모지 등

### Python에서 Unicode 다루기

- Python3 부터 string 타입은 유니코드 표준을 사용한다.
- `ord()` : 문자를 유니코드 code print로 변환한다.
- `chr()` : Code print를 문자로 변환한다.

```python
ord('A') # 65
hex(ord('A')) # '0x41'
chr(65) # 'A'
chr(0x41) # 'A'

ord('가') # 44032
hex(ord('가')) # '0xac00'
chr(44032) # '가'
chr(0xac00) # '가'
```

### Unicode와 한국어

한국어는 한자 다음으로 유니코드에서 많은 부분을 차지하고 있는 문자이다.

- 완성형 : 현대 한국어의 자모 조합으로 나타낼 수 있는 모든 완성형 한글 11,172자
    - `len('가') # 1`
- 조합형 : 조합하여 글자를 만들 수 있는 초성, 중성, 종성
    - `len('가') # 2`

### 토크나이징

텍스트를 토큰 단위로 나누는 것. 단어(띄어쓰기 기준), 형태소, subword 등 여러 토큰 기준이 사용된다.

- Subword 토크나이징
    - 자주 쓰이는 글자 조합은 한 단위로 취급하고, 자주 쓰이지 않는 조합은 subword로 쪼갠다.
    - '아버지 가방에 들어가신다' = '아버지', '가', '##방', '##에', '들어', '##가', '##신', '##다'
- BPE (Byte-Pair Encoding)
    - 데이터 압축용으로 제안된 알고리즘. NLP에서 토크나이징용으로 활발하게 사용되고 있다.
    - 방법
        1. 가장 자주 나오는 글자 단위 Bigram(or Byte Pair)를 다른 글자로 치환한다.
        2. 치환된 글자를 저장해둔다.
        3. 1~2번을 반복한다.

## Looking into the Dataset

### KorQuAD

> LG CNS가 AI 언어지능 연구를 위해 공개한 질의응답/기계독해 한국어 데이터셋  

- SQuAD v1.0의 데이터 수집 방식을 벤치마크하여 표준성을 확보함
- HuggingFace datasets 라이브러리에서 `squad_kor_v1`, `squad_kor_v2`로 불러올 수 있음

### KorQuAD 예시

'Title' : 알렉산더_헤이그

'Context': 알렉산더 메이그스 헤이그 2세(영어:AlexanderMeigsHaig,Jr.,1924년 12월 2일 ~2010년 2월 20일)는 미국의 국무장관을 지낸 미국의 군인,관료 및 정치인이다.로널드 레이건 대통령 밑에서 국무장관을 지냈으며,리처드 닉슨과 제럴드 포드 대통령 밑에서 백악관 비서실장을 지냈다.또한 그는 미국 군대에서 2번째로 높은 직위인 미국 육군 부참모 총장과 나토 및 미국 군대의 유럽연합군 최고사령관이었다. 한국 전쟁 시절 더글러스 맥아더 유엔군 사령관의 참모로 직접 참전하였으며,로널드 레이건 정부 출범당시 초대 국무장관직을 맡아 1980년대 대한민국과 미국의 관계를 조율해 왔다.저서로 회고록 《경고:현실주의, 레이건과 외교 정책》(1984년 발간)이 있다.

'Question-Answer Pairs': [
{'question':'미국 군대 내 두번째로 높은 직위는 무엇인가?’, 
'answers':[{'answer_start':204,'text':'미국 육군 부참모 총장'}],'id':'6521755-0-0',} 
{'question':'알렉산더 헤이그가 로널드 레이건 대통령 밑에서 맡은 직책은 무엇이었나?, 
''answers':[{'answer_start':345,'text':'국무장관'}], 'id':'6539187-0-1',}
]