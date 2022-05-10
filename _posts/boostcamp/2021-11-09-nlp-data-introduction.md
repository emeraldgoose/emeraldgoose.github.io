---
title: 자연어 처리 데이터 소개
categories:
  - BoostCamp
tags: [dataset]
---
## 국내 언어 데이터의 구축 프로젝트

- 국가 주도 : 21 세기 세종 계획(국립 국어원) → 엑소브레인(ETRI) → 모두의 말뭉치(국립 국어원, AI 허브, NIA)
- 민간 주도 : KLUE(Upstage), KorQuAD(LG CNS), KorNLU(Kakaobrain)

## 21세기 세종 계획과 모두의 말뭉치

### 21세기 세종 계획

한국의 국어 정보화 중장기 발전 계획으로 총 2억 어절의 자료 구축, 공개 XML 형식, 언어정보나눔터 누리집을 통해 배포하다 중단 후 DVD로만 배포

세종 형태 분석 태그표는 MeCab, Khaii, 꼬마, 한나눔, 코모란 등에서 채택하고 있다.

### 모두의 말뭉치

인공지능의 한국어 처리 능력 향상에 필수적인 한국어 학습 자료 공개 플랫폼. 일상 대화, 메신저, 웹 문서 등 구어체 자료의 비중을 높임. 다층위 주석 말뭉치 포함(형태, 구문, 어휘 의미, 의미역, 개체명, 상호 참조 등)

JSON 형식, 모두의 말뭉치 누리집(https://corpus.korean.go.kr/)에서 배포

학습, 검증, 평가용 데이터가 나누어져 있지 않으므로 사용자가 직접 나누어 사용해야 함

## 엑소브레인

엑소브레인은 인간의 지적 노동을 보조할 수 있는 언어처리 분야의 AI 기술개발을 위해, 전문직 종사자(금융, 법률 또는 특허 등)의 조사, 분석 등의 지식노동을 보조 가능한 

- 언어 문법 분석을 넘어선 언어의 의미 추론 기술 개발
- 전문분야 원인, 절차, 상관관계 등 고차원 지식 학습 및 축적 기술 개발
- 전문분야 대상 인간과 기계의 연속적인 문답을 통한 심층질의응답 기술 개발 및 국내외 표준화

를 통해 핵심 IPR을 확보하는 인공지능 국가 R&D 프로젝트

ETRI의 오픈 AI API, DATA 서비스 누리집에서 데이터셋 배포

## AI허브

AI허브는 AI 기술 및 제품, 서비스 개발에 필요한 AI 인프라(AI 데이터, AI SW API, 컴퓨팅 자원)를 지원하는 누구나 활용하고 참여하는 AI 통합 플랫폼

데이터별로 데이터 설명서, 구축활용 가이드 제공

JSON, 엑셀 등 다양한 형식의 데이터 제공

실제 산업계 수요 조사를 반영하여 다양한 Task를 수행할 수 있는 자원 구축

## 민간 주도 데이터셋

### KLUE

한국어 이해 능력 평가를 위한 벤치마크

- 뉴스 헤드라인 분류
- 문장 유사도 비교
- 자연어 추론
- 개체면 인식
- 관계 추출
- 형태소 및 의존 구문 분석
- 기계 독해 이해
- 대화 상태 추적

### KorQuAD 1.0 & 2.0

한국어 기계 독해(Machine Reading Comprehension) 데이터셋. 스탠포드 대학교에서 공개한 SQuAD를 벤치마킹

### KorNLU

영어로 된 자연어 추론(NLI, Natural Language Inference) 및 문장 의미 유사도(STS, semantic textual similarity) 데이터셋을 기계 번역하여 공개

## Question Answering

### SQuAD

위키피디아 데이터를 기반으로 제작한 기계독해 및 질의응답 데이터

### SQuAD 1.0 - 데이터 구축

1. 구축 대상 기사 추출 : 위키피디아 상위 10,000 기사 중 500자 이하인 536 기사 무작위 추출
2. 크라우드 소싱을 통한 질의 응답 수집 : 각 문단마다 다섯 개의 질문과 답변 수집
3. 추가 응답 수집 : 평가를 통해서 각 질문 당 최소 두 개의 추가적인 답변 수집. 기사의 단락과 질문 노출 후 가장 짧은 대답 선택

### SQuAD 2.0 - 데이터 구축

1. 크라우드 소싱 플랫폼을 통한 대답하기 어려운 질문(unanswerable question) 수집
    - 각 문단마다 각 문단 만으로는 대답할 수 없는 다섯 개의 질문 생성
    - 적합한 질문을 25개 이하로 남김
2. 적합한 질문이 수집되지 않은 기사 삭제
3. 학습, 검증, 평가용 데이터 분할

## Machine Translation

### WMT 데이터셋

2014년부터 시행된 기계 번역 학회에서 공개한 데이터셋 다국어 번역 데이터이며, 두 언어간의 병렬 말뭉치로 구성됨. 뉴스, 바이오, 멀티 모달 데이터 등이 제공됨

### WMT 데이터셋 - 데이터 구축(2018년 기준)

평가용 데이터 : 1,500개의 영어 문장을 다른 언어로 번역 + 1,500개의 문장은 다른 언어에서 영어 문장으로 번역

훈련용 데이터 : 기존에 존재하는 병렬 말뭉치와 단일 언어 말뭉치를 제공 

## Text Summarization

### CNN/Daily Mail

추상 요약 말뭉치. 기사에 대하여 사람이 직접 작성한 요약문이 짝을 이루고 있음. 

## Dialogue

### DSTC - Dialog System Technology Challenges

- DSTC1 : human-computer dialogs in the bus timetable domain
- DSTC2 and DSTC3 : human-computer dialogs in the restaurant information domain
- DSTC4 and DSTC5 : DSTC4 human-human dialogs in the tourist information domain
- DSTC6 이후 : End-to-End Goal Oriented Dialog Learning, End-to-End Conversation Modeling, and Dialogue Breakdown Detection로 확장

### Wizard-of-Oz

WoZ방식으로 수집된 데이터셋이며 대화 상태 추적 데이터와 유사한 형태로 이루어짐

WoZ방식은 대화 수집 방식의 하나로, 참여자가 대화 시스템을 통해 대화를 하고 있다고 생각하게 한 뒤 실제로는 실제 사람이 참여자의 발화에 맞추어 응답을 제시하고 대화를 이끌어내면서 대화를 수집하는 방식

### UDC (Ubuntu Dialogue Corpus)

우분투 플랫폼 포럼의 대화를 수집한 데이터

특별한 레이블이 주석되어 있지 않음. 대화 상태 추적과 블로그 등에서 보이는 비구조적 상호작용의 특성을 모두 가지고 있음