---
title: 관계 추출 과제의 이해
categories:
  - BoostCamp
tags: [dataset]
---
## 관계 추출 관련 과제의 개요

### 개체명(Entity) 인식(NER, Named Entity Recognition)

개체명이란 인명, 지명, 기관명 등과 같은 고유명사나 명사구를 의미한다.

개체명 인식 태스크는 문장을 분석 대상으로 삼아서 문장에 출현한 개체명의 경계를 인식하고, 각 개체명에 해당하는 태그를 주석함.

가장 널리 알려진 챌린지는 MUC-7, CoNLL 2003이 있다. 한국에서는 TTA의 개체명 태그 세트 및 태깅 말뭉치 문서가 표준이다.

### 관계(Relation) 추출(RE, Relation Extract)

관계 추출은 문장에서 나타난 개체명 쌍(Entity Pair)의 관계(Relation)을 판별하는 태스크이다.

개체명 쌍은 관계의 주체(Subject)와 대상(Object)로 구성된다.

관계 추출 태크 표는 대표적으로 TAC KBP 2016, TAC RED가 있다.

### 개체명 연결(EL, Entity Linking)

개체명을 인식(Named Entity Recognition)하고 모호성을 해소(Named Entity Disambiguation)하는 과제를 결합한 것.

텍스트에서 추출된 개체명을 지식 베이스(knowledge base)와 연결하여 모호성을 해소함.

AIDA CoNLL-YAGO Dataset 또는 TAC KBP English Entity Linking Comprehensive and Evaluation Data 2010등이 있음

![](https://lh3.google.com/u/0/d/1kX_2NhKoYfFyNizJ36WVbYDgR-RIcLTs)

- 출처: Robust Disambihuation of Named Entities in Text(2011)

### 이러한 데이터를 만드는 이유?

NER, RE, EL은 기본적으로 비구조화된 텍스트에서 정보를 추출하여 구조화하려는 것이 목적입니다.

따라서 이 과정에서 지식 베이스가 활용되기도 하고, 이 결과물이 지식 베이스가 되기도 합니다.

정보처리의 관점에서 구조화된 정보의 활용도가 높기 때문에 이러한 시도는 앞으로도 계속 될 것입니다.

## 과제별 차이점

- 개체를 보는 관점
- 대상 개체의 분류 레이블(태크 체계) 차이
- 관계에 대한 주석 여부, 참조 자원

## 데이터 제작시 문제점

### KLUE 데이터 구축시 문제점 : NER

- 2개 이상의 태그로 주석될 수 있는 개체명 → 맥락에 기반한 주석
    - **서울시**는 정책을 발표했다. → Organization
    - 그 카페는 **서울시** 서대문구 연희동에 있다. → Location
- 주석 대상의 범주 → 구체적 범주 및 기준 명시
    - A급, B급, C급, 삼류(3류)

### KLUE 데이터 구축시 문제점 : RE

- 한국어 데이터 현실에 맞지 않는 주석 → 태그 통폐합 및 추가
    - 지역 관련 태그 통합, 사람, 기관의 작품 및 생산물 관련 태그 추가
- KB(Knowledge base)의 활용 → 일부만 활용

### 데이터 구축시 문제점 : EL

- 적합한 KB(Knowledge base) 선정의 문제
    - 현재 AI hub에 공개된 KB의 경우 제한적인 저작권 아래서 활용이 가능함
    - 위키 데이터를 활용하여 자체적인 지식베이스를 구출하여 활용하거나, 서비스 도메인에 맞는 지식베이스를 구축하여 활용할 수 있음
    - 지식베이스를 구축하는 것 자체가 많은 비용과 자원이 드는 일이므로 이에 대한 대비가 필요함