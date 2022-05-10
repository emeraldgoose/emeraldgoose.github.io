---
title: Intro to Natural Language Processing(NLP) 정리
categories:
  - BoostCamp
tags: [nlp]
---
- 자연어 처리는 기본적으로 컴퓨터가 주어진 단어나 문장, 보다 더 긴 문단이나 글을 이해하는 Natural Language Understanding(NLU)가 있고 이런 자연어를 상황에 따라 적절히 생성할 수 있는 Natural Language Generation(NLG) 두 종류의 task로 구성됩니다.

- Natural language processing (major conferences: ACL, EMNLP, NAACL)
    - Low-level parsing
        - Tokenization : 문장을 단어(Token) 단위로 나누는 과정
        - stemming : study, studying과 같이 어미가 달라지면서 단어 의미가 변하는데 그 과정에서 어근은 고정되어 있도록 의미를 추출하는 과정
    - Word and phrase level
        - Named entity recognition(NER) : Newyork Times와 같이 단어를 나눠서 해석하지 않는 단일단어, 고유명사를 인식하는 과정
        - part-of-speech(POS) tagging : word들이 문장내에서 어떤 역할을 하는지 알아내는 과정(주어, 본동사, ...)
        - noun-phrase chunking
        - dependency parsing
        - conference resolution
    - Sentence level
        - Sentiment analysis : 문장의 느낌(긍정 혹은 부정)
        - machine translation : 주어진 문장을 이해하고 다른 언어로 번역하는 과정
    - Multi-sentence and paragraph level
        - Entailment prediction : 두 문장간의 논리적인 내포, 모순을 예측
        - question answering : 독해기반의 문장을 이해하고 답을 하는 과정
        - dialog systems : 챗봇
        - summarization : 주어진 문서를 요약하는 과정
- Text mining(major conference: KDD, The WebConf (formely, WWW), WSDM, CIKM, ICWSM)
    - 특정 키워드의 빈도수를 파악해서 트렌드를 분석하거나 상품을 출시하여 소비자 반응을 얻어내는데 활용될 수 있다.
    - Document clustering (topic modeling)
    - computational social science와 매우 밀접하여 소셜 미디어에 기반하여 사회 과학적인 인사이트를 발견하는데 활용되고 있다.
- Information retrieval (major conference: SIGIR, WSDM, CIKM, RecSys)
    - 정보검색 분야이고 상당히 성숙되어 있는 분야이다.
    - 추천 시스템분야 또한 이곳에 속한다.