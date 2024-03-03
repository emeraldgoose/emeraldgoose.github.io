---
title: On the Effect of Pretraining Corpora on In-context Learning by a LLM
categories:
 - paper
tags:
    - Corpus
---

> 모두의연구소에서 논문저자가 직접 논문을 리뷰해주는 세미나가 열렸습니다. 주제가 재밌어 보여 발표를 듣고 논문을 다시 읽어보았습니다.

## Motivation
GPT라는 Large Scale Langauge Model이 등장하면서 언어 모델의 새로운 시대를 열게 되었습니다. GPT에서 In-context Learning이라는 방식을 사용하는 점이 특징입니다.

### In-context Learning
In-context Learning은 사전학습 모델에 풀고자 하는 태스크를 input으로 넣는 방식을 말합니다. 예제에 따라 zero-shot, one-shot, few-shot learning으로 나뉘어집니다.
- Task description + example(zero/one/few shot) -> answer

## Task Definition
모델은 HyperCLOVA를 사용하고 1.3B의 파라미터를 가지고 있습니다.  

Corpus는 네이버블로그, 네이버카페, 네이버뉴스, 네이버댓글, 네이버지식인, 위키피디아, 모두의말뭉치를 사용합니다.  

Downstream Task로는 NSMC(영화리뷰), KorQuAD, KLUE-YNAT(뉴스제목분류), AI hub(한영/영한 번역)로 진행합니다.  

## Experimental Results
### Effect of Corpus Source
![](https://lh3.google.com/u/0/d/1lkOE5QdbilT_WV80gvNVn-J6-IJLtIoT){:width="800"}  
테이블에서 말뭉치의 종류에 따라 다르게 in-context learning 성능이 나타는 것을 볼 수 있습니다.  
- 블로그 데이터(Blog)로 학습한 모델이 카페(Cafe)나 뉴스(News)로 학습한 모델보다 few-shot 성능이 ALL 모델에 근접합니다.
    - ALL 모델은 모든 데이터를 학습한 모델을 말합니다.
- 모두의말뭉치(Modu)로 학습한 모델은 카페나 뉴스로 학습한 모델보다 좋은 성능을 냅니다. 하지만 Modu 사이즈는 카페나 뉴스 말뭉치의 1/10배보다 작습니다.  

### Effect of Corpus size
![](https://lh3.google.com/u/0/d/1igaRjlK06EdM7kY6YAlRN5oePI8sQPeW){:width="700"}    
말뭉치 사이즈를 150B에서 56B로 줄였을 때는 성능이 비슷합니다. 말뭉치 사이즈가 성능을 감소시키는 것은 아닙니다.  
하지만 6B 토큰으로 학습한 모델은 150B 토큰의 카페와 뉴스 말뭉치로 학습한 모델의 성능보다 낮게 나옵니다.  
위의 Table 2에서는 블로그 54B 토큰과 27B 토큰 데이터를 학습한 결과가 있습니다. 블로그 150B 토큰 데이터와 54B 토큰 데이터로 학습한 모델의 성능이 비슷하지만 ALL 6B 토큰과 블로그 27B 토큰 데이터는 블로그 54B 토큰 데이터로 학습한 모델보다 성능이 나오지 않습니다.

![](https://lh3.google.com/u/0/d/1xx1eT9X21wpvZypYbhc1n6Orl7rUPuLM){:width="600"}    

fig 3에서 모델의 사이즈와 토큰 사이즈에 대해 학습한 결과를 보여줍니다. 
150B 토큰으로 학습한 모델보다 56B 토큰으로 학습한 모델의 성능 감소가 크게 나타나지는 않습니다.  

### Effect of Combining Corpora
![](https://lh3.google.com/u/0/d/1lmabxCG-WuuldhZs4NRPmr1oetgdIABI){:width="800"}    
Table 4에서 in-context learning에서 능력이 두 말뭉치 조합에 의해 발생될 수 있음을 보여줍니다.  
- 지식인(KiN) + 위키피디아(Ency) 모델은 대부분의 태스크에서 성능이 좋습니다.
- 카페(Cafe) + 지식인(KiN) 모델도 대부분의 태스크에서 성능이 좋아졌습니다. 카페와 지식인 각각의 데이터로만 학습한 모델들은 해당 태스크에서 성능이 좋지 못했습니다.(Table 2)  

이런 현상은 Multi Task Learning에 의해 발생한 것으로 생각됩니다. Multi Task Learning은 연관있는 Task를 같이 학습하는 방법을 말하며 Object function이 다양한 Next Word Prediction을 학습하기 때문에 일반화 과정이 없는 능력을 촉진한 것으로 보입니다. 지식인 데이터와 위키피디아 데이터로 Next Word Prediction을 수행하면서 MRC task를 학습한 것으로 보여집니다. 

그러나 말뭉치 조합이 항상 성능을 향상시키지는 않습니다.  
- 카페와 뉴스 데이터를 조합한 경우 KorQuAD 태크스에서 각각 말뭉치로 학습한 성능보다 조금 나아지지만 나머지 태스크(NSMC, KLUE-YNAT)에서 성능이 낮아집니다.

### Effect of Domain Relevance
Table 2에서 말뭉치와 Downstream task와의 관계가 few-shot 성능을 항상 보장하지는 않습니다.  
- 지식인(KiN)과 위키피디아(Ency) 모델은 KorQuAD 태스크를 잘 수행하지 못합니다.
- KLUE-YNAT에는 뉴스 헤드라인 쿼리가 포함되어 있음에도 뉴스(News) 모델은 KLUE-YNAT 태스크를 잘 수행하지 못합니다.

Table 4에서 뉴스 + 지식인 + 위키 모델이 지식인 + 위키 모델보다 KLUE-YNAT F1 스코어가 낮습니다.  
- YNAT에서 성능이 좋았던 지식인 + 위키 모델에서 뉴스를 추가한 것뿐인데 성능이 낮아졌습니다.  

![](https://lh3.google.com/u/0/d/1tshVYl37oO4AIgsXz5MfB3s_hq0uJ9Sf){:width="600"}    
말뭉치와 태스크와의 Vocabulary overlap을 조사한 히트맵입니다.  
높은 Vocabulary overlap이 높은 Downstream task 성능을 가져오지는 못합니다.
- 모두의말뭉치(Modu)는 번역(AI Hub)태스크와 높은 오버래핑을 보이지만 블로그(Blog)나 지식인(KiN) 모델보다 성능이 낮습니다.

하지만 지식인(KiN) 모델에서 번역(AI Hub) 태스크에서 반례가 존재합니다.  
- 지식인 데이터는 한국어로 영어에 대한 질문이 많습니다. 지식인 모델은 한영 문장 패턴들을 학습할 수 있습니다.  
- 지식인 모델이 한영 번역 태스크는 성능이 나오지만 나머지 태스크에서는 성능이 낮게 나옵니다.  

![](https://lh3.google.com/u/0/d/11aNe9bnSwZhtoYQJuy5hu1hMtqLTQMfs){:width="800"}    
![](https://lh3.google.com/u/0/d/1-z_oVEl3f5ubpMoLls8QXcLn1lsGR5D5){:width="800"}    

Zero-shot에서 도메인 관련성이 이런 현상을 더 잘 나타냅니다.  
- 뉴스 말뭉치는 KLUE-YNAT 태스크에 도움을 줍니다. Table 3, 5에서 뉴스 말뭉치를 추가했을 때 성능이 오르는 것을 볼 수 있습니다. 모든 데이터를 학습하는 ALL 모델보다 나은 성능을 보입니다.
- 지식인 말뭉치 자체로는 Zero-shot에서 few-shot보다 낮은 성능을 보입니다. 그러나 지식인 데이터를 다른 말뭉치와 조합하는 경우 성능이 오르는 것을 볼 수 있습니다.  

### Perplexity and Downstream Task
Figure 2는 각 in-context learning 성능을 ALL 모델의 성능으로 나눠 Normalizing한 결과입니다. Perplexity와 in-context 성능간의 관계를 알아보기 위해 진행했으나 둘의 관계가 있다는 가설을 세울 수 없었습니다.  
![](https://lh3.google.com/u/0/d/1HGg4mxcZRfR6qmE6eYkDR1CfduGKyLoZ){:width="600"}    
- Table 2에서 블로그 모델은 낮은 PPL을 가지지만 높은 성능을 가집니다. 반면, 위키피디아 모델은 높은 PPL을 가지지만 낮은 성능을 가집니다.

## Conclusion
이번 연구에서 발견한 점은 다음과 같습니다.  
1. 말뭉치에 따라 성능이 크게 달라질 수 있다. Perplexity가 높다고 성능이 꼭 높은 것은 아니다.
    - 학습 말뭉치 선정 시 Few-shot에서 PPL 기반이 적합하지 않을 수 있습니다.
2. 말뭉치를 섞으면 없던 능력이 생기기도 한다.
3. 태스크와 비슷한 말뭉치가 사전훈련에 포함되더라도 성능이 꼭 좋아지는 것은 아니다.

이번 연구에서는 하이퍼클로바를 이용하여 말뭉치 역할이 중요함을 알 수 있었습니다. 아쉬운 점으로는 한국어 데이터와 한국어 모델에 대해서만 실험했기 때문에 영어로 확장할 필요가 있어 보입니다. 그리고 사전학습 방법이 Next Word Prediction만 있는 것도 추가 연구가 필요해보입니다. 

하지만 모델이나 학습 방법에서의 관점이 아닌 Corpus 관점에서 연구가 진행된 점에서 생각보다 재밌었던 논문이었습니다. 