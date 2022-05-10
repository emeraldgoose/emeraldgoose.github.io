---
title: Self-supervised Pre-training Models 정리
categories:
  - BoostCamp
tags: [nlp]
---
## Recent Trends

- Transformer 모델과 self-attention block의 범용적인 sequence encoder와 decoder가 다양한 자연어 처리 분야에 좋은 성능을 내고 있다.
- Transformer의 구조적인 변경없이 인코더와 디코더의 스택을 12개 혹은 24개로 사용하고 이 상태에서 Fine-Tuning을 하는 방향으로 사용된다.
- 추천 시스템, 신약 개발, 영상처리 등으로 확장된다.
- NLG에서 self-attention 모델이 아직도 greedy decoding에서 벗어나지 못하는 단점도 존재한다.

## GPT-1

### Improving Language Understanding by Generative Pre-training

- GPT-1

    ![](https://drive.google.com/uc?export=view&id=1Dt6MzObB1bLToXuUl4s113c_2y2kvuiH)

    - 출처 : [openai.com/blog/language-unsupervised](https://openai.com/blog/language-unsupervised)

    - 다양한 special token들을 제안해서 간단한 task 뿐만 아니라 다양한 자연어 처리 task에서 모두 사용할 수 있는 통합된 모델을 제안했다.
    - 만약, Classification을 위해 학습된 모델이 있고 다른 Task를 수행하고 싶을 때, 마지막 layer를 떼어내고 random initialized된 layer를 붙여 해당 Task를 위해 다시 학습을 한다.
    - 학습을 할 때는 마지막 layer는 학습을 해야 하지만 이미 학습된 나머지는 상대적으로 learning_rate을 작게 줌으로써 큰 변화가 일어나지 않도록 한다.
    - 이전 학습 내용이 잘 담겨있도록 하여 그 정보를 원하는 Task에 활용할 수 있는 형태로 Pre-training 및 메인 Task를 위한 Fine-Tuning과정이 일어나게 된다.
    - Self-supervised learning이라는 것은 대규모 데이터로부터 얻은 지식을 소량의 데이터를 학습하는데 지식을 전이하여 성능을 향상하는 방법이다.

## BERT

### BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

- BERT도 GPT와 마찬가지로 Language Modeling이란 Task로써 문장에 있는 일부 단어를 맞추도록 Pre-trained된 모델이다.
- Transformer 이전에 LSTM을 이용하여 접근한 ELMO 방법도 존재한다. ELMO는 인코더와 디코더를 모두 LSTM으로 대체한 모델이다.

### Masked Language Model

- Motivation
    - 기존 한계점 : 이전 예측된 단어를 보고 다음 단어를 예측해야 하는 단점. 즉, 한쪽 방향에서의 정보만으로 예측한다.
    - 앞뒤의 문맥을 보고 예측하는 것이 좀 더 좋은 성능을 낼 것이다.

### Pre-training Tasks in BERT

![](https://drive.google.com/uc?export=view&id=1DCM7-K5IQcaIR1x1GUK-zj1iig_S11Mm)

- 출처 : [nlp.stanford.edu/seminar/details/jdevlin.pdf](nlp.stanford.edu/seminar/details/jdevlin.pdf)

- 중간에 단어를 [MASK]로 치환해서 단어가 무엇인지 맞추는 형태로 학습을 진행한다.
- 예측할 단어의 비율을 하이퍼 파라미터로 둔다.
    - 너무 적게 masking 한 경우 : 학습시 cost가 비싸다.(ex. 100단어를 읽어서 1단어를 예측함)
    - 너무 많이 masking 한 경우 : context를 잡아내는 것이 충분하지 않다.
- 여기서는 k=15%로 두고 학습을 진행했지만 [MASK]를 100% 모두 예측할 수는 없다.
    - [MASK]라는 것에 익숙해진 모델이 나올 수 있다. 그래서 [MASK]를 치환할 때도 아래와 같이 다르게 치환한다.
    - [MASK]로 치환할 단어들 중 80%는 실제로 [MASK]로 치환한다.
    - 남은 20% 중 10%는 random word로 치환한다.
    - 남은 10%는 그대로 둔다. 다른 단어로 바뀌어야 한다고 할 때 원래 있는 단어와 동일해야 한다고 알려주기 위함이다.

### Pre-training Tasks in BERT: Next Sentence Prediction

![](https://drive.google.com/uc?export=view&id=1wj90_COt5PwbCmbAeUstBvYW29Cj4Jj9)

- 문장 레벨에서 Pre-training하기 위한 기법이다.
- [MASK]위치의 단어를 예측하고 [CLS]토큰을 위해 Binary Classification layer를 하나 추가하여 입려된 두 문장이 연결되었는지 혹은 연결되지 않았는지 판단하도록 한다.

### BERT Summary

- 모델의 구조는 Trasformer 구조를 그대로 사용한다.
- Input Representation
    - WordPiece embeddings : 각각의 sequence의 단위를 subword라고 부르는 단위로 임베딩하여 입력벡터로 넣어준다.
    - Learned positional embeddings : positional embedding 벡터 또한 학습시킨다.
    - [CLS] - Classification embeddings
    - [SEP] - Packed sentence embedding
    - Segment Embedding : 또다른 Positional Embedding 벡터인데 문장레벨에서의 position을 반영한 벡터이다.

### BERT vs. GPT

- GPT는 특정 타임스텝 이후의 단어의 접근을 허용하지 않는다. BERT는 특정 타임스텝에서도 모든 단어의 접근을 허용한다.([MASK]와 같은 토큰도 포함)
- Traning-data size
    - GPT는 BookCorpus(800M)인 반면, BERT는 BookCorpus(2500M)을 학습했다.
- Training special toekns during training
    - BERT는 [SEP], [CLS], sentence A/B를 구분할 수 있는 Segment embedding을 넣어 학습한다.
- Batch size
    - BERT - 128,000 words; GPT - 32,000 words
    - BERT는 큰 batch size를 사용했을 때 안정된 결과를 얻을 수 있는데 이는 gradient descent 알고리즘을 수행할 때 다수의 데이터로 도출된 gradient로 학습하기 때문이다. 하지만 batch size를 키우기 위해 한번에 load해야 하는 메모리 + backpropagation을 위한 메모리가 너무 크다.
- Task-specific fine-tuning
    - GPT는 모든 fine-tuning에서 동일한 learning rate을 사용했지만 BERT는 task마다 learning rate를 fine-tuning했다.
    - BERT도 GPT와 동일하게 마지막 layer를 제거하고 새로운 task를 수행하기 위한 layer를 붙이고 남은 구조의 learning rate을 작게 해서 학습을 진행했다.

### Machine Reading Comprehension (MRC), Question Answering

- 기계독해 기반
- 문서와 질문을 이해하고 질문에 답을 찾는 Task
- SQuAD Dataset
    - v1.1 : 질문과 문서에 답이 존재
    - v2.0 : 질문이 있지만 문서에 답이 없는 경우도 존재
- SWAG Dataset

### BERT: Ablation Study

- layer와 paramter를 점진적으로 늘리면서 학습하면 성능이 지속적으로 증가한다.


## GPT-2

### GPT-2: Language Models are Unsupervised Multi-task Learners

- transformer의 레벨을 더 쌓아서 크기를 키운 모델이다. 또한, pre-trained 모델은 Language Model Task를 수행한다.
- 40GB의 text로 학습을 진행했고 데이터의 퀄리티를 증가시켰다.
- 여러 downstream task가 zero-shot setting으로써 다루어질 수 있다.
    - zero-shot setting : 어떠한 파라미터나 아키텍쳐의 변경이 없는 세팅

### GPT-2: Motivation (decaNLP)

- "The Natual Language Decathlon: Multitask Learning as Question Answering" 논문에 영향을 받았다.
- 모든 종류의 자연어 처리 task가 모두 Quesition Answering task로 바꿀 수 있다는 논문으로 자연어 처리 task들의 통합에 대해 다루고 있다.

### GPT-2: Datasets

- Reddit의 글에 있는 문서와 외부링크가 첨부되어 있는 경우 외부링크를 타서 내부 문서까지 모두 크롤링한다.
- Byte pair encoding (BPE)를 사용한다.

### GPT-2: Model

- Layer normalization이 각 sub-block의 입력으로 이동했다.
- 각 레이어를 random initialization할 때, 레이어의 인덱스에 비례해서 더 작은 값으로 초기화한다.
    - 레이어가 위로 올라갈 수록 거기에 쓰이는 선형변환의 해당하는 값들이 점점 더 0에 가까워지도록 하여 위쪽 레이어가 하는 역할이 점점 더 줄어들게 의도했다.

### GPT-2: Question Answering

- 대화형의 질의응답 데이터(Conversion question answering dataset(CoQA))가 있을 때, 학습데이터를 쓰지않고 질문을 읽고 답을 하게 했을 경우 55의 F1 score를 얻었다고 한다.
    - Fine-Tuned BERT는 89의 F1 score을 얻었다.

### GPT-2: Summarization

- CNN and Daily Mail Dataset
- Article 뒤에 TL;DR(Too long, didn't read)이라는 것을 삽입하여 TL;DR이 나올경우 앞쪽 모든 문단에 대해 요약하는 Task를 수행한다.

### GPT-2: Translation

- WMT14 en-fr dataset


## GPT-3

### GPT-3: Language Models are Few-Shot Learners

- GPT-3는 GPT-2를 개선한 모델인데 개선방향은 GPT-2의 파라미터보다 훨씬 많은 파라미터와 훨씬 더 많은 레이어를 쌓고 더 많은 데이터, 더 큰 배치 사이즈로 학습한 모델이다.

    ![](https://drive.google.com/uc?export=view&id=1IfMt9cN6S_Shf4wzA4bvgp56jYMtEGjq)

- Few-shot Learners

    ![](https://drive.google.com/uc?export=view&id=12dXEgelAp4Gdc2ha7WPS1i43Ogq4r6iB)

    - Zero-shot: task-description을 가지고 바로 answer를 답하도록 하는 task
    - One-shot: 한 가지 예를 주고 answer를 답하도록 하는 task
    - Few-shot: 여러가지 예를 주고 answer를 답하도록 하는 task
    - 이전에는 downstream을 위해 output layer를 추가하여 학습시키는 과정을 거쳤지만 지금은 변경하지 않고 inference과정 중에 예시를 input text에 example를 포함시키는 방법을 사용한다. → zero-shot보다 성능이 좋아졌음
- 모델 사이즈를 키울수록 zero-shot, few-shot 성능이 빠르게 올라간다.

    ![](https://drive.google.com/uc?export=view&id=1JHHgOz0D46RmVx5SHegfJsfgIHcMjrBZ)

## ALBERT: A Lite BERT for Self-supervised Learning of Language Representations

- 많은 pre-trained모델은 많은 파라미터와 많은 레이어를 가지고 있는데 이로 인해 많은 메모리와 학습 시간이 필요로 했다.
- ALBERT는 기존 BERT모델보다 가볍게 하면서 성능 하락이 없는 모델이다.
    - Factorized Embedding Parameterization
    - Cross-layer Parameter Sharing
    - (For Performance) Sentence Order Prediction ← 새롭게 제안
- Factorized Embedding Parameterization
    - Transformer에는 입력의 dimension이 첫 self-attention 입력에도 유지가 되고 다음 레이어의 입력에도 유지가 된다.
    - ALBERT는 Embedding Layer의 dimension을 줄이는 기법을 제시했다.
        - 기존 BERT는 word embedding의 dimension이 입력의 dimension과 같아야 했다.
        - ALBERT는 word embedding의 dimension을 입력의 dimension보다 작게 하고 다시 입력의 dimension과 같게 할 수 있는 layer를 추가한다.
        - 만약, 기존 BERT의 embedding vector의 크기가 (500 $\times$ 100)라고 하고 ALBERT의 embedding vector의 크기는 (500 $\times$ 15), weight matrix의 크기 (15 $\times$ 100)를 가지게 된다고 가정하자.
        - 이때, BERT는 50000의 크기를 가지는데 반해 ALBERT는 7500 + 1500 = 9000의 크기만을 가지게 되어 메모리를 절약할 수 있다.
- Cross-layer Parameter Sharing
    - 선형변환 matrix를 공유하는 아이디어이다.
    - Shared-FFN: feed-forward network parameter를 공유
    - Shared-attention: attention parameter를 공유
    - All-shared: Shared-FFN + Shared-attention

    ![](https://drive.google.com/uc?export=view&id=1D-sF2dFk_xHfZ76naH542k0eAIkWOYNA)

    - 모두 공유했을 때 가장 적은 파라미터 수를 가지고 성능에 있어서는 하락폭이 크지 않다.
- Sentence Order Prediction
    - 기존 BERT모델에서는 Next Sentence Prediction task는 실효성이 없었다. → NSP를 좀 더 유의미한 pre-training을하고 싶었다.
    - 연속적인 두 문장을 가져와서 순서대로 SEP을 추가한 데이터와 순서를 반대로 한 데이터를 만들어  binary classification task를 수행하도록 했다.
        - Negative sample 또한 두 문서(docs)에서 랜덤하게 추출된 두 문장을 연결해서 next sentence가 아니도록 학습을 진행했다.

 

## ELECTRA: Efficiently Learning an Encoder that Classifies Token Replacements Accurately

![](https://drive.google.com/uc?export=view&id=17k8Gjv6xybgtgjvJW3aYXN-Ymi65hSjJ)

- Generator는 MLM을 통해 [MASK]단어를 복원한다.
- Discriminator는 self-attention을 쌓은 모델인데 각 단어별로 original단어인지 어색하여 replaced된 모델인지 판단하는 모델이다.
- 두 가지 모델이 Adversarial(적대적)관계로 학습이 진행된다. (Generative adversarial network의 아이디어와 같다.)
- 이 모델의 경우 Discriminator를 pre-trained 모델로 사용하게 된다.

## Light-weight Models

### DistillBERT

- Teacher모델과 Student모델이 있는데 Teacher모델은 Student모델을 가르치는 역할을 한다. Student모델은 Layer나 파라미터 수가 Teacher모델보다 작지만 Teacher모델의 output을 잘 모사할 수 있도록 학습된다.
- Student모델의 ground truth probability는 Teacher 모델의 probability가 된다.

### TinyBERT

- knowledge distillation을 이용하여 DistilBERT와 비슷하게 Teacher모델과 Student모델이 존재한다.
- Student 모델이 Teacher모델의 probability를 따라할 뿐만 아니라 각 self-attention block이 가지는 $W_Q$, $W_K$, $W_V$와 같은 Attention matrix와 그 결과로 나오는 히든 스테이트 벡터들 까지도 유사해지도록 학습을 진행한다.
- Mean Squared Error(MSE)를 통해 가까워지도록 한다. 그러나 일반적으로 Student 모델의 차원은 Teacher 모델의 차원보다 작을 수 있기 때문에 일반적인 loss를 적용할 수는 없다.
- 그래서 Teacher 모델의 학습가능한 Fully connected layer를 하나 두어서 차원이 다른 벡터의 mismatch를 줄이는 방식으로 학습을 진행한다.

## Fusing Knowledge Graph into Language Model

- BERT 모델들은 주어진 문장이 있을 때 문맥을 파악하고 단어들 간의 유사도나 관계를 파악하는 것은 잘 하지만 주어져 있는 문장에 포함되어 있지 않은 정보가 필요한 경우에는 잘 활용하지 못한다.
    - 예를들어, "꽃을 심기 위해 땅을 팠다"와 "건물을 짓기 위해 땅을 팠다"를 학습했을 때 "땅을 판 도구는 무엇인가?"라는 질문에 대해 제대로 답을 하지 못한다.
    - 인간의 경우 꽃을 심기 위해 부삽을 사용하고 건물을 짓기 위해 중장비를 이용한다는 외부지식(상식)을 가지고 있어 위의 답을 할 수 있다. 따라서, 모델들도 이런 상식이 답을 하기 위한 외부지식이 필요하다고 판단할 수 있다.
    - Knowledge Graph는 이런 외부지식들을 담고 있는 그래프이다. "(땅을)파다"라는 객체와 "땅"이라는 객체가 연결되어 있고 다시 땅을 파기 위한 어떤 도구들이 연결되어 있다.

### ERNIE: Enhanced Language Representation with Information Entities

- Information fusion layer는 token embedding과 entity embedding을 concatenation해준다.

### KagNET: Knowledge-Aware Graph Networks for Commonsense Reasoning

- A knowledge-aware reasoning framework for learning to answer commonsense questions
- 질문과 답 pair에 대해 외부 knowledge graph에서 sub-graph검색하여 관련 지식을 capture한다.