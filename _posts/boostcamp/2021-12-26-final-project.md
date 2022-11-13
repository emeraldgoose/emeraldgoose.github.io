---
title: 악성채팅 탐지 시스템 구현
categories:
  - BoostCamp
tags: [cnn, nlp]
---
## 목적

> 유저가 많은 방의 채팅 속도가 빨라 모든 악성채팅을 거르지 못하기 때문에 실시간으로 탐지하여 관리자에게 악성유저들을 리포트할 수 있는 애플리케이션을 개발하고자 했습니다.
> 

## 계획

---

### 모델

실시간으로 문장의 성향을 파악하기 위해서는 매우 무겁고 느리기 때문에 기존 언어모델을 사용할 수 없었습니다.

그리고 Toxicity text detection은 비교적 쉬운 task에 속합니다. 

또한, 서버의 환경을 생각해보면 보통 GPU가 있기보단 좋은 CPU와 많은 메모리를 사용하여 서버를 구동하고 있습니다.

따라서 빠른 속도를 가진 CNN 기반의 모델을 구현하여 성능을 어느정도 손해보더라도 속도를 가져가려고 했습니다.

### 데이터

악성채팅은 보통 욕설, 혐오표현, 공격적인 표현 등으로 구성되어 있습니다.

하지만 이런 분류 모두 사람의 주관적 시선으로 만들어져 기준이 모두 다를 수 있습니다. 따라서 저희는 욕설과 혐오표현을 위주로 학습을 진행했고 해당 채팅을 친 유저의 판단은 관리자에게 맡기는 방향으로 계획했습니다.

## 악성채팅의 유형

---

### 욕설, 혐오표현의 유형

욕설의 경우, 한 단어로 이루어진 경우가 많습니다.  하지만 필터링 기술이 발전함에 따라 필터링을 피하기 위해 욕설 사이에 의미없는 문자를 넣는 방식을 많이 사용합니다.

혐오표현의 경우 혐오적인 단어를 보통 포함합니다. 특히 특정 집단을 일컫는 단어를 사용합니다.

### 트위치 채팅의 유형

트위치는 스트리머마다 특정 밈(Meme)을 사용하는 경우가 많습니다. 젊은 사람들이 많이 보기 때문에 인터넷 커뮤니티 단어들이 사용되고 짧은 문장의 채팅이 자주 올라옵니다.  

그러나, 트위치는 구독을 했을 시 스트리머마다 다른 이모지를 제공하며 그 이름도 다릅니다. 그리고 그림을 그리는 경우도 존재합니다.

## Challenge

---

### Word embedding

문장을 그대로 사용할 수는 없으므로 Huggingface의 Tokenizer를 사용하여 Word embedding을 진행했습니다.

corpus만 확보하면 BertWordPieceTokenizer로 빠르게 학습할 수 있고 다루기 쉬운 장점이 있습니다.

그리고 Rust로 구현되어있기 때문에 굉장히 빠릅니다.

### Punctuation

인터넷 댓글들은 중간에 특수문자가 들어가는 경우가 많습니다. 오타이거나 필터링을 피하기 위해서 등의 이유를 들 수 있는데 이런 문장 또한 포함되는 것을 가정해야 하므로 문장의 단어 중간에 Punctuation을 삽입하여 모델이 Robust하도록 유도했습니다.

단어 사이에 Punctuation을 넣는 것도 생각해보았지만 Tokenizer가 특수문자가 들어간 단어를 [UNK]로 대체하면서 성능이 떨어졌습니다.

이후, Meta Pseudo labeling을 구현하면서 위의 전처리 방법을 augmentation하는데 사용했습니다.

### CNN

NLP task를 위한 CNN 모델을 구현하는 것이 처음이라서 여러가지 블로그들을 찾아봤는데 공통적인 구조는 Embedding layer에서 Conv layer로 쓸고 pooling을 하는식으로 설계되어 있습니다. 저 또한 이러한 구조를 크게 벗어나지 않게 설계했습니다.

### BiLSTM

CNN만으로 확보한 데이터셋 기준으로 어느 정도의 성능을 내주지만 성능에 대한 욕심이 조금 더 있었고 NLP에 적합한 BiLSTM을 사용하여 성능을 더 올릴 수 있었습니다.

### Benchmark

|Model|Best Acc|Best F1|Inference time(128 batch, CPU)|
|-|-|-|-|
|Rule based|0.81|0.690|**0.01s**(1300 words)|
|CNN|0.90|0.878|0.03s|
|BiLSTM|**0.93**|0.90|0.28s|
|CNN + BiLSTM|0.92|**0.911**|0.3s|

CNN과 BiLSTM을 같이 사용할 때 CNN 대비 10배 느려지지만 좀 더 다양한 표현을 잡으려면 꼭 사용해야 한다고 판단했습니다.  
병렬처리를 사용한다면 시간이 빨라지긴 하겠지만 아직 공부하지 않았으므로 패스합니다.

### Semi-Supervised Learning

저희의 목적은 트위치 채팅 데이터의 성향을 판단하는 것이므로 트위치 문장들을 데이터로 모델에 먹일 필요가 있습니다. 그러나 레이블링되지 않은 데이터를 사람 손으로 레이블링하기에는 데이터의 양이 너무 많습니다.

Pseudo Labeling이란 방법이 있지만 Pseudo label을 생성하는 모델에 의존해야 하며 사용하려는 모델이 Pseudo label을 생성하고 다시 학습하게되면 Bias가 증폭되는 단점을 갖고 있습니다.

이를 해결하기 위해 Meta Pseudo Labels라는 방법을 사용하여 좀 더 정확한 레이블링을 할 수 있도록 구현하고자 했습니다.

### Meta Pseudo Labels

- Reference : [https://arxiv.org/abs/2003.10580](https://arxiv.org/abs/2003.10580)

겉핥기로 공부해서 논문 내용을 정확하게 보지는 않았지만 대략적인 설명은 다음 그림과 같습니다.

![](https://drive.google.com/uc?export=view&id=14HfiaSGFEWCmyNdNaD-_aTCrXWZg0o3B){:width=400}  

기존 Pseudo label 방법은 Pre-trained Teacher로부터 생성된 Pseudo label을 Student에게 학습하는 과정이지만 MPL 방법은 Student가 Pseudo label을 학습하지만 Student의 loss를 Teacher가 학습합니다.

![](https://drive.google.com/uc?export=view&id=1VB9geH6mpHxgfR827u8iUHsPiFs2uG3m){:width=400}  

Supervised learning보다 MPL이 더 좋은 퍼포먼스를 보여준다는 일러스트입니다. 

그리고 이 논문에서는 ImageNet 벤치마크에서 EfficientNet-L2와 EfficientNet-B0을 가지고 MPL를 적용했는데 현재 SOTA급의 성능을 내고 있는 중입니다.

![](https://drive.google.com/uc?export=view&id=1vFXRUOcIu5g-eKLLqo2TvPz2Gc4WPg-0){:width=400}  

가장 중요한 점은 Teacher-Student 형태를 가지고 있지만 Teacher의 좀 더 정확한 Pseudo Labels를 생성하기 위해 최적화하는 과정입니다.

### MLflow

모델 버전관리와 배포가 쉬운 도구인 MLflow를 사용했습니다. 

MLflow의 tracking server를 GCP(Google Cloud Platform)으로 지정했는데 이때 Docker를 사용하면 굉장히 쉽게 배포가 가능합니다.

```
FROM python:3.10.1-slim-buster

COPY . /mlflow
WORKDIR /mlflow

RUN apt-get update && \
    pip install pip --upgrade && \
    pip install -r requirements.txt

ENV BACKEND_STORE_URI=mysql+pymysql://{Google-cloud-storage}
ENV DEFAULT_ARTIFACT_ROOT=gs://{tracking-server}
ENV MLFLOW_TRACKING_URI=http://localhost:5000
EXPOSE 80
```
python 이미지 위에 mlflow를 설치하고 환경변수를 넣어주는 방법을 사용했습니다. 원래는 환경변수를 넣어주는 것이 좋지는 않지만 CLI에서 넣을때 에러가 발생하는 경우가 있어서 dockerfile에 넣어주는 걸로 결정했습니다.

또한, 새로운 기능을 넣거나 버전을 다운그레이드 시킬때도 아티팩트들이 그대로 google-cloud-storage에 저장되어 있기 때문에 새롭게 학습시킬 필요가 없습니다.

그러나 [[BUG] 'mlflow.pyfunc.load_model()' need to be added keyword argument #4749](https://github.com/mlflow/mlflow/issues/4749)와 같이 gpu상에서 학습시킨 모델을 cpu상의 시스템에서 사용할 수 없습니다.

torch에서는 gpu에서 학습한 모델을 cpu만 사용하는 시스템에서 불러올 수 있게 `torch.load()`시 `map_location`의 인자를 'cpu'로 주어 cpu상의 시스템으로 옮길 수 있도록 하지만 mlflow는 이런것을 지원하지 않았습니다.

그래서 gpu가 없는 서버에 모델을 배포하려면 cpu로 학습시켜야 한다는 단점이 있었습니다. 다행히 애플리케이션을 제공받은 gpu가 있는 서버에서 시연하기로 해서 cuda로 학습할 수 있었습니다.

## 회고

---

airflow까지 해보고 싶었는데 SQLAlchemy하고 충돌때문인지 `airflow db init` 를 실행하면 자꾸 에러나서 다운그레이드도 해봤는데 결국 해결하지 못했습니다.

그리고 MLflow의 파이프라인 작성을 해보지 못해서 이와 관련된 airflow와 mlflow의 파이프라인 작성을 연습해보려고 합니다.

마지막으로 Docker를 제대로 써본건 이번이 처음인데 정말 편한 도구였습니다. k8s도 해보고 싶네요.
