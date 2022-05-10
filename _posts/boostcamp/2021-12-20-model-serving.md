---
title: Model Serving 정리
categories:
  - BoostCamp
tags: [serving]
---
## Serving Basic

### Serving

> 머신러닝 모델을 개발하고, 현실 세계(앱, 웹, Production 환경)에서 사용할 수 있게 만드는 행위
> 

크게 2가지 방식 존재

- Online Serving
- Batch Serving
- 그 외에 클라이언트(모바일 기기, IoT Device 등)에서 Edge Serving도 존재
- Inference : 모델에 데이터가 제공되어 예측하는 경우, 사용하는 관점, Serving과 용어가 혼재되어 사용되는 경우도 존재

## Online Serving

- 웹 서버
    - HTTP를 통해 웹 브라우저에서 요청하는 HTML 문서나 오브젝트를 전송해주는 서비스 프로그램
    - 요청(Request)을 받으면 요청한 내용을 보내주는(Response) 프로그램
- API(Application Programming Interface)
    - 운영체제나 프로그래밍 언어가 제공하는 기능을 제어할 수 있게 만든 인터페이
- Online Serving
    - 요청(Request)이 올 때마다 **실시간으로 예측**
    - 클라이언트(Application)에서 ML 모델 서버에 HTTP 요청(Reqeust)하고, 머신러닝 모델 서버에서 예측한 후, 예측값(응답)을 반환(Response)한다.
    - Serving Input - Single Data Point
        - **단일 데이터**를 받아 실시간으로 예측을 하는 예제
        - 기계 고장 예측 모델
            - 센서의 실시간 데이터가 제공되면 특정 기계 부품이 앞으로 N분안에 고장날지 아닐지를 예측
        - 음식 배달 소요 시간 예측
            - 해당 지역의 과거 평균 배달 시간, 실시간 교통 정보, 음식 데이터 등을 기반으로 음식 배달 소요 시간 예측
    - ML모델 서버에 요청할 때, 필요시 ML 모델 서버에서 데이터 전처리를 해야할 수 있음(혹은 분리를 위해 전처리 서버 / ML 모델 서버로 나눌 수 있음)
    - 서비스의 서버에 ML 서버를 포함하는 경우도 있고 ML 서버를 별도로 운영하는 경우도 존재
    - Online Serving을 구현하는 방식
        - 직접 API 웹 서버 개발 : Flask, FastAPI
        - 클라우드 서비스 활용 : AWS의 SageMaker, GCP의 Vertex AI
        - Serving 라이브러리 활용 : Tensorflow Serving, Torch Serve, MLFlow, BentoML 등
- Online Serving에서 고려할 부분
    - 실시간 예측을 하기 때문에 **지연시간**(Latency)를 최소화해야 함
    - Input 데이터를 기반으로 Database에 있는 데이터를 추출해서 모델 예측해야 하는 경우 데이터 추출을 위한 쿼리 실행, 결과를 받기까지의 시간 소요
    - 모델이 수행하는 연산 시간 → 경량화 작업이 필요할수도
    - 결과값에 대한 보정이 필요한 시간 → 유효하지 않은 예측값이 반환될 경우
    

### 클라우드 서비스 활용

- 클라우드 서비스 장점
    - 직접 구축해야 하는 MLOps의 다양한 부분이 만들어짐
    - 사용자 관점에선 PyTorch 사용하듯 학습 코드만 제공하면 API 서버가 만들어짐
- 클라우드 서비스 단점
    - 클라우드 서비스가 익숙해야 잘 활용할 수 있음
    - 비용문제 : 직접 만드는 것보단 더 많은 비용이 나갈 수 있음
- 소수의 인원만 존재하며, 소수의 인원이 많은 업무를 해야 하는 경우 유용
- 클라우드 내부 실행 구조를 잘 알아야 문제 상황이 발견되었을 때 잘 해결할 수 있음
- 클라우드 서비스에선 어떤 방식으로 AI제품을 만들었는지 확인할 수 있어서 사용해본느 것도 좋음

### Serving 라이브러리 활용

- 다양한 방식으로 개발할 수 있지만, 매번 추상화된 패턴을 가질 수 있음
- 추상화된 패턴을 잘 제공하는 오픈소스를 활용하는 방식

### Serving 방법을 선택하는 가이드

- 어떤 방법을 쓰느냐는 주어진 환경(회사에서 주어진 일정, 인력, 예산, 요구 성능 등)에 따라 다름

### Online Serving에서 고려할 부분

Serving 할 때 Python 버전, 패키지 버전 등 Dependency가 굉장히 중요하다.

"재현 가능"하지 않은 코드는 Risk를 가지고 있는 코드이므로 배포할 수 없다.

관련한 도구로는 Virtualenv, Poetry, Docker 등이 있다.

## Batch Serving

- 주기적으로 학습을 하거나 예측을 하는 경우
- 일정 주기 마다 **최근 데이터**를 가지고 예측
- Batch 묶음, 한번에 많은 예측을 실행
- Batch Serving 관련한 라이브러리는 따로 존재하지 않고 Airflow, Cron Job 등으로 스케쥴링 작업(**Workflow Scheduler**)을 진행한다.
    - 학습 주기와 예측 주기가 다를 수도 있다.
    - 아파치의 Airflow가 데이터 엔지니어링에서 자주 활용된다.
- 예를들면, 추천시스템은 1일 전에 생성된 컨텐츠에 대한 추천 리스트를 예측한다.
- 실시간이 필요 없는 대부분의 방식에서 활용 가능하다.
- 장점으로는 코드를 함수화한 후 주기적으로 실행만 하면 되므로 간단한 구조를 가지고 있으며 한번에 많은 데이터를 처리하기 때문에 Latency 문제가 없음
- 단점으로는 실시간으로 활용할 수 없고 Cold Start 문제가 있다.
    - Cold Start 문제 : 오늘 새로 생긴 컨텐츠는 추천할 수 없음

## Online Serving vs. Batch Serving

### Input 관점

- 데이터 하나씩 요청하는 경우 : Online
- 여러가지 데이터가 한꺼번에 처리되는 경우 : Batch

### Output 관점

- Inference output을 어떻게 활용하는지에 따라 다름
    - API형태로 바로 결과를 반환 → Online
    - 서버와 통신 필요 → Online
    - 1시간에 1번씩 예측해도 좋은 경우 → Batch

### Serving 형태 선택

실시간 모델 결과가 어떻게 활용되는지에 대한 생각이 중요하다.

예측해도 활용이 되지 않는다면 Batch로 진행해도 무방하다.

Batch Serving의 결과를 Database에 저장하고, 서버는 Database의 데이터를 쿼리해서 주기적으로 조회하는 방식으로 사용할 수 있음