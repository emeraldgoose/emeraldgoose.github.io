---
title: 좋은 모델 찾기
categories:
  - BoostCamp
tags: [AutoML]
---
## Image augmentation

### Data Augmentation

- 기존 훈련 데이터에 변화에 가해, 데이터를 추가로 확보하는 방법
- 데이터가 적거나 Imbalance된 상황에서도 유용하게 활용가능
- 직관적으로는 모델에 데이터의 불변하는 성질을 전달 → Robust해짐

### 경량화, AutoML 관점의 augmentation?

경량화 관점에서는 직접적으로 연결이 되지는 않으나, 성능 향상을 위해서는 필수적으로 적용되어야 하는 깁버

Augmentation 또한 일종의 파라미터로, AutoML의 search space에 포함이 가능한 영역

### Object detection에서의 대표적인 Augmentation

- MixUP
- CutMix
- Crop, Rotation, Flip,...
- Mosaic
- Blur
- ...

## Image augmentation 논문 리뷰

### Image Augmentation: issue

- Task, Dataset의 종류에 따라 적절한 Augmentation의 종류, 조합, 정도(magnitude가 다를 것)

### AutoAugment: AutoML로 augmentation policy를 찾자

- Data로부터 Data augmentation policy를 학습
- 총 5개의 sub policy, 각 sub policy는 2개의 augmentation type, 각 probability와 magnitude를 가짐
- 학습을 하는 방법은 딥러닝을 사용한다
- 성능 향상을 보이지만, 학습에 매우 큰 자원이 필요

### RandAugment

- 2개의 파라미터(N: 한번의 몇개 적용, M: 공통 magnitude)로 search space를 극단적으로 줄임
- 이 설정으로, policy를 찾는 여러 알고리즘과 거의 동등한 성능을 보임

## AutoML 구동 및 결과 분석

### 가상의 시나리오 설정(현업에서는 훨씬 더 구체적)

- PC 1대 리소스로 cifar 10 classification 태스크에 대해서 파라미터 적고, 성능 좋은 모델 요구
- Q: Objective는?
    - A: Multi object 문제(min Param, max Acc)
- Q: 해당 태스크를 한번 수행하는데 걸리는 시간?
    - 모델의 크기에 따라 다르지만 약 2~4시간
- Q: 해당 리소스에서 최대 몇개의 세션까지 구동가능?
    - 대략 2~3개까지 가능
- 일반적으로 100~200회는 되어야 만족스러운 결과를 얻어낼 수 있다는 것을 경험적으로 알고 있을 때 하루에 24 트라이만 가능
- 따라서 시간을 줄여야 한다.
    - 학습 시간을 줄이고 싶은데, 데이터셋을 소규모로 sample해서 모델을 찾아도 될까?
    - 모델의 성능에 가장 큰 영향을 주는 인자만 Search space로 넣고 싶은데, 어떤 것들이 있을까?
    - ...
- Search space : 7개의 블록, 하이퍼파라미터는 전부 고정
- wandb와 같이 기록하는 플랫폼을 사용하여 베이스라인과 비교하도록 한다.

### 정리

- AutoML로 데이터셋, task에 특화된 모델을 찾는 것이 가능
- 현실적인 제약(시간, 리소스 등)들을 해소하기 위한 엔지니어링은 아직 사람의 노하우가 필요
- 데이터셋이 계속 추가되고 변화하는 현실 상황에서 이전 결과를 지속적으로 활용하려면 시스템을 어떻게 구성해야 할까?(AutoML에서의 CI/CD)