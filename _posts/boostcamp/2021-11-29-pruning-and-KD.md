---
title: Pruning and Knowledge Distillation
categories:
  - BoostCamp
tags: [lightweight, pruning, knowledge Distillation]
---
## Pruning

- Pruning은 중요도가 낮은 파라미터를 제거하는 것
- 어떤 단위로 Pruning? → Structured(group) / Unstructured(fine grained)
- 어떤 기준으로 Pruning? → 중요도 정하기(Magnitude(L2, L1), BN scaling facgtor, Energy-based, Feature map, ...)
- 기준은 어떻게 적용? → Network 전체로 줄 세워서(global), Layer 마다 동일 비율로 기준(local)
    - global : 전체 n%, 어떤 layer는 많이, 어떤 layer는 적게
    - local : 모든 layer를 균일하게 n%
- 어떤 phase에? → 학습된 모델에(trained model) / initialize 시점에(pruning at initialization)

### Structured Pruning

- 파라미터를 그룹 단위로 pruning(그룹 단위는 channel/filter/layer level 등이 가능)
- Masked (0으로 pruning된) filter 제거 시 실질적 연산 횟수 감소로 직접적인 속도 향상
- BN
    - Learning Efficient Convolutional Networks through Network Slimming(ICCV 2017)
    - Scaling factor $\gamma$
        - BN의 scaling factor $\gamma$는 Conv의 Out Channel에 곱해지는 형태
        - $\gamma$가 크면, 해당 Filter의 weight크기는 일반적으로 클 것
        - $\gamma$의 크기를 기준으로 작은 p%의 채널을 pruning
        - $\gamma$에 L1-norm을 Regularizer로 사용, Insignificant Channels은 자연스럽게 Prune되도록 유도
- Rank
    - Rank를 기준으로 한 논문 : HRank: Filter Pruning using High-Rank Feature Map (CVPR 2020)
    - Weight가 아닌 실제 Feature map output을 보자!
    - Feature map output의 각 Filter에 SVD를 적용, Rank를 계산
    - 이미지에 따라 Feature map output은 당연히 달라지므로, 그때마다 SVD rank 개수는 달라지는 것 아닌가?
        - 각 다른 Batch 이미지들로 Feature map output을 계산, Rank를 구했을 때, 차이가 없음을 실험적으로 보임
    - Rank 계산을 구현할 때는 `torch.matrix_rank()` 함수를 사용하면 된다.
    

### Unstructured Pruning

- 파라미터 각각을 독립적으로 Pruning
- Pruning을 수행할 수록 네트워크 내부의 행렬이 점점 희소(Sparse)해짐
- Structured Pruning과 달리 Sparse Computation에 최적화된 소프트웨어 또는 하드웨어에 적절한 기법
    - Sparse computation을 지원하는 소프트웨어 혹은 하드웨어에서만 속도 향상이 있고 그 외에는 없음
    - 왜나하면, 각 레이어의 일부 가중치만 0이 될 뿐이고 실질적은 레이어 개수는 변함이 없음
- The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks(ICLR 2019)
    - Prune된 sparse한 모델들은 일반적으로 초기치로부터 학습이 어려움
    - Lottery Ticket Hypothesis
        - Dense, randomly-initialized, feed-forward net은 기존의 original network와 필적하는 성능을 갖는 sub networks(winning tickets)를 갖는다.
    - 이 Lottery Ticket(sub network)을 찾는 한 방법을 제안함
    - Identifying winning tickets
        1. 네트워크 임의로 초기화 $f(x;\theta\_0)$
        2. 네트쿼으 j번 학습하여 파라미터 $\theta\_j$를 도출
        3. 파라미터 $\theta\_j$를 $p$%만큼 Pruning하여 mask $m$을 생성($p$는 보통 20%)
        4. Mask되지 않은 파라미터를 $\theta\_0$로 되돌리고(초기화하고), 이를 winning ticket이라 지칭 $f(x;m \odot\theta\_0)$
        5. Target sparsity에 도달할 때까지 2-4 반복
- Stabilizing the Lottery Ticket Hypothesis(arXiv 2019): Weight Rewinding
    - LTH의 경우 데이터셋 또는 네트워크의 크기가 커졌을 때 불안정한 모습을 보임
    - k번째 epoch에서 학습한 파라미터로 네트워크를 초기화 하면 학습이 안정화 됨을 보임
    - 논문에서는 총 iteration의 0.1% ~ 7% 정도의 지점을 rewinding point로 언급
- Comparing Rewinding And Fine-tuning In Neural Network Pruning(ICLR 2020): Learning Rate Rewinding
    - Weight rewinding 대신, weight는 그대로 유지하고, 학습했던 Learning rate scheduling을 특정 시점(k)로 rewinding하는 전략을 제안
    - 어느 시점의 weight를 rewind할지에 대한 파라미터 고민 없이, Learning rate를 0으로 설정한 후 다시 학습하면 대체로 좋은 성능을 보임
- LTH 관련 기타 추천 논문
    - 왜 잘되는지 아직 명쾌히 설명되지 않음(but RL, NLP 등등 다양한 분야에서 적용이 가능함)
    - 특정 initialization에 대해서, 학습 후에 도달하는 공간이 유사하다? → "네트워크 학습 및 수렴" 관련 연구와 접목되는 트렌드
- Linear Mode Connectivity and the Lottery Ticket Hypothesis(ICML 2020)
    - 네트워크의 학습 및 수렴 관련된 실험
    - 특정 학습 시점(epoch at 0, k)에서 seed를 변경하여 두개의 Net을 학습 → SGD를 통한 계산 결과가 다르므로, 다른 곳으로 수렴
    - 둘 간의 weight를 Linear interpolation하여, 성능을 비교
    - 두 weight 공간 사이의 interpolated net들의 성능을 확인
    - 특정 시점부터 네트워크가 수렴하는 공간은 한 plateau 위에 있는것이 아닌가?라는 해석을 제시
- Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask(NeurIPS 2019)
    - LTH는 $w\_f$의 L1 norm으로 줄을 세워서 masking하는데 $w\_i$를 고려하면?
    - $w\_i, w\_f$를 함께 고려하면?(NLP 쪽의 movement pruning처럼)
- Pruning at Initialization(unstructured)
    - Train이전에 "중요도"를 보고 Pruning을 수행하자. 그 후 학습하면 시간이 훨씬 절약된다.
    - 중요도는 어떻게 계산?
        - SNIP(ICLR 2018): Training 데이터 샘플, Forward해서 Gradient와 Weight의 곱의 절댓값으로!
        - GraSP(ICLR 2019): Training 데이터 샘플, Forward해서 Hessian-gradient product와 Weight의 곱으로!
        - SynFlow(NeurIPS 2020): 전부 1로 된 가상 데이터를 Forward해서 Gradient와 Weight의 곱으로!
    - 네트워크의 가능성을 본다 → Training Free NAS/AutoML Zero-const proxies for Lightweight NAS(ICLR 2021)
        - Pruning at initialization의 기법들을 일종의 score로 network를 정렬
        - 각 기법의 score와 실제 학습 결과의 상관관계(Spearman 상관계수)를 확인
        - 생각보다 score와 실 학습 결과의 상관 계수가 높고, voting을 하면 더욱 높다
        - Synflow가 CV외 여러 task(NLP, ASR)에도 높은 상관계수를 보여줌
- 아쉬운점
    - Structured/Unstructured 모두 간단한 Architecture로 벤치마크를 수행(Modern architecture는 잘 다루지 않음)
    - 다양한 Architecture에 대해서 여러 예외처리, 추가 검증이 필요함
    - 파라미터 수, FLOPs로 비교를 많이 하는 편이나 실질적 latency 향상 정도와는 차이가 있음
    - Unstructured는 특히 아직 가속되는 HW가 많지 않음

## Knowledge Distillation

### KD의 핵심

> Teacher의 정보를 어떻게 빼내는가?
> 

### Response-Based Knowledge Distillation

- Teacher model의 last output layer를 활용하는 기법, 즉 직접적인 final prediction을 활용
- 대표적으로 hinton loss가 있음

### Feature-Based Knowledge Distillation

- Teacher의 layer의 중간 중간의 intermediate representations(feature)를 student가 학습하도록 유도
- $F\_t(x)$,  $F\_s(t)$를 각각 teacher, student의 feature map이라 하고 $T_t$, $T_s$를 각각 techer, student의 transformation function이라 하자.
- Distance $d$가 있을 때, feature based KD의 loss는 아래처럼 정의된다.
    - $L\_{distill} = d(T\_t(F\_t), T\_s(F\_s))$
- 가장 간단하게는 $d$은 L2 norm, $T\_t$는 identity, $T\_s$는 learnable한 linear transformation matrix로 정의한다.
    - feature channel 수가 보통은 teacher가 많으므로, 이를 맞춰주기 위해 도입
- 논문들의 주된 방향은, Feature distillation 과정에서 유용한 정보는 가져오고, 중요하지 않은 정보는 가져오지 않도록 Transformation function과 distance, 위치 등을 조정하는 것
    - 중간 결과를 가져오므로, network 구조에 크게 의존함

### Relation-Based Knowledge Distillation

- 다른 레이어나 Sample들 간의 관계를 정의하여 Knowledge distllation을 수행
- Relation among examples represented by teacher to student

### 주의

- 재현이 잘 되지 않는다는 주장을 하는 논문이 다수 존재
- Contrastive representation distillation, ICLR 2019
- NLP의 경우 Transformer, Transfer Learning이 main stream이어서 KD 연구가 활발

### 추가 논문

- 크기가 문제가 아닌 성능이 문제라면?
- Self-training with noisy student improves imagenet classification(CVPR 2019)
    - 아래와 같은 방식을 사용
        - Train a teacher model on labeled images
        - Use the teacher to generate pseudo labels on unlabeled images
        - Train a student model on the combination of labeled images and pseudo labeled images
    - Student를 다음 iteration에서의 teacher로, student는 유지하거나 점점 키움
    - Unlabeled 데이터를 사용하지만, 당시 ImageNet SOTA 기록
    - KD와의 차이점?
        - Noise를 추가(input noise: RandAug, Model noise: dropout, stochastic depth function)
        - Student가 점점 커지는 framework
        - Soft target을 사용한다는 점에서 KD와의 유사성 존재