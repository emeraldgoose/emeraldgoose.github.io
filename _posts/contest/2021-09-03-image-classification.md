---
title: P1 Image Classification 대회 회고
categories:
  - Contest
tags: [boostcamp]
---
## 프로젝트 목표
---
이번 대회는 인물 이미지를 통해 나이, 성별, 마스크 착용에 대해 총 18개의 클래스로 나누고 이를 예측하는 대회입니다. 저는 앙상블을 만들기 위해 적합한 모델을 찾고 다양한 방식을 적용하여 F1 score를 높이는 목표를 가지고 있었습니다.

## 프로젝트를 수행하기 위한 실험
---
### loss function에 대한 실험

- 실험목적 : 단일 loss function을 사용했을 때와 multiple loss function을 사용했을 때 성능 변화 관찰
- 실험모델 : ViT(vit_tiny_patch16_224 in timm)
    - 통제변인 : batch_size = 32, learning_rate = 0.002, transforms, optimizer = torch.optim.Adam
    - 조작변인 : loss_fn
- 실험관리도구 : Weights & Biases
- 실험내용 :
    - 단일 loss function의 적용 (CrossEntropyLoss, F1 Loss, Focal Loss, Label smoothing)
    - 가중치 $\alpha$를 사용해 $loss = loss_1 \times (1-\alpha) + loss_2 \times \alpha$의 식을 적용해서 실험 진행
        - F1 Loss & CrossEntropyLoss
        - CrossEntropyLoss & Focal Loss
        - F1 Loss & Label smoothing (smoothing=0.1, 0.3)
- 실험결과 :
    - 단일 loss function을 적용한 경우 label smoothing을 적용했을 때 가장 성능이 좋았습니다. Focal Loss와 CrossEntropyLoss는 최종적으로 비슷한 성능을 내주지만 F1 Loss는 일정 Epoch 이후 성장을 멈춥니다.
    - Multiple loss function을 적용한 경우 $\alpha$=0.8일때 F1 Loss와 CrossEntropyLoss를 섞어준 경우가 가장 성능이 좋습니다.
    - Label smoothing은 다른 loss function과 섞은 경우와 단일 loss 함수로 사용한 경우에서 비슷한 성능을 내줍니다. 따라서 Label smoothing은 단독으로 사용하는 것이 오히려 좋습니다.
- 아쉬운점
    - 이 실험에서는 loss 값들에 가중치를 주고 실험을 진행했는데 각각 optimizer를 적용한 후 까지 진행했어야 하는 것을 몰랐습니다. stackoverflow나 다른 논문들을 찾고 실험계획을 했어야 했는데 미흡한 부분이 있던 것 같습니다. 다음에는 관련 내용을 좀 더 찾고 실험을 정교화하도록 하겠습니다.

## 모델 개선 방법
---
1. CNN 모델에서 Transformer 모델로 전환하여 학습시키는 방법
    - resnet, VGG, GoogLeNet을 사용해봤고 VGG와 GoogLeNet은 resnet보다 느리다고 판단하여 대회 초반에는 resnet에 집중했습니다.
    - 그러나 resent에서의 성장에 한계가 보이는 듯하여 transformer로 눈을 돌렸고 ViT 모델을 가지고 여러가지를 적용하여 테스트를 진행했습니다.
        - ViT 모델이 가지는 한계때문에 학습 속도가 매우 느려 학습 속도가 빠른 모델을 찾게 되었습니다.
    - 최근 발표한 MLP mixer모델의 경우 ViT보다 빠른 속도를 가지면서 비슷한 성능을 낸다고 하여 사용해봤으나 확실히 ViT보다 빠르지만 efficientnet과 비슷한 속도이고 efficientnet의 layer가 더 많기 때문에 대회 데이터에서 성능이 확실한 efficientnet에 집중하는 것이 옳다고 생각되었습니다.
2. 데이터 전처리 방법을 Cutmix로 사용
    - Cutmix를 적용하기 전과 후의 성능의 차이는 LB에서 F1 score 0.07의 성능 향상을 보여주었습니다.
3. train dataset과 validation set의 클래스별 분포를 비슷하게 나눠 학습
    - random으로 8:2로 나눈 것보다 성능 향상을 이뤘습니다.
    - stratifiedKFold를 사용하여 n_split=5를 주고 학습을 진행했습니다. 이를 통해 validation set의 중요성을 알게 되었습니다.

## 대회 회고
---
대회를 진행하면서 baseline 코드를 손으로 직접 작성하는 경험을 할 수 있어서 좋았습니다. 덕분에 baseline 코드로 여러가지 작업을 하면서 빈 화면을 pytorch 코드로 채울 수 있었습니다.  
대회를 진행하면서 아쉬웠던 점은 초반에 CNN 모델에 집중한 것입니다. 빨리 눈을 돌려 여러가지 Transformer모델을 사용해보고 분석했다면 좋지 않았을까 하는 생각이 듭니다.   
또한, Validation set에 대한 중요성을 늦게 알게 되었습니다. 제공된 baseline 코드에 stratifiedKFold가 소개되어 있는데 꼼꼼히 확인하고 작성하는 코드에 적용했어야 했습니다.  
특히, autocast가 학습 속도를 높여준다는 것을 대회 마감날에 알게 되서 더 아쉽습니다. 거의 40%의 시간이 줄어드는 효과가 있어 더 많은 실험을 할 수 있었을 것입니다.