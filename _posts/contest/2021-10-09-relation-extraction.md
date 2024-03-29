---
title: P2 KLUE Relation Extraction(RE) 대회 회고
categories:
  - Contest
tags: [boostcamp]
---
## 대회 목적
---
이번 대회는 KLUE의 Relation Extraction Task를 수행을 평가합니다. 문장 내 단어의 관계를 알아내면 knowledge graph를 생성하는데 key역할을 하며 생성된 knowlege graph를 통해 QA나 Summarization과 같은 Task를 수행하는데 도움을 줄 수 있습니다. 저는 이번 대회에서 적절한 성능을 내는 모델과 special token을 넣어 성능에 어떤 변화를 일으키는지 확인하는 실험을 진행했습니다.

## 성능향상을 위한 실험
- 먼저 baseline 코드의 성능을 알아보았습니다.
    - Public LB 62.963 (eval step 2000)

### Preprocessing with special tokens

- s' 는 sentence의 일부분을 뜻합니다.
- Type1
    - [CLS] s' [SUB] subject_word [/SUB] s' [OBJ] object_word [/OBJ] s' [SEP]
- Type2
    - [CLS] s' [ENT] subject_word [/ENT] s' [ENT] object_word [/ENT] s' [SEP]
- Type3
    - [CLS] s' [SUB] s' [OBJ] s' [SEP] subject_word [SEP] object_word [SEP]
- Type4
    - [CLS] subject_word [SEP] subject_type [SEP] object_word [SEP] object_type [SEP] sentence [SEP]
- Type5
    - Type4를 기반으로 하고 [UNK], [unused] 토큰을 각각 10개와 99개를 add_special_tokens로 넣었습니다.
    - Testset에는 Trainset에 없는 단어들도 있다고 하여 [UNK]로 대체되어 생긴 문제라는 가설을 세우고 [unused]토큰을 사용해서 대체하도록 했습니다.
- Type6
    - Type5와 달리 [UNK]는 10개, [unused] 토큰을 200개를 add_special_tokens로 넣었습니다.
    - monologg님의 koELECTRA [깃 이슈](https://github.com/monologg/KoELECTRA/issues/12)에서 [unused] 토큰의 개수가 200개인 것을 보고 [unused] 토큰 99개로는 데이터셋을 모두 커버할 수 없다고 생각하게 되어 200개로 늘렸습니다.
- Type7
    - [CLS] subject_word [SEP] subject_type [SEP] object_word [SEP] object_type [SEP] s' @ * subject_type * subject_word @ s' # ^ object_type ^ object_word # s' [SEP]

### Models

- KLUE
    - Bert-base
    - roberta-base, large
- Multilingul Models
    - XLM-roberta-base
    - bert-base-multilingul-cased
    - XLM-roberta-base와 bert-base-multilingul-cased모델을 선택한 이유는 trainset을 보면 subject_word 혹은 object_word가 한국어가 아닌 영어, 일본어, 한자까지 있기 때문에 multilingul모델의 vocab을 사용하면 모두 커버가 되지 않을까?해서 선택하게 되었습니다.

### Results

- 먼저, 저는 eval_auprc를 기준으로 삼았습니다. f1의 측정 방식에서 no_relation label을 제외하고 측정하기 때문에 전체를 보려면 auprc를 봐야한다고 생각했습니다.
- Type1, Type2, Type3의 경우 모든 모델에 대해 학습이 전혀 되지 않았습니다.
    - Stratified 5-Fold를 사용하여 학습을 해도 모든 각 fold에 대해 testset의 예측결과로 거의 대부분 no_relation으로 출력합니다. 또 다른 특징은 eval auprc가 대부분 70을 넘지 못합니다. 최대 68정도이고 많은 경우 50대를 유지합니다.
    - 따라서, baseline code의 전처리 방식을 따라가면서 word의 정보를 주는것이 맞다고 판단하여 Type4의 실험을 진행했습니다.
- Type4의 경우, klue/roberta, klue/bert에서 baseline보다 좋은 성능을 내줍니다.
    - xlm-roberta-base의 성능은 5-fold에서 eval aurpc 60대를 기록하여 roberta나 bert보다 성능이 좋지 않습니다. vocab의 수가 너무 많아서 optima에 수렴하는 것을 어려워한 것이 아닌가 생각됩니다.
    - 데이터 불균형때문에 label smoothing factor(0.1, 0.3)를 주어서 학습을 진행했습니다. 하지만 오히려 성능이 떨어지는 결과를 얻었습니다. 이후 실험에서는 label smoothing을 적용하지 않았습니다.
- Type5의 경우, klue/bert-base를 이용하여 실험을 진행하였습니다.
    - eval_auprc가 최소 72에서 78까지 다양하게 학습된다.
    - Public LB에서 f1 1~2정도 오르는 효과를 볼 수 있었습니다.
- Type6의 경우, klue/roberta-large를 이용하여 실험을 진행하였습니다.
    - eval_auprc가 최소 80이상을 기록했으며, Public LB에서 폴드 평균을 내면 71, 폴드마다 71에서 73을 기록했습니다.
    - dataset의 Out of Vocabulary가 심한 상태이고 [unused] 토큰의 전략이 잘 먹혔다고 판단되었습니다. 하지만 f1보다 auprc가 낮기 때문에 Priviate에서 점수가 떨어질 것으로 예상했습니다.
    - [unused] 토큰을 250개로 늘려서 똑같은 환경에서 실험해봤으나 [unused] 토큰을 마냥 늘리는 것이 성능 향상에 도움이 되지 않음을 알 수 있었습니다.
- Type7의 경우, klue/roberta-base, larget 모델을 이용해서 실험을 진행하였습니다.
    - Type6를 적용했을 때와 비슷한 성능을 내주지만 넘어서는 성능을 내지 못했습니다.

## 회고
대회 첫 주에 baseline 코드를 이해하느라 시간을 너무 많이 사용했습니다. 두 번째 주가 되어서야 전처리나 다른 모델을 사용할 수 있어서 너무 아쉬웠습니다. 이번 대회를 통해 huggingface를 어느정도 이해할 수 있었고 대회가 끝나면 최적화 방법들을 공부하고 팀에 공유할 생각입니다.  
대회 이후 상위권 팀의 리뷰에서 Type7과 같은 Punctuation 방식을 사용했는데 subject와 object를 앞에 추가하지 않고 성능이 오르는 효과가 있었다고 합니다. 그래서 더욱 아쉽습니다.