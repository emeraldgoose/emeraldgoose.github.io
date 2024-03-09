---
title: BERT 경량화
categories:
  - BoostCamp
tags: [bert, lightweight]
---
## Overview

### CV 경량화와 NLP 경량화의 차이점?

- Original task → Target task로의 fine tuning하는 방식이 주된 흐름
- 기본적으로 transforemer 구조
- 모델 구조가 거의 유사해서, 논문의 재현 가능성이 높고, 코드 재사용성도 높음

### BERT Profiling

### Profiling: Model size and computations

- Embedding layer: look up table이므로, FLOPs는 없음
- Linear before Attn: k, q, v mat 연산으로 linear after attn 대비 3배
- MHA: matmul, softmax등의 연산, 별도 파라미터는 없음
- FFN: 가장 많은 파라미터, 연산횟수
- 효율적인 GPU 연산을 위해 단독으로 CPU에서 사용하는 메모리보다 소비량이 더 큼(연산속도를 위해 CPU, GPU에 동일 텐서를 들고 있거나 등등)
- MHA 파트는 이론 연산 횟수 대비 속도가 매우 느린데, 이는 여러 연산의 조합(matmul, softmax) 때문인 것으로 보여짐
- Linear Layer에서는 GPU가 빠른 속도를 보여주나, CPU는 이론 연산 횟수와 유사한 경향을 보임
- FFN 파트가 모델의 주된 bottleneck

## Paper

### Are Sixteen Heads Really Better than One?(NeurIPS 2019)

- MHA는 구조상 복합적인 정보를 담아두기 위해 제안됨.
- 실험에서 많은 헤드가 지워지더라도 성능에 영향을 주지 않았음
- Attn을 Pruning하여 17.5% 속도 향상을 이룸
- Are All Attention Heads Important?
    - 헤드를 하나만 남겨두고 나머지를 지운 후 성능 드랍을 기록하는 실험
    - 눈에 띌 만큼 성능드랍이 없었음
- Are Important Heads the Same Across Datasets?
    - 하지만 역할 수행에 꼭 필요한 헤드가 있다.
    - 다른 데이터셋에서도 같은 헤드가 중요한가? → 중요한 헤드는 다른 데이터셋에서도 중요한 경우가 많다.
- 지금까진 특정한 layer에서 head를 제거하여 효과를 확인했다. 여러 layer에 대한 여러 head를 제거했을 때는 어떤 현상이 발생하는가?
- Iterative Pruning of Attention Heads
    - Attn 헤드를 중요도를 간접적으로 계산한다.
        
        ![](https://lh3.googleusercontent.com/fife/ALs6j_FLysvHedgXmhA37J6Se4X9s3w5FPWzDqewkYchqESlpjYlnofuuEN0URJqq4ztppdU3GxfQjD8rlpdnKqPn6LeEbnaZIufLJUUyEnEKU9NMur--HwlXW0BL3Vfvax_q0xDseH0YIJw48y6ym9j8fdz6Z21hIWohkBfIiNWVX8dt2ToEm0oshYszA3W_evHKiEbs3iKcNXDqFQ5A80pBJzexBipKkNybjhxzPRw1Se33gtKfJoJZ-eHFYlQrEcuJlxCR7JbFN49HUQlocnQcNlWd3yc2Nr2We8dqQszd0Wi3iF0aJRu4BHcZ7kfFfAJxaN8g0QEAXxZ1IHIM6ZQMNLCL30I0N4-YbD2roqn91WM_Hp5YK641ShDwUkeX9GnTSxLZOByoEf8i-TIaOMcB72Q1TLfPU0awdvPH4YXOp_x1ZYtuk_uuI9W9qfUfBfvCq44TCq65mIYcQSWlVKm2TxraLTzHUKAOGlzLfFnBPAAqyu3fX07uHqqSogyWEJt3aGRZT_fHqXTYSkib4OQeq8dVPUZaC8ZjU0wxEYP6yQpHUb5fl28CkzwgrHR7HZ3FI5mDzl7EhJ8o1nlSXOg1ZYOO1leulnZ3o_xXpdT14UiqfoyR4ukVeWjqG4a_sAy3mVM_JyqOMTfAHJ7ZNU6UYgx2-Io1LX9HZa2LLM0Qx0HRrtO9awPUPoALkFx48Yxq7GfFRqEdLPEqxKh9HignMs3RXXBAQxxZsy8403XuCqKgIG2FKxwCFzT5LWOAKcWljHGeYkjePjBePQUwWiQoUodjN9r2PJzqBUlrYUvYDaCLtaMVv9XXkAV1QMKJO_mZDkmRHz3jcJVSBNFAPmC9AmSmiqvQ-dQdj62SaygGBHJtxLNAmgb1H93QPpGH48ImdBrscIi98rnnfZhiOUJ1CgBeXjsZUQ7EXd53hb-n03cy59bgc-qYqSwCED2fXU67Cvwl4LvPanv41VPV8yX83XUrh9y8z0b3Sokh87Zfj9FPNgscmcaiSbLATsWl7APagwfIfGgyjAlHQdLe9Ho90ibV0tCWvwpyTWOx0l0Rfjlu57HL8jptgpk7Z9-heKEKV0KGp5We9J7LuARUJeggoTU6Xv-Enxtll3cSvfPFWBZ9ijFBoRQFNMkzIYOwWfWiKQ5FSJIV5ENzGYmJ9aLKokPakA4SnybFUr4QKS0EttjnrfEBGJz_bsRDRHaxZbBI1-9QxdXLv_9f7l3Y8lzTlBJ9RKqcGDgomT9JtLMaNeeSNT0u0iE5KSe-h9FD38d_ZHZXiExQNJ4MWAUrsYd4e8f9q_XxcsAz2aSfz--y7UPCHwJ7xt8CXsZk6bWkcx3i6YzWlWsXylw6mJqTMrldV8Z-DOPeY6J4NldBkoKBJgEfAGK1UmO3-jCbQrNaBqkEZPMO2Uc6zKcwwbTLDB1mMuubRCAFIHw3RSaF7KnIMQlNMlWhn5J8dSXBqm3dx_owFNv0lxeTl-HIomY9-mZIEkaSutuPeoADa-URIex4vsvnGyq0SXlI8NH1ZROS9atKPKNmspwgkwze6V_MmeJ4sCbyuHAxHTISETW1-x6tGLV8F_E9QYZ4NnScaD3mu_1ykW1DHs-yddOTyiCAAQhj9gvJ_wHKyNtmUrS6uCDuLGoe1Pysg)
        
        - $\xi_h$ : mask variables, 0: masked(removed), 1: unmaksed, $X$: data distribution
    - 데이터샘플에 대해서 특정 헤드를 살린 것에 대한 Loss와 해당 헤드를 지운 것에 대한 Loss의 차이를 뒤의 식으로 approximate하여 사용
        
        ![](https://lh3.googleusercontent.com/fife/ALs6j_F8Cqr6c8SuKNcMSi-1fp-hT2XNmTRJt_GD60Q4mrCAwPBVB-S_oXTwCXKIA2VnT-vnaKKFSViUuykzzv0n-61e60yrilV71QVkhLgnoFYHc27wf8WlHbiaOUk5ZNgElWX0LUpMn4C4o50gDyhF7KkPQqUcjaQpIU6leHDEH5LebEWEHIu4h3aLFRXepkillz4lHdYiFq0le7U7EDwzL_KcF4dpU7hU1pMrh7Engzbo46E-ePQ8gYgSNWWH2Y_GJdiqn8T05tArjUlowE2Ip-ivC98Bjj7WV8dhvqfA0BuC-txlRB-98fS3238hRScmN13jTfWMHi0lPvBh9hvgN2iadqbGUYJTnek-FRxsIOAHxEP3ha1NYkGrqtblw-gxupKRdBIB9H1GrobI7yooZAd2vR68NAjT0aGaG-4CiQnQds9Bu4_PZXb5Mmeql0XUYCs5FXklEJsR-mm0dIj-y95JSWXzVpMTAfEQe3hY7gy6YrYW27CUivaMLKoBb6LPJkQP1JHKNGMmnu6GwRTbDXlpWZlPn954T3V_07pU3Y9mIAnVt9Y01OR0xnRnKnOzb2BrOv5rLipJ2ljV5bTcPAqm-uZTJCkn0VtFCiLjvI1uoYJZGnYrE6-QgrI9aRf7QLXH8GXKCm6YASRJ1OSLcrY7AXBU_Zsx43U0l_7XHuesTBR89NX9PzXHvwEQBrq0feTEIfIxERL2nmD6Fw3spufBzz3aX2IX6ihsayUodbF97PENOjvDjkkuiojxY2vFZqbGKhxLkoeFrN8xXMUZteKliHjrYfiX0EqvbiTRN_PSPchVj82KqjOGbq4tpRGPttvlxKg4gaY3Hn_TbHWwtBZfqk5C5fO7BKAfAjg96ZiWg4hmGjI0rAIu6fDrXaDhP4JZR6jYjCQz7SZ1STU3D8gsmVMwY1WDqrSTE2YAh5hr5rMA4MXO5fNcGj3qMm6h1aP-QDZ5LLzoe9QbbcdojXwd4tAYIV0DtDExuWu_Bz_vdtxCGKHVsw3KreDWhoh-DVcRDlZL2c8Wm99qdQo473z5njGYbVzTLuiwVyBFb3wI8ABGNtxwBQ5YDwx3BrVIaLs1UqLGNcnxLwHuN_UkzNENpx1zB5BTk_WoeNmQl269L3vYZ0Q32uinhmFn9umU2uWSh_YMBoZvTBT8DlZxpfnBe0XFS6d8ZI1ZreYIMxTC3gq4sWmxBUwROB_40C_LEMEEt5hFKRFW_csXtBjlo8oIEpGG6kDnYylgiuDxLG5Z361_XB7PHRypOYKL7HemdCBnsXttoiV1fDv0kURs3mdwe8oQz1UumtCFYIwf7qa_AKm6KFJ6K8mHJcqXm2lHBrc6uSyev6gyB5H3nkoPLTjrp0wmL1qP7odws_z_2c4XjJt0W-FHJfVypv0qrrEES4F2K4HeJ-dX06BEcHPZ7mCdX03kTOhN4Shut7HWSlNkESUixHkT8Gelg2p6-wdXp9ecHriHhbmST6LuW1v-fASgw-e14cKMyA3qKiqopIexgMA84UaE40xxR2bc5EOWaqfOGJ4VPXsZzvEzzTCGV-B-WACNuE4TZRAYTVBdIPPRDprg18idD-hNTRcBTorh3jZpuIIo7wWOEbnArMh8h2CPD_JMpbBHA8iCSZKiwZEQrpjy6w)
        
- Effect of Pruning on Efficiency
    - 배치사이즈가 작은 경우 효과가 좋지 않지만 큰 경우 효과가 좋았음

### Movement Pruning: Adaptive Sparsity by Fine-Tuning(NeurIPS 2019)

- Transfer learning에서는 Magnitude pruning의 효과가 좋지 않음
    - 왜냐? original task에서 target task로 transfer learning 과정에서 Weight 값의 변화가 그리 크지 않음
- 값이 큰 Weight은 그대로 값이 큼 → Weight 값은 많이 변하지 않음
- Original model에서의 큰 Weight은 Original task에서 중요한 의미를 갖는 Weight일 가능성이 큼
    - Fine-tuned model에서 큰 Weight은 Target task에서 중요하지 않은 Weight일 수도 있음
- Magnitude Pruning에서는 Original task에서만 중요했던 Weight들이 살아남을 수 있음
    - → Movement Pruning : Transfer Learning 과정에서, Weight의 움직임을 누적해가며 Pruning할 Weight를 결정하자!
- Background: Score-Based Pruning(Unstructured pruning formulation)
    
    ![](https://lh3.googleusercontent.com/fife/ALs6j_H8SRN7Q2zHPE58RpdJ1bfa92PpNpN35ABlTXbxF9GAgctlmN7Kj3VzUA27wF-5mgzrM2RXsk0XBhnQwT6zeYs_lY4GKwhCdnqV-W3LSuxwzpgcKJH6O_lyTsgtOccG8QW_iEnjOehNLk0J4fyeQ6epMwwhJfLifB9jzTF6E-z7xpDq9x1Xj5aA7v3eZF8GBA-QGlOFIDUXynyRdBvbK3eS3eBEHByBq2Ov-92_XhC3Zu_HXCOxULTz-9Q-M6pdEXGXFpNwrku6-u-YLhdYP_uBxl4gDwI0ihp6DuOv97Lhqo4eQt_H-LVHLBfTpt6z-oE6nBd-9rx_9WbEMhhm4jW8GXD_Og-PuhgyqhpkbE8AnL236e0uMLPpzujB_2MJbKq0tLeIbcBlTlui7YFyYXI-KX7TlO3lntwWIPq-3wq9I27H4wwnKeejTfnJjVi65u9hObLBGlLZnaMOmORy5DE7fhCoXFvFeNaL4VQCFCsPnkuxtkJZICx8CemMSvNYuWhevmr-zXlvkUvbC18a30xPqnJzPHaERjBYYa5xohCBgdvlkYEyyXOv2_zBdQmgzRsREFbDvwTYan81qHt6e7mRD6n_d8dic5WVhtDQe26U_QCs8GenL6amEsGo7wa5ZyeZR_itFHUslsrj5rQPtxE0hHIjiCvRsJzKmfLLxylEjmms82CNt0glZ1PagfaOmFd9BYDLoxY858qxIklYacUXvDuPfQdSkdnstLt25lNyaUrqJvEihA0HZkogGtY6Z59JTN3fgYZMQWgj9MYkY0V96e63ooNx6baa3z38JrCuiZeladRHKw_DVCFU6PFNHQnCmZlZ5a3W7mLtS7uV0CFmGgfXCNZsCG98JEYKHpioVkGtH9KIrtmjJSO-6iKYydaqcJWbjSPQkBCsMeeDQprUL5N3j1YhdbM_pvn7BRejRsJ9cwRoTQYlIAlcNAlw1TI3av2BAswkSMI7Cmz9L6eVXvIoWeHR10dL4Ne2bliuxHr4Xl7x2GtVMbdD5rGG-YYi-l-vZVnN9P87JRYJIa3gmV2-iSFkBW9Yx2__SjsxfFNOzlhC2OD-O2ySp7orOEc_Z0VtMESDm-YEBwAzAKkP9b3LhPUVERv4a0qciFHvwSIpWKvXJrJpICVUnHemd4aSP6uuXxt2xbE8o8-vlaqPHB0ktp5PiVcAKRzm7UJPdkreQrvK9kcfRyN7kknjpNFi1l4IubvNvfru9j1qIW-tQ7Lxgq7fQRCtEZtQaNYhM1TOKn5O_gBOKz4BaJaw51XEfXbZHbMGYE-h-J55KpP7V4XtsShlIAH7laYjNWlOCJqQWeMWO3X0cPqpLIMzfMWP953pigcRYwXQqwL7LD4SOsRWyWEch_bYRK5oQyHbreo-rkSn9s6siOYH3THL8Vf_tgBZoDnAcmEhNJSniIOYmgfCFUvOkbSXmh8JyHXAgQH7vvWv-BUKfYvVIRfaf89wUGQiFoo1QCiZpPzg4XBu5lBCuJoI7C2QnCfiQVSViXV9wlYO_udBTupHVfUlkZNP4ANCUE3MKKChO-DelivE_7HnO4B39XY0w3Zpt-L9JIW9NGh3FNfhWtNo9vfQWCUFvghu9pya9CcHFowTEIZCGRW8ItnushpIOVTj8VRiuQSOlg_)
    
    - $S = (\|W\_{i,j}\|)\_{1<i,j<n}$
- Method Interpretation
    - 0에서부터 멀어지는 weight를 고르는 방법
    - Movement Pruning의 score 유도
        - Masks are computed using the $M = Top\_v(S)$
        - Learn both weights $W$, and importance score $S$ during training(fine-tuning)
        - Forward pass, we compute all $i, a\_i = \sum\_{k=1}^n W\_{i,k}M\_{i,k}x\_k$
        - Forward 과정에서, Top에 속하지 못한 나머지는 masking이 0이 되어, gradient 값이 없어진다.
        - straight-through estimator(Quantization function의 back propagation에서 자주 사용되는 Technique)를 통하여 gradient를 계산
        - 단순히, gradient가 "straight-through"하게 $S$로 전달($M$ → $S$)
        - $S$의 변화에 따른 Loss의 변화는 아래와 같이 정리
        - $\frac{\partial L}{\partial S\_{i,j}} = \frac{\partial L}{\partial a\_i} \frac{\partial a\_i}{\partial S\_{i,j}} = W\_{i,j} x\_{j}$
    - Movement Pruning의 score 해석
        - Gradient descent를 생각해 볼 때($w = w - \alpha \frac{\partial L}{\partial w}$, $\alpha$ : lr rate)
        - $W\_{i,j} > 0, \frac{\partial L}{\partial W\_{i,j}} < 0$이면, $W\_{i,j}$는 증가하는 방향(이미 양수에서 더 커짐)
        - $W\_{i,j} < 0, \frac{\partial L}{\partial W\_{i,j}} > 0$이면, $W\_{i,j}$는 감소하는 방향(이미 음수에서 더 작아짐)
        - 즉, $\frac{\partial L}{\partial S\_{i,j}} < 0$의 경우의 수 두 가지에 대해서 모두 $W\_{i,j}$의 Magnitude가 커지는 방향
        - $W\_{i,j}$가 0에서 멀어지는 것($\frac{\partial L}{\partial S\_{i,j}} < 0$) → $S\_{i,j}$가 커지는 것($S\_{i,j} = S\_{i,j} - \alpha \frac{\partial L}{\partial S\_{i,j}}$)
    - Score는 Weight가 fine tuning 되면서 함께 학습
        - 기존 score를 계산하는 방법에서는 잘못된 에러에 대해서 수정할 기회가 없었다.
        - 여기에서는 학습되는 과정에서 score를 계산해서 학습이 진행되면서 일종의 self-correction하는 효과가 있다.

### Pruning 추천 논문

- Encoder의 각 위치별로 어떤 Knowledge를 가지고 있는가?
    - On the Effect of Dropping Layers of Pre-trained Transformer Models
    - Pretrained information(general linguistic knowledge)는 input에 가까운 encoder들에 저장되어 있고, head에 가까운 부분들은 task specific한 정보를 저장한다.
    - pretraining 모델에서 head 쪽 레이어를 없애도 fine-tuning 시의 성능이 크게 떨어지지 않는다.
- Pretraining fine-tuning paradigm이 왜 성능, generalization capability가 더 좋은가?
    - Visualizing and Understanding the Effectiveness of BERT
    - 사전학습 모델을 fine-tuning하는 과정에서 loss surface가 평탄하기 때문에 학습이 더 잘되고 generalization capability가 더 좋다.

## Weight Factorizatino & Weight Sharing

### ALBERT: A Lite BERT for Self-supervised Learning of Language Representations(ICLR 2020)

- 큰 모델들이 SOTA 퍼포먼스를 얻게 되는데 메모리 한계때문에 사이즈를 크게 키우는 것에 한계가 있다.
- 또한, distributed training으로부터 Communication overhead가 존재한다.
- 이 논문에서 제안하는 세 가지 방법
    - Cross-layer parameter sharing : 파라미터 수 감소
    - Next Sentence Prediction → Sentence Order Prediction : 성능 향상
    - Factored Embedding Parameterization : 파라미터 수 감소
- ALBERT는 모델을 효율적으로 만들어서 더 큰 모델을 만들자는 목적

### Cross-layer parameter sharing

- Weight sharing은 input, output embeddings의 L2 dist, Cosine similarity를 계산해봤을 때 network parameters를 stabilizing하는 효과가 있다.

### Sentence Ordering Objectives

- Next Sentence Prediction(NSP)가 너무 쉽기 때문에 Sentence Ordering Object(SOP)를 수행하도록 함
- NSP loss는 SOP 수행에 도움을 주지 않지만 SOP loss는 NSP 수행에 도움을 줄 수 있다.

### Factorized Embedding Parameterization

- WordPiece embeddings($E$)는 context-independent 표현을 학습한다.
- Hidden layer embeddings($H$)는 context-dependent 표현을 학습한다.
- BERT에서는 $E$와 $H$의 dimension이 같다. → BERT는 context-dependent 표현의 학습에 효과적인 구조.
- 그렇다면, 왜 BERT 레이어가 context-independent representation인 WordPiece embedding에 묶여야 할까?
- WordPiece embedding 사이즈 $E$를 hidden layer 사이즈 $H$로부터 풀어내자.
- Untying dimensions by using decomposition
    - 원래 $O(V \times H)$를 $O(V \times E + E \times H)$로 파라미터 수를 줄임

## Knowledge Distillation

### DitilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter

- 세 가지 loss를 제안
    - 기존 masked  language loss, distillation(Hinton) loss, cosine-similarity loss
- Triple loss = MLM($L\_{mlm}$)+Hinton($L\_{ce}$)+Cosine embedding($L\_{cos}$)
    - Masked Language Modeling Loss : CE-loss
    - Distillation(Hinton) Loss : KL div of teacher, student softmax prob with temperature
    - Cosine embedding loss : between teacher, student hidden state vectors
- Student architecture & initialization(paper)
    - token-type embedding, pooler 제거
    - 레이어 개수를 절반으로 줄이고 hidden size dimension은 그대로 두었음(dimension을 줄이는 것이 computation에서 큰 효과가 없었다.)
    - 초기화는 student의 첫 번째 레이어는 teacher의 두 번째, student의 두 번째 레이어는 teacher의 네 번째 레이어에서 가져오는 것으로 한다.

### TinyBERT: Distilling BERT for Natural Language Understanding

- Transformer distillation method: 3 types of loss
    - From the output embedding layer
    - From the hidden states and attention matrices
    - From the logits output by the prediction layer
    - Teacher $N$ layers, student $M$ layers → N개에서 M개를 고른다.
    - Teacher와 student의 레이어 맵핑을 하는 $n = g(m)$이라는 함수를 정의한다.(논문에서는 g(m) = 3 *m으로 정의했고 1번 레이어를 3번 레이어로 맵핑한 의미이다.)
    - $L\_{model} = \sum\_{x \in \chi} \sum\_{m=0}^{M+1} \lambda\_m L\_{layer}(f\_m^S(x), f\_{g(x)}^T(x))$
        - 학습데이터 $x$에 대하여 Student layer $m$마다 각 레이어의 로스를 구하고 각 레이어의 중요도 가중치 $\lambda$를 곱한 것의 합
        - Layer loss는 $m$번째 student의 특정 output(Attn, hidden, logits), $g(m)$번째 teacher의 특정 output(Attn, hidden, logit)
        - $m = 0$ 일때, $L\_{layer}$는 $L\_{embd}$, $0<m≤M$ 일때, $L\_{layer}$는 $L\_{hidden}+L\_{attn}$, $m = M+1$ 일때, $L\_{pred}$
    - Transformer-layer Distillation(Attention based)
        - $L\_{attn} = \frac{1}{h} \sum\_{i=1}^{h} MSE(A\_i^S, A\_i^T)$
        - A는 teacher 또는 student의 attention matrix인데 이 논문에서는 unnormalized attention matrix로 설정함
            - unnormalized가 빠르고 성능이 더 좋았음
    - Transformer-layer Distillation(Hidden state)
        - $L\_{hidn} = MSE(H^SW\_h,H^T)$
        - $H^T$, $H^S$ : teacher hidden state, student hidden state
        - $W\_h$ : Learnable linear transformation(Student의 hidden state를 teacher의 dimension만큼 키워서 MSE를 계산하기 위함)
    - Embedding-layer Distillation loss
        - $L\_{embd} = MSE(E^SW\_e,E^T)$
        - $W\_e$는 $W\_h$와 같은 역할을 한다.
    - Prediction-layer Distillation loss
        - $L\_{pred} = CE(z^T/t, z^S/t)$
- Two stage learning framework
    - General Distillation : Large-scale Text Corpus → General TinyBERT
    - Task-specific Distillation : General TinyBERT → Fine-tuned TinyBERT

### 기타 논문 추천

- MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices
- Exploring the Boundaries of Low-Resource BERT Distillation
- AdaBERT: Task-Adaptive BERT Compression with Differentiable Neural Architecture Search

## Quantization

- 장점 : low memory footprint, inference speed(low precision operation 증가 추세)
- 주로 Quantization 과정에서 발생하는 accuracy drop을 줄이는 방향에 대한 연구
- QAT(Quantizatino Aware Training), Quantization range 계산 방법 제안 등

### Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT

- 민감도가 높은 layer는 큰 precision, 낮은 layer는 작은 precision
- group-wise quantization이라는 새로운 방법 제시
- BERT의 bottleneck 조사
- Hessian spectrum(eigenvalues)
    - Higher Hessian spectrum을 가지는 NN layer에서의 파라미터들은 더 민감함(Hessian의 top eigenvalues의 크기가 해당 레이어의 민감도와 연관이 있다)
    - Hessian spectrum 계산의 복잡성 → power iteration method로 해결(Large sparse matrix에서의 빠른 수렴)
    - 같은 데이터셋이더라도 Hessian spectrum의 var이 매우 큼 → mean, std를 함께 고려하여 민감도를 정렬
- Group-wise Quantization method
    - key, query, value, output 모두 같은 Quantization range로 정했었는데 이 range가 커버해야 하는 matrices의 단위가 너무 크다.
    - query, key, value의 분포가 다를 수 있는데 이때 에러가 커진다.
    - 따라서, Multi-head 별로 따로따로 quantization을 진행함

## 정리

### Pruning

- Structured : 모델 사이즈 감소, 속도 향상, 성능 드랍
- Unstructured : 모델 사이즈 감소, 적은 성능 드랍, 속도 향상 X(별도 처리가 없는 경우)

### KD

- 파라미터 수 감소, 다양한 range 모델, 큰 속도 향상 여지(LSTM 등 다른 구조로 distillation), 비교적 복잡한 학습 구성, code maintain

### Weight

- Matrix decompostion(factorization) : 모델 사이즈 감소, 적은 성능 감소, 속도 향상(주로 CPU), 속도 변화 미미(GPU)
- Param(weight) sharing : 모델 사이즈 감소, 학습 관련 이점, 적은 속도 개선

### Quantization

- 모델 사이즈 감소 탁월, 적은 성능 하락, 속도 향상 불투명(지원 증가 추세)