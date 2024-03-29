---
title: Probability 정리
categories:
 - BoostCamp
tags: [probability]
---
## 딥러닝에서 확률론이 왜 필요한가요?
- 머신러닝은 어떤 방식에서든지 결국 예측을 수행해야 합니다. 예를들어, 강화학습 방식에서도 에이전트가 할 수 있는 행동들 중에서 보상을 가장 많이 받을 수 있는 확률을 고려해야 합니다.
- 머신러닝에서 사용하는 손실함수(loss function)들의 작동 원리는 데이터 공간을 통계적으로 해석해서 유도하게 됩니다.
- 이 밖에도 많은 부분에서 확률과 통계 내용이 필요합니다. 기계학습을 이해하려면 확률론의 기본 개념을 알아야 합니다.

## 확률분포는 데이터의 초상화
- 우리가 손에 얻게 되는 데이터들은 확률변수(random variable)에서 실현(realized)된 표본(sample)입니다.
    - 확률변수는 분포모형(distributed model)과 모수(parameter)를 가지게 됩니다.
- 데이터공간을 $X \times Y$라 표기하고 $D$는 데이터공간에서 데이터를 추출하는 분포입니다.
    - $(x,y) \in X \times Y$로 데이터 $(x,y)$는 데이터공간 상의 관측가능한 데이터에 해당합니다.
    - 이때 데이터는 확률변수로 $(x,y) \sim D$ 로 표기합니다.

## 이산확률변수 vs. 연속확률변수
- 확률변수는 확률분포 D에 따라 이산형(discrete)와 연속형(continuous) 확률변수로 나뉘게 됩니다.
- 이산형 확률변수는 확률변수가 가질 수 있는 경우의 수를 모두 고려하여 확률을 더해서 모델링합니다.
    - $P(X \in A) = \sum_{x \in A}P(X = x)$
- 연속형 확률변수는 데이터 공간에 정의된 확률변수의 밀도(density) 위에서의 적분을 통해 모델링합니다.
    - $P(X \in A) = \int_{A}P(x)dx$
    
## 조건부확률
- 조건부확률 $P(B \vert A)$는 입력변수 $A$에 대해 정답이 $B$일 확률을 의미합니다.
    - 연속확률분포의 경우 $P(y \vert x)$는 확률이 아니고 밀도로 해석합니다.
    
## 기댓값
- 확률분포가 주어지면 여러 종류의 통계적 범함수(statistical function)을 계산할 수 있습니다.
- 기댓값(expection)은 데이터를 대표하는 통계량이면서 확률분포를 통해 다른 통계적 범함수를 계산하는데 사용할 수 있습니다.
    - 연속확률분포 : $E_{x \sim P(x)}[f(x)] = \int_{X}f(x)P(x)dx$
    - 이산확률분포 : $E_{x \sim P(x)}[f(x)] = \sum_{x \in X}f(x)P(x)dx$
    
    
## 몬테카를로 샘플링
- 기계학습의 많은 문제들은 확률분포를 명시적으로 모를 때가 대부분입니다.
- 확률분포를 모를 때 데이터를 이용하여 기댓값을 계산하려면 몬테카를로(Monte Carlo) 샘플링 방법을 사용해야 합니다.
    - $E_{x \sim P(x)}[f(x)] \approx \frac{1}{N}\sum_{i=1}^Nf(x^{(i)}), \space x^{(1)}\overset{\underset{\mathrm{i.i.d.}}{}}{\sim}P(x)$
    - 몬테카를로 샘플링은 이산형이든 연속형이든 상관없이 성립합니다.
    - i.i.d.는 독립항등분포(independent and identically distribution)로 random variable이 독립적(independent)이고 같은 항등분포(identically distribution)일 때 i.i.d.라고 합니다.
    - 몬테카를로 샘플링은 독립추출만 보장된다면 대수의 법칙(law of large number)에 의해 수렴성을 보장 받습니다.