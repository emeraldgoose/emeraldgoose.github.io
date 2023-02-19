---
title: Byte Pair Encoding
categories:
  - nlp
tags: [nlp]
---

## Reference
[https://ratsgo.github.io/nlpbook/docs/preprocess/bpe/](https://ratsgo.github.io/nlpbook/docs/preprocess/bpe/)

## Byte Pair Encoding
> BPE 알고리즘에 대한 설명은 위에 링크한 곳에 잘 설명되어 있습니다. 여기서는 참고한 곳의 내용을 바탕으로 구현방법에 대해 설명하겠습니다.

### get vocabulary
토크나이징을 위해 문서내에 등장한 단어의 등장횟수가 기록된 `dictionary`를 사용하여 단어 집합인 `vocabulary`를 만들어야 합니다.

<script src="https://gist.github.com/emeraldgoose/26465d827bd585a22796ba0461b10014.js"></script>
위 코드는 `dictionary`의 단어들을 구성하는 글자들만을 추출하여 `vocabulary`에 저장한 코드입니다. 알고리즘 내에서 `vocabulary`를 사용하고 다시 업데이트를 반복할 것입니다.

모든 단어에 대한 bigram 쌍을 `pairs`라는 딕셔너리에 저장하면서 횟수를 저장합니다.

<script src="https://gist.github.com/emeraldgoose/c2726eaffd762b1498a689bf751745a7.js"></script>

`pairs`에서 가장 많이 등장한 순대로 정렬 후 가장 많은 단어를 `vocabulary`에 저장합니다.

<script src="https://gist.github.com/emeraldgoose/79fc8c2d59f6647c4b8a2ef289c429ae.js"></script>

여기까지의 과정을 반복합니다.

<script src="https://gist.github.com/emeraldgoose/98cf430f0d69833753dc969b5b1560fd.js"></script>

### tokenizing
토크나이징을 하려면 `vocabulary`가 필요합니다. `vocabulary`가 list 자료구조를 이용하는 이유는 등장 횟수가 많은 단어를 순서대로 추가하기 때문입니다. 

<script src="https://gist.github.com/emeraldgoose/200425885cfbf199eb560967cda36768.js"></script>

`<unk>` 토큰은 토크나이저가 `vocabulary`에 없는 단어를 대체할 때 사용합니다. 이런 문제를 OOV(Out of Vocabulary)라 하는데 여기서는 사전의 크기가 크지 않아 발생합니다. 보통 `vocabulary`를 구축할 때는 많은 corpus로부터 생성하므로 발생빈도가 `vocabulary` 크기에 따라 달라집니다.

다시 돌아와서 위의 코드는 hugs를 h,u,g,s로 나누고 h,ug,s로 병합하는 과정이었습니다. 이 과정을 더 이상 병합하지 않을 때까지 반복하여 토크나이징을 수행합니다.

<script src="https://gist.github.com/emeraldgoose/8e75ed76cec6574d11d34296861c4cbe.js"></script>

이 코드를 함수로 정의하고 문장에 대해 토크나아징할 수 있습니다.

<script src="https://gist.github.com/emeraldgoose/10c50f9835dae2d01d0aa0d56ac02991.js"></script>

이렇게 BPE 알고리즘을 통해 subword를 분리할 수 있어 합성어를 사전에 포함하지 않아도 되는 장점이 있습니다. 또한, 사전에 단어가 포함되지 않아 발생하는 OOV(Out of vocabulary) 문제를 해결하는 방법이기도 합니다. 

여기까지 BPE 토크나이징 기법을 python으로 구현해봤습니다.
