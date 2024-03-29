---
title: Wordpiece Tokenizer
categories:
  - nlp
tags: [nlp]
---
## Reference
> Wordpiece 토크나이저는 BERT를 사전학습할때 사용했던 토크나이저입니다. BPE(Byte-Pair Encoding) 토크나이저와 방식은 거의 똑같은데 단어를 합치는 부분이 다른점이 특징입니다. 

> [https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt](https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt)

## WordPiece tokenization
BPE와 동일하게 모든 단어들은 알파벳으로 나눈 후 머지하는 과정을 진행합니다. Wordpiece는 머지할 두 단어를 선택하는 방법이 BPE와 다릅니다. Wordpiece는 다음과 같은 공식을 통해 머지할 pair를 선택합니다.

- score = (freq_of_pair) / (freq_of_first_element X freq_of_second_element)

위의 공식을 따르면 pair를 구성하는 각각이 덜 등장하지만 pair 등장 빈도가 높은 경우 머지할 우선순위가 높습니다.

## Get Vocabulary
vocabulary를 얻기 위해 corpus를 모두 단어 단위로 분리합니다. 각 단어의 첫 번째 알파벳을 제외한 나머지 알파벳에 대해 "##"인 prefix를 붙입니다.

<script src="https://gist.github.com/emeraldgoose/e6694089638b13e235ed5f0acf86be27.js"></script>

각 단어들을 분리한 splits를 `compute_pair_scores` 함수로 score들을 계산합니다. 

<script src="https://gist.github.com/emeraldgoose/d7928e87b4cf9643816c527c44b0aa90.js"></script>

`merge_pair`라는 함수를 사용하여 머지가 필요한 쌍을 머지하고 splits에 저장합니다. 지정한 vocab_size가 모두 채워질 때까지 반복합니다.

<script src="https://gist.github.com/emeraldgoose/b16d32e9f311f286d6d8efcfcb2b22d1.js"></script>

## Tokenization
문장내 각 단어들을 vocab에 있는 단어에 따라 토크나이징합니다. 인코딩에서는 첫번째 글자부터 가장 큰 subword를 찾아 분리한 후 반복합니다.

<script src="https://gist.github.com/emeraldgoose/f7007594074af53650898799ecf2179d.js"></script>