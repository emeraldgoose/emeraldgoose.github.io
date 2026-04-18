---
title: Byte Pair Encoding
categories:
  - nlp
tags: [nlp]
---
## Reference
> BPE 알고리즘에 대한 설명은 링크한 곳에 잘 설명되어 있습니다. 여기서는 참고한 곳의 내용을 바탕으로 직접 구현했습니다.

> [https://ratsgo.github.io/nlpbook/docs/preprocess/bpe/](https://ratsgo.github.io/nlpbook/docs/preprocess/bpe/)

### Get Vocabulary
토크나이징을 위해 문서내에 등장한 단어의 등장횟수가 기록된 `dictionary`를 사용하여 단어 집합인 `vocabulary`를 만들어야 합니다.

```python
dictionary = {
    'hug':10,
    'pug':5,
    'pun':12,
    'bun':4,
    'hugs':5
}

vocabulary = list(set(sum([list(word) for word in dictionary.keys()],[])))
# ['p', 'b', 'n', 'g', 'h', 'u', 's']
```

위 코드는 `dictionary`의 단어들을 구성하는 글자들만을 추출하여 `vocabulary`에 저장한 코드입니다. 알고리즘 내에서 `vocabulary`를 사용하고 다시 업데이트를 반복할 것입니다.

모든 단어에 대한 bigram 쌍을 `pairs`라는 딕셔너리에 저장하면서 횟수를 더해줍니다.

```python
import collections

pairs = collections.defaultdict(int)
for word, cnt in dictionary.items():
    i, word_splited = 0,[] # word_splited는 vocabulary에 있는 단어들로 word를 분리하고 저장합니다
    while i < len(word):
        if i+1<len(word) and word[i]+word[i+1] in vocabulary:
            word_splited.append(word[i]+word[i+1])
            i+=2
        else:
            word_splited.append(word[i])
            i+=1
for word1,word2 in zip(word_splited[:-1],word_splited[1:]):
  pairs[word1+word2]+=cnt
 
print(pairs.items())

# dict_items([('hu', 15), ('ug', 20), ('pu', 17), ('un', 16), ('gs', 5), ('bu', 4)])
```

`pairs`에서 가장 많이 등장한 순대로 정렬 후 가장 많은 단어를 `vocabulary`에 저장합니다.

```python
subword = sorted(list(pairs.items()),key=lambda x : -x[1])[0][0]
vocabulary.append(subword)
print(vocabulary)
# ['p', 'b', 'n', 'g', 'h', 'u', 's', 'ug']
```

여기까지의 과정을 반복합니다.

```python
for _ in range(3):
    pairs = collections.defaultdict(int)
    for word, cnt in dictionary.items():
        i, word_splited = 0,[]
        while i < len(word):
            if i+1<len(word) and word[i]+word[i+1] in vocabulary:
                word_splited.append(word[i]+word[i+1])
                i+=2
            else:
                word_splited.append(word[i])
                i+=1
        for word1,word2 in zip(word_splited[:-1],word_splited[1:]):
            pairs[word1+word2]+=cnt
    subword = sorted(list(pairs.items()),key=lambda x : -x[1])[0][0]
    vocabulary.append(subword)

# iter = 0
# [('ug', 20), ('pu', 17), ('un', 16), ('hu', 15), ('gs', 5), ('bu', 4)]
# ['p', 'b', 'n', 'g', 'h', 'u', 's', 'ug']

# iter = 1
# [('un', 16), ('hug', 15), ('pu', 12), ('pug', 5), ('ugs', 5), ('bu', 4)]
# ['p', 'b', 'n', 'g', 'h', 'u', 's', 'ug', 'un']

# iter = 2
# [('hug', 15), ('pun', 12), ('pug', 5), ('ugs', 5), ('bun', 4)]
# ['p', 'b', 'n', 'g', 'h', 'u', 's', 'ug', 'un', 'hug']
```

이렇게 `vocabulary`인 `['p', 'b', 'n', 'g', 'h', 'u', 's', 'ug', 'un', 'hug']`를 얻을 수 있습니다. corpus에서 가장 많이 등장한 bigram 쌍이 순서대로 추가되었습니다.

Huggingface의 토크나이저의 경우 이 `vocabulary`가 `vocab.json`으로 출력되어 단어와 단어 인덱스 값이 key-value 형태로 저장되어 있습니다.

### Tokenizing
토크나이징을 하려면 위에서 구한 `vocabulary`가 필요합니다. `vocabulary`에 따라 단어들을 병합합니다.

```python
UNK = '<unk>'
word = list('hugs')

i, r = 0, []
while i < len(word):
    if i + 1 < len(word) and word[i] + word[i+1] in vocabulary:
        r.append(word[i] + word[i+1])
        i += 1
    elif word[i] in vocabulary:
        r.append(word[i])
    else:
        r.append(UNK)
    i += 1

print(r)

# ['h', 'ug', 's']
```

`<unk>` 토큰은 토크나이저가 `vocabulary`에 없는 단어를 대체할 때 사용합니다. 이런 문제를 OOV(Out of Vocabulary)라 하는데 여기서는 corpus의 크기가 크지 않아 발생합니다.

다시 돌아와서 위의 코드는 hugs를 h,u,g,s로 나누고 h,ug,s로 병합하는 과정이었습니다. 이 과정을 더 이상 병합하지 않을 때까지 반복하여 토크나이징을 수행합니다.

```python
word = list('hugs')

while True:
    i, r = 0, []
    while i < len(word):
        if i + 1 < len(word) and word[i] + word[i+1] in vocabulary:
            r.append(word[i] + word[i+1])
            i += 1
        elif word[i] in vocabulary:
            r.append(word[i])
        else:
            r.append(UNK)
        i += 1
      if word == r:
          break
    word = r
    print(word)

# ['h', 'ug', 's']
# ['hug', 's']
```

이 코드를 함수로 정의하고 문장에 대해 토크나아징할 수 있습니다.

```python
# BPE Tokenizer
sentence = 'pug bug mug hug hugs'
UNK = '<unk>'

def word_tokenizing(word, vocabulary):
    while True:
        i, r = 0, []
        while i < len(word):
            if i + 1 < len(word) and word[i] + word[i+1] in vocabulary:
                r.append(word[i] + word[i+1])
                i += 1
            elif word[i] in vocabulary:
                r.append(word[i])
            else:
                r.append(UNK)
            i += 1
          if word == r:
              # 더이상 병합되지 않는다면 종료
              break
          word = r
    return word

r = []
for word in sentence.split():
    r += word_tokenizing(list(word), vocabulary)

print(r)

# ['p', 'ug', 'b', 'ug', '<unk>', 'ug', 'hug', 'hug', 's']
```

이렇게 BPE 알고리즘을 통해 subword를 분리할 수 있어 합성어를 사전에 포함하지 않아도 되는 장점이 있습니다. 또한, 사전에 단어가 포함되지 않아 발생하는 OOV(Out of vocabulary) 문제를 해결하는 방법이기도 합니다. 

여기까지 BPE 토크나이징 기법을 python으로 구현해봤습니다.
