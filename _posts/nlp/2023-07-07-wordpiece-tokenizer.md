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

```python
from collections import defaultdict

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

def pre_tokenize_str(text):
    """
    tokenizer 라이브러리에 있는 함수를 구현한 함수입니다.
    완전히 똑같은 함수는 아니므로 결과값만 참고해주세요.
    """
    import re
    words_with_offsets = []
    offset = 0
    text = re.split(r'([.,?!@#$%^&*]+)',text)
    special_characters = ".,?!@#$%^&*"
    for subtext in text:
        subtext = subtext.split()
        for word in subtext:
            if word in special_characters:
                offset -= 1
            next_offset = offset + len(word)
            words_with_offsets.append((word, (offset, next_offset)))
            offset = next_offset + 1
    return words_with_offsets

word_freqs = defaultdict(int) # (단어, 빈도 수)
for text in corpus:
    words_with_offsets = pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

# word_freqs
# {'This': 3, 'is': 2, 'the': 1, 'Hugging': 1, 'Face': 1, 'Course': 1, '.': 4, 
#  'chapter': 1, 'about': 1, 'tokenization': 1, 'section': 1, 'shows': 1, 
#  'several': 1, 'tokenizer': 1, 'algorithms': 1, 'Hopefully': 1, ',': 1, 
#  'you': 1, 'will': 1, 'be': 1, 'able': 1, 'to': 1, 'understand': 1, 'how': 1, 
#  'they': 1, 'are': 1, 'trained': 1, 'and': 1, 'generate': 1, 'tokens': 1}

alphabet = []
for word in word_freqs.keys():
    if word[0] not in alphabet:
        alphabet.append(word[0])
    for letter in word[1:]:
        if f'##{letter}' not in alphabet:
            alphabet.append(f'##{letter}')

alphabet.sort()
print(alphabet)

# alphabet
# [
#  '##a', '##b', '##c', '##d', '##e', '##f', '##g', 
#  '##h', '##i', '##k', '##l', '##m', '##n', '##o', 
#  '##p', '##r', '##s', '##t', '##u', '##v', '##w', 
#  '##y', '##z', ',', '.', 'C', 'F', 'H', 'T', 'a', 
#  'b', 'c', 'g', 'h', 'i', 's', 't', 'u', 'w', 'y'
# ]
```

각 단어들을 분리한 splits를 `compute_pair_scores` 함수로 score들을 계산합니다. 

```python
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()

splits = {
    word : [c if i==0 else f'##{c}' for i,c in enumerate(word)] for word in word_freqs.keys()
}

print(splits)
# {
#  'This': ['T', '##h', '##i', '##s'], 
#  'is': ['i', '##s'], 
#  'the': ['t', '##h', '##e'], 
#  'Hugging': ['H', '##u', '##g', '##g', '##i', '##n', '##g'], 
#  'Face': ['F', '##a', '##c', '##e'], 
#  'Course': ['C', '##o', '##u', '##r', '##s', '##e'], 
#  '.': ['.'],
#  ...}

def compute_pair_scores(splits):
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split_ = splits[word]
        if len(split_) == 1:
            letter_freqs[split_[0]] += freq
            continue
        for i in range(len(split_) - 1):
            pair = (split_[i], split_[i+1])
            letter_freqs[split_[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split_[-1]] += freq

    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]]) for pair, freq in pair_freqs.items()
    }
    return scores
  
pair_scores = compute_pair_scores(splits)
for i, key in enumerate(pair_scores.keys()):
    print(f'{key}: {pair_scores[key]}')

# ('T', '##h'): 0.125
# ('##h', '##i'): 0.03409090909090909
# ('##i', '##s'): 0.02727272727272727
# ('i', '##s'): 0.1
# ('t', '##h'): 0.03571428571428571
# ('a', '##b'): 0.2
```

`merge_pair`라는 함수를 사용하여 머지가 필요한 쌍을 머지하고 splits에 저장합니다. 지정한 vocab_size가 모두 채워질 때까지 반복합니다.

```python
def merge_pair(a,b,splits):
    for word in word_freqs:
        split_ = splits[word]
        if len(split_) == 1:
            continue
        i = 0
        while i < len(split_) - 1:
            if split_[i] == a and split_[i+1] == b:
                merge = a + b[2:] if b.startswith('##') else a + b
                split_ = split_[:i] + [merge] + split_[i+2:]
            else:
                i += 1
        splits[word] = split_
    return splits
    
vocab_size = 70
# vocab_size를 채울 때까지 반복한다
while len(vocab) < vocab_size:
    scores = compute_pair_scores(splits)
    best_pair, max_score = "", None
    for pair, score in scores.items():
        if max_score is None or max_score < score:
            best_pair = pair
            max_score = score
    splits = merge_pair(*best_pair, splits)
    new_token = (
        best_pair[0] + best_pair[1][2:]
        if best_pair[1].startswith("##")
        else best_pair[0] + best_pair[1]
    )
    vocab.append(new_token)
    
print(vocab)

# [
#  '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 
#  '##a', '##b', '##c', '##d', '##e', '##f', '##g', '##h', 
#  '##i', '##k', '##l', '##m', '##n', '##o', '##p', '##r', '##s', 
#  '##t', '##u', '##v', '##w', '##y', '##z', ',', '.', 
#  'C', 'F', 'H', 'T', 'a', 'b', 'c', 'g', 'h', 'i', 's', 't', 'u', 'w', 'y', 
#  'ab', '##fu', 'Fa', 'Fac', '##ct', '##ful', '##full', 
#  '##fully', 'Th', 'ch', '##hm', 'cha', 'chap', 'chapt', 
#  '##thm', 'Hu', 'Hug', 'Hugg', 'sh', 'th', 'is', '##thms', '##za', '##zat', '##ut'
# ]
```

## Tokenization
문장내 각 단어들을 vocab에 있는 단어에 따라 토크나이징합니다. 인코딩에서는 첫번째 글자부터 가장 큰 subword를 찾아 분리한 후 반복합니다.

```python
def encode_word(word):
    tokens = []
    while len(word) > 0:
        i = len(word)
        # vocab에 있는 단어가 있다면 tokens에 추가한다
        while i > 0 and word[:i] not in vocab:
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(word[:i])
        word = word[i:]
        if len(word) > 0:
            word = f"##{word}"
    return tokens
  
def tokenize(text):
    pre_tokenize_result = pre_tokenize_str(text)
    pre_tokenize_text = [word for word, offset in pre_tokenize_result]
    encoded_words = [encode_word(word) for word in pre_tokenize_text]
    return sum(encoded_words, [])
  
print(tokenize("This is the Hugging Face course!"))

# [
#  'Th', '##i', '##s', 'is', 'th', '##e', 
#  'Hugg', '##i', '##n', '##g', 'Fac', '##e', 
#  'c', '##o', '##u', '##r', '##s', '##e', '[UNK]'
# ]
```