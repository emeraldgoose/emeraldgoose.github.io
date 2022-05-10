---
title: "Huggingface에 사전학습한 모델 업로드하기"
categories:
  - BoostCamp
tags: [huggingface]
---

> 이번 데이터셋 구축을 진행하면서 klue/bert-base처럼 pretraining한 모델을 업로드해보는 경험이 필요하다고 생각하게 되어 실행에 옮겼습니다.

## 사전학습 준비
먼저, 확보한 데이터는 문장단위였습니다. bert는 MLM(Masked Language Model)과 NSP(Next Sentence Prediction) task를 수행하는데 문장 단위의 데이터로 NSP를 수행할 수 없었습니다.  

그래서 bert의 MLM만을 이용하여 pretraining을 진행하였습니다.


## tokenizer 학습
bert 뿐만 아니라 tokenizer 또한 확보한 문장에 맞춰 학습을 해야합니다. tokenizer의 학습 결과로 `vocab.txt`을 저장할 수 있기 때문입니다.

Wordpiece와 Sentencepiece 두 가지가 있고 저는 Wordpiece를 이용하여 문장을 분리하고 vocab을 저장하려고 했습니다.

```python
def tokenizer_train():
  # load sentences
  tokenizer = BertWordPieceTokenizer(
    vocab=None,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False,
    lowercase=False,
    wordpieces_prefix="##",
  )

  limit_alphabet = 1000
  vocab_size = 30000

  tokenizer.train(
    files='./sentence.txt',
    vocab_size=vocab_size,
    limit_alphabet=limit_alphabet,
  )

  tokenizer.save("./tokenizer.json", True)  # save tokenizer.json, pretty=True로 두시면 json 형식이 보기좋게 저장됩니다
  tokenizer.save_model('./')  # save vocab.txt
```

## Bert 데이터 준비
bert의 MLM을 수행하기 위해 먼저 필요한 데이터를 확인합니다.
- input_ids, token_type_ids, attention_mask, labels

input_ids 토큰들 중 랜덤으로 마스킹하는데 labels은 마스킹된 토큰들만 들어가는 것이 아닌 원래 input_ids 토큰 리스트이어야 합니다.

```python
def preprocessing(dataset):
  """ preprocessing(random word convert to "[MASK]") """

  mask = 4  # [MASK] 토큰번호
  label = []
  for idx, words in enumerate(dataset['input_ids']):
    maxlen = 0
    for i in range(len(words)): # [PAD], [SEP]을 제외한 나머지 토큰들의 범위
      if not (0 <= words[i] <= 4):
        maxlen = max(maxlen, i)

      masked_idx = torch.randint( # 간단하게 전체 중의 15%의 토큰 중 80%의 토큰만 마스킹합니다
        size=(int((maxlen-1)*0.15*0.8),), low=1, high=maxlen)
      tmp = words.clone().detach() # label은 원래의 input_ids를 복사한 것입니다
      label.append(tmp)

      for index in masked_idx:
        words[index] = mask
      dataset['input_ids'][idx] = words

  return dataset, label
```

## 모델 학습
위에서 bert가 MLM만을 수행해야 하므로 transformer에서 `BertForMaskedLM`모델을 불러옵니다. 참고로 `BertForPretraining`모델은 MLM 데이터와 NSP 데이터 모두 필요하므로 현재 가지고 있는 데이터로는 사전학습을 할 수 없었습니다.

또한, `BertConfig을` 불러와서 원하는 설정을 한 후 `BertForMaskedLM`에 config설정을 해줍니다.

```python
  # training 준비
  config = BertConfig(
      ...
  )

  tokenizer = AutoTokenizer.from_pretrained('tokenizer path') # 학습한 토크나이저가 저장된 경로로 지정해줍니다

  model = BertForMaskedLM(config=config)
  model.resize_token_embeddings(config.vocab_size) # 기본으로 설정된 vocab size를 우리가 만든 vocab size로 임베딩 크기를 맞춰야 합니다
  model.to(device)
```

여기서 vocab size가 맞지 않으면 CUDA 에러가 발생합니다.  
아래 에러들은 발생했던 에러들입니다. 아래 에러들의 공통점은 레이어의 사이즈가 맞지 않을 때 발생합니다.
- CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling 'cublasCreate(handle)'
- RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling 'cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)'
- CUDA error: device-side assert triggered

`training_args` 를 설정한 후 `Trainer`를 통해 모델을 학습하면 됩니다.  
`trainer.save_model(output_dir='')`를 통해 `pytorch_model.bin`을 포함하여 여러가지 파일들이 저장이 됩니다.

## 허깅페이스 업로드
모델 학습이 완료되면 output_dir 경로에 학습된 결과물들이 저장되어있습니다.

이 모델들을 허깅페이스에 업로드하기 전에 허깅페이스 가입을 완료한 후 New Model을 통해 레포지토리를 생성하고 git clone 해옵니다.

아까 output_dir 위치에 있는 파일들을 clone된 디렉토리에 옮긴 후에 git lfs를 활성화 해야 합니다.

> git lfs란? 기존 git 명령은 최대 10M을 초과하는 파일을 원격 레포지토리에 업로드 할 수 없습니다. git lfs를 통해 큰 파일을 옮길 수 있게됩니다.

git lfs는 `sudo apt-get install git lfs`를 통해 git lfs를 설치하고 아까 clone한 디렉토리에 가서 `git lfs install` 명령어를 통해 해당 디렉토리가 git lfs를 사용할 수 있도록 합니다.

이후에는 `git add .`, `git commit -m "message"`, `git push` 순으로 명령어를 치시면 git lfs를 통해 레포지토리로 업로드를 시작합니다.

### tokenizer_config.json
`tokenizer_config.json`은 허깅페이스의 inference api를 이용하는데 필요한 파일입니다.

이 파일을 보면 never_split 이후에 여러가지 항목들이 있을 텐데 나머지를 날리고 허깅페이스로 업로드 하시면 inference api를 정상적으로 사용할 수 있게 됩니다.

또한, inference api의 example을 바꾸고 싶다면 READEME.md에 아래와 같은 글을 상단에 넣으시면 됩니다.

```
---
language: ko
mask_token: "[MASK]"
widget:
  - text: 산악 자전거 경기는 상대적으로 새로운 [MASK] 1990년대에 활성화 되었다.
---
```

