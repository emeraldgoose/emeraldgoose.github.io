---
title: Finetuning Large Language Models 정리
categories:
  - coursera
tags: [LLM, nlp]
---
> [Finetuning Large Language Models - Deeplearning.ai](https://www.coursera.org/projects/finetuning-large-language-models-project)

# Why Finetune?
---
Finetuning은 GPT-3와 같은 범용 모델을 사용하여 채팅을 잘 할 수 있는 ChatGPT 혹은 자동으로 코드를 완성하는 co-pilot과 같은 모델로 전문화하는 것을 말합니다. finetuning은 모델을 보다 일관된 동작으로 조정하는 것 외에도, 환각(hallucination)을 줄이는 데 도움이 될 수 있습니다.

finetuning은 prompt engineering과 차이점이 존재합니다. prompt engineering은 시작하는데 데이터가 필요하지 않습니다. 초기비용이 적게 든다는 장점이 있고 시작하기 위한 기술적 지식이 필요하지 않지만 환각에 대한 문제가 존재합니다. 모델이 이미 학습한 잘못된 정보를 수정하는 것이 어렵기 때문에 모델이 잘못된 정보를 출력하는 경우가 많습니다.

finetuning은 많은 수의 데이터를 사용할 수 있고 모델이 새로운 정보를 학습하는데 좋습니다. 따라서 이전에 학습했을 잘못된 정보를 수정하거나 이전에 학습되지 않은 최근 정보를 입력할 수도 있습니다. 그러나 좋은 품질의 더 많은 데이터와 컴퓨팅 리소스가 필요합니다.

따라서, prompt engineering은 다양한 사이드 프로젝트나 프로토타입에 적합하고 finetuning은 엔터프라이즈와 프로덕션에 적합합니다.

finetuning을 할 수 있는 라이브러리는 3가지가 존재합니다.

- Pytorch(Meta) : Low-level interface
- Huggingface : Pytorch 보다 고수준의 인터페이스
- Llama(Llamini) : 세 라이브러리 중 가장 높은 수준의 인터페이스
  - 여기서는 주로 Llamini 라이브러리를 사용하고 있습니다.

```python
# Llamini Library
from llama import BasicModelRunner
non_finetuned = BasicModelRunner("meta-llama/Llama-2-7b-hf")
non_finetuned_output = non_finetuned("Tell me how to train my dog to sit")

# Output
# .
# Tell me how to train my dog to sit. 
# I have a 10 month old puppy and I want to train him to sit. 
# I have tried the treat method and he just sits there and looks at me like I am crazy. 
# I have tried the "sit" command and he just looks at me like I am crazy. 
# I have tried the "sit" command and he just looks at me like I am crazy. 
...
# I have tried the "sit" command and he just looks at me like I am crazy. 
# I have tried the "sit" command and he just looks

finetuned_model = BasicModelRunner("meta-llama/Llama-2-7b-chat-hf")
finetuned_otuput = finetuned_model("Tell me how to train my dog to sit")

# Output
# on command.
# Training a dog to sit on command is a basic obedience command that can be achieved with patience, consistency, and positive reinforcement. Here's a step-by-step guide on how to train your dog to sit on command:

# 1. Choose a quiet and distraction-free area: Find a quiet area with minimal distractions where your dog can focus on you.
# 2. Have treats ready: Choose your dog's favorite treats and have them ready to use as rewards.
# 3. Stand in front of your dog: Stand in front of your dog and hold a treat close to their nose.
# 4. Move the treat up and back: Slowly move the treat up and back, towards your dog's tail, while saying "sit" in a calm and clear voice.
# 5. Dog will sit: As you move the treat, your dog will naturally sit down to follow the treat. The moment their bottom touches the ground, say "good sit" and give them the treat.
# 6. Repeat the process: Repeat steps 3-5 several times, so your dog starts to associate the command "sit" with
```

# Where finetuning fits in
---
Finetuning은 사전훈련(Pretrain) 단계 뒤에 진행합니다. 사전훈련은 전혀 세팅되지 않은 모델을 사용하며 다음 토큰을 예측하는 목표를 가지고 훈련합니다. 예를들어 ‘wants’ 뒤에 ‘upon’이라는 단어를 예측하도록 훈련하며 라벨링되지 않은 거대한 양의 데이터를 학습합니다. 이러한 학습 방법을 Self-supervised Learning이라 합니다.

Finetuning은 모델의 동작을 변화시킵니다. 모델 응답을 일관성있게 하거나 질문에 좀 더 집중할 수 있게 합니다. 또한, 모델의 능력을 발현(?)시킬 수 있습니다. 대화에 더 능숙해져 다양한 주제에 대해 이야기 할 수 있게 됩니다. 이전에는 이러한 정보를 얻기 위해 많은 Prompt engineering을 해야 했지만 finetuning은 쉽게 가능합니다.

Finetuning을 처음 하게 된다면 추천하는 몇가지 단계에 대해 소개합니다.

1. Identify task(s) by prompt-engineering a large LLM
2. Find tasks that you see an LLM doing ~OK at
3. Pick one task
4. Get ~1000 inputs and outputs for the task
  - Better than the ~OK from the LLM
5. Finetune a small LLM on this data

```python
import jsonlines
import itertools
import pandas as pd
from pprint import pprint

filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)

examples = instruction_dataset_df.to_dict()

prompt_template_qa = """### Question:
{question}

### Answer:
{answer}"""

num_examples = len(examples["question"])
finetuning_dataset_text_only = []
finetuning_dataset_question_answer = []

for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]

  text_with_prompt_template_qa = prompt_template_qa.format(
    question=question,
    answer=answer
  )
  finetuning_dataset_text_only.append(
    {
      "text": text_with_prompt_template_qa
    }
  )

  text_with_prompt_template_q = prompt_template_q.format(
    question=question
  )
  finetuning_dataset_question_answer.append(
    {
      "question": text_with_prompt_template_q,
      "answer": answer
    }
  )

pprint(finetuning_dataset_text_only[0])

# Output
# {'text': '### Question:\n'
#          'What are the different types of documents available in the '
#          "repository (e.g., installation guide, API documentation, developer's "
#          'guide)?\n'
#          '\n'
#          '### Answer:\n'
#          'Lamini has documentation on Getting Started, Authentication, '
#          'Question Answer Model, Python Library, Batching, Error Handling, '
#          'Advanced topics, and class documentation on LLM Engine available at '
#          'https://lamini-ai.github.io/.'}

pprint(finetuning_dataset_question_answer[0])

# Output
# {'answer': 'Lamini has documentation on Getting Started, Authentication, '
#            'Question Answer Model, Python Library, Batching, Error Handling, '
#            'Advanced topics, and class documentation on LLM Engine available '
#            'at https://lamini-ai.github.io/.',
#  'question': '### Question:\n'
#              'What are the different types of documents available in the '
#              'repository (e.g., installation guide, API documentation, '
#              "developer's guide)?\n"
#              '\n'
#              '### Answer:'}
```

# Instruction finetuning
---
Finetuning 중 Instruction Finetuning(instruction-tuned, instruction-following)이라는 finetuning이 있습니다. 이 방식은 모델을 챗봇과 같이 행동하도록 조정하는 방법입니다. Instruction-following 데이터셋의 예로는 FAQs, Customer support conversation, Slak 메시지 등이 있습니다. 만약 데이터가 없다면 LLM을 프롬프트 템플릿을 사용하여 Non-QnA 데이터를 QnA 데이터로 변환하는 방법도 존재합니다.

```python
# Instruction-tuning
import itertools
import jsonlines

from datasets import load_dataset
from pprint import pprint

instruction_tuned_dataset = load_dataset(
	"tatsu-lab/alpaca", 
	split="train", 
	streaming=True
)

# Two prompt templates
prompt_template_with_input = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""

prompt_template_without_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

# Hydrate prompts (add data to prompts)
processed_data = []
for j in top_m:
  if not j["input"]:
    processed_prompt = prompt_template_without_input.format(instruction=j["instruction"])
  else:
    processed_prompt = prompt_template_with_input.format(instruction=j["instruction"], input=j["input"])

  processed_data.append({"input": processed_prompt, "output": j["output"]})

# Save data to jsonl
with jsonlines.open(f'alpaca_processed.jsonl', 'w') as writer:
    writer.write_all(processed_data)
```
```python
from llama import BasicModelRunner
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # Tokenize
  input_ids = tokenizer.encode(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=max_input_tokens
  )

  # Generate
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
  )

  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

  # Strip the prompt
  generated_text_answer = generated_text_with_prompt[0][len(text):]

  return generated_text_answer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")

finetuning_dataset_path = "lamini/lamini_docs"
finetuning_dataset = load_dataset(finetuning_dataset_path)

test_sample = finetuning_dataset["test"][0]
print(inference(test_sample["question"], model, tokenizer))

# Output
# I have a question about the following:
# 
# How do I get the correct documentation to work?
# 
# A:
# 
# I think you need to use the following code:
# 
# A:
# 
# You can use the following code to get the correct documentation.
# 
# A:
# 
# You can use the following code to get the correct documentation.
# 
# A:
# 
# You can use the following
```

# Data preparation
---
데이터 준비에 필요한 몇가지가 있습니다.

1. 고품질의 데이터
2. 데이터의 다양성
3. 데이터의 양
  - 데이터의 양보다 품질이 더 중요함

데이터를 준비하는 스텝은 다음과 같습니다.

1. instruction-response 쌍을 수집한다.
2. 이러한 쌍을 연결(concatenate)하거나 Prompt 템플릿을 추가한다.
3. 데이터를 토크나이징하고 패딩을 추가하거나 데이터를 잘라 알맞은 크기로 모델에 입력하도록 한다.
4. 해당 데이터를 train과 test로 분리한다.

토크나이징은 텍스트 데이터를 가져와 각각의 텍스트 조각을 숫자로 변환하는 작업입니다. 다양한 토크나이징 도구가 있으며 모델들은 훈련된 특정 토크나이저와 연관되어 있습니다. 잘못된 토크나이저를 모델에 사용하면 모델과 토크나이저가 서로 다른 문자 및 단어를 나타내어 모델이 혼란스러워질 수 있습니다.

```python
import pandas as pd
import datasets

from pprint import pprint
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)
examples = instruction_dataset_df.to_dict()

prompt_template = """### Question:
{question}

### Answer:"""

num_examples = len(examples["question"])
finetuning_dataset = []
for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]
  text_with_prompt_template = prompt_template.format(question=question)
  finetuning_dataset.append(
    {
      "question": text_with_prompt_template, 
      "answer": answer
    }
  )

def tokenize_function(examples):
  tokenizer.pad_token = tokenizer.eos_token
  tokenized_inputs = tokenizer(
    text,
    return_tensors="np",
    padding=True,
  )

  max_length = min(
    tokenized_inputs["input_ids"].shape[1],
    2048
  )
  tokenizer.truncation_side = "left"
  tokenized_inputs = tokenizer(
    text,
    return_tensors="np",
    truncation=True,
    max_length=max_length
  )

  return tokenized_inputs

finetuning_dataset_loaded = datasets.load_dataset(
  "json", 
  data_files=filename, 
  split="train"
)

tokenized_dataset = finetuning_dataset_loaded.map(
  tokenize_function,
  batched=True,
  batch_size=1,
  drop_last_batch=True
)

tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])
split_dataset = tokenized_dataset.train_test_split(
  test_size=0.1, 
  shuffle=True, 
  seed=123
)
print(split_dataset)

# Output
# DatasetDict({
#     train: Dataset({
#         features: ['question', 'answer', 'input_ids', 'attention_mask', 'labels'],
#         num_rows: 1260
#     })
#     test: Dataset({
#         features: ['question', 'answer', 'input_ids', 'attention_mask', 'labels'],
#         num_rows: 140
#     })
# })
```

# Training process
---
기존 신경망에서의 학습은 학습 데이터를 추가하고 loss를 계산하여 역전파과정을 거쳐 파라미터를 업데이트합니다. 하이퍼파라미터는 learning rate, learning rate scheduler, optimizer hyperparameters가 된다. 그리고 Pytorch와 llamini를 사용하면 학습 프로세스가 아래 코드와 같은 형태로 작성됩니다.

```python
# Pytorch
for epoch in range(num_epochs):
  for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Llamini
from llama import BasicModelRunner

model = BasicModelRunner("EleutherAI/pythia-410m")
model.load_data_from_jsonlines("lamini_docs.jsonl", input_key="question", output_key="answer")
model.train(is_public=True)
```
```python
import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import time
import torch
import transformers
import pandas as pd
import jsonlines

from utilities import *
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from llama import BasicModelRunner


logger = logging.getLogger(__name__)
global_config = None

# Load the Lamini docs dataset
dataset_name = "lamini_docs.jsonl"
dataset_path = f"/content/{dataset_name}"
use_hf = False

dataset_path = "lamini/lamini_docs"
use_hf = True

# Set up model, training config, and tokenizer
model_name = "EleutherAI/pythia-70m"
training_config = {
  "model": {
    "pretrained_name": model_name,
    "max_length" : 2048
  },
  "datasets": {
    "use_hf": use_hf,
    "path": dataset_path
  },
  "verbose": True
}
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset, test_dataset = tokenize_and_split_data(training_config, tokenizer)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(model_name)

device_count = torch.cuda.device_count()
if device_count > 0:
  logger.debug("Select GPU device")
  device = torch.device("cuda")
else:
  logger.debug("Select CPU device")
  device = torch.device("cpu")

base_model.to(device)

# Define function to carry out inference
def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # Tokenize
  input_ids = tokenizer.encode(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=max_input_tokens
  )

  # Generate
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
  )

  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

  # Strip the prompt
  generated_text_answer = generated_text_with_prompt[0][len(text):]

  return generated_text_answer

# Setup trainig
max_steps = 3
trained_model_name = f"lamini_docs_{max_steps}_steps"
output_dir = trained_model_name

training_args = TrainingArguments(

  # Learning rate
  learning_rate=1.0e-5,

  # Number of training epochs
  num_train_epochs=1,

  # Max steps to train for (each step is a batch of data)
  # Overrides num_train_epochs, if not -1
  max_steps=max_steps,

  # Batch size for training
  per_device_train_batch_size=1,

  # Directory to save model checkpoints
  output_dir=output_dir,

  # Other arguments
  overwrite_output_dir=False, # Overwrite the content of the output directory
  disable_tqdm=False, # Disable progress bars
  eval_steps=120, # Number of update steps between two evaluations
  save_steps=120, # After # steps model is saved
  warmup_steps=1, # Number of warmup steps for learning rate scheduler
  per_device_eval_batch_size=1, # Batch size for evaluation
  evaluation_strategy="steps",
  logging_strategy="steps",
  logging_steps=1,
  optim="adafactor",
  gradient_accumulation_steps = 4,
  gradient_checkpointing=False,

  # Parameters for early stopping
  load_best_model_at_end=True,
  save_total_limit=1,
  metric_for_best_model="eval_loss",
  greater_is_better=False
)

model_flops = (
  base_model.floating_point_ops(
    {
      "input_ids": torch.zeros(
        (1, training_config["model"]["max_length"])
      )
    }
  )
  * training_args.gradient_accumulation_steps
)

trainer = Trainer(
  model=base_model,
  model_flops=model_flops,
  total_steps=max_steps,
  args=training_args,
  train_dataset=train_dataset,
  eval_dataset=test_dataset,
)

# Train a few steps
training_output = trainer.train()

# Save model locally
save_dir = f'{output_dir}/final'

trainer.save_model(save_dir)

finetuned_slightly_model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True)
finetuned_slightly_model.to(device)

# Finetune a model in 3 lines of code using Lamini
model = BasicModelRunner("EleutherAI/pythia-410m") 
model.load_data_from_jsonlines("lamini_docs.jsonl", input_key="question", output_key="answer")
model.train(is_public=True)

out = model.evaluate()
lofd = []
for e in out['eval_results']:
  q  = f"{e['input']}"
  at = f"{e['outputs'][0]['output']}"
  ab = f"{e['outputs'][1]['output']}"
  di = {'question': q, 'trained model': at, 'Base Model' : ab}
  lofd.append(di)
df = pd.DataFrame.from_dict(lofd)
style_df = df.style.set_properties(**{'text-align': 'left'})
style_df = style_df.set_properties(**{"vertical-align": "text-top"})

# Output
# question : Does Lamini have the ability to understand and generate code for audio processing tasks?
# trained model : Yes, Lamini has the ability to understand and generate code.
# Base Model : A: Lamini is a very good language for audio processing.\nA: I think...
```

# Evaluation and iteration
---
모델을 훈련한 후에는 다음 단계로 모델을 평가하고 성능을 확인해야 합니다. 이 과정은 시간이 지남에 따라 모델을 개선하는데 도움을 줄 수 있는 반복적인 과정입니다.

생성 모델은 명확한 측정 지표가 없어 평가하는 것이 어렵습니다. 그래서 Human expert evaluation이 종종 가장 신뢰성있는 방법이며 해당 도메인을 이해하는 전문가들이 평가하는 것입니다.

LLM 벤치마크는 여러 평가 방법의 모음으로 구성됩니다.

- ARC : 초등학교 질문의 모음
- HellaSwag : 상식 테스트
- MMLU : 다양한 초등학교 과목
- TruthfulQA : 모델이 온라인에서 흔히 찾을 수 있는 거짓말을 재현하는 능력을 측정

모델을 분석하고 평가하는데 사용되는 방법 중 “Error Analysis”가 있습니다. Error Analysis는 에러를 범주화하여 매우 일반적인 에러 유형을 이해하고 가장 흔한 에러와 매우 치명적인 에러를 우선적으로 해결하는 것입니다.

모델을 finetuning하기 전에 기본 모델의 에러를 분석한 후 finetuning을 했을 때 가장 큰 효과를 줄 수 있는 데이터 종류를 파악할 수 있습니다.

에러의 범주는 다음과 같습니다.

- Misspelling(맞춤법 오류)
- Too long(길이) : 데이터셋이 덜 장황하도록 하여 모델이 질문에 명확하게 답변할 수 있도록 하는 것이 중요합니다.
- Repetitive(반복) : 모델이 반복적인 응답을 할 수 있는데 이를 해결하는 한 가지 방법은 stop token들을 사용하거나(eos 토큰을 가리키는 것 같습니다) Prompt 템플릿을 이용하는 것입니다. 데이터셋에 반복이 적고 다양성이 있는 예제를 포함하는 것도 중요합니다.

이 강의에서는 벤치마크에 너무 집착하지 말라고 합니다. 모델을 순위를 매기는 방식이긴 하지만, 실제 사용사례와 다를 수 있기 때문입니다. 따라서 finetuning된 모델은 다양한 task에 맞게 조정될 수 있으며, 이는 다양한 평가 방법이 필요합니다.

```python
import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import logging
import difflib
import pandas as pd

import transformers
import datasets
import torch

from tqdm import tqdm
from utilities import *
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)
global_config = None

dataset = datasets.load_dataset("lamini/lamini_docs")

test_dataset = dataset["test"]

model_name = "lamini/lamini_docs_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Setup a really basic evaluation function
def is_exact_match(a, b):
  return a.strip() == b.strip()

model.eval() # dropout과 같은 기능이 비활성화되었는지 확인해야 한다

def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # Tokenize
  tokenizer.pad_token = tokenizer.eos_token
  input_ids = tokenizer.encode(
      text,
      return_tensors="pt",
      truncation=True,
      max_length=max_input_tokens
  )

  # Generate
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
  )

  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

  # Strip the prompt
  generated_text_answer = generated_text_with_prompt[0][len(text):]

  return generated_text_answer

# Run model and compare to expected answer
generated_answer = inference(test_question, model, tokenizer)

answer = test_dataset[0]["answer"]

exact_match = is_exact_match(generated_answer, answer)
print(exact_match)

# Output
# False
# generated_answer와 answer가 같은지 단순히 확인하는 간단한 방법입니다
# 다른 방법으로는 generated_answer와 answer를 LLM에 입력하고 같은 답변인지 점수로 매겨 얼마나 가까운지 확인할 수 있다

# Run over entire dataset
n = 10
metrics = {'exact_matches': []}
predictions = []
for i, item in tqdm(enumerate(test_dataset)):
    print("i Evaluating: " + str(item))
    question = item['question']
    answer = item['answer']

    try:
      predicted_answer = inference(question, model, tokenizer)
    except:
      continue
    predictions.append([predicted_answer, answer])

    #fixed: exact_match = is_exact_match(generated_answer, answer)
    exact_match = is_exact_match(predicted_answer, answer)
    metrics['exact_matches'].append(exact_match)

    if i > n and n != -1:
      break
```

# Consideration on getting started now
---
Finetuning과정에 대한 실용적인 접근을 소개합니다.

1. Task를 먼저 이해해야 합니다.
2. Task의 입력과 출력에 관련된 데이터를 수집합니다.
3. 데이터가 충분하지 않다면 데이터를 생성합니다.
    - Prompt Template
4. 먼저 작은 모델을 finetuning하는 것이 좋습니다.
    - 4억(400M)에서 10억(1B) 파라미터 모델을 추천합니다.
5. 데이터의 양을 변경하면서 모델의 성능을 측정합니다.
6. 모델을 평가하면서 잘 되고 있는지 확인합니다.
7. 모델을 개선하기 위해 더 많은 데이터를 수집합니다.
8. Task를 좀 더 복잡하게 해봅니다.
    - 글쓰기 작업은 읽기 작업보다 더 많은 토큰을 생성해야 하므로 어렵습니다.
    - 여러 가지 작업을 묶어 모델이 여러 작업을 수행하도록 할 수도 있습니다.
9. 성능을 위해 모델의 사이즈를 늘려봅니다.

작업 복잡성에 따른 모델 크기를 감안할 때 하드웨어에 대한 고려가 필요합니다. v100 1장은 16GB 메모리를 가지고 있어 추론시 최대 7B의 모델을 사용할 수 있지만 훈련시 최대 1B 모델만 사용가능합니다. 더 큰 모델을 사용하려면 다른 옵션을 고려해야 합니다.

더 큰 모델을 사용하려는 경우 PEFT(Parameter-efficient Fine-tuning)나 LoRa(Low-Rank Adaptation)와 같은 다양한 방법들을 고려해볼 수 있습니다. 이 중 LoRa는 main pretrained weight들을 freezing하고 일부 레이어안에 새로운 가중치를 훈련시키는 방법입니다. 새롭게 훈련된 가중치를 메인 가중치로 다시 병합하여 fine-tuning된 모델을 더 효율적으로 얻을 수 있습니다.