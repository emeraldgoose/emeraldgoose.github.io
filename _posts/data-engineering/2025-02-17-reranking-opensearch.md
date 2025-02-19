---
title: "Reranking - Opensearch"
categories:
 - data-engineer
tags: [data-engineering, Opensearch]
---
## Reranker
문서 검색 과정에서 문서를 임베딩 벡터로 변환하는 과정과 검색 시간 단축을 위해 Approximate Nearest Neighbor search(ANNs)와 같이 근사 기법으로 인해 정보 손실이 발생합니다. 
이로 인해 필요한 문서가 누락될 가능성이 있으며, 이를 보완하기 위해 가져오는 문서의 개수 k를 조정할 수 있습니다. 그러나 문서의 순위가 낮아질 수 있으며, LLM이 해당 문서를 참조하지 않거나 LLM 토큰 비용이 증가하는 비효율적인 상황이 발생하게 됩니다. 
이를 해결하기 위해 Reranker를 도입하여 Retriever가 가져온 문서와 사용자 질문 간의 관련성을 측정하여 보다 연관성이 높은 문서의 순위를 조정하는 방법을 활용할 수 있습니다.

## Opensearch Machine Learning
RAG 시스템에 Reranker를 추가하려면 AWS Sagemaker와 같은 클라우드 서비스를 이용해 모델을 배포하거나 Cohere.ai와 같은 상용 서비스를 사용하는 방법이 있지만 비용이 추가됩니다. 
Opensearch는 클러스터에 직접 아티팩트를 배포하고 사용할 수 있는 기능을 지원하고 있습니다. 
만약 Opensearch 클러스터를 운영하고 있다면 대안으로 클러스터에 직접 아티팩트를 배포할 수 있고 REST API를 통해 모델을 사용할 수 있습니다. 또한, 파이프라인에 포함시킬 수도 있습니다.

### Settings
프로덕션 환경에서의 Opensearch의 ML 작업은 ML 노드에서 실행하는 것을 추천합니다. 
프로덕션 환경이라면 아래 쿼리에서 `plugins.ml_commons.only_run_on_ml_node: true`로 설정합니다. 그렇지 않다면 아무 노드에서나 ml 작업을 할 수 있도록 `false`로 설정합니다. 
`native_memory_threshold` 옵션은 ML작업을 하는데 메모리 사용률 한계를 설정할 수 있습니다. 100이면 아무런 제한도 하지 않는 것이고 90이라면 90%까지 허용하며 넘어가는 경우 에러를 일으키게 됩니다. 클러스터 자원에 따라 자유롭게 조정합니다. 
```json
PUT _cluster/settings
{
  "persistent": {
    "plugins.ml_commons.only_run_on_ml_node": "false",
    "plugins.ml_commons.model_access_control_enabled": "true",
    "plugins.ml_commons.native_memory_threshold": "99"
  }
}
```

### Register a model group
ML 모델을 배포하기 전에 모델 그룹을 먼저 등록해야 합니다. 모델 그룹의 역할은 모델 버전을 관리하고 접근 제어에 활용됩니다. 
모델 그룹을 등록하려면 다음 요청을 보내야 합니다.
```json
POST /_plugins/_ml/model_groups/_register
{
  "name": "local_model_group",
  "description": "A model group for local models"
}
```
요청이 올바르게 전달되었다면 다음의 응답을 받게 됩니다.
```json
{
 "model_group_id": "wlcnb4kBJ1eYAeTMHlV6",
 "status": "CREATED"
}
```

### Register a model
모델을 등록하는 단계입니다. 
Opensearch에서 제공하는 pretrained model 또는 Custom model를 등록할 수 있습니다. 
Custom model인 경우, torchscript 또는 ONNX로 변환되어 있어야 합니다. 
저는 Opensearch에서 제공하는 Reranking 모델인 `cross-encoders/ms-marco-MiniLM-L-12-v2`를 배포하겠습니다.

아래 요청에서 `model_group_id는` 위에서 등록된 `model_group_id`와 같아야 합니다.
```json
POST /_plugins/_ml/models/_register
{
    "name": "cross-encoders/ms-marco-MiniLM-L-12-v2", 
    "version": "1.0.2", 
    "description": "The model can be used for Information Retrieval: Given a query, encode the query will all possible passages (e.g. retrieved with ElasticSearch). Then sort the passages in a decreasing order.", 
    "model_group_id": "wlcnb4kBJ1eYAeTMHlV6",
    "model_format": "TORCH_SCRIPT", 
    "function_name": "TEXT_SIMILARITY", 
    "model_task_type": "TEXT_SIMILARITY", 
    "model_content_hash_value": "628183b3d249dd45b82626972828125f7ad98c3b11c0ef1f2261c575600102d6", 
    "model_config": {
        "model_type": "bert", 
        "embedding_dimension": 384, 
        "framework_type": "huggingface_transformers", 
        "all_config": "{\n  \"_name_or_path\": \"cross-encoder/ms-marco-MiniLM-L-6-v2\",\n  \"architectures\": [\n    \"BertForSequenceClassification\"\n  ],\n  \"attention_probs_dropout_prob\": 0.1,\n  \"classifier_dropout\": null,\n  \"gradient_checkpointing\": false,\n  \"hidden_act\": \"gelu\",\n  \"hidden_dropout_prob\": 0.1,\n  \"hidden_size\": 384,\n  \"id2label\": {\n    \"0\": \"LABEL_0\"\n  },\n  \"initializer_range\": 0.02,\n  \"intermediate_size\": 1536,\n  \"label2id\": {\n    \"LABEL_0\": 0\n  },\n  \"layer_norm_eps\": 1e-12,\n  \"max_position_embeddings\": 512,\n  \"model_type\": \"bert\",\n  \"num_attention_heads\": 12,\n  \"num_hidden_layers\": 6,\n  \"pad_token_id\": 0,\n  \"position_embedding_type\": \"absolute\",\n  \"sbert_ce_default_activation_function\": \"torch.nn.modules.linear.Identity\",\n  \"transformers_version\": \"4.22.1\",\n  \"type_vocab_size\": 2,\n  \"use_cache\": true,\n  \"vocab_size\": 30522\n}\n"
    },
    "created_time": 1676073973126,
    "url": "https://artifacts.opensearch.org/models/ml-models/huggingface/cross-encoders/ms-marco-MiniLM-L-12-v2/1.0.2/torch_script/cross-encoders_ms-marco-MiniLM-L-12-v2-1.0.2-torch_script.zip"
}
```

요청이 올바르게 전달되었으면 아래와 같은 응답을 받게 됩니다.
```json
{
  "task_id": "cVeMb4kBJ1eYAeTMFFgj",
  "status": "CREATED"
}
```

task_id를 이용해 태스크 상태를 조회할 수 있습니다.
```
GET /_plugins/_ml/tasks/cVeMb4kBJ1eYAeTMFFgj
```
```json
{
  "model_id": "cleMb4kBJ1eYAeTMFFg4",
  "task_type": "REGISTER_MODEL",
  "function_name": "TEXT_SIMILARITY",
  "state": "COMPLETED",
  "worker_node": [
    "XPcXLV7RQoi5m8NI_jEOVQ"
  ],
  "create_time": 1689793598499,
  "last_update_time": 1689793598530,
  "is_async": false
}
```
조회 결과로 `model_id`가 없고 state가 `CREATED` 상태라면 아직 아티팩트를 다운로드 받고 있는 것이므로 기다려야 합니다. 
state가 `FAILED`라면 `error`가 응답에 포함됩니다. 
완료되었다면 `model_id`를 따로 메모합니다.

### Deploy a model
모델 인덱스로부터 읽어와 메모리에 모델을 적재하는 단계입니다. 
모델을 메모리에 올리기 위해 다음의 요청을 실행합니다. 
/models 다음에는 위 단계에서의 등록된 `model_id`를 넣어줍니다.
```
POST /_plugins/_ml/models/{model_id}/_deploy
```
이전처럼 응답으로부터 task_id를 받을 수 있는데 마찬가지로 태스크를 조회할 수 있습니다.
```json
{
  "task_id": "vVePb4kBJ1eYAeTM7ljG",
  "status": "CREATED"
}
```
```
GET /_plugins/_ml/tasks/vVePb4kBJ1eYAeTM7ljG
```
```json
{
  "model_id": "cleMb4kBJ1eYAeTMFFg4",
  "task_type": "DEPLOY_MODEL",
  "function_name": "TEXT_EMBEDDING",
  "state": "COMPLETED",
  "worker_node": [
    "n-72khvBTBi3bnIIR8FTTw"
  ],
  "create_time": 1689793851077,
  "last_update_time": 1689793851101,
  "is_async": true
}
```
배포 도중이라면 state가 `RUNNING`일 것이고 완료되었다면 위와 같이 `COMPLETED`로 표시됩니다.

모델 배포가 완료되었다면 Opensearch Dashboard에서 모델 리스트를 확인할 수 있습니다. **Opensearch plugins > Machine Learning** 탭에서 다음의 사진처럼 모델 목록을 확인할 수 있습니다.

<figure>
    <img src="https://opensearch.org/docs/latest/images/ml/ml-dashboard/deployed-models.png" alt="01" style="max-width: 100%; height: auto;">
    <figcaption>Opensearch Dashboard Machine Learning (refs: https://opensearch.org/docs/latest/ml-commons-plugin/ml-dashboard/)</figcaption>
</figure>

배포가 완료되었다면 Status에 Responding이라고 표시됩니다.

### Reranker Predict
모델을 사용할 때는 아래의 요청을 보내야 합니다.
```json
POST _plugins/_ml/models/<model_id>/_predict
{
    "query_text": "today is sunny",
    "text_docs": [
        "how are you",
        "today is sunny",
        "today is july fifth",
        "it is winter"
    ]
}
```
이 요청의 응답으로 다음과 같이 오게 됩니다.
```json
{
  "inference_results": [
    {
      "output": [
        {
          "name": "similarity",
          "data_type": "FLOAT32",
          "shape": [
            1
          ],
          "data": [
            -6.077798
          ],
          "byte_buffer": {
            "array": "Un3CwA==",
            "order": "LITTLE_ENDIAN"
          }
        }
      ]
    },
    {
      "output": [
        {
          "name": "similarity",
          "data_type": "FLOAT32",
          "shape": [
            1
          ],
          "data": [
            10.223609
          ],
          "byte_buffer": {
            "array": "55MjQQ==",
            "order": "LITTLE_ENDIAN"
          }
        }
      ]
    },
    {
      "output": [
        {
          "name": "similarity",
          "data_type": "FLOAT32",
          "shape": [
            1
          ],
          "data": [
            -1.3987057
          ],
          "byte_buffer": {
            "array": "ygizvw==",
            "order": "LITTLE_ENDIAN"
          }
        }
      ]
    },
    {
      "output": [
        {
          "name": "similarity",
          "data_type": "FLOAT32",
          "shape": [
            1
          ],
          "data": [
            -4.5923924
          ],
          "byte_buffer": {
            "array": "4fSSwA==",
            "order": "LITTLE_ENDIAN"
          }
        }
      ]
    }
  ]
}
```

만약 `opensearchpy`라이브러리를 이용하고 있다면 아래의 코드로 구현할 수 있습니다.
```python
from opensearchpy import OpenSearch

client = Opensearch(
    hosts=[{'host':'localhost','port':9200}],
    ...
)

payload = {
    "query_text": "today is sunny",
    "text_docs": [
        "how are you",
        "today is sunny",
        "today is july fifth",
        "it is winter"
    ]
}

response = client.transport.perform_request(
    method='POST',
    url=f'/_plugins/_ml/models/{model_id}/_predict',
    body=payload,
)
```

### Query
한국어 문서를 이용해 테스트를 진행했습니다. 경제 관련 문서들이 저장되어 있고 아래 쿼리를 이용해 결과를 확인했습니다.
```json
// Lexical Search
{
  "_source": {
    "include": ["text"]
  },
  "size": 5,
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "text": {
              "query": "2024 하반기 은행 가계대출 변화량을 설명해봐",
              "minimum_should_match": "0%",
              "operator": "or"
            }
          }
        }
      ]
    }
  }
}
```
```json
// Lexical Search + Rerank
{
  "_source": {
    "include": ["text"]
  },
  "size": 5,
  "query": {
    "bool": {
      "must": [
        {
          "match": {
            "text": {
              "query": "2024 하반기 은행 가계대출 변화량을 설명해봐",
              "minimum_should_match": "0%",
              "operator": "or"
            }
          }
        }
      ]
    }
  },
  "ext": {
    "rerank": {
      "query_context": {
        "query_text": "2024 하반기 은행 가계대출 변화량을 설명해봐"
      }
    }
  }
}
```

Rerank를 쿼리에 포함하려면 pipeline 정의가 필요합니다. 저는 아래 요청을 통해 파이프라인을 정의했습니다.

```json
PUT /_search/pipeline/rerank_pipeline
{
  "response_processors": [
    {
      "rerank": {
        "ml_opensearch": {
          "model_id": "{model_id}"
        },
        "context": {
          "document_fields": ["{passage_text_field}"]
        }
      }
    }
  ]
}
```

그 결과로 아래 이미지와 같이 Lexical search만 사용한 결과와 Rerank 모델을 같이 사용한 결과가 다름을 알 수 있습니다. 
첫 번째 이미지는 Lexical search만 사용한 결과이고 두 번째 이미지는 Lexical search + Rerank를 사용한 결과입니다.

<figure>
  <a href="https://lh3.googleusercontent.com/d/1j5YfrSJw9SLLRQlorbGL3optQR9gJg_D" onclick="return false;" data-lightbox="gallery">
    <img src="https://lh3.googleusercontent.com/d/1j5YfrSJw9SLLRQlorbGL3optQR9gJg_D" alt="02" style="max-width: 100%; height: auto;">
  </a>
  <figcaption>Only Lexical search</figcaption>
</figure>
<figure>
  <a href="https://lh3.googleusercontent.com/d/1SrB8HE_f5oo60dNTwXPBH1Yskyu3I9O4" onclick="return false;" data-lightbox="gallery">
    <img src="https://lh3.googleusercontent.com/d/1SrB8HE_f5oo60dNTwXPBH1Yskyu3I9O4" alt="03" style="max-width: 100%; height: auto;">
  </a>
  <figcaption>Lexical search + Rerank</figcaption>
</figure>

실제로 문서의 순위가 변경된 것을 볼 수 있습니다. 
이처럼 Opensearch 클러스터 내에서 Rerank 모델이 동작하여 문서의 순위를 변경할 수 있지만 그 만큼 지연시간이 증가하는 단점이 있습니다. 


## References
- [한국어 Reranker를 활용한 검색 증강 생성(RAG) 성능 올리기](https://aws.amazon.com/ko/blogs/tech/korean-reranker-rag/)
- [Retrieval-Augmented Generation (RAG) for Large Language Models on AWS](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot)
- [Opensearch docs - Machine learning](https://opensearch.org/docs/latest/ml-commons-plugin/)
