---
title: "Databricks Vector Search Index"
categories:
 - databricks
tags: [data-engineering, databricks]
---
## Databricks Vector Search
Databricks에는 벡터 검색을 위한 벡터 검색 엔드포인트 및 인덱스를 생성하고 사용할 수 있습니다. Databricks의 Unity Catalog에 등록되었으며 ChangeDataFeed가 활성화된 테이블에 대해서만 인덱스를 생성할 수 있고 Sync를 통해 테이블 업데이트시 인덱스도 같이 업데이트가 가능합니다.

이름처럼 Semantic 검색을 지원하고 있지만 BM25와 같은 Lexical Search도 함께 지원하고 있다는 점이 잘 알려지지 않는 것 같습니다. 그래서 이 글에선 Vector search 활용에 대해 정리해보려고 합니다.

## Vector Search Endpoint
벡터 서치 엔드포인트는 인덱스를 관리할 수 있는 인스턴스 같은 개념입니다. 워크스페이스 왼쪽 항목에서 Compute에 Vector Search 탭이 있고 여기에 생성한 엔드포인트 목록이 보이게 되고 각 엔드포인트에는 생성한 인덱스 목록도 함께 보이게 됩니다.

## Vector Search Index
벡터 서치 인덱스는 임베딩 벡터가 포함된 인덱스를 말합니다. 기존 테이블에서 임베딩 대상이 되는 컬럼에 대한 임베딩 벡터를 생성해서 `__db_review_vector`라는 이름의 컬럼에 삽입됩니다. 인덱스도 Unity Catalog의 관리를 받게 되고 테이블 목록과 함께 나타납니다.

### databricks-vectorsearch
Databricks에선 워크스페이스 환경에서 벡터 서치를 이용할 수 있는 파이썬 기반의 라이브러리를 제공 중에 있습니다. 아래 명령어를 통해 설치하고 사용할 수 있습니다.
```
%pip install databricks-vectorsearch
```

사용법은 간단합니다.
```python
from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()
index = client.get_index(endpoint_name="vector_search_endpoint", index_name="{catalog}.{schema}.{index_name}")

results = index.similarity_search(
    query_text="Their Outback Oatmeal and Austin Almond Biscotti were dry and flavorless.",
    columns=["review"],
    num_results=5,
)
```
```
# Output
{
    'manifest': {'column_count': 2, 'columns': [{'name': 'review'}, {'name': 'score'}]},
    'result': {
        'row_count': 5,
        'data_array': [
            ["Bakehouse in Wynwood, Miami, has disappointed me. ... I expected more from a renowned cookie company.", 0.9040976460331299],
            ...
        ]
    }
}
```
similarity_search의 기본값은 ANN이기 때문에 `query_type`에 아무런 값을 주진 않았다면 Semantic Search가 자동으로 적용됩니다. 

이름이 벡터 서치이기 때문에 Semantic Search만 지원하는 건가 싶지만 사실 그렇지는 않습니다. Lexical Search만 이용할 수도 있고 Semantic Search와 함께 사용하는 Hybrid Search도 지원하고 있습니다.
```python
from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()
index = client.get_index(endpoint_name="vector_search_endpoint", index_name="{catalog}.{schema}.{index_name}")

results = index.similarity_search(
    query_text="Their Outback Oatmeal and Austin Almond Biscotti were dry and flavorless.",
    columns=["review"],
    num_results=5,
    query_type="FULL_TEXT", # bm25
)

results = index.similarity_search(
    query_text="Their Outback Oatmeal and Austin Almond Biscotti were dry and flavorless.",
    columns=["review"],
    num_results=5,
    query_type="hybrid", # hybrid
)
```
단, Hybrid Search의 경우 RRF(Reciprocal Rank Fusion) 알고리즘으로 계산됩니다. 만약, CC(Convex Combination)를 적용하고 싶다면 `query_type`을 `FULL_TEXT`와 `ANN`으로 각각 검색하여 스코어 정규화를 거쳐 계산하는 함수를 작성해야 합니다.

CC는 간단하게 다음과 같이 구현했습니다.
```python
def get_cc_ensemble_results(lexical, semantic, alpha, beta):
    assert alpha + beta == 1

    max_scores = (
        lexical_results['result']['data_array'][0][1], 
        semantic_results['result']['data_array'][0][1]
    )
    lexical_search = [
        [doc[0], doc[1] / max_scores[0]] 
        for doc in lexical_results['result']['data_array']
    ]
    semantic_search = [
        [doc[0], doc[1] / max_scores[1]] 
        for doc in semantic_results['result']['data_array']
    ]

    assert lexical_search[0][1] == semantic_search[0][1] == 1.0

    ensemble_results = []
    for lexical_doc, semantic_doc in zip(lexical, semantic):
        score = alpha * lexical_doc[1] + beta * semantic_doc[1]
        ensemble_results.append([lexical_doc[0], score])
    ensemble_results.sort(key=lambda x: x[1], reverse=True)
    return ensemble_results

cc_ensemble_results = get_cc_ensemble_results(lexical_search, semantic_search, 0.8, 0.2)
print(cc_ensemble_results)
```

추가적으로, debug_level=1을 넣어주면 debug_info라는 필드가 추가되면서 처리 시간이 나타는데 `query_type`에 따라 달라지는 것을 볼 수 있습니다.
``` python
# query_type=ann
'debug_info': {'response_time': 321.0, 'ann_time': 98.0, 'embedding_gen_time': 215.0}

# query_type=FULL_TEXT
'debug_info': {'response_time': 22.0, 'ann_time': 16.0}

# query_type=hybrid
'debug_info': {'response_time': 364.0, 'ann_time': 96.0, 'embedding_gen_time': 260.0}
```

이 debug_info를 보면 lexical search에서 ann_time이라는 필드가 나오는 것으로 봐서 bm25 알고리즘이 아닐 수 있냐는 의문이 들 수도 있습니다. 내부 처리 로직을 볼 수는 없어서 자세히는 알 수 없지만 제 생각으로는 응답 시간 자체가 매우 짧아서 bm25 알고리즘은 맞고 출력에 같은 필드를 사용해서 저런 결과가 나온 것 같습니다.

### Reranker
databricks-vectorsearch에선 reranker를 사용할 수 있는 옵션도 제공하고 있습니다. 문서에도 설명되어 있고 사용할 수 있는 줄로 알고 있는데 Databricks Free-edition 워크스페이스에선 지원하지 않아 확인하지 못했습니다. Private Preview라던가 하는 안내도 없어서 아마 엔터프라이즈 워크스페이스에선 가능할 수도 있습니다.

사용법은 아래와 같습니다.
```python
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.reranker import DatabricksReranker

client = VectorSearchClient()
index = client.get_index(endpoint_name="vector_search_endpoint", index_name="{catalog}.{schema}.{index_name}")

results = index.similarity_search(
  query_text="Their Outback Oatmeal and Austin Almond Biscotti were dry and flavorless.",
  columns=["review"],
  num_results=10,
  query_type="full_text",
  reranker=DatabricksReranker(
      columns_to_rerank=["review"]
  )
)
```

만약, 위와 같은 기능을 엔터프라이즈 워크스페이스에서도 사용이 불가하다면 Reranker 모델을 직접 MLflow 모델로 등록하고 Serving Endpoint로 배포하여 사용해야 합니다. 사용한 모델은 'cross-encoder/ms-marco-MiniLM-L6-v2'입니다.
```python
import os
import sys
from typing import *
import pandas as pd

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

from sentence_transformers import CrossEncoder

def _patch_stream_isatty(): # isatty 에러를 방지하기 위해 사용했지만 피하는 법을 모르겠습니다
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and not hasattr(stream, "isatty"):
            stream.isatty = lambda: False

class CrossEncoderModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_name):
        self.model_name = model_name

    def load_context(self, context):
        _patch_stream_isatty()
        self.model = CrossEncoder(self.model_name)

    def predict(self, context, model_input: List[List[str]]) -> List[float]:
        return self.model.predict(model_input).tolist()


input_example = [
    ["How many people live in Berlin?", "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."],
    ["How many people live in Berlin?", "Berlin is well known for its museums."],
]

output_example = [-8.1875, 5.26171875]
signature = infer_signature(input_example, output_example)

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model( # log와 동시에 register까지 수행
        python_model=CrossEncoderModel("cross-encoder/ms-marco-MiniLM-L6-v2"),
        name="ms-marco-MiniLM-L6-v2",
        signature=signature,
        input_example=input_example,
        registered_model_name=f"{catalog}.{schema}.ms-marco-MiniLM-L6-v2" ,
        pip_requirements=[
            "sentence-transformers==5.2.2",
            "torch==2.10.0",
            "transformers==5.1.0"
        ]
    )
```

MLflow Model로 등록한 뒤 Serving Endpoint로 등록하면 생성된 엔드포인트로 입력을 보내 Score를 받아볼 수 있습니다.
```python
import mlflow.deployments
documents = [doc[0] for doc in results['result']['data_array']]

client = mlflow.deployments.get_deploy_client("databricks")
response = client.predict(
    endpoint="ms-marco-MiniLM-L6-v2",
    inputs={"inputs": [["Their Outback Oatmeal was dry and flavorless.", doc] for doc in documents]}
)

reranked = sorted([[doc, score] for doc, score in zip(documents, response['predictions'])], key=lambda x: -x[1])
print(reranked)
```

마지막으로, Databricks는 Vector Search라는 강력한 검색 기능을 제공하고 실제로 사용할 수 있습니다. 또한, Search 기능을 적용하려는 도메인에 따라 검색 방향을 다르게 잡아야 원하는 성능에 도달할 수 있습니다.
- 예를 들어, 주로 검색 쿼리와 문서가 문맥 관계가 높다면 Semantic Search 비중을 크게 주어야 하고 전문적인 단어가 자주 등장하면 Lexical Search 비중을 크게 주어야 합니다. 

Databricks의 Vector Search는 다양한 도메인에 적용할 수 있도록 옵션을 제공하고 있으며 Reranker를 통해 검색 순위를 재정렬할 수도 있습니다. 또한, Reranker를 배포한 것처럼 Colbert나 다른 검색에 특화된 모델도 배포하여 사용할 수 있다는 장점도 있기 때문에 여러가지 방법을 두고 가장 좋은 전략을 찾을 수 있으면 좋겠습니다.