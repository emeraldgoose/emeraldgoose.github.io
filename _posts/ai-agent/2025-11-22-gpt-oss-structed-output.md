---
title: Langchain GPT-OSS 구조화된 출력 트러블슈팅
categories:
  - ai-agent
tags: [ai-agent]
---
## Databricks-gpt-oss
사용한 LLM은 gpt-oss-20b 모델이고 databricks workspace에 통합된 모델을 사용했습니다. huggingface에 업로드된 gpt-oss 모델을 사용해본 경험은 없지만 같은 에러가 발생할 것으로 예상됩니다. 만약 똑같은 문제가 일어났다면 아래 해결 방법이 도움될 수 있을 것 같습니다.

```python
from databricks_langchain import ChatDatabricks

llm = ChatDatabricks(
    model='databricks-gpt-oss-20b',
    temperature=0.,
    extra_params={"reasoning_effort": "low"}
)
llm.invoke("hello?")
```

## v1 이전 Structured Output
Langchain이 최근 v1.0으로 업데이트되면서 기존 애플리케이션을 마이그레이션해야 하는 상황에 놓였습니다.

업데이트 이전 Langchain 애플리케이션에선 아래 코드처럼 ResponseSchema를 정의하여 Prompt에 통합하여 사용했습니다.
```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(
        name="answer",
        description="Answer",
        type="string"
    ),
    ResponseSchema(
        name="reason",
        description="Reason",
        type="string"
    ),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

chain = prompt | llm
response = chain.invoke({
    "question": question,
    "format_instruction": parser.get_format_instructions()
})

content = json.loads(response.content)[-1]['text']
parsed_content = parser.parse(content)
```

## v1 이후 Structured Output
최신 Langchain에선 Pydantic BaseModel을 이용해 표준화된 형태로만 사용 가능하도록 변경되었습니다. 아래는 Langchain 공식 문서의 예제입니다.

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from pydantic import BaseModel

class OutputSchema(BaseModel):
    summary: str
    sentiment: str

# Using ToolStrategy
agent = create_agent(
    model="gpt-4o-mini",
    tools=tools,
    # explicitly using tool strategy
    response_format=ToolStrategy(OutputSchema)  
)
```

## gpt-oss의 구조화된 출력 문제
타 모델의 경우 위의 예시처럼 response_format을 이용해 모델의 출력을 쉽게 제어할 수 있어 편리한 점이 있었지만 gpt-oss의 경우 제대로 적용되지 않았습니다.

랭체인에선 구조화된 출력을 제어할 수 있도록 langchain.agents.structured_output.ToolStrategy라는 것을 제공하고 있지만 gpt-oss에선 오히려 아래와 같은 에러가 발생하여 제대로 출력되지 않습니다.
```
BadRequestError: Error code: 400 - {'error_code': 'BAD_REQUEST', 'message': 'Model output is not in expected format, adjust parameters or prompt and try again.\n'}
```

### Solution
이를 해결하기 위해 랭체인에서 제공하는 ToolStrategy를 사용하지 않고 스키마를 직접 작성한 뒤 LLM에 bind하여 해결할 수 있습니다.

```python
from pydantic import BaseModel, Field

class ResponseFormat(BaseModel):
    answer: str = Field("Answer for the question")
    reason: str = Field("Reason for the answer")

response_format = {
    "type": "json_schema",
    "json_schema": {
        "strict": True,
        "name": ResponseFormat.__name__,
        "schema": ResponseFormat.model_json_schema(),
    },
}

llm.bind(response_format=response_format).invoke("hello?")
```

이를 이용하면 langchain.agents.create_agent에도 사용할 수 있습니다. create_agent의 response_schema를 이용하지 않고 bind된 LLM 모델을 넣어 사용하여 구조화된 출력을 기대할 수 있습니다.

```python
from langchain.agents import create_agent
from langchain.tools import tool

from pydantic import BaseModel, Field

class ResponseFormat(BaseModel):
    answer: str = Field("Answer for the question")
    reason: str = Field("Reason for the answer")

response_format = {
    "type": "json_schema",
    "json_schema": {
        "strict": True,
        "name": ResponseFormat.__name__,
        "schema": ResponseFormat.model_json_schema(),
    },
}

agent = create_agent(
    model=llm.bind(response_format=response_format),
    tools=[],
    system_prompt="You are helpful agent and your output should be in korean",
)

response = agent.invoke({"messages": [{"role": "human", "content": "What is aEtbdwetq?"}]})

import json
json.loads(response['messages'][-1].content)
```
```text
Output:
[
    {
        'type': 'reasoning',
        'summary': [
            {
                'type': 'summary_text',
                'text': 'Need to respond in Korean, JSON format with answer and reason. Unknown term. Probably explain unknown.'
            }
        ]
    },
    {
        'type': 'text',
        'text': '{
            "answer":"aEtbdwetq는 일반적으로 알려진 용어나 개념이 아니며, 현재로서는 특정한 의미를 파악하기 어렵습니다. 혹시 오타이거나 특정 분야에서 사용되는 전문 용어일 가능성이 있으니, 추가적인 맥락이나 정확한 철자를 알려주시면 보다 정확한 답변을 드릴 수 있습니다.",
            "reason":"사용자가 제시한 용어가 표준 사전이나 일반적인 지식 데이터베이스에 존재하지 않으며, 문맥이 부족해 정확한 정의를 제공하기 어려웠습니다. 따라서 추가 정보를 요청하는 것이 가장 합리적인 접근입니다."
        }'
    }
]
```

그러나 agent의 tool을 사용하면 이 구조가 깨지는 문제가 또 발생합니다. 이 경우에는 이전 방법처럼 프롬프트에 출력 스키마를 포함시켜 최종 출력에 스키마 구조를 따르도록 지시해야 합니다.
```python
from langchain.agents import create_agent
from langchain.tools import tool

from pydantic import BaseModel, Field

class ResponseFormat(BaseModel):
    answer: str = Field("Answer for the question")
    reason: str = Field("Reason for the answer")
    search: str = Field("Search result")

response_format = {
    "type": "json_schema",
    "json_schema": {
        "strict": True,
        "name": ResponseFormat.__name__,
        "schema": ResponseFormat.model_json_schema(),
    },
}

@tool
def search(query: str):
    "anything related to the query"
    return {"result": "Not found :("}

agent = create_agent(
    model=llm.bind(response_format=response_format),
    tools=[search],
    system_prompt=(
        "You are helpful agent and your output should be in korean."
        "you should to use structed format." 
        "output should be respected:\n{response_format}".format(
            response_format=ResponseFormat.model_json_schema()
        )
    )
)

response = agent.invoke({"messages": [{"role": "human", "content": "What is aEtbdwetq?"}]})

import json
json.loads(response['messages'][-1].content)
```
```text
Output
{
    'answer': 'aEtbdwetq는 일반적으로 알려진 용어나 개념이 아니며, 검색 결과에서도 해당 단어에 대한 정보를 찾을 수 없었습니다.',
    'reason': '검색 결과가 없었고, 해당 문자열이 특정 약어, 코드, 혹은 오타일 가능성이 높아 명확한 정의를 제공하기 어려웠습니다.',
    'search': 'Not found :('
}
```

또는 아래와 같은 프롬프트로 출력을 제어할 수도 있습니다.

{% raw %}
```python
class ResponseFormat(BaseModel):
    answer: str = Field("Answer for the question")
    reason: str = Field("Reason for the answer")
    search: str = Field("Search result")

json_schema = ResponseFormat.model_json_schema()

prompt = """You are helpful agent and your output should be in korean.
...
### Output Format
The output should be a Markdown code snippet formatted in the following schema, including the leading and trailing:
{{
    "type": "object",
    "properties": {json_schema},
    "required": {required}
}}""".format(
    json_schema=json_schema["properties"], 
    required=list(json_schema["properties"])
)
```
{% endraw %}

## Reference
- [Langchain Docs - Structured output](https://docs.langchain.com/oss/javascript/langchain/structured-output#structured-output)
- [[LangChain 번역] Structured output](https://rudaks.tistory.com/entry/LangChain-%EB%B2%88%EC%97%AD-Structured-output)
