---
title: Airflow task 디자인
categories:
  - airflow
tags: [airflow, Ops]
---
> __Apache Airflow 기반의 데이터 파이프라인__ 책의 내용 중 일부를 정리한 내용입니다.

# 태스크 디자인
Airflow의 백필링과 재실행 태스크는 원자성과 멱등성을 고려하여 태스크를 디자인해야 합니다.

## 원자성
Airflow에서 원자성 트랜잭션은 데이터베이스와 같이 모두 발생하거나 전혀 발생하지 않는, 더 이상 나눌 수 없는 작업으로 간주됩니다. Airflow의 태스크는 성공적으로 수행하여 적절한 결과를 생성하거나 시스템 상태에 영향을 미치지 않고 실패하도록 정의합니다.

```python
def _calculate_stats(**context):
    """이벤트 데이터 통계 계산하기"""
    input_path = context['templates_dict']['input_path']
    output_path = contesxt['templates_dict']['output_path']

    events = pd.read_json(input_path)
    stats = events.groupby(['date', 'user']).size().reset_index()
    stats.to_csv(output_path, index=False)

    email_stats(stats, email='user@example.com')

def _email_stats(stats, email):
    """Send an email..."""
    print(f"Sending stats to {email}...")

calculate_stats = PythonOperator(
    task_id="calculate_stats",
    python_callable=_calculate_stats,
    templates_dict={
        "input_path": "/data/events/{{ds}}.json",
        "output_path": "/data/stats/{{ds}}.csv",
    },
    dag=dag,
)
```

위의 코드의 문제점은 CSV 작성 후 이메일을 보내면 단일 기능에서 두 가지 작업을 수행하게 되어 원자성이 깨지게 됩니다. 만약 `email_stats` 함수가 실패하면 이미 output_path 경로에 통계에 대한 출력 파일이 저장되어 있기 때문에 통계 발송이 실패했음에도 작업이 성공한 것처럼 보이게 됩니다. 이 기능을 원자성을 유지하는 방식으로 구현하기 위해 이메일 발송 기능을 별도의 태스크로 분리하여 구현합니다.

```python
def _send_stats(email, **context):
    stats = pd.read_csv(context['templates_dict']['stats_path'])
    email_stats(stats, email=email)

send_stats = PythonOperator(
    python_callable=_send_stats,
    task_id='send_stats',
    op_kwargs={"email":"user@example.com"},
    templates_dict={"stats_path":"/data/stats/{{ds}}.csv"}
    dag=dag,
)

calculated_stats >> send_stats
```

이렇게 하면 이메일 전송이 실패해도 더 이상 `calculate_stats` 작업의 결과에 영향을 주지 않고 `_send_stats`만 실패하도록 하여 두 작업 모두 원자성을 유지할 수 있습니다.

하지만 모든 작업을 개별 태스크로 분리하여 모든 태스크를 원자성을 유지할 수 있다고 생각해서는 안됩니다. 

만약 이벤트 API를 호출하기 전에 로그인해야 하는 경우, 일반적으로 API를 인증하기 위한 인증 토큰을 가져오기 위해 추가적인 API 호출이 필요하며 그 이후에 이벤트 API를 호출할 수 있습니다. "하나의 작업 = 하나의 태스크"라는 접근성에 따라 개별 작업으로 분리했을 때 두 번째 태스크(이벤트 API 호출) 바로 전에 로그인을 위한 토큰을 가져오는 API를 호출하는 태스크를 반드시 수행해야 하므로 두 태스크 사이에 강한 의존성이 발생합니다. 이러한 의존성은 단일 태스크 내에서 두 작업을 모두 유지하여 하나의 일관된 태스크 단위를 형성하는 것이 더 나을 수 있습니다.

대부분의 airflow 오퍼레이터는 이미 원자성을 유지하도록 설계되어, 오퍼레이터가 내부적으로 인증과 같이 밀접하게 연결된 작업을 수행할 수 있는 옵션들이 있습니다. 좀 더 유연한 파이썬 및 배시 오퍼레이터 사용시 태스크가 원자성을 유지할 수 있도록 주의를 기울여야 할 필요가 있습니다.

## 멱등성
Airflow의 멱등성은 동일한 입력으로 동일한 태스크를 여러 번 호출해도 결과는 항상 같아야 하는 속성입니다.

```python
fetch_events = BashOperator(
    task_id='fetch_events',
    bash_command=(
      'mkdir -p /data/events && ',
      'curl -o /data/events/{{ds}}.json',
      'http://localhost:5000/events?',
      'start_date={{ds}}&',
      'end_date={{next_ds}}'
    ),
    dag=dag,
)
```

특정 날짜에 이 태스크를 다시 실행하면 이전 실행과 동일한 이벤트 데이터 세트를 가져오고 /data/events 폴더에 있는 기존 JSON 파일에 동일한 결과를 덮어쓰게 됩니다. 따라서 이 이벤트 가져오기 태스크는 효럭이 없게 됩니다.

보통 데이터를 쓰는 태스크는 기존 결과를 확인하거나 이전 태스크 결과를 덮어쓸지 여부를 확인하여 멱등성을 유지합니다. 시간별로 파티션 데이터 세트가 저장되는 경우 파티션 범위로 결과를 덮어쓸 수 있기 때문에 Upsert를 이용하면 비교적 간단하게 작업할 수 있습니다. 보다 일반적인 애플리케이션에서는 작업의 모든 과정에서 오류 발생 상황을 고려해 멱등성이 보장되는지 확인해야 합니다.
