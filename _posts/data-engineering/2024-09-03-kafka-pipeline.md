---
title: "ksqlDB: 실시간 데이터 처리 후 시각화까지"
categories:
 - data-engineer
tags: [data-engineering]
---
# ksqlDB
ksqlDB는 Kafka Streams에 기반하는 SQL 엔진입니다. ksqlDB는 Kafka topic에 이벤트 스트리밍 애플리케이션을 구축할 수 있는 쿼리 계층을 제공합니다. Kafka Streams와 달리 ksqlDB는 SQL로 새로운 스트림을 생성하거나 Materialized View를 생성할 수 있습니다.

## 왜 ksqlDB?
다음과 같이 실시간 데이터를 처리하기 위한 파이프라인을 가정해볼 수 있습니다.
```
Source Database -> CDC(Debezium-connector) -> Kafka Cluster -> Kafka-sink-connector -> Sink Database
```

만약, 실시간 처리가 필요한 데이터가 개인정보라면 Kafka Cluster에서 개인식별 정보를 마스킹 작업을 하는 스트리밍 프로세스가 추가되어야 합니다. 또는 kafka connector를 직접 개발하여 중간에 데이터를 처리하는 로직을 추가해야 할지도 모릅니다.

하지만, ksqlDB는 스트리밍 데이터 파이프라인을 작성할 수 있고 Kafka를 저장소로 활용하게 됩니다. 여러개의 스트림과 테이블을 sql로 관리할 수 있어 번거로운 과정을 줄일 수 있습니다. 단순히 스트림이나 테이블을 생성하는 것이 아닌 SQL처럼 스트림-스트림, 스트림-테이블, 테이블-테이블을 조인할 수도 있습니다.

## Persistent, Push, Pull query
1. Persistent Query:  
Persistent query는 이벤트 행을 무기한으로 처리하는 서버측 쿼리입니다. `CREATE STREAM AS SELECT`나 `CREATE TABLE AS SELECT`가 이에 해당합니다.

2. Push Query:  
Push query는 클라이언트가 실시간으로 변경된는 결과를 구독하는 쿼리입니다. push query를 사용하면 스트림이나 테이블의 결과를 실시간으로 받아볼 수 있습니다. 이러한 쿼리는 비동기 작업에 적합한 쿼리입니다.

3. Pull Query:  
Pull query는 클라이언트가 현재 결과를 가져오게 하는 쿼리입니다. 새 이벤트가 도착하면 pull query를 통해 Materialized View를 업데이트합니다. 이러한 쿼리는 요청/응답 작업에 적합한 쿼리입니다.

## 파이프라인
> 파이프라인을 구축할 때는 Docker Desktop에서 제공하는 쿠버네티스를 사용했고 모니터링 툴은 k9s와 Docker Desktop을 이용했습니다.

> 로컬 쿠버네티스를 이용하므로 하나의 노드만 사용하게 됩니다. 그래서 각각의 네임스페이스를 생성하고 네임스페이스에 맞게 배포했습니다.
<figure>
  <img width="1365" alt="19" src="https://github.com/user-attachments/assets/8f4dacc6-dd6b-449a-8bd0-1c2b1c755c64">
  <figcaption>pipeline</figcaption>
</figure>

### 데이터 생성
Source 데이터베이스인 Postgresql에 가상의 은행 데이터를 생성하여 적재했습니다.

account, bank_transaction, transaction_log 테이블을 만들고 sqlalchemy를 이용해 source db에 데이터를 적재했습니다. 아래는 테이블의 스키마를 정의한 것입니다.

```python
# entities.py
from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Account(Base):
    __tablename__ = 'account'

    account_id = Column(String, primary_key=True, nullable=False)
    account_holder_name = Column(String, nullable=False) # hash
    account_balance = Column(Integer, nullable=False)


class Transaction(Base):
    __tablename__ = "bank_transaction"
    
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    date = Column(TIMESTAMP, nullable=False)
    account_id = Column(String, ForeignKey(Account.account_id), nullable=False)
    amount = Column(Integer, nullable=False)
    type = Column(String, nullable=False) # deposit, withdraw


class Transaction_log(Base):
    __tablename__ = 'transaction_log'
    
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    transaction_id = Column(Integer, ForeignKey(Transaction.id), nullable=True)
    date = Column(TIMESTAMP, nullable=False)
    event = Column(String, nullable=False) # Success, Failed
```

랜덤하게 계좌를 생성하거나 거래를 시도하게 되고 거래 시 금액도 랜덤하게 결정됩니다. 만약 출금시 계좌 잔액이 부족한 경우 bank_transaction 테이블에 기록되지 않지만 로그는 남게 했습니다. 그래서 transaction_log 테이블에 모든 거래 시도에 대해 성공 혹은 실패를 기록하게 됩니다.

### Postgres 배포
Postgresql을 k8s에 배포하기 위해 deployment.yaml, configmap.yaml, service.yaml을 작성했습니다.
```
# postgresql/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    postgresql: postgresql
  name: postgresql
spec:
  replicas: 1
  selector:
    matchLabels:
      postgresql: postgresql
  template:
    metadata:
      labels:
        postgresql: postgresql
    spec:
      containers:
        - name: postgresql
          image: bitnami/postgresql:latest
          env:
            - name: POSTGRESQL_PASSWORD
              value: postgres
          volumeMounts:
            - name: postgres-config-volume
              mountPath: /bitnami/postgresql/conf/postgresql.conf
              subPath: postgresql.conf
          ports:
            - containerPort: 5432 # original port 5432
      hostname: postgresql
      volumes:
        - name: postgres-config-volume
          configMap:
            name: postgres-config
```
configmap은 postgresql이 initdb를 실행할 때 설정값을 변경하기 위해 필요합니다.

listen_address는 기본값이 127.0.0.1이므로 다른 파드에서 접근하기 위해 모두 허용하는 0.0.0.0으로 변경해야 합니다. wal_level(Write Ahead Log)는 Debezium에서 캡쳐를 위해 logical 설정이 필요하기 때문에 변경했습니다.
```
# postgresql/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
data:
  postgresql.conf: |
    wal_level = logical
    listen_addresses = '0.0.0.0'
```

service.yaml은 내부 도메인을 이용하므로 포트만 연결해줍니다. 클러스터 내 파드와 연결된 서비스끼리 통신할 수 있도록 `{pod.spec.hostname}.{namespace}.svc.cluster.local`이라는 도메인이 자동 부여되고 파드 내에서만 사용할 수 있습니다. 저 도메인을 사용하려면 hostname을 deployment.yaml에 넣어줘야 합니다.
```
# postgresql/service.yaml
apiVersion: v1
kind: Service
metadata:
  labels:
    postgresql: postgresql
  name: postgresql
spec:
  type: ClusterIP
  ports:
    - name: "5432"
      port: 5432
      targetPort: 5432
  selector:
    postgresql: postgresql
```

아래 명령어를 입력하여 postgresql을 배포합니다.
```
kubectl create ns postgres
kubectl apply -f postgres/ -n postgres
```

postgresql이 배포되면 기본적으로 postgres 데이터베이스가 생성됩니다. 다음 Kafka로부터 데이터를 받을 sink 데이터베이스를 생성하기 위해 아래 명령어를 확인해서 pod 이름을 조회한 후 파드에 접속해 sink 데이터베이스를 생성합니다.
```
kubectl get po -n postgres
```
```
kubectl exec -it <postgresql-pod-name> -n postgres -- createdb -U postgres sink
```
명령어를 실행하면 패스워드를 물어보는데 deployment.yaml에 POSTGRESQL_PASSWORD에 설정한 값을 입력합니다.

만약 db가 생성되었는지 확인하고 싶다면 아래 명령어로 확인가능합니다.
```
kubectl exec -it <postgresql-pod-name> -n postgres -- psql -U postgres -c "\list"
```

### Kafka 배포
Kafka는 kafka, kafka-connect, kafka-ui, ksqldb-server, schema-registry, zookeeper를 배포해야 합니다.

저는 deployment, service를 모두 작성하지는 않았고 docker-compose.yml로 먼저 작성 후에 kompose를 사용해서 deployment와 service로 변환 후 환경변수들만 수정해서 사용했습니다. 모든 yaml을 올리는 것은 너무 길어지므로 환경변수들만 작성하겠습니다.

```
# kafka/broker-deployment.yaml
KAFKA_LISTENERS=INTERNAL://:9092,EXTERNAL://:29092
KAFKA_ADVERTISED_LISTENERS=INTERNAL://broker:9092,EXTERNAL://localhost:29092
KAFKA_BROKER_ID=1
KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS=0
KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=INTERNAL:PLAINTEXT,EXTERNAL:PLAINTEXT
KAFKA_INTER_BROKER_LISTENER_NAME=INTERNAL
KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1
KAFKA_TRANSACTION_STATE_LOG_MIN_ISR=1
KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR=1
KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
```
```
# kafka/ksqldb-server-deployment.yaml
KSQL_BOOTSTRAP_SERVERS=broker:9092
KSQL_KSQL_LOGGING_PROCESSING_STREAM_AUTO_CREATE=true
KSQL_KSQL_LOGGING_PROCESSING_TOPIC_AUTO_CREATE=true
KSQL_LISTENERS=http://0.0.0.0:28088
KSQL_KSQL_SCHEMA_REGISTRY_URL=http://schema-registry:8081
```
```
# kafka/schema-registry-deployment.yaml
SCHEMA_REGISTRY_HOST_NAME=schema-registry.kafka.svc.cluster.local
SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS=broker:9092
SCHEMA_REGISTRY_LISTENERS=http://0.0.0.0:8081
```
```
# kafka/zookeeper-deployment.yaml
ZOOKEEPER_CLIENT_PORT=2181
ZOOKEEPER_TICK_TIME=2000
```
```
# kafka/kafka-connect-deployment.yaml
CONNECT_BOOTSTRAP_SERVERS=broker:9092
CONNECT_GROUP_ID=1
CONNECT_CONFIG_STORAGE_TOPIC=kafka-configs
CONNECT_CONFIG_STORAGE_REPLICATION_FACTOR=1
CONNECT_OFFSET_STORAGE_TOPIC=kafka-offsets
CONNECT_OFFSET_STORAGE_REPLICATION_FACTOR=1
CONNECT_STATUS_STORAGE_TOPIC=kafka-status
CONNECT_STATUS_STORAGE_REPLICATION_FACTOR=1
CONNECT_KEY_CONVERTER=org.apache.kafka.connect.json.JsonConverter
CONNECT_VALUE_CONVERTER=org.apache.kafka.connect.json.JsonConverter
CONNECT_REST_ADVERTISED_HOST_NAME=kafka-connect.kafka.svc.cluster.local
CONNECT_VALUE_CONVERTER_SCHEMA_REGISTRY_URL=http://schema-registry:8081
```
postgresql과 연결하기 위해 source connector로 debezium connector를 사용하고 sink connector로 jdbc sink connector를 사용할 것입니다. 카프카 커넥터는 confluent hub를 이용해 설치를 진행합니다. 카프카 커넥터를 배포할 때는 따로 Dockerfile을 작성하고 빌드한 이미지를 사용해야 합니다. Dcokerfile을 작성해야 하는 이유는 confluent-hub로 설치한 커넥터는 목록으로 바로 잡히지 않고 재시작해야 목록으로 잡히기 때문입니다.  

실제로 confluentinc/cp-kafka-connect 이미지를 실행시켜 플러그인 설치 직후 `curl -X GET localhost:8083/connector-plugins` 명령어로 플러그인 목록을 보게 되면 설치한 플러그인들이 목록으로 나타나지 않습니다. 이 목록에 없다면 당연히 커넥터 설정 시 해당 커넥터를 사용하지 못하기 때문에 배포 전에 플러그인을 받고 커넥터를 시작하는 이미지를 만들어야 합니다.
```
# kafka-connector/Dockerfile
FROM confluentinc/cp-kafka-connect-base:latest

RUN confluent-hub install --no-prompt confluentinc/kafka-connect-jdbc:10.7.11 && \
    confluent-hub install --no-prompt debezium/debezium-connector-postgresql:2.5.4
```
```
docker build --no-cache -t kafka-connector:latest kafka/kafka-connector/
```

대시보드 배포를 위해 `provectuslabs/kafka-ui` 이미지를 사용했습니다. 대안으로 confluent/control-center도 있습니다. 대시보드는 localhost:8080으로 접속할 수 있도록 service.yaml에서 type을 LoadBalancer 혹은 NodePort로 변경해야 합니다.
```
# kafka/kafka-ui-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kafka-ui
spec:
  selector:
    matchLabels:
      app: kafka-ui
  replicas: 1
  template:
    metadata:
      labels:
        app: kafka-ui
    spec:
      containers:
      - name: kafka-ui
        image: provectuslabs/kafka-ui:latest
        ports:
          - containerPort: 8080
        env:
          - name: KAFKA_CLUSTERS_0_NAME
            value: broker
          - name: DYNAMIC_CONFIG_ENABLED
            value: "true"
          - name: KAFKA_CLUSTERS_0_KAFKACONNECT_0_NAME
            value: kafka-connect
          - name: KAFKA_CLUSTERS_0_KAFKACONNECT_0_ADDRESS
            value: "http://kafka-connect:8083"
          - name: KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS
            value: "broker:9092"
          - name: KAFKA_CLUSTERS_0_ZOOKEEPER
            value: "zookeeper:2181"
          - name: KAFKA_CLUSTERS_0_KSQLDBSERVER
            value: "http://ksqldb-server:28088"
          - name: KAFKA_CLUSTERS_0_SCHEMAREGISTRY
            value: "http://schema-registry:8081"
```
```
# kafka/kafka-ui-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: kafka-ui
spec:
  type: LoadBalancer
  selector:
    app: kafka-ui
  ports:
  - protocol: TCP
    port: 8080
    targetPort: 8080
```
대시보드가 배포된 후 localhost:8080으로 접속하면 아래와 같은 화면을 볼 수 있습니다.

<figure>
    <a href="https://drive.google.com/file/d/1XUDVuAmSgqtcWPh8ST-f1N6bfhU8PGG0/view?usp=sharing"><img src="https://1drv.ms/i/s!AoC6BbMk0S9Qmzx_0fT-PPlqG5Wm?embed=1&width=2544&height=1195"></a>
    <figcaption>UI For Apache Kafka</figcaption>
</figure>

Kafka Connect와 KSQL DB 탭은 kafka-connect나 ksqldb-server와 연결되지 않으면 사이드바 목록에 나타나지 않습니다. 만약 사이드바에 없다면 kafka-connect와 ksqldb의 상태를 확인해야 합니다.

이전 배포한 Postgresql과 연결하기 위해 Connector를 생성합니다.
```
{
    "name": "debezium-postgresql-source-connector",
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgresql.postgres.svc.cluster.local",
    "database.port": "5432",
    "database.user": "postgres",
    "database.password": “postgres”,
    "database.dbname": "postgres",
    "tasks.max": "1",
    "slot.name": "debezium_test",
    "plugin.name": "pgoutput",
    "topic.prefix": "integrate",
    "key.converter": "io.confluent.connect.avro.AvroConverter",
    "key.converter.schemas.enable": "true",
    "key.converter.schema.registry.url": "http://schema-registry.kafka.svc.cluster.local:8081"
    "value.converter": "io.confluent.connect.avro.AvroConverter",
    "value.converter.schemas.enable": "true",
    "value.converter.schema.registry.url": "http://schema-registry.kafka.svc.cluster.local:8081",
}
```

kafka-connect가 접속하는 `database.user`의 role이 superuser, replication을 가지고 있어야 합니다. connector가 postgresql에 접속해서 slot을 생성해야 하는데 이 권한이 superuser와 replication이기 때문입니다. 만약 username이 postgres가 아니라면 다음의 명령어를 사용해 권한을 부여합니다.
```
kubectl exec -it <postgres-pod> -n postgres -- psql -U postgres -c "alter user <username> with superuser replication;"
```

schema-registry를 배포했기 때문에 AvroConverter를 사용할 수 있습니다. schema-registry는 데이터 스키마를 저장하고 불러오는 공간으로 사용자가 스키마를 입력하지 않아도 알아서 스키마를 관리합니다. 뒤에 나올 ksqldb를 사용할 때도 스키마 입력을 하지 않아도 되는 장점도 있습니다. connector가 연결되면 아래와 같이 커넥터 목록들을 볼 수 있습니다.

<figure>
    <a href="https://drive.google.com/file/d/1yNMwz2tnhVfvBVyfvV_h0rg4p2i8YG0K/view?usp=sharing"><img src="https://1drv.ms/i/s!AoC6BbMk0S9QmzpLv8yOSVwta8q4?embed=1&width=2345&height=687"></a>
    <figcaption>Connectors</figcaption>
</figure>

**ksqlDB 설정**
대시보드에서 ksqlDB 탭으로 들어가면 다음과 같은 화면을 볼 수 있습니다.

<figure>
    <a href="https://drive.google.com/file/d/1zyoubY48lF0Yzb7cmJDE5vf5zuhrXNC3/view?usp=sharing"><img src="https://1drv.ms/i/s!AoC6BbMk0S9QmzlLlxoK6QOdJKjj?embed=1&width=2342&height=409"></a>
    <figcaption>ksqlDB</figcaption>
</figure>

이미 스트림과 테이블을 정의했기 때문에 목록에 보이고 있고 초기 상태는 스트림에 한개만 정의되어 있습니다.

오른쪽 상단에 `Execute KSQL Request`버튼을 누르면 쿼리를 실행할 수 있는 공간을 볼 수 있습니다. 아래 사진은 bank_transaction_base라는 스트림을 생성하고 실행한 모습입니다.

<figure>
    <a href="https://drive.google.com/file/d/1e9xQBs4njpdKk4Pio6nukwE2xZYugYsR/view?usp=sharing"><img src="https://1drv.ms/i/s!AoC6BbMk0S9Qmz0O6ZFyrCda0Gob?embed=1&width=2343&height=1175"></a>
</figure>

source-connector로 데이터를 가져오면 topic에 메시지가 쌓이게 되는데 ksqlDB에서는 이 메시지를 바로 사용하지 못합니다. 그래서 stream을 먼저 생성하고 그 스트림을 사용하여 어떠한 처리 로직이 반영된 스트림 혹은 테이블을 만들 수 있습니다.

```sql
create stream account with (kafka_topic=‘integrate.public.account’, value_format=‘avro’);
```

AvroConvert를 사용하고 있으므로 format을 avro로 설정하면 스트림을 생성할 수 있습니다. 만약 AvroConvert를 사용하지 않는다면 스키마를 지정해줘야 합니다.

스키마를 지정해야 하는 경우 주의해야 할 점은 source db의 스키마가 아닌 connector가 생성하는 구조로 스키마를 지정해야 합니다. kafka topic 메시지 하나가 다음과 같다면 작성해야 할 sql 예시를 들어보겠습니다.

```json
{
    "after": {
        "account_id": "566740694069",
        "account_holder_name": "8202b6c341fe41f74b0749678fa87f9d56d89f9b83f48277a91a0a61a01cd183",
        "account_balance": 0
    },
    "source": {
        "version": "2.5.4.Final",
        "connector": "postgresql",
        "name": "integrate",
        "ts_ms": 1725335089064,
        "snapshot": "false",
        "db": "postgres",
        "sequence": "[\"26901096\",\"26901200\"]",
        "schema": "public",
        "table": "transaction_log",
        "txId": 748,
        "lsn": 26901200
    },
    "op": "c",
    "ts_ms": 1725335089547
}
```
```sql
create stream account (
    after struct<account_id varchar, account-holder-name varchar, account-balance bigint>, 
    ts_ms bigint
) with (
    kafka_topic=‘integrate.public.account’, 
    value_format=‘json’
);
```
메시지 구조와 스키마 구조가 맞지 않다면 push query 실행 시 데이터는 null로 채워집니다.

저는 bank_transaction 데이터를 사용해서 count_type_by_account, success_ratio_by_account, transaction_volume_per_minute이라는 테이블을 생성했습니다.

```sql
create table count_type_by_account as
    select 
        bt.after->account_id, 
        count(case when bt.after->type='deposit' then 1 end) as deposit_count, 
        count(case when bt.after->type='withdraw' then 1 end) as withdraw_count
    from bank_transaction_base bt
    group by bt.after->account_id
    emit changes;
```
이 쿼리처럼 account_id로 group by하려면 반드시 select 절에 account_id가 등장해야 합니다. `emit changes`는 참조하는 스트림에 변화가 생길때 이 테이블을 업데이트합니다.
```sql
create table success_ratio_by_account as
  select
    bt.after->account_id as account_id,
    count(case when tl.after->event='Success' then 1 end) as success_count,
    count(*) as tot,
    round(cast(count(case when tl.after->event = 'Success' then 1 end) as double) / cast(count(*) as double), 2) as success_rate
  from bank_transaction_base bt
  left join transaction_log_base tl within 10 minutes on bt.after->id = tl.after->transaction_id
  group by bt.after->account_id
  emit changes;
```
이 쿼리에서 스트림인 bank_transaction_base와 transaction_log_base를 조인하여 현재 시점을 기준으로 10분 이내로 들어온 메시지들을 사용하여 테이블을 업데이트합니다.
```sql
create table transaction_volume_per_minute as
    select
        after->account_id as account_id,
        sum(after->amount) as total_amount,
        windowstart as window_start,
        windowend as window_end,
        from_unixtime(windowstart) as time_start,
        from_unixtime(windowend) as time_end
    from bank_transaction_base
    window tumbling(size 1 minute)
    group by after->account_id
    emit changes;
```
`window tumbling(size 1 minute)`은 1분간격을 의미합니다. 즉, account_id마다 1분간 amount 총합을 계산하는 쿼리입니다.

테이블과 스트림을 생성하면 topic에 자동 등록되고 메시지로 쌓이게 됩니다.

<figure>
    <a href="https://drive.google.com/file/d/1XKq0Pe5ikGZw3GBXiNRseJaWf9NbcoLD/view?usp=sharing"><img src="https://1drv.ms/i/s!AoC6BbMk0S9QmzcQq59oeLhF9RJl?embed=1&width=2339&height=1005"></a>
    <figcaption>topics</figcaption>
</figure>

그리고 schema-registry에도 스키마가 등록됩니다.

<figure>
    <a href="https://drive.google.com/file/d/1QQa_1B0vc-zz7WMiiwcYc-iL5h-JNsTl/view?usp=sharing"><img src="https://1drv.ms/i/s!AoC6BbMk0S9Qmzh4uvzmuM_QLZMt?embed=1&width=2340&height=1123"></a>
    <figcaption>schema-registry</figcaption>
</figure>

다시 kafka-connect로 돌아와서 Sink 커넥터 설정을 진행합니다. 저는 count_type_by_account, success_ratio_by_account, transaction_volume_per_minute topic을 sink db에 적재했습니다.
```
{
    "name": "postgresql-sink-connector-topic-success-ratio-by-account",
    "connector.class": "io.confluent.connect.jdbc.JdbcSinkConnector",
    "connection.url": "jdbc:postgresql://postgresql.postgres.svc.cluster.local:5432/sink",
    "connection.user": "postgres",
    "connection.password": "postgres",
    "topics": "SUCCESS_RATIO_BY_ACCOUNT",
    "tasks.max": "1",
    "insert.mode": "upsert",
    "auto.evolve": "true",
    "auto.create": "true",
    "pk.mode":"record_key",
    "pk.field":"account_id",
    "key.converter": "io.confluent.connect.avro.AvroConverter",
    "key.converter.schemas.enable": "true",
    "key.converter.schema.registry.url": "http://schema-registry.kafka.svc.cluster.local:8081"
    "value.converter": "io.confluent.connect.avro.AvroConverter",
    "value.converter.schemas.enable": "true",
    "value.converter.schema.registry.url": "http://schema-registry.kafka.svc.cluster.local:8081",
}
```

### Superset 배포
Superset은 따로 yaml로 작성하지 않고 helm을 이용하여 배포했습니다. helm 통해 superset, redis, postgresql, superset-worker가 배포됩니다.

먼저 superset 네임스페이스를 생성합니다.
```
kubectl create ns superset
```

다음 helm repo를 추가하고 업데이트합니다.
```
helm repo add superset http://apache.github.io/superset/
helm repo update
```

superset 최근 이슈 중에 default secret key 문제가 있어 values.yaml 파일에 secret key를 등록해야 합니다. secret key는 openssl 명령어로 실행된 값을 넣어줍니다.
```
openssl rand -base64 42
```
```
# superset/values.yaml
extraSecretEnv:
  SUPERSET_SECRET_KEY: {SECRET_KEY}
```
```
helm install superset superset/superset -f superset/values.yaml -n superset
```
또는 --set 명령어로 values.yaml 파일을 overwrite합니다.
```
helm install superset superset/superset -n superset --set extraSecretEnv.SUPERSET_SECRET_KEY=$(openssl rand -base64 42)
```

다음 superset에 접속하기 위해 포트포워드를 설정하고 localhost:8088로 접속합니다. 기본 id와 passwowrd는 superset입니다.
```
kubectl port-forward superset -n superset 8088:8088
```

접속 후 database를 먼저 연결해줍니다. 오른쪽 상단 `+`에서 데이터베이스를 연결하거나 `Settings`에서 데이터베이스 커넥션 관리 탭을 눌러 관리할 수 있습니다.

<figure>
    <a href="https://drive.google.com/file/d/1MNCZA5gVFc4Ok1hiQ8_M9UuS_PvAZStN/view?usp=sharing"><img src="https://1drv.ms/i/s!AoC6BbMk0S9Qm0IU71HSJmWMKCVj?embed=1&width=2553&height=1219"></a>
    <figcaption>superset, connect a database</figcaption>
</figure>

PostgreSQL을 눌러 다음과 같이 입력하고 `Connect`버튼을 누릅니다.
```
host: postgresql.postgres.svc.cluster.local
port: 5432
database: postgres
username: postgres
password: postgres
display name: PostgreSQL-source
```
```
host: postgresql.postgres.svc.cluster.local
port: 5432
database: sink
username: postgres
password: postgres
display name: PostgreSQL-sink
```

다음 추가 설정으로 SQL Lab 탭에서 Allow DML을 체크하여 `Finish` 버튼을 누릅니다.

SQL Lab에서는 데이터베이스에 쿼리를 날릴 수 있습니다. 오른쪽에 데이터베이스를 선택하면 해당 데이터베이스에 직접 쿼리를 날려 결과를 볼 수 있습니다.
<figure>
    <a href="https://drive.google.com/file/d/1gctLb0021N_7nV9OWO4uJCeZbGrYAMHG/view?usp=sharing"><img src="https://1drv.ms/i/s!AoC6BbMk0S9Qm0RWMGZmtZXGd9z4?embed=1&width=2543&height=1212"></a>
    <figcaption>superset, sql lab</figcaption>
</figure>

위 사진에서는 COUNT_TYPE_BY_ACCOUNT 테이블을 이용해 새로운 데이터로 조회했습니다. 이것을 오른쪽 중간에 SAVE 버튼 옆 확장 표시를 눌러 Datasets으로 저장할 수 있습니다. 또는 CREATE CHART 버튼을 눌러 바로 차트를 만들 수도 있습니다.

몇 가지 쿼리를 만들어 간단하게 대시보드를 구성해봤습니다.
<figure>
    <a href="https://drive.google.com/file/d/1L22GmE9SBNjaJ-q0-aGHrQeNUZm7dEsc/view?usp=sharing"><img src="https://1drv.ms/i/s!AoC6BbMk0S9Qm0U95QrnHpAev6Xi?embed=1&width=2558&height=1209"></a>
    <figcaption>superset, dashboard transaction</figcaption>
</figure>

<figure>
    <a href="https://drive.google.com/file/d/1l-CHHk2PcepbZGiCYemRgzEAkBSn3uVn/view?usp=sharing"><img src="https://1drv.ms/i/s!AoC6BbMk0S9Qm0ZjPZ3xzWCi0V2X?embed=1&width=2552&height=1214"></a>
    <figcaption>superset, dashboard transaction-event</figcaption>
</figure>

각 차트 우측 상단 ...을 누르게 되면 refresh interval을 설정하거나 차트를 공유할 수도 있습니다.

## 마무리
실시간으로 데이터를 생성하고 kafka-ksqlDB를 거쳐 superset으로 시각화까지 이어지도록 해봤습니다. ksqlDB이 window 관련 내용이 재밌는게 많았지만 깊게 파지 못한 점이 아쉬웠습니다. 실제로 해본 느낌으로는 kafka 이후 배치 파이프라인이 필요없어질 수 있겠다는 생각과 kafka에서 superset으로 바로 연결가능한 솔루션이 있었으면 좋겠다는 생각이 들었습니다.