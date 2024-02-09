---
title: airflow 체험기_최종
categories:
  - airflow
tags: [airflow, Ops]
---
> 이전 글에서 SequentialExecutor에서 CeleryExecutor로 변경하기 위해 삽질한 경험글입니다.

## CeleryExecutor
Celery는 Postgresql과 Mysql만 db로 사용하고 있어서 기존 sqlite를 postgresql로 바꾸는 작업을 진행했습니다. Mysql은 에러가 자주나서 Postgresql을 선택했습니다.  
먼저, `airflow db init`을 하게 되면 `AIRFLOW_HOME`에 `airflow.cfg`라는 설정 파일이 생성됩니다. 바꿔야 하는 설정은 다음과 같습니다.  
```python
# port는 모두 기본포트를 사용하고 있어서 명시해줄 필요가 없었습니다.
executor = CeleryExecutor
sql_alchemy_conn = postgresql+psycopg2://id:password@addr/dbname
broker_url = amqp://id:password@addr/mqname
result_backend = db+postgresql://id:password@addr/dbname
```
`broker_url`에는 보통 메시지 큐를 삽입하는데 RabbitMQ를 사용하기로 했습니다. 보통 Redis는 캐시, RabbitMQ는 메시지 큐로 사용한다고 합니다.  

그렇다면, 도커를 추가로 올려야 할 것은 RabbitMQ, Postgresql입니다.

## RabbitMQ, Postgresql
RabbitMQ와 Postgresql은 Docker Hub에 이미지파일이 있어서 latest버전으로 pull했습니다.  

### RabbitMQ 설정
RabbitMQ는 도커로 실행하면 자동으로 서버가 실행되도록 작성되어 있습니다. (CMD ["rabbitmq-server"])  
그러나, 여기서 airflow라는 유저와 airflow라는 가상호스트를 추가해야 합니다.
```python
# entrypoint.sh
rabbitmqctl add_user airflow airflow
rabbitmqctl add_vhost airflow
rabbitmqctl set_user_tags airflow airflow
rabbitmqctl set_permissions -p airflow airflow ".*" ".*" ".*"
```
dockerfile에서 CMD로 실행시켜버리면 rabbitmq 이미지의 CMD 명령어가 실행되지 않고 위의 명령어들이 실행되면서 **rabbitmq 서버를 찾을 수 없다**는 에러를 볼 수 있습니다.  
어쩔 수 없이 쉘 스크립트로 만들어서 컨테이너에서 쉘을 실행시키는 방법으로 해결했습니다.  
```python
# dockerfile
FROM rabbitmq:latest

ADD ./entrypoint.sh
RUN chmod +x entrypoint.sh
```

### Postgresql
Postgresql은 미리 데이터베이스를 만들어둬야 합니다. 그래서 다음과 같은 sql 쿼리를 작성해서 컨테이너에서 실행할 수 있도록 작성했습니다.  
```python
# entrypoint.sql
CREATE DATABASE airflow;
CREATE USER airflow WITH PASSWORD '1234' SUPERUSER;
```

dockerfile 또한 entrypoint.sql을 실행할 수 있도록 권한을 변경해주고 postgres를 실행할 수 있게 했습니다. 
```python
# dockerfile
FROM postgres:latest

ADD ./entrypoint.sql .
CMD chmod +x entrypoint.sql && \
    su postgres
```
이후에 컨테이너에서 `psql -U postgres -f entrypoint.sql`로 작성한 sql 쿼리를 실행할 수 있습니다.
쉘 스크립트 내에서 sql을 실행시킬 수 있는 방법이 있지만 아직 어려워서 사용하지 못했습니다.  

## airflow
위의 CeleryExecutor에서 바꿔야 할 설정파일들을 자동으로 적용되게 하고 싶었습니다. 스크립트에서 설정파일을 바꿀 수 있는 명렁어가 있는지 찾아보다가 `sed`를 찾게 되어 바로 적용했습니다.  

```python
#!/bin/bash
cp /redis/mydags.py /dags
cp /redis/func.py /dags
cp /redis/redisqueue.py /dags
cp /redis/constant.py /dags

# create airflow.cfg
airflow db init

# exampel=False, set celery worker
sed -i "s/load_examples = True/load_examples = False/g" airflow.cfg
sed -i "s/executor = SequentialExecutor/executor = CeleryExecutor/g" airflow.cfg
sed -i "s^sql_alchemy_conn = sqlite:///./airflow.db^sql_alchemy_conn = postgresql+psycopg2://postgres:1234@172.17.0.2/airflow^g" airflow.cfg
sed -i "s^broker_url = redis://redis:6379/0^broker_url = amqp://airflow:airflow@172.17.0.5/airflow^g" airflow.cfg
sed -i "s^result_backend = db+postgresql://postgres:airflow@postgres/airflow^result_backend = db+postgresql://postgres:1234@172.17.0.2:5432/airflow^g" airflow.cfg

# create account
airflow users create --username admin --password 1234 --firstname a --lastname b --role Admin --email smk6221@naver.com

# run airflow
airflow celery worker -D
airflow celery flower -D
airflow scheduler -D
airflow webserver -p 8080
```
`cp`명령어로 file sharing으로 로컬에서 작성한 파일들을 컨테이너로 옮기는 과정입니다.  
이후에 db 초기화를 진행하고 설정파일을 `sed`명렁어로 바꿔줍니다. 이때 주소는 바뀔 수 있어서 airflow 컨테이너를 가장 마지막에 올렸습니다.  
로그인할 계정을 만들어주고 celery, scheduler, server를 차례대로 실행시켜 줍니다. 웹서버를 제외한 나머지를 background로 돌리고 웹서버만 foreground로 돌리게했습니다.  

### DAG
다음, mydags.py 내용을 수정했습니다. sqlite의 테이블이 있는지 확인하고 없으면 생성하는 태스크를 할 수 있도록 함수 추가와 오퍼레이터를 추가했습니다.  
```python
# func.py
def create_table():
    con = sqlite3.connect('./dags/sqlite.db')
    con.execute('create table if not exists logging (time, level, id)')
    con.close()
```
```python
# mydags.py
t2 = PythonOperator(
    task_id='if_exists_table',
    python_callable=create_table,
    depends_on_past=True,
    owner='gooose',
    retries=3,
    retry_delay=timedelta(minutes=1)
)
```

## 결과
최종 그래프는 다음과 같습니다. 
![](https://lh3.google.com/u/0/d/1Qtqbduiw72XpNNSeQeH7pGWufVc-VT1-){:width=400}  
collector들이 연두색 테두리를 가지고 있는데 RUNNING되고 있는 것을 나타내고 있으며 병렬처리되고 있음을 알 수 있습니다.  
추가적으로 collector들이 db에 접근하기 때문에 table lock이 필요한가?에 대해서 찾아봤는데 sqlite는 트랜잭션을 실행할 때 테이블을 잠가버리기 때문에 따로 구현할 필요는 없다고 합니다.  

![](https://lh3.google.com/u/0/d/1dtOKXsliL3tkOZaqZ9WOmgbz05_8hXIP){:width=400}  
collector 하나의 log를 살펴본 이미지인데 에러가 아닌(check=0) 로그들이 문제없이 db로 들어가고 있습니다. 지금 출력이 2개씩 되고 있는것은 코드 작성에서 print가 두 번 실행되고 있기 때문입니다.  

## 회고
### 도커
5개의 컨테이너를 하나씩 돌려보면서 docker-compose를 사용하는 이유를 알 수 있었습니다. 나중에 한 번 연습해보려고 합니다.  
그리고, 도커를 실행하면 바로 exit(0)되는 경우가 자주 있었는데 이건 도커에 대한 이해를 제대로 하지 못해서였습니다. 도커는 vmware나 virtual box가 아닌 가상 컨테이너에서 명령을 실행하는 도구로 인식해야 함을 알게 되었습니다.
그래서 명령을 다 실행하게되면 자동으로 도커가 stop되는 것이고 서버처럼 계속 돌아가고 있게 하려면 서버를 foreground로 구동해야 합니다. 컨테이너를 돌릴 때 while문으로 메시지(예를들면, "still alive...")를 계속 출력하게 하는 방법도 있다고 합니다.  
마지막으로 RabbitMQ나 Postgres를 사용해보면서 dockerfile 마지막에 CMD로 쉘 스크립트를 실행시키려고 했지만 서버가 실행되지 않으면서 스크립트 명령어가 제대로 작동하지 않았습니다. 이유는 from으로 들어오는 이미지의 도커파일 마지막에 CMD로 서버를 실행시키는 명령이 있고 저의 dockerfile에서 CMD를 사용해버리면서 대체되어 버리기 때문입니다. 이것을 해결하는 방법은 아직 찾지 못했습니다.  

### 쉘 스크립트
부캠때도 많이 사용하지 않은 쉘 스크립트를 이번에 많이 사용하게 되었습니다. 아직 초보 수준이고 if문이나 다른 문법들을 공부할 필요를 느낄 수 있었습니다.  
특히 `sed`의 경우 모두 블로그에서 치환자(?)를 `/`로 많이 사용해서 치환할 문자열이 주소인 경우 너무 난감했습니다. 하나의 블로그에서 다른 치환자가 가능하다는 것을 알려줘서 주소가 들어간 경우 `^`로 사용할 수 있었습니다.  

### 에어플로우
먼저, `bashOperator`의 경우 임시 폴더에서 실행되는 것을 알 수 있었습니다. 맥 기준 /var/private/.../...(확실하진 않지만 비슷한 경로입니다)와 같은 폴더에서 실행되어 환경변수 `AIRFLOW_HOME` 설정이 정말 중요하다는 것을 알 수 있었습니다. 도커파일에는 다른 것들이 들어가지 않아서 `.`으로 설정했지만 디렉토리 구분이 필요한 경우 환경변수 세팅이 중요해보입니다.  
다음, cron 표기인데 (초, 분, 시간, 일, 월, 년) 순으로 되어 있는 것을 몰라서 0/1 * * * *로 했다가 1초마다 갱신되는 지옥을 보았습니다. 나중에 수정해서 5분마다 갱신되도록 했습니다.  

## Reference
- [https://www.slideshare.net/YoungHeonKim1/airflow-workflow](https://www.slideshare.net/YoungHeonKim1/airflow-workflow)
- [http://sanghun.xyz/2017/12/airflow-4.-celeryexecutor-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0/](http://sanghun.xyz/2017/12/airflow-4.-celeryexecutor-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0/)
- [https://stackoverflow.com/questions/36822515/configuring-airflow-to-work-with-celeryexecutor](https://stackoverflow.com/questions/36822515/configuring-airflow-to-work-with-celeryexecutor)
- [http://daplus.net/postgresql-%EC%99%B8%EB%B6%80%EC%97%90%EC%84%9C-%EB%8F%84%EC%BB%A4-%EC%BB%A8%ED%85%8C%EC%9D%B4%EB%84%88%EC%9D%98-postgresql%EC%97%90-%EC%97%B0%EA%B2%B0/](http://daplus.net/postgresql-%EC%99%B8%EB%B6%80%EC%97%90%EC%84%9C-%EB%8F%84%EC%BB%A4-%EC%BB%A8%ED%85%8C%EC%9D%B4%EB%84%88%EC%9D%98-postgresql%EC%97%90-%EC%97%B0%EA%B2%B0/)
- [https://forums.docker.com/t/unable-to-run-psql-inside-a-postgres-container/90623/7](https://forums.docker.com/t/unable-to-run-psql-inside-a-postgres-container/90623/7)
