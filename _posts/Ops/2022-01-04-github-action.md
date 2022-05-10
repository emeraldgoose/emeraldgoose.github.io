---
title: Github Action 정리
categories:
  - github
tags: [Ops]
---
> CI/CD를 위한 도구로 jenkins, github action 등이 있는데 이 중 github action을 설정하고 pytest와 슬랙 메시지까지 전송해보는 것을 정리했습니다.

## Github Action 설정
먼저, public으로 Github action을 활성화할 repo를 생성합니다. (private은 사용 요금이 청구되는 것으로 알고 있습니다.)  
다음, Actions 탭에서 New workflow로 새로운 워크플로우를 생성합니다.

![](https://drive.google.com/uc?export=view&id=1B0qZbAkEBeZOTmEj67Cg7LvTAzqzFkTC){:width=400}

워크플로우를 선택하는 페이지에서 `Python application`으로 생성합니다.

이제 자동으로 `.github/workflows`에 `python-app.yml`이라는 파일이 생성이 되고 기본 코드는 다음과 같습니다.
```python
# This workflow will install Python dependencies, run tests and lint with a single version of Python
# For more information see: https://help.github.com/actions/language-and-framework-guides/using-python-with-github-actions

name: Python application

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:

    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python 3.10
      uses: actions/setup-python@v2
      with:
        python-version: "3.10"
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 pytest
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    - name: Lint with flake8
      run: |
        # stop the build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    - name: Test with pytest
      run: |
        pytest
```

이 코드에서 크게 on은 Event, jobs는 Stpes의 조합으로 이루어져 있습니다.
- on은 특정 branch로 push, PR할 때 실행하는 것을 의미하며 특정 시간대(cron)에 실행할 수도 있습니다.
- 하나의 jobs는 여러개의 steps를 가질 수 있고 여러개의 jobs이 있는 경우 병렬로 실행됩니다.
  - 다른 jobs에 의존 관계를 가질 수도 있습니다. (job A가 success -> run job B)

### jobs
```python
build:

    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python 3.10
      uses: actions/setup-python@v2
      with:
        python-version: "3.10"
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 pytest
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    - name: Lint with flake8
      run: |
        # stop the build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    - name: Test with pytest
      run: |
        pytest
```

runs-on은 action을 실행할 os를 말합니다. 보통 ubuntu에서 돌립니다.  
steps에 `-`단위로 순차적으로 실행됩니다. 
![](https://drive.google.com/uc?export=view&id=119I6XBZ0AodU-BtzYJS3iM7gj3amr6Td){:width=400}  

## Pytest
Github Action을 사용하는 이유는 테스트, 빌드, 배포를 자동화하는 도구이므로 pytest를 사용하여 원하는 코드를 테스트할 수 있습니다.  
저는 간단하게 `math_.py`와 `test_math_.py`를 만들어 함수들을 테스트했습니다.
테스트코드를 작성하면서 따로 설정할 것은 없었습니다. 테스트코드를 만들어 두면 `Test with pytest` 단계에서 테스트한 결과를 볼 수 있습니다.

![](https://drive.google.com/uc?export=view&id=1JVmteiLG7BU_MCzbjyV-QpSDtdArDJ2V){:width=400}  

단, pytest를 사용할 때 테스트 파일의 전치사가 `test_`로 시작해야 합니다. 그렇지 못하면 `Test with pytest`의 run에서 `pytest filename.py`로 바꿔줘야 합니다.

## Slack message
이제 테스트가 성공인지 실패인지 알림이 왔으면 좋겠습니다. 개발자들이 많이 사용하는 슬랙으로 알림을 보내려고 합니다.  
슬랙을 사용하기 전에 알림을 받고 싶은 스페이스에서 webhook을 설정해야 합니다. 이건 다른 블로그에서 많이 소개하고 있어서 참고하시면 되겠습니다.  
하지만 공개된 repo에 webhook url을 설정할 수는 없으므로 깃허브에 환경변수로 등록하려고 합니다.
깃허브 repo의 Settings 탭에서 Secretes로 가면 환경변수를 등록할 수 있습니다.

![](https://drive.google.com/uc?export=view&id=1cErmGBRZUoAydx2ZEaW_o_5ikR_z3KwQ){:width=400}  

`YOUR_SECRET_NAME`에 `SLACK_WEBHOOK_URL`이라 입력하고 `Value`에 url을 입력하시고 등록하면 `python-app.yml`에서 해당 환경변수를 사용할 수 있습니다.  
이름은 원하는데로 설정할 수 있습니다.  

이제 `python-app.yml`에 하나를 추가해야 합니다.
```python
- name: action-slack
      uses: 8398a7/action-slack@v3
      with:
        status: ${{  jobs.status  }}
        fields: workflow,job,commit,repo,ref,author,took
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
      if: always() # Pick up events even if the job fails or is canceled.
```
`steps`에 `Test with pytest`밑에 위 코드를 추가하면 테스트가 통과했을 때 아래 사진과 같이 슬랙으로 알림이 날라오게 됩니다.

![](https://drive.google.com/uc?export=view&id=1d3KLHipOBFTukVpKjUavbw6A--rgwTPz){:width=400}  

다음 코드처럼 다른 방식도 존재합니다.
```python
- name: action-slack
      uses: 8398a7/action-slack@v3
      with:
        status: custom
        fields: workflow,job,commit,repo,ref,author,took
        custom_payload: |
          {
            attachments: [{
              color: '${{ job.status }}' === 'success' ? 'good' : '${{ job.status }}' === 'failure' ? 'danger' : 'warning',
              text: `${process.env.AS_WORKFLOW}\n${process.env.AS_JOB} (${process.env.AS_COMMIT}) of ${process.env.AS_REPO}@${process.env.AS_REF} by ${process.env.AS_AUTHOR} ${{ job.status }} in ${process.env.AS_TOOK}`,
            }]
          }
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
      if: always() # Pick up events even if the job fails or is canceled.
```
위 코드에 대한 슬랙 알림은 다음과 같습니다.
![](https://drive.google.com/uc?export=view&id=1ulXEhcH-9wXD54aOH9J_DmUaPv5CBY6E){:width=400}  

## 다른 Apps
위에서 Github Action에서 원하는 애플리케이션을 추가하는 방법을 정리했습니다.  
다른 애플리케이션을 추가하고 싶다면 [https://github.com/marketplace?category=&query=&type=actions](https://github.com/marketplace?category=&query=&type=actions)로 들어가서 똑같은 방식으로 추가하면 됩니다.  
