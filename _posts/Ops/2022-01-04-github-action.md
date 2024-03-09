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

![](https://lh3.google.com/u/0/d/1B0qZbAkEBeZOTmEj67Cg7LvTAzqzFkTC){:width=400}

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
![](https://lh3.google.com/u/0/d/119I6XBZ0AodU-BtzYJS3iM7gj3amr6Td){:width=400}  

## Pytest
Github Action을 사용하는 이유는 테스트, 빌드, 배포를 자동화하는 도구이므로 pytest를 사용하여 원하는 코드를 테스트할 수 있습니다.  
저는 간단하게 `math_.py`와 `test_math_.py`를 만들어 함수들을 테스트했습니다.
테스트코드를 작성하면서 따로 설정할 것은 없었습니다. 테스트코드를 만들어 두면 `Test with pytest` 단계에서 테스트한 결과를 볼 수 있습니다.

![](https://lh3.google.com/u/0/d/1JVmteiLG7BU_MCzbjyV-QpSDtdArDJ2V){:width=400}  

단, pytest를 사용할 때 테스트 파일의 전치사가 `test_`로 시작해야 합니다. 그렇지 못하면 `Test with pytest`의 run에서 `pytest filename.py`로 바꿔줘야 합니다.

## Slack message
이제 테스트가 성공인지 실패인지 알림이 왔으면 좋겠습니다. 개발자들이 많이 사용하는 슬랙으로 알림을 보내려고 합니다.  
슬랙을 사용하기 전에 알림을 받고 싶은 스페이스에서 webhook을 설정해야 합니다. 이건 다른 블로그에서 많이 소개하고 있어서 참고하시면 되겠습니다.  
하지만 공개된 repo에 webhook url을 설정할 수는 없으므로 깃허브에 환경변수로 등록하려고 합니다.
깃허브 repo의 Settings 탭에서 Secretes로 가면 환경변수를 등록할 수 있습니다.

![](https://lh3.google.com/u/0/d/1cErmGBRZUoAydx2ZEaW_o_5ikR_z3KwQ){:width=400}  

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

![](https://lh3.googleusercontent.com/fife/ALs6j_Fk96_ZPKxJPMWsRGA_TjufCtF8K7V0SDZ7g5yZcdxjR_PXzjSPYyKhnMrS3ODMlbNIBA0F3YztVsaLFW88X3MpxZFOUmcZSV_2RLv1iq2Nb1qyNlhQyvx6nqiZBHH2L_m7HLNQEA1uVznphQLiiIwYKBKmAvqanV-NV8tPJXHQ_hzQVIAFn_TKHen4j0swWeukpJDLJTm7wdtCgw5Jeksk8Zt-JWmEiwLG4lcMwaG9F3YqMNhOroGVp9aoQxHqBwaesWVcQAhWUUd6oQJyHhOVB2imHsYsiI3X2aUMUQNKfaWaWJJikE_NonH9WW3YFExYF6WnlFTsR-WQALk4_fSdIS4xi2jNKb_xpZNn09Xvn3bd6Oa9gjS1s_lcC4z3gFQeYczLpOXLKiSvCAeT_qUJNgLCUvr1tf7bDNWwDhvvXYxO647U6TXvwo7O_xVttglkGAmNISo8t73mXb7PaX8PcOwc7QuXQzfjoJSceXNahYB9q2NR_eGl_dB-N9V1Hm51Q798wJikX3QIbhm9G4pVa4X19Ktvlvosv1YSbZ0rlK80PBwAjdOQu-Kk-IVGLdv5_5koYyUfnZJ1iehRsJn_Q2aD71aG0uL1THuvb3Xv9wjvg07nJsvmH3KO4GA2HA-goupvMaH6AQoxHSAFAmBXUBnndq87CTQg0fY1nt8AbNzyv5-jGOcCK9H2ijRaSfoz27QWQrduMR8xXA0qHpMjczVZXLoWjnSt_OCm-5Nf2pUdIV6vnIuPU3DDP-meHyzu_fxMWtPu9CKwQ3fiDUYwQmtiq0gzHMkAq06Mma0gQoIoxAfPKj1988zDdtb_Ze--CZHCkOTZHgUEdSphJnd5Czudu4LzROMsBne3rg7MR-5BxTPrZPQY-Dtn2ftsZCwGsdCt1dvxI0bGpZwCkMHWtAm5Ppl3h7yn8iOU_qZPWpRgrw3_5_0wgD-Y65MRrHwm6Re3YZCORks8iHUXtZBEImCwUFSknpMaXtyWYoXydOi1dFwVNDsKVrnU9vYjVdcmT0JIbf8RVJuf7iB-Pt49bBZQRMY0U96b6sbK7mG2rDoL7IzFv-IfoUBwGTVS8_on1m0iHw4jPZKPrevUr88cq0qSw_4_EyOSCiHpkUW9aZX5GJ8BMaaBjODB597bxY3-hLzQN3bTO68YpFXe50drRCe3Kv5rSA-cMfUU0hUuq1LCbaHrbd12PrG2Kc_24HshNh2P5LwDiIQcZd5vQDjT5g6bE2j_nRACXryGgnlvSP5R_iuc34sjd5JtCI8V2UNmqzvDXPUUySJyShN81XJE5os5ouLosb80mlR7B4j_q5_NxYJtvFWxJlk_MlWeSNfe9fkJfuD30nvp1NCOd1uW3VNw0mNN4p8o-VoX9sXTj6ur4iR0T_HAKnrTqYxFH5cMzwL1MAxVyzRcjs8Rgd0mJAGvSuWJmjiqSVySK3tsh4Zdyv0tHipLtw1fMkMCyq-Qb90MxvJP-M-RrkVTApz5cq6z-fjyN_pLLr4IrappiZGo9pYDfDy8dsjTQAYUVog9ZMcdkzZoX7Kjlmi78-wi8mlS9U-mpyoz4CUIRpqBVDYER4tvjAcAzWUwXR57JyLVTSaWVFAjZAq66AZA5312t2JWZm6lTfCfM80hZLHsrz0b5Q){:width=400}  

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
![](https://lh3.googleusercontent.com/fife/ALs6j_E7604MVdw1EbyuXVwjvbo-aAl2Jlzs7QNeEQTZM08qoyUSHOcQvFw1sMC9nUjG-LU4DnUl9KRZg4gwbXpm6ThGNlP0DmLfGoCoBMDrEo5OYHrksqeZaFF5EzGuRXe-2iIG0_nMhnl9OpHvj8R6SDXBk2gy2M6hmWxs_E6CvD3vcgQq2dLhpTrNV2ihbnEr3n83cgThDbLnogNM0UGNJKBPb8Tk6l5-n3j1F8brgP6BbIUol1YSFrvw5k2yB5Bj_p9uCfTxtGQqRCjl4M1rYrfOlTmIsYDgbRDdzQKtmb1CVKV3vtfHlZbqARGJvyRyfLT_H_ZL_6xtp88yO6eAa9qagp_v487N_tCBSv3DzKK6-OeejpifmGrGzbmg84QhdNaUOtseMRyMkvNblY3MCxEJUVFGKZwD9XWDGZEr95A2hZpmQQ7eJvomHNYsAkNnF9LJQSLuArU6JXuof4aEnM0_ALYEQ9uQaoBYxqBAvt9A9CI--9GpD-DhV-pw7hQ1vdhvzN0D-jk6nvq_rJvYuNPItlzqoRIwv7wzGc-UHy2jkeshC1wNTniBF9qSNBcg3otm6NZHW2Fb9FwVr2PPtHJxV0dc0drlRVcPkOFUWsfL8-9QFWhDccMXPOw1Z6LWeMvljMwXgm2iPlYvXcJQOHY4eV7r5TY08T3YTqERGrzdEO-zgRFJvod_I3yIJtsqWY2MzfD67tQFfpotjXHQvDkGWFbmXcJG7rktymrj0s0Mqty-R6KTJrKuAfAIEW-OyCl8ruulJO92t1udS0Dv2Wsjut-LXZTHnB8tdymkkUQsAwDc_8egRWzByOcAa6_z3vHssVxNjvpYWkm7ucosu_rQtq8kVONYEmRiXDClj37JHEzsH-84BmcLOrrANnNRW-zbl90_iLTQ7k2ecURNQgqrJFTZSD9Ze28732aBuqEvnHBRBy8sOsZviuJQ3iOAqziSK0Os-6JrjknUqwY5CIHjdnVS0AolPq-eFvR7Y2JugzAjVZyS4iep7-dXLyrQzD9NkuJEodBkXOFPkPCneaA6Q5FhaqifwG1y8Y3lpSqG765a5LR8u3wI6300R6TV58w3XxxZnIsUl0-h5GF4MPB3nn9MoDAr81Y1IUENW_3-vo7Ckc-qwhhEpgyqwEuJQIE7DSIxnSFG7BJRjcsgvZW2ErGRl7FiEo-GmQH7pa-Q4WV5sI8YA0pniF1EnpYHW7YxnSJkDMybpuIz5xLgnC0LkNbEdGQkFJBPQFzTfz38u98Uagg-qsZ18_RHvcccU0RQQFlcObzlVjV561IfDzzDsxI8IFZNY_MgzeDbAJdqFinFPfWP-KVPlXbLz6g8ffRh79J_sRMuNfcFf2iGjjHbx9UVImvlViL93st95iYYIejDwHDadXfIh4ukcucIkvKRXm40-AZmZaKJ9b6saR_uqv98-q-nxdIiMU7AZDYaBVaDHaGBIpApD8tyXyN-kaePz7UhF6Ti-yVWGTozeCpWe8Owd54QC5MW-8z-pZu3JsaDqrIEsrW0aDRMPiEmxQVVN6fazKP5s8YSxlFfQprHgLj_HZrZy0QaFeB5w218E-fJ1mmviyRASU80hfcAabVPX4eMirYdC_3y7CMWQNo2XwzIJRCDG2UaEjS8lox01W_ZhA){:width=400}  

## 다른 Apps
위에서 Github Action에서 원하는 애플리케이션을 추가하는 방법을 정리했습니다.  
다른 애플리케이션을 추가하고 싶다면 [https://github.com/marketplace?category=&query=&type=actions](https://github.com/marketplace?category=&query=&type=actions)로 들어가서 똑같은 방식으로 추가하면 됩니다.  
