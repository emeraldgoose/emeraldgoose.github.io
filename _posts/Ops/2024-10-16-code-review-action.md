---
title: Github Action 코드리뷰 봇 만들기(Gemini-1.5-flash)
categories:
  - github
tags: [Ops]
---
## Github Action 설정
Github Action은 사용자가 원하는 트리거에 따라 워크플로우를 실행할 수 있는 CI(Continuous Integration) 도구입니다. 구글의 Gemini-1.5-flash 모델을 사용하여 Pull Request시 코드 변경사항에 대해 LLM이 코드리뷰할 수 있도록 워크플로우를 생성하는 과정을 설명할 것입니다. 이 워크플로우를 실행하기 위해 필요한 것은 Pull Request 권한이 주어진 GITHUB TOKEN과 Gemini API KEY가 필요합니다.

### workflow.yml
repository에 `.github/workflows`에 워크플로우가 정의된 yml파일이 있으면 github action을 사용할 수 있습니다.
```yaml
name: code-review

on:
  pull_request:
    types: [opened, reopened, synchronize]
  workflow_dispatch:

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    steps:
      - name: Checkout Repo
        uses: actions/checkout@v3
      - name: Set up Node
        uses: actions/setup-node@v3
      - name: Install GoogleGenerativeAI
        run: |
          npm install @google/generative-ai
      - name: Get git diff
        run: |
          git fetch origin "${% raw %}{{ github.event.pull_request.base.ref }}{% endraw %}"
          git fetch origin "${% raw %}{{ github.event.pull_request.head.ref }}{% endraw %}"
          git diff --unified=0 "origin/${% raw %}{{ github.event.pull_request.base.ref }}{% endraw %}" > "diff.txt"
      - name: Run Gemini-1.5-flash
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require("fs");
            const diff_output = fs.readFileSync("diff.txt",'utf8');
            
            const { GoogleGenerativeAI } = require("@google/generative-ai");
            const genAI = new GoogleGenerativeAI("${% raw %}{{ secrets.GEMINI_API_KEY }}{% endraw %}");
            const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash"});
            
            const prompt = `You are a senior software engineer and need to perform a code review based on the results of a given git diff. Review the changed code from different perspectives and let us know if there are any changes that need to be made. If you see any code that needs to be fixed in the result of the git diff, you need to calculate the exact line number by referring to the “@@ -0,0 +0,0 @@” part. The output format is \[{“path”:“{ filepath }”, “line”: { line }, “text”: { review comment }, “side”: “RIGHT"}\] format must be respected.\n<git diff>${diff_output}</git diff>`;
            const result = await model.generateContent(prompt);
            const response = await result.response;
            const text = response.text();
            
            fs.writeFileSync('res.txt',text);
            console.log('Save Results!')
      - name: output
        id: store
        run: |
          COMMENT=$(sed '/^```/d' res.txt | jq -c .)
          echo "comment=$COMMENT" >> $GITHUB_OUTPUT
      - name: Add Pull Request Review Comment
        uses: nbaztec/add-pr-review-comment@v1.0.7
        with:
          comments: ${% raw %}{{ steps.store.outputs.comment }}{% endraw %}
          repo-token: ${% raw %}{{ secrets.GITHUB_TOKEN }}{% endraw %}
          repo-token-user-login: 'github-actions[bot]' # The user.login for temporary GitHub tokens
          allow-repeats: false # This is the default
```

### Workflow
이 워크플로우의 구성요소에 대해 하나씩 설명하겠습니다.

```yaml
on:
  pull_request:
    types: [opened, reopened, synchronize]
  workflow_dispatch:
```
on은 워크플로우가 시작하기 위한 트리거를 의미합니다. open은 PR이 열리는 경우, reopened는 PR이 다시 open되는 경우, synchronize는 PR에 커밋이 되어 업데이트 되는 경우를 말합니다. 이러한 이벤트가 실행될 때마다 github action은 실행되며, 변경 코드에 대해 리뷰를 남기게 됩니다.

```yaml
- name: Checkout Repo
  uses: actions/checkout@v3
```
레포를 체크아웃하여 워크플로우가 레포에 접근할 수 있도록 하는 작업입니다. `git diff`명령어를 실행하기 위해 action이 동작하는 환경에서 코드에 접근하도록 하기 위함입니다.

```yaml
- name: Set up Node
  uses: actions/setup-node@v3
- name: Install GoogleGenerativeAI
  run: |
    npm install @google/generative-ai
```
이 단계는 Node 환경을 구축하고 필요한 패키지를 설치하는 단계입니다. 저는 Gemini를 이용하므로 GoogleGenerativeAI 패키지를 설치합니다.

```yaml
- name: Get git diff
  run: |
    git fetch origin "${% raw %}{{ github.event.pull_request.base.ref }}{% endraw %}"
    git fetch origin "${% raw %}{{ github.event.pull_request.head.ref }}{% endraw %}"
    git diff --unified=0 "origin/${% raw %}{{ github.event.pull_request.base.ref }}{% endraw %}" > "diff.txt"
```
이 단계에서는 PR이 되는 base 브랜치와 head 브랜치의 코드 변경 내역을 조회하고 "diff.txt"에 저장합니다. `git diff` 명령어의 `--unified`는 수정되기 전 몇개의 라인을 남겨놓을지 설정하는 옵션입니다. 기본값은 3으로 설정되어 있고 저는 0으로 설정하여 모든 줄이 드라나도록 했습니다. 이 이유는 원본 파일의 라인 넘버가 필요한데 생략하게 되면 정확한 위치를 LLM이 파악하기 어려워했기 때문입니다.

```yaml
- name: Run Gemini-1.5-flash
  uses: actions/github-script@v7
  with:
    script: |
    const fs = require("fs");
    const diff_output = fs.readFileSync("diff.txt",'utf8');
    
    const { GoogleGenerativeAI } = require("@google/generative-ai");
    const genAI = new GoogleGenerativeAI("${% raw %}{{ secrets.GEMINI_API_KEY }}{% endraw %}");
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash"});
    
    const prompt = `You are a senior software engineer and need to perform a code review based on the results of a given git diff. Review the changed code from different perspectives and let us know if there are any changes that need to be made. If you see any code that needs to be fixed in the result of the git diff, you need to calculate the exact line number by referring to the “@@ -0,0 +0,0 @@” part. The output format is \[{“path”:“{ filepath }”, “line”: { line }, “text”: { review comment }, “side”: “RIGHT"}\] format must be respected.\n<git diff>${diff_output}</git diff>`;
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();
    
    fs.writeFileSync('res.txt',text);
    console.log('Save Results!')
```
이 작업에서 "diff.txt"파일을 diff_output에 저장하고 Gemini에게 지시와 함께 전달합니다. LLM은 [{"path": filepath, "line": line_number, "text": "review content", "side": "RIGHT"}]형태로 정리하여 출력하고 결과값을 "res.txt"에 저장합니다. 하지만 LLM의 출력이 마크다운 코드블럭(```) 형태로 출력되어 다음 단계에서 내용물을 파싱합니다.

```yaml
- name: output
  id: store
  run: |
    COMMENT=$(sed '/^```/d' res.txt | jq -c .)
    echo "comment=$COMMENT" >> $GITHUB_OUTPUT
```
이 작업에서는 LLM 출력을 코드블럭 형식으로부터 파싱하고 GITHUB_OUTPUT에 저장합니다. 파싱 방법은 코드블럭 형식을 제거한 뒤 jq라는 도구를 이용해 json 직렬화 변환을 수행합니다. GITHUB OUTPUT은 (Key=Value)형태로 저장하고 같은 jobs 내에서 `steps.{id}.outputs.{key}`로 접근할 수 있게 해줍니다. 만약 다른 job에서 접근하려면 GLOBAL 세팅을 해야 하는 것으로 알고 있습니다.

```yaml
- name: Add Pull Request Review Comment
  uses: nbaztec/add-pr-review-comment@v1.0.7
  with:
    comments: ${% raw %}{{ steps.store.outputs.comment }}{% endraw %}
    repo-token: ${% raw %}{{ secrets.GITHUB_TOKEN }}{% endraw %}
    repo-token-user-login: 'github-actions[bot]' # The user.login for temporary GitHub tokens
    allow-repeats: false # This is the default
```
마지막 작업은 파싱된 내용을 바탕으로 PR Review Comment를 생성하는 작업입니다. 다른 사람이 만들어둔 Action을 사용하여 코멘트를 등록합니다.

## Sample
아래는 실제 review라는 워크플로우의 일부를 캡쳐한 사진입니다. Gemini-1.5-flash를 사용하여 출력된 결과물이 ```처럼 코드블럭으로 보여지고 있습니다.
<figure>
    <img src="https://1drv.ms/i/s!AoC6BbMk0S9Qm369AZoebGqnqsyD?embed=1&width=1631&height=1094">
</figure>

아래 사진은 실제로 PR 리뷰를 위한 Comment가 등록된 모습입니다. 이는 LLM을 실행할 때 프롬프트를 어떻게 작성하냐에 따라 다르게 리뷰할 수도 있습니다.
<figure>
    <img src="https://1drv.ms/i/s!AoC6BbMk0S9Qm30-fNFY1PqDL6DL?embed=1&width=928&height=451">
</figure>

한가지 문제점으로는 Line Number 계산이 쉽지 않을 수 있습니다. 원본 파일을 그대로 올리는 것이 아니고 git diff의 결과로 유추해야만 합니다. 따라서, 리뷰 코멘트의 코드 위치가 잘못된 위치에 가있을 수도 있는 문제가 있습니다.

### PR Link
- [https://github.com/emeraldgoose/github-action-test/pull/3](https://github.com/emeraldgoose/github-action-test/pull/3)