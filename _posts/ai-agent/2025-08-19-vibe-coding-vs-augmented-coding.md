---
title: 바이브 코딩과 증강 코딩(Vibe Coding & Augmented Coding)
categories:
  - ai-agent
tags: [ai-agent]
---
## Vibe Coding
바이브 코딩은 자연어 프롬프트를 이용해 AI가 코드를 작성하고 사용자는 **작동 결과**에 관심을 기울이는 형태의 프로그래밍(?) 방식입니다. 만약, 작동 결과가 사용자의 의도와 맞지 않다면, 피드백을 통해 AI가 코드를 수정하는 방식이며 이러한 방식은 프로토타입을 생성하는데 적합합니다.

바이브 코딩의 장점은 다음과 같습니다.
1. 뛰어난 접근성: 숙련자 뿐만 아니라 비숙련자도 AI를 통해 작업물을 생성할 수 있으며, 실제 잘 작동하는 결과물이 나오고 있습니다.
2. 빠른 프로토타이핑

하지만, 단점은 다음과 같습니다.
1. 낮은 코드 품질
2. 보안 취약 및 유지보수 어려움
3. 맥락 이해 부족 가능성: 보안, 성능 제약에 대한 고려가 이루어지지 않거나 컨벤션이 깨지거나 중복된 함수를 새로 생성하는 등의 동작을 수행할 수 있으며, 사용자 요청에만 맞추기 위한 코드를 생성할 수 있습니다.

바이브 코딩은 속도와 창의성이 필요한 작업에 어울리는 소규모 프로젝트에 적합하며 빠른 실험 및 PoC에 적합한 방식입니다.

## Augmented Coding
> 켄트 벡의 Augmented Coding: Beyond the Vibes를 참고하여 작성합니다

Augmented coding은 사용자가 AI 결과물을 보며 피드백하고 코드 구조, 품질에 사용자가 개입해야 합니다. AI의 도움으로 코드를 작성하되, **작동하는 깔끔한 코드**(Tidy Code That Works)를 목표로 합니다.

특히, 코드의 복잡성 및 테스트 커버리지를 중요하게 생각합니다. 켄트 벡은 자신의 B+ Tree 라이브러리를 Rust와 Python으로 작성하면서 TDD와 코드 안정성을 위해 AI를 사용했고 이 경험을 이야기 하고 있습니다.

또한, AI가 Rust 기반 라이브러리와 Python 기반 라이브러리를 비교하는 성능 벤치마크를 작성했으며, Python 기반의 코드를 Rust 기반의 코드로 변환을 요청하여 괜찮은 결과과 나왔다는 경험도 소개하고 있습니다.

켄트 백의 글에선 AI가 잘못된 방향으로 가고 있을 때의 위험 신호는 다음과 같이 나열하고 있습니다.
1. 루프: AI가 비슷한 코드를 생성하거나, 해결되지 않는 문제에 갇혀 반복 행동을 하는 경우
2. 요청하지 않은 기능을 구현
3. AI가 테스트 코드를 삭제 또는 비활성화하는 등의 치팅 행위

Augmented coding은 AI와의 협업하여 코드를 작성하고 사용자가 코드의 품질을 검도하며 완성도를 높이는 방식입니다. 이는 코드 품질과 테스트가 중요하며 신뢰성 높은 결과물을 원할 때 적합한 방식입니다.

실제로 아래와 같이 AI를 활용해서 태스크 케이스를 생성하거나 코드 개선 사례가 있습니다.
- [생성형 AI를 활용하여 자동차 소프트웨어 요구사항을 위한 테스트 케이스 생성하기](https://aws.amazon.com/ko/blogs/tech/using-generative-ai-to-create-test-cases-for-software-requirements/)
- [Amazon Q Developer 를 이용한 엑심베이의 JDK 자동화 업그레이드 사례](https://aws.amazon.com/ko/blogs/tech/q-developer-eximbay-journey/)

위 사례가 Augmented coding에 직접적인 사례로 들 수는 없을 것 같지만, 개발자를 AI가 대체하는 것이 아닌 개발자가 AI를 잘 다루어 업무 효율성을 높일 수 있는 형태로 사용하는 형태로 변화될 것 같고 AI를 다루는 역량도 점차 중요하게 생각될 것 같습니다.

## Reference
- [Augmented Coding: Beyond the Vibes](https://tidyfirst.substack.com/p/augmented-coding-beyond-the-vibes)
- [켄트 벡의 Augmented Coding: Beyond the Vibes 요약 및 추가 의견](https://www.linkedin.com/pulse/%EC%BC%84%ED%8A%B8-%EB%B2%A1%EC%9D%98-augmented-coding-beyond-vibes-%EC%9A%94%EC%95%BD-%EB%B0%8F-%EC%B6%94%EA%B0%80-%EC%9D%98%EA%B2%AC-toby-lee-mcy8e/?utm_source=chatgpt%2Ecom&originalSubdomain=kr)
- [증강형 코딩: 바이브를 넘어서](https://news.hada.io/topic?id=21733)