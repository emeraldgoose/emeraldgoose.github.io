---
title: "rakazo 에이전트 설정 및 3일 사용 후기"
categories:
  - ai-agent
tags: [ai-agent]
---

## rakazo
> The open source Grok Bot alternative you actually own

<figure style="text-align:center;">
    <a href="https://github.com/elie222/rakazo/blob/main/docs/readme-hero.png?raw=true" data-lightbox="gallery" style="text-align:center;">
        <img src="https://github.com/elie222/rakazo/blob/main/docs/readme-hero.png?raw=true" alt="01" style="max-width:80%;height:auto;">
        <figcaption>https://github.com/elie222/rakazo</figcaption>
    </a>
</figure>

rakazo는 Grok Bot의 기능들을 구현한 오픈소스 프로젝트로 Grok Bot, OpenAI Dots처럼 봇이 컴퓨터를 가지고 장기적으로 동작할 수 있는 환경을 가지고 있는 것이 특징입니다. rakazo가 오픈소스 프로젝트이기 때문에 Self-hosted라는 점도 앞에 내새우고 있으며 각자 자신의 컴퓨터에 직접 설치하여 사용할 수 있습니다. 이 글은 rakazo를 직접 설치하고  3일간 사용해본 후기를 작성합니다.

### 환경
rakazo가 떠 있어야 하는 호스트는 상시 켜져있어야 하기 때문에 메인으로 사용하는 컴퓨터가 아닌 다른 컴퓨터에 설치했습니다. 또한, 각 기기들은 모두 Tailscale을 통해 VPN 망으로 구성되어 있어 Tailscale을 이용해 설정을 하고 있다는 점을 알려드립니다. Tailscale 설정 방법은 각 기기마다 tailscale을 설치하고 같은 아이디로 로그인하면 자동으로 설정되기 때문에 따로 가이드는 하지 않겠습니다.

네트워크 구조를 그리면 다음과 같습니다.

<script type="module"> import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11.16.0/+esm' </script>
<pre class="mermaid">
flowchart LR
    mobile
    desktop
    laptop
    tailscale

    subgraph Mac
        subgraph docker
            postgres
            api
            web
            supervisor
            computer
            worker

            web --> api
            api <--> postgres
            api <--> worker
            
            worker --> supervisor --> computer

        end        
    end

    mobile & laptop & desktop -->|:443| tailscale -->|:5173| web
</pre>

스마트폰, 데스크톱 등 외부에서 rakazo에 접근할 때 tailscale을 통해 접근하게 됩니다. 안드로이드 앱, 아이폰 앱, 맥용 앱도 제공하고 있지만 설치된 rakazo 서버에 접근할 때는 https를 사용하도록 강제되어 있습니다. 이 때문에 tailscale의 https 서비스를 이용하였습니다.
- 아이폰 앱은 앱스토어에서 받을 수 있고 안드로이드, 맥용 앱은 직접 패키지 파일을 다운로드받아 설치해야 합니다.

### 설치

rakazo의 깃허브에 안내하고 있는 방법대로 production 버전으로 설치합니다.

```
mkdir -p rakazo
cd rakazo 
curl -fsSLO https://raw.githubusercontent.com/elie222/rakazo/main/infra/compose/install-images.sh
nbash install-images.sh
```

docker-compose 파일이 다운로드되면 자동으로 띄어지는데 이걸 끊고 다음처럼 수정이 필요합니다.

rakazo 디렉토리 내 `docker-compose.images.yml`파일을 다음과 같이 수정합니다. 아래는 수정이 필요한 내용만 적혀있고 api, worker 등 extra_hosts, ports들은 모두 주석처리합니다. 마지막 `ts-sidecar` 컨테이너는 도커 컴포즈에 같이 띄어져 외부의 통신을 app 네트워크에 연결된 :5173으로 라우팅하는 역할을 수행합니다.

```yaml
name: rakazo
x-app-environment: &app-environment
  # Runtime
  NODE_ENV: production
  API_URL: https://rakazo.tail*******.ts.net
  
services:
  postgres:
  computer:
  data-init:
  supervisor:
  migrate:
  api:
    # extra_hosts:
    #  - "host.docker.internal:host-gateway"
    environment:
      BETTER_AUTH_URL: https://rakazo.tail*******.ts.net
      WEB_ORIGIN: https://rakazo.tail*******.ts.net
    # ports:
    #  - "127.0.0.1:${RAKAZO_API_PORT:-3100}:3100"
  worker:
    # extra_hosts:
    #  - "host.docker.internal:host-gateway"
  web:
    environment:
      RAKAZO_HOST: rakazo.tail*******.ts.net
    # ports:
    #  - "127.0.0.1:${RAKAZO_WEB_PORT:-5173}:5173"
  ts-sidecar:
    image: tailscale/tailscale:latest
    container_name: ts-sidecar
    hostname: rakazo
    extra_hosts:
      - "host.docker.internal:host-gateway"
    devices:
      - /dev/net/tun:/dev/net/tun
    volumes:
      - ts-state:/var/lib/tailscale
      - ./serve.json:/config/serve.json:ro
    environment:
      - TS_AUTHKEY=tskey-
      - TS_HOSTNAME=rakazo
      - TS_STATE_DIR=/var/lib/tailscale
      - TS_SERVE_CONFIG=/config/serve.json
    networks:
      - app
    cap_add:
      - NET_ADMIN
    restart: unless-stopped

volumes:
  pgdata:
  appdata:
  webassets:
  ts-state:

networks:
  app:
  data:
    internal: true
```

같은 위치에 `serve.json` 파일을 생성한 뒤 아래 json을 복사합니다. 이 `serve.json`이 tailscale 주소로 통신을 보내면 5173 포트로 보내주고 따로 인증서 발급을 하지 않아도 되기 때문에 편합니다.

```json
{
  "TCP": { "443": { "HTTPS": true } },
  "Web": {
    "${TS_CERT_DOMAIN}:443": {
      "Handlers": { "/": { "Proxy": "http://web:5173" } }
    }
  }
}
```

저장 후 docker compose를 띄어서 정상적으로 db 초기화 및 설정이 완료되는지 확인합니다.

```
docker compose -f docker-compose.images.yml up -d
```

```
➜  ~ docker ps
CONTAINER ID   IMAGE                             COMMAND                   CREATED      STATUS                PORTS                          NAMES
0134e39ca09d   ghcr.io/elie222/rakazo/app:edge   "docker-entrypoint.s…"   2 days ago   Up 2 days             3100/tcp, 5173/tcp, 7091/tcp   rakazo-web-1
b15972d0ed45   ghcr.io/elie222/rakazo/app:edge   "docker-entrypoint.s…"   2 days ago   Up 2 days (healthy)   3100/tcp, 5173/tcp, 7091/tcp   rakazo-worker-1
e91f6f3f7d87   ghcr.io/elie222/rakazo/app:edge   "docker-entrypoint.s…"   2 days ago   Up 2 days (healthy)   3100/tcp, 5173/tcp, 7091/tcp   rakazo-api-1
3aa9aa88a553   ghcr.io/elie222/rakazo/app:edge   "docker-entrypoint.s…"   2 days ago   Up 2 days (healthy)   3100/tcp, 5173/tcp, 7091/tcp   rakazo-supervisor-1
08094b91be8e   postgres:16                       "docker-entrypoint.s…"   2 days ago   Up 2 days (healthy)   5432/tcp                       rakazo-postgres-1
70f31f5be093   tailscale/tailscale:latest        "/usr/local/bin/cont…"   2 days ago   Up 2 days                                            ts-sidecar
➜  ~
```

도커 컨테이너 내부가 아닌 tailscale 망에서 에서 tailscale을 통해 제대로 접속할 수 있는지 아래 명령어를 통해 테스트해봅니다.

```
nslookup rakazo.tail*******.ts.net
```

위 명령어로 했을 때 tailscale에 보이는 IP와 일치해야 합니다.

```
nc -zv rakazo.tail*******.ts.net 443
```

"Connection to rakazo.tail*******.ts.net port 443 [tcp/https] succeeded!"라는 문구가 나오면 tailscale 망으로 연결된 기기들이 정상적으로 https 통신을 할 수 있습니다.

```
curl https://rakazo.tail*******.ts.net
```

이 명령어를 통해 html이 리턴되면 정상적으로 외부에서 직접 설치한 rakazo로 연결할 수 있습니다.

### 사용
초기에 로그인 창이 나오는데 가입한 뒤 로그인하고 설정에 들어가 API 설정을 진행합니다. 설정은 좌측 하단 프로필을 누르면 설정 화면으로 이동 가능합니다.

<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQRgswK5qGV9QoNHAkNq-sdJAURor4t2sYtjTjJhbCxIhVU" data-lightbox="gallery" style="text-align:center;">
        <img src="https://1drv.ms/i/c/502fd124b305ba80/IQRgswK5qGV9QoNHAkNq-sdJAURor4t2sYtjTjJhbCxIhVU" alt="01" style="max-width:80%;height:auto;">
        <figcaption>rakazo: 설정 화면</figcaption>
    </a>
</figure>

서비스 항목에서 연결 가능한 서비스들을 추가합니다. 항목에 보이지 않는 경우, OpenAI-compatible로 선택해서 base_url, api_key, model을 넣어서 설정합니다. 모델을 세팅한 뒤 메인으로 나와 봇과 채팅을 시도합니다.

#### computer
rakazo의 bot들은 개인의 computer를 가지거나 팀의 computer를 사용할 수 있습니다. supervisor가 직접 컨테이너를 띄울 수 있어서 각 컨테이너가 bot의 computer로 할당됩니다. 우측 상단에 모니터 버튼을 누르면 bot이 사용할 수 있는 컴퓨터 화면이 보이고 직접 조작권을 빼앗아 사용한 뒤 되돌려 줄 수도 있습니다. bot이 못하는 로그인 인증, 결제 등을 사용자가 대신할 수 있다는 점이 특징입니다.

<figure style="text-align:center;">
    <a href="https://1drv.ms/i/c/502fd124b305ba80/IQR5E_nvIwa3S58t8oz5evKNAXQuK2SMLofhKaSpBEF7NCc" data-lightbox="gallery" style="text-align:center;">
        <img src="https://1drv.ms/i/c/502fd124b305ba80/IQR5E_nvIwa3S58t8oz5evKNAXQuK2SMLofhKaSpBEF7NCc" alt="01" style="max-width:80%;height:auto;">
        <figcaption>rakazo: 챗봇 화면</figcaption>
    </a>
</figure>

### 후기
일단 rakazo는 베타 버전의 서비스라는 점을 생각하면 꽤 괜찮다고 생각합니다. 이전에는 hermes agent를 설치해서 사용했었고 나름 좋은 점이 있었지만 외부에서 사용하기 위해서는 web을 제외한 discord, slack과 같은 외부 채널을 이용해야 하는 점이 불편했습니다. 그런 점에서 rakazo에서 제공하는 web, app들이 통일성이 있어 나름 메인으로 써봐도 괜찮을 것 같습니다.

저는 이전 hermes에서 사용하던 cron 작업을 여기서 루틴으로 만들었고 다른 기능들도 사용해보려고 합니다. 그리고 이미지들을 보면 말풍선 바깥으로 글이 나가거나 자잘한 버그가 아직은 있기 때문에 현재 버전을 메인으로 쓰기보다는 grok bot이나 openai dots와 같은 에이전트에 입문하는 용도로도 좋아보입니다.

### References
- [rakazo](https://rakazo.com/)
- [rakazo github](https://github.com/elie222/rakazo)
