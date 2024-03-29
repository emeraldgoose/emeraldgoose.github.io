---
title: Kubernetes
categories:
  - Ops
tags: [k8s]
---
## 컨테이너 기반 배포
애플리케이션 배포 방식은 물리적인 컴퓨터에 OS와 APP을 설치하여 서비스하던 방식에서 가상화 배포 방식으로 변화했습니다. 가상화 방식은 가상머신 성능을 각각 관리하면서 자원 상황에 따라 조절할 수 있습니다. 그러나 가상머신마다 OS를 새로 설치해야 하고 용량 또한 큽니다.

컨테이너 기반 배포가 등장하면서 Host OS 위에 컨테이너 런타임이 올라가고 그 위로 컨테이너가 올라가게 되었습니다. 컨테이너 기반 배포는 가상화 배포 방식과 비슷하지만 컨테이너는 Host OS의 API를 가져다 쓰는 가상머신입니다. 가상머신이라는 격리된 메모리, 디스크, 네트워크, IO 인터페이스 등을 할당받지 않아서 오버헤드를 줄일 수 있습니다.

### 쿠버네티스
도커는 컨테이너 기반의 오픈소스 가상화 플랫폼을 말합니다. 도커를 사용해서 애플리케이션을 서비스 단위로 분할하여 배포할 수 있지만 서비스마다 배포해야 하는 단점이 있습니다. 다수의 컨테이너를 관리할 필요가 생기면서 **컨테이너 오케스트레이터**라는 기능을 많은 회사들이 서비스로 내놓았습니다.

그 중 하나가 **쿠버네티스**입니다. 쿠버네티스는 규모에 맞는 컨테이너를 배포하는데 필요한 오케스트레이션 및 관리기능을 제공합니다.

### 쿠버네티스 기능
- 시간대별로 서비스의 트래픽이 달라지는 경우 Auto Scaling 기능을 통해 자원의 양을 조절할 수 있습니다.
- Auto Healing 기능을 통해 장애가 난 서버의 자리를 여분의 서버 하나가 자동으로 대체합니다.
- 서비스의 버전 업데이트가 있는 경우 Deployment 오브젝트를 통해 업데이트 방식에 대해 자동적으로 처리되도록 합니다.

## 쿠버네티스 Overview
쿠버네티스 클러스터는 서버 하나를 Master, 다른 서버를 Node라 하고 하나의 Master에 여러 Node들이 연결되게 합니다. 마스터는 쿠버네티스의 전반적인 기능들을 컨트롤하고 노드들이 자원을 제공합니다. 만약, 클러스터의 자원을 늘리고 싶다면 노드들을 추가해주면 됩니다.

클러스터 안에 쿠버네티스 오브젝트들은 독립된 Namespace라는 공간에 들어갑니다. Namespace에는 Pod들이 있고 외부로부터 IP를 할당해주는 서비스가 있습니다. 그리고 다른 Namespace안 Pod끼리는 연결이 불가능합니다. Pod가 문제가 생겨 재할당이 되면 기존 Pod 안 데이터들이 날라갑니다. 그래서 Volume을 만들어 데이터를 하드에 별도로 저장하게 할 수 있습니다.

Namespace에는 ResourceQuota와 LimitRange를 달아서 하나의 Namespace에서 사용할 수 있는 자원의 양(CPU, 메모리)을 한정할 수 있습니다.

Pod를 컨트롤하는 Controller는 다음과 같습니다.
- ReplicaSet
  - Pod이 죽으면 다시 살리거나 스케일링이 가능합니다.
- Deployment
  - 배포 후 Pod들을 업그레이드합니다. 문제 발생 시 롤백이 가능합니다.
- DaemonSet
  - 한 노드에 Pod 하나씩만 유지되도록 합니다.
- Job
  - 어떤 특정 작업만 하고 종료를 시켜야 할 때 사용합니다.
  - 이런 Job을 주기적으로 실행해야 한다면 Cronjob을 사용합니다.

## minikube
쿠버네티스를 정확하게 실습하려면 3대의 서버가 필요하지만 간단하게 `minikube`라는 도구를 이용하여 실습해볼 수 있습니다.
- 맥 기준 홈브류를 이용해서 쉽게 설치할 수 있습니다. 윈도우는 따로 설치파일이 필요합니다.
- `brew install minikube`

### minikube 명령어
- 버전관리 `minikube version`
- 가상머신 시작
  - x86 `minikube start -driver=hyperkit`
  - M1 `minikube start --driver=docker`
  - driver 에러 발생 시 `minikube start --driver=virtualbox`
  - 특정 버전 실행 `minikube start --kubernetes-version=v.1.23.1`
- 상태확인 `minikube status`
- 정지 `minikube stop`
- 삭제 `minikube delete`
- IP확인 `minikube ip`

### kubectl 명령어
`kubectl`은 쿠버네티스 클러스터에 명령어를 전달하는 도구입니다. 쿠버네티스에서는 대시보드를 제공하지만 노출되지 말아야할 값들이 노출되어서 대시보드를 권장하지 않습니다.
- `apply` : 원하는 상태를 적용, -f 옵션으로 파일과 함께 사용한다.
- `get` : 리소스 목록을 보여준다.
- `describe` : 리소스 목록을 자세히 보여준다.
- `delete` : 리소스를 제거한다.
- `logs` : 컨테이너 로그를 보여준다.
- `exec` : 컨테이너에 명령을 전달한다. 컨테이너에 접근할 때 주로 사용한다.
- `config` : kubectl 설정을 관리한다.

## Reference
- [대세는 쿠버네티스(인프런)](https://www.inflearn.com/course/%EC%BF%A0%EB%B2%84%EB%84%A4%ED%8B%B0%EC%8A%A4-%EA%B8%B0%EC%B4%88/dashboard)
- [https://subicura.com/k8s/](https://subicura.com/k8s/)
- [쿠버네티스 알아보기 1편](https://samsungsds.com/kr/story/220222_kubernetes1.html)
- [쿠버네티스(Kubernetes) 개념, 사용방법, 차이점](https://www.redhat.com/ko/topics/containers/what-is-kubernetes)