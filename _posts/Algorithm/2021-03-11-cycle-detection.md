---
title: Cycle Detection
categories:
  - Algorithm
tag: [graph]
---
사이클 탐지기법은 dfs를 사용하는 방법과 union-find 방식이 있는데 여기서는 dfs를 사용한 방법을 소개하고자 한다.

dfs를 실행하게 되면 dfs spanning tree가 만들어지는데, 사이클이 있다는 것은 tree에서 back edge(역방향 간선)가 존재한다는 것이다.

즉, 이 back edge를 찾게되면 사이클이 있다는 것을 알 수 있다.

### Back edge

---

Back edge란, 다음의 그림에서 빨간색 선이다.

![](https://lh3.google.com/u/0/d/1CVQ7T6HLMMPYy9e6dMvojEF_aiB5QGca)

정점 1, 2, 5가 사이클을 형성하는 것을 볼 수 있다. 사이클을 찾기 위해서는 Back edge를 찾아야 한다.

## Undirected Graph

---

방향성이 없는 그래프는 이전에 방문한 노드를 다시 방문하면 cycle이 존재한다.(단, 직전 방문한 노드는 해당되지 않는다.)

```cpp
vector<vector<int>> edges;
vector<bool> visited;

bool isCycle(int u, int parent) {
    visited[u]=true;
    for(int v : edges[u]) {
        if(!visited[v]) {
            if(isCycle(v,u))
                return true;
        }
        else if(v!=parent) { // 직전 방문한 노드가 아니면서 방문한 적이 있는 노드라면
            return true;
        }
    }
    return false;
}
```

## Directed Graph

---

방향성이 있는 그래프는 조금 다르다. 무향 그래프와 다르게 이전 방문한 노드와 사이클을 이룰 수 있다.

그리고 무향 그래프에서 사용했던 코드를 다시 사용할 수는 없다. 

예를들면, 아래 그래프에서 dfs로 1 → 2 → 3 → 4까지 내려간 후 길이 없어 2번 정점까지 리턴 될 것이다.

![](https://lh3.google.com/u/0/d/1f_T-Jk36ZmbibyOv9ugiW6IKnqpmHXgu)

이제, 2번에서 5번으로 이동 후 4번으로 가려고 했을 때 4번이 이미 방문한 지점이라 위의 코드에서 true를 반환하게 될 것이다. 하지만 실제로는 사이클이 아니다.

그래서 이번에는 stack처럼 사용할 다른 배열의 공간이 필요하다. 이것을 `recStack[]`이라 하자.

![](https://lh3.google.com/u/0/d/1Fx6rRNCjZMXGubDT-LZhS2T27y3yW3wi)

처음 1번에서 시작하여 dfs를 진행한다.

![](https://lh3.google.com/u/0/d/1r_alnZpEfMHMWlMK8yQsISnDJr4Z2EXn)

차례대로 4번 정점까지 들어가면 두 배열의 상태는 다음과 같다.

- visited[] : [1,1,1,1,0]
- recStack[] : [1,1,1,1,0]

![](https://lh3.google.com/u/0/d/1-4pqEpBFCIeuCCp_XVvvyCzQdeOnO-QK)

다시 반환되어 2번 정점까지 되돌아온다. 이때 배열의 상태는 다음과 같다.

- visited[] : [1,1,1,1,0]
- recStack[] : [1,1,0,0,0]

다음 2번에서 5번으로 진행한다.

![](https://lh3.google.com/u/0/d/1lvNfnHRWhqanV1vo4IybgEP_nWnozieb)

이때 배열의 상태는 다음과 같다.

- visited[] : [1,1,1,1,1]
- recStack[] : [1,1,0,0,1]

마지막으로 5번 정점에서 1번 정점으로 진행하면 recStack[1]=1이므로 true가 반환되어 이 그래프가 사이클이 있다는 것을 판단할 수 있다.

```cpp
vector<vector<int>> edges;
vector<int> visited;
vector<int> recStack;

bool isCycle(int u) {
    visited[u]=true;
    recStack[u]=true;
    for(int v : edges[u]) {
        if(!visited[v]) {
            if(isCycle(v))
                return true;
        }
        else if(recStack[v]) { // 스택에 있는 정점이라면 사이클이 존재
            return true;
        }
    }
    recStack[u]=false; // 사이클을 형성하지 못하는 노드는 스택에서 제거
    return false;
}
```