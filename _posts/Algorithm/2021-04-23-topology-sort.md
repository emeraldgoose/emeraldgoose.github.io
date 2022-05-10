---
title: 위상정렬 (Topology Sort)
categories:
  - Algorithm
tag: [sort]
---

### 위상 정렬 이해
---

위상정렬은 유향 그래프(DAG)에서 각 정점들의 선행 순서를 위배하지 않으면서 나열하는 것이다.  
예를들면, 스타크래프트에서 스포닝 풀을 짓고 히드라 덴을 지을 수 있듯이 순서를 지키면서 정렬하는 알고리즘이다.  
위상정렬에 사용되는 배열을 degree(진입 차수)라 한다. 이 배열은 어떤 정점이 호출된 수를 저장해 놓은 것인데 진입 차수가 0이라는 것은 현재 정점을 탐색할 수 있음을 나타낸다.  

### 위상 정렬 구현
---

```cpp
#include <iostream>
#include <queue>
using namespace std;

int main() {
    int n; // 정점의 개수
    vector<int> graph[n+1]; // (x -> graph[x][0])
    vector<int> degree(n+1,0);
    for(int i=1;i<=n;i++) { // 호출되는 정점의 차수를 증가
        for(int j=0;j<graph[i].size();j++) {
            int v=graph[i][j];
            degree[v]++;
        }
    }
	
    // topology sort
    queue<int> q;
    for(int i=1;i<=n;i++) { // 진입 차수가 0인 정점을 큐에 삽입
        if(degree[i]==0) q.push(i);
    }
    vector<int> ret; // 정렬순서를 저장할 배열
    for(int i=1;i<=n;i++) {
        if(q.empty()) {
            // n개를 방문하기 전에 큐가 비어버리면 사이클이 존재
            cout<<"cycle detected"<<endl;
            return 0;
        }
        int x=q.front();
        q.pop();
        ret.push_back(x);
        for(int j=0;j<graph[x].size();j++) {
            int v=graph[x][j];
            if(--degree[v]==0) q.push(v);
        }
    }
    // ret 출력
    for(auto it : ret) cout<<it<<' ';
    return 0;
}
```

### 위상 정렬 성능
---
위상정렬은 정점의 개수 $V$ + 간선의 개수 $E$ 번을 탐색하므로 $O(V+E)$의 시간 복잡도를 가진다.  