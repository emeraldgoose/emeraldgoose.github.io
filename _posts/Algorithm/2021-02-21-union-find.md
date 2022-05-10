---
title: Union-Find
categories:
  - Algorithm
tags: [tree]
---

## Union-Find 이해
---
Union-Find(혹은 Disjoint Set)란,  여러개의 노드들을 집합으로 묶어주고 다시 노드들이 어떤 집합에 속하는지 확인할 수 있는 알고리즘이다.

## Union-Find 구현

---

이 알고리즘에서는 두 가지 연산이 필요한데, Find연산과 Union연산이다.

먼저, Find 연산의 경우 노드 `x`가 어떤 집합에 속하는지 확인한다.

```cpp
int parent[MAX_SIZE];

int Find(int x) {
    if(x==parent[x]) {
        return x;
    }
    return Find(parent[x]);
}
```

위와 같이 코드를 작성할 경우 매 Find함수를 호출할 때마다 부모가 나올 때까지 반복하기 때문에 $O(N)$의 시간 복잡도를 가진다. 이 코드로는 트리로 구성하는 의미가 없어진다.

그래서 “**경로 압축**”이라는 최적화 기법이 들어간다.

경로 압축 기법의 중요한 점은 위의 코드에서 `x`가 `parent[x]`가 다른 경우 `parent[x]`를 인자로 Find함수를 다시 호출하는 것을 줄이는 것이다.

그래서, `parent[x]`의 값을 `Find(parent[x])`의 반환값으로 저장하여 처음 쿼리에서 `x`의 부모노드를 바꿔줄 수 있다. 이후에 다시 호출할 때 바로 부모노드를 반환할 수 있다.

```cpp
int Find(int x) {
    if(x==parent[x]) {
        return x;
    }
    return parent[x]=Find(parent[x]);
}
```

다음, Union 연산은 두 인자 x, y가 같은 집합에 속할 수 있다면 두 집합을 합쳐주는 역할을 한다.

```cpp
void Union(int x, int y) {
    int x=Find(x);
    int y=Find(y);
    if(x!=y) {
        parent[y]=x;
    }
}
```

위의 코드는 `x`의 집합과 `y`의 집합을 합쳐야 하는 상황에서 `y`의 부모를 `x`로 바꿔주는 코드이다. 

그러나, 이 코드는 깊이가 깊은 정점이 깊이가 낮은 정점의 밑으로 들어가면 트리의 깊이가 깊어지는 문제가 생길 수 있다. 

그래서 깊이를 저장하는 배열을 하나 선언하여 이 문제를 해결하도록 한다.  이 배열을 rank라 한다.

- 이러한 최적화를 Union-by-rank라 한다.

```cpp
void Union(int x, int y) {
    int x=Find(x);
    int y=Find(y);

    // 두 노드의 부모가 같다면 Union을 할 필요가 없다
    if(x==y) return;

    if(rank[x]>rank[y]) { // x가 y보다 더 깊은 노드라면
        parent[x]=y;
    }
    else { // x가 y보다 깊이가 낮거나 같다면
        parent[y]=x;
    }

    // 두 노드가 깊이가 같다면, y의 부모가 x가 되므로 y의 깊이를 +1한다
    if(rank[x]==rank[y]) {
        level[y]++;
    }
}
```

## Union-Find 성능

---

Union 함수는 Find함수만 시간에 영향을 주므로 Find만 계산하면 된다.

먼저, Find 함수에서 “경로 압축”을 사용하지 않으면 사향 트리(Skewed Tree)가 될 가능성이 있어서 이때의 시간 복잡도는 $O(N)$을 가진다. 

그러나, “경로 압축”을 사용할 경우 Find함수에 의해 호출할 때마다 수행 시간이 변하기 때문에 분석하기가 어렵다. 

그래서 경로 압축을 사용한 Find함수의 시간 복잡도는 $O(\alpha(N))$이라 한다.

- $\alpha(N)$은 아커만 함수라 하며 거의 모든 $N$에 대해서 4이하의 시간을 가진다고 한다. (사실상 상수 시간이다)