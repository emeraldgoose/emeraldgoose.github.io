---
title: 좌표 압축 (Coordinate Compression)
categories:
  - Algorithm
tags: [compression]
---

좌표의 범위가 너무 큰 경우 인덱싱으로 좌표 사이 갭을 없애는 좌표압축 기법(Coordinate Compression)을 사용하여 문제를 해결할 수 있다.  
세그먼트 트리와 같은 자료구조를 이용해서 쿼리당 $O(lgN)$의 시간 복잡도를 가지지만 좌표가 너무 큰 범위를 가지는 경우 공간 복잡도가 매우 커져 사용하기 힘들다.  
좌표압축은 해당 좌표를 0, 1, 2, ... 의 값으로 대치하는 방법이다.  
아래 코드는 [BOJ 18870번 : 좌표압축](https://www.acmicpc.net/problem/18870)의 코드이다.  

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#define endl '\n'
using namespace std;

int main(void) {
    ios::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);
    freopen("input.txt", "r", stdin);
    int N;
    cin>>N;
    vector<int> v(N), idx(N); // v배열과 idx배열은 입력되는 좌표를 저장한다.
    for(int i=0;i<N;i++) {
        cin>>v[i];
        idx[i]=v[i];
    }
		
    // lower_bound를 사용하기 위해 정렬과 중복되는 원소를 제거
    sort(v.begin(),v.end()); 
    v.erase(unique(v.begin(),v.end()), v.end());
    
    vector<int> ans;
    for(int i=0;i<idx.size();i++) {
        int res=lower_bound(v.begin(),v.end(),idx[i])-v.begin();
        ans.push_back(res);
    }

    for(auto it : ans) cout<<it<<' ';
    return 0;
}
```

`lower_bound`는 binary search를 사용하기 때문에 쿼리당 $O(NlogN)$의 시간 복잡도와 $O(2N)$의 공간 복잡도를 가지게 된다.