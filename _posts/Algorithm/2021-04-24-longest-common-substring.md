---
title: LCS(Longest Common Substring)
categories:
  - Algorithm
tags: [string, dp]
---
## LCS

LCS는 Longest Common Substring의 약자로 최장 공통 부분 문자열이라 한다. LCS(Longest Common Subsequence)와 혼동해서는 안됩니다.

**Substring**이라면 연속된 부분 문자열이고 **Subsequence**라면 연속되지 않아도 되는 부분 문자열을 말합니다.

## Problem

> String A와 String B의 연속된 공통 문자열의 길이를 찾는 알고리즘을 작성하라.

## Dynamic Programming

---

브루트포스나 두 개의 포인터를 이용하는 것보다 DP를 사용하는 방법이 쉽고 간단하기 때문에 DP 방법만 소개하려고 한다.

이제 DP를 다음과 같이 정의할 수 있다.

- `dp[i][j]` : `A`의 $i-1$번째와 `B`의 $j-1$번째까지 LCS를 이루는 문자열의 길이

만약, `A`의 $i-1$번째 문자와 `B`의 $j-1$번째 문자가 같다면 $dp[i][j]=dp[i-1][j-1]+1$이 될 것이다.

다르다면  $dp[i][j]=0$이 저장되어야 한다

```cpp
int lcs(string a, string b) {
  int ret=0; // lcs의 길이
  vector<vector<int>> dp(a.size()+1, vector<int>(b.size()+1,0));
  for(int i=0;i<=a.size();i++) {
    for(int j=0;j<=b.size();j++) {
      if(i==0 || j==0) {
        dp[i][j]=0;
      }
      else if(a[i-1]==b[j-1]) {
        dp[i][j]=dp[i-1][j-1]+1;
        ret=max(ret,dp[i][j]);
      }
    }
  }
  return ret;
}
```

## 성능

---

문자열 `a`의 길이를 $N$, 문자열 `b`의 길이를 $M$이라 했을 때, 반복문으로 인해 $O(NM)$의 시간 복잡도를 가진다.