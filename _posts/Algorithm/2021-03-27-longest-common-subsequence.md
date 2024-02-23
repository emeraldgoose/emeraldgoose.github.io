---
title: LCS(Longest Common Subsequence)
categories:
  - Algorithm
tags: [string, dp]
---
### LCS
LCS는 Longest Common Subsequence의 약자로 최장 공통 부분 문자열이다. [LCS(Longest Common Substring)]과 혼동해서는 안된다.

### LCS의 이해
DP(Dynamic Programming)를 사용하여 특정 범위의 LCS를 구하고 다른 범위의 LCS를 구할 때 이전 범위의 값을 이용하여 효율적으로 해결할 수 있다.

비교한 문자가 다르다면 이전 문자까지의 LCS들을 비교하여 가장 큰 값을 현재 값에 저장한다.

즉, 두 가지의 경우가 생긴다.

1. 문자열이 서로 같을 때 : $dp[i][j]=dp[i-1][j-1]+1$
2. 문자열이 서로 다를 때 : $dp[i][j]=max(dp[i-1][j],dp[i][j-1])$

### LCS 구현
```cpp
#include <iostream>
#define LEN 1001 // 입력으로 들어올 문자열의 최대 길이
using namespace std;

string LCS(string &s1, string &s2) { // LCS 문자열을 반환
  // 동적할당으로 dp 선언
  int **dp;
  dp=new int *[s2.size()+1];
  for(int i=0;i<=s2.size();i++) dp[i]=new int[s1.size()+1];
	
  for(int i=0;i<=s2.size();i++) { // dp 초기화
    for(int j=0;j<=s1.size();j++) dp[i][j]=0;
  }

  int lcs_length=0;
  for(int i=1;i<=s2.size();i++) {
    for(int j=1;j<=s1.size();j++) {
      if(s2[i-1]==s1[j-1]) {
        dp[i][j]=dp[i-1][j-1]+1;
      }
      else {
        dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
      }

      lcs_length=max(lcs_length,dp[i][j]);
    }
  }

  string ret="";
  int len=lcs_length;
  for(int i=s2.size();i>=1;i--) {
    for(int j=s1.size();j>=1;j--) {
      if(dp[i][j]==len && dp[i-1][j]==len-1 && dp[i-1][j-1]==len-1 && dp[i][j-1]==len-1) {
        ret=s2[i-1]+ret;
        len--;
        break;
      }
    }
  }

  // 동적할당 해제
  for(int i=0;i<s2.size();i++) free(dp[i]);
  free(dp);

  return ret;
}
```

### LCS 성능

---

두 수열의 길이를 $n, m$이라 할 때, 동적 계획법에 의해 시간 복잡도는 $O(nm)$을 가지게 된다.

## Reference

[LCS(Longest Common Subsequence) 알고리즘](https://twinw.tistory.com/126)