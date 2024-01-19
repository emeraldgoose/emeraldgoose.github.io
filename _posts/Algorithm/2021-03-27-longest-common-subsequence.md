---
title: LCS(Longest Common Subsequence)
categories:
  - Algorithm
tags: [string, dp]
---
### LCS

---

LCS는 Longest Common Subsequence의 약자로 최장 공통 부분 문자열이다. [LCS(Longest Common Substring)]과 혼동해서는 안된다.

### LCS의 이해

---

DP(Dynamic Programming)를 사용하여 특정 범위의 LCS를 구하고 다른 범위의 LCS를 구할 때 이전 범위의 값을 이용하여 효율적으로 해결할 수 있다.

먼저, 서로 다른 문자열 'ACAYKP'와 'CAPCAK'가 있다고 가정하자.

문자열 'ACAYKP'를 기준으로 'CAPCAK'의 LCS를 구하려고 한다.

![](https://lh3.google.com/u/0/d/1xgheEaXJq1YSY1r4G7fhyl_mabu64BwR)

배열의 첫 번째 행과 열은 모두 0으로 채워준다. 다음 행인 'C'에 대해 LCS를 구하면 다음 그림과 같다.

![](https://lh3.google.com/u/0/d/1yktKtvTsuc5_PnQoEnExFekFrRC11Y_9)

문자열 'ACAYKP'는 'C'이후로 공통된 문자열이 하나이므로 1을 채워준다. 다음 A에 대해 채워주면 다음과 같다.

![](https://lh3.google.com/u/0/d/1jmY-h48FbMzv1qB9s6nXm1e_ASMVSR_y)

2번째 행을 채울 때 (1,2)가 1이고 두 문자열의 'A'가 같으므로 (1, 2)의 값 + 1인 값이 (2, 3)에 저장된다.

![](https://lh3.google.com/u/0/d/1qte5z8UL3MRNlHUdRt1pgj68hhHxCTFJ)

'P'의 행도 반복하면 위의 그림과 같아진다.

비교한 문자가 다르다면 이전 문자까지의 LCS들을 비교하여 가장 큰 값을 현재 값에 저장한다.

즉, 두 가지의 경우가 생긴다.

1. 문자열이 서로 같을 때 : $dp[i][j]=dp[i-1][j-1]+1$
2. 문자열이 서로 다를 때 : $dp[i][j]=max(dp[i-1][j],dp[i][j-1])$

위의 예시에 대해 모든 표를 채우면 다음과 같다.

![](https://lh3.google.com/u/0/d/1-jchdMr0YI42az3MWC4Hu_B_AMDfzx0q)

여기서 LCS의 길이가 아닌 문자열을 찾고 싶다면 큰 숫자가 시작되는 위치를 찾으면 된다.

![](https://lh3.google.com/u/0/d/1XzvdLXfkXl5sPTNh5ZCGR1k3YqQm9bUJ)

이때, 각각의 행과 열에는 단 하나의 체크만 있어야 한다. 

- 그림출처: [https://twinw.tistory.com/126](https://twinw.tistory.com/126)

### LCS 구현

---

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