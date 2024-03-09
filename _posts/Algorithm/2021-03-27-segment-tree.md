---
title: 세그먼트 트리(Segment Tree)
categories:
  - Algorithm
tags: [tree]
---
## 세그먼트 트리란?

여러 개의 데이터가 연속적으로 존재할 때 특정한 범위의 데이터 합을 가장 빠르고 간단하게 구할 수 있는 자료구조이다.  
예를들면 길이가 N인 배열 A에서 A[i]부터 A[j]까지의 부분합을 구하고 A[k]=V로 바꾸어라.  

- 부분합을 구하는 시간 복잡도 : $O(N)$
- A[k]=V로 바꾸는 시간 복잡도 : $O(1)$
    → 쿼리가 M개인 경우 총 시간 복잡도는 $O(MN)$을 가진다.

그러나, 세그먼트 트리를 이용하면 두 쿼리 모두 $O(logN)$의 시간 복잡도를 가지게 된다.

## 세그먼트 트리 구조

길이 10의 순열을 세그먼트 트리로 구성하면 다음 그림과 같다.

![](https://lh3.googleusercontent.com/fife/ALs6j_Fnm3ZEEDj0X8A8u7P4zCO-EmRvvfEB-oEIDcI5gh3EmJeqy6tCTzRwiLb4NCGIqhvrQrig2CQkN44m5V4LK4Iag7P8N_RzT0z7xsyEuXGuz0WU4412NbfYHFrHcrthnzqg0Qf8gSOf5o4xTUsP19Gpb6PpyrPolQuYLixIpOUu5T4X2yIOhKMyVy0jpNAmVCqaYE1116vawrPQQvNQ_iQqT-KugBJ5jZ33y-GgxmTxOKGPn9FHP9-9u1HU808IXZnzlPg8p4R5uQI-lBlA-kqhaPztmnBxojDvRRzegBtuPwhAR367G2k5XsS2QFCJgz7aEGdMRkUSU51BqeIzbw2GnZQeVMeG_gX7t9rPyQDUOrnKmX-RYtPFxGvQc_2UxEAewW1MH6wjE6fRxr00OmlGTrAhIqTCSWeUJyYu5bbL12s5KPXnuzhrpxWO6qf-agoKXYQ2c6ycGYisNC4tBkEoi0iaJdRR2GFGYauNmFBzqxQUuh1IB2vNaLNtrPnciU0Vn-efCMymcTAoglnWaZYpL5l8C9QixaY09jy_3Cu2pSuC8nkDIeNh4Pp_-qeOVYag--TyFw6RdFLQ7NEj6aLk_HUD07JVrc_qq1HmpwSKSEgMfWDmN5W6OP-_Pw0psqmtxpDkTTRfr5pF1CsFK4lNLJneHrF72N9d6PAWB7FQtCSskIq6qqM8W2eOoKFgWir596GqIpsUxk48nDiY3nMQqCDG5J2XNonf05ojTbzvh2PH8PZKTMz_k9DBng-c-jf0ZGMBLwwMlqyCOsPdeQ-q93X3Dm6ZPJwUujjDNJYlsL1Ys0LkX1fQuEnZTYqZE3r-C_xlqOkw4G65X-KjJxAp7iXAUTnvB2KzTR0GqNDlFbY6FKFbhUFt-A-f_SBjhuS9xarqNejZmg_jAUFgz49JzJjvIC2ptGqOvHKJG9NHkzlv-02UM_B5Utyzax4UBW6ndOdlZP6zPoVm6pYoG56Nd23KkhayGiOhbDCpob-vXKC1BPgT6hM8mGRz3CnssP0FDY42sH4d0UFOjba3ZjccFWAv-2puGO62flMiwJz2Uf3TOBDOPINnsHPpzUI_3gpFzsX6PghQdQUlOFOBA1J3OgoT2gKOlzIt8GUCpKBsm7X0sGXT8sq_sSKddiApIJa_1LfVNEA0fPL5XZpO4q5dn_HA119dHL1HGQfbl9IQa19vxzcRnA0H8SE4cXBC3PNSUwnQ_mK-oFiTem_a7UseyBUMmyngJK96Rd3WAXeWR5_jQDD2T9JQznk_ZRFnuiDKYdYy2Zo_G0aYjN-Uc14FoTV0OkFchD1knENwR2pdg0E12b4Yq9Sx2sfVX_g0FlLSAW-fNIb7LHVS3F2uuaRmPkmcWeDulkOK6sJ3eWRrnj91EuYAJG-uDJ23uSTy7squyKmXPyouPkrQjnAGfQ3grrqFdMrI8WcN2R248xi26qEsjp2Ul91ixdtQ4Y1jwPCVIpS4-OVylpGnwGqqurhqGgDA0-9DsjOZwlRPSIGt5vlFlOT8s7L_jRJbOL6H_McM8BNUw4LiaXv5S2cd2n2Lrrf75cwkfYAV6GmZpk2YKzD5g7FKTU-czE0IaSqcoUGT1QfG2re2XUX8FTYkWkU9aRMSSMxuRqP3fINjFlslOpw_Jw)

트리를 만드는 방법은 다음과 같다.

1. 루트는 전체 순열의 합이 들어간다.
2. 자식 노드는 부모의 데이터를 절반씩 나누어 구간 합을 저장한다.

이 과정을 반복하면 구간 합 트리의 전체 노드를 구할 수 있다.  
이때, 루트의 번호는 0이 아니라 1을 의미한다.   
왜냐하면 다음 왼쪽 자식이 2, 오른쪽 자식이 3을 가리키게 되면서 부모노드 번호에서 2를 곱하면 왼쪽 자식노드를 의미하기 때문에 효과적이다. 아래 그림을 보면 이해하기 쉽다.  

![](https://lh3.googleusercontent.com/fife/ALs6j_H-ZgdVezHkKTo0kpSIZldSQXkETY3aU7iQIeaGQDBA48D5SWJkNcQRC9iMWPdM0LKGhIl9iu3q2vFPnwI73_ApxyXWP8PPB3kEdDD3-TTVy5rZKWWPDsLHhBkCVNSAPN0mGnrkdf5ADI24QtnGZxMkwPte9wHXQCcM-YniGFGryYWfDhM4SsYQBCr6KeVBEZDyaf70OKNj1Ib_s-BbFHzfKhzgV9Z3dm4wlQ6t5SKz_tPBkVMXq-PGfrqVoWsF2tmRtnfYJwkLBHHF8w6zfyqcv96_dNPagsattXgK7J1EEm6-CpEHdGIgxQBVSTSmTKioR5k7nzmacDgAlGxz0B5mvf63Wv7Qn8A0-uYnfpZ_VucEQIQqv8glErEvXlDQp7dAuSdIchrcuWH71Nv5MPdITr3wC7tDbR_ARNrV9V0gTPwguhrsoWyAOhPgSAEsD80c0i8ywehVMxpVlM7xZkS2dF-R9fmSiwOEs4QKi4aITcjJ2xFN53gkww9FUSocYP2fAdVWFHk6Ycz14UID1vEDDm1Y_Y3IR_SqWLQFLVIsWtREooOwhLt1zIERHZNT_J8WrY9HQdI3YvdInIVvTSJx1AiX2vNJOpmnSOaHEZALg35Z2g5px62honUGzh7lnPVw-Cw1oXvrcZVMp-6LezmsogT8Mdv9xe3QsgqhsqBLFoaRiO22czyQfZsNCNI9EVtN-7TE7wKW0Avl8oQ936SYwT4VPZX53Fu_zV482w1tPsRxsU23-xSCDTAIFbzvVamqsoHBCDgGEyXDVAN-Dv63JJ23yJxPlPZeqmnsfxlY4R8JbIGuLjBjnG3_5faJXEmAZLOhI2xcAPNCIMHBLJMBsV-GcUfxdyPQZhn3qYgmMTmaBcskt5hicebhcfmyGkKit6Lo4uO-1Pl0Xzift7AeM6Ag1CUgEctGFjUC29ug_gurM9QHMusJFS5ccc_s9Ique0n5OHVbt29XYuLpqMGz7lLbQeLVJfIfdmQcnITzUiGSSoeXs-HZlqqffcFtatlztAeKU3movF_DYCCoQZdvAU42eNtmMCP2kLwkPd1T00DcaKEocdG8TJd1szTEJV_XigxrzlhMfcJY8TsmzQeWkQIaCS98iP0Gn9C5qLysi_OTm6-B1JihWDwHohIfPBdSuAeifAGoXygXEtEyolQHg_vu6ABLgvRbELeFQUVHWZmbyuNBkOzLR8THgy9nPnUBP1mTlyQ4D-oEfoE0YDytWSkKZ-pWlw2cSZG44_jjS3ilxJQuPSxkh_f_MMkJlOu0Woogh1CM-7DAK3j_lRmFf15z8AWHie7OojQvXekyUqk0U-6f49rqw1HE-TssMOmKMo44I3DRyQ8IOjTXZcJp28gnaaYcclI2WaKanYbbK_LHMFjOcazIDbmZUkhmjKP6wvOG_EY51dauG7NhZECXlaLl12YgzRwZdY873ZuYbGNdHXfZ4QmmtFf-rUiS_zlRZ51icZlRGcUTv140nu7DI4AgDHky9cSCtsJ_7RpWu30DQLNrlZK3yTz259nUYaf8CkXeEsC0_TIB3E8n0fZpLPO0bXhZzXuTQxFbnNU90UrQsNON5g7pF_1_LnAiNLZubQL7doRKdvmWU2nO7hEjx8FgmpFUdilqliO8sGycWiD06Q)

또한, 구현 방식은 재귀적으로 구하는 것이 더 간단하다.  

```cpp
// start: 시작 인덱스, end: 끝 인덱스
int init(int start, int end, int node) {
    if(start == end) return tree[node] = a[start];

    int mid = (start + end) / 2;

    // 재귀적으로 두 부분으로 나눈 뒤에 그 합을 자기 자신으로 한다.
    return tree[node] = init(start, mid, node * 2) + init(mid + 1, end, node * 2 + 1);
}
```

## 구간 합 구하기

![](https://lh3.googleusercontent.com/fife/ALs6j_HlgjAR9T7_P_2QgWPo8ZmeB9LWnzNPjzLnqRt0o--Bf9hVdKZMDB9MZiFYVEUn_AKGNa4VUPaCbfsp7X3Z8u5EHKcyvdtq-VgajO5bro0rByRmC4eN3POeJzHm9HQd6jZJzpOsy-LfwKwyHvFK_ORkKp2n8hUzorDt0mye11k-4eAYDW-YGJnAM01R7__f6bFZD5JIsK8HrQ6aUgY6tIz5G0w1VvUMejjr2f0Qr9WqQ65ECLkDofpOxyMpSaDj9DUczNVJymZjQy601ghNj7nmKVMHuEqALsuzaLHE05cE74r_dqiBclciiTEhwobOYfoGmv1S5Ldlc1n2SWOxwDQl0nrcNqX4VyV8sgdknArvj9J8RqB_XoQPWN1JXPkj7vACQZiFMUaM_yjUnvuBbeiuIStz5nwoG-v-I5G-yn3UQX0lQSOGFMDMoIlsVeS3FQALe0uH9jpBG2K4j3bXXN5gfyb4l3gV1UmZaLe4fEudiAfYY__wCr43CCUZCo1KNWTp_VFDjBg20Omg-XKh0dPfwx1hCFt_QoUVF4lnh6PRVQIFRxdJ7s1-xRqrq3iSCBvrtDlVwKE52nankI5-Hf72e_3zusI-cW2_E25RTt9JH0eLhAKM2NWDGj9sFFNSe-JXkYcZcqf9is3_mxSAjutxJ4J2bWOGlC3WXse81vfm7sH4ALGV5c7eC6FMfA_4H60ELODX-5FcEqwgvqXRVGMCYJleunRgQXLSJWFW7ix69jWlBMqgqdQcKoD7iSM2AUpWqvViiZpApQkKsVdM1UP-QiKMD89lnXiVxqNKseIkVstPPgx532OD94JoOkjQzkPGvrONxiU5cTP6oFZzJ3wLOLxvLYyMkp7qngT_wEXvtYdvwDJ7ZzUVtdW98aEy471fhkhfBP1hDSdauCDPFVkY_VZLOcoUI2QrKjXS2q8-yeo8oBfZCtvppE-Z1SuS7S1DAJBQiFpyAb1s5sfb-6rtx743mtULd1Zn77nyb4FKi83S0E8mr_Pp4xxDY_SRLPZvwaeFW3ewp1WDGP8YFSLAx67mrzl868VteZESA4ZMfVIVqfCiCwyJjfShpsskmoa4-w20txSQTYxJ1Nm3ymSGnJhP5fAHd9DFq1_1N1lfTPnSr57LGRfyyN4PtaOP5cyNFbCfooNaHsSVXSW2Yp7gDGb47dSVF-oljxUqNQoL0GnZQJN1IHJkIT2qEwNn0TQvqmDA87ZrLBSxwDcvK7gQKrspVVTHoVph4JlpQ-HRRn_-EurSQMFIYbH4f-ikHo8lbzMOsBNzP65rZz-Lyo5pv3ktUacQ6PlcPAXbLu-d8_6P9TYL0TzWUh5CEk0QvsokfVMIXyvWuO0xtEL38nRlpTkc3XSSHvpiMupR0SGq_wRVC9Q-q5Vlr4h8Wrg4oSXiQc6I0PpYEu-tuXqAimj2yPWvxqD0b-V3iB5qFIWYdnBaR8mQu4ruNhpanRGLcmz0O3FyiNgJLuHWc2gBq8O255DZCoAmPNNKUZ7ixgc8dFagA9_iPpwil2tyGAZu27IHqFEK8vPfw9QcOWqpHF1MY3YUX_RiI29ZYR52xkst2d_eoxaPB5IfSKk8S3_iolLqBKlOgm7qFd4JYPerN4JEoD8-ngCU0_qhQTWXhHIEZ4DXUA)

위의 그림에서 a[4] ~ a[8]의 구간합을 구하고 싶다면 녹색으로 칠해진 노드의 값만 더해주면 된다.

```cpp
// start : 시작 인덱스, end : 끝 인덱스
// left, right : 구간 합을 구하고자 하는 범위
int sum(int start, int end, int node, int left, int right) {
    // 범위 밖에 있는 경우
    if(left > end || right < start) return 0;

    // 범위 안에 있는 경우
    if(left <= start && end <= right) return tree[node];

    // 그렇지 않다면 두 부분으로 나누어 합을 구하기
    int mid = (start + end) / 2;

    return sum(start, mid, node * 2, left, rigth) + sum(mid + 1, end, node * 2 + 1);
}
```

## 특정 원소의 값 바꾸기

특정 원소의 값을 바꾸고 싶다면 해당 원소를 포함하고 있는 모든 구간 합 노드를 갱신해야 한다.

```cpp
// start : 시작 인덱스, end : 끝 인덱스
// index : 구간 합을 수정하고자 하는 노드
// dif : 수정할 값과 원래의 값의 차이 (val - a[index])
void update(int start, int end, int node, int index, int dif) {
    // 범위 밖에 있는 경우
    if(index < start || index > end) return;

    // 범위 안에 있으면 내려가며 다른 원소도 갱신
    tree[node] += dif;

    if(start == end) return;

    int mid = (start + end) / 2;

    update(start, mid, node * 2, index, dif);
    update(mid + 1, end, node * 2 + 1, index, dif);
}
```

## 전체 코드

```cpp
#include <iostream>
#include <vector>
#define NUM 13
using namespace std;

int a[]={1,9,3,8,4,5,5,9,10,3,4,5};
int tree[4*NUM];
/*
    4를 곱하면 모든 범위를 커버할 수 있다.
    갯수에 대해서 2의 제곱 형태의 길이를 가지기 되기 때문
*/

int init(int start, int end, int node) {
    if(start == end)
        return tree[node] = a[start];

    int mid = (start + end) / 2;

    return tree[node] = init(start, mid, node * 2) + init(mid + 1, end, node * 2 + 1);
}

int sum(int start, int end, int node, int left, int right) {
    if(left > end || right < start) return 0;

    if(left <= start && end <= right) return tree[node];

    int mid = (start + end) / 2;

    return sum(start, mid, node * 2, left, right) + sum(mid + 1, end, node * 2 + 1, left, right);
}

void update(int start, int end, int node, int index, int dif) {
    if(index < start || index > end) return;

    tree[node] += dif;

    if(start == end) return;

    int mid = (start + end) / 2;

    update(start, mid, node * 2, index, dif);
    update(mid + 1, end, node * 2 + 1, index, dif);
}

int main(void) {
    init(0, NUM-1, 1);

    cout<<"0부터 12까지의 구간 합: "<<sum(0, NUM-1,1,0,12)<<endl;

    cout<<"3부터 8까지의 구간 합: "<<sum(0,NUM-1,1,3,8)<<endl;

    cout<<"인덱스 5의 원소를 0으로 수정"<<endl;
    update(0,NUM-1,1,5,-5); // val - a[index]

    cout<<"3부터 8까지의 구간 합: "<<sum(0,NUM-1,1,3,8)<<endl;

    return 0;
}

/*
    결과:
    0부터 12까지의 구간 합 : 66
    3부터 8까지의 구간 합 : 41
    인덱스 5의 원소를 0으로 수정
    3부터 8까지의 구간 합 : 36
*/
```

## Reference
[안경잡이개발자 : 네이버 블로그](https://blog.naver.com/ndb796/221282210534)
