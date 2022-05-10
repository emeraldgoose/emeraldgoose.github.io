---
title: Modular Exponentiation
categories:
  - Algorithm
tags: [math]
---

### 모듈러 거듭제곱법 이해
---

거듭제곱한 값에 모듈러 연산을 하는 방식으로 RSA 암호에서 자주 사용되는 방법이다.  
정수 $b$를 $e$번 거듭제곱한 값에 $m$으로 나머지 연산을 한 값을 $c$라 할 때 다음의 식으로 표현할 수 있다.  
- $c=b^2 \space mod \space m,$ $(0 ≤ c ≤ m)$

### Memory-efficient Method
---
간단하게 $b$를 $e$번 거듭제곱한 후에 $m$으로 나머지 연산을 수행한다면 $O(e)$의 시간 복잡도를 가질 것이다.  
그러나, 너무 큰 수를 계산하려면(long long 범위를 넘거나 big integer) 너무 많은 메모리를 차지하게 된다.  
그래서 다음의 성질을 이용하여 메모리 효율을 증가시킬 수 있다.  
- $(a\cdot b)\space mod \space m = [(a \space mod \space m) \cdot (b \space mod \space m)] \space mod \space m$

이 방법의 알고리즘은 다음의 과정을 따라간다.  
1. $Set\space c=1, e'=0$
2. $Increase\space e'\space by\space 1$
3. $Set\space c=(b\cdot c)\space mod\space m$
4. $if\space e'<e,\space \text{go to step 2.}$ 
$\text{Else, } c\space \text{contais the correct solution to }  c=b^2(mod\space m).$

예를들어, $4^{13}$mod 497을 계산해보자. ($b$=4, $e$=13, $m$=497)  
- $e'=1, \space c=(1\cdot4)mod\space 497=4\space mod\space497=4$
- $e'=2,\space c=(4\cdot4)mod\space 497=16\space mod\space497=16$
- $e'=3,\space c=(16\cdot4)mod\space 497=64\space mod\space497=64$
- $e'=4,\space c=(64\cdot4)mod\space 497=256\space mod\space497=256$
- ...
- $e'=12,\space c=(121\cdot 4)mod\space 497=484\space mod\space 497=484$
- $e'=13,\space c=(484\cdot 4)mod\space 497=1936\space mod\space 497=445$

위의 시간 복잡도는 똑같이 $O(e)$를 가지지만 메모리에서 크게 이득을 볼 수 있다. 오히려 작은 수를 계산에 사용하므로 단순 계산에 걸리는 시간보다 빠르게 계산이 가능하다.  

```cpp
#define ll long long

// e_prime이 0부터 시작해야 한다.
void modular_pow(ll base, ll e_prime, ll e, ll mod, ll &c) {
    if(e==0) {
        c=1; return;
    }

    if(e_prime<e) {
        c=(base*c)%mod;
        modular_pow(base,e_prime+1,e,mod,c);
    }
}
```

### 분할 정복을 이용한 모듈러 거듭제곱
---
단순히 $b$를 $e$번 곱할 때 시간 복잡도는 $O(e)$를 가지게 된다. 그러나, 분할 정복을 이용하면 $O(loge)$의 시간 복잡도로 계산이 가능하다.  
거듭제곱의 계산에서의 분할 정복은 다음의 성질을 이용한다.  
- $b^e=b^{e/2}\cdot b^{e/2}$
- $b^{e+1}=b^{e/2}\cdot b^{e/2}\cdot b$

$e$가 1이 될때까지 나누어 주고 다시 합쳐줄 때 거듭제곱과 모듈러 연산을 수행한다.  
```cpp
#define ll long long

ll modular_pow(ll base, ll index, ll mod) {
    if(index==1) return base; // 지수가 1이면 base 반환

    if(index%2==0) { // 지수승이 짝수라면
        ll r=modular_pow(base,index/2,mod); // 지수/2로 꾸준히 나누어 준다
        return (r*r)%mod;
    }
	
    // 지수승이 홀수라면 index에서 1을 빼주고 반환되는 값에 base를 한번 곱해준다
    return base*modular_pow(base,index-1,mod)%mod;
}
```

위의 코드를 좀 더 최적화 시키면 다음과 같다.  
```cpp
#define ll long long

ll modular_pow(ll base, ll index, ll mod) {
    ll r=1;
	
    while(index) { // index > 0
        if(index&1) r=(r*base)&mod; // index가 홀수일때

        // base를 제곱하고 index를 2로 나누어 b^e=b^(e/2)*b^(e/2)역할을 한다.
        base=(base*base)%mod; 
        index>>=1; // index/=2
    }

    return r;
}
```