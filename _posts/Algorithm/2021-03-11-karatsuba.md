---
title: 카라츠바 알고리즘 (Karatsuba algorithm)
categories:
  - Algorithm
tags: [math]
---

## 카라츠바 알고리즘 이해

---

카라츠바 알고리즘 (Karatsuba algorithm)은 큰 수에 대한 효과적인 알고리즘이다. 이 방법은 두 $n$자리 수 곱셈을 $3n^{log_2{3}}=3n^{1.585}$개의 한 자리 곱셈으로 줄일 수 있다.

또한, 카라츠바 알고리즘은 분할 정복 (Divide and Conquer)에 기반하고 있다.

카라츠바 알고리즘은 두 큰 수 $x$, $y$의 절반인 수들의 곱 3번과 시프트 연산을 이용하는 것이다.

왜냐하면, 곱셈의 경우 $O(n^2)$의 시간 복잡도를 가지고 있고 덧셈, 뺄셈의 경우 $O(n)$의 시간 복잡도를 가지고 있다. 카라츠바 알고리즘을 이용하여 곱셈의 횟수를 줄이고 덧셈, 뺄셈의 횟수를 늘리는 알고리즘이다.

$x$와 $y$를 $B$진법의 $n$자리 수라고 하자. $n$보다 작은 양수 $m$에 대해 $x$, $y$를 다음과 같이 쪼갤 수 있다.

- $x=x_1B^m+x_0 \space(x_0<B^m)$
- $y=y_1B^m+y_0 \space(y_0<B^m)$

위의 식을 다음과 같이 다시 정의한다.

- $z_2=x_1y_1$
- $z_1=x_1y_0+x_0y_1$
- $z_0=x_0y_0$

이제, x와 y의 곱은 다음과 같이 정리할 수 있다.

- $xy=(x_1B^m+x_0)(y_1B^m+y_0)=z_2B^{2m}+z_1B^m+z_0$

한편, $z_1$에 대해 $z_2$와 $z_0$를 이용하여 다음과 같이 정리할 수 있다.

- $\begin{matrix} z_1 &=& (x_1y_1+x_1y_0+x_0y_1+x_0y_0)-x_1y_1-x_0y_0=x_1y_0+x_0y_1 \\ &=&(x_1+x_0)(y_1+y_0)-z_2-z_0 \end{matrix}$

## 카라츠바 알고리즘 구현

---

두 큰 수 $x$, $y$는 벡터를 이용할 것이다.

```cpp
void nomalize(vector<int> &num) {
    num.push_back(0);
    int size=num.size();
	
    for(int i=0;i<size-1;i++) { // 자릿수 올림 계산
        if(num[i]<0) {
            int carry=(abs(num[i])+9)/10;
            num[i+1]-=carry;
            num[i]+=carry*10;
        }
        else {
            num[i+1]+=num[i]/10;
            num[i]%=10;
        }
    }
    while(num.size()>1 && num.back()==0) // 앞에 남은 0을 제거한다
        num.pop_back();
}

// multiply
vector<int> multiply(vector<int> &a, vector<int> &b) {
    vector<int> c(a.size()+b.size()+1,0);
	
    for(int i=0;i<a.size();i++) {
        for(int j=0;j<b.size();j++) {
            c[i+j]+=a[i]*b[j];
        }
    }
	
    normalize(c);
    return c;
}

// a+=b*10^k
void add(vector<int> &a, vector<int> &b, int k) {
    int ASize=a.size(); // 원래 a 길이
    if(a.size()<b.size()) {
        a.resize(b.size()+k);
    }
    a.push_back(0);

    for(int i=ASize;i<a.size();i++) a[i]=0;
    for(int i=0;i<b.size();i++) a[i+k]+=b[i];
    normalize(a);	
}

// a-=b (a>=b)
void sub(vector<int> &a, vector<int> &b) {
    for(int i=0;i<b.size();i++) a[i]-=b[i];
    normalize(a);
}

vector<int> karatsuba(vector<int> &a, vector<int> &b) {
    // a.size()<b.size() -> swap(a,b)
    if(a.size()<b.size()) return karatsuba(b,a);
	
    // 기저사례1 : a나 b가 비어있는 경우
    if(a.size()==0 || b.size()==0) return vector<int>();

    // 기저사례2 : a가 비교적 짧은 경우, O(n^2)곱셈으로 변경한다(성능을 위함)
    if(a.size()<=50) return multiply(a,b);
	
    // a와 b를 a0, a1, b0, b1으로 분리
    int half=a.size()/2;
    vector<int> a0(a.begin(),a.begin()+half);
    vector<int> a1(a.begin()+half,a.end());
    vector<int> b0(b.begin(),b.begin()+min(b.size(),half));
    vector<int> b1(b.begin()+min(b.size(),half),b.end());

    // z2=a1*b1
    vector<int> z2=karatsuba(a1,b1);

    // z0=a0*b0
    vector<int> z0=karatsuba(a0,b0);

    // z1=(a0+a1)(b0+b1)-z2-z0
    add(a0,a1,0);
    add(b0,b1,0);
    vector<int> z1=karatsuba(a0,b0);
    sub(z1,z0);
    sub(z1,z2);

    // 병합 과정
    vector<int> ret(z2.size()+half*2,0);
    add(ret,z0,0);
    add(ret,z1,half);
    add(ret,z2,half*2);

    return ret;
}
```

## 카라츠바 알고리즘의 성능

---

$a$, $b$의 자릿수 $n$이 $2^k$ 라고 할 때, 재귀 호출의 깊이는 $k$가 된다. 한 번 쪼갤 때 마다 수행해야 할 곱셈의 연산이 3배씩 증가하므로 마지막 단계에서 $3^k$번의 연산이 필요하다. 그러나 마지막 단계는 한 자리수 곱셈이므로 곱셈 한 번이면 충분하다.

따라서 곱셈 횟수는 $O(3^k)$의 시간 복잡도를 가지고 있는데 $n=2^k$이므로 $k=logn$이다.

그래서 $O(3^k)=O(3^{logn})=O(n^{log3})$의 시간 복잡도를 가지게 된다. $(a^{logb}=b^{loga})$

반면, 병합과정에 있어 덧셈과 뺄셈 과정만 하므로 O(n)의 시간 복잡도를 가지는데 재귀 호출의 깊이가 증가할 때마다 수의 길이는 절반이 되고 연산의 횟수는 3배가 되므로 깊이 k에서 연산 횟수는 $(\frac{3}{2})^kn$가 된다.

총 연산 수는 $n\sum_{i=0}^{log(n)}(\frac{3}{2})^i$인데 $n^{log(3)}$과 같이 따라가므로 최종 시간 복잡도는 $O(n^{log3})$이 된다.

## Reference

[카라츠바의 빠른 곱셈 (Karatsuba algorithm)](https://invincibletyphoon.tistory.com/13)