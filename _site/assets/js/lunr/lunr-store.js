var store = [{
        "title": "Union-Find",
        "excerpt":"Union-Find 이해 Union-Find(혹은 Disjoint Set)란, 여러개의 노드들을 집합으로 묶어주고 다시 노드들이 어떤 집합에 속하는지 확인할 수 있는 알고리즘이다. Union-Find 구현 이 알고리즘에서는 두 가지 연산이 필요한데, Find연산과 Union연산이다. 먼저, Find 연산의 경우 노드 x가 어떤 집합에 속하는지 확인한다. int parent[MAX_SIZE]; int Find(int x) { if(x==parent[x]) { return x; } return...","categories": ["Algorithm"],
        "tags": ["tree"],
        "url": "/algorithm/union-find/",
        "teaser": null
      },{
        "title": "0-1 BFS",
        "excerpt":"Problem $V$ 정점과 $E$ 간선들을 가지고 있는 그래프 $G$가 있다고 가정하자. 이 그래프의 간선들은 0과 1의 가중치를 가지고 있다. $source$ 로부터 가장 짧은 거리를 계산할 수 있는 효율적인 코드를 작성하라. Naive Solution 위의 문제에서 나이브한 솔루션으로는 다익스트라 알고리즘이다. 다익스트라 알고리즘은 $O(E+VlogV)$의 시간 복잡도를 가지지만 모든 상황에서 빠르지 않다. 0-1 BFS...","categories": ["Algorithm"],
        "tags": ["bfs"],
        "url": "/algorithm/0-1-BFS/",
        "teaser": null
      },{
        "title": "Cycle Detection",
        "excerpt":"사이클 탐지기법은 dfs를 사용하는 방법과 union-find 방식이 있는데 여기서는 dfs를 사용한 방법을 소개하고자 한다. dfs를 실행하게 되면 dfs spanning tree가 만들어지는데, 사이클이 있다는 것은 tree에서 back edge(역방향 간선)가 존재한다는 것이다. 즉, 이 back edge를 찾게되면 사이클이 있다는 것을 알 수 있다. Back edge Back edge란, 다음의 그림에서 빨간색 선이다. 정점...","categories": ["Algorithm"],
        "tags": ["graph"],
        "url": "/algorithm/cycle-detection/",
        "teaser": null
      },{
        "title": "카라츠바 알고리즘 (Karatsuba algorithm)",
        "excerpt":"카라츠바 알고리즘 이해 카라츠바 알고리즘 (Karatsuba algorithm)은 큰 수에 대한 효과적인 알고리즘이다. 이 방법은 두 $n$자리 수 곱셈을 $3n^{log_2{3}}=3n^{1.585}$개의 한 자리 곱셈으로 줄일 수 있다. 또한, 카라츠바 알고리즘은 분할 정복 (Divide and Conquer)에 기반하고 있다. 카라츠바 알고리즘은 두 큰 수 $x$, $y$의 절반인 수들의 곱 3번과 시프트 연산을 이용하는 것이다....","categories": ["Algorithm"],
        "tags": ["math"],
        "url": "/algorithm/karatsuba/",
        "teaser": null
      },{
        "title": "LCA(Lowest Common Ancestor)",
        "excerpt":"LCA 이해 출처 : https://velog.io/@syd1215no/자료구조-Tree-Graph LCA란, 위 그림과 같이 트리를 이루고 있는 어떠한 노드 ‘E’와 ‘M’의 최소 공통 조상(Lowest Common Ancester)를 찾는 알고리즘이다. 간단하게 생각하면 노드 ‘E’에서 루트까지 올라오고 노드 ‘M’에서 루트까지 올라오면서 처음 겹치는 구간이 LCA임을 알 수 있다. 그러나, 노드의 개수를 $N$이라 하면 최악의 경우 $O(N^2)$의 시간 복잡도를...","categories": ["Algorithm"],
        "tags": ["tree"],
        "url": "/algorithm/lca/",
        "teaser": null
      },{
        "title": "Modular Exponentiation",
        "excerpt":"모듈러 거듭제곱법 이해 거듭제곱한 값에 모듈러 연산을 하는 방식으로 RSA 암호에서 자주 사용되는 방법이다. 정수 $b$를 $e$번 거듭제곱한 값에 $m$으로 나머지 연산을 한 값을 $c$라 할 때 다음의 식으로 표현할 수 있다. $c=b^2 \\space mod \\space m,$ $(0 ≤ c ≤ m)$ Memory-efficient Method 간단하게 $b$를 $e$번 거듭제곱한 후에 $m$으로...","categories": ["Algorithm"],
        "tags": ["math"],
        "url": "/algorithm/modular-exponentiation/",
        "teaser": null
      },{
        "title": "Panasonic Programming Contest (AtCoder Beginner Contest 195)",
        "excerpt":"A. Health M Death Takahashi가 M만큼 몬스터를 때릴 수 있고 몬스터의 체력이 H일 때, H가 M의 배수가 아니라면 공격해도 소용이 없다. 즉, H가 M의 배수인지 확인하는 문제이다. #include &lt;iostream&gt; #include &lt;cstring&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; using namespace std; typedef long long ll; typedef pair&lt;int,int&gt; pii; int main(){ ios::sync_with_stdio(0); cin.tie(0); cout.tie(0);...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc195/",
        "teaser": null
      },{
        "title": "AtCoder Beginner Contest 196",
        "excerpt":"A. Difference Max 정수 \\(a,b,c,d\\)에 대해 \\(a ≤ x ≤ b, c ≤ y ≤ d\\)인 \\(x,y\\)가 주어질 때, \\(x-y\\)의 최댓값을 구하는 문제 최댓값은 결국 경계선 값들에 의해 결정되므로 \\(a-c, a-d, b-c, b-d\\) 중 최댓값을 구하면 된다. (사실 b-c가 답이다.) #include &lt;iostream&gt; #include &lt;cstring&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; #define endl...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc196/",
        "teaser": null
      },{
        "title": "0-1 Knapsack problem",
        "excerpt":"Problem 도둑이 보석가게에 부피 W인 배낭을 매고 침입했다. 도둑은 매장에 진열된 보석의 크기 v와 가격 p을 알고 있다. 이때 도둑이 훔친 보석의 가격의 최대 합은 얼마인가? 0-1 Knapsack 문제의 이해 한번 생각해보면 매장에 진열된 보석들의 가격을 기준으로 정렬한 후 가장 큰 가격이 있는 순대로 집어넣으면 될 것처럼 보인다. 그러나 아래와...","categories": ["Algorithm"],
        "tags": ["dp"],
        "url": "/algorithm/knapsack/",
        "teaser": null
      },{
        "title": "좌표 압축 (Coordinate Compression)",
        "excerpt":"좌표의 범위가 너무 큰 경우 인덱싱으로 좌표 사이 갭을 없애는 좌표압축 기법(Coordinate Compression)을 사용하여 문제를 해결할 수 있다. 세그먼트 트리와 같은 자료구조를 이용해서 쿼리당 $O(lgN)$의 시간 복잡도를 가지지만 좌표가 너무 큰 범위를 가지는 경우 공간 복잡도가 매우 커져 사용하기 힘들다. 좌표압축은 해당 좌표를 0, 1, 2, … 의 값으로 대치하는...","categories": ["Algorithm"],
        "tags": ["compression"],
        "url": "/algorithm/Coordinate-Compression/",
        "teaser": null
      },{
        "title": "LCS(Longest Common Subsequence)",
        "excerpt":"LCS LCS는 Longest Common Subsequence의 약자로 최장 공통 부분 문자열이다. [LCS(Longest Common Substring)]과 혼동해서는 안된다. LCS의 이해 DP(Dynamic Programming)를 사용하여 특정 범위의 LCS를 구하고 다른 범위의 LCS를 구할 때 이전 범위의 값을 이용하여 효율적으로 해결할 수 있다. 먼저, 서로 다른 문자열 ‘ACAYKP’와 ‘CAPCAK’가 있다고 가정하자. 문자열 ‘ACAYKP’를 기준으로 ‘CAPCAK’의 LCS를...","categories": ["Algorithm"],
        "tags": ["string","dp"],
        "url": "/algorithm/longest-common-subsequence/",
        "teaser": null
      },{
        "title": "세그먼트 트리(Segment Tree)",
        "excerpt":"세그먼트 트리란? 여러 개의 데이터가 연속적으로 존재할 때 특정한 범위의 데이터 합을 가장 빠르고 간단하게 구할 수 있는 자료구조이다. 예를들면 길이가 N인 배열 A에서 A[i]부터 A[j]까지의 부분합을 구하고 A[k]=V로 바꾸어라. 부분합을 구하는 시간 복잡도 : $O(N)$ A[k]=V로 바꾸는 시간 복잡도 : $O(1)$ → 쿼리가 M개인 경우 총 시간 복잡도는 $O(MN)$을...","categories": ["Algorithm"],
        "tags": ["tree"],
        "url": "/algorithm/segment-tree/",
        "teaser": null
      },{
        "title": "트라이 (Trie)",
        "excerpt":"트라이 이해 트라이(Trie)는 문자열 집합을 표현하는 아래 그림과 같은 자료구조이다. 출처 : https://aerocode.net/377 기본적으로 K진 트리구조를 이루고 있으며 문자열의 prefix를 빠르게 찾을 수 있다. 그러나 많은 공간이 필요하다는 단점이 존재한다. 예를들어 알파벳의 경우 총 26자리의 배열을 각 노드마다 확보해야 한다. 따라서 공간 복잡도는 O(포인터 크기 * 포인터 배열 개수 *...","categories": ["Algorithm"],
        "tags": ["String","Tree"],
        "url": "/algorithm/trie/",
        "teaser": null
      },{
        "title": "AtCoder Beginner Contest 194",
        "excerpt":"A. I Scream milk solid와 milk fat을 비교하여 ice cream, ice milk, lacto ice, flavored ice로 나누는 문제 입력으로 주어지는 milk-solid-not fat과 milk fat을 더하면 milk solid의 값을 알 수 있다. #include &lt;iostream&gt; #include &lt;vector&gt; #include &lt;numeric&gt; using namespace std; typedef long long ll; typedef pair&lt;int,int&gt; pii; int main(){ ios::sync_with_stdio(0);...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc194/",
        "teaser": null
      },{
        "title": "스타트업 코딩 페스티벌 2021(스코페) 후기",
        "excerpt":"1차 대회 1차 대회의 문제는 총 6문제이고 각 스타트업에서 한 문제씩 출제한 것으로 보였습니다. 문제들은 전반적으로 쉬웠습니다. 1번 문제 : 문자열 2번 문제 : DP 3번 문제 : 구현 4번 문제 : 정렬 5번 문제 : BFS 6번 문제 : DP 2차 대회 2차 대회는 1차보다 어려운 문제들이 4문제가 출제되었습니다....","categories": ["Contest"],
        "tags": ["scofe"],
        "url": "/contest/startup-coding-festival/",
        "teaser": null
      },{
        "title": "Caddi Programming Contest 2021 (AtCoder Beginner Contest 193)",
        "excerpt":"A. Discount A엔에서 B엔으로 할인할 때의 할인율을 구하는 문제 #include &lt;iostream&gt; #include &lt;cmath&gt; #define endl '\\n' #define ll long long using namespace std; int main() { ios::sync_with_stdio(0); cin.tie(0); cout.tie(0); double a, b; cin&gt;&gt;a&gt;&gt;b; cout&lt;&lt;fixed; cout.precision(10); cout&lt;&lt;(a-b)*100/a; return 0; } B. Play Snuke 0.5, 1.5, 2.5, … 분 마다 Snuke가 1씩...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc193/",
        "teaser": null
      },{
        "title": "AtCoder Beginner Contest 197（Sponsored by Panasonic)",
        "excerpt":"A. Rotate 입력되는 문자열 S의 길이가 3일 때, S의 첫번째 글자를 S의 맨 뒤로 보낸 문자열 S’을 출력하는 문제이다. #include &lt;iostream&gt; #include &lt;cstring&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; #define endl '\\n' using namespace std; typedef long long ll; typedef pair&lt;int,int&gt; pii; typedef pair&lt;ll,ll&gt; pll; const ll INF=1e10+1; int main(){ ios::sync_with_stdio(0); cin.tie(0);...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc197/",
        "teaser": null
      },{
        "title": "LeetCode Weekly Contest 230",
        "excerpt":"A. Count Items Matching a Rule 입력으로 주어지는 items배열에서 rulekey와 ruleValue에 맞는 항목의 개수를 리턴하는 문제이다. class Solution { public: int countMatches(vector&lt;vector&lt;string&gt;&gt;&amp; items, string ruleKey, string ruleValue) { int find; if(ruleKey==\"type\") find=0; else if(ruleKey==\"color\") find=1; else find=2; int ans=0; for(vector&lt;string&gt; it : items) { if(it[find]==ruleValue) ans++; } return ans; }...","categories": ["LeetCode"],
        "tags": ["Weekly"],
        "url": "/leetcode/lc-wc230/",
        "teaser": null
      },{
        "title": "LeetCode Weekly Contest 231",
        "excerpt":"A.Check if Binary String Has at Most One Segment of Ones ‘1’이 연속으로 붙어있는 부분이 있는지 확인하는 문제이다. class Solution { public: bool checkOnesSegment(string s) { int last=-1; for(int i=s.size()-1;i&gt;0;i--) { if(s[i]=='1') { last=i; break; } } for(int i=1;i&lt;=last;i++) { if(s[i]=='0') return false; } return true; } }; B. Minimum...","categories": ["LeetCode"],
        "tags": ["Weekly"],
        "url": "/leetcode/lc-wc231/",
        "teaser": null
      },{
        "title": "LIS(Longest Increasing Subsequence)",
        "excerpt":"원소 $N$개인 배열의 일부 원소를 골라 증가하는 부분 수열을 이룰 때, 가장 긴 증가하는 부분 수열의 길이를 구하는 코드를 작성하라 LIS 알고리즘의 구현 만약 {1,2,4,5,3,4,5}인 $a$배열이 주어졌을 때 {1,2,3,4,5}의 부분 수열의 길이가 5인 수열이 LIS의 길이가 된다. DP를 이용하면 쉽게 구현할 수 있다. $dp[i]$ : $i$번째 원소까지 부분 수열을 체크했을...","categories": ["Algorithm"],
        "tags": ["dp"],
        "url": "/algorithm/lis/",
        "teaser": null
      },{
        "title": "Codeforces Round 697 (Div.3)",
        "excerpt":"A. Odd Divisor 정수 $n$이 주어질 때 홀수인 수 $x$로 나누어 떨어지는지 확인하는 문제이다. 먼저, $n$이 홀수일 때 홀수인 수 $x$로 반드시 나누어 떨어진다. 다음, $n$이 짝수일 때 $n$을 만드는 방법은 세 가지가 있다. 짝수 * 짝수 = 짝수 짝수 * 홀수 = 짝수 홀수 * 짝수 = 짝수 $n$을...","categories": ["Codeforces"],
        "tags": ["Div3"],
        "url": "/codeforces/cf-round697/",
        "teaser": null
      },{
        "title": "Codeforces Round 702 (Div.3)",
        "excerpt":"A. Dense Array $a_i$와 $a_{i+1}$이 $2a_i ≤ a_{i+1}$ (or $2a_{i+1} ≤ a_i$)의 식을 만족하지 않으면 두 수 사이에 조건을 만족하는 숫자를 집어넣는 문제 $a_i$에서 시작하여 2씩 곱해가면서 $a_{i+1}$보다 같거나 작을 때까지 반복하면 된다. #include &lt;iostream&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; #define endl '\\n' #define ll long long using namespace std; void...","categories": ["Codeforces"],
        "tags": ["Div3"],
        "url": "/codeforces/cf-round702/",
        "teaser": null
      },{
        "title": "Codeforces Round 677 (Div.3)",
        "excerpt":"A. Boring Apartments 입력되는 $x$가 1일때, boring apartments는 1, 11, 111, 1111이므로 1이 등장하는 횟수는 1+2+3+4 = 10이다. 따라서, $x-1$까지 등장하는 숫자의 횟수는 10의 배수임을 알 수 있다. 이때, $len$을 $x$의 길이라고 한다면 우리가 구해야할 정답은 다음의 식을 만족해야 한다. $answer = 10\\cdot(dig-1)+\\frac{len(len+1)}{2}$, ($dig$은 입력된 $x$의 반복되는 숫자) #include &lt;iostream&gt;...","categories": ["Codeforces"],
        "tags": ["Div3"],
        "url": "/codeforces/cf-round677/",
        "teaser": null
      },{
        "title": "Codeforces Round 686 (Div.3)",
        "excerpt":"A. Special Permutation $p_i=i$가 되지 않게 하는 순열을 출력하는 문제 ($1$ ≤ $i$ ≤ $n$) 그냥 2부터 $n$까지 출력한 후 마지막에 1을 출력하면 된다. #include &lt;iostream&gt; using namespace std; int main() { int t; cin&gt;&gt;t; while(t--) { int n; cin&gt;&gt;n; for(int i=2;i&lt;=n;i++) cout&lt;&lt;i&lt;&lt;' '; cout&lt;&lt;1&lt;&lt;endl; } return 0; } B....","categories": ["Codeforces"],
        "tags": ["Div3"],
        "url": "/codeforces/cf-round686/",
        "teaser": null
      },{
        "title": "Codeforces Round 690 (Div.3)",
        "excerpt":"A. Favorite Sequence 일반적으로, 가장 왼쪽 요소는 모든 홀수 위치에 배치하고 가장 오른쪽 요소는 모든 짝수 위치에 배치한 후 왼쪽 포인터는 앞으로, 오른쪽 포인터는 하나씩 뒤로 이동 #include &lt;iostream&gt; using namespace std; void solve() { int n; cin&gt;&gt;n; int arr[300]; for(int i=0;i&lt;n;i++) cin&gt;&gt;arr[i]; int ans[300]; int left=0, right=n-1; for(int i=0;i&lt;n;i++)...","categories": ["Codeforces"],
        "tags": ["Div3"],
        "url": "/codeforces/cf-round690/",
        "teaser": null
      },{
        "title": "Educational Codeforces Round 105 (Div.2)",
        "excerpt":"A. ABC String 문자열 s는 A, B, C로 이루어져 있고, 각 문자열은 '(' 또는 ')'로 치환 될 수 있다. 이때, \"()\", \"(()())\" 처럼 정상적인 괄호 표현식으로 나타낼 수 있는지 확인하는 문제 문자 A, B, C에 대해 '(' 와 ')'를 완전탐색 해야 문제를 해결할 수 있다. 정규 괄호식 테스트하는 함수 test()는...","categories": ["Codeforces"],
        "tags": ["Div2"],
        "url": "/codeforces/cf-round105/",
        "teaser": null
      },{
        "title": "Codeforces Round 674 (Div.3)",
        "excerpt":"A. Floor Number 먼저, $n$≤2이라면, 답은 1이다. 나머지는 $\\frac{n-3}{x}+2$를 만족한다. (왜냐하면 3층 이후부터 $x$만큼 차지하기 때문) #include &lt;iostream&gt; #define endl '\\n' using namespace std; void solve() { int n,x; cin&gt;&gt;n&gt;&gt;x; if(n&lt;=2) { cout&lt;&lt;1&lt;&lt;endl; return; } cout&lt;&lt;(n-3)/x+2&lt;&lt;endl; return; } int main() { ios::sync_with_stdio(0); cin.tie(0); cout.tie(0); //freopen(\"input.txt\", \"r\", stdin); int t; cin&gt;&gt;t;...","categories": ["Codeforces"],
        "tags": ["Div3"],
        "url": "/codeforces/cf-round674/",
        "teaser": null
      },{
        "title": "Educational Codeforces Round 96 (Div.2)",
        "excerpt":"A. Number of Apartments 방이 3개, 5개, 7개짜리로 구성된 아파트에서 각각의 방의 개수를 구하는 문제 가장 간단하게, $n$이 1, 2, 4로 구성된 아파트인 경우 방의 개수가 맞지 않으므로 -1을 출력해야 한다. 방 3개를 기준으로 $[3,6,9,..]$, 5개를 1추가하면 $[5,8,11,…]$, 7개를 1추가하면 $[7,10,13,…]$의 수열을 얻을 수 있다. 즉, 위의 수열 중 1,...","categories": ["Codeforces"],
        "tags": ["Div2"],
        "url": "/codeforces/cf-round96/",
        "teaser": null
      },{
        "title": "AtCoder Beginner Contest 198",
        "excerpt":"A. Div A와 B가 N개의 서로 다른 사탕을 나누려고 할 때, 가능한 조합의 개수를 구하는 문제이다. A와 B는 적어도 1개 이상의 사탕을 가져야 하는데 N의 크기가 작기 때문에 이중for문으로 문제를 해결할 수 있다. #include &lt;iostream&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; #include &lt;cmath&gt; #include &lt;numeric&gt; #define endl '\\n' using namespace std; typedef...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc198/",
        "teaser": null
      },{
        "title": "위상정렬 (Topology Sort)",
        "excerpt":"위상 정렬 이해 위상정렬은 유향 그래프(DAG)에서 각 정점들의 선행 순서를 위배하지 않으면서 나열하는 것이다. 예를들면, 스타크래프트에서 스포닝 풀을 짓고 히드라 덴을 지을 수 있듯이 순서를 지키면서 정렬하는 알고리즘이다. 위상정렬에 사용되는 배열을 degree(진입 차수)라 한다. 이 배열은 어떤 정점이 호출된 수를 저장해 놓은 것인데 진입 차수가 0이라는 것은 현재 정점을 탐색할...","categories": ["Algorithm"],
        "tags": ["sort"],
        "url": "/algorithm/topology-sort/",
        "teaser": null
      },{
        "title": "LCS(Longest Common Substring)",
        "excerpt":"LCS LCS는 Longest Common Substring의 약자로 최장 공통 부분 문자열이라 한다. LCS(Longest Common Subsequence)와 혼동해서는 안됩니다. Substring이라면 연속된 부분 문자열이고 Subsequence라면 연속되지 않아도 되는 부분 문자열을 말합니다. Problem String A와 String B의 연속된 공통 문자열의 길이를 찾는 알고리즘을 작성하라. Dynamic Programming 브루트포스나 두 개의 포인터를 이용하는 것보다 DP를 사용하는 방법이...","categories": ["Algorithm"],
        "tags": ["string","dp"],
        "url": "/algorithm/longest-common-substring/",
        "teaser": null
      },{
        "title": "AtCoder Beginner Contest 199（Sponsored by Panasonic）",
        "excerpt":"A. Square Inequality $A^2+B^2&lt;C^2$이 성립하는지 확인하는 문제이다. #include &lt;iostream&gt; #include &lt;cstring&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; #define endl '\\n' using namespace std; typedef long long ll; typedef pair&lt;int,int&gt; pii; typedef pair&lt;ll,ll&gt; pll; const ll INF=1e10+1; int main(){ ios::sync_with_stdio(0); cin.tie(0); cout.tie(0); //freopen(\"input.txt\",\"r\",stdin); ll a,b,c; cin&gt;&gt;a&gt;&gt;b&gt;&gt;c; if(a*a+b*b&lt;c*c) { cout&lt;&lt;\"Yes\"; } else cout&lt;&lt;\"No\"; return...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc199/",
        "teaser": null
      },{
        "title": "ZONe Energy Programming Contest",
        "excerpt":"A. UFO Invasion 길이 12인 문자열 S가 주어질 때, ZONe문자열이 몇번 들어가 있는지 확인하는 문제 #include &lt;iostream&gt; #include &lt;cstring&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; #define endl '\\n' using namespace std; typedef long long ll; typedef pair&lt;int,int&gt; pii; typedef pair&lt;ll,ll&gt; pll; const ll INF=1e10+1; int main(){ ios::sync_with_stdio(0); cin.tie(0); cout.tie(0); //freopen(\"input.txt\",\"r\",stdin); string s;...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/zone/",
        "teaser": null
      },{
        "title": "KYOCERA Programming Contest 2021（AtCoder Beginner Contest 200）",
        "excerpt":"A. Century 주어진 년도를 보고 몇 세기인지 출력하는 문제이다. 1년부터 100년까지가 1세기, 101년부터 200년까지가 2세기이므로 100으로 나누어 떨어지지 않는 경우부터 다음 세기로 넘어가는 것을 알 수 있다. #include &lt;iostream&gt; #include &lt;cstring&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; #define endl '\\n' using namespace std; typedef long long ll; typedef pair&lt;int,int&gt; pii; typedef pair&lt;ll,ll&gt;...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc200/",
        "teaser": null
      },{
        "title": "Mynavi Programming Contest 2021（AtCoder Beginner Contest 201）",
        "excerpt":"A. Tiny Arithmetic Sequence 어떠한 배열 A에 대해 $A_3-A_2=A_2-A_1$이 되는 경우를 찾아내는 문제이다. 배열 A에 대해 순열을 구하면서 위의 조건이 맞는지 확인하면 된다. #include &lt;iostream&gt; #include &lt;cstring&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; #define endl '\\n' using namespace std; typedef long long ll; typedef pair&lt;int,int&gt; pii; typedef pair&lt;ll,ll&gt; pll; const ll INF=1e10+1;...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc201/",
        "teaser": null
      },{
        "title": "AISing Programming Contest 2021（AtCoder Beginner Contest 202）",
        "excerpt":"A. Three Dice 3개의 주사위를 굴렸을 때 윗면에 있는 숫자를 보고 밑면의 숫자를 더하는 문제이다. (1,6), (2,5), (3,4)가 맞은편에 있는 것을 구현하면 된다. #include &lt;iostream&gt; #include &lt;cstring&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; #define endl '\\n' using namespace std; typedef long long ll; typedef pair&lt;int,int&gt; pii; typedef pair&lt;ll,ll&gt; pll; const ll INF=1e10+1;...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc202/",
        "teaser": null
      },{
        "title": "LeetCode Weekly Contest 239",
        "excerpt":"A. Minimum Distance to the Target Element nums[i]==target이면서 abs(i-start)의 값이 최소인 값을 리턴하는 문제이다. 그대로 구현해주면 된다. class Solution { public: int getMinDistance(vector&lt;int&gt;&amp; nums, int target, int start) { int ret=1e9; for(int i=0;i&lt;nums.size();i++) { if(nums[i]==target &amp;&amp; abs(i-start)&lt;ret) ret=abs(i-start); } return ret; } }; B. Splitting a String Into Descending Consecutive...","categories": ["LeetCode"],
        "tags": ["Weekly"],
        "url": "/leetcode/lc-wc239/",
        "teaser": null
      },{
        "title": "AtCoder Beginner Contest 203（Sponsored by Panasonic）",
        "excerpt":"A. Chinchirorin a,b,c가 주어질 때, 입력된 수 중 2개가 겹친다면 나머지 하나를 출력하고 아무것도 겹치지 않는다면 0을 출력하는 문제이다. #include &lt;iostream&gt; #include &lt;cstring&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; #define endl '\\n' using namespace std; typedef long long ll; typedef pair&lt;int,int&gt; pii; typedef pair&lt;ll,ll&gt; pll; const ll INF=1e10+1; int main(){ ios::sync_with_stdio(0); cin.tie(0);...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc203/",
        "teaser": null
      },{
        "title": "AtCoder Beginner Contest 204",
        "excerpt":"A. Rock-paper-scissors 세 명이서 가위바위보를 진행할 때, 두 사람이 내는 것을 보고 나머지 한 사람이 비기기위헤 무엇을 내야하는지 출력하는 문제이다. 만약 두 사람이 같은 것을 내면 남은 한 사람도 같은 것을 내면 되고, 두 사람이 다른 것을 내면 남은 한 사람은 그 둘과 다른 것을 내면 비길 수 있다. #include...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc204/",
        "teaser": null
      },{
        "title": "LeetCode Weekly Contest 246",
        "excerpt":"A. Largest Odd Number in String 주어진 숫자 문자열에서 연속된 부분문자열(substring)에서 가장 큰 홀수를 찾는 문제이다. 단순하게 num이 홀수인지 확인한 후 뒤에서부터 잘라가면서 홀수인지 판단하여 홀수일 때 리턴하면 된다. class Solution { public: string largestOddNumber(string num) { if((num.back()-'0')%2!=0) return num; for(int i=num.size()-1;i&gt;=0;i--) { if((num.back()-'0')%2==0) num.pop_back(); else break; } return num;...","categories": ["LeetCode"],
        "tags": ["Weekly"],
        "url": "/leetcode/lc-wc246/",
        "teaser": null
      },{
        "title": "AtCoder Beginner Contest 206（Sponsored by Panasonic)",
        "excerpt":"A. Maxi-Buying $\\lfloor 1.08 \\times N \\rfloor$ 값과 206을 비교하는 문제이다. #include &lt;iostream&gt; #include &lt;cstring&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; #define endl '\\n' using namespace std; typedef long long ll; typedef pair&lt;int,int&gt; pii; typedef pair&lt;ll,ll&gt; pll; const ll INF=1e10+1; int main(){ ios::sync_with_stdio(0); cin.tie(0); cout.tie(0); //freopen(\"input.txt\",\"r\",stdin); int n; cin&gt;&gt;n; int res=1.08*n; if(res&lt;206)...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc206/",
        "teaser": null
      },{
        "title": "AtCoder Beginner Contest 205",
        "excerpt":"A. kcal 100 밀리리터마다 A킬로칼로리를 섭취하게 된다. B밀리리터를 먹었을 때 섭취한 칼로리의 양을 구하는 문제이다. #include &lt;iostream&gt; #include &lt;cstring&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; #define endl '\\n' using namespace std; typedef long long ll; typedef pair&lt;int,int&gt; pii; typedef pair&lt;ll,ll&gt; pll; const ll INF=1e10+1; int main(){ ios::sync_with_stdio(0); cin.tie(0); cout.tie(0); //freopen(\"input.txt\",\"r\",stdin); double a,b;...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc205/",
        "teaser": null
      },{
        "title": "Convolution 정리",
        "excerpt":"Convolution 신호처리에서 정말 많이 등장하는 컨볼루션 연산입니다. 기계학습에서도 컨볼루션을 사용하여 학습을 하게 됩니다. 아래 식은 각 뉴련들이 선형모델과 활성함수로 모두 연결된(fully connected) 구조를 가졌습니다. $h_i = \\sigma(\\sum_{j=1}^{p}W_{i,j}x_j)$, $\\sigma$는 활성함수, $W_{i,j}$는 가중치 행렬입니다. 각 성분 $h_i$에 대응하는 가중치 행 $W_i$이 필요합니다. 컨볼루션 연산은 커널(kernel)을 입력벡터 상에서 움직여가면서 선형모델과 합성함수가 적용되는 구조입니다....","categories": ["BoostCamp"],
        "tags": ["convolution"],
        "url": "/boostcamp/convolution/",
        "teaser": null
      },{
        "title": "경사하강법 정리",
        "excerpt":"미분 함수 f의 주어진 점(x,f(x))에서의 미분을 통해 접선의 기울기를 구할 수 있다. 접선의 기울기는 어느 방향으로 점을 움직여야 함수값이 증가하는지/감소하는지 알 수 있습니다. 미분값을 더하면 경사상승법(gradient ascent)라 하며 극댓값의 위치를 구할 때 사용합니다. 미분값을 빼면 경사하강법(gradient descent)라 하며 극소값의 위치를 구할 때 사용합니다. 경사상승법과 경사하강법 모두 극값에 도착하면 움직임을 멈추게...","categories": ["BoostCamp"],
        "tags": ["gradient_descent"],
        "url": "/boostcamp/gradient-descent/",
        "teaser": null
      },{
        "title": "Probability 정리",
        "excerpt":"딥러닝에서 확률론이 왜 필요한가요? 머신러닝은 어떤 방식에서든지 결국 예측을 수행해야 합니다. 예를들어, 강화학습 방식에서도 에이전트가 할 수 있는 행동들 중에서 보상을 가장 많이 받을 수 있는 확률을 고려해야 합니다. 머신러닝에서 사용하는 손실함수(loss function)들의 작동 원리는 데이터 공간을 통계적으로 해석해서 유도하게 됩니다. 이 밖에도 많은 부분에서 확률과 통계 내용이 필요합니다. 기계학습을...","categories": ["BoostCamp"],
        "tags": ["probability"],
        "url": "/boostcamp/probability/",
        "teaser": null
      },{
        "title": "RNN 정리",
        "excerpt":"시퀀스 데이터 소리, 문자열, 주가 등의 데이터를 시퀀스(sequence) 데이터로 분류한다. 시퀀스 데이터는 독립동등분포(Independent and Identically Distributed, i.i.d.)가정을 잘 위배하기 때문에 순서를 바꾸거나 과거 정보에 손실이 발생하면 데이터의 확률분포도 바뀌게 된다. 시퀀스 데이터 핸들링 이전 시퀀스 정보를 가지고 앞으로 발생할 데이터의 확률분포를 다루기 위해 조건부확률을 이용할 수 있다. $P(X_1,…,X_t) = P(X_t|X_1,…,X_{t-1})P(X1,…,X_{t-1})$...","categories": ["BoostCamp"],
        "tags": ["rnn"],
        "url": "/boostcamp/rnn/",
        "teaser": null
      },{
        "title": "Statistics 정리",
        "excerpt":"통계적 모델링 적절한 가정 위에서 확률분포를 추정(inference)하는 것이 목표이며, 기계학습과 통계학이 공통적으로 추구하는 목표입니다. 그러나 유한개의 데이터만 관찰해서 모집단의 분포를 정확하게 알아내는 것은 불가능하므로, 근사적으로 확률분포를 추정해야 합니다. 예측모형의 목적은 분포를 정확하게 맞추는 것보다 데이터의 추정 방법의 불확실성을 고려해서 위험(risk)을 최소화하는 것입니다. 데이터가 특정 확률분포를 따른다고 가정한 뒤 그 분포를...","categories": ["BoostCamp"],
        "tags": ["statistics"],
        "url": "/boostcamp/statistics/",
        "teaser": null
      },{
        "title": "CNN 정리",
        "excerpt":"Convolution Continuous convolution $(f * g)(t) = \\int f(\\tau)g(t-\\tau)d\\tau = \\int f(t-\\tau)g(t)d\\tau$ Discrete convolution $(f * g)(t) = \\sum_{i=-\\infty}^{\\infty} f(\\tau)g(t-\\tau)d\\tau = \\sum_{i=-\\infty}^{\\infty} f(t-\\tau)g(t)d\\tau$ 2D image convolution $(I * K)(i,j) = \\sum_m \\sum_n I(m,n)K(i-m,j-n) = \\sum_m \\sum_n I(i-m,i-n)K(m,n)$ filter 값과 이미지의 값을 컨볼루션 연산한 값을 출력한다. 2D 컨볼루션 연산으로 Blur, Emboss,...","categories": ["BoostCamp"],
        "tags": ["cnn"],
        "url": "/boostcamp/cnn/",
        "teaser": null
      },{
        "title": "Generative models 정리",
        "excerpt":"Generative Models Learning a Generative Model 개의 이미지가 있다고 가정하면 개의 이미지에 대한 확률분포 $p(x)$를 알 수 있다. Generation : 만약 $p(x)$에서 $x_{new}$를 sample할 수 있다면 $x_{new}$ 또한 개의 이미지이다. (샘플링, sampling) Density estimation : $x$가 개인지 고양인지 알아 낼 수 있다. (이상 탐지, anomaly detection) 입력이 주어졌을 때 입력에...","categories": ["BoostCamp"],
        "tags": ["model"],
        "url": "/boostcamp/generative-model/",
        "teaser": null
      },{
        "title": "MLP 정리",
        "excerpt":"Neural Networks Neural Networks are function approximators that stack affine transformations followed by nonlinear transforms. 뉴럴 네트워크는 수학적이고 비선형 연산이 반복적으로 일어나는 어떤 함수를 모방(근사)하는 것이다. Linear Neural Networks 간단한 예제를 들어보자 Data : $D = (x_i,j_i)_{i=1}^N$ Model : $\\hat{y} = wx + b$ Loss : loss$= \\frac{1}{N}\\sum_{i=1}^N (y_i-\\hat{y_i})^2$ 그러면...","categories": ["BoostCamp"],
        "tags": ["mlp"],
        "url": "/boostcamp/mlp/",
        "teaser": null
      },{
        "title": "Optimization 정리",
        "excerpt":"Intorduction Gradient Descent 1차 반복적인 최적화 알고리즘은 미분 가능한 함수의 극소값을 찾기 위한 알고리즘이다. Important Concepts in Optimization Generalization 일반화 성능을 높이는 것이 목표 학습을 진행하게 되면 학습 데이터에 대한 에러가 줄어든다. 그러나 테스트는 일정 에러 이후에 오히려 늘어난다. 이 차이를 Generalization gap이라 한다. “좋은 Generalization performance를 가지고 있다”라는 의미는...","categories": ["BoostCamp"],
        "tags": ["optimization"],
        "url": "/boostcamp/optimization/",
        "teaser": null
      },{
        "title": "Sequential Model 정리",
        "excerpt":"Recurrent Neural Networks Sequential Model 시퀀설 데이터를 처리하는 데 가장 어려운 점은 입력의 차원을 알 수 없다는 점이다. $p(x_t|x_{t-1},x_{t-2},…)$에서 $x_{t-1},x_{t-2},…$는 과거에 내가 고려해야 하는 정보량이다. $p(x_t|x_{t-1},…,x_{t-\\tau}$에서 $x_{t-\\tau}$는 과거에 내가 볼 정보량을 고정하는 것이다. Markov model (first-order autogressive model) $p(x_1,…,x_T) = p(x_T|x_{T-1})p(x_{T-1}|x_{T-2})…p(x_2|x_1)p(x_1)=\\Pi_{t=1}^Tp(x_t|x_{t-1})$ 이 모델의 장점이 joint distribution을 표현하는 것이 쉬워진다 Latent...","categories": ["BoostCamp"],
        "tags": ["model"],
        "url": "/boostcamp/sequential-model/",
        "teaser": null
      },{
        "title": "Attention Is All You Need",
        "excerpt":"Astract 대표적인 문장번역모델들은 인코더와 디코더를 포함한 rnn 혹은 cnn에 기반하고 있습니다. 또한, 가장 성능이 좋은 모델들은 인코더와 디코더가 attention 매커니즘에 의해 연결되어있습니다. 우리는 recurrent와 컨볼루션한 것들을 모두 배제하고 트랜스포머라는 새로운 간단한 네트워크 구조를 제안합니다다. 두번의 기계 번역에서의 실험은 이러한 모델들이 더 병렬화가 가능하고 훈련하는데 시간이 적게 소요됨과 동시에 퀄리티가 좋다는...","categories": ["paper"],
        "tags": ["transformer"],
        "url": "/paper/Attention_Is_All_You_Need/",
        "teaser": null
      },{
        "title": "AtCoder Beginner Contest 215",
        "excerpt":"A. Your First Judge 입력 S가 “Hello,World!”와 같은지 판단하는 문제 #include &lt;iostream&gt; #include &lt;cstring&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; #define endl '\\n' using namespace std; typedef long long ll; typedef pair&lt;int,int&gt; pii; typedef pair&lt;ll,ll&gt; pll; const ll INF=1e10+1; const ll MOD=1e9+7; int main(){ ios::sync_with_stdio(0); cin.tie(0); cout.tie(0); //freopen(\"input.txt\",\"r\",stdin); string s; cin&gt;&gt;s; if(s==\"Hello,World!\")...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc215/",
        "teaser": null
      },{
        "title": "P1 Image Classification 대회 회고",
        "excerpt":"프로젝트 목표 이번 대회는 인물 이미지를 통해 나이, 성별, 마스크 착용에 대해 총 18개의 클래스로 나누고 이를 예측하는 대회입니다. 저는 앙상블을 만들기 위해 적합한 모델을 찾고 다양한 방식을 적용하여 F1 score를 높이는 목표를 가지고 있었습니다. 프로젝트를 수행하기 위한 실험 loss function에 대한 실험 실험목적 : 단일 loss function을 사용했을 때와...","categories": ["Contest"],
        "tags": ["boostcamp"],
        "url": "/contest/image-classification/",
        "teaser": null
      },{
        "title": "Bag-of Words 정리",
        "excerpt":"현재 NLP는 Transformer가 주도하고 그 이전에는 RNN과 같이 Recurrent한 모델이 주도했지만 그 이전에는 단어 및 문서를 숫자 형태로 나타내는 Bag-of Words 기법을 정리한다. Bag-of-Words Representation Step 1. Constructing the vocabulary containing unique words → 사전을 만드는 과정 “John really really loves this movie”, “Jane really likes this song” Vocabulary :...","categories": ["BoostCamp"],
        "tags": ["nlp"],
        "url": "/boostcamp/bag-of-words/",
        "teaser": null
      },{
        "title": "Intro to Natural Language Processing(NLP) 정리",
        "excerpt":"자연어 처리는 기본적으로 컴퓨터가 주어진 단어나 문장, 보다 더 긴 문단이나 글을 이해하는 Natural Language Understanding(NLU)가 있고 이런 자연어를 상황에 따라 적절히 생성할 수 있는 Natural Language Generation(NLG) 두 종류의 task로 구성됩니다. Natural language processing (major conferences: ACL, EMNLP, NAACL) Low-level parsing Tokenization : 문장을 단어(Token) 단위로 나누는 과정 stemming...","categories": ["BoostCamp"],
        "tags": ["nlp"],
        "url": "/boostcamp/introduction-nlp/",
        "teaser": null
      },{
        "title": "Word Embedding 정리",
        "excerpt":"Word Embedding이란, 자연어가 단어들을 정보의 기본 단위로 해서 이런 단어들의 sequence라 볼때, 각 단어들을 특정한 차원으로 이루어진 공간상의 한 점 혹은 그 점의 좌표를 나타내는 벡터로 변환해주는 기법이다. 예를들어, cat과 Kitty는 비슷한 단어이므로, 거리가 가깝다. 그러나 cat과 hamburger는 비슷하지 않는 단어이므로 거리가 멀게 표현될 수 있다. Word embedding을 통해 두...","categories": ["BoostCamp"],
        "tags": ["nlp"],
        "url": "/boostcamp/word-embedding/",
        "teaser": null
      },{
        "title": "AtCoder Beginner Contest 214",
        "excerpt":"A. New Generation ABC 입력받은 대회 회차에 따라 몇 문제가 출제되었는지 그대로 구현하는 문제이다. #include &lt;iostream&gt; #include &lt;cstring&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; #define endl '\\n' using namespace std; typedef long long ll; typedef unsigned long long ull; typedef pair&lt;int,int&gt; pii; typedef pair&lt;ll,ll&gt; pll; const ll INF=1e10+1; const ll MOD=1e9+7; int...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc214/",
        "teaser": null
      },{
        "title": "LSTM & GRU 정리",
        "excerpt":"Long Short-Term Memory (LSTM) Original RNN에서의 Long term dependency를 개선한 모델이다. 매 타임스텝마다 변화하는 hidden state vector를 단기기억을 담당하는 기억소자로 볼 수 있다. 단기기억을 이 시퀀스가 타임스텝별로 진행됨에 따라서 단기기억을 보다 길게 기억할 수 있도록 개선한 모델이다. LSTM은 RNN과 달리 이전 state에서 두 가지의 정보가 들어오는데 $C_{t-1}$와 $h_{t-1}$이다. $C_{t-1}$을 Cell...","categories": ["BoostCamp"],
        "tags": ["nlp"],
        "url": "/boostcamp/long-short-term-memory/",
        "teaser": null
      },{
        "title": "Basics of Recurrent Nerual Networks (RNN)",
        "excerpt":"Types fo RNNs 각 타임스텝에서 들어오는 입력벡터 $X_t$와 이전 state에서 계산된 $h_{t-1}$을 입력으로 받아서 현재 타임스텝에서의 $h_t$를 출력으로 내어주는 구조를 가지고 있다. 모듈 A가 재귀적으로 호출되는 구조를 가지고 있다. 왼쪽의 그림을 rolled 버전, 오른쪽 그림을 unrolled 버전이라 한다. Recurrent Neural Network $h_{t-1}$ : old hidden-state vector $x_t$ : input vector...","categories": ["BoostCamp"],
        "tags": ["nlp"],
        "url": "/boostcamp/recurrent-neural-network/",
        "teaser": null
      },{
        "title": "Encoder-Decoder & Attention mechanism 정리",
        "excerpt":"Seq2Seq Model seq2seq는 RNN의 구조 중 Many-to-many 형태에 해당한다. 인코더와 디코더는 서로 공유하지 않는 모듈을 사용한다. 인코더는 마지막까지 읽고 마지막 타임스텝의 hidden state vector는 디코더의 $h_0$ 즉, 이전 타임스텝의 hidden state vector로써 사용된다. 디코더에 첫 번째 단어로 넣어주는 어떤 하나의 특수문자로서 토큰 혹은 (Start of Sentence)토큰이라 부른다. 이런 특수한 토큰은...","categories": ["BoostCamp"],
        "tags": ["attention"],
        "url": "/boostcamp/attention-mechanism/",
        "teaser": null
      },{
        "title": "Transformer 정리",
        "excerpt":"Attention is all you need No more RNN or CNN modules Attention만으로 sequence 입력, 출력이 가능하다. Bi-Directional RNNs 왼쪽에서 오른쪽으로 흐르는 방식을 Forward라 하면 오른쪽에서 왼쪽으로 흐르는 것을 Backward라 할 수 있다. Forward는 “go”와 왼쪽에 있던 정보까지를 $h_2^f$, Backward는 “go”와 오른쪽에 있던 정보까지를 $h_2^b$라 하자. 인코딩 벡터는 왼쪽에 있던 정보...","categories": ["BoostCamp"],
        "tags": ["transformer"],
        "url": "/boostcamp/transformer/",
        "teaser": null
      },{
        "title": "Self-supervised Pre-training Models 정리",
        "excerpt":"Recent Trends Transformer 모델과 self-attention block의 범용적인 sequence encoder와 decoder가 다양한 자연어 처리 분야에 좋은 성능을 내고 있다. Transformer의 구조적인 변경없이 인코더와 디코더의 스택을 12개 혹은 24개로 사용하고 이 상태에서 Fine-Tuning을 하는 방향으로 사용된다. 추천 시스템, 신약 개발, 영상처리 등으로 확장된다. NLG에서 self-attention 모델이 아직도 greedy decoding에서 벗어나지 못하는 단점도...","categories": ["BoostCamp"],
        "tags": ["nlp"],
        "url": "/boostcamp/self-supervised-pretraining-model/",
        "teaser": null
      },{
        "title": "P2 KLUE Relation Extraction(RE) 대회 회고",
        "excerpt":"대회 목적 이번 대회는 KLUE의 Relation Extraction Task를 수행을 평가합니다. 문장 내 단어의 관계를 알아내면 knowledge graph를 생성하는데 key역할을 하며 생성된 knowlege graph를 통해 QA나 Summarization과 같은 Task를 수행하는데 도움을 줄 수 있습니다. 저는 이번 대회에서 적절한 성능을 내는 모델과 special token을 넣어 성능에 어떤 변화를 일으키는지 확인하는 실험을 진행했습니다....","categories": ["Contest"],
        "tags": ["boostcamp"],
        "url": "/contest/relation-extraction/",
        "teaser": null
      },{
        "title": "jekyll 마크다운과 maxjax 충돌 해결 방법",
        "excerpt":"Latex을 사용하게 될 때 수식의 아래첨자를 사용하기 위해 _를 사용하는 경우가 많습니다. 그러나 $$안에 _가 두 번 쓰이는 경우 마크다운 렌더링 때문에 _italic_되어 수식이 제대로 렌더링되지 않는 경우가 생깁니다. 이런 경우는 markdown 문법과 Latex의 문법의 충돌 때문에 일어나는데, 이를 해결하기 위해서는 kramdown을 수정하는 방법이 있습니다. 하지만 이 블로그의 경우도 그렇고...","categories": ["jekyll"],
        "tags": ["error"],
        "url": "/jekyll/md-latex/",
        "teaser": null
      },{
        "title": "Introduction to MRC",
        "excerpt":"Introduction to MRC Machine Reading Comprehension(MRC)의 개념 기계독해(Machine Reading Comprehension) 주어진 지문(Context)를 이해하고, 주어진 질의(Query/Question)의 답변을 추론하는 문제 MRC의 종류 Extraction Answer Datasets 질의(question)에 대한 답이 항상 주어진 지문(context)의 segment (or span)으로 존재 SQuAD, KorQuAD, NewsQQ, Natural Questions, etc Descriptive/Narrative Answer Datasets 답이 지문 내에서 추출한 span이 아니라, 질의를 보고...","categories": ["BoostCamp"],
        "tags": ["MRC"],
        "url": "/boostcamp/intro-to-mrc/",
        "teaser": null
      },{
        "title": "Extraction-based MRC",
        "excerpt":"Extraction-based MRC Extraction-based MRC 문제 정의 질문(question)의 답변(answer)이 항상 주어진 지문(context)내에 span으로 존재. Extraction-based MRC 평가 방법 Exact Match (EM) Score 예측값과 정답이 캐릭터 단위로 완전히 똑같은 경우에만 1점 부여. 하나라도 다르면 0점 F1 Score 예측값과 정답의 overlap을 비율로 계산. 0점과 1점 사이의 부분점수를 받을 수 있음 Pre-processing Tokenization 텍스트를...","categories": ["BoostCamp"],
        "tags": ["MRC"],
        "url": "/boostcamp/extraction-based-mrc/",
        "teaser": null
      },{
        "title": "Generation-based MRC",
        "excerpt":"Generation-based MRC Generation-based MRC 문제 정의 MRC 문제를 푸는 방법 Extraction-based mrc : 지문(context) 내 답의 위치를 예측 → 분류문제(classification) Generation-based mrc : 주어진 지문과 질의(question)을 보고, 답변을 생성 → 생성문제(generation) generation-based mrc는 지문에 답이 없어도 답을 생성하는 방법이다. 모든 extraction-based mrc문제들은 generation-based mrc 문제로 치환할 수 있다. 반대로 generation-based...","categories": ["BoostCamp"],
        "tags": ["MRC"],
        "url": "/boostcamp/generation-based-mrc/",
        "teaser": null
      },{
        "title": "Passage Retrieval - Dense Embedding",
        "excerpt":"Introduction to Dense Embedding Limitations of sparse embedding 자주 등장하는 단어의 경우 사실상 0이 되면서 90%이상의 벡터 디멘션들이 0이 되는 경우가 발생한다. 차원의 수가 매우 크다 → compressed format으로 극복가능 가장 큰 단점은 유사성을 고려하지 못한다. Dense Embedding Complementary to sparse representations by design 더 작은 차원의 고밀도 벡터(length =...","categories": ["BoostCamp"],
        "tags": ["MRC"],
        "url": "/boostcamp/passage-retrieval-dense/",
        "teaser": null
      },{
        "title": "Passage Retrieval - Sparse Embedding",
        "excerpt":"Introduction to Passage Retrieval Passage Retrieval 질문(query)에 맞는 문서(passage)를 찾는 것. Passage Retrieval with MRC Open-domain Question Answering: 대규모의 문서 중에서 질문에 대한 답을 찾기 Passage Retrieval과 MRC를 이어서 2-Stage로 만들 수 있음 Overview of Passage Retrieval Query와 Passage를 임베딩한 뒤 유사도로 랭킹을 매기고, 유사도가 가장 높은 Passage를 선택함 질문이...","categories": ["BoostCamp"],
        "tags": ["MRC"],
        "url": "/boostcamp/passage-retrieval-sparse/",
        "teaser": null
      },{
        "title": "Scaling Up with FAISS",
        "excerpt":"Passage Retrieval and Similarity Search How to find the passage in real time? ⇒ Similarity Search MIPS (Maximum Inner Product Search) 주어진 질문(query) 벡터 q에 대해 Passage 벡터 v들 중 가장 질문과 관련된 벡터를 찾아야함 여기서 관련성은 내적(inner product)이 가장 큰 것 MIPS &amp; Challenge 실제로 검색해야 할 데이터는 훨씬...","categories": ["BoostCamp"],
        "tags": ["MRC"],
        "url": "/boostcamp/scaling-up-with-faiss/",
        "teaser": null
      },{
        "title": "Linking MRC and Retrieval",
        "excerpt":"Introduction to ODQA Linking MRC and Retrieval: Open-domain Question Answering (ODQA) MRC: 지문이 주어진 상황에서 질의응답 ODQA: 지문이 따로 주어지지 않음. 방대한 World Knowledge에 기반해서 질의응답 Modern search engines: 연관문서 뿐만 아니라 질문의 답을 같이 제공 History of ODQA Text retrieval conference (TREC) - QA Tracks (1999-2007): 연관문서만 반환하는 information...","categories": ["BoostCamp"],
        "tags": ["MRC"],
        "url": "/boostcamp/linking-mrc-and-retrieval/",
        "teaser": null
      },{
        "title": "Reducing Training Bias",
        "excerpt":"Definition of Bias Bias의 종류 Bias in learning 학습할 때 과적합을 막거나 사전 지식을 주입하기 위해 특정 형태의 함수를 선호하는 것 (inductive bias) A Biased World 현실 세계가 편향되어 있기 때문에 모델에 원치 않는 속성이 학습되는 것 (historical bias) 성별과 직업 간 관계 등 표면적인 상관관계 때문에 원치 않는 속성이...","categories": ["BoostCamp"],
        "tags": ["MRC"],
        "url": "/boostcamp/reducing-training-bias/",
        "teaser": null
      },{
        "title": "Closed-book QA with T5",
        "excerpt":"Closed-book Question Answering 모델이 이미 사전학습으로 대량의 지식을 학습했다면, 사전학습 언어모델 자체가 이미 하나의 knowledge storage라고 볼 수 있지 않을까? → 굳이 다른 곳에서 지식을 가져와야할 필요가 있을까? Zero-shot QA performance of GPT-2 사전학습 시 전혀 본적없는 Natural Question 데이터셋에도 어느정도 대답이 가능함 Open-book QA vs. Closed-book QA 지식을 찾는...","categories": ["BoostCamp"],
        "tags": ["MRC"],
        "url": "/boostcamp/closed-book-qa-t5/",
        "teaser": null
      },{
        "title": "QA with Phrase Retrieval",
        "excerpt":"Phrase Retrieval in Open-Domain Qeustion Answering Current limitation of Retriever-Reader approach Error Propagation: 5-10개의 문서만 reader에게 전달 Query-dependent encoding: query에 따라 정답이 되는 answer span에 대한 encoding이 달라짐 How does Document Search work? 사전에 Dense 또는 Sparse vector를 만들고 indexing. MIPS와 같은 방법으로 검색 Retrieve-Read 두 단계 말고 정답을 바로...","categories": ["BoostCamp"],
        "tags": ["MRC"],
        "url": "/boostcamp/OA-phrase-retrieval/",
        "teaser": null
      },{
        "title": "데이터 제작",
        "excerpt":"데이터 제작의 중요성 인공지능 서비스 개발 과정과 데이터 서비스 기획 = 문제 정의 데이터 준비(수집, 정제) - 학습(train) 데이터 모델 학습 - 학습(train) 데이터 모델 검증 → 분석 → 다시 학습 - 검증(validation) 데이터, 개발(development) 데이터 모델 평가 - 평가(test) 데이터 배포 데이터 구축 과정 원시 데이터 선정 및 확보(저작권,...","categories": ["BoostCamp"],
        "tags": ["dataset"],
        "url": "/boostcamp/creating-data/",
        "teaser": null
      },{
        "title": "자연어 처리 데이터 기초",
        "excerpt":"인공지능 모델 개발을 위한 데이터 데이터 종류 말뭉치 류(실제 텍스트 기반의 데이터) 대화문, 기사, SNS 텍스트, 댓글 등 사전/데이터베이스 류(텍스트 분석 시 참조로 사용되는 자원) 온톨로지, 워드넷, 시소러스 등 인공지능 기술의 발전 규칙기반(rule based) → 통계기반(statistics based) → 기계 학습 기반(machne learning based) 언어 모델 평가를 위한 종합적인 벤치마크 등장...","categories": ["BoostCamp"],
        "tags": ["dataset"],
        "url": "/boostcamp/nlp-data/",
        "teaser": null
      },{
        "title": "원시 데이터 수집과 가공",
        "excerpt":"원시 데이터 정의 원시 데이터란? 과제를 해결허기 위해 특정 도메인, 장르, 주제 등에 대하여 조건에 맞춰 수집하였으나, 주석 단계를 거치지 않은 상태의 데이터 원하는 형태로 가공하기 이전의 데이터로 목적에 맞는 전처리 과정을 거쳐 가공이 되어야 활용할 수 있음 원시 텍스트 수집 시 검토 사항 What (수집 대상, 포함 요소(메타정보)) :...","categories": ["BoostCamp"],
        "tags": ["dataset"],
        "url": "/boostcamp/crawling-data/",
        "teaser": null
      },{
        "title": "자연어 처리 데이터 소개",
        "excerpt":"국내 언어 데이터의 구축 프로젝트 국가 주도 : 21 세기 세종 계획(국립 국어원) → 엑소브레인(ETRI) → 모두의 말뭉치(국립 국어원, AI 허브, NIA) 민간 주도 : KLUE(Upstage), KorQuAD(LG CNS), KorNLU(Kakaobrain) 21세기 세종 계획과 모두의 말뭉치 21세기 세종 계획 한국의 국어 정보화 중장기 발전 계획으로 총 2억 어절의 자료 구축, 공개 XML...","categories": ["BoostCamp"],
        "tags": ["dataset"],
        "url": "/boostcamp/nlp-data-introduction/",
        "teaser": null
      },{
        "title": "데이터 구축 가이드라인 작성 기초",
        "excerpt":"가이드라인 유형 목적 : 수집을 위한 가이드라인, 주석을 위한 가이드라인(필수), 검수를 위한 가이드라인 제시 방식 : 문서형, 화면 노출형 튜토리얼형(필수) 가이드라인 구성 요소 데이터 구축 목적 정의, 데이터 구축 시 고려 사항, 사용 용어 정의 목적을 어느정도 말해주는 것이 데이터 품질이 향상될 수 있지만 날것이 필요한 경우(일상대화 등) 어느정도 숨겨야...","categories": ["BoostCamp"],
        "tags": ["dataset"],
        "url": "/boostcamp/guideline/",
        "teaser": null
      },{
        "title": "데이터 구축 작업 설계",
        "excerpt":"데이터 구축 프로세스 과제 정의 데이터 수집 데이터 정제 데이터 주석 ↔ 데이터 검수 데이터 학습 ↔ 데이터 검수 MATTER cycle : Model → Annotate → Train → Test → Evaluate → Revise MAMA cycle : Model → Annotate → Model → Annotate 데이터 구축 프로세스 - 예시 파일럿 구축...","categories": ["BoostCamp"],
        "tags": ["dataset"],
        "url": "/boostcamp/pipelining-dataset/",
        "teaser": null
      },{
        "title": "관계 추출 과제의 이해",
        "excerpt":"관계 추출 관련 과제의 개요 개체명(Entity) 인식(NER, Named Entity Recognition) 개체명이란 인명, 지명, 기관명 등과 같은 고유명사나 명사구를 의미한다. 개체명 인식 태스크는 문장을 분석 대상으로 삼아서 문장에 출현한 개체명의 경계를 인식하고, 각 개체명에 해당하는 태그를 주석함. 가장 널리 알려진 챌린지는 MUC-7, CoNLL 2003이 있다. 한국에서는 TTA의 개체명 태그 세트 및...","categories": ["BoostCamp"],
        "tags": ["dataset"],
        "url": "/boostcamp/relation-extraction-understanding/",
        "teaser": null
      },{
        "title": "관계 추출 데이터 구축 실습",
        "excerpt":"과제 정의 과제 정의 시 고려할 요소 과제의 목적 데이터 구축 규모 원시 데이터 둘 이상의 개체와 개체 간의 관계를 추출할 만한 문장이 포함된 텍스트 선정 데이터의 주석 체계 데이터 주석 도구 주석 단계 세분화 후, 주석 도구 결정 트리플(Triplet)형태의 주석이 가능한 도구 선정 필요 필요기능 문자열에서 개체명 선택 →...","categories": ["BoostCamp"],
        "tags": ["dataset"],
        "url": "/boostcamp/re-data-exc/",
        "teaser": null
      },{
        "title": "P2 MRC 대회 회고",
        "excerpt":"대회 목적 Open Domain에 해당하는 Wikipedia 문서를 불러와 Query에 답을 하는 Open Domain Question Answering(ODQA) 대회입니다. 저는 주어진 Query를 받고 Context내에서 답을 찾는 Reader 모델의 모델링을 맡았습니다. 성능향상을 위한 실험 베이스라인 기존 베이스라인 코드에서 Huggingface의 AutoModelForQuestionAnswering을 불러 학습 시킨 후 Sparse Embedding을 이용한 Retriever를 통해 얻은 문서에서 답을 찾는 구성이었습니다....","categories": ["Contest"],
        "tags": ["boostcamp"],
        "url": "/contest/mrc/",
        "teaser": null
      },{
        "title": "KEYENCE Programming Contest 2021 (AtCoder Beginner Contest 227)",
        "excerpt":"Last Card $K$개의 카드가 있고 1부터 $N$번호를 가진 사람이 있을 때 $A$번호부터 카드를 주게되면 맨 마지막 카드를 받는 사람의 번호가 무엇인지 찾는 문제이다. #include &lt;iostream&gt; #include &lt;cstring&gt; #include &lt;vector&gt; #include &lt;algorithm&gt; #include &lt;numeric&gt; #define endl '\\n' #define fastio ios::sync_with_stdio(0); cin.tie(0); cout.tie(0); using namespace std; typedef long long ll; typedef unsigned...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc227/",
        "teaser": null
      },{
        "title": "Huggingface에 사전학습한 모델 업로드하기",
        "excerpt":"이번 데이터셋 구축을 진행하면서 klue/bert-base처럼 pretraining한 모델을 업로드해보는 경험이 필요하다고 생각하게 되어 실행에 옮겼습니다. 사전학습 준비 먼저, 확보한 데이터는 문장단위였습니다. bert는 MLM(Masked Language Model)과 NSP(Next Sentence Prediction) task를 수행하는데 문장 단위의 데이터로 NSP를 수행할 수 없었습니다. 그래서 bert의 MLM만을 이용하여 pretraining을 진행하였습니다. tokenizer 학습 bert 뿐만 아니라 tokenizer 또한 확보한...","categories": ["BoostCamp"],
        "tags": ["huggingface"],
        "url": "/boostcamp/huggingface-upload-model/",
        "teaser": null
      },{
        "title": "모델 경량화 소개",
        "excerpt":"경량화의 목적 On device AI Smart Phone, Watch, Other IoT devices Limitation: Power Usage(Battery), Ram Memory Usage, Storage, Computing Power AI on cloud(or server) 배터리, 저장 공간, 연산능력의 제약은 줄어드나, latency와 throughput의 제약이 존재 한 요청의 소요시간, 단위 시간당 처리 가능한 요청 수 같은 자원으로 더 적은 latency와 더 큰...","categories": ["BoostCamp"],
        "tags": ["lightweight"],
        "url": "/boostcamp/lightweight-model/",
        "teaser": null
      },{
        "title": "AutoML 이론",
        "excerpt":"Overview Conventional DL Training Pipeline Data Engineering 일반적으로 Data Engineering은 Data Cleansing, Preprocessing, Feature Engineering, Select ML Algorihm, Set Hyperparameters 등의 일련을 거친다. Model Architecture와 Hyperparamter를 선택한 후 Train &amp; Evaluate를 진행하는데 성능이 잘 나오면 Hyperparameter tuning을 진행하고 성능이 나오지 않으면 다른 모델을 선택해서 반복한다. 가장 큰 문제는 위의 과정을...","categories": ["BoostCamp"],
        "tags": ["AutoML"],
        "url": "/boostcamp/automl/",
        "teaser": null
      },{
        "title": "좋은 모델 찾기",
        "excerpt":"Image augmentation Data Augmentation 기존 훈련 데이터에 변화에 가해, 데이터를 추가로 확보하는 방법 데이터가 적거나 Imbalance된 상황에서도 유용하게 활용가능 직관적으로는 모델에 데이터의 불변하는 성질을 전달 → Robust해짐 경량화, AutoML 관점의 augmentation? 경량화 관점에서는 직접적으로 연결이 되지는 않으나, 성능 향상을 위해서는 필수적으로 적용되어야 하는 깁버 Augmentation 또한 일종의 파라미터로, AutoML의 search...","categories": ["BoostCamp"],
        "tags": ["AutoML"],
        "url": "/boostcamp/automl-augmentation/",
        "teaser": null
      },{
        "title": "Pruning and Knowledge Distillation",
        "excerpt":"Pruning Pruning은 중요도가 낮은 파라미터를 제거하는 것 어떤 단위로 Pruning? → Structured(group) / Unstructured(fine grained) 어떤 기준으로 Pruning? → 중요도 정하기(Magnitude(L2, L1), BN scaling facgtor, Energy-based, Feature map, …) 기준은 어떻게 적용? → Network 전체로 줄 세워서(global), Layer 마다 동일 비율로 기준(local) global : 전체 n%, 어떤 layer는 많이, 어떤...","categories": ["BoostCamp"],
        "tags": ["lightweight","pruning","knowledge Distillation"],
        "url": "/boostcamp/pruning-and-KD/",
        "teaser": null
      },{
        "title": "BERT 경량화",
        "excerpt":"Overview CV 경량화와 NLP 경량화의 차이점? Original task → Target task로의 fine tuning하는 방식이 주된 흐름 기본적으로 transforemer 구조 모델 구조가 거의 유사해서, 논문의 재현 가능성이 높고, 코드 재사용성도 높음 BERT Profiling Profiling: Model size and computations Embedding layer: look up table이므로, FLOPs는 없음 Linear before Attn: k, q, v...","categories": ["BoostCamp"],
        "tags": ["bert","lightweight"],
        "url": "/boostcamp/lightweight-bert/",
        "teaser": null
      },{
        "title": "MLOps 정리",
        "excerpt":"모델 개발 프로세스 Research 문제정의 → EDA → Feature Engineering → Train → Predict 위 프로세스는 보통 자신의 컴퓨터, 서버 인스턴스 등에서 실행하고 고정(Static)된 데이터를 사용해 학습한다. 학습된 모델을 앱, 웹 서비스에서 사용할 수 있도록 만드는 과정이 필요하다. 이런 경우 Real World, Production 환경에 모델을 배포한다고 표현한다. Production 웹, 앱...","categories": ["BoostCamp"],
        "tags": ["MLOps"],
        "url": "/boostcamp/MLOps/",
        "teaser": null
      },{
        "title": "Model Serving 정리",
        "excerpt":"Serving Basic Serving 머신러닝 모델을 개발하고, 현실 세계(앱, 웹, Production 환경)에서 사용할 수 있게 만드는 행위 크게 2가지 방식 존재 Online Serving Batch Serving 그 외에 클라이언트(모바일 기기, IoT Device 등)에서 Edge Serving도 존재 Inference : 모델에 데이터가 제공되어 예측하는 경우, 사용하는 관점, Serving과 용어가 혼재되어 사용되는 경우도 존재 Online...","categories": ["BoostCamp"],
        "tags": ["serving"],
        "url": "/boostcamp/model-serving/",
        "teaser": null
      },{
        "title": "머신러닝 라이프 사이클",
        "excerpt":"문제 정의의 중요성 문제 정의란, 특정 현상을 파악하고 그 현상에 있는 문제(Problem)을 정의하는 과정을 말한다. 문제를 잘 풀기(Solve) 위해선 문제 정의(Problem Definition)이 매우 중요함 풀려고 하는 문제가 명확하지 않으면 그 이후 무엇을 해야할지 결정하기 어려워짐 프로젝트 Flow 문제를 해결하기 위한 flow는 다음과 같다. 현상파악 목적, 문제 정의 → 계속 생각하기,...","categories": ["BoostCamp"],
        "tags": ["serving"],
        "url": "/boostcamp/ml-life-cycle/",
        "teaser": null
      },{
        "title": "악성채팅 탐지 시스템 구현",
        "excerpt":"목적 유저가 많은 방의 채팅 속도가 빨라 모든 악성채팅을 거르지 못하기 때문에 실시간으로 탐지하여 관리자에게 악성유저들을 리포트할 수 있는 애플리케이션을 개발하고자 했습니다. 계획 모델 실시간으로 문장의 성향을 파악하기 위해서는 매우 무겁고 느리기 때문에 기존 언어모델을 사용할 수 없었습니다. 그리고 Toxicity text detection은 비교적 쉬운 task에 속합니다. 또한, 서버의 환경을 생각해보면...","categories": ["BoostCamp"],
        "tags": ["cnn","nlp"],
        "url": "/boostcamp/final-project/",
        "teaser": null
      },{
        "title": "Pytorch layer 초기화 함수",
        "excerpt":"모델링을 하게되면 초기화를 신경쓰지 않게 되는데 어떤식으로 이루어지는지 잘 모르고 있었습니다. 그래서 Linear layer를 선언했을 때 weight와 bias를 어떻게 초기화하는지 알아보고자 합니다. Class Linear Linear 레이어 모듈을 살펴보기 위해 pytorch 코드를 가져왔습니다. class Linear(Module): __constants__ = ['in_features', 'out_features'] in_features: int out_features: int weight: Tensor def __init__(self, in_features: int, out_features: int,...","categories": ["Pytorch"],
        "tags": ["torch","initialize"],
        "url": "/pytorch/reset-parameters/",
        "teaser": null
      },{
        "title": "Github Action 정리",
        "excerpt":"CI/CD를 위한 도구로 jenkins, github action 등이 있는데 이 중 github action을 설정하고 pytest와 슬랙 메시지까지 전송해보는 것을 정리했습니다. Github Action 설정 먼저, public으로 Github action을 활성화할 repo를 생성합니다. (private은 사용 요금이 청구되는 것으로 알고 있습니다.) 다음, Actions 탭에서 New workflow로 새로운 워크플로우를 생성합니다. 워크플로우를 선택하는 페이지에서 Python application으로 생성합니다....","categories": ["github"],
        "tags": ["Ops"],
        "url": "/github/github-action/",
        "teaser": null
      },{
        "title": "AtCoder Beginner Contest 233",
        "excerpt":"A. 10yen Stamp 10엔 스탬프를 모아서 현재 $X$에서 목표치 $Y$를 달성할 때까지 몇 번을 봉투에 넣어야 하는지 계산하는 문제이다. $X$가 $Y$보다 큰 경우 0을 출력해주는 예외처리만 해주면 된다. #include &lt;iostream&gt; #include &lt;vector&gt; #include &lt;cstring&gt; #include &lt;algorithm&gt; #include &lt;queue&gt; #include &lt;cmath&gt; #define endl '\\n' #define fastio ios::sync_with_stdio(0); cin.tie(0); cout.tie(0); using namespace...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc233/",
        "teaser": null
      },{
        "title": "airflow 체험기",
        "excerpt":"부캠 때 에러로 사용못한 airflow를 이제서야 체험해본 것을 정리한 글입니다. (무지성 주의) airflow 시작 데이터 엔지니어링 도구인 airflow를 로컬에 설치한 후 몇가지를 살펴보았는데 airflow는 데이터를 관리하는 것이 아닌 함수 단위나 스크립트 단위로 실행시켜주는 도구라는 것을 깨닫게 되었습니다. 그렇다면 데이터를 생성해서 스케줄링을 돌리자!라는 생각이 들어 재밌어보여서 바로 실행에 옮겼습니다. 다시 어떤...","categories": ["airflow"],
        "tags": ["Ops"],
        "url": "/airflow/airflow/",
        "teaser": null
      },{
        "title": "airflow 체험기_최종",
        "excerpt":"이전 글에서 SequentialExecutor에서 CeleryExecutor로 변경하기 위해 삽질한 경험글입니다. CeleryExecutor Celery는 Postgresql과 Mysql만 db로 사용하고 있어서 기존 sqlite를 postgresql로 바꾸는 작업을 진행했습니다. Mysql은 에러가 자주나서 Postgresql을 선택했습니다. 먼저, airflow db init을 하게 되면 AIRFLOW_HOME에 airflow.cfg라는 설정 파일이 생성됩니다. 바꿔야 하는 설정은 다음과 같습니다. # port는 모두 기본포트를 사용하고 있어서 명시해줄 필요가...","categories": ["airflow"],
        "tags": ["docker","Ops"],
        "url": "/airflow/airflow-final/",
        "teaser": null
      },{
        "title": "AtCoder Beginner Contest 250",
        "excerpt":"A. Adjacent Squares (H, W) 크기의 행렬이 주어졌을 때 위치 (R, C)에서 인접한 원소의 수를 출력하는 문제이다. H와 W가 1인 경우를 예외처리해주면 쉽게 풀 수 있다. #include &lt;bits/stdc++.h&gt; #define endl '\\n' #define fastio ios::sync_with_stdio(0); cin.tie(0); cout.tie(0); using namespace std; typedef long long ll; typedef pair&lt;int,int&gt; pii; typedef pair&lt;ll,ll&gt; pll; const...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc250/",
        "teaser": null
      },{
        "title": "Kubernetes",
        "excerpt":"컨테이너 기반 배포 애플리케이션 배포 방식은 물리적인 컴퓨터에 OS와 APP을 설치하여 서비스하던 방식에서 가상화 배포 방식으로 변화했습니다. 가상화 방식은 가상머신 성능을 각각 관리하면서 자원 상황에 따라 조절할 수 있습니다. 그러나 가상머신마다 OS를 새로 설치해야 하고 용량 또한 큽니다. 컨테이너 기반 배포가 등장하면서 Host OS 위에 컨테이너 런타임이 올라가고 그 위로...","categories": ["Ops"],
        "tags": ["k8s"],
        "url": "/ops/kubenetes-overview/",
        "teaser": null
      },{
        "title": "On the Effect of Pretraining Corpora on In-context Learning by a LLM",
        "excerpt":"모두의연구소에서 논문저자가 직접 논문을 리뷰해주는 세미나가 열렸습니다. 주제가 재밌어 보여 발표를 듣고 논문을 다시 읽어보았습니다. Motivation GPT라는 Large Scale Langauge Model이 등장하면서 언어 모델의 새로운 시대를 열게 되었습니다. GPT에서 In-context Learning이라는 방식을 사용하는 점이 특징입니다. In-context Learning In-context Learning은 사전학습 모델에 풀고자 하는 태스크를 input으로 넣는 방식을 말합니다. 예제에 따라...","categories": ["paper"],
        "tags": ["Corpus"],
        "url": "/paper/On-the-effect-of-corpora/",
        "teaser": null
      },{
        "title": "Python으로 MLP 구현하기",
        "excerpt":"코딩테스트로 Python으로만 MLP를 구현하는 문제가 나왔던 적이 있습니다. 당시에 역전파 구현을 하지 못해 코딩테스트에서 떨어졌었고 완전히 바닥에서부터 구현해보고자 시작한 프로젝트입니다. Multi-Layer Perceptron Multi-Layer Perceptron(MLP)은 퍼셉트론으로 이루어진 층(layer)들이 쌓여 신경망을 이루는 모델입니다. 구현이 간단하기 때문에 딥러닝을 바닥부터 구현하는 프로젝트를 시작하는데 좋은 모델입니다. 저는 MNIST를 데이터셋으로 하여 모델을 훈련시키고 classification task를 수행해볼...","categories": ["Pytorch"],
        "tags": ["torch"],
        "url": "/pytorch/dl-implement/",
        "teaser": null
      },{
        "title": "python으로 CNN 구현하기",
        "excerpt":"Related 이전 포스트에서 MLP를 구현했고 이번에는 CNN을 구현하는 삽질을 진행했습니다. 여기서는 Conv2d의 구현에 대해서만 정리하려고 합니다. 밑바닥부터 구현하실때 도움이 되었으면 좋겠습니다. CNN CNN은 [Conv2d + Pooling + (Activation)] 레이어가 수직으로 쌓여있는 뉴럴넷을 말합니다. 구현해보려는 CNN의 구조는 다음과 같습니다. CNN( (layer1): Sequential( (0): Conv2d(1, 5, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))...","categories": ["Pytorch"],
        "tags": ["torch"],
        "url": "/pytorch/cnn-implementation/",
        "teaser": null
      },{
        "title": "데이터 파이프라인 구축해보기",
        "excerpt":"Motivation 빅데이터를 지탱하는 기술을 읽다가 데이터 엔지니어링에 사용되는 플랫폼들을 전체 파이프라인으로 구축해보고 싶어서 이 사이드 프로젝트를 진행하게 되었습니다. Data 먼저, 수집할 데이터는 nginx로부터 나오는 로그를 생각했습니다. 하지만 많은 양의 다양한 로그를 생산하려면 nginx로부터 나오게 하기는 어려워서 python 코드로 비슷한 nginx 로그를 생성하고 /var/log/httpd/access_log/*.log에 logging 모듈로 기록하는 방법으로 로그를 생산했습니다. 생산되는...","categories": ["data-engineer"],
        "tags": ["data-engineering"],
        "url": "/data-engineer/data-pipeline/",
        "teaser": null
      },{
        "title": "python으로 RNN 구현하기",
        "excerpt":"Related 이전 포스트에서 CNN을 구현했고 이번에는 RNN을 구현하는 과정을 정리하려고 합니다. RNN RNN은 Recurrent Neural Network의 약자로 계산을 담당하는 Cell이 순환되는 구조를 말합니다. 이전 timestep의 hidden state vector가 현재 timestep의 계산에 활용되는 특징을 가지고 있습니다. 이번에 구현한 RNN 레이어를 학습하기 위해 계속 사용하던 MNIST를 사용할 생각입니다. MNIST를 (28, 28)의 크기를...","categories": ["Pytorch"],
        "tags": ["torch"],
        "url": "/pytorch/rnn-impl/",
        "teaser": null
      },{
        "title": "Byte Pair Encoding",
        "excerpt":"Reference BPE 알고리즘에 대한 설명은 링크한 곳에 잘 설명되어 있습니다. 여기서는 참고한 곳의 내용을 바탕으로 직접 구현했습니다. https://ratsgo.github.io/nlpbook/docs/preprocess/bpe/ Get Vocabulary 토크나이징을 위해 문서내에 등장한 단어의 등장횟수가 기록된 dictionary를 사용하여 단어 집합인 vocabulary를 만들어야 합니다. 위 코드는 dictionary의 단어들을 구성하는 글자들만을 추출하여 vocabulary에 저장한 코드입니다. 알고리즘 내에서 vocabulary를 사용하고 다시 업데이트를...","categories": ["nlp"],
        "tags": ["nlp"],
        "url": "/nlp/byte-pair-encoding/",
        "teaser": null
      },{
        "title": "python으로 LSTM 구현하기",
        "excerpt":"Related RNN에 이어서 LSTM을 구현했습니다. LSTM LSTM(Long-Short Term Memory)은 RNN의 long-term dependencies 문제를 해결한 모델입니다. 장기의존성(long-term dependencies) 문제란, 과거의 정보가 먼 미래까지 전달되기 어려운 문제를 말합니다. 이러한 long-term dependencies는 gradient vanishinig problem과도 관련이 있습니다. gradient vanishing problem이란, 미분 기울기가 0과 1사이의 값을 가지고 여러번 반복적으로 곱하게 되면 기울기가 0으로 수렴되는...","categories": ["Pytorch"],
        "tags": ["torch"],
        "url": "/pytorch/lstm-implementation/",
        "teaser": null
      },{
        "title": "Wordpiece Tokenizer",
        "excerpt":"Reference Wordpiece 토크나이저는 BERT를 사전학습할때 사용했던 토크나이저입니다. BPE(Byte-Pair Encoding) 토크나이저와 방식은 거의 똑같은데 단어를 합치는 부분이 다른점이 특징입니다. https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt WordPiece tokenization BPE와 동일하게 모든 단어들은 알파벳으로 나눈 후 머지하는 과정을 진행합니다. Wordpiece는 머지할 두 단어를 선택하는 방법이 BPE와 다릅니다. Wordpiece는 다음과 같은 공식을 통해 머지할 pair를 선택합니다. score = (freq_of_pair)...","categories": ["nlp"],
        "tags": ["nlp"],
        "url": "/nlp/wordpiece-tokenizer/",
        "teaser": null
      },{
        "title": "GAMEFREAK Programming Contest 2023 (AtCoder Beginner Contest 317)",
        "excerpt":"A. Potions 현재 체력 H에서 $P_i$만큼 회복시킬 수 있는 포션을 먹을 때 체력이 X만큼 차면서 X에 가장 가까운 i를 찾는 문제이다. 포션이 오름차순으로 입력되므로 포션과 체력을 더해준 뒤 lower_bound로 문제에서 원하는 포션 인덱스를 구한다. #include &lt;bits/stdc++.h&gt; #define endl '\\n' #define fastio ios::sync_with_stdio(0); cin.tie(0); cout.tie(0); using namespace std; typedef long long...","categories": ["AtCoder"],
        "tags": ["ABC"],
        "url": "/atcoder/abc317/",
        "teaser": null
      },{
        "title": "AI Village Capture the Flag @ DEFCON31 후기",
        "excerpt":"AI 관련 CTF가 있는 줄은 몰랐는데 Kaggle에서 해당 대회가 열려 한번 참가하여 한 달간 풀어봤습니다. 대회에서 사용되는 Capture the Flag(CTF) 방식은 취약점을 통해 주최자가 숨겨둔 플래그를 찾아 문제를 해결할 수 있습니다. 이 대회는 27개의 문제로 이루어졌고 테스트 문제 한 개를 제외한 26문제에 각각 혼자서 도전하게 됩니다. 풀어본 문제 중에 일부분의...","categories": ["Contest"],
        "tags": ["kaggle"],
        "url": "/contest/ai-ctf/",
        "teaser": null
      },{
        "title": "Finetuning Large Language Models 정리",
        "excerpt":"Finetuning Large Language Models - Deeplearning.ai Why Finetune? Finetuning은 GPT-3와 같은 범용 모델을 사용하여 채팅을 잘 할 수 있는 ChatGPT 혹은 자동으로 코드를 완성하는 co-pilot과 같은 모델로 전문화하는 것을 말합니다. finetuning은 모델을 보다 일관된 동작으로 조정하는 것 외에도, 환각(hallucination)을 줄이는 데 도움이 될 수 있습니다. finetuning은 prompt engineering과 차이점이 존재합니다....","categories": ["coursera"],
        "tags": ["LLM","nlp"],
        "url": "/coursera/finetuning-LLMs/",
        "teaser": null
      }]
