---
title: Generative models 정리
categories:
  - BoostCamp
tags: [model]
---
## Generative Models

### Learning a Generative Model
- 개의 이미지가 있다고 가정하면 개의 이미지에 대한 확률분포 $p(x)$를 알 수 있다.
    - Generation : 만약 $p(x)$에서 $x_{new}$를 sample할 수 있다면 $x_{new}$ 또한 개의 이미지이다. (샘플링, sampling)
    - Density estimation : $x$가 개인지 고양인지 알아 낼 수 있다. (이상 탐지, anomaly detection)
        - 입력이 주어졌을 때 입력에 대한 확률 값을 얻어낼 수 있는 모델을 explicit model이라 한다.
            - 예를들어, 이미지를 만들어내고 그 이미지를 분류할 수 있는 모델을 말한다.
        - 반대로 generation만 할 수 있는 모델을 implicit model이라 한다.
    - Unsupervised representation learning : 이미지가 가지고 있는 특징들을 학습할 수 있다. (feature learning)
        - 예를들면 개의 꼬리, 귀와 같은 것들
- how can we represent $p(x)$?

### Basic Discrete Distributions
- 베르누이 분포 : (biased) coin flip -> 0 or 1
    - D = {Heads, Tails}
    - Specify $P(X=Heads) = p$. Then $P(X=Tails) = 1 - p$
    - $X \sim Ber(p)$
    - 이 확률분포를 표현하는 숫자가 하나 필요하다.
- 카테코리 분포 : (biased) m-sided dice
    - D = {1,...,m}
    - Specify $P(Y=i) = p_i$, such that $\sum_{i=1}^m p_i=1$
    - $Y \sim Cat(p_1,...,p_m)$
    - 6면의 주사위를 던지는 데에는 5개의 파라미터만 있으면 된다. 왜냐하면 합계를 통해 나머지 한 면을 알 수 있기 때문

### Example
- RGB joint distribution을 모델링
- $(r,g,b) \sim p(R,G,B)$
- Number of case : 256 $\times$ 256 $\times$ 256
- 이때 파라미터의 수는 256 $\times$ 256 $\times$ 256 - 1
- 파라미터의 수가 많아질수록 학습은 어려워지기 때문에 파라미터 수를 줄이기 위한 노력이 필요하다

### Structure Through Independence
- 만약, $X_1, ..., X_n$이 모두 independent하다면 $p(x_1,...,x_n)=p(x_1)p(x_2)...p(x_n)$로 표현할 수 있다.
- state의 수는 $2^n$개가 된다.
- $p(x_1,...,x_n)$을 표현하기 위한 파라미터의 수는 $n$개이다.
- $2^n$개의 엔트리를 단지 $n$개의 수로 표현가능하다. 그러나 independent하다는 가정이 너무 어렵다

### Conditional Independence
- 3가지의 중요한 룰
    - Chain rule:
        - $p(x_1,...,x_n)=p(x_1)p(x_2\|x_1)..p(x_n\|x_1,...,x_{n-1})$
        - $p(x_1)$ : 1 parameter
        - $p(x_2\|x_1)$ : 2 parameter ($p(x_2\|x_1)=0$이라면 $p(x_2\|x_1)=1$)
        - $p(x_3\|x_1,x_2)$ : 4 parameter
        - 따라서, $1 + 2 + 2^2 + ... + 2^{n-1} = 2^n-1$이므로 이전과 같다.
    - Bayes' rule:
        - $p(x\|y) = \frac{p(x,y)}{p(y)} = \frac{p(y\|x)p(x)}{p(y)}$
    - Conditional independence:
        - if $x \bot y \|z$, then $p(x \|y,z) = p(x\|z)$
        - $z$가 주어졌을 때 $x$와 $y$가 independent할 때, $p(x\|y,z)$는 $p(x\|z)$와 같기 때문에 $y$는 상관이 없다

- 이제 $X_{i+1} \bot X_1,..., X_{i-1}\|X_i$ (Markov assumption)이라고 할 때
    - $p(x_1,...,x_n) = p(x_1)p(x_2\|x_1)...p(x_n\|x_{n-1})$
    - 이제 파라미터 수는 $2n-1$이 되었다.
    - 따라서 Markov assumption을 통해 파라미터 수를 줄일 수 있었다.
    
### Auto-regressive Model
- 28 x 28 바이너리 픽셀이 있다고 가정하자
- $p(x) = p(x_1,...,x_{784}), x \in \[0,1\] ^{784}$
- 이것을 $p(x)$로 어떻게 표현할 수 있는가?
    - chain rule을 사용
    - $p(x_{1:784})=p(x_1)p(x_2\|x_1)p(x_3\|x_{1:2})...$
    - 이러한 것을 autoregressive model이라 한다.
        - autoregressive model은 어떤 하나의 이전 정보가 이전 정보들에 대해 dependent한 것을 말한다.
        - Markov assumption을 통해 i번째 픽셀이 i-1번째 픽셀에만 dependent한 것도 autoregressvie model이다.
        - i번째 픽셀이 1부터 i-1번째 모두에 dependent한 것도 autoregressive model이다.
    - 또한, 이미지 도메인과 같이 순서를 정하는 것이 필요하다.
        - 순서를 매기는 것에 따라 성능과 방법이 달라질 수 있다.
- markov assumption을 주거나 어떤 임의의 conditional independence를 주는 것이 chain rule입장에서 joint distribution을 쪼개는 거에 어떤 관계가 있는 건지 잘 생각해봐야 한다.

### NADE: Neural Autoregressive Density Estimator
- i번째 픽셀을 1부터 i-1에만 dependent하게 만든것
- $p(x_1\|x_{1:i-1}) = \sigma(\alpha_ih_i+b_i)$ where $h_i = \sigma(W{<i}x_{1:i-1}+c)$
- NADE는 explicit model이고 입력의 density를 계산한다.
    - 784개의 바이너리 픽셀이 있다고 가정하면
    - $p(x_1,..,x_{784})=p(x_1)p(x_2\|x_1)...p(x_{784}\|x_{1:784})$
    - 각 conditional probability $p(x_i\|x_{1:i-1})$를 독립적으로 계산한다.
- continuous random variable을 모델링하는 경우 가우시안 믹스쳐 모델을 사용한다.

### Pixel RNN
- 픽셀들을 만들어 내고 싶어하는 generative model이다.
- 예를들어, n x n RGB 이미지를 원한다면
    - $P(x) = \Pi_{i=1}^{n^2}p(x_{i,R}\|x_{<i})p(x_{i,G}\|x_{<i},x_{i,R})p(x_{i,B}\|x_{<i},x_{i,R},x_{i,G})$
    - $p(x_{i,R}\|x_{<i})$ : Prob. i-th Red
    - $p(x_{i,G}\|x_{<i},x_{i,R})$ : Prob. i-th Green
    - $p(x_{i,B}\|x_{<i},x_{i,R},x_{i,G})$ : Prob. i-th Blue
- ordering에 따라 두 가지 모델로 나뉜다.
    - Row LSTM : 위쪽의 있는 정보를 활용하는 방법
    - Diagonal BiLSTM : 자기 이전정보들을 모두 활용한 방법

## Latent Variable Models
---
- autoencoder가 generative model인가?

### Variational Auto-encoder (VA)
- Variational inference (VI)
    - Variational distribution을 찾는 목적은 posterior distribution을 찾기 위함이다.
    - Posterior distribution : $p_{\theta}(z\|x)$
        - Posterior distribution이란 observation이 주어졌을 때 관심있어하는 random variable의 확률분포이다.
        - 여기서 z는 Latent Variable이 된다.
    - Variational distribution : $q_{\phi}(z\|x)$
        - Posterior distribution을 계산하는 것이 너무 힘들기 때문에 근사한 분포를 Variatoinal distribution이라 한다.
- 따라서 관심있는 posterior사이의 KL divergence를 최소화하는(근사하는) variational distribution을 찾고 싶은 것을 Variational inference라 한다.

- posterior가 뭔지도 모르는데 이것을 근사하는 variational distribution을 찾는 다는 것은 말이 안된다.
- 그러나 target이 뭔지 모르는데 loss function을 줄일 수 있게 해주는 것이 VI의 ELBO(Evidence Lower Bound)다.
    - ELBO를 키움으로써 intractable objective를 줄이는 방법이다.
    - In $P_{\theta}(D) = E_{q_{\phi}(z\|x)} \[In \frac{p_{\phi}(x,z)}{q_{\phi}(z\|x)}\](ELBO)+ D_{KL}(q_{\phi}(z\|x)\|\|p_{\theta}(z\|x)) (Objective)$
- 다시 ELBO를 나누게 되면
    - $E_{q_{\phi}(z\|x)} [In \frac{p_{\phi}(x,z)}{q_{\phi}(z\|x)}] = \int In\frac{p_{\theta}(x\|z)p(z)}{q_{\phi}(z\|x)}q_{\phi}(z\|x)dz$
    - = $E_{q_{\phi}(z\|x)}[p_{\theta}(x\|z)] - D_{KL}(q_{\theta}(z\|x)\|\|p(z))$
    - = Reconstruction Term - Prior Fitting Term
        - 인코터를 통해서 x라는 입력을 latent space로 보냈다가 다시 디코더로 돌아오는 reconstruction loss를 줄이는 것이 reconstruction term이다.
        - Prior Fitting Term은 이미지들을 latent space로 올리면 이미지들이 이루는 분포와 내가 가정하고 있는 latent space에서의 prior distribution을 비슷하게 만들어주는 것이 두 가지 분포를 모두 만족하는 것과 같다

- 결국 Variational Auto-encoder는 어떤 입력이 주어지고 latent space로 보내서 무언가를 찾고 다시 reconstruction하는 term으로 만들어지는데 generative model이 되기 위해서는 latent space된 prior distribution z를 샘플링 하고 디코더에 넣어서 나오는 어떤 output 도메인들의 값들이 바로 generation result라고 본다.
- 하지만 엄밀한 의미에서 입력을 넣어서 아웃풋이 나오는  auto-encoder는 generative model이 아니다.

- Key limitation:
    - explicit 모델이 아니다. 어떤 입력이 주어졌을 때 얼마나 likelihood한지 알기 어렵다.
    - prior fitting term에서 KL divergence는 가우시안을 제외하고 closed form이 나오는 경우가 거의 없다.
    - 일반적으로는 isotropic Gaussian을 사용한다.
        - 모든 output dimension이 independent한 분포
        - $D_{KL}(q_{\phi}(z\|x)\|\|N(0,I)) = \frac{1}{2}\sum_{i=1}^D(\sigma_{z_i}^2+\mu_{z_i}^2-ln(\sigma_{z_i}^2)-1)$

### Adversarial Auto-encoder (AA)
- auto encoder의 가장 큰 단점은 가우시안을 활용해야 한다는 점이다.
    - 가우시안이 아닌 다른 분포를 활용하기 위해 adversarial auto-encoder를 사용한다.
- VA보다 AA가 성능이 좋은 경우도 많다.

### Generative Adversarial Network (GAN)
- generator는 discriminator를 속이기위해 fake data를 생성한다.
- discriminator는 real data와 fake data를 판별한다.
- disciminator의 성능이 올라갈수록 generator 또한 성능이 같이 올라간다.
- $min_G max_D V(D,G) = E_{x\sim P_{data}(x)}[log D(x)] + E_{z \sim p_z(z)}[log(1-D(G(z)))]$
- implicit 모델이다.

### GAN vs. VAE
- VAE : x라는 이미지 혹은 입력 도메인이 들어오면 인코더를 통해 latent vector로 갔다가 z를 샘플링해서 디코더를 통해 x가 generative 결과가 된다.
- GAN : z를 통해서 fake가 생성된다. Discriminator는 분류기를 학습하고 generator는 discriminator 입장에서 true가 나오도록 다시 업데이트하고 discriminator는 업데이트된 generator의 생성 결과물을 분류할 수 있도록 성장한다.

### GAN Objective
- generator와 discriminator 사이의 minmax 게임이다.
    - minmax 게임은 한쪽은 높이고 싶어하고 한쪽은 낮추고 싶어하는 게임
- For discriminator
    - $max_D V(G,D) = E_{x \sim p_{data}}[log D(x)] + E_{x \sim p_G} [log(1-(D(x))$
    - where the optimal discriminator is
        - $D_G^*(x) = \frac{P_{data}(x)}{P_{data}(x)+P_G(x)}$
- For generator
    - $min_G V(G,D) = E_{x \sim p_{data}}[log D(x)] + E_{x \sim p_G} [log(1-(D(x))$
    - Plugging in the optimal discriminator,
        - $V(G, D_G^*(x)) = D_{KL}[P_{data},\frac{P_{data}+P_G}{2}] + D_{KL}[P_G,\frac{P_{data}+P_G}{2}] - log 4$
    - 2 x Jenson-Shannon Divergence (JSD) = $2D_{JSD}[P_{data},P_G] - log 4$
    - GAN의 Objective가 많은 경우 true data distribution를 real data distribution에 가까워지게 하는 것
        - 즉, Jenson-Shannon Divergence를 최소화하는 것이다.

## DCGAN

### Info-GAN
- z와 class를 의미하는 c를 통해 한 곳에 집중시켜주는 방법

### Text2Image
- 문장이 주어지면 이미지를 생성

### Puzzle-GAN
- 이미지 안에 subpatch가 있고 subpatch를 가지고 원래 이미지로 복원

### CycleGAN
- Cycle-consistency loss
- GAN 구조가 두 개 들어가 있는 구조

### Star-GAN
- 이미지를 다른 도메인으로 바꾸는 것이 아닌 특정 영역을 컨트롤 할 수 있는 구조

### Progressive-GAN
- 4x4에서 시작해서 1024x1024로 점차적으로 학습시키는 구조
- 한번에 고해상도 이미지를 만드는 것은 무리가 간다.