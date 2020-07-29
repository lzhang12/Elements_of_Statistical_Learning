# Exercise of Chapter 3

$$
\def\Ep#1#2{\mathrm{E}_{#1}\left[{#2}\right]}
\def\bm#1{\boldsymbol{#1}}
\def\argmin#1#2{\mathrm{argmin}_{#1}\left[{#2}\right]}
$$

#### Ex. 3.1
Show that the $F$ statistic (3.13) for dropping a single coefficient from a model is equal to the square of the corresponding $z$-score (3.12).

> The z-score for $i$th feature is 
>
> $$z_i = \frac{\hat{\beta}_i}{\hat{\sigma} \sqrt{\nu_i}}$$,
>
> where $\hat{\beta} = (\bm{X}^T\bm{X})^{-1}\bm{X}^T y$, $\hat{\sigma}^2 = \frac{1}{N-p-1} ||y - \hat{y}||^2$ and $\nu_i$ is the $i$th diagonal element of $(\bm{X}^T\bm{X})^{-1}$.
>
> The $F$ statistic for dropping the $i$th feature from the full model is
>
> $$F = \frac{\mathrm{RSS}_r - \mathrm{RSS}}{\mathrm{RSS}/(N-p-1)}$$,
>
> where $\mathrm{RSS} = ||y - \hat{y}||^2$ and $\mathrm{RSS}_r = ||y - \hat{y}_r||^2$ are the residual sum of squares with full model and reduced model without $i$. $\hat{y}_{i,r}$ is the prediction of the reduced model, $\hat{y}_{i,r} = \bm{X} \hat{\beta}_r$, where $\hat{\beta}_r$ is the $p+1$ coefficient vector of the reduced model, where the $i$the element is 0. Obvisouly, $\hat{\sigma}^2 = \mathrm{RSS}/(N-p-1)$, and the difference of the residual is
>
> $$\begin{aligned} \mathrm{RSS}_r - \mathrm{RSS} & = \sum\limits_{i=1}^N (y_i-\hat{y}_{i,r})^2 - (y_i-\hat{y}_{i})^2 \\ & = \sum\limits_{i=1}^N 2(y_i - \hat{y}_i)(\hat{y}_i-\hat{y}_{i,r}) + (\hat{y}_i-\hat{y}_{i,r})^2 \\ (\text{vector form}) & = 2(y-\hat{y})^T(\hat{y}-\hat{y}_r) + (\hat{y}-\hat{y}_r)^T(\hat{y}-\hat{y}_r) \\ & = 2(y-\hat{y})^T \bm{X}(\hat{\beta}-\hat{\beta}_r) + (\hat{y}-\hat{y}_r)^T(\hat{y}-\hat{y}_r) \\ (\bm{X}^T(y-\hat{y})=0)) & = (\hat{y}-\hat{y}_r)^T(\hat{y}-\hat{y}_r) \\ & = (\hat{\beta} -\hat{\beta}_r)^T \bm{X}^T \bm{X} (\hat{\beta} -\hat{\beta}_r)\end{aligned}$$
>
> To get the form of $\hat{\beta}_r$, we can formulate the optimization problem using Lagrangian multiplier
>
> $$\hat{\beta}_r = \mathrm{argmin}_{\beta} \left[ (y - \bm{X}\beta)^T(y-\bm{X}\beta) + \lambda \delta_i^T \beta \right]$$
>
> where $\delta_i$ is a $p+1$ vector with 1 at $i$th position and zero at other positions. It adds the constraint $\delta_i^T \beta = 0$, requiring the $i$th coefficient to be zero. The reduced coefficient is obtained when the derivative is zero, i.e., 
>
> $$\dfrac{\mathrm{d} \left[ (y - \bm{X}\beta)^T(y-\bm{X}\beta) + \lambda \delta_i^T \beta \right]}{\mathrm{d} \beta} = -2 \bm{X}^T (y-\bm{X}\beta) + \lambda \delta_i = 0$$,
>
> this gives
>
> $$\hat{\beta}_r = (\bm{X}^T\bm{X})^{-1}\bm{X}^Ty - \frac{1}{2}(\bm{X}^T\bm{X})^{-1}\lambda\delta_i$$. 
>
> Due to the constraint $\delta_i^T \beta = 0$, we get ​
>
> $$\lambda = \dfrac{\delta_i^T(\bm{X}^T\bm{X})^{-1}\bm{X}^Ty}{\frac{1}{2}\delta_i^T(\bm{X}^T\bm{X})^{-1}\delta_i} = \dfrac{2\delta_i^T \hat{\beta}}{\nu_i} = \dfrac{2\hat{\beta}_i}{\nu_i}$$.
>
> So
>
> $\hat{\beta}-\hat{\beta}_r = (\bm{X}^T\bm{X})^{-1}\delta_i \frac{\hat{\beta}_i}{\nu_i}$
>
> and
>
> $\mathrm{RSS}_r - \mathrm{RSS} = \left( (\bm{X}^T\bm{X})^{-1}\delta_i \frac{\hat{\beta}_i}{\nu_i} \right)^T \bm{X}^T\bm{X} (\bm{X}^T\bm{X})^{-1}\delta_i \frac{\hat{\beta}_i}{\nu_i} = \frac{\hat{\beta}_i^2}{\nu_i^2}\delta_i^T (\bm{X}^T\bm{X})^{-1}\delta_i = \frac{\hat{\beta_i}^2}{\nu_i}$
>
> Finally,
>
> $$F =  \frac{\mathrm{RSS}_r - \mathrm{RSS}}{\mathrm{RSS}/(N-p-1)} = \frac{\hat{\beta}_i^2}{\hat{\sigma}^2 \nu_i } = z_i^2$$.


#### Ex. 3.2
Given data on two variables $X$ and $Y$ , consider fitting a cubic polynomial regression model $f(x) = \sum\limits_{j=0}^{3} \beta_j X^j$. In addition to plotting the fitted curve, you would like a 95% confidence band about the curve. Consider the following two approaches:
1. At each point $x_0$, form a 95% confidence interval for the linear function $a^T\beta = \sum\limits_{j=0}^3 \beta_j x_0^j$.
2. Form a 95% confidence set for $\beta$ as in (3.15), which in turn generates confidence intervals for $f(x_0)$.

How do these approaches differ? Which band is likely to be wider? Conduct a small simulation experiment to compare the two methods.

> First, let's denote $\beta$ and $\hat{\beta}$ as the true coefficients and the computed coefficients from linear regression.
>
> If we assume that $Y = \sum\limits_{j=1}^3 \beta_j X^j + \epsilon$, where $\epsilon \sim N(0, \sigma^2)$, then $\hat{\beta}$ follows a Gaussian distribution $N(\beta, (\bm{X}^T\bm{X})^{-1}\sigma^2)$.
>
> For 1, since $\hat{\beta}$ is normally distributed, $a^T\hat{\beta}$ is also a normally-distributed variable with mean $a^T\beta$. And $\mathrm{Var}(a^T\hat{\beta}) = a^T \mathrm{Var}(\hat{\beta})a = \sigma^2 a^T (\bm{X}^T\bm{X})^{-1} a$. And $\sigma^2$ is estimated by $\hat{\sigma}^2 = \frac{1}{N-p-1}\sum\limits_{i=1}^{N} (y_i - \hat{y}_i)^2$. So to form a 95% confidence interval for $a^T\hat{\beta}$ is to compute the quantile function of a normal distribution for $p=0.05$, which gives 1.96 multiplied by the squared root of $\hat{\sigma}^2 a^T (\bm{X}^T\bm{X})^{-1} a$.
>
> For 2, we need to first form a 95% confidence set for the coeffiecient vector $\hat{\beta}$. Denote $\gamma = \bm{P}(\hat{\beta} - \beta)$ as a new transformed variable and $\bm{P}$ is an invertible matrix. Since $\mathrm{E}(\beta) = \mathrm{E}(\hat{\beta})$, expectation of $\gamma$ is zero. The variance of $\gamma$ is
>
> $$\mathrm{Var}(\gamma) = \mathrm{Var}(\bm{P}(\hat{\beta} - \beta)) = \bm{P}\mathrm{Var}(\hat{\beta}-\beta) \bm{P}^T = \hat{\sigma}^2 \bm{P}(\bm{X}^T\bm{X})^{-1}
> \bm{P}^T = \hat{\sigma}^2 (\bm{P}^{-T}\bm{X}^T\bm{X} \bm{P}^{-1})^{-1}
> $$.
>
> Notice that the eigen decomposition of $\bm{X}^T\bm{X}$ is 
>
> $$\bm{X}^T\bm{X} = \bm{V}\bm{D}^2\bm{V}^T$$,
>
> where $\bm{V}$ is a $4$-dimensional orthonormal matrix and $\bm{D}$ is a diagonal matrix with singular values of $\bm{X}$. If we take $\bm{P}^{-1} = \bm{V}\bm{D}^{-1}$, i.e., $\bm{P} = \bm{D}\bm{V}^T$, then 
>
> $$\mathrm{Var}(\gamma) = \hat{\sigma}^2 \bm{I}_p$$.
>
> Therefore, $\gamma/\hat{\sigma}$ follows a 4-dimensional standard normal distribution, and the square length of $\gamma/\hat{\sigma}$ will follow a 4-degree chi-square distribution, i.e.,
>
> $$\gamma^T\gamma = (\hat{\beta}-\beta)^T\bm{P}^T\bm{P}(\hat{\beta}-\beta) = (\hat{\beta}-\beta)^T(\bm{D}\bm{V}^T)^T\bm{D}\bm{V}^T(\hat{\beta}-\beta) = (\hat{\beta}-\beta)^T\bm{X}^T\bm{X}(\hat{\beta}-\beta) \sim \hat{\sigma}^2 \chi_4$$.
>
> So, the confidence set of $\hat{\beta}$ is $C(\beta) =\{\beta|(\hat{\beta}-\beta)^T \bm{X}^T \bm{X} (\hat{\beta}-\beta) \leq \hat{\sigma}^2 {\chi^2_4}^{1-\alpha} \}$ with $\alpha=0.95$ for a 95% confidence. Note that $C(\beta)$ is essentially a 4-dimensional sphere with radius of $\hat{\sigma}\sqrt{{\chi_4^2}^{1-0.05}}$, and thus we can sample $\beta$ on the boundary of $C(\beta)$.



#### Ex. 3.3

Gauss–Markov theorem:

(a) Prove the Gauss–Markov theorem: the least squares estimate of a parameter $a^T\beta$ has variance no bigger than that of any other linear unbiased estimate of $a^T\beta$ (Section 3.2.2).

(b) The matrix inequality $\bm{B}\leq  \bm{A}$ holds if $\bm{A} − \bm{B}$ is positive semidefinite. Show that if $\hat{\bm{V}}$ is the variance-covariance matrix of the least squares estimate of $\beta$ and $\tilde{\bm{V}}$ is the variance-covariance matrix of any other linear unbiased estimate, then $\hat{\bm{V}} \leq \tilde{\bm{V}}$.

> (a) 设线性无偏估计$\hat{\theta} = c^T y$，根据无偏性
>
> $$\mathrm{E}(c^T y) = c^T \mathrm{E}(y) = a^T \beta$$
>
> 因为$\bm{X}\beta$是$y$的无偏估计，所以
>
> $$c^T \mathrm{E}(y) = c^T \bm{X} \beta = a^T \beta$$，
>
> 根据$a$的任意性可知$c^T\bm{X} = a^T$。
>
> $\hat{\theta}$的方差为$\mathrm{Var}(\hat{\theta}) = c^T\mathrm{Var}(y)c$。假设$y$变量独立同分布且方差为$\sigma$，则$\mathrm{Var}(y) = \sigma^2\bm{I}$。因此
>
> $$\begin{aligned} \mathrm{Var}(\hat{\theta}) - \mathrm{Var}(a^T\beta) & = \sigma^2 c^Tc - \sigma^2 a^T (\bm{X}^T\bm{X})^{-1}a \\ & = \sigma^2 c^Tc - \sigma^2 c^T \bm{X}(\bm{X}^T\bm{X})^{-1}\bm{X}^Tc \\ & = \sigma^2 c^T (\bm{I} - \bm{X}(\bm{X}^T\bm{X})^{-1}\bm{X}^T)c \\ & = \sigma^2c^T \bm{Q} c\end{aligned}$$
>
> 注意到矩阵$\bm{Q} = \bm{I} - \bm{X}(\bm{X}^T\bm{X})^{-1}\bm{X}^T$存在$\bm{Q}^2 = \bm{Q}$，因此$\bm{Q}$为半正定矩阵，所以上式大于等于零，即$a^T\beta$是所有线性无偏估计中方差最小的。
>
> 利用三角不等式的证明参见[https://github.com/szcf-weiya/ESL-CN/issues/70](https://github.com/szcf-weiya/ESL-CN/issues/70)。
>
> (b)  $\tilde{\beta}$为一个线性无偏估计的系数，且$\hat{\theta} = a^T \tilde{\beta}$。$\hat{V} = \mathrm{Var}(\beta), \tilde{V} = \mathrm{Var}(\tilde{\beta})$，根据(a)的结论，$$a^T \tilde{V}a - a^T\hat{V}a = a^T(\tilde{V} - \hat{V}) a \geq 0$$，由$a$的任意性知$\hat{V} \leq \tilde{V}$。



Ex 3.4 略



#### Ex 3.5 

Consider the ridge regression problem (3.41)

$$\hat{\beta}^{\mathrm{ridge}} = \underset{\beta}{\mathrm{argmin}}\left\{ \sum\limits_{i=1}^{N} (y_i - \beta_0 -\sum\limits_{j=1}^p x_{ij}\beta_j)^2 +\lambda \sum\limits_{j=1}^p \beta_j^2 \right\} $$. Show that this problem is equivalent to the problem

$$\hat{\beta}^{\mathrm{c}} = \underset{\beta^c}{\mathrm{argmin}}\left\{ \sum\limits_{i=1}^{N} (y_i - \beta_0^c -\sum\limits_{j=1}^p (x_{ij}-\overline{x}_j)\beta_j^c)^2 +\lambda \sum\limits_{j=1}^p (\beta^c_j)^2 \right\} $$.

Give the correspondence between $\beta^c$ and the original $\beta$ in (3.41). Characterize the solution to this modified criterion. Show that a similar result holds for the lasso.

> 求解以上优化问题。

#### Ex 3.6

Show that the ridge regression estimate is the mean (and mode) of the posterior distribution, under a Gaussian prior $\beta \sim \mathcal{N}(0,\tau\bm{I})$, and Gaussian sampling model $y \sim \mathcal{N}(\bm{X}\beta, \sigma^2 \bm{I})$. Find the relationship between the regularization parameter $\lambda$ in the ridge formula, and the variances $\tau$ and $\sigma^2$.

> 参见纸质版。