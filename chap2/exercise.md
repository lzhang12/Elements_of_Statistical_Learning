# Exercise of Chapter 2

**Ex. 2.1**
Suppose each of K-classes has an associated target $t_k$, which is a vector of all zeros, except a one in the $k$th position. Show that classifying to the largest element of $\hat{y}$ amounts to choosing the closest target, $\text{min}_k ||t_k − \hat{y}||$, if the elements of $\hat{y}$ sum to one.


> Essetially, it is to prove $\text{argmin}_k||t_k - \hat{y}||$ is equvalent to $\text{argmax}_k \hat{y}_k$, if $\sum\limits_i \hat{y}_i = 1$.
>
> $$\begin{aligned} \text{argmin}_k ||t_k - \hat{y}||^2 & = \text{argmin}_k \big( \sum\limits_{i\neq k} \hat{y}_i^2 + (1-\hat{y}_k)^2 \big) \\
& = \text{argmin}_k \big(\sum\limits_i \hat{y}_i^2 - 2\hat{y}_k + 1 \big) \\
& = \text{argmin}_k (-\hat{y}_k) \\
& = \text{argmax}_k \hat{y}_k \end{aligned}$$
> Note that $\sum\limits_i \hat{y}_i = 1$ is not necessary.

**Ex. 2.2**
Show how to compute the Bayes decision boundary for the simulation example in Figure 2.5.


> The data generation process first produces 10 mean points $m_k$ for each class, then randomly pick one of these mean points to generate the data points by a bivariate Gaussian distribution $\mathcal{N}(m_k, \bm{I}_m)$, where the covariance matrix $\bm{I}_m =\begin{bmatrix} 0.2 & 0 \\ 0 & 0.2 \end{bmatrix}$.
>
> The Bayesian desicion boundary is given by the Bayesian classifier, which predicts the class by
>
> $$\hat{G} = \underset{g \in \mathcal{G}}{\text{argmax}} \, \text{Pr} (g| X = x)$$
>
> The conditional probability $\text{Pr} (g| X = x)$ can be inferred from the data generation process. For example, for the blue class ($g=0$), the condional probability is the sum of the generation probabilities from the 10 mean points, i.e,
>
> $$\begin{aligned}
\text{Pr} (g=0| X = x) & = \sum\limits_{k=1}^{10} f(x) \\
 & = \sum\limits_{k=1}^{10} \frac{1}{2\pi \sqrt{|\bm{I}_m|}} \exp\big(-\frac{1}{2} (x-m^{(0)}_k)^T\bm{I}_m^{-1} (x-m^{(0)}_k)\big)
\end{aligned}$$
>
> Similarly,
>
> $$\text{Pr} (g=1| X = x) = \sum\limits_{k=1}^{10} \frac{1}{2\pi \sqrt{|\bm{I}_m|}} \exp\big(-\frac{1}{2} (x-m^{(1)}_k)^T\bm{I}_m^{-1} (x-m^{(1)}_k)\big).$$
>
> And the Bayesian decision boundary locates at where
>
> $$\text{Pr} (g=0| X = x) = \text{Pr} (g=1| X = x).$$
>
> Sample python code
```python
class BayesianClassifier(object):
    """
    Bayesian classifier using the data generator
    """
    def __init__(self, n_class, mean, var):
        self.n_class = n_class
        self.mean = mean
        self.cov = var*np.identity(n_class)

    def Pr(self, X):
        """
        Compute probability of each class
        """
        p = np.zeros((X.shape[0], self.n_class))
        for i in range(self.n_class):
            for m in self.mean[i]:
                p[:,i] += multivariate_normal.pdf(X, mean=m, cov=self.cov)
        return p

    def predict(self, X):
        """
        Make prediction
        """
        Y = np.argmax(self.Pr(X), axis=1)
        return Y
```

**Ex. 2.3**
Derive equation (2.24). Consider $N$ data points uniformly distributed in a $p$-dimensional unit ball centered at the origin. Suppose we consider a nearest-neighbor estimate at the origin. The median distance from the origin to the closest data point is given by the expression

$$d(p, N) = (1-(\frac{1}{2})^{1/N})^{1/p}$$.

> The $p$-dimensional volume of a Euclidean ball of radius $r$ in $p$-dimensional Euclidean space is [^1]
>
> $$V_p(r) = \frac{\pi^{p/2}}{\Gamma(p/2 + 1)} r^p.$$
>
> Denote the distance from the origin as a random variable $Y$, and the cumulative distribution function (CDF), denoted as $F$, of the distance is proportional to the volume of the hypersphere,
>
> $$F_Y (r) = \text{Pr}(Y<r) \sim r^p$$,
> Since $\text{Pr}(Y<1)=1$, $F_Y (r) = r^p$.
>
> Assume $N$ independent random variables $Y_1, Y_2, \cdots, Y_N$, and denote the variable with the closest distance as $Y_\mathrm{min}$. The cumulative distribution function of the closest distance is
>
> $$F_{Y_\mathrm{min}}(x; N) = \text{Pr} (Y_\mathrm{min}\leq x; N) = \text{Pr}(\#[(Y_1, Y_2, \cdots, Y_N)\leq x]\geq 1),$$
>
> where $\#[(Y_1, Y_2, \cdots, Y_N)\leq x]$ denotes the number of variables less than $x$.
>
> Due to the independence of the variables,
>
> $$\begin{aligned}
F_{Y_\mathrm{min}}(r; N) & = \sum\limits_{i=1}^N {N \choose i} F_Y(r)^i \big( 1-F_Y(r)\big)^{N-i} \\
& = \sum\limits_{i=0}^N {N \choose i} F_Y(r)^i \big( 1-F_Y(r)\big)^{N-i} - \big(1-F_Y(r)\big)^{N} \\
& = \big( F_Y(r) + 1-F_Y(r) \big)^N - \big(1-F_Y(r)\big)^{N} \\
& = 1 - \big(1-F_Y(r)\big)^{N}
\end{aligned}$$
>
> The median distance is obtained when $F_{Y_\mathrm{min}}(r; N) = 1/2$, thus
>
> $$ d(p, N) = \text{median}(Y_\mathrm{min})= (1-(\frac{1}{2})^{1/N})^{1/p}$$
>

**Ex. 2.4**
The edge effect problem discussed on page 23 is not peculiar to uniform sampling from bounded domains. Consider inputs drawn from a spherical multinormal distribution $X \sim \mathcal{N}(0, \bm{I}_p)$. The squared distance from any sample point to the origin has a $\chi_p^2$ distribution with mean $p$. Consider a prediction point $x_0$ drawn from this distribution, and let $a = x_0/||x_0||$ be an associated unit vector. Let $z_i = a^\mathrm{T} x_i$ be the projection of each of the training points on this direction.

Show that the $z_i$ are distributed $\mathcal{N}(0, 1)$ with expected squared distance from the origin 1, while the target point has expected squared distance $p$ from the origin. Hence for $p = 10$, a randomly drawn test point is about 3.1 standard deviations from the origin, while all the training points are on average one standard deviation along direction $a$. So most prediction points see themselves as lying on the edge of the training set.

> According to the Corollary[^2]:
>
> Distribution of $Z = b\cdot X$, where $X\sim \mathcal{N}(\mu, \Sigma)$, and $b$ is a constant vector with the same number of elements as X and the dot indicates the dot product, is univariate Gaussian with
>
> $$Z\sim\mathcal{N}\left(b\cdot\mu, b^{\rm T}\bm{\Sigma} b\right),$$
> the projection $z_i = a^\mathrm{T} x_i$ has a distribution $\mathcal{N}(a\cdot 0, a^\mathrm{T} \bm{I}_p a) = \mathcal{N}(0, 1)$. The squared distance $z_i^2$ follows a $\chi_1^2$ distribution with a mean value of 1. The expected distance $|z_i|$ follows a $\chi_1$ distribution, and the mean value is about $0.8$ (not $1$ claimed in the question).
>
> For a training point sampled from $\mathcal{N}(0,\bm{I}_p)$, the expected squared distance follows a $\chi_p^2$ distribution, and the mean value is $p$ ($\mathrm{E}(\chi_p^2) = p$). To calculate the mean distance from the origin, simply taking the squared root of $10$, approximately $3.16\approx 3.2$, is not correct. Instead we should use the mean of the $\chi_p$ distribution, which is
>
> $$\mu(\chi_p) = \sqrt(2) \frac{\Gamma((k+1)/2)}{\Gamma(k/2)},$$
> where $\Gamma(x)$ is the gamma function. $\mu(\chi_{10}) \approx 3.08 \approx 3.1$.
>
> Therefore, a random prediction point $x_0$ is about $3.1$ away from the origin, while the mean distance of the other points projected on the direction of $x_0$ reduced to about $0.8$. From the perspective of $x_0$, it sees itself lie on the outer ege of the training set.

**Ex. 2.5**
(a) Derive equation (2.27). The last line makes use of (3.8) through a
conditioning argument.
(b) Derive equation (2.28), making use of the cyclic property of the trace
operator $[\text{trace}(AB) = \text{trace}(BA)]$, and its linearity (which allows us to interchange the order of trace and expectation).

> **(a)** This is an example to show the bias-variance decomposition of error.
>
> Firstly, the training data is generated by a given process, that is, the output $Y$ is related to the input $X$ by $Y=X^\mathrm{T}\beta + \epsilon$ with $\epsilon\sim\mathcal{N}(0,\sigma^2)$. We use the linear regression model to fit the data.
> Given a training set $\mathcal{T}$ and a test point $x_0$, we want to know the expected prediction error of our linear model in theory.
> This error is essentially the mean squared error conditioned at $x_0$, first averaged over the training set $\mathcal{T}$, then averaged over $y_0$, i.e.,
>
> $$\mathrm{EPE}(x_0) = \mathrm{E}_{y_0|x_0}\left[ \mathrm{E}_\mathcal{T} \left[ (y_0 - \hat{y}_0)^2\right]\right].$$
>
> Notice that the expectation of of training set $\mathcal{T}$ includes two random subprocesses (1) generate random training input $\boldsymbol{X}$, (2) generate random noise $\epsilon$. To avoid ambituity, we use $\mathcal{T}_X$ to represent the first one and $\mathcal{T}_\epsilon$ the second one. By $\mathrm{E}_\mathcal{T}[f(\boldsymbol{X}, \epsilon)]$, we mean $\mathrm{E}_{\mathrm{T}_\boldsymbol{X}}[\mathrm{E}_{\mathcal{T}_\epsilon}[f(\boldsymbol{X},\epsilon)|\boldsymbol{X}]].$
>
> The inner expectation w.r.t. the training $\mathcal{T}$ has no influence on $y_0$, i.e., $\mathrm{E}_\mathcal{T}\left[y_0\right] = y_0$, thus
>
> $$\begin{aligned}\mathrm{E}_\mathcal{T}\left[(y_0 -\hat{y}_0)^2\right] = y_0^2 -2y_0 \mathrm{E}_\mathcal{T}\left[\hat{y}_0\right] + \mathrm{E}_\mathcal{T}\left[\hat{y}_0^2\right] \end{aligned}.$$
>
> Further,
>
> $$\begin{aligned}\mathrm{E}_\mathcal{T}\left[\hat{y}_0^2\right] & = \mathrm{E}_\mathcal{T}\left[(\hat{y}_0 - \mathrm{E}_\mathcal{T}[\hat{y}_0] + \mathrm{E}_\mathcal{T}^2[\hat{y}_0])\right] \\ \text{(crossed term is 0)} & = \mathrm{E}_\mathcal{T}[(\hat{y}_0 - \mathrm{E}_\mathcal{T}[\hat{y}_0])^2] + \mathrm{E}_\mathcal{T}^2[\hat{y}_0] \\ & = \mathrm{Var}_\mathcal{T} \left[\hat{y}_0 \right] + \mathrm{E}_\mathcal{T}^2 [\hat{y}_0] \end{aligned}$$
>
> Similarly,
>
> $$\mathrm{E}_\mathcal{y0|x_0}\left[y_0^2\right] = \mathrm{Var}\left[y_0\right] + \mathrm{E}_{y_0|x_0}^2 [y_0].$$
>
> Therefore,
>
> $$\begin{aligned}\mathrm{EPE}(x_0) & = \mathrm{Var}\left[y_0\right] + \mathrm{E}_{y_0|x_0}^2 [y_0] - 2\; \mathrm{E}_{y_0|x_0}[y_0]\; \mathrm{E}_\mathcal{T}[\hat{y}_0] + \mathrm{E}_\mathcal{T}^2 [\hat{y}_0] + \mathrm{Var}_\mathcal{T}\left[\hat{y}_0 \right] \\ & =  \mathrm{Var}\left[y_0\right] + (\mathrm{E}_{y_0|x_0}[y_0] - \mathrm{E}_\mathcal{T}[\hat{y}_0])^2 + \mathrm{Var}_\mathcal{T}\left[\hat{y}_0 \right] \\ & = \mathrm{Var}\left[y_0\right] + \mathrm{Bias}^2[\hat{y}_0] + \mathrm{Var}_\mathcal{T}\left[\hat{y}_0 \right] \end{aligned}$$
>
> Since $y_0 = x_0^\mathrm{T} \beta + \epsilon$, $y_0$ is in fact a Gaussian distribution with $\mathcal{N}(x_0^\mathrm{T} \beta, \sigma^2)$, and
>
> $$\mathrm{E}_{y_0|x_0} [y_0]=x_0^\mathrm{T} \beta.$$
>
> $$\mathrm{Var}[y_0]=\sigma^2.$$
>
> With the help of the normal solution, the linear regression model predicts
>
> $$\begin{aligned}\hat{y}_0 & = x_0^\mathrm{T} (\boldsymbol{X}^\mathrm{T}\boldsymbol{X})^{-1}\boldsymbol{X}^\mathrm{T} (\boldsymbol{X}\beta + \epsilon) \\ & = x_0^\mathrm{T}\beta + x_0^\mathrm{T} (\boldsymbol{X}^\mathrm{T}\boldsymbol{X})^{-1}\boldsymbol{X}^\mathrm{T} \epsilon \\ & = x_0^\mathrm{T}\beta + b^\mathrm{T} \epsilon. \end{aligned}$$
> where we define a vector $b^T = x_0^\mathrm{T} (\boldsymbol{X}^\mathrm{T}\boldsymbol{X})^{-1}\boldsymbol{X}^\mathrm{T}$.
>
> This is actually an affine transformation of the Gaussian distribution $\epsilon$. According to the corollary[^2],
>
> $$\mathrm{E}_\mathcal{T} = \mathrm{E}_{\mathcal{T}_X}[\mathrm{E}_{\mathcal{T}_\epsilon}[\hat{y}_0]] = \mathrm{E}_{\mathcal{T}_X} [x_0^\mathrm{T} \beta]$$
> where we have used $\mathrm{E}_{\mathcal{T}_\epsilon}[b^\mathrm{T}\epsilon] = 0$, and
>
> $$\begin{aligned} \mathrm{Var}_\mathcal{T} [\hat{y}_0] & = \mathrm{E}_{\mathcal{T}_X}[\mathrm{E}_{\mathcal{T}_\epsilon} [(\hat{y}_0 - \mathrm{E}_{\mathcal{T}_X}[\mathrm{E}_{\mathcal{T}_\epsilon}[\hat{y}_0]])^2]] \\ & = \mathrm{E}_{\mathcal{T}_X}[\mathrm{E}_{\mathcal{T}_\epsilon} [(\hat{y}_0 - \mathrm{E}_{\mathcal{T}_X}[\mathrm{E}_{\mathcal{T}_\epsilon}[\hat{y}_0]])^2]] \\ & = \mathrm{E}_{\mathcal{T}_X}[\mathrm{E}_{\mathcal{T}_\epsilon}[(b^\mathrm{T}\epsilon)^2]] \\ & = \mathrm{E}_{\mathcal{T}_X}[b^\mathrm{T}b \, \mathrm{E}_{\mathcal{T}_\epsilon} [\epsilon^\mathrm{T}\epsilon]]\\ & = \mathrm{E}_{\mathcal{T}_X} [x_0^\mathrm{T} (\boldsymbol{X}^\mathrm{T}\boldsymbol{X})^{-1} x_0 ] \sigma^2\end{aligned}$$
>
> Therefore,
>
> $$\mathrm{EPE}(x_0) = \sigma^2 + 0^2 + \mathrm{E}_{\mathcal{T}_X} \left[x_0^\mathrm{T} (\boldsymbol{X}^\mathrm{T}\boldsymbol{X})^{-1} x_0\right] \sigma^2$$
>
>**(b)**
> The second part of this proof is to give an estimation of the $\mathrm{EPE}$ above by assuming large number of samples $N$, $\mathrm{E}[X] = 0$ and a randomly picked $\mathcal{T}$.
>
> The covariance of a $d$-dimensional random vector $X$ is
>
> $$\mathrm{Cov}[X] = \mathrm{E}[X X^\mathrm{T}] - \mathrm{E}[X]\, \mathrm{E}[X]^\mathrm{T} = \mathrm{E}[X X^\mathrm{T}],$$
> where $\mathrm{E}[X] = 0$. When $N$ is large, $\mathrm{E}[X X^\mathrm{T}]$ can be approximated by $\bm{X}^\mathrm{X} \bm{X}/N$ ($\bm{X}$ is input matrix with each row for a data point), thus
>
> $$\bm{X}^\mathrm{T}\bm{X} \approx N \mathrm{Cov}[X].$$
>
> Take the average of $\mathrm{EPE}$ w.r.t $x_0$,
>
> $$\begin{aligned} \mathrm{E}_{x_0} [\mathrm{EPE}|\mathcal{T}] & = \mathrm{E}_{x_0} \left[ x_0^\mathrm{T} \mathrm{Cov}[X]^{-1} x_0 \right]\sigma^2/N + \sigma^2 \\ (x_0^\mathrm{T} \mathrm{Cov}[X]^{-1} x_0 \text{ is a scalar)} & = \mathrm{E}_{x_0} \left[ \mathrm{trace}\left[x_0^\mathrm{T} \mathrm{Cov}[X]^{-1} x_0 \right]\right]\sigma^2/N + \sigma^2 \\ \text{(cyclic property of trace)} & = \mathrm{E}_{x_0} \left[ \mathrm{trace}\left[ \mathrm{Cov}[X]^{-1} x_0 x_0^\mathrm{T} \right]\right]\sigma^2/N + \sigma^2 \\ \text{(linearity of trace)} & = \mathrm{trace}\left[ \mathrm{E}_{x_0} \left[ \mathrm{Cov}[X]^{-1} x_0 x_0^\mathrm{T} \right]\right]\sigma^2/N + \sigma^2 \\ \text{(linearitty of E)} & = \mathrm{trace}\left[ \mathrm{Cov}[X]^{-1} \; \mathrm{E}_{x_0} \left[  x_0 x_0^\mathrm{T} \right]\right]\sigma^2/N + \sigma^2  \\ (\mathrm{E}_{x_0}[x_0 x_0^\mathrm{T}] = \mathrm{Cov}[x_0] ) & = \mathrm{trace}\left[ \mathrm{Cov}[X]^{-1} \; \mathrm{Cov}[x_0] \right]\sigma^2/N + \sigma^2 \\ (X, x_0 \text{ from same distribution} ) & = \mathrm{trace}\left[ \bm{I}_d\right]\sigma^2/N + \sigma^2 \\ & = \frac{p}{N} \sigma^2 + \sigma^2  \end{aligned}$$
>
> (2.28) proved.

[^1]: https://en.wikipedia.org/wiki/Volume_of_an_n-ball
[^2]: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Affine_transformation
