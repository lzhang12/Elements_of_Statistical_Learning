# Exercise of Chapter 2

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

[^1]: https://en.wikipedia.org/wiki/Volume_of_an_n-ball
