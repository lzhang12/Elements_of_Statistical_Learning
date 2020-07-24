# Exercise of Chapter 5

#### Ex. 5.1
Show that the truncated power basis function in (5.3) 

$$\begin{aligned} h_1(X) = 1, h_3(X) = X^2, h_5(X) = (X - \xi_1)_+^3 \\ h_2(X) = X, h_4(X) = X^3, h_6(X) = (X-\xi_2)_+^3 \end{aligned}$$

represent a basis for a cubic spline with the two knots as indicated.

> Assume the first section $(0, \xi_1)$ is represented by a basis of $(h_1, h_2, h_3, h_4)$, i.e.,
>
> $$f_1(X) = \sum\limits_{i=1}^4 a_i h_i(X)$$
>
> and the second section $(\xi_1, \xi_2)$ is represented by
>
> $$f_2(X) = \sum\limits_{i=1}^4 b_i h_i(X)$$
>
> And there are three constraints at $\xi_1$
>
> $$f_1(\xi_1) = f_2 (\xi_1), f_1'(\xi_1) = f_2'(\xi_2), f_1''(\xi_1) = f_2''(\xi_2)$$.
>
> Since the $a_i$ are known, we have 3 equations and 4 variables ($b_i$). Let's assume $b_4$ is fixed, and we can obtain the other $b_i$ by solving the constraints
>
> $$\begin{aligned} b_3 & = a_3 + 3(a_4-b_4) \xi_1 \\ b_2 & = a_2 -3 (a_4-b_4) \xi_1^2 \\ b_1 & = a_1 + (a_4-b_4)\xi_1^3 \end{aligned}$$
>
> Then $f_2$ can be written as
>
> $$\begin{aligned} f_2(X) & = f_1(X) + (a_4-b_4)\xi_1^3 h_1(X) - 3(a_4-b_4) \xi^2 h_2(X) + 3(a_4 - b_4) \xi h_3(X) - (a_4-b_4) h_4(X) \\ & = f_1(X) - (a_4-b_4)(X -\xi_1)^3 \end{aligned}$$.
>
> So $f_2(X)$ can be seen as a combination of $f_1(X)$ and a new basis function $(X-\xi_1)^3$ when $X\ge\xi_1$. Similarly, it can be proved that $(X-\xi_2)^3$ is the extra basis function for the next section.



#### Ex. 5.4
Consider the truncated power series representation for cubic splines with K interior knots. Let

$$f(X) = \sum\limits_{j=0}^3 \beta_j X^j + \sum\limits_{k=1}^K \theta_k (X-\xi_k)_+^3$$.

Prove that the natural boundary conditions for natural cubic splines imply the following linear constraints on the coefficients:

$$\begin{aligned} & \beta_2=0, \sum\limits_{k=1}^K \theta_k = 0 \\ & \beta_3=0, \sum\limits_{k=1}^K \xi_k\theta_k =0  \end{aligned}$$

Hence derive the basis

$$N_1(X)= 1, N_2(X) = X, N_{k+2}(X) = d_k(X) - d_{K-1}(X)$$

where

$$d_k(X) = \frac{(X-\xi_k)_+^3 - (X-\xi_K)_+^3}{\xi_K - \xi_k} $$.

> ​                   
>
> If $X < \xi_1$, $f(X) = \sum\limits_{j=0}^3 \beta_j X^j$, $f''(X) = 2\beta_2 + 6\beta_3 X$. The natural boundary condition requires that $f''(X) = 0$, which gives $\beta_2 =0, \beta_3 = 0$ since $X$ is arbitrary.
>
> Similarly if $X>\xi_K$, the natural boundary condition leads to $\sum\limits_{k=1}^K\theta_k = 0, \sum\limits_{k=1}^K\xi_k \theta_k = 0$.
>
> Then solve $\theta_{k-1}$ and $\theta_k$ using the two equations above, and replace them in $f(X)$, this gives another basis function.