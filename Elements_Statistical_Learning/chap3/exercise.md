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



