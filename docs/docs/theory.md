# Theory

## Panel Cointegrating Regression

The model estimated by PyCupFM is:

$$y_{it} = \alpha_i + \beta' x_{it} + e_{it}, \quad i = 1,\ldots,N; \; t = 1,\ldots,T$$

where the error $e_{it}$ has a **common factor structure**:

$$e_{it} = \lambda_i' F_t + u_{it}$$

### Variables

- $y_{it}$: dependent variable (I(1))
- $x_{it}$: $k \times 1$ vector of I(1) regressors with $x_{it} = x_{i,t-1} + v_{it}$
- $F_t$: $r \times 1$ vector of **common factors** — I(1) global stochastic trends
- $\lambda_i$: $r \times 1$ **heterogeneous** factor loadings
- $u_{it}$: idiosyncratic error (stationary, may be serially correlated)
- $\alpha_i$: unit-specific fixed effects
- $\beta$: $k \times 1$ **homogeneous** cointegrating coefficients

---

## The 5 Estimators

### 1. LSDV (Within / Fixed-Effects)

Standard panel within estimator. Eliminates $\alpha_i$ by demeaning:

$$\hat\beta_{LSDV} = \left(\sum_i \tilde{X}_i' \tilde{X}_i\right)^{-1} \sum_i \tilde{X}_i' \tilde{y}_i$$

where $\tilde{X}_i = X_i - \bar{X}_i$. **Biased and inconsistent** when $F_t \sim I(1)$.

### 2. Bai FM (Two-Step Fully Modified)

From Bai & Kao (2005), Equations 7–8. Non-iterative:

1. Estimate LSDV → get $\hat\beta_0$, extract factors $\hat{F}$, $\hat\Lambda$
2. Construct FM correction using Bartlett long-run covariance $\hat\Omega$
3. Apply bias correction once

### 3. CupFM — Continuously-Updated FM ★ Recommended

From BKN (2009), Theorem 3, Equation 16. Iterates the Bai FM procedure:

$$\hat\beta^{(j)} \to \text{residuals} \to \text{PCA} \to \hat\Omega \to \text{FM correction} \to \hat\beta^{(j+1)}$$

**Asymptotic distribution** (BKN 2009, Theorem 3):

$$\sqrt{NT}\left(\hat\beta_{CupFM} - \beta\right) \xrightarrow{d} N\left(0, 6\Omega_{uu \cdot x} / \Sigma_{xx}\right)$$

Exhibits the **smallest bias** in all BKN Monte Carlo experiments.

### 4. CupFM-bar (Z-bar Variant)

Uses the instrument $\bar{Z}_i = \bar{x}_i - \hat{F}\hat\delta_i$ instead of $X_i$.

### 5. CupBC (Continuously-Updated Bias-Corrected)

From BKN (2009), Theorem 2. Iterates plain Cup-LS:

$$\hat\beta_{BC}^{(j)} = \left(\sum_i X_i' M_{\hat{F}} X_i\right)^{-1} \sum_i X_i' M_{\hat{F}} y_i$$

then applies bias correction at convergence.

---

## Long-Run Covariance Estimation

The long-run covariance matrix is estimated using kernel methods:

$$\hat\Omega = \frac{1}{T}\sum_{j=-M}^{M} w_j \sum_t z_t z_{t-j}'$$

### Available Kernels

| Kernel | Weight Function |
|--------|----------------|
| **Bartlett** | $w_j = 1 - \|j\|/(M+1)$ |
| **Parzen** | Smooth taper (cubic) |
| **Quadratic Spectral** | Andrews (1991) |

### Bandwidth Selection

- **Manual**: `bandwidth=5` (BKN default)
- **Auto (Newey-West)**: $M = \lfloor 4(T/100)^{2/9} \rfloor$
- **Auto (Andrews)**: AR(1) plug-in rule

---

## Factor Selection

The number of factors $r$ is selected by minimizing the Bai & Ng (2002) information criterion:

$$IC_1(k) = \ln V(k, \hat{F}) + k \cdot \frac{N+T}{NT} \cdot \ln\frac{NT}{N+T}$$

where $V(k, \hat{F}) = \frac{1}{NT}\sum_i\sum_t (e_{it} - \hat\lambda_i'\hat{F}_t)^2$.

---

## References

1. **Bai, J., Kao, C. & Ng, S. (2009)**. Panel cointegration with global stochastic trends. *Journal of Econometrics*, 149(1), 82-99.
2. **Bai, J. & Kao, C. (2005)**. On the estimation and inference of a panel cointegration model with cross-sectional dependence. SSRN-1815227.
3. **Bai, J. & Ng, S. (2002)**. Determining the number of factors in approximate factor models. *Econometrica*, 70(1), 191-221.
4. **Andrews, D.W.K. (1991)**. Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica*, 59(3), 817-858.
