# API Reference

## Main Class

### `CupFM`

```python
from pycupfm import CupFM

model = CupFM(
    n_factors=2,          # int or 'auto'
    kernel='bartlett',    # 'bartlett', 'parzen', 'qs'
    bandwidth=5,          # int or 'auto'
    max_iter=20,          # int (1-500)
    tol=1e-4,             # float
    auto_rmax=8,          # int (max r for auto-selection)
    verbose=False,        # bool
)
```

#### `model.fit(y, X, panel_id, time_id, var_names=None, dep_var='y')`

Fit all 5 panel cointegration estimators.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | array-like (N*T,) | Dependent variable |
| `X` | array-like (N*T, k) | Independent variables |
| `panel_id` | array-like (N*T,) | Panel unit identifiers |
| `time_id` | array-like (N*T,) | Time period identifiers |
| `var_names` | list of str | Regressor names |
| `dep_var` | str | Dependent variable name |

**Returns:** `CupFMResults`

#### `model.summary()`

Print the summary table from the last fit.

#### `model.plot(kind='all', save_prefix=None)`

Generate plots. `kind` options: `'all'`, `'coefficients'`, `'factors'`, `'loadings'`, `'convergence'`, `'omega'`, `'ic'`.

---

## Results Class

### `CupFMResults`

Returned by `model.fit()`.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `beta` | ndarray (k,) | CupFM coefficients (primary) |
| `tstat` | ndarray (k,) | CupFM t-statistics |
| `pvalue` | ndarray (k,) | CupFM p-values |
| `betas` | dict | All 5 estimator coefficients |
| `tstats` | dict | All 5 t-statistics |
| `pvalues` | dict | All 5 p-values |
| `se` | dict | All 5 standard errors |
| `ci` | dict | All 5 confidence intervals |
| `F_hat` | ndarray (T, r) | Estimated factors |
| `Lambda` | ndarray (N, r) | Estimated loadings |
| `N`, `T`, `r` | int | Panel dimensions |
| `convergence_path` | ndarray | β across iterations |

#### Methods

| Method | Description |
|--------|-------------|
| `summary()` | Print publication-quality table |
| `to_dataframe()` | Convert to pandas DataFrame |
| `to_latex(caption=None)` | Generate LaTeX table |
| `to_excel(filename)` | Export to Excel |
| `to_csv(filename)` | Export to CSV |

---

## Simulation

### `simulate_panel(N, T, k, r, beta, ...)`

Simulate panel data from the BKN (2009) DGP.

```python
from pycupfm import simulate_panel

sim = simulate_panel(N=20, T=40, k=1, r=2, beta=2.0, seed=42)
```

### `monte_carlo(n_reps, N, T, ...)`

Run a full Monte Carlo experiment.

```python
from pycupfm import monte_carlo

mc = monte_carlo(n_reps=100, N=20, T=40, verbose=True)
```

---

## Data

### `load_grunfeld()`

Load the classic Grunfeld (1958) investment dataset.

```python
from pycupfm.datasets import load_grunfeld
df = load_grunfeld()  # N=10, T=20
```

---

## Plotting Functions

All functions return `matplotlib.figure.Figure`.

| Function | Description |
|----------|-------------|
| `plot_coefficients(results)` | Forest plot with 95% CIs |
| `plot_factors(results)` | Common factor time series |
| `plot_loadings(results)` | Loading bar/scatter |
| `plot_convergence(results)` | Iteration path |
| `plot_omega_heatmap(results)` | Ω covariance heatmap |
| `plot_factor_ic(ic_values)` | Bai-Ng IC vs r |
| `plot_loadings_heatmap(results)` | N×r loading heatmap |
| `plot_all(results)` | All plots at once |

---

## Kernel Functions

```python
from pycupfm import long_run_covariance, auto_bandwidth

Omega, Delta_plus = long_run_covariance(U, bandwidth=5, kernel='bartlett')
bw = auto_bandwidth(residuals, method='nw')
```

Available kernels: `'bartlett'`, `'parzen'`, `'qs'` / `'quadratic_spectral'`
