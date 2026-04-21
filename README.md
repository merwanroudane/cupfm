# PyCupFM — Panel Cointegration with Common Factors

<p align="center">
  <a href="https://pypi.org/project/pycupfm/"><img src="https://img.shields.io/pypi/v/pycupfm?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/pycupfm/"><img src="https://img.shields.io/pypi/pyversions/pycupfm" alt="Python"></a>
  <a href="https://github.com/merwanroudane/cupfm/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://merwanroudane.github.io/cupfm/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue" alt="Docs"></a>
</p>

<p align="center">
  <em>Python implementation of all 5 panel cointegration estimators from<br>
  Bai, Kao & Ng (2009, Journal of Econometrics) and Bai & Kao (2005, SSRN)</em>
</p>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **5 Estimators** | LSDV, Bai FM, CupFM ★, CupFM-bar, CupBC |
| **Auto Factor Selection** | Bai & Ng (2002) information criterion |
| **3 Kernels** | Bartlett, Parzen, Quadratic Spectral |
| **Auto Bandwidth** | Newey-West and Andrews plug-in rules |
| **Beautiful Plots** | 9 publication-quality visualization types |
| **Export** | LaTeX, Excel, CSV, HTML tables |
| **Monte Carlo** | Built-in DGP simulation (BKN 2009 design) |
| **Pandas Native** | DataFrame input/output |

## 📦 Installation

```bash
pip install pycupfm
```

## 🚀 Quick Start

```python
from pycupfm import CupFM
from pycupfm.datasets import load_grunfeld

# Load classic Grunfeld investment data (N=10 firms, T=20 years)
df = load_grunfeld()

# Fit all 5 estimators
model = CupFM(n_factors=1, bandwidth=3, max_iter=25)
results = model.fit(
    y=df['linvest'],
    X=df[['lmvalue', 'lkstock']],
    panel_id=df['firm'],
    time_id=df['year'],
    var_names=['lmvalue', 'lkstock'],
    dep_var='linvest'
)

# Publication-quality summary table
results.summary()

# Beautiful visualizations
model.plot(kind='all')
```

## 📊 Output Example

```
==============================================================================
  cupfm — Panel Cointegration with Common Factors        v1.0.0
  Bai, Kao & Ng (2009, JoE 149:82-99)  |  Bai & Kao (2005, SSRN-1815227)
==============================================================================

  Panel Information
  --------------------------------------------------------------------------
  Dependent variable       : linvest         Regressors         : lmvalue, lkstock
  Cross-sections (N)       : 10              Time periods (T)   : 20
  Observations (N×T)       : 200             Panel type         : Balanced
  Common factors (r)       : 1               Bandwidth (M)      : 3 (bartlett)
  --------------------------------------------------------------------------

  Estimation Results
  --------------------------------------------------------------------------
      Variable  |       LSDV     Bai FM      CupFM    CupFM-bar      CupBC
  --------------+----------------------------------------------------------
       lmvalue  |  0.1234***  0.1189***  0.1156***  0.1180***  0.1145***
                |   (  5.67)   (  5.23)   (  5.01)   (  5.12)   (  4.95)
       lkstock  |  0.3456***  0.3312***  0.3289***  0.3301***  0.3278***
                |   (  8.91)   (  8.34)   (  8.22)   (  8.27)   (  8.15)
  --------------------------------------------------------------------------
```

## 📐 Model

Panel cointegrating regression with common factor structure:

$$y_{it} = \alpha_i + \beta' x_{it} + \lambda_i' F_t + u_{it}$$

where $F_t$ are $r$ common I(1) stochastic trends, $\lambda_i$ are heterogeneous factor loadings, and $\beta$ is the homogeneous cointegrating vector.

### Estimators

| Estimator | Method | Iterates | Reference |
|-----------|--------|----------|-----------|
| **LSDV** | Within/FE | No | Biased baseline |
| **Bai FM** | FM correction (1-step) | No | Bai & Kao (2005) |
| **CupFM** ★ | FM + continuous updating | Yes | BKN (2009) Thm 3 |
| **CupFM-bar** | FM + Z-bar instrument | Yes | BKN (2009) |
| **CupBC** | Bias-corrected + updating | Yes | BKN (2009) Thm 2 |

## 🎨 Visualization Gallery

```python
# Individual plots
model.plot(kind='coefficients')   # Forest plot with 95% CIs
model.plot(kind='factors')        # Estimated common factors
model.plot(kind='loadings')       # Factor loadings scatter/bar
model.plot(kind='convergence')    # CupFM iteration path
model.plot(kind='omega')          # Long-run covariance heatmap
model.plot(kind='ic')             # Bai-Ng IC values vs r
```

## 🔬 Monte Carlo Simulation

```python
from pycupfm import simulate_panel, monte_carlo

# Simulate BKN (2009) DGP
sim = simulate_panel(N=20, T=40, k=1, r=2, beta=2.0, seed=42)

# Run Monte Carlo experiment
mc = monte_carlo(n_reps=100, N=20, T=40, verbose=True)
```

## 📤 Export Results

```python
# Export to multiple formats
results.to_latex()           # LaTeX table
results.to_excel('out.xlsx') # Excel workbook
results.to_csv('out.csv')    # CSV file
results.to_dataframe()       # Pandas DataFrame

# Or all at once:
from pycupfm import export_results
export_results(results, 'cupfm_results', fmt='all')
```

## 🔗 Also Available for Stata

This package is the Python companion to the Stata `cupfm` package:
```stata
ssc install cupfm
cupfm y x1 x2, nfactors(2) bandwidth(5) plot
```

## 📚 References

- **Bai, J., Kao, C. & Ng, S. (2009)**. Panel cointegration with global stochastic trends. *Journal of Econometrics*, 149(1), 82-99. [DOI](https://doi.org/10.1016/j.jeconom.2008.10.012)
- **Bai, J. & Kao, C. (2005)**. On the estimation and inference of a panel cointegration model with cross-sectional dependence. CPR Working Paper No. 75. [SSRN](https://ssrn.com/abstract=1815227)
- **Bai, J. & Ng, S. (2002)**. Determining the number of factors in approximate factor models. *Econometrica*, 70(1), 191-221.

## 👤 Author

**Dr. Merwan Roudane**
- Email: merwanroudane920@gmail.com
- GitHub: [@merwanroudane](https://github.com/merwanroudane)

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
