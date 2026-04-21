<div class="hero-banner" markdown>

# 🔬 PyCupFM

<p class="hero-subtitle">
<strong>Panel Cointegration with Common Factors</strong><br>
Python implementation of all 5 estimators from Bai, Kao & Ng (2009, <em>Journal of Econometrics</em>)
and Bai & Kao (2005)
</p>

<div class="hero-badges">
<a href="https://pypi.org/project/pycupfm/"><img src="https://img.shields.io/pypi/v/pycupfm?color=00bfa5&label=PyPI&style=for-the-badge" alt="PyPI"></a>
<a href="https://pypi.org/project/pycupfm/"><img src="https://img.shields.io/pypi/pyversions/pycupfm?style=for-the-badge&color=7c4dff" alt="Python"></a>
<a href="https://github.com/merwanroudane/cupfm"><img src="https://img.shields.io/badge/license-MIT-00c853?style=for-the-badge" alt="License"></a>
<a href="https://github.com/merwanroudane/cupfm"><img src="https://img.shields.io/badge/Stata-SSC-ff6d00?style=for-the-badge" alt="Stata"></a>
</div>

</div>

<div class="install-box" markdown>

### 📦 Install in one line

`pip install pycupfm`

</div>

---

<div class="section-header" markdown>

## ✨ Features at a Glance

</div>

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
<div class="feature-icon">📐</div>

### 5 Estimators

LSDV, Bai FM, **CupFM** ★, CupFM-bar, CupBC — faithfully translated from the original GAUSS source code

</div>

<div class="feature-card" markdown>
<div class="feature-icon">🎨</div>

### Beautiful Visualizations

9 publication-quality plot types with premium academic aesthetics using matplotlib

</div>

<div class="feature-card" markdown>
<div class="feature-icon">🔍</div>

### Auto Factor Selection

Bai & Ng (2002) IC for automatic factor number selection + auto-bandwidth

</div>

<div class="feature-card" markdown>
<div class="feature-icon">📤</div>

### Export Everywhere

LaTeX, Excel, CSV, HTML tables — ready for your next paper submission

</div>

<div class="feature-card" markdown>
<div class="feature-icon">🧪</div>

### Monte Carlo Tools

Built-in DGP simulation replicating BKN (2009) Tables 1-4

</div>

<div class="feature-card" markdown>
<div class="feature-icon">🐼</div>

### Pandas Native

DataFrame input/output with automatic variable name inference

</div>

</div>

---

<div class="section-header" markdown>

## 🚀 Quick Start — 5 Lines of Code

</div>

```python
from pycupfm import CupFM
from pycupfm.datasets import load_grunfeld

df = load_grunfeld()  # N=10 firms, T=20 years
model = CupFM(n_factors=1, bandwidth=3, max_iter=25)
results = model.fit(
    y=df['linvest'], X=df[['lmvalue', 'lkstock']],
    panel_id=df['firm'], time_id=df['year'],
    var_names=['lmvalue', 'lkstock'], dep_var='linvest'
)
results.summary()
```

---

<div class="section-header" markdown>

## 📊 Output — Publication-Quality Summary Table

</div>

<div class="output-block">
==============================================================================
  cupfm — Panel Cointegration with Common Factors        v1.0.0
  Bai, Kao & Ng (2009, JoE 149:82-99)  |  Bai & Kao (2005, SSRN-1815227)
==============================================================================

  Panel Information
  --------------------------------------------------------------------------
  Dependent variable     : linvest         Regressors         : lmvalue, lkstock
  Cross-sections (N)     : 10              Time periods (T)   : 20
  Observations (N×T)     : 200             Panel type         : Balanced
  Common factors (r)     : 1               Bandwidth (M)      : 3 (bartlett)
  Max iterations         : 25              CupFM iterations   : 25
  --------------------------------------------------------------------------

  Estimation Results
  --------------------------------------------------------------------------
      Variable  |       LSDV     Bai FM      CupFM  CupFM-bar      CupBC
  --------------+----------------------------------------------------------
       lmvalue  |  0.5224***  0.5840***  0.7421***  0.6952***  0.5824***
                |   (  6.81)   ( 11.41)   ( 10.43)   (  7.58)   ( 11.38)
  --------------+----------------------------------------------------------
       lkstock  |  0.0827***  0.0789***  0.0664**   0.0716*    0.0791***
                |   (  3.35)   (  2.92)   (  2.34)   (  1.87)   (  2.93)
  --------------------------------------------------------------------------
  t-statistics in parentheses  |  *** p<0.01  ** p<0.05  * p<0.10
  CupFM = recommended (BKN 2009, Theorem 3)  |  CupFM-bar = Z-bar variant
</div>

---

<div class="section-header" markdown>

## 🎨 Visualization Gallery

</div>

<div class="plot-gallery" markdown>

<div class="plot-card">
<img src="assets/images/coef_plot.png" alt="Coefficient Forest Plot">
<div class="plot-caption">📊 <strong>Coefficient Comparison</strong> — All 5 estimators with 95% confidence intervals</div>
</div>

<div class="plot-card">
<img src="assets/images/factors_plot.png" alt="Factor Time Series">
<div class="plot-caption">📈 <strong>Estimated Common Factors</strong> — F̂ₜ time series with r = 1</div>
</div>

<div class="plot-card">
<img src="assets/images/loadings_plot.png" alt="Factor Loadings">
<div class="plot-caption">📉 <strong>Factor Loadings</strong> — Heterogeneous λᵢ across cross-sections</div>
</div>

<div class="plot-card">
<img src="assets/images/convergence_plot.png" alt="Convergence Path">
<div class="plot-caption">🔄 <strong>CupFM Convergence</strong> — β estimates across iterations</div>
</div>

<div class="plot-card">
<img src="assets/images/omega_plot.png" alt="Omega Heatmap">
<div class="plot-caption">🌡️ <strong>Long-Run Covariance Ω̂</strong> — Kernel-estimated covariance matrix</div>
</div>

<div class="plot-card">
<img src="assets/images/ic_plot.png" alt="Factor IC Plot">
<div class="plot-caption">⭐ <strong>Bai-Ng IC</strong> — Automatic factor number selection (r* = optimal)</div>
</div>

</div>

---

<div class="section-header" markdown>

## 📐 The Model

</div>

Panel cointegrating regression with common factor structure:

$$y_{it} = \alpha_i + \beta' x_{it} + \lambda_i' F_t + u_{it}$$

where $F_t$ are $r$ common I(1) stochastic trends and $\lambda_i$ are heterogeneous loadings.

<table class="estimator-table">
<thead>
<tr><th>Estimator</th><th>Method</th><th>Iterates</th><th>Reference</th><th>Status</th></tr>
</thead>
<tbody>
<tr><td>LSDV</td><td>Within / FE</td><td>✗</td><td>Biased baseline</td><td>⚠️ Biased</td></tr>
<tr><td>Bai FM</td><td>FM correction (1-step)</td><td>✗</td><td>Bai & Kao (2005)</td><td>✅ Consistent</td></tr>
<tr><td class="star">CupFM ★</td><td>FM + continuous updating</td><td>✓</td><td>BKN (2009) Thm 3</td><td>⭐ Recommended</td></tr>
<tr><td>CupFM-bar</td><td>FM + Z-bar instrument</td><td>✓</td><td>BKN (2009)</td><td>✅ Consistent</td></tr>
<tr><td>CupBC</td><td>BC + updating</td><td>✓</td><td>BKN (2009) Thm 2</td><td>✅ Consistent</td></tr>
</tbody>
</table>

---

<div class="section-header" markdown>

## 🔗 Also Available for Stata

</div>

```stata
ssc install cupfm
cupfm y x1 x2, nfactors(2) bandwidth(5) plot
```

---

<div class="author-card" markdown>

### 👤 Dr. Merwan Roudane

📧 [merwanroudane920@gmail.com](mailto:merwanroudane920@gmail.com)

🔗 [GitHub](https://github.com/merwanroudane) · [PyPI](https://pypi.org/project/pycupfm/) · [Stata SSC](https://ideas.repec.org/c/boc/bocode/s459351.html)

</div>
