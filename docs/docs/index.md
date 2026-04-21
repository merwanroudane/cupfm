# PyCupFM

## Panel Cointegration with Common Factors

<div style="text-align: center; margin: 2rem 0;">
<p style="font-size: 1.2rem; color: #555;">
Python implementation of all 5 panel cointegration estimators from<br>
<strong>Bai, Kao & Ng (2009, Journal of Econometrics)</strong> and <strong>Bai & Kao (2005, SSRN)</strong>
</p>
</div>

---

## ✨ Highlights

<div class="grid cards" markdown>

-   :material-calculator-variant:{ .lg .middle } **5 Estimators**

    ---

    LSDV, Bai FM, **CupFM** ★, CupFM-bar, CupBC — all from the official GAUSS source code

-   :material-chart-line:{ .lg .middle } **Beautiful Visualizations**

    ---

    9 publication-quality plot types with premium academic aesthetics

-   :material-auto-fix:{ .lg .middle } **Automatic Selection**

    ---

    Bai & Ng (2002) IC for factor number, auto-bandwidth via Newey-West

-   :material-export:{ .lg .middle } **Export Everywhere**

    ---

    LaTeX, Excel, CSV, HTML tables — ready for your paper

</div>

## Quick Start

```python
from pycupfm import CupFM
from pycupfm.datasets import load_grunfeld

df = load_grunfeld()
model = CupFM(n_factors=1, bandwidth=3)
results = model.fit(
    y=df['linvest'],
    X=df[['lmvalue', 'lkstock']],
    panel_id=df['firm'],
    time_id=df['year']
)
results.summary()
```

## Installation

```bash
pip install pycupfm
```

## Estimators

| Estimator | Method | Iterates | Consistent if F~I(1) |
|-----------|--------|:--------:|:--------------------:|
| LSDV | Within/FE | ✗ | ✗ |
| Bai FM | FM (1-step) | ✗ | ✓ |
| **CupFM** ★ | FM + updating | ✓ | ✓ |
| CupFM-bar | FM + Z-bar | ✓ | ✓ |
| CupBC | BC + updating | ✓ | ✓ |

## Also Available for Stata

```stata
ssc install cupfm
cupfm y x1 x2, nfactors(2) bandwidth(5) plot
```

## Author

**Dr. Merwan Roudane**  
📧 [merwanroudane920@gmail.com](mailto:merwanroudane920@gmail.com)  
🔗 [GitHub](https://github.com/merwanroudane)
