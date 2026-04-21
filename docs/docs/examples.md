# Examples

## Example 1: Basic Usage

```python
from pycupfm import CupFM
from pycupfm.datasets import load_grunfeld

df = load_grunfeld()
model = CupFM(n_factors=1, bandwidth=3)
results = model.fit(
    y=df['linvest'], X=df[['lmvalue', 'lkstock']],
    panel_id=df['firm'], time_id=df['year']
)
results.summary()
```

## Example 2: Auto Factor Selection

```python
model = CupFM(n_factors='auto', auto_rmax=5, bandwidth=3)
results = model.fit(y=df['linvest'], X=df[['lmvalue', 'lkstock']],
                    panel_id=df['firm'], time_id=df['year'])
print(f'Auto-selected r = {results.r}')
```

## Example 3: All Visualizations

```python
model = CupFM(n_factors=1, bandwidth=3)
results = model.fit(y=df['linvest'], X=df[['lmvalue']],
                    panel_id=df['firm'], time_id=df['year'])
model.plot(kind='all', save_prefix='grunfeld')
```

## Example 4: Monte Carlo Simulation

```python
from pycupfm import simulate_panel, monte_carlo

# Single simulation
sim = simulate_panel(N=20, T=40, k=1, r=2, beta=2.0, seed=42)

# Monte Carlo experiment
mc = monte_carlo(n_reps=100, N=20, T=40, verbose=True)
summary = mc.groupby('estimator')['bias'].agg(['mean', 'std'])
print(summary)
```

## Example 5: Bandwidth Sensitivity

```python
for bw in [2, 3, 5, 8, 10]:
    model = CupFM(n_factors=1, bandwidth=bw)
    r = model.fit(y=df['linvest'], X=df[['lmvalue']],
                  panel_id=df['firm'], time_id=df['year'])
    print(f'BW={bw}: CupFM β = {r.beta[0]:.4f}')
```

## Example 6: Different Kernels

```python
for kern in ['bartlett', 'parzen', 'qs']:
    model = CupFM(n_factors=1, bandwidth=5, kernel=kern)
    r = model.fit(y=df['linvest'], X=df[['lmvalue']],
                  panel_id=df['firm'], time_id=df['year'])
    print(f'{kern}: CupFM β = {r.beta[0]:.4f}')
```

## Example 7: Export Results

```python
results.to_latex()                  # LaTeX table string
results.to_excel('results.xlsx')    # Excel file
results.to_csv('results.csv')       # CSV file

from pycupfm import export_results
export_results(results, 'output', fmt='all')  # All formats
```

## Example 8: Post-Estimation Analysis

```python
import numpy as np

# Compare estimators
for name in results.ESTIMATOR_NAMES:
    b = results.betas[name]
    t = results.tstats[name]
    p = results.pvalues[name]
    print(f'{name:12s}: β = {b[0]:.4f}, t = {t[0]:.2f}, p = {p[0]:.4f}')

# LSDV bias
bias = results.betas['LSDV'] - results.betas['CupFM']
print(f'LSDV bias vs CupFM: {bias}')
```
