# Getting Started

## Installation

```bash
pip install pycupfm
```

### Requirements

- Python ≥ 3.9
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- Pandas ≥ 1.3
- Matplotlib ≥ 3.5
- Seaborn ≥ 0.12

## Your First Model

```python
from pycupfm import CupFM
from pycupfm.datasets import load_grunfeld

# 1. Load data
df = load_grunfeld()

# 2. Create model
model = CupFM(
    n_factors=1,        # number of common factors
    bandwidth=3,        # Bartlett kernel bandwidth
    kernel='bartlett',  # kernel type
    max_iter=25,        # max iterations
)

# 3. Fit all 5 estimators
results = model.fit(
    y=df['linvest'],
    X=df[['lmvalue', 'lkstock']],
    panel_id=df['firm'],
    time_id=df['year'],
    var_names=['lmvalue', 'lkstock'],
    dep_var='linvest'
)

# 4. View results
results.summary()

# 5. Generate plots
model.plot(kind='all')

# 6. Export
results.to_latex()
results.to_excel('results.xlsx')
```

## Data Format

PyCupFM expects **long-format** panel data:

| panel_id | time_id | y | x1 | x2 |
|----------|---------|---|----|-----|
| 1 | 1 | 5.2 | 3.1 | 2.4 |
| 1 | 2 | 5.8 | 3.5 | 2.6 |
| ... | ... | ... | ... | ... |
| N | T | 4.1 | 2.8 | 1.9 |

!!! important "Requirements"
    - **Balanced panel**: every unit must have exactly T observations
    - **I(1) variables**: all variables should be integrated of order 1
    - **No missing values**: remove or impute NaN before fitting

## Automatic Factor Selection

```python
# Let Bai-Ng (2002) IC choose the number of factors
model = CupFM(n_factors='auto', auto_rmax=8)
results = model.fit(y, X, panel_id, time_id)
print(f'Selected r = {results.r}')
```

## Using Your Own Data

```python
import pandas as pd
from pycupfm import CupFM

df = pd.read_csv('my_panel_data.csv')

model = CupFM(n_factors=2, bandwidth=5)
results = model.fit(
    y=df['gdp'],
    X=df[['investment', 'exports']],
    panel_id=df['country'],
    time_id=df['year']
)
results.summary()
```
