"""
PyCupFM — Panel Cointegration with Common Factors
==================================================

Python implementation of all 5 estimators from:
    Bai, Kao & Ng (2009). Panel cointegration with global stochastic trends.
    Journal of Econometrics, 149(1), 82-99.
    Bai & Kao (2005). SSRN-1815227.

Estimators: LSDV, Bai FM, CupFM (★), CupFM-bar, CupBC

Author: Dr. Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane/cupfm

Quick Start
-----------
>>> from pycupfm import CupFM
>>> from pycupfm.datasets import load_grunfeld
>>> df = load_grunfeld()
>>> model = CupFM(n_factors=1, bandwidth=3)
>>> results = model.fit(
...     y=df['linvest'], X=df[['lmvalue', 'lkstock']],
...     panel_id=df['firm'], time_id=df['year']
... )
>>> results.summary()
"""

__version__ = "1.0.0"
__author__ = "Dr. Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"

from .core import CupFM
from .results import CupFMResults
from .simulation import simulate_panel, monte_carlo
from .datasets import load_grunfeld
from .factors import extract_factors, bai_ng_ic, rotation_weights
from .kernels import long_run_covariance, auto_bandwidth
from .plotting import (
    plot_coefficients,
    plot_factors,
    plot_loadings,
    plot_convergence,
    plot_omega_heatmap,
    plot_factor_ic,
    plot_loadings_heatmap,
    plot_all,
)
from .export import export_results

__all__ = [
    "CupFM",
    "CupFMResults",
    "simulate_panel",
    "monte_carlo",
    "load_grunfeld",
    "extract_factors",
    "bai_ng_ic",
    "rotation_weights",
    "long_run_covariance",
    "auto_bandwidth",
    "plot_coefficients",
    "plot_factors",
    "plot_loadings",
    "plot_convergence",
    "plot_omega_heatmap",
    "plot_factor_ic",
    "plot_loadings_heatmap",
    "plot_all",
    "export_results",
]
