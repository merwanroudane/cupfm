"""
CupFM — Main estimator class for panel cointegration with common factors.

Provides a scikit-learn-style API:
    model = CupFM(n_factors=2, bandwidth=5)
    results = model.fit(y, X, panel_id, time_id)
    results.summary()

Author: Dr. Merwan Roudane <merwanroudane920@gmail.com>

References:
    Bai, J., Kao, C. & Ng, S. (2009). Panel cointegration with global
    stochastic trends. Journal of Econometrics, 149(1), 82-99.
    Bai, J. & Kao, C. (2005). On the estimation and inference of a panel
    cointegration model with cross-sectional dependence.
    CPR Working Paper No. 75, Syracuse University.
"""

import numpy as np
import pandas as pd

from .validation import validate_panel, validate_params
from .factors import bai_ng_ic, extract_factors
from .estimators import run_all_estimators, _long2wide
from .kernels import auto_bandwidth
from .results import CupFMResults
from .plotting import (
    plot_coefficients, plot_factors, plot_loadings,
    plot_convergence, plot_omega_heatmap, plot_factor_ic,
    plot_loadings_heatmap, plot_all,
)
from .export import export_results


class CupFM:
    """
    Panel Cointegration with Common Factors.

    Implements all five estimators from Bai, Kao & Ng (2009) and
    Bai & Kao (2005):

    - **LSDV**: Within/fixed-effects (biased baseline)
    - **Bai FM**: One-shot 2-step Fully Modified
    - **CupFM**: Continuously-Updated FM ★ recommended
    - **CupFM-bar**: CupFM with Z-bar instrument
    - **CupBC**: Continuously-Updated Bias-Corrected

    Parameters
    ----------
    n_factors : int or str
        Number of common factors r. Use 0 or 'auto' for automatic
        selection via Bai & Ng (2002) information criterion.
    kernel : str
        Kernel for long-run covariance: 'bartlett' (default),
        'parzen', or 'quadratic_spectral'/'qs'.
    bandwidth : int or str
        Bartlett kernel bandwidth M. Use 'auto' for Newey-West plug-in.
        Default: 5.
    max_iter : int
        Maximum CupFM/CupBC iterations. Default: 20.
    tol : float
        Convergence tolerance. Default: 1e-4.
    auto_rmax : int
        Maximum r in automatic selection. Default: 8.
    verbose : bool
        Print iteration progress. Default: False.

    Examples
    --------
    >>> from pycupfm import CupFM
    >>> from pycupfm.datasets import load_grunfeld
    >>> df = load_grunfeld()
    >>> model = CupFM(n_factors=1, bandwidth=3)
    >>> results = model.fit(
    ...     y=df['linvest'], X=df[['lmvalue', 'lkstock']],
    ...     panel_id=df['firm'], time_id=df['year'],
    ...     var_names=['lmvalue', 'lkstock'], dep_var='linvest'
    ... )
    >>> results.summary()
    """

    def __init__(
        self,
        n_factors: int | str = "auto",
        kernel: str = "bartlett",
        bandwidth: int | str = 5,
        max_iter: int = 20,
        tol: float = 1e-4,
        auto_rmax: int = 8,
        verbose: bool = False,
    ):
        self.n_factors = n_factors
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.auto_rmax = auto_rmax
        self.verbose = verbose
        self.results_ = None

    def fit(
        self,
        y,
        X,
        panel_id,
        time_id,
        var_names: list = None,
        dep_var: str = "y",
    ) -> CupFMResults:
        """
        Fit all 5 panel cointegration estimators.

        Parameters
        ----------
        y : array-like of shape (N*T,)
            Dependent variable (I(1)).
        X : array-like of shape (N*T, k)
            Independent variables (I(1) regressors).
        panel_id : array-like of shape (N*T,)
            Cross-section unit identifiers.
        time_id : array-like of shape (N*T,)
            Time period identifiers.
        var_names : list of str, optional
            Names for regressors. Defaults to x1, x2, ...
        dep_var : str
            Name of dependent variable.

        Returns
        -------
        CupFMResults
            Results object with .summary(), .plot(), etc.
        """
        # Convert inputs
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        panel_arr = np.asarray(panel_id)
        time_arr = np.asarray(time_id)

        # Validate panel
        dims = validate_panel(y_arr, X_arr, panel_arr, time_arr)
        N, T, k = dims["N"], dims["T"], dims["k"]

        # Resolve bandwidth
        if isinstance(self.bandwidth, str) and self.bandwidth == "auto":
            bw = auto_bandwidth(y_arr[:T].reshape(-1, 1))
        else:
            bw = int(self.bandwidth)

        # Resolve n_factors
        nf = self.n_factors
        if isinstance(nf, str) and nf == "auto":
            nf = 0

        # Validate params
        params = validate_params(nf, bw, self.max_iter, self.tol, N, T)

        # Sort data by panel_id then time_id
        sort_idx = np.lexsort((time_arr, panel_arr))
        y_sorted = y_arr[sort_idx].reshape(-1, 1)
        X_sorted = X_arr[sort_idx]

        # Auto-select r if needed
        r = params["r_use"]
        ic_values = None
        if params["do_auto"]:
            from .estimators import _demean, _long2wide as l2w
            Xdm = _demean(X_sorted, N, T)
            ydm = _demean(y_sorted, N, T)
            XX_inv = np.linalg.pinv(Xdm.T @ Xdm)
            beta_init = XX_inv @ (Xdm.T @ ydm)
            uhat = y_sorted - X_sorted @ beta_init
            U_wide = l2w(uhat, N, T, 1)
            rmax_use = min(self.auto_rmax, min(N, T) // 2)
            rmax_use = max(rmax_use, 1)
            r, ic_values = bai_ng_ic(U_wide, rmax_use)
            if self.verbose:
                print(f"  Auto-selected r = {r} (Bai-Ng 2002 IC)")

        # Run all estimators
        if self.verbose:
            print(f"  Running CupFM: N={N}, T={T}, k={k}, r={r}, bw={bw}")

        raw = run_all_estimators(
            y_sorted, X_sorted, N, T, r, bw,
            kernel=self.kernel, max_iter=self.max_iter,
            verbose=self.verbose,
        )

        # Infer var_names from DataFrame columns if available
        if var_names is None:
            if isinstance(X, pd.DataFrame):
                var_names = list(X.columns)
            else:
                var_names = [f"x{j+1}" for j in range(k)]

        if isinstance(y, pd.Series) and dep_var == "y":
            dep_var = y.name or "y"

        self.results_ = CupFMResults(raw, var_names=var_names, dep_var=dep_var)
        self.ic_values_ = ic_values
        return self.results_

    def summary(self):
        """Print summary of last fit."""
        if self.results_ is None:
            raise RuntimeError("Call .fit() before .summary().")
        return self.results_.summary()

    def plot(self, kind="all", save_prefix=None):
        """
        Generate plots from last fit.

        Parameters
        ----------
        kind : str
            'all', 'coefficients', 'factors', 'loadings',
            'convergence', 'omega', 'ic'.
        save_prefix : str, optional
            Base filename for saving plots.
        """
        if self.results_ is None:
            raise RuntimeError("Call .fit() before .plot().")

        r = self.results_
        sp = lambda name: f"{save_prefix}_{name}.png" if save_prefix else None

        if kind == "all":
            return plot_all(r, save_prefix=save_prefix)
        elif kind == "coefficients":
            return plot_coefficients(r, save_path=sp("coef"))
        elif kind == "factors":
            return plot_factors(r, save_path=sp("factors"))
        elif kind == "loadings":
            return plot_loadings(r, save_path=sp("loadings"))
        elif kind == "convergence":
            return plot_convergence(r, save_path=sp("convergence"))
        elif kind == "omega":
            return plot_omega_heatmap(r, save_path=sp("omega"))
        elif kind == "ic":
            if self.ic_values_ is not None:
                return plot_factor_ic(self.ic_values_, save_path=sp("ic"))
            else:
                print("No IC values (r was specified, not auto-selected).")
        else:
            raise ValueError(f"Unknown plot kind: {kind}")

    def __repr__(self):
        return (
            f"CupFM(n_factors={self.n_factors}, kernel='{self.kernel}', "
            f"bandwidth={self.bandwidth}, max_iter={self.max_iter})"
        )
