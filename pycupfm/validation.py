"""
Input validation for CupFM panel cointegration.

Validates:
    - Balanced panel structure
    - Minimum N and T dimensions
    - Missing values
    - Parameter bounds (nfactors, bandwidth, maxiter)

Author: Dr. Merwan Roudane <merwanroudane920@gmail.com>
"""

import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Custom exception for CupFM validation errors."""
    pass


def validate_panel(
    y: np.ndarray,
    X: np.ndarray,
    panel_id: np.ndarray,
    time_id: np.ndarray,
) -> dict:
    """
    Validate panel data inputs.

    Parameters
    ----------
    y : array-like of shape (N*T,)
        Dependent variable.
    X : array-like of shape (N*T, k)
        Independent variables.
    panel_id : array-like of shape (N*T,)
        Panel unit identifiers.
    time_id : array-like of shape (N*T,)
        Time period identifiers.

    Returns
    -------
    dict
        Panel dimensions: N, T, k, N_obs, units, times.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    panel_id = np.asarray(panel_id)
    time_id = np.asarray(time_id)

    n_obs = len(y)
    if X.shape[0] != n_obs:
        raise ValidationError(
            f"y has {n_obs} obs but X has {X.shape[0]} rows."
        )
    if len(panel_id) != n_obs or len(time_id) != n_obs:
        raise ValidationError(
            "panel_id and time_id must have the same length as y."
        )

    # Check for missing values
    if np.any(np.isnan(y)):
        raise ValidationError("y contains missing values (NaN).")
    if np.any(np.isnan(X)):
        raise ValidationError("X contains missing values (NaN).")

    # Panel dimensions
    units = np.unique(panel_id)
    times = np.unique(time_id)
    N = len(units)
    T_total = len(times)
    k = X.shape[1]

    if N < 2:
        raise ValidationError(
            f"CupFM requires at least 2 cross-section units (got N={N})."
        )
    if T_total < 5:
        raise ValidationError(
            f"CupFM requires at least 5 time periods (got T={T_total})."
        )

    # Check balanced panel
    counts = pd.Series(panel_id).value_counts()
    T_min, T_max = counts.min(), counts.max()
    if T_min != T_max:
        raise ValidationError(
            f"CupFM requires a balanced panel. "
            f"Min T={T_min}, Max T={T_max}."
        )

    if n_obs != N * T_total:
        raise ValidationError(
            f"Expected N*T = {N}*{T_total} = {N * T_total} obs, "
            f"got {n_obs}."
        )

    return {
        "N": N,
        "T": T_total,
        "k": k,
        "N_obs": n_obs,
        "units": units,
        "times": times,
    }


def validate_params(
    n_factors: int,
    bandwidth: int,
    max_iter: int,
    tol: float,
    N: int,
    T: int,
) -> dict:
    """
    Validate estimation parameters.

    Returns
    -------
    dict
        Validated parameters with auto settings resolved.
    """
    rmax = min(N, T) // 2

    if isinstance(n_factors, str) and n_factors == "auto":
        do_auto = True
        r_use = 1
    elif n_factors == 0:
        do_auto = True
        r_use = 1
    else:
        do_auto = False
        r_use = int(n_factors)
        if r_use < 0:
            raise ValidationError(
                "n_factors must be non-negative (0 or 'auto' for auto-select)."
            )
        if r_use > rmax:
            raise ValidationError(
                f"n_factors={r_use} exceeds max allowed = min(N,T)/2 = {rmax}."
            )

    if bandwidth < 1:
        raise ValidationError("bandwidth must be at least 1.")
    if bandwidth > T - 2:
        raise ValidationError(
            f"bandwidth={bandwidth} too large for T={T}. "
            f"Max allowed: {T - 2}."
        )

    if max_iter < 1 or max_iter > 500:
        raise ValidationError("max_iter must be between 1 and 500.")
    if tol <= 0:
        raise ValidationError("tol must be positive.")

    return {
        "do_auto": do_auto,
        "r_use": r_use,
        "rmax": rmax,
        "bandwidth": bandwidth,
        "max_iter": max_iter,
        "tol": tol,
    }
