"""
Monte Carlo simulation and DGP for panel cointegration.

Implements the exact DGP from Bai, Kao & Ng (2009, Tables 1-4):
    y_it = alpha_i + beta'x_it + lambda_i'F_t + u_it
    x_it = x_{i,t-1} + v_it    (I(1) regressors)
    F_t  = F_{t-1} + eta_t     (I(1) common factors)
    u_it = rho*u_{i,t-1} + e_it (AR(1) idiosyncratic error)

Author: Dr. Merwan Roudane <merwanroudane920@gmail.com>
"""

import numpy as np
import pandas as pd


def simulate_panel(
    N: int = 20,
    T: int = 40,
    k: int = 1,
    r: int = 2,
    beta: float | np.ndarray = 2.0,
    rho: float = 0.3,
    factor_ratio: float = 1.0,
    mu_lambda: float = 0.1,
    corr_uv: float = -0.4,
    burn_in: int = 200,
    seed: int = None,
) -> dict:
    """
    Simulate a panel dataset following the BKN (2009) Monte Carlo DGP.

    Parameters
    ----------
    N : int
        Number of cross-section units.
    T : int
        Number of time periods.
    k : int
        Number of regressors.
    r : int
        Number of common I(1) factors.
    beta : float or array
        True cointegrating coefficients.
    rho : float
        AR(1) coefficient for idiosyncratic error.
    factor_ratio : float
        Scale of factor component in y.
    mu_lambda : float
        Mean of factor loadings distribution.
    corr_uv : float
        Correlation between u and v innovations.
    burn_in : int
        Number of burn-in periods.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: 'y', 'X', 'panel_id', 'time_id', 'data' (DataFrame),
        'F_true', 'Lambda_true', 'beta_true', 'alpha_true'.
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(beta, (int, float)):
        beta = np.full(k, float(beta))
    beta = np.asarray(beta, dtype=np.float64)

    T_tot = T + burn_in

    # Correlation matrix W for (eta_1,...,eta_r, e, v_1,...,v_k)
    m = r + 1 + k
    W = np.eye(m)
    if m > r + 1:
        W[r, r + 1] = corr_uv
        W[r + 1, r] = corr_uv
    L_chol = np.linalg.cholesky(W)

    # Factor loadings
    Lambda_dgp = mu_lambda + np.random.randn(N, r)

    # Fixed effects
    alpha_i = np.random.uniform(0, 10, N)

    # Common I(1) factors
    F_innov = np.random.randn(T_tot, r)
    F_dgp = np.cumsum(F_innov, axis=0)
    F_dgp = F_dgp[burn_in:]

    Y_long = np.empty(N * T)
    X_long = np.empty((N * T, k))

    for i in range(N):
        Z_raw = np.random.randn(T_tot, m)
        eps_raw = Z_raw @ L_chol.T

        # AR(1) idiosyncratic error
        u_innov = eps_raw[:, r]
        u_i = np.zeros(T_tot)
        for t in range(1, T_tot):
            u_i[t] = rho * u_i[t - 1] + u_innov[t]
        u_i = u_i[burn_in:]

        # I(1) regressors
        xi_innov = eps_raw[:, r + 1:r + 1 + k]
        x_i = np.cumsum(xi_innov, axis=0)
        x_i = x_i[burn_in:]

        # Generate y
        y_i = alpha_i[i] + x_i @ beta + (
            F_dgp @ Lambda_dgp[i].reshape(-1, 1)
        ).ravel() * factor_ratio + u_i

        idx = slice(i * T, (i + 1) * T)
        Y_long[idx] = y_i
        X_long[idx] = x_i

    # Build panel identifiers
    panel_id = np.repeat(np.arange(1, N + 1), T)
    time_id = np.tile(np.arange(1, T + 1), N)

    # Build DataFrame
    data = pd.DataFrame({
        "panel_id": panel_id,
        "time_id": time_id,
        "y": Y_long,
    })
    for j in range(k):
        data[f"x{j+1}"] = X_long[:, j]

    return {
        "y": Y_long,
        "X": X_long,
        "panel_id": panel_id,
        "time_id": time_id,
        "data": data,
        "F_true": F_dgp,
        "Lambda_true": Lambda_dgp,
        "beta_true": beta,
        "alpha_true": alpha_i,
    }


def monte_carlo(
    n_reps: int = 100,
    N: int = 20,
    T: int = 40,
    k: int = 1,
    r: int = 2,
    beta: float = 2.0,
    bandwidth: int = 5,
    max_iter: int = 20,
    seed: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run a Monte Carlo experiment replicating BKN (2009) Tables.

    Parameters
    ----------
    n_reps : int
        Number of Monte Carlo replications.
    N, T, k, r : int
        Panel dimensions and factor count.
    beta : float
        True coefficient.
    bandwidth : int
        Bartlett kernel bandwidth.
    max_iter : int
        CupFM/CupBC max iterations.
    seed : int
        Base random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Columns: rep, estimator, variable, beta_hat, bias, etc.
    """
    from .estimators import run_all_estimators
    from .factors import bai_ng_ic

    results = []
    beta_arr = np.full(k, float(beta))

    for rep in range(n_reps):
        if verbose and (rep + 1) % 10 == 0:
            print(f"  MC replication {rep + 1}/{n_reps}")

        sim = simulate_panel(N=N, T=T, k=k, r=r, beta=beta, seed=seed + rep)

        try:
            raw = run_all_estimators(
                sim["y"], sim["X"], N, T, r, bandwidth,
                max_iter=max_iter, verbose=False,
            )

            est_names = ["lsdv", "baifm", "cupfm", "cupfm_bar", "cupbc"]
            est_labels = ["LSDV", "Bai FM", "CupFM", "CupFM-bar", "CupBC"]
            for key, label in zip(est_names, est_labels):
                bhat = raw[f"beta_{key}"]
                for j in range(k):
                    results.append({
                        "rep": rep + 1,
                        "estimator": label,
                        "variable": f"x{j+1}",
                        "beta_hat": bhat[j],
                        "bias": bhat[j] - beta_arr[j],
                        "true_beta": beta_arr[j],
                    })
        except Exception:
            continue

    df = pd.DataFrame(results)

    if verbose and len(df) > 0:
        print("\n  Monte Carlo Summary:")
        summary = df.groupby("estimator")["bias"].agg(
            ["mean", "std", "count"]
        )
        summary["RMSE"] = np.sqrt(
            df.groupby("estimator")["bias"].apply(lambda x: (x**2).mean())
        )
        print(summary.to_string())

    return df
