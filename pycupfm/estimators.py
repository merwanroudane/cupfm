"""
Core estimator functions for panel cointegration with common factors.

Implements all 5 estimators from:
    Bai, J., Kao, C. & Ng, S. (2009). Panel cointegration with global
    stochastic trends. Journal of Econometrics, 149(1), 82-99.
    Bai, J. & Kao, C. (2005). On the estimation and inference of a panel
    cointegration model with cross-sectional dependence.

Estimators:
    1. LSDV     - Within/fixed-effects (biased baseline)
    2. Bai FM   - One-shot 2-step Fully Modified
    3. CupFM    - Continuously-Updated FM (recommended)
    4. CupFM-bar- CupFM with Z-bar instrument
    5. CupBC    - Continuously-Updated Bias-Corrected

Author: Dr. Merwan Roudane <merwanroudane920@gmail.com>
"""

import numpy as np
from numpy.linalg import inv
from .factors import extract_factors, rotation_weights
from .kernels import long_run_covariance


# ── Utility functions ─────────────────────────────────────────────────────

def _long2wide(V_long: np.ndarray, N: int, T: int, m: int) -> np.ndarray:
    """Reshape long (N*T, m) -> wide (T, N*m)."""
    V_wide = np.empty((T, N * m))
    for i in range(N):
        idx = slice(i * T, (i + 1) * T)
        V_wide[:, i * m:(i + 1) * m] = V_long[idx]
    return V_wide


def _wide2long(V_wide: np.ndarray, N: int, T: int, m: int) -> np.ndarray:
    """Reshape wide (T, N*m) -> long (N*T, m)."""
    V_long = np.empty((N * T, m))
    for i in range(N):
        idx = slice(i * T, (i + 1) * T)
        V_long[idx] = V_wide[:, i * m:(i + 1) * m]
    return V_long


def _fdif(X: np.ndarray) -> np.ndarray:
    """First differences: ΔX_t = X_t - X_{t-1}."""
    return X[1:] - X[:-1]


def _demean(X_long: np.ndarray, N: int, T: int) -> np.ndarray:
    """Within-demean each cross-section unit."""
    Xdm = X_long.copy()
    for i in range(N):
        idx = slice(i * T, (i + 1) * T)
        Xdm[idx] -= Xdm[idx].mean(axis=0)
    return Xdm


def _safe_inv(A: np.ndarray) -> np.ndarray:
    """Invert A, falling back to pseudo-inverse if singular."""
    try:
        return inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


# ── 1. LSDV Estimator ────────────────────────────────────────────────────

def lsdv_estimate(
    y_long: np.ndarray, X_long: np.ndarray, N: int, T: int
) -> dict:
    """
    LSDV (within/fixed-effects) estimator.

    Biased under cross-sectional dependence from I(1) common factors.
    Included as baseline for comparison.

    Parameters
    ----------
    y_long : ndarray (N*T, 1)
    X_long : ndarray (N*T, k)
    N, T : int

    Returns
    -------
    dict with keys: beta, tstat, se, sigma2
    """
    k = X_long.shape[1]
    NT = N * T

    Xdm = _demean(X_long, N, T)
    ydm = _demean(y_long, N, T)

    XX = Xdm.T @ Xdm
    XX_inv = _safe_inv(XX)
    beta = XX_inv @ (Xdm.T @ ydm)

    uhat = ydm - Xdm @ beta
    sigma2 = float(uhat.T @ uhat) / max(NT - N - k, 1)
    sigma2 = max(sigma2, 1e-12)

    var_beta = np.diag(sigma2 * XX_inv)
    se = np.sqrt(np.maximum(var_beta, 0))
    tstat = beta.ravel() / np.where(se > 1e-12, se, 1e-12)

    return {
        "beta": beta.ravel(),
        "tstat": tstat,
        "se": se,
        "sigma2": sigma2,
    }


# ── 2. CUP Plain LS (CupBC inner loop) ──────────────────────────────────

def _cup_pls_beta(
    X_long: np.ndarray, y_long: np.ndarray,
    F: np.ndarray, N: int, T: int, k: int,
) -> np.ndarray:
    """Plain Cup-LS beta with factor projection (GAUSS Mul_panelbeta)."""
    invFtF = _safe_inv(F.T @ F)
    XX = np.zeros((k, k))
    Xy = np.zeros((k, 1))
    for i in range(N):
        idx = slice(i * T, (i + 1) * T)
        xi = X_long[idx]
        yi = y_long[idx]
        FtX = F.T @ xi
        FtY = F.T @ yi
        XX += xi.T @ xi - FtX.T @ invFtF @ FtX
        Xy += xi.T @ yi - FtX.T @ invFtF @ FtY
    return _safe_inv(XX) @ Xy


# ── 3. Full FM Beta ──────────────────────────────────────────────────────

def _fm_beta(
    y_wide_T1, X_long_T1, X_wide_T1, aik,
    F_T1, dF, u1_wide_T1,
    du2_T1_long, bw, N, T1, k, r, kernel,
):
    """
    Full FM beta computation (Bai & Kao 2005 / BKN 2009).

    Returns dict with beta_fm1, beta_fm2, tstat_fm1, tstat_fm2,
    Omega_bar, cond_var.
    """
    m = 1 + k + r

    # A. ΔN̄x_i
    du2_T1_wide = _long2wide(du2_T1_long, N, T1, k)
    du2N_wide = np.zeros((T1, N * k))
    for ip in range(N):
        sumd = np.zeros((T1, k))
        for kp in range(N):
            sumd += du2_T1_wide[:, kp * k:(kp + 1) * k] * aik[ip, kp]
        du2N_wide[:, ip * k:(ip + 1) * k] = (
            du2_T1_wide[:, ip * k:(ip + 1) * k] - sumd / N
        )

    # A.2 x̄_i (aik-weighted demean of x)
    xbar_wide = np.zeros((T1, N * k))
    for ip in range(N):
        sumx = np.zeros((T1, k))
        for kp in range(N):
            sumx += X_wide_T1[:, kp * k:(kp + 1) * k] * aik[ip, kp]
        xbar_wide[:, ip * k:(ip + 1) * k] = (
            X_wide_T1[:, ip * k:(ip + 1) * k] - sumx / N
        )

    # B. Kernel long-run covariance
    Omega_sum = np.zeros((m, m))
    Dplus_sum = np.zeros((m, m))
    for ku in range(N):
        u1k = u1_wide_T1[:, ku:ku + 1]
        du2Nk = du2N_wide[:, ku * k:(ku + 1) * k]
        W_k = np.hstack([u1k, du2Nk, dF])
        Om_k, Dp_k = long_run_covariance(W_k, bw, kernel)
        Omega_sum += Om_k
        Dplus_sum += Dp_k
    Omega_bar = Omega_sum / N
    Dplus_bar = Dplus_sum / N

    # C. Partition Omega
    Omega_ub = Omega_bar[0:1, 1:m]
    Omega_b = Omega_bar[1:m, 1:m]
    invOmb = _safe_inv(Omega_b)
    gab = Omega_ub @ invOmb

    # D. FM y-transformation
    y_plus = np.empty((T1, N))
    for ip in range(N):
        dN_ip = du2N_wide[:, ip * k:(ip + 1) * k]
        for ti in range(T1):
            temp5 = np.concatenate([dN_ip[ti:ti + 1], dF[ti:ti + 1]], axis=1).T
            y_plus[ti, ip] = y_wide_T1[ti, ip] - float(gab @ temp5)

    # E. Serial-correlation correction
    abu = Dplus_bar[1:m, 0:1]
    ab_ = Dplus_bar[1:m, 1:m]
    obu = Omega_bar[1:m, 0:1]
    obup = abu - ab_ @ invOmb @ obu

    # F. δ̄ correction
    invFF = _safe_inv(F_T1.T @ F_T1)
    db_all = np.zeros((N, r * k))
    for ip in range(N):
        x_ip = X_wide_T1[:, ip * k:(ip + 1) * k]
        db_i = invFF @ (F_T1.T @ x_ip)
        db_all[ip, :] = db_i.ravel()
    dbsum_all = np.zeros((N, r * k))
    for ip in range(N):
        for kp in range(N):
            dbsum_all[ip, :] += db_all[kp, :] * aik[ip, kp]
    dbsum1_all = db_all - dbsum_all / N
    db_bar = dbsum1_all.sum(axis=0).reshape(r, k)

    db2 = np.zeros((r, N * k))
    for ip in range(N):
        db2[:, ip * k:(ip + 1) * k] = dbsum1_all[ip, :].reshape(r, k)

    # G. Correction term
    obup_x = obup[:k]
    obup_f = obup[k:k + r]
    temp3 = T1 * (N * obup_x - db_bar.T @ obup_f)

    # H. beta_fm1 - main CupFM estimate
    invFF1 = _safe_inv(F_T1.T @ F_T1)
    XX = np.zeros((k, k))
    Xy = np.zeros((k, 1))
    for ip in range(N):
        xi = X_long_T1[ip * T1:(ip + 1) * T1]
        yi = y_plus[:, ip:ip + 1]
        FtX = F_T1.T @ xi
        FtY = F_T1.T @ yi
        XX += xi.T @ xi - FtX.T @ invFF1 @ FtX
        Xy += xi.T @ yi - FtX.T @ invFF1 @ FtY
    Xy -= temp3
    invXX = _safe_inv(XX)
    beta_fm1 = invXX @ Xy

    Omega_uu = Omega_bar[0, 0]
    cond_var = float(Omega_uu - Omega_ub @ invOmb @ Omega_ub.T)
    if cond_var <= 0:
        cond_var = 1e-12

    diag_invXX = np.diag(invXX)
    tstat_fm1 = beta_fm1.ravel() / np.sqrt(
        np.maximum(diag_invXX * cond_var, 1e-20)
    )

    # I. beta_fm2 - CupFM-bar (Z-bar variant)
    z1 = xbar_wide - F_T1 @ db2
    z_long = _wide2long(z1, N, T1, k)

    XXz = np.zeros((k, k))
    Xyz = np.zeros((k, 1))
    for ip in range(N):
        zi = z_long[ip * T1:(ip + 1) * T1]
        yi = y_plus[:, ip:ip + 1]
        XXz += zi.T @ zi
        Xyz += zi.T @ yi
    Xyz -= temp3
    invXXz = _safe_inv(XXz)
    beta_fm2 = invXXz @ Xyz
    tstat_fm2 = beta_fm2.ravel() / np.sqrt(
        np.maximum(np.diag(invXXz) * cond_var, 1e-20)
    )

    return {
        "beta_fm1": beta_fm1.ravel(),
        "beta_fm2": beta_fm2.ravel(),
        "tstat_fm1": tstat_fm1,
        "tstat_fm2": tstat_fm2,
        "Omega_bar": Omega_bar,
        "cond_var": cond_var,
    }


# ── 4. Main Engine ───────────────────────────────────────────────────────

def run_all_estimators(
    y_long: np.ndarray,
    X_long: np.ndarray,
    N: int,
    T: int,
    r: int,
    bandwidth: int,
    kernel: str = "bartlett",
    max_iter: int = 20,
    verbose: bool = False,
) -> dict:
    """
    Run all 5 panel cointegration estimators.

    This is the main engine, translated from the Stata/Mata cupfm_main().

    Parameters
    ----------
    y_long : ndarray (N*T, 1)
    X_long : ndarray (N*T, k)
    N, T : int
    r : int
        Number of common factors.
    bandwidth : int
    kernel : str
    max_iter : int
    verbose : bool

    Returns
    -------
    dict
        All estimator results.
    """
    y_long = y_long.reshape(-1, 1) if y_long.ndim == 1 else y_long
    k = X_long.shape[1]

    # ── 1. LSDV ──
    lsdv = lsdv_estimate(y_long, X_long, N, T)

    # ── 2. Initial PCA ──
    uhat_long = y_long - X_long @ lsdv["beta"].reshape(-1, 1)
    U_wide = _long2wide(uhat_long, N, T, 1)

    F_hat, L_hat = extract_factors(U_wide, r)

    # ── 3. First differences Δx ──
    T1 = T - 1
    du2_T1_long = np.empty((N * T1, k))
    for i in range(N):
        idx_i = slice(i * T, (i + 1) * T)
        du2_T1_long[i * T1:(i + 1) * T1] = _fdif(X_long[idx_i])

    dF = _fdif(F_hat)

    # ── 4. Idiosyncratic residuals u1 ──
    FL_wide = np.zeros((T, N))
    for i in range(N):
        FL_wide[:, i] = (F_hat @ L_hat[i:i + 1].T).ravel()
    FL_long = _wide2long(FL_wide, N, T, 1)
    u1_wide = _long2wide(
        y_long - X_long @ lsdv["beta"].reshape(-1, 1) - FL_long, N, T, 1
    )

    aik = rotation_weights(L_hat)

    # ── 5. T-1 versions for FM ──
    F_T1 = F_hat[:T1]
    u1_wide_T1 = u1_wide[:T1]
    y_wide = _long2wide(y_long, N, T, 1)
    y_wide_T1 = y_wide[:T1]

    X_long_T1 = np.empty((N * T1, k))
    for i in range(N):
        X_long_T1[i * T1:(i + 1) * T1] = X_long[i * T:(i + 1) * T][:T1]
    X_wide_T1 = _long2wide(X_long_T1, N, T1, k)

    # ── 6. One-shot Bai FM ──
    bfm = _fm_beta(
        y_wide_T1, X_long_T1, X_wide_T1, aik,
        F_T1, dF, u1_wide_T1,
        du2_T1_long, bandwidth, N, T1, k, r, kernel,
    )

    # ── 7. CupFM iteration ──
    beta_cur = bfm["beta_fm1"].reshape(-1, 1)
    convergence_path = [beta_cur.ravel().copy()]

    for itr in range(1, max_iter + 1):
        # Factor step
        uhat_long = y_long - X_long @ beta_cur
        U_wide = _long2wide(uhat_long, N, T, 1)
        F_hat, L_hat = extract_factors(U_wide, r)
        aik = rotation_weights(L_hat)

        F_T1 = F_hat[:T1]
        dF = _fdif(F_hat)

        FL_wide = np.zeros((T, N))
        for i in range(N):
            FL_wide[:, i] = (F_hat @ L_hat[i:i + 1].T).ravel()
        FL_long = _wide2long(FL_wide, N, T, 1)
        u1_wide = _long2wide(y_long - X_long @ beta_cur - FL_long, N, T, 1)
        u1_wide_T1 = u1_wide[:T1]

        # FM step
        cup_res = _fm_beta(
            y_wide_T1, X_long_T1, X_wide_T1, aik,
            F_T1, dF, u1_wide_T1,
            du2_T1_long, bandwidth, N, T1, k, r, kernel,
        )
        beta_cur = cup_res["beta_fm1"].reshape(-1, 1)
        convergence_path.append(beta_cur.ravel().copy())

        if verbose:
            print(f"CupFM iter {itr}: beta = {beta_cur.ravel()}")

    cupfm_result = cup_res
    niter_cupfm = max_iter

    # ── 8. CupBC ──
    uhat_long = y_long - X_long @ lsdv["beta"].reshape(-1, 1)
    U_wide = _long2wide(uhat_long, N, T, 1)
    F_bc, L_bc = extract_factors(U_wide, r)

    for itr_bc in range(1, max_iter + 1):
        beta_bc = _cup_pls_beta(X_long, y_long, F_bc, N, T, k)
        uhat_long = y_long - X_long @ beta_bc
        U_wide = _long2wide(uhat_long, N, T, 1)
        F_bc, L_bc = extract_factors(U_wide, r)
        if verbose:
            print(f"CupBC iter {itr_bc}: beta = {beta_bc.ravel()}")

    aik_bc = rotation_weights(L_bc)
    dF_bc = _fdif(F_bc)
    F_T1_bc = F_bc[:T1]

    FL_wide_bc = np.zeros((T, N))
    for i in range(N):
        FL_wide_bc[:, i] = (F_bc @ L_bc[i:i + 1].T).ravel()
    FL_long_bc = _wide2long(FL_wide_bc, N, T, 1)
    u1_wide_bc = _long2wide(
        y_long - X_long @ beta_bc - FL_long_bc, N, T, 1
    )
    u1_wide_T1_bc = u1_wide_bc[:T1]
    X_wide_T1_bc = _long2wide(X_long_T1, N, T1, k)

    cupbc_fm = _fm_beta(
        y_wide_T1, X_long_T1, X_wide_T1_bc, aik_bc,
        F_T1_bc, dF_bc, u1_wide_T1_bc,
        du2_T1_long, bandwidth, N, T1, k, r, kernel,
    )

    # ── Assemble results ──
    return {
        # LSDV
        "beta_lsdv": lsdv["beta"],
        "tstat_lsdv": lsdv["tstat"],
        "se_lsdv": lsdv["se"],
        # Bai FM
        "beta_baifm": bfm["beta_fm1"],
        "tstat_baifm": bfm["tstat_fm1"],
        # CupFM
        "beta_cupfm": cupfm_result["beta_fm1"],
        "tstat_cupfm": cupfm_result["tstat_fm1"],
        # CupFM-bar
        "beta_cupfm_bar": cupfm_result["beta_fm2"],
        "tstat_cupfm_bar": cupfm_result["tstat_fm2"],
        # CupBC
        "beta_cupbc": cupbc_fm["beta_fm1"],
        "tstat_cupbc": cupbc_fm["tstat_fm1"],
        # Factor structure
        "F_hat": F_hat,
        "Lambda": L_hat,
        "Aik": aik,
        "Omega_cupfm": cupfm_result["Omega_bar"],
        "Omega_cupbc": cupbc_fm["Omega_bar"],
        "cond_var_cupfm": cupfm_result["cond_var"],
        "cond_var_cupbc": cupbc_fm["cond_var"],
        # Convergence
        "n_iter": niter_cupfm,
        "convergence_path": np.array(convergence_path),
        # Dimensions
        "N": N,
        "T": T,
        "r": r,
        "bandwidth": bandwidth,
        "kernel": kernel,
    }
