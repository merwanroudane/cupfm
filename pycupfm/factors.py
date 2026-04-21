"""
Factor extraction and selection for panel cointegration.

Implements:
    - SVD-based PCA factor extraction (T>=N and T<N branches)
    - Bai & Ng (2002) information criterion for automatic r selection
    - Rotation weight matrix (BKN 2009, Section 2)

Author: Dr. Merwan Roudane <merwanroudane920@gmail.com>
Reference:
    Bai, J. & Ng, S. (2002). Determining the number of factors in
    approximate factor models. Econometrica, 70(1), 191-221.
"""

import numpy as np
from numpy.linalg import svd


def extract_factors(U_wide: np.ndarray, r: int) -> tuple:
    """
    SVD-based PCA factor extraction.

    Normalization: F'F/T -> I_r, Lambda'Lambda/N -> Sigma_lambda

    Parameters
    ----------
    U_wide : ndarray of shape (T, N)
        Wide-format residual matrix.
    r : int
        Number of factors to extract.

    Returns
    -------
    F_hat : ndarray of shape (T, r)
        Estimated common factors.
    L_hat : ndarray of shape (N, r)
        Estimated factor loadings.
    """
    T, N = U_wide.shape
    r = min(r, min(T, N))

    if T >= N:
        P, D, Qh = svd(U_wide, full_matrices=False)
        F_hat = P[:, :r] * T
        L_hat = U_wide.T @ F_hat / (T ** 2)
    else:
        # For short panels: eigendecompose U'U
        UU = U_wide.T @ U_wide
        P, D, Qh = svd(UU, full_matrices=False)
        L_hat = P[:, :r] * np.sqrt(N)
        F_hat = U_wide @ L_hat / N

    return F_hat, L_hat


def rotation_weights(Lambda: np.ndarray) -> np.ndarray:
    """
    Compute rotation weight matrix a_ik (BKN 2009, Section 2).

    a_ik = Lambda * (Lambda'Lambda / N)^{-1} * Lambda'

    Parameters
    ----------
    Lambda : ndarray of shape (N, r)
        Factor loading matrix.

    Returns
    -------
    aik : ndarray of shape (N, N)
        Rotation weight matrix.
    """
    N = Lambda.shape[0]
    SigL = Lambda.T @ Lambda / N
    try:
        SigL_inv = np.linalg.inv(SigL)
    except np.linalg.LinAlgError:
        SigL_inv = np.linalg.pinv(SigL)
    return Lambda @ SigL_inv @ Lambda.T


def bai_ng_ic(U_wide: np.ndarray, rmax: int) -> tuple:
    """
    Bai-Ng (2002) information criterion for number of factors.

    IC_1(k) = log(V(k,F)) + k * (N+T)/(NT) * log(NT/(N+T))

    Parameters
    ----------
    U_wide : ndarray of shape (T, N)
        Wide-format residual matrix.
    rmax : int
        Maximum number of factors to evaluate.

    Returns
    -------
    r_opt : int
        Optimal number of factors.
    ic_values : ndarray
        IC values for r = 1, ..., rmax.
    """
    T, N = U_wide.shape
    NT = N * T
    rmax = min(rmax, min(N, T) // 2)
    rmax = max(rmax, 1)

    ic_values = np.full(rmax, np.inf)
    best_r = 1
    best_ic = np.inf

    for ri in range(1, rmax + 1):
        F_r, L_r = extract_factors(U_wide, ri)
        resid = U_wide - F_r @ L_r.T
        V_r = np.sum(resid ** 2) / NT
        if V_r <= 0:
            continue
        penalty = ri * (N + T) / NT * np.log(NT / (N + T))
        ic_r = np.log(V_r) + penalty
        ic_values[ri - 1] = ic_r
        if ic_r < best_ic:
            best_ic = ic_r
            best_r = ri

    return best_r, ic_values
