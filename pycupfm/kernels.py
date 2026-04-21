"""
Long-run covariance kernel estimators for panel cointegration.

Implements Bartlett, Parzen, and Quadratic Spectral kernels
with automatic bandwidth selection (Andrews 1991, Newey-West 1994).

Author: Dr. Merwan Roudane <merwanroudane920@gmail.com>
Reference:
    Bai, Kao & Ng (2009). Panel cointegration with global stochastic trends.
    Journal of Econometrics, 149(1), 82-99.
"""

import numpy as np
from numpy.linalg import inv


def bartlett_weight(j: int, bandwidth: int) -> float:
    """Bartlett kernel weight: w_j = 1 - |j|/(M+1)."""
    return 1.0 - abs(j) / (bandwidth + 1)


def parzen_weight(j: int, bandwidth: int) -> float:
    """Parzen kernel weight."""
    z = abs(j) / (bandwidth + 1)
    if z <= 0.5:
        return 1.0 - 6.0 * z**2 + 6.0 * z**3
    else:
        return 2.0 * (1.0 - z) ** 3


def qs_weight(j: int, bandwidth: int) -> float:
    """Quadratic Spectral (Andrews 1991) kernel weight."""
    if j == 0:
        return 1.0
    z = 6.0 * np.pi * abs(j) / (5.0 * (bandwidth + 1))
    return (25.0 / (12.0 * np.pi**2 * (abs(j) / (bandwidth + 1))**2)) * (
        np.sin(z) / z - np.cos(z)
    )


KERNEL_FUNCTIONS = {
    "bartlett": bartlett_weight,
    "parzen": parzen_weight,
    "quadratic_spectral": qs_weight,
    "qs": qs_weight,
}


def auto_bandwidth(residuals: np.ndarray, method: str = "nw") -> int:
    """
    Automatic bandwidth selection.

    Parameters
    ----------
    residuals : ndarray of shape (T, m)
        Residual matrix.
    method : str
        'nw' for Newey-West (1994) plug-in rule, 'andrews' for Andrews (1991).

    Returns
    -------
    int
        Selected bandwidth.
    """
    T = residuals.shape[0]
    if method == "nw":
        # Newey-West rule: M = floor(4*(T/100)^{2/9})
        bw = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    elif method == "andrews":
        # Andrews (1991) AR(1) plug-in for Bartlett
        if residuals.ndim == 1:
            u = residuals
        else:
            u = residuals[:, 0]
        rho_hat = np.corrcoef(u[1:], u[:-1])[0, 1] if len(u) > 2 else 0.0
        rho_hat = min(max(rho_hat, -0.99), 0.99)
        alpha_hat = (4.0 * rho_hat**2) / ((1.0 - rho_hat)**2 * (1.0 + rho_hat)**2)
        bw = int(np.ceil(1.1447 * (alpha_hat * T) ** (1.0 / 3.0)))
    else:
        bw = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))

    return max(1, min(bw, T - 2))


def long_run_covariance(
    U: np.ndarray,
    bandwidth: int,
    kernel: str = "bartlett",
) -> tuple:
    """
    Kernel-based long-run covariance estimation.

    Estimates:
        Omega      = Sigma + Gamma + Gamma'  (two-sided)
        Delta_plus = Sigma + Gamma           (one-sided)

    Parameters
    ----------
    U : ndarray of shape (T, m)
        Residual/innovation matrix.
    bandwidth : int
        Lag truncation parameter M.
    kernel : str
        Kernel function name: 'bartlett', 'parzen', or 'qs'/'quadratic_spectral'.

    Returns
    -------
    Omega : ndarray of shape (m, m)
        Two-sided long-run covariance.
    Delta_plus : ndarray of shape (m, m)
        One-sided long-run covariance.
    """
    T, m = U.shape
    kernel_fn = KERNEL_FUNCTIONS.get(kernel, bartlett_weight)

    # Sample covariance at lag 0
    Sigma = U.T @ U / T

    # Weighted autocovariances
    Gamma = np.zeros((m, m))
    for j in range(1, bandwidth + 1):
        if T - j < 1:
            break
        wj = kernel_fn(j, bandwidth)
        Gamma_j = U[j:].T @ U[:T - j] / T
        Gamma += wj * Gamma_j

    Delta_plus = Sigma + Gamma
    Omega = Sigma + Gamma + Gamma.T

    return Omega, Delta_plus
