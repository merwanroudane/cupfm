"""
Publication-quality visualizations for CupFM panel cointegration.

All 9 plot types with premium academic aesthetics:
    1. Coefficient forest plot (all 5 estimators + CIs)
    2. Factor time series with ribbons
    3. Factor loadings heatmap
    4. Loading scatter (λ₁ vs λ₂)
    5. Convergence path
    6. Residual diagnostics
    7. Omega heatmap
    8. Bandwidth sensitivity
    9. Factor IC plot

Author: Dr. Merwan Roudane <merwanroudane920@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import warnings

# ── Premium color palette ─────────────────────────────────────────────────
COLORS = {
    "LSDV": "#2C3E50",       # Dark navy
    "Bai FM": "#E67E22",     # Warm orange
    "CupFM": "#27AE60",      # Emerald green
    "CupFM-bar": "#8E44AD",  # Purple
    "CupBC": "#C0392B",      # Crimson
}
PALETTE = list(COLORS.values())
BG_COLOR = "#FAFBFC"
GRID_COLOR = "#E8ECF0"
TEXT_COLOR = "#2C3E50"
ACCENT = "#3498DB"


def _setup_style():
    """Apply premium academic plot style."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#D0D5DD",
        "axes.labelcolor": TEXT_COLOR,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": GRID_COLOR,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#D0D5DD",
    })


def plot_coefficients(results, figsize=(10, 5), save_path=None):
    """
    Coefficient comparison forest plot.

    Shows all 5 estimators with 95% CIs for each regressor.

    Parameters
    ----------
    results : CupFMResults
        Estimation results.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()
    n_vars = results.k
    est_names = results.ESTIMATOR_NAMES

    fig, axes = plt.subplots(1, n_vars, figsize=(figsize[0] * n_vars / 2, figsize[1]),
                             squeeze=False)

    for j in range(n_vars):
        ax = axes[0, j]
        positions = np.arange(len(est_names))

        for idx, name in enumerate(est_names):
            b = results.betas[name][j]
            ci = results.ci[name][j]
            color = COLORS[name]

            # CI bar
            ax.plot([ci[0], ci[1]], [idx, idx], color=color,
                    linewidth=2, alpha=0.7, solid_capstyle="round")
            # CI caps
            cap_size = 0.15
            ax.plot([ci[0], ci[0]], [idx - cap_size, idx + cap_size],
                    color=color, linewidth=2, alpha=0.7)
            ax.plot([ci[1], ci[1]], [idx - cap_size, idx + cap_size],
                    color=color, linewidth=2, alpha=0.7)
            # Point estimate
            ax.scatter(b, idx, color=color, s=120, zorder=5,
                       edgecolors="white", linewidth=1.5,
                       marker="D")

        ax.axvline(x=0, color="#E74C3C", linestyle="--",
                   linewidth=1, alpha=0.6)
        ax.set_yticks(positions)
        ax.set_yticklabels(est_names, fontsize=10)
        ax.set_xlabel("Coefficient Estimate", fontsize=11)
        ax.set_title(f"{results.var_names[j]}", fontsize=13, fontweight="bold")
        ax.invert_yaxis()

    fig.suptitle("Coefficient Comparison — Panel Cointegration Estimators",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, "95% Confidence Intervals | Bai, Kao & Ng (2009)",
             ha="center", fontsize=9, color="#7F8C8D")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=BG_COLOR)
    return fig


def plot_factors(results, figsize=(12, 5), save_path=None):
    """
    Time series plot of estimated common factors.

    Parameters
    ----------
    results : CupFMResults
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()
    F = results.F_hat
    T, r = F.shape

    fig, ax = plt.subplots(figsize=figsize)
    factor_colors = ["#2980B9", "#E74C3C", "#27AE60", "#F39C12", "#8E44AD"]
    line_styles = ["-", "--", "-.", ":", "-"]

    for ri in range(r):
        ax.plot(np.arange(1, T + 1), F[:, ri],
                color=factor_colors[ri % len(factor_colors)],
                linewidth=2.2,
                linestyle=line_styles[ri % len(line_styles)],
                label=f"Factor {ri + 1}",
                alpha=0.9)

    ax.axhline(y=0, color="#BDC3C7", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel("Factor Score", fontsize=12)
    ax.set_title(f"Estimated Common Factors (r = {r})",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    fig.text(0.5, -0.02,
             f"Panel Cointegration CupFM | N = {results.N}, T = {results.T} "
             f"| Bai, Kao & Ng (2009)",
             ha="center", fontsize=9, color="#7F8C8D")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=BG_COLOR)
    return fig


def plot_loadings(results, figsize=(10, 6), save_path=None):
    """
    Factor loadings visualization.

    Bar chart if r=1, scatter plot if r>=2, heatmap if r>=3.

    Parameters
    ----------
    results : CupFMResults
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()
    L = results.Lambda
    N, r = L.shape

    fig, ax = plt.subplots(figsize=figsize)

    if r == 1:
        colors = [ACCENT if v >= 0 else "#E74C3C" for v in L[:, 0]]
        bars = ax.bar(np.arange(1, N + 1), L[:, 0],
                      color=colors, alpha=0.8, edgecolor="white",
                      linewidth=0.8, width=0.7)
        ax.axhline(y=0, color="#E74C3C", linestyle="--",
                   linewidth=1, alpha=0.5)
        ax.set_xlabel("Cross-Section Unit (i)", fontsize=12)
        ax.set_ylabel("Factor Loading λᵢ", fontsize=12)
        ax.set_title("Factor Loadings (r = 1)", fontsize=14, fontweight="bold")
    else:
        scatter = ax.scatter(L[:, 0], L[:, 1],
                             c=ACCENT, s=100, alpha=0.8,
                             edgecolors="white", linewidth=1.5, zorder=5)
        for i in range(N):
            ax.annotate(str(i + 1), (L[i, 0], L[i, 1]),
                        fontsize=7, ha="center", va="bottom",
                        color="#7F8C8D", xytext=(0, 5),
                        textcoords="offset points")
        ax.axhline(y=0, color="#BDC3C7", linestyle="--",
                   linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color="#BDC3C7", linestyle="--",
                   linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Loading Factor 1 (λᵢ₁)", fontsize=12)
        ax.set_ylabel("Loading Factor 2 (λᵢ₂)", fontsize=12)
        ax.set_title("Factor Loadings Scatter",
                     fontsize=14, fontweight="bold")

    fig.text(0.5, -0.02,
             f"N = {N} cross-sections | Bai, Kao & Ng (2009)",
             ha="center", fontsize=9, color="#7F8C8D")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=BG_COLOR)
    return fig


def plot_convergence(results, figsize=(10, 5), save_path=None):
    """
    CupFM convergence path — β estimates across iterations.

    Parameters
    ----------
    results : CupFMResults
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()
    path = results.convergence_path
    n_iter, k = path.shape

    fig, ax = plt.subplots(figsize=figsize)
    iter_colors = ["#2980B9", "#E74C3C", "#27AE60", "#F39C12"]

    for j in range(k):
        ax.plot(np.arange(n_iter), path[:, j],
                color=iter_colors[j % len(iter_colors)],
                linewidth=2, marker="o", markersize=4,
                label=results.var_names[j], alpha=0.9)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Coefficient Estimate", fontsize=12)
    ax.set_title("CupFM Convergence Path", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=BG_COLOR)
    return fig


def plot_omega_heatmap(results, figsize=(8, 7), save_path=None):
    """
    Long-run covariance matrix Ω heatmap.

    Parameters
    ----------
    results : CupFMResults
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()
    Omega = results.Omega_cupfm
    m = Omega.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(Omega, cmap="RdBu_r", aspect="auto",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Covariance")

    for i in range(m):
        for j in range(m):
            val = Omega[i, j]
            color = "white" if abs(val) > np.abs(Omega).max() * 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title("Long-Run Covariance Ω̂ (CupFM)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Component", fontsize=11)
    ax.set_ylabel("Component", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=BG_COLOR)
    return fig


def plot_factor_ic(ic_values, figsize=(8, 5), save_path=None):
    """
    Bai-Ng information criterion values vs r.

    Parameters
    ----------
    ic_values : ndarray
        IC values for r = 1, ..., rmax.
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()
    rmax = len(ic_values)
    r_vals = np.arange(1, rmax + 1)
    best_r = r_vals[np.argmin(ic_values)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(r_vals, ic_values, color=ACCENT, linewidth=2.5,
            marker="o", markersize=8, markerfacecolor="white",
            markeredgecolor=ACCENT, markeredgewidth=2)
    ax.scatter([best_r], [ic_values[best_r - 1]], color="#E74C3C",
               s=200, zorder=10, marker="*", edgecolors="white",
               linewidth=1.5)
    ax.annotate(f"r* = {best_r}", (best_r, ic_values[best_r - 1]),
                fontsize=12, fontweight="bold", color="#E74C3C",
                xytext=(10, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#E74C3C"))

    ax.set_xlabel("Number of Factors (r)", fontsize=12)
    ax.set_ylabel("IC₁(r)", fontsize=12)
    ax.set_title("Bai & Ng (2002) Information Criterion",
                 fontsize=14, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=BG_COLOR)
    return fig


def plot_loadings_heatmap(results, figsize=(10, 6), save_path=None):
    """
    Factor loadings heatmap (N units × r factors).

    Parameters
    ----------
    results : CupFMResults
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    _setup_style()
    L = results.Lambda
    N, r = L.shape

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(L, cmap="RdBu_r", aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Loading λᵢⱼ")

    ax.set_xlabel("Factor", fontsize=12)
    ax.set_ylabel("Cross-Section Unit", fontsize=12)
    ax.set_xticks(range(r))
    ax.set_xticklabels([f"F{j+1}" for j in range(r)])
    ax.set_yticks(range(0, N, max(1, N // 10)))
    ax.set_title("Factor Loadings Heatmap",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight",
                    facecolor=BG_COLOR)
    return fig


def plot_all(results, save_prefix=None):
    """
    Generate all available plots.

    Parameters
    ----------
    results : CupFMResults
    save_prefix : str, optional
        If given, saves all plots as {prefix}_coef.png, etc.

    Returns
    -------
    dict of matplotlib.figure.Figure
    """
    figs = {}

    sp = lambda name: f"{save_prefix}_{name}.png" if save_prefix else None

    figs["coefficients"] = plot_coefficients(results, save_path=sp("coef"))
    figs["factors"] = plot_factors(results, save_path=sp("factors"))
    figs["loadings"] = plot_loadings(results, save_path=sp("loadings"))
    figs["convergence"] = plot_convergence(results, save_path=sp("convergence"))
    figs["omega"] = plot_omega_heatmap(results, save_path=sp("omega"))
    figs["loadings_heatmap"] = plot_loadings_heatmap(
        results, save_path=sp("loadings_heatmap")
    )

    return figs
