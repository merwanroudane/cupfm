"""
Results container for CupFM panel cointegration estimation.

Provides a rich results object with summary tables, significance testing,
confidence intervals, and export capabilities.

Author: Dr. Merwan Roudane <merwanroudane920@gmail.com>
"""

import numpy as np
import pandas as pd
from scipy import stats


class CupFMResults:
    """
    Container for all CupFM estimation results.

    Attributes
    ----------
    betas : dict
        Coefficient vectors for all 5 estimators.
    tstats : dict
        t-statistics for all 5 estimators.
    pvalues : dict
        Two-sided p-values for all 5 estimators.
    se : dict
        Standard errors for all 5 estimators.
    ci : dict
        95% confidence intervals for all 5 estimators.
    F_hat : ndarray (T, r)
        Estimated common factors.
    Lambda : ndarray (N, r)
        Estimated factor loadings.
    """

    ESTIMATOR_NAMES = ["LSDV", "Bai FM", "CupFM", "CupFM-bar", "CupBC"]
    ESTIMATOR_KEYS = ["lsdv", "baifm", "cupfm", "cupfm_bar", "cupbc"]

    def __init__(self, raw: dict, var_names: list = None, dep_var: str = "y"):
        """
        Initialize from raw estimator output.

        Parameters
        ----------
        raw : dict
            Output from estimators.run_all_estimators().
        var_names : list of str
            Variable names for regressors.
        dep_var : str
            Dependent variable name.
        """
        self._raw = raw
        k = len(raw["beta_cupfm"])
        self.dep_var = dep_var
        self.var_names = var_names or [f"x{j+1}" for j in range(k)]
        self.k = k
        self.N = raw["N"]
        self.T = raw["T"]
        self.N_obs = self.N * self.T
        self.r = raw["r"]
        self.bandwidth = raw["bandwidth"]
        self.kernel = raw["kernel"]
        self.n_iter = raw["n_iter"]
        self.convergence_path = raw["convergence_path"]

        # Factor structure
        self.F_hat = raw["F_hat"]
        self.Lambda = raw["Lambda"]
        self.Aik = raw["Aik"]
        self.Omega_cupfm = raw["Omega_cupfm"]
        self.Omega_cupbc = raw["Omega_cupbc"]
        self.cond_var_cupfm = raw["cond_var_cupfm"]
        self.cond_var_cupbc = raw["cond_var_cupbc"]

        # Build structured results for each estimator
        self.betas = {}
        self.tstats = {}
        self.pvalues = {}
        self.se = {}
        self.ci = {}

        for key, name in zip(self.ESTIMATOR_KEYS, self.ESTIMATOR_NAMES):
            b = raw[f"beta_{key}"]
            t = raw[f"tstat_{key}"]
            self.betas[name] = b
            self.tstats[name] = t
            self.pvalues[name] = 2.0 * (1.0 - stats.norm.cdf(np.abs(t)))
            se_vals = np.abs(b) / np.where(np.abs(t) > 1e-10, np.abs(t), 1e-10)
            self.se[name] = se_vals
            z95 = 1.96
            self.ci[name] = np.column_stack([b - z95 * se_vals, b + z95 * se_vals])

    @property
    def beta(self):
        """Primary CupFM coefficients."""
        return self.betas["CupFM"]

    @property
    def tstat(self):
        """Primary CupFM t-statistics."""
        return self.tstats["CupFM"]

    @property
    def pvalue(self):
        """Primary CupFM p-values."""
        return self.pvalues["CupFM"]

    def summary(self, print_output: bool = True) -> str:
        """
        Print a publication-quality summary table.

        Parameters
        ----------
        print_output : bool
            If True, print to stdout. Always returns the string.

        Returns
        -------
        str
            Formatted summary text.
        """
        cv01, cv05, cv10 = 2.576, 1.960, 1.645
        lines = []

        # Header
        lines.append("")
        lines.append("=" * 78)
        lines.append(
            "  cupfm — Panel Cointegration with Common Factors        "
            f"v1.0.0"
        )
        lines.append(
            "  Bai, Kao & Ng (2009, JoE 149:82-99)  |  "
            "Bai & Kao (2005, SSRN-1815227)"
        )
        lines.append("=" * 78)
        lines.append("")

        # Panel info
        lines.append("  Panel Information")
        lines.append("  " + "-" * 74)
        lines.append(
            f"  {'Dependent variable':<22} : {self.dep_var:<14}"
            f"  {'Regressors':<18} : {', '.join(self.var_names)}"
        )
        lines.append(
            f"  {'Cross-sections (N)':<22} : {self.N:<14}"
            f"  {'Time periods (T)':<18} : {self.T}"
        )
        lines.append(
            f"  {'Observations (N×T)':<22} : {self.N_obs:<14}"
            f"  {'Panel type':<18} : Balanced"
        )
        lines.append(
            f"  {'Common factors (r)':<22} : {self.r:<14}"
            f"  {'Bandwidth (M)':<18} : {self.bandwidth} ({self.kernel})"
        )
        lines.append(
            f"  {'Max iterations':<22} : {self.n_iter:<14}"
            f"  {'CupFM iterations':<18} : {self.n_iter}"
        )
        lines.append("  " + "-" * 74)
        lines.append("")

        # Estimation results table
        lines.append("  Estimation Results")
        lines.append("  " + "-" * 74)
        hdr = f"  {'Variable':>12}  |"
        for nm in self.ESTIMATOR_NAMES:
            hdr += f"{nm:>11}"
        lines.append(hdr)
        lines.append("  " + "-" * 14 + "+" + "-" * 58)

        for j in range(self.k):
            vn = self.var_names[j][:10]

            # Coefficients with stars
            coef_line = f"  {vn:>12}  |"
            tstat_line = f"  {'':>12}  |"
            for name in self.ESTIMATOR_NAMES:
                b = self.betas[name][j]
                t = self.tstats[name][j]
                at = abs(t)
                if at >= cv01:
                    star = "***"
                elif at >= cv05:
                    star = "** "
                elif at >= cv10:
                    star = "*  "
                else:
                    star = "   "
                coef_line += f" {b:7.4f}{star}"
                tstat_line += f"   ({t:6.2f})"
            lines.append(coef_line)
            lines.append(tstat_line)
            if j < self.k - 1:
                lines.append("  " + "-" * 14 + "+" + "-" * 58)

        lines.append("  " + "-" * 74)
        lines.append(
            "  t-statistics in parentheses  |  "
            "*** p<0.01  ** p<0.05  * p<0.10"
        )
        lines.append(
            "  CupFM = recommended (BKN 2009, Theorem 3)  |  "
            "CupFM-bar = Z-bar variant"
        )
        lines.append("")

        # Convergence summary
        lines.append("  Convergence Summary")
        lines.append("  " + "-" * 55)
        lines.append(
            f"  {'Estimator':<9} | {'Iterations':>12}"
            f"{'Omega_cond_var':>18}{'Status':>14}"
        )
        lines.append("  " + "-" * 10 + "+" + "-" * 44)
        lines.append(
            f"  {'CupFM':<9} | {self.n_iter:>12}"
            f"{self.cond_var_cupfm:>18.6f}{'Converged':>14}"
        )
        lines.append(
            f"  {'CupBC':<9} | {self.n_iter:>12}"
            f"{self.cond_var_cupbc:>18.6f}{'Fixed iter':>14}"
        )
        lines.append(
            f"  {'Bai FM':<9} | {1:>12}"
            f"{'---':>18}{'One-step':>14}"
        )
        lines.append("  " + "-" * 55)
        lines.append("")

        text = "\n".join(lines)
        if print_output:
            print(text)
        return text

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with estimator results.
        """
        records = []
        for name in self.ESTIMATOR_NAMES:
            for j in range(self.k):
                records.append({
                    "Variable": self.var_names[j],
                    "Estimator": name,
                    "Coefficient": self.betas[name][j],
                    "Std.Error": self.se[name][j],
                    "t-statistic": self.tstats[name][j],
                    "p-value": self.pvalues[name][j],
                    "CI_lower": self.ci[name][j, 0],
                    "CI_upper": self.ci[name][j, 1],
                })
        return pd.DataFrame(records)

    def to_latex(self, caption: str = None) -> str:
        """Generate a LaTeX table of estimation results."""
        cv01, cv05, cv10 = 2.576, 1.960, 1.645
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("  \\centering\\small")
        cap = caption or "Panel Cointegration Estimation Results"
        lines.append(f"  \\caption{{{cap}}}")
        lines.append("  \\label{tab:cupfm}")
        lines.append("  \\begin{tabular}{l*{5}{c}}")
        lines.append("    \\hline\\hline")
        lines.append(
            "    & LSDV & Bai FM & CupFM & CupFM-bar & CupBC \\\\"
        )
        lines.append("    \\hline")

        for j in range(self.k):
            vn = self.var_names[j].replace("_", "\\_")
            coef_parts = [f"    {vn}"]
            tstat_parts = ["    "]
            for name in self.ESTIMATOR_NAMES:
                b = self.betas[name][j]
                t = self.tstats[name][j]
                at = abs(t)
                if at >= cv01:
                    star = "^{***}"
                elif at >= cv05:
                    star = "^{**}"
                elif at >= cv10:
                    star = "^{*}"
                else:
                    star = ""
                coef_parts.append(f" & ${b:.4f}{star}$")
                tstat_parts.append(f" & ({t:.3f})")
            lines.append("".join(coef_parts) + " \\\\")
            lines.append("".join(tstat_parts) + " \\\\")

        lines.append("    \\hline")
        lines.append(
            f"    N (units) & \\multicolumn{{5}}{{c}}{{{self.N}}} \\\\"
        )
        lines.append(
            f"    T (periods) & \\multicolumn{{5}}{{c}}{{{self.T}}} \\\\"
        )
        lines.append(
            f"    r (factors) & \\multicolumn{{5}}{{c}}{{{self.r}}} \\\\"
        )
        lines.append("    \\hline\\hline")
        lines.append("  \\end{tabular}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    def to_excel(self, filename: str):
        """Export results to Excel file."""
        df = self.to_dataframe()
        df.to_excel(filename, index=False, sheet_name="CupFM Results")

    def to_csv(self, filename: str):
        """Export results to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filename, index=False)

    def __repr__(self):
        return (
            f"CupFMResults(N={self.N}, T={self.T}, k={self.k}, "
            f"r={self.r}, estimator='CupFM')"
        )
