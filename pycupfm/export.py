"""
Export utilities for CupFM results.

Author: Dr. Merwan Roudane <merwanroudane920@gmail.com>
"""


def export_results(results, filename, fmt="all"):
    """
    Export CupFM results to external files.

    Parameters
    ----------
    results : CupFMResults
    filename : str
        Base filename (no extension).
    fmt : str
        'excel', 'latex', 'csv', 'html', or 'all'.
    """
    if fmt in ("csv", "all"):
        results.to_csv(f"{filename}.csv")
    if fmt in ("excel", "all"):
        results.to_excel(f"{filename}.xlsx")
    if fmt in ("latex", "all"):
        tex = results.to_latex()
        with open(f"{filename}.tex", "w") as f:
            f.write(tex)
    if fmt in ("html", "all"):
        df = results.to_dataframe()
        html = df.to_html(index=False, float_format="%.4f")
        with open(f"{filename}.html", "w") as f:
            f.write(f"""<!DOCTYPE html>
<html><head>
<style>
body {{ font-family: 'Segoe UI', sans-serif; padding: 2rem; background: #FAFBFC; }}
table {{ border-collapse: collapse; margin: 1rem auto; }}
th {{ background: #2C3E50; color: white; padding: 8px 12px; }}
td {{ padding: 6px 12px; border-bottom: 1px solid #E8ECF0; }}
tr:hover {{ background: #F0F4F8; }}
h1 {{ color: #2C3E50; text-align: center; }}
</style>
</head><body>
<h1>CupFM — Panel Cointegration Results</h1>
<p style="text-align:center;color:#7F8C8D;">
Bai, Kao & Ng (2009) | Dr. Merwan Roudane
</p>
{html}
</body></html>""")
