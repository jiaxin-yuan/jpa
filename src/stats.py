"""Exact McNemar test on paired joint hits, 95% CI of the paired difference, Holm correction."""
import numpy as np
from scipy.stats import binomtest


def mcnemar(a, b):
    """Paired comparison of two hit vectors; diff and CI in points (A - B)."""
    n = len(a)
    n_ab = int(np.sum(a & ~b))
    n_ba = int(np.sum(~a & b))
    p = binomtest(n_ab, n_ab + n_ba, 0.5).pvalue if n_ab + n_ba > 0 else 1.0
    diff = 100.0 * (n_ab - n_ba) / n
    var = ((n_ab + n_ba) - (n_ab - n_ba) ** 2 / n) / n ** 2
    half = 100.0 * 1.959964 * np.sqrt(max(var, 0.0))
    return {"diff": diff, "lo": diff - half, "hi": diff + half, "p": p}


def holm(rows):
    """Holm-Bonferroni adjusted p-values over one family, written to row['p_holm']."""
    order = np.argsort([r["p"] for r in rows])
    m, running = len(rows), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[i]["p"]))
        rows[i]["p_holm"] = running
    return rows
