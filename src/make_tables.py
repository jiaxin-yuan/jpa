"""Compute Tables 1-5 of the paper from per-instance predictions -> tables/values.json + .tex."""
import itertools
import json
import os

import numpy as np
import pandas as pd

import config as C
import jpa
import naive
import stats

DENSITY_DECODES = ["mean", "median", "eps_mode"]
GRID_PICKS = [0, 4, 8, 12, 16, 20, 24]                     # Table 3 columns; 12 = eps*
GRID_LABELS = ["86.4 ms", "1.1 s", "14.8 s", "193 s", "42.1 min", "9.2 h", "5.0 d"]
ALPHA, MAGNITUDE, WIDE = 0.05, 2.0, 0.5                     # P5 thresholds (points)


def load_all():
    """Load every model on every log once."""
    return {ds: {m: jpa.load_model(ds, m) for m in C.MODELS} for ds in C.LOGS}


def eps_mode_hits(r, k=C.EPS_STAR_IDX):
    """Joint hits under the eps-mode decode at grid tolerance k."""
    return jpa.hits(r, jpa.decode(r, "eps_mode", k), C.EPS_GRID[k])


def point_hits(r):
    """Joint hits at eps* under the decode each model natively reports (THP-M: mean)."""
    return jpa.hits(r, r["mean"] if r["point"] is None else r["point"], C.EPS_STAR)


def scored_hits(r, m, k=C.EPS_STAR_IDX):
    """Hits under the decode used to compare models: THP-B point, all others eps-mode."""
    if m == "THP-B":
        return jpa.hits(r, r["point"], C.EPS_GRID[k])
    return eps_mode_hits(r, k)


def table1(D):
    """Raw-log counts, |S|, and share of scored prefixes with tau < eps*."""
    raw = pd.read_csv(os.path.join(C.RESULTS, "log_stats.csv")).set_index("log")
    out = {}
    for ds in C.LOGS:
        r, s = D[ds]["THP-M"], C.SHORT[ds]
        out[s] = [int(raw.loc[s, "cases"]), int(raw.loc[s, "events"]),
                  int(raw.loc[s, "activities"]), int(len(r["gap"])),
                  100.0 * (r["gap"] < C.EPS_STAR).mean()]
    return out


def table2(D):
    """Best model and margin to the runner-up under Acc, cMAE and JPA at eps*.

    cMAE uses the median decode, JPA the eps-mode decode; THP-B enters with its point
    prediction. Acc and JPA margins are taken between the values as printed (1 decimal).
    """
    out = {}
    for ds in C.LOGS:
        P = D[ds]
        acc = {m: 100.0 * P[m]["correct"].mean() for m in C.MODELS}
        cm = {m: jpa.cmae(P[m], P[m]["point"] if m == "THP-B" else P[m]["median"])
              for m in C.MODELS}
        jp = {m: 100.0 * scored_hits(P[m], m).mean() for m in C.MODELS}
        row = []
        for v, lower, nd in ((acc, False, 1), (cm, True, None), (jp, False, 1)):
            order = sorted(v, key=v.get, reverse=not lower)
            a, b = v[order[0]], v[order[1]]
            row += [order[0], abs(a - b) if nd is None else abs(round(a, nd) - round(b, nd))]
        row.append(row[4] not in (row[0], row[2]))
        out[C.SHORT[ds]] = row
    return out


def table3(D):
    """Spread of eps-mode JPA over the four models with a density, winner flips, THP-M range.

    Flips count winner changes along the 25-value grid over all scored variants: the four
    eps-mode curves plus THP-B's point prediction.
    """
    out, cols = {}, []
    K = range(len(C.EPS_GRID))
    for ds in C.LOGS:
        curve = {m: np.array([100.0 * eps_mode_hits(D[ds][m], k).mean() for k in K])
                 for m in C.MODELS}
        mat = np.stack([curve[m] for m in C.MODELS])
        spread = mat.max(0) - mat.min(0)
        pt = np.array([100.0 * scored_hits(D[ds]["THP-B"], "THP-B", k).mean() for k in K])
        win = np.vstack([mat, pt]).argmax(0)
        flips = int((win[1:] != win[:-1]).sum())
        rng = float(curve["THP-M"].max() - curve["THP-M"].min())
        out[C.SHORT[ds]] = [float(spread[k]) for k in GRID_PICKS] + [flips, rng]
        cols.append(out[C.SHORT[ds]])
    out["median"] = [float(np.median(c)) for c in zip(*cols)]
    return out


def b1_hits(ds):
    """B1 joint hits at eps* and its activity accuracy."""
    hit, gap = naive.b1(ds)
    return hit & (gap < C.EPS_STAR), 100.0 * hit.mean()


def b1_family(D):
    """B1 vs every model (THP-B point, others eps-mode), one Holm family of 40 tests."""
    rows = []
    for ds in C.LOGS:
        h, _ = b1_hits(ds)
        for m in C.MODELS:
            rows.append(dict(log=C.SHORT[ds], a="B1", b=m,
                             **stats.mcnemar(h, scored_hits(D[ds][m], m))))
    return stats.holm(rows)


def table4(D, fam_b1):
    """Every decode at eps*, the B2 null, skill of the native point output, and B1."""
    dag = [[r["log"], r["b"]] for r in fam_b1 if r["p_holm"] >= ALPHA]
    out = {"dagger": dag}
    for ds in C.LOGS:
        row = {}
        for m in C.MODELS:
            r = D[ds][m]
            cells = [None if r["point"] is None else jpa.jpa(r, r["point"])]
            cells += [jpa.jpa(r, jpa.decode(r, d)) for d in DENSITY_DECODES]
            j_null = 100.0 * jpa.null_hits(r, C.EPS_STAR).mean()
            native = 100.0 * point_hits(r).mean()
            row[m] = cells + [j_null, jpa.skill(native, j_null)]
        h, acc = b1_hits(ds)
        row["B1"] = [acc, 100.0 * h.mean()]
        out[C.SHORT[ds]] = row
    return out


def table5(D, fam_b1):
    """Paired differences: selected rows of the 60 model pairs, then B1 non-rejections."""
    pairs = []
    for ds in C.LOGS:
        h = {m: scored_hits(D[ds][m], m) for m in C.MODELS}
        for a, b in itertools.combinations(C.MODELS, 2):
            pairs.append(dict(log=C.SHORT[ds], a=a, b=b, **stats.mcnemar(h[a], h[b])))
    stats.holm(pairs)
    sig = sorted([r for r in pairs if r["p_holm"] < ALPHA], key=lambda r: abs(r["diff"]))
    dens = [r for r in sig if "THP-B" not in (r["a"], r["b"])]
    ns = sorted([r for r in pairs if r["p_holm"] >= ALPHA], key=lambda r: r["hi"] - r["lo"])
    pick = [ns[0], sig[0], dens[(len(dens) - 1) // 2], ns[-1], sig[-1]]
    pick = sorted(pick, key=lambda r: C.LOGS.index("UQ_" + r["log"].replace("-", "_")))
    pick += [r for r in fam_b1 if r["p_holm"] >= ALPHA]
    rows = []
    for r in pick:
        rows.append([r["log"], r["a"], r["b"], r["diff"], r["lo"], r["hi"], r["p_holm"],
                     bool(r["p_holm"] < ALPHA), bool(abs(r["diff"]) >= MAGNITUDE),
                     bool(r["lo"] < -WIDE or r["hi"] > WIDE)])
    return {"rows": rows, "n_pairs": len(pairs), "n_significant": len(sig)}


# ------------------------------------------------------------------ LaTeX
def f1(x):
    """One decimal, en-dash for missing, math minus for negatives."""
    return "--" if x is None else (f"${x:.1f}$" if x < 0 else f"{x:.1f}")


def tabular(cols, head, body):
    """Minimal booktabs tabular."""
    return "\n".join([f"\\begin{{tabular}}{{{cols}}}", "\\toprule", head + " \\\\",
                      "\\midrule"] + [b + " \\\\" for b in body]
                     + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def write_tex(T):
    """One .tex tabular per table, values only (captions live in the paper)."""
    t = T["table1"]
    body = [" & ".join([s, f"{v[0]:,}".replace(",", "\\,"), f"{v[1]:,}".replace(",", "\\,"),
                        str(v[2]), f"{v[3]:,}".replace(",", "\\,"), f"{v[4]:.1f}"])
            for s, v in t.items()]
    tex = {"tab1_datasets": tabular("lrrrrr", "Log & Cases & Events & Activities & $N$ & "
                                    "Inside $\\varepsilon^\\star$ (\\%)", body)}
    body = [" & ".join([s, v[0], f"{v[1]:.1f}", v[2], f"{v[3]:.3f}", v[4], f"{v[5]:.1f}",
                        "$\\bullet$" if v[6] else ""]) for s, v in T["table2"].items()]
    tex["tab2_best"] = tabular("llrlrlrc", "Log & best Acc & $\\Delta$ & best cMAE & $\\Delta$ & "
                               "best JPA$_{\\varepsilon^\\star}$ & $\\Delta$ & differs?", body)
    body = [" & ".join([s] + [f"{x:.1f}" for x in v[:7]] + [f"{v[7]:.0f}", f"{v[8]:.1f}"])
            for s, v in T["table3"].items()]
    tex["tab3_spread"] = tabular("l" + "r" * 9, "Log & " + " & ".join(GRID_LABELS)
                                 + " & flips & range", body)
    body, dag = [], {tuple(d) for d in T["table4"]["dagger"]}
    for s in (C.SHORT[d] for d in C.LOGS):
        row = T["table4"][s]
        for i, m in enumerate(C.MODELS):
            v = list(row[m])
            em = f1(v[3]) + ("$^\\dagger$" if (s, m) in dag else "")
            if row["B1"][1] > v[3]:
                em = f"\\textbf{{{em}}}"
            b1 = [f"{row['B1'][0]:.1f}", f"{row['B1'][1]:.1f}"] if i == 0 else ["", ""]
            body.append(" & ".join([s if i == 0 else "", m, f1(v[0]), f1(v[1]), f1(v[2]), em,
                                    f1(v[4]), f1(v[5])] + b1))
    tex["tab4_baselines"] = tabular("llrrrrrrrr", "Log & Method & point & mean & median & "
                                    "$\\varepsilon$-mode & B2 null & skill (\\%) & B1 Acc & B1 JPA",
                                    body)
    yn = {True: "yes", False: "no"}
    body = []
    for r in T["table5"]["rows"]:
        p = "$<10^{-300}$" if r[6] < 1e-300 else (f"{r[6]:.2f}" if r[6] >= 0.01 else
                                                  "${:.1f}\\times10^{{{}}}$".format(
                                                      *_mant_exp(r[6])))
        body.append(" & ".join([r[0], f"{r[1]}$-${r[2]}", f"${r[3]:+.2f}$",
                                f"$[{r[4]:+.2f},{r[5]:+.2f}]$", p] + [yn[x] for x in r[7:]]))
    tex["tab5_paired"] = tabular("llrrrccc", "Log & Comparison & $\\Delta$ (pt) & 95\\% CI & "
                                 "$p_{\\mathrm{Holm}}$ & signif. & magn. & wide", body)
    for name, s in tex.items():
        with open(os.path.join(C.TABLES, f"{name}.tex"), "w") as f:
            f.write(s)


def _mant_exp(p):
    """Mantissa and exponent of p for scientific notation."""
    e = int(np.floor(np.log10(p)))
    return p / 10 ** e, e


def main():
    """Compute all tables, write values.json and the .tex files."""
    D = load_all()
    fam_b1 = b1_family(D)
    T = {"table1": table1(D), "table2": table2(D), "table3": table3(D),
         "table4": table4(D, fam_b1), "table5": table5(D, fam_b1)}
    os.makedirs(C.TABLES, exist_ok=True)
    with open(os.path.join(C.TABLES, "values.json"), "w") as f:
        json.dump(T, f, indent=1)
    write_tex(T)
    print(f"wrote {C.TABLES}/values.json and tab1-5 .tex")


if __name__ == "__main__":
    main()
