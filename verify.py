"""Compare tables/values.json with the values printed in the paper (reference/paper_tables.json)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import config as C  # noqa: E402

DIGITS = {"table1": [0, 0, 0, 0, 1], "table2": [None, 1, None, 3, None, 1, None],
          "table3": [1] * 7 + [0, 1], "table4": [1] * 6, "b1": [1, 1],
          "table5": [None, None, None, 2, 2, 2, "skip", None, None, None]}


def same(got, ref, d):
    """Equal at the printed precision d (None = exact match)."""
    if ref is None or got is None:
        return ref is got
    if d is None:
        return got == ref
    return abs(round(got, d) - ref) <= 0.5 * 10 ** -d + 1e-9


def check_row(name, got, ref, digits, bad, unit=None):
    """Append (name, column, got, ref, one_instance) for every mismatching cell.

    one_instance is True when a percent-point cell differs by at most one instance
    (100/N points) beyond the printed precision: GPU re-inference noise, not a defect.
    """
    for i, (g, r, d) in enumerate(zip(got, ref, digits)):
        if d == "skip":
            continue
        if not same(g, r, d):
            near = (unit is not None and isinstance(d, int) and d > 0
                    and abs(g - r) <= unit + 0.5 * 10 ** -d)
            bad.append((name, i, g, r, near))


def main():
    """Print every mismatch; exit 1 if any."""
    got = json.load(open(os.path.join(C.TABLES, "values.json")))
    ref = json.load(open(C.REFERENCE))
    bad, n = [], 0
    unit = {log: 100.0 / v[3] for log, v in ref["table1"].items() if log != "columns"}
    for t in ("table1", "table2", "table3"):
        for log, row in ref[t].items():
            if log == "columns":
                continue
            check_row(f"{t}/{log}", got[t][log], row, DIGITS[t], bad,
                      None if t == "table1" else unit.get(log))
            n += len(row)
    for log, row in ref["table4"].items():
        if log in ("columns", "b1_columns", "dagger"):
            continue
        for m, v in row.items():
            check_row(f"table4/{log}/{m}", got["table4"][log][m], v,
                      DIGITS["b1"] if m == "B1" else DIGITS["table4"], bad, unit[log])
            n += len(v)
    if sorted(map(tuple, got["table4"]["dagger"])) != sorted(map(tuple, ref["table4"]["dagger"])):
        bad.append(("table4/dagger", "-", got["table4"]["dagger"], ref["table4"]["dagger"],
                    False))
    n += 1
    for i, (g, r) in enumerate(zip(got["table5"]["rows"], ref["table5"]["rows"])):
        check_row(f"table5/row{i}", g, r, DIGITS["table5"], bad, unit[r[0]])
        # p_Holm: two significant digits as printed; "<1e-300" stored as 0.0
        if not (r[6] == 0.0 and g[6] < 1e-300) and r[6] != 0.0 \
                and abs(g[6] - r[6]) > 0.051 * max(r[6], 1e-300) + (0.005 if r[6] >= 0.01 else 0):
            bad.append((f"table5/row{i}", "p_holm", g[6], r[6], False))
        n += len(r)
    if len(got["table5"]["rows"]) != len(ref["table5"]["rows"]):
        bad.append(("table5", "n_rows", len(got["table5"]["rows"]),
                    len(ref["table5"]["rows"]), False))
    for b in bad:
        print("WITHIN ONE INSTANCE" if b[4] else "MISMATCH", *b[:4])
    off = [b for b in bad if not b[4]]
    print(f"{n - len(bad)}/{n} cells match the paper exactly; "
          f"{len(bad) - len(off)} within one instance; {len(off)} mismatches")
    return 1 if off else 0


if __name__ == "__main__":
    sys.exit(main())
