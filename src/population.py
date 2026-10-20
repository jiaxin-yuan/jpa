"""Evaluation prefix log S: prefix-level intersection of THP and seq2seq rows, with checks V0-V2.

Prefixes are keyed by (case, prefix length). V0: the seq2seq prefix lengths agree
elementwise across models and with the stored keys. V1: each seq2seq target gap,
de-standardised, equals the gap read off the log. V2: on paired instances both sides agree
on the gap (<= 2 s, timestamp resolution) and on the activity names. Cases whose event
sequences diverge between the two logs are excluded whole; positions inside reordered
timestamp ties are excluded row-wise. Writes results/population/<log>.npz.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

import config as C

S2S_OUT = os.path.join(C.RESULTS, "seq2seq")
MODELS = ["SuTraN", "ED-LSTM"]
CASE, TIME = "case:concept:name", "time:timestamp"
STD_ATOL = 1e-5            # float32 round-trip of a standardised label
GAP_ATOL_SEC = 2.0         # V2 gap tolerance
ACT_MISMATCH_ABORT = 0.10  # above this share of activity disagreement the log is aborted


def _pkl(ds, name):
    """Load one seq2seq metadata pickle."""
    with open(os.path.join(C.S2S_DATA, ds, f"{ds}_strict_{name}.pkl"), "rb") as f:
        return pickle.load(f)


def _mapped_log(ds, cols=None):
    """Seq2seq test log sorted stably by (case, time)."""
    log = pd.read_csv(os.path.join(C.S2S_DATA, ds, f"{ds}_strict_test_mapped.csv"),
                      usecols=cols, dtype={CASE: str})
    log[TIME] = pd.to_datetime(log[TIME])
    return log.sort_values([CASE, TIME], kind="mergesort")


def _load(ds, model, name):
    """One stored seq2seq test tensor."""
    return torch.load(os.path.join(S2S_OUT, ds, model, "test", name), map_location="cpu",
                      weights_only=False)


def seq2seq_rows(ds):
    """Seq2seq test rows with keys, de-standardised step-0 gap (days) and validity; V0, V1."""
    keys = pd.read_csv(os.path.join(C.S2S_DATA, ds, "test_prefix_keys.csv"),
                       dtype={"case_id": str})
    pl = [_load(ds, m, "pref_len.pt").numpy() for m in MODELS]
    assert np.array_equal(pl[0], pl[1]) and np.array_equal(pl[0], keys["prefix_len"]), "V0"
    mean, std = _pkl(ds, "train_means_dict")["timeLabel_df"][0], \
        _pkl(ds, "train_std_dict")["timeLabel_df"][0]
    labels = _load(ds, MODELS[0], "labels.pt")
    z = labels[0][:, 0, 0].numpy()
    valid = labels[2][:, 0].numpy() > 0

    epoch = {c: g[TIME].values.astype("datetime64[ns]").astype(np.int64) / 1e9
             for c, g in _mapped_log(ds).groupby(CASE, sort=False)}
    rec = np.full(len(keys), np.nan)
    for i, (c, k) in enumerate(zip(keys["case_id"], pl[0])):
        ts = epoch.get(c)
        if ts is not None and 1 <= k < len(ts):
            rec[i] = ts[k] - ts[k - 1]
    rec_z = ((rec - mean) / std).astype(np.float32)
    chk = valid & np.isfinite(rec_z)
    assert chk.any() and (np.abs(rec_z[chk] - z[chk]) <= STD_ATOL).all(), "V1"
    return pd.DataFrame({"case_id": keys["case_id"].to_numpy(), "prefix_len": pl[0],
                         "bl_row": np.arange(len(keys)),
                         "bl_gap": (z.astype(np.float64) * std + mean) / C.SEC_PER_DAY,
                         "valid": valid})


def thp_rows(ds):
    """One row per THP prefix: the longest position of each prefix-expanded sequence."""
    k = np.load(os.path.join(C.KEYS, f"{ds}_keys.npz"), allow_pickle=True)
    df = pd.DataFrame({"exp_case": k["exp_case"].astype(str),
                       "case_id": k["orig_case"].astype(str),
                       "prefix_len": k["prefix_len"].astype(np.int64),
                       "recon_gap": k["recon_gap"].astype(np.float64),
                       "last_act": k["last_act"].astype(str),
                       "target_act": k["target_act"].astype(str),
                       "thp_row": np.arange(len(k["exp_case"]))})
    last = df.loc[df.groupby("exp_case", sort=False)["prefix_len"].idxmax()]
    assert not last.duplicated(["case_id", "prefix_len"]).any()
    return last.reset_index(drop=True)


def diverging_cases(ds):
    """Cases whose THP event sequence is not a prefix of the seq2seq one (by timestamps)."""
    thp = pd.read_csv(os.path.join(C.THP_DATA, f"test_fold0_variation0_{ds}.csv"),
                      usecols=["CaseID", "Timestamp"], dtype={"CaseID": str})
    thp["Timestamp"] = pd.to_datetime(thp["Timestamp"], format="%Y/%m/%d %H:%M:%S.%f")
    size = thp.groupby("CaseID", sort=False).size().rename("n").reset_index()
    size["orig"] = size["CaseID"].str.rsplit("_p", n=1).str[0]
    longest = set(size.loc[size.groupby("orig")["n"].idxmax(), "CaseID"])
    to_orig = dict(zip(size["CaseID"], size["orig"]))
    T = {to_orig[c]: g["Timestamp"].sort_values(kind="mergesort").tolist()
         for c, g in thp[thp["CaseID"].isin(longest)].groupby("CaseID", sort=False)}
    B = {c: g[TIME].sort_values(kind="mergesort").tolist()
         for c, g in _mapped_log(ds, [CASE, TIME]).groupby(CASE, sort=False)}
    return {c for c, t in T.items() if c in B and t != B[c][:len(t)]}


def build(ds):
    """Intersect, run V2, drop tie-reordered positions; return the population table."""
    m = seq2seq_rows(ds).merge(thp_rows(ds), on=["case_id", "prefix_len"], how="inner")
    m = m[m["valid"] & ~m["case_id"].isin(diverging_cases(ds))].reset_index(drop=True)
    assert len(m) and ((m["bl_gap"] - m["recon_gap"]).abs() * C.SEC_PER_DAY
                       <= GAP_ATOL_SEC).all(), "V2 gap"
    inv = {v: k for k, v in _pkl(ds, "categ_mapping")["concept:name"].items()}
    acts = {c: [inv.get(a, "?") for a in g["concept:name"]]
            for c, g in _mapped_log(ds).groupby(CASE, sort=False)}
    last = [acts[c][k - 1] if 0 < k <= len(acts[c]) else "?"
            for c, k in zip(m["case_id"], m["prefix_len"])]
    targ = [acts[c][k] if k < len(acts[c]) else "?"
            for c, k in zip(m["case_id"], m["prefix_len"])]
    ok = (np.array(last) == m["last_act"].to_numpy()) & \
         (np.array(targ) == m["target_act"].to_numpy())
    assert (~ok).mean() <= ACT_MISMATCH_ABORT, "V2 activity"
    return m[ok].reset_index(drop=True), int((~ok).sum())


def main(logs):
    """Write results/population/<log>.npz for each log."""
    os.makedirs(C.POP, exist_ok=True)
    for ds in logs:
        m, n_tie = build(ds)
        np.savez_compressed(os.path.join(C.POP, f"{ds}.npz"),
                            case_id=m["case_id"].to_numpy().astype(str),
                            prefix_len=m["prefix_len"].to_numpy(np.int32),
                            bl_row=m["bl_row"].to_numpy(np.int64),
                            thp_row=m["thp_row"].to_numpy(np.int64),
                            gap_days=m["bl_gap"].to_numpy(np.float64))
        print(f"{ds}: |S| = {len(m)} (V0-V2 passed, {n_tie} tie-reordered positions dropped)")


if __name__ == "__main__":
    main(sys.argv[1:] or C.LOGS)
