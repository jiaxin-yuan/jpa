"""Key every THP test row by (case, prefix length), replaying the THP dataloader order -> results/keys."""
import os
import re
import sys

import numpy as np
import pandas as pd

import config as C

_PREFIX_RE = re.compile(r"_p\d+$")


def build(ds):
    """One row per scored THP position, in evaluation order (cases by first appearance)."""
    df = pd.read_csv(os.path.join(C.THP_DATA, f"test_fold0_variation0_{ds}.csv"))
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed")
    g = df.groupby("CaseID", sort=False)
    df["gap"] = g["Timestamp"].diff().dt.total_seconds().div(C.SEC_PER_DAY)
    rows = []
    for exp_case, grp in df.groupby("CaseID", sort=False):
        acts, gaps = grp["Activity"].tolist(), grp["gap"].to_numpy()
        orig = _PREFIX_RE.sub("", str(exp_case))
        for t in range(1, len(acts)):          # target position t, prefix length t
            rows.append((orig, str(exp_case), t, acts[t - 1], acts[t], gaps[t]))
    return pd.DataFrame(rows, columns=["orig_case", "exp_case", "prefix_len",
                                       "last_act", "target_act", "recon_gap"])


def main(logs):
    """Write results/keys/<log>_keys.npz."""
    os.makedirs(C.KEYS, exist_ok=True)
    for ds in logs:
        k = build(ds)
        np.savez_compressed(os.path.join(C.KEYS, f"{ds}_keys.npz"),
                            orig_case=k["orig_case"].to_numpy().astype(str),
                            exp_case=k["exp_case"].to_numpy().astype(str),
                            prefix_len=k["prefix_len"].to_numpy().astype(np.int32),
                            last_act=k["last_act"].to_numpy().astype(str),
                            target_act=k["target_act"].to_numpy().astype(str),
                            recon_gap=k["recon_gap"].to_numpy().astype(np.float32))
        print(f"{ds}: {len(k)} THP rows keyed")


if __name__ == "__main__":
    main(sys.argv[1:] or C.LOGS)
