"""B1 naive baseline: most frequent successor of the last activity (training split), tau_hat = 0."""
import os

import numpy as np
import pandas as pd

import config as C


def b1(ds, keys_dir=C.KEYS, pop_dir=C.POP, thp_data=C.THP_DATA):
    """Return (activity hits, true gaps) of B1 on S, row-aligned with jpa.load_model."""
    tr = pd.read_csv(os.path.join(thp_data, f"train_fold0_variation0_{ds}.csv"),
                     usecols=["CaseID", "Activity"])
    pairs = pd.DataFrame({"last": tr.groupby("CaseID", sort=False)["Activity"].shift(1),
                          "next": tr["Activity"]}).dropna()
    top = pairs["next"].value_counts().idxmax()
    table = pairs.groupby("last")["next"].agg(lambda s: s.value_counts().idxmax())

    keys = np.load(os.path.join(keys_dir, f"{ds}_keys.npz"), allow_pickle=True)
    pop = np.load(os.path.join(pop_dir, f"{ds}.npz"), allow_pickle=True)
    idx = pop["thp_row"]
    hit = (pd.Series(keys["last_act"][idx]).map(table).fillna(top).to_numpy()
           == keys["target_act"][idx])
    return hit, pop["gap_days"].astype(np.float64)
