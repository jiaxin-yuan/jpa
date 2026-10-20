"""SuTraN and ED-LSTM per-instance predictions on S, with the Eq. (6) wrapper -> results/predictions."""
import os
import pickle
import sys

import numpy as np
import torch

import config as C
import wrapper
from export_thp import save

S2S_OUT = os.path.join(C.RESULTS, "seq2seq")
TAGS = {"SuTraN": "sutran", "ED-LSTM": "edlstm"}


def step0(ds, model, split, mean, std):
    """Step-0 (next event) point prediction and true gap in days, activity hit, validity."""
    def load(name):
        return torch.load(os.path.join(S2S_OUT, ds, model, split, name), map_location="cpu",
                          weights_only=False)
    ttne, labels, acts = load("suffix_ttne_preds.pt"), load("labels.pt"), \
        load("suffix_acts_decoded.pt")
    gt = labels[2][:, 0].numpy()
    return {"point": (ttne[:, 0].numpy() * std + mean) / C.SEC_PER_DAY,
            "gap": (labels[0][:, 0, 0].numpy() * std + mean) / C.SEC_PER_DAY,
            "correct": acts[:, 0].numpy() == gt, "valid": gt > 0}


def run(ds):
    """Export both seq2seq models for one log."""
    base = os.path.join(C.S2S_DATA, ds)
    with open(os.path.join(base, f"{ds}_strict_train_means_dict.pkl"), "rb") as f:
        mean = pickle.load(f)["timeLabel_df"][0]
    with open(os.path.join(base, f"{ds}_strict_train_std_dict.pkl"), "rb") as f:
        std = pickle.load(f)["timeLabel_df"][0]
    pop = np.load(os.path.join(C.POP, f"{ds}.npz"), allow_pickle=True)
    idx = pop["bl_row"]
    for model, tag in TAGS.items():
        v = step0(ds, model, "val", mean, std)
        fit = wrapper.fit(v["point"][v["valid"]], v["gap"][v["valid"]])
        t = step0(ds, model, "test", mean, std)
        assert (np.abs(t["gap"][idx] - pop["gap_days"]) * C.SEC_PER_DAY <= 1.0).all(), \
            "row indexing disagrees with the population"
        point = t["point"][idx]
        dens = wrapper.decode(wrapper.params(point, *fit), C.EPS_GRID)
        save(ds, tag, pop, t["correct"][idx], t["gap"][idx], point, dens)
        print(f"{ds}: {model} on |S| = {len(idx)} "
              f"(wrapper pi={fit[0]:.3f} delta={fit[1]:+.3f} s={fit[2]:.3f})")


if __name__ == "__main__":
    for d in sys.argv[1:] or C.LOGS:
        run(d)
