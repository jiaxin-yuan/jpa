"""JPA criterion (Eq. 2), B2 null, skill (Eq. 4), cMAE (Eq. 7) and the per-model loader."""
import os

import numpy as np

import config as C


def _read(ds, tag, pred_dir):
    """Read one per-instance file; every array is already restricted to S."""
    d = np.load(os.path.join(pred_dir, ds, f"{tag}.npz"), allow_pickle=True)
    r = {"correct": d["correct"] > 0,
         "gap": d["true_gap"].astype(np.float64),
         "point": d["pred_point"].astype(np.float64),
         "case": d["case_id"].astype(str),
         "plen": d["prefix_len"].astype(np.int64)}
    if int(d["has_density"]):
        r["mean"] = d["pred_mean"].astype(np.float64)
        r["median"] = d["pred_median"].astype(np.float64)
        r["eps_mode_grid"] = d["pred_eps_mode"].astype(np.float64)   # [25, |S|]
    return r


def load_model(ds, model, pred_dir=C.PRED):
    """Return one model's activity hits, true gaps and every declared decode on S.

    THP-M carries a native density and reports no separate point decode. THP-B, SuTraN
    and ED-LSTM report a point decode; their density decodes come from the wrapper (Eq. 6).
    """
    if model == "THP-M":
        r = _read(ds, "thp_mixture", pred_dir)
        r["point"] = None
        return r
    if model == "THP-B":
        r = _read(ds, "thp_baseline", pred_dir)
        w = _read(ds, "thp_baseline_zilm", pred_dir)
        assert np.array_equal(r["point"], w["point"]) and np.array_equal(r["gap"], w["gap"])
        for k in ("mean", "median", "eps_mode_grid"):
            r[k] = w[k]
        return r
    return _read(ds, {"SuTraN": "sutran", "ED-LSTM": "edlstm"}[model], pred_dir)


def hits(r, pred, eps):
    """Per-instance joint hit A_n * T_n."""
    return r["correct"] & (np.abs(pred - r["gap"]) < eps)


def jpa(r, pred, eps=C.EPS_STAR):
    """JPA_eps in percent."""
    return 100.0 * hits(r, pred, eps).mean()


def decode(r, name, k=C.EPS_STAR_IDX):
    """Point prediction of decode `name`; eps-mode is matched to grid tolerance k."""
    return r["eps_mode_grid"][k] if name == "eps_mode" else r[name]


def null_hits(r, eps):
    """B2: the model's own activity predictions with tau_hat = 0."""
    return r["correct"] & (r["gap"] < eps)


def skill(j, j_null):
    """Skill score of Eq. (4), both inputs and output in percent."""
    return 100.0 * (j - j_null) / (100.0 - j_null)


def cmae(r, pred):
    """cMAE of Eq. (7) in days, over the instances with a correct activity."""
    c = r["correct"]
    return float(np.abs(pred[c] - r["gap"][c]).mean())
