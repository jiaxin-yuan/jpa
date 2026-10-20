"""THP-M and THP-B per-instance predictions on S (plus THP-B's wrapper) -> results/predictions."""
import os
import sys

import numpy as np
import torch

import config as C
import wrapper
import zilm

sys.path.insert(0, os.path.join(C.ROOT, "models", "thp"))
from preprocess.dataset import df_to_dict, get_dataloader   # noqa: E402
from transformer.model import Transformer, get_non_pad_mask  # noqa: E402

N_MIX = 5
BATCH_SIZE = 64
ULP_MULT = 8               # row check tolerance: float32 ULPs of the largest timestamp


def build_model(num_types, kind, device):
    """THP encoder (4 layers, 4 heads, d=36) with a mixture or a point time head."""
    return Transformer(num_types=num_types, d_model=36, d_rnn=256, d_inner=128, n_layers=4,
                       n_head=4, d_k=16, d_v=16, dropout=0.1, n_mix=N_MIX,
                       model_type=kind).to(device)


def load_model(ds, kind, num_types, device):
    """Load the checkpoint selected by best validation next-event MAE."""
    name = f"fold0_variation0_{ds}_{kind}_best_model_best_mae.pth"
    state = torch.load(os.path.join(C.THP_CKPT, name), map_location=device)
    model = build_model(num_types, kind, device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    ts = state.get("time_stats")
    return model, (ts["dt_mean"], ts["dt_std"]) if ts else (0.0, 1.0)


def forward(model, loader, kind, device, dt_stats):
    """Teacher-forced pass: position t predicts t+1; returns arrays over valid positions."""
    acc = {k: [] for k in ("gap", "correct", "point", "pi", "log_w", "mu", "sigma")}
    max_time = 0.0
    with torch.no_grad():
        for batch in loader:
            event_time, _, _, event_type = (x.to(device) for x in batch)
            max_time = max(max_time, float(event_time.max()))
            non_pad = get_non_pad_mask(event_type).squeeze(-1)
            _, heads = model(event_type, event_time)
            vm = ((non_pad[:, 1:] > 0) & (event_type[:, 1:] > 0)).reshape(-1)
            if kind == "mixture":
                tie_logit, time_params, mark_logit = heads
                prm = zilm.extract_params(tie_logit[:, :-1], time_params[:, :-1], N_MIX)
                for key, x in zip(("pi", "log_w", "mu", "sigma"), prm):
                    x = x.reshape(-1) if key == "pi" else x.reshape(-1, N_MIX)
                    acc[key].append(x[vm].double().cpu())
            else:
                mark_logit = heads[2]
                pt = (heads[0].squeeze(-1) * dt_stats[1] + dt_stats[0])[:, :-1]
                acc["point"].append(pt.reshape(-1)[vm].cpu())
            gt_mark = event_type[:, 1:] - 1
            acc["gap"].append((event_time[:, 1:] - event_time[:, :-1]).reshape(-1)[vm].cpu())
            acc["correct"].append((mark_logit[:, :-1, :].argmax(-1) == gt_mark)
                                  .reshape(-1)[vm].cpu())
    return {k: torch.cat(v) for k, v in acc.items() if v}, max_time


def check_rows(ds, gap, max_time):
    """Pin every row to its (case, prefix length) key: true gaps must equal the keyed gaps."""
    rec = np.load(os.path.join(C.KEYS, f"{ds}_keys.npz"))["recon_gap"].astype(np.float64)
    tol = ULP_MULT * float(np.spacing(np.float32(max_time)))
    assert len(rec) == len(gap) and (np.abs(rec - gap) <= tol).all(), "row order changed"


def save(ds, tag, pop, correct, gap, point, dens=None):
    """Write one per-instance file restricted to S (float32, as in the original export)."""
    f32 = lambda x: np.asarray(x).astype(np.float32)  # noqa: E731
    out = {"correct": correct.astype(np.int8), "true_gap": f32(gap), "pred_point": f32(point),
           "case_id": pop["case_id"], "prefix_len": pop["prefix_len"],
           "has_density": np.int8(dens is not None)}
    if dens is not None:
        out.update(pred_mean=f32(dens[0]), pred_median=f32(dens[1]),
                   pred_eps_mode=f32(dens[2]))
    os.makedirs(os.path.join(C.PRED, ds), exist_ok=True)
    np.savez_compressed(os.path.join(C.PRED, ds, f"{tag}.npz"), **out)


def run(ds, device):
    """Export thp_mixture, thp_baseline and thp_baseline_zilm for one log."""
    tr, va, te = df_to_dict(directory=C.THP_DATA, fold_filename=f"fold0_variation0_{ds}")
    loaders = {s: get_dataloader(d[s], batch_size=BATCH_SIZE, shuffle=False)
               for s, d in (("val", va), ("test", te))}
    pop = np.load(os.path.join(C.POP, f"{ds}.npz"), allow_pickle=True)
    idx = pop["thp_row"]

    model, st = load_model(ds, "mixture", tr["dim_process"], device)
    a, mt = forward(model, loaders["test"], "mixture", device, st)
    gap = a["gap"].numpy().astype(np.float64)
    check_rows(ds, gap, mt)
    prm = tuple(a[k][idx] for k in ("pi", "log_w", "mu", "sigma"))
    dens = wrapper.decode(prm, C.EPS_GRID, device)
    save(ds, "thp_mixture", pop, a["correct"].numpy()[idx], gap[idx], dens[0], dens)

    model, st = load_model(ds, "baseline_act", tr["dim_process"], device)
    a, mt = forward(model, loaders["test"], "baseline_act", device, st)
    gap = a["gap"].numpy().astype(np.float64)
    check_rows(ds, gap, mt)
    point = a["point"].numpy().astype(np.float32).astype(np.float64)[idx]
    correct = a["correct"].numpy()[idx]
    save(ds, "thp_baseline", pop, correct, gap[idx], point)

    v, _ = forward(model, loaders["val"], "baseline_act", device, st)
    fit = wrapper.fit(v["point"].numpy().astype(np.float64), v["gap"].numpy().astype(np.float64))
    dens = wrapper.decode(wrapper.params(point, *fit), C.EPS_GRID, device)
    save(ds, "thp_baseline_zilm", pop, correct, gap[idx], point, dens)
    print(f"{ds}: THP-M, THP-B, THP-B+wrapper on |S| = {len(idx)} "
          f"(wrapper pi={fit[0]:.3f} delta={fit[1]:+.3f} s={fit[2]:.3f})")


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in sys.argv[1:] or C.LOGS:
        run(d, dev)
