"""Post-hoc density of Eq. (6) for a frozen point model, and chunked mean/median/eps-mode decoding."""
import numpy as np
import torch

import zilm

T_FLOOR_DAYS = 1e-6        # clamp before log of a point prediction
TIE_THRESH_DAYS = 1e-6     # |tau| below this counts as tau = 0
DECODE_CHUNK = 20000


def fit(pred_val, gap_val):
    """Fit (pi_hat, delta_hat, s_hat) on validation: tie rate, mean/std of log-residuals."""
    tie = np.abs(gap_val) < TIE_THRESH_DAYS
    mu = np.log(np.clip(pred_val[~tie], T_FLOOR_DAYS, None))
    resid = np.log(np.clip(gap_val[~tie], T_FLOOR_DAYS, None)) - mu
    return float(tie.mean()), float(resid.mean()), float(max(resid.std(), 1e-3))


def params(pred, pi_hat, delta_hat, s_hat):
    """Per-instance K=1 ZILM parameters (pi, log_w, mu, sigma) around point predictions."""
    n = len(pred)
    return (torch.full((n,), pi_hat, dtype=torch.float64),
            torch.zeros(n, 1, dtype=torch.float64),
            torch.tensor(np.log(np.clip(pred, T_FLOOR_DAYS, None)) + delta_hat,
                         dtype=torch.float64).unsqueeze(-1),
            torch.full((n, 1), s_hat, dtype=torch.float64))


def decode(prm, eps_list, device="cpu"):
    """Mean, median and eps-mode (one row per tolerance) of a ZILM, as float64 numpy."""
    pi, log_w, mu, sigma = prm
    n = pi.shape[0]
    mean, median = np.empty(n), np.empty(n)
    modes = np.empty((len(eps_list), n))
    for lo in range(0, n, DECODE_CHUNK):
        hi = min(lo + DECODE_CHUNK, n)
        p, w, m, s = (x[lo:hi].to(device) for x in (pi, log_w, mu, sigma))
        mean[lo:hi] = zilm.zilm_mean(p, w, m, s).cpu().numpy()
        median[lo:hi] = zilm.quantile(0.5, p, w, m, s).cpu().numpy()
        for k, e in enumerate(eps_list):
            modes[k, lo:hi] = zilm.eps_mode(float(e), p, w, m, s).cpu().numpy()
    return mean, median, modes
