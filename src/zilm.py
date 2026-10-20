"""Zero-inflated LogNormal mixture (Eq. 5): parameters, mean, median and eps-mode decodes.

Copied unchanged from the experiment code (generative_eval.py).
"""
import math

import torch
import torch.nn.functional as F

_SQRT2 = math.sqrt(2.0)


def extract_params(tie_logit, time_params, K):
    """Head outputs -> (pi [.], log_w [., K], mu [., K], sigma [., K])."""
    pi = torch.sigmoid(tie_logit)
    log_w = F.log_softmax(time_params[..., :K], dim=-1)
    mu = time_params[..., K:2 * K]
    sigma = F.softplus(time_params[..., 2 * K:]) + 1e-4
    return pi, log_w, mu, sigma


def cont_cdf(t, log_w, mu, sigma):
    """CDF of the continuous part at t (>0); 0 for t <= 0.

    t may be [N] (one point per event) or [N, C] (C candidates per event,
    with params [N, K]).
    """
    if t.dim() == mu.dim():                       # [N, C] vs [N, K]
        log_w, mu, sigma = (x.unsqueeze(-2) for x in (log_w, mu, sigma))
    z = (torch.log(t.clamp(min=1e-12)).unsqueeze(-1) - mu) / sigma
    phi = 0.5 * (1.0 + torch.erf(z / _SQRT2))
    c = (log_w.exp() * phi).sum(-1)
    return torch.where(t > 0, c, torch.zeros_like(c))


def zilm_cdf(t, pi, log_w, mu, sigma):
    """Full CDF F(t) = pi * 1[t >= 0] + (1 - pi) * cont_cdf(t)."""
    c = cont_cdf(t, log_w, mu, sigma)
    if t.dim() > pi.dim():
        pi = pi.unsqueeze(-1)
    return torch.where(t >= 0, pi + (1.0 - pi) * c, torch.zeros_like(c))


def zilm_mean(pi, log_w, mu, sigma):
    """E[dt] = (1 - pi) * sum_k w_k exp(mu_k + sigma_k^2 / 2) (legacy decode)."""
    comp = torch.exp((mu + 0.5 * sigma ** 2).clamp(max=20.0))
    return (1.0 - pi) * (log_w.exp() * comp).sum(-1)


def quantile(q, pi, log_w, mu, sigma, iters=60):
    """Quantile of the ZILM, vectorized [N]. q: scalar or [N].

    Returns 0 where q <= pi (the atom absorbs the quantile); otherwise solves
    cont_cdf(t) = (q - pi) / (1 - pi) by bisection in log-t.
    """
    q = torch.as_tensor(q, dtype=pi.dtype, device=pi.device).expand_as(pi)
    target = ((q - pi) / (1.0 - pi).clamp(min=1e-12)).clamp(min=0.0, max=1.0 - 1e-9)
    lo = (mu - 10.0 * sigma).min(dim=-1).values                     # log-t bracket
    hi = (mu + 10.0 * sigma).max(dim=-1).values
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        go_right = cont_cdf(mid.exp(), log_w, mu, sigma) < target
        lo = torch.where(go_right, mid, lo)
        hi = torch.where(go_right, hi, mid)
    t = (0.5 * (lo + hi)).exp()
    return torch.where(q <= pi, torch.zeros_like(t), t)


def eps_ball_mass(t, eps, pi, log_w, mu, sigma):
    """P(|dt - t| < eps) = (1-pi)[C(t+eps) - C(max(t-eps, 0))] + pi 1[t < eps].

    t: [N] or [N, C]; eps: scalar (days).
    """
    upper = cont_cdf(t + eps, log_w, mu, sigma)
    lower = cont_cdf((t - eps).clamp(min=0.0), log_w, mu, sigma)
    if t.dim() > pi.dim():
        pi = pi.unsqueeze(-1)
    return (1.0 - pi) * (upper - lower) + pi * (t < eps)


def eps_mode(eps, pi, log_w, mu, sigma, n_grid=49, refine=3):
    """Decode t-hat = argmax_t P(|dt - t| < eps): the JEA-optimal point estimate.

    Coarse candidates per event: 0, the K component modes exp(mu - sigma^2),
    the K component medians exp(mu), the ZILM mean and median, and an n_grid
    log-spaced grid spanning [min(mu - 3 sigma), max(mu + 3 sigma)] in log-t.
    The best candidate is then refined by `refine` rounds of local log-space
    grid search (the ball mass is sharply peaked when eps is small relative to
    the density scale, so a coarse grid alone can miss the maximizer).
    Including mean and median guarantees the decode never scores below either.
    """
    zero = torch.zeros_like(pi).unsqueeze(-1)                       # [N, 1]
    modes = torch.exp(mu - sigma ** 2)                              # [N, K]
    medians = torch.exp(mu)                                         # [N, K]
    extra = torch.stack(
        [zilm_mean(pi, log_w, mu, sigma), quantile(0.5, pi, log_w, mu, sigma)],
        dim=-1)                                                     # [N, 2]
    glo = (mu - 3.0 * sigma).min(dim=-1, keepdim=True).values       # [N, 1]
    ghi = (mu + 3.0 * sigma).max(dim=-1, keepdim=True).values
    steps = torch.linspace(0.0, 1.0, n_grid, dtype=pi.dtype, device=pi.device)
    grid = torch.exp(glo + (ghi - glo) * steps)                     # [N, G]
    cands = torch.cat([zero, modes, medians, extra, grid], dim=-1)  # [N, 3+2K+G]
    mass = eps_ball_mass(cands, eps, pi, log_w, mu, sigma)
    idx = mass.argmax(dim=-1, keepdim=True)
    best_t = cands.gather(-1, idx).squeeze(-1)                      # [N]

    spacing = (ghi - glo).squeeze(-1) / (n_grid - 1)                # [N] log-t step
    local_steps = torch.linspace(-1.0, 1.0, 17, dtype=pi.dtype, device=pi.device)
    for _ in range(refine):
        center = torch.log(best_t.clamp(min=1e-12)).unsqueeze(-1)
        local = torch.exp(center + spacing.unsqueeze(-1) * local_steps)
        # keep the incumbent so refinement can only improve; skip refining t=0
        local = torch.where(best_t.unsqueeze(-1) > 0, local, torch.zeros_like(local))
        cand2 = torch.cat([best_t.unsqueeze(-1), local], dim=-1)
        m2 = eps_ball_mass(cand2, eps, pi, log_w, mu, sigma)
        idx2 = m2.argmax(dim=-1, keepdim=True)
        best_t = cand2.gather(-1, idx2).squeeze(-1)
        spacing = spacing * (2.0 / 16.0)
    return best_t
