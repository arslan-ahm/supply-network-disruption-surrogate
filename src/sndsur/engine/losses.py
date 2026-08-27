"""Loss terms for the surrogate.

Three ideas govern the choices here.

**Masking, not imputation.** Time-to-impact and recovery time are genuinely
undefined for a demand point the disruption never reached. Filling those in with
zero would teach the model that every unaffected point is hit immediately, and
would silently change what the reported timing error means. Every timing term is
masked and reports how many rows contributed.

**Heteroscedastic NLL for the interval.** The impact head is trained with
Gaussian NLL over a predicted log-variance (Nix & Weigend, 1994; used for
ensembles by Lakshminarayanan et al., 2017). The point prediction also gets a
plain L1 term, because pure NLL lets the model reduce loss by inflating variance
on hard rows instead of fitting them, and the point estimate is what the ranking
uses.

**L1 rather than L2 on impact.** The label distribution has a large mass at
exactly zero (most disruptions are absorbed) and a long right tail. Squared error
puts nearly all its gradient on the tail, and the first version of this model
trained that way produced a good MAE on big events and ranked the bottom
two-thirds of nodes at random — which is useless for screening, since screening
is exactly the job of separating small from zero.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Log-variance clamp. Below the floor the NLL gradient explodes on a row the
#: model happens to fit exactly; above the ceiling the model can silence any row
#: by declaring it unpredictable.
LOGVAR_MIN, LOGVAR_MAX = -8.0, 4.0


def masked_l1(pred: Tensor, target: Tensor) -> tuple[Tensor, int]:
    """L1 over rows where ``target`` is finite.

    Returns:
        ``(loss, n_contributing)``. The loss is a zero tensor that still carries
        a graph when nothing contributes, so the caller can add it unconditionally.
    """
    mask = torch.isfinite(target)
    n = int(mask.sum().item())
    if n == 0:
        return (pred.sum() * 0.0, 0)
    return ((pred[mask] - target[mask]).abs().mean(), n)


def gaussian_nll(pred: Tensor, logvar: Tensor, target: Tensor) -> Tensor:
    """Mean Gaussian negative log-likelihood, up to an additive constant.

    ``0.5 * (logvar + (y - mu)^2 / exp(logvar))``. The ``log(2*pi)`` term is
    dropped: it is constant, so it changes the printed number but not the
    gradient, and leaving it out keeps the reported NLL comparable only within
    this repository — which is stated wherever the number appears.
    """
    lv = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
    return (0.5 * (lv + (target - pred) ** 2 / lv.exp())).mean()


def quantile_loss(pred: Tensor, target: Tensor, levels: tuple[float, ...]) -> Tensor:
    """Pinball loss summed over ``levels``.

    Args:
        pred: ``(R, Q)`` predictions, one column per level.
        target: ``(R,)`` truth.
        levels: The quantile levels, aligned with ``pred``'s columns.

    Quantile regression is the second uncertainty route measured in this project.
    It makes no distributional assumption, which matters because the impact
    distribution is a spike at zero plus a right tail and is nowhere near
    Gaussian — so the Gaussian interval is expected to be the worse-calibrated of
    the two, and the results table says whether it is.
    """
    if pred.numel() == 0:
        return pred.sum() * 0.0
    t = target.unsqueeze(1)
    q = torch.as_tensor(levels, dtype=pred.dtype, device=pred.device).unsqueeze(0)
    diff = t - pred
    return torch.maximum(q * diff, (q - 1.0) * diff).mean()


def trajectory_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 over the trajectory, weighted towards early periods.

    Weights decay as ``1 / sqrt(1 + t)``. The early periods are what a planner
    acts on — a shortage six periods out is actionable and one twenty-four
    periods out is a forecast — and the untapered version spent its capacity on
    the long flat tail, where the target is almost always zero.
    """
    if pred.numel() == 0:
        return pred.sum() * 0.0
    p = min(pred.shape[1], target.shape[1])
    t_idx = torch.arange(p, dtype=pred.dtype, device=pred.device)
    w = 1.0 / torch.sqrt(1.0 + t_idx)
    w = w / w.mean()
    return ((pred[:, :p] - target[:, :p]).abs() * w.unsqueeze(0)).mean()


def surrogate_loss(out, batch, cfg) -> tuple[Tensor, dict[str, float]]:
    """Total loss and a dictionary of its parts.

    Args:
        out: A :class:`sndsur.models.surrogate.SurrogateOutput`.
        batch: A :class:`sndsur.engine.batching.Batch`.
        cfg: The full config; ``cfg.train`` supplies the weights.

    Returns:
        ``(total, parts)`` where ``parts`` also records how many rows
        contributed to each masked term, so ``history.jsonl`` shows whether a
        timing term was trained on 30 rows or 3000.
    """
    tc = cfg.train
    l_impact, _ = masked_l1(out.impact, batch.impact)
    l_traj = trajectory_loss(out.traj, batch.traj)
    # Timing targets are in *periods* (0-30), while impact is a fraction (0-1).
    # Left unscaled, the timing L1 came out around 12 against an impact L1 around
    # 0.05, so even at w_timing=0.2 more than 98% of the gradient went to the
    # auxiliary head and the headline target was effectively untrained. Dividing
    # both prediction and target by the horizon puts every term on a comparable
    # scale; the head still emits periods, so predictions stay directly readable.
    h = float(max(cfg.sim.horizon, 1))
    l_tti, n_tti = masked_l1(out.time_to_impact / h, batch.tti / h)
    l_rec, n_rec = masked_l1(out.recovery_time / h, batch.recovery / h)

    total = tc.w_impact * l_impact + tc.w_traj * l_traj + tc.w_timing * (l_tti + l_rec)
    parts = {
        "l_impact": float(l_impact.detach()),
        "l_traj": float(l_traj.detach()),
        "l_tti": float(l_tti.detach()),
        "l_recovery": float(l_rec.detach()),
        "n_tti": n_tti,
        "n_recovery": n_rec,
    }

    if out.logvar is not None and tc.w_nll > 0:
        l_nll = gaussian_nll(out.impact, out.logvar, batch.impact)
        total = total + tc.w_nll * l_nll
        parts["l_nll"] = float(l_nll.detach())
    if out.quantiles is not None and tc.w_quantile > 0:
        l_q = quantile_loss(out.quantiles, batch.impact, tuple(cfg.model.quantiles))
        total = total + tc.w_quantile * l_q
        parts["l_quantile"] = float(l_q.detach())

    parts["loss"] = float(total.detach())
    return total, parts
