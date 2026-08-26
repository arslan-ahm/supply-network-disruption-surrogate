"""Surrogate fidelity against simulator ground truth.

Two families, and the distinction between them is the most important thing a
reader of this repository needs to hold onto.

**Magnitude fidelity** (MAE, RMSE, per-row relative error) says whether the
surrogate reproduces *how much* service is lost. It is what you need if the
surrogate's output goes into a cost calculation.

**Rank fidelity** (Spearman, Kendall, top-k agreement) says whether it reproduces
the *ordering*. It is what you need for screening, which is the use case this
project actually argues for: a planner asks "which link should I look at first",
and a surrogate that gets every magnitude wrong by a factor of two but the
ordering right answers that question perfectly.

The two can diverge sharply and this module reports both, unaggregated, so a
reader can see which one was achieved. A single "accuracy" number would hide it.

Rank correlations are computed **within scenario** by default and then averaged,
not pooled across scenarios. Pooling lets an easy between-scenario signal (big
disruptions hurt more than small ones) inflate the correlation, and the question
of interest is which demand point inside a given scenario is worst hit.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {x.shape} vs {y.shape}")
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error over rows where both are finite."""
    p, t = _finite_pair(pred, true)
    return float(np.abs(p - t).mean()) if p.size else float("nan")


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Root mean squared error over rows where both are finite."""
    p, t = _finite_pair(pred, true)
    return float(np.sqrt(((p - t) ** 2).mean())) if p.size else float("nan")


def bias(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean signed error. Positive means the surrogate over-predicts impact.

    Worth reporting separately from MAE because the two failure modes have
    opposite consequences: a surrogate that systematically over-states impact
    wastes mitigation budget, and one that under-states it misses the outage.
    """
    p, t = _finite_pair(pred, true)
    return float((p - t).mean()) if p.size else float("nan")


def r2(pred: np.ndarray, true: np.ndarray) -> float:
    """Coefficient of determination.

    Returns NaN when the truth has zero variance — which happens on a split where
    every scenario was absorbed. A hard-coded 0.0 there would look like a failed
    model rather than an undefined metric.
    """
    p, t = _finite_pair(pred, true)
    if p.size < 2:
        return float("nan")
    denom = ((t - t.mean()) ** 2).sum()
    if denom <= 0:
        return float("nan")
    return float(1.0 - ((p - t) ** 2).sum() / denom)


def spearman(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman rank correlation. NaN when either side is constant."""
    p, t = _finite_pair(pred, true)
    if p.size < 3 or np.all(p == p[0]) or np.all(t == t[0]):
        return float("nan")
    return float(stats.spearmanr(p, t).statistic)


def kendall_tau(pred: np.ndarray, true: np.ndarray) -> float:
    """Kendall's tau-b. Slower than Spearman but robust to heavy ties.

    The zero-inflated target produces a lot of ties, and Spearman's average-rank
    handling of ties is optimistic where tau-b's is not, so both are reported.
    """
    p, t = _finite_pair(pred, true)
    if p.size < 3 or np.all(p == p[0]) or np.all(t == t[0]):
        return float("nan")
    return float(stats.kendalltau(p, t, variant="b").statistic)


def grouped_spearman(
    pred: np.ndarray, true: np.ndarray, group: np.ndarray, min_size: int = 3
) -> tuple[float, int, np.ndarray]:
    """Mean within-group Spearman correlation.

    Args:
        pred: Predictions.
        true: Ground truth.
        group: Group label per row, e.g. scenario id.
        min_size: Groups smaller than this are skipped, since a correlation over
            two points is either +1 or -1 and carries no information.

    Returns:
        ``(mean, n_groups_contributing, per_group_values)``. The count is returned
        because a mean over 40 of 900 groups is a different claim from a mean over
        all 900, and the tables print both.
    """
    vals = []
    for g in np.unique(group):
        m = group == g
        if int(m.sum()) < min_size:
            continue
        r = spearman(pred[m], true[m])
        if np.isfinite(r):
            vals.append(r)
    arr = np.asarray(vals, dtype=np.float64)
    return (float(arr.mean()) if arr.size else float("nan"), int(arr.size), arr)


def top1_agreement(
    pred: np.ndarray, true: np.ndarray, group: np.ndarray
) -> tuple[float, int]:
    """Fraction of groups where the highest-predicted row is truly the highest.

    A blunt but very interpretable screening metric: "if I look at the one demand
    point the surrogate says is worst hit, is it?" Groups where the truth is
    entirely tied (every demand point unaffected) are excluded, because there is
    no correct answer to agree with.

    Returns:
        ``(fraction, n_groups_contributing)``.
    """
    hits, n = 0, 0
    for g in np.unique(group):
        m = group == g
        t = true[m]
        if not np.isfinite(t).any() or np.allclose(t, t[np.isfinite(t)][0]):
            continue
        n += 1
        if int(np.argmax(pred[m])) in set(np.flatnonzero(t == np.nanmax(t)).tolist()):
            hits += 1
    return (hits / n if n else float("nan"), n)


def trajectory_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Errors on the excess-unmet trajectory.

    ``peak_period_error`` is the absolute difference in the period at which the
    shortage peaks, over rows where the true trajectory is not all zero. That is
    the trajectory number a planner would actually read off — when it gets worst —
    and it is not implied by a good mean absolute error.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    k = min(p.shape[1], t.shape[1])
    p, t = p[:, :k], t[:, :k]
    active = t.sum(axis=1) > 1e-9
    out = {
        "traj_mae": float(np.abs(p - t).mean()),
        "traj_rmse": float(np.sqrt(((p - t) ** 2).mean())),
        "traj_rows": int(p.shape[0]),
        "traj_active_rows": int(active.sum()),
    }
    if active.any():
        out["traj_mae_active"] = float(np.abs(p[active] - t[active]).mean())
        out["peak_period_error"] = float(
            np.abs(np.argmax(p[active], axis=1) - np.argmax(t[active], axis=1)).mean()
        )
        out["traj_total_rel_error"] = float(
            np.abs(p[active].sum(1) - t[active].sum(1)).sum()
            / max(t[active].sum(), 1e-9)
        )
    else:
        out["traj_mae_active"] = float("nan")
        out["peak_period_error"] = float("nan")
        out["traj_total_rel_error"] = float("nan")
    return out


def timing_metrics(
    tti_pred: np.ndarray,
    tti_true: np.ndarray,
    rec_pred: np.ndarray,
    rec_true: np.ndarray,
) -> dict[str, float]:
    """Time-to-impact and recovery-time errors, on defined rows only.

    Both counts are reported. In this dataset only a minority of
    (scenario, demand point) rows have a defined time-to-impact — most demand
    points are simply never affected — so a timing MAE quoted without its ``n``
    is close to meaningless.
    """
    out: dict[str, float] = {}
    for name, p, t in (
        ("time_to_impact", tti_pred, tti_true),
        ("recovery_time", rec_pred, rec_true),
    ):
        pp, tt = _finite_pair(p, t)
        out[f"{name}_mae"] = float(np.abs(pp - tt).mean()) if pp.size else float("nan")
        out[f"{name}_n"] = int(pp.size)
        out[f"{name}_spearman"] = spearman(pp, tt)
    return out


def fidelity_report(
    pred: np.ndarray,
    true: np.ndarray,
    group: np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, float]:
    """Every scalar fidelity metric for one set of impact predictions."""
    out = {
        f"{prefix}mae": mae(pred, true),
        f"{prefix}rmse": rmse(pred, true),
        f"{prefix}bias": bias(pred, true),
        f"{prefix}r2": r2(pred, true),
        f"{prefix}spearman_pooled": spearman(pred, true),
        f"{prefix}kendall_pooled": kendall_tau(pred, true),
        f"{prefix}n": int(np.isfinite(np.asarray(true, dtype=np.float64)).sum()),
    }
    if group is not None:
        m, n, _ = grouped_spearman(pred, true, group)
        out[f"{prefix}spearman_within_scenario"] = m
        out[f"{prefix}spearman_groups"] = n
        frac, n1 = top1_agreement(pred, true, group)
        out[f"{prefix}top1_agreement"] = frac
        out[f"{prefix}top1_groups"] = n1
    return out
