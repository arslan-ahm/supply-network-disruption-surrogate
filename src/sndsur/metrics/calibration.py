"""Calibration and sharpness of the surrogate's predictive intervals.

A surrogate used for planning must be able to say when it does not know. That
requires two things, and reporting either alone is misleading:

**Coverage.** Does a nominal 90% interval contain the truth 90% of the time? An
interval that covers 99% at the 90% level is not "safe", it is uninformative.

**Sharpness.** How wide is it? A model can achieve perfect coverage by predicting
``[0, 1]`` everywhere, so width is reported alongside coverage in every table.

Two interval constructions are measured, because they make different assumptions
and the target distribution violates one of them:

* **Gaussian**, ``mu +- z * sigma``, from the heteroscedastic head (and, for the
  ensemble, from the combined predictive variance of Lakshminarayanan et al.,
  2017). Assumes symmetry, which a target with a spike at zero and a right tail
  does not have.
* **Quantile**, from the pinball-trained head, which assumes nothing about shape
  but must learn each level separately.

Beyond intervals, the *ordering* property matters most for the out-of-distribution
claim: does error grow with predicted uncertainty? That is what
:func:`sparsification` and :func:`error_uncertainty_auroc` measure, and it is the
claim the shifted-topology tables are built to test.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _z(level: float) -> float:
    """Two-sided normal quantile for a nominal coverage ``level``."""
    return float(stats.norm.ppf(0.5 + level / 2.0))


def gaussian_interval(
    mu: np.ndarray, sigma: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric Gaussian interval, clipped at zero below.

    The clip is not cosmetic: service-level loss is non-negative, so the part of
    a Gaussian interval below zero is not a possible outcome and counting it as
    width would flatter the sharpness numbers.
    """
    z = _z(level)
    lo = np.clip(mu - z * sigma, 0.0, None)
    hi = mu + z * sigma
    return lo, hi


def coverage_and_width(
    lo: np.ndarray, hi: np.ndarray, true: np.ndarray
) -> tuple[float, float, int]:
    """Empirical coverage, mean width, and contributing row count."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    m = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(t)
    if not m.any():
        return (float("nan"), float("nan"), 0)
    inside = (t[m] >= lo[m]) & (t[m] <= hi[m])
    return (float(inside.mean()), float((hi[m] - lo[m]).mean()), int(m.sum()))


def calibration_table(
    mu: np.ndarray,
    sigma: np.ndarray,
    true: np.ndarray,
    levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
    label: str = "gaussian",
) -> list[dict[str, float | str]]:
    """One row per nominal level with coverage, width and the coverage gap."""
    rows: list[dict[str, float | str]] = []
    for lv in levels:
        lo, hi = gaussian_interval(mu, sigma, lv)
        cov, width, n = coverage_and_width(lo, hi, true)
        rows.append({
            "interval": label,
            "nominal": lv,
            "coverage": cov,
            "width": width,
            "coverage_gap": cov - lv,
            "n": n,
        })
    return rows


def quantile_calibration(
    q_pred: np.ndarray, levels: tuple[float, ...], true: np.ndarray
) -> list[dict[str, float | str]]:
    """Per-level exceedance check plus the interval formed by the outer levels.

    For each level ``tau`` the fraction of truths at or below the predicted
    quantile should be ``tau``. The central interval between the lowest and
    highest level is then reported with the same coverage/width pair as the
    Gaussian table so the two are directly comparable.
    """
    rows: list[dict[str, float | str]] = []
    q = np.asarray(q_pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != len(levels) or q.shape[0] != t.shape[0]:
        return rows
    for j, lv in enumerate(levels):
        m = np.isfinite(q[:, j]) & np.isfinite(t)
        rows.append({
            "interval": "quantile_level",
            "nominal": lv,
            "coverage": float((t[m] <= q[m, j]).mean()) if m.any() else float("nan"),
            "width": float("nan"),
            "coverage_gap": (float((t[m] <= q[m, j]).mean()) - lv) if m.any() else float("nan"),
            "n": int(m.sum()),
        })
    if len(levels) >= 2:
        lo, hi = q[:, 0], q[:, -1]
        # Quantile heads can cross (nothing in the pinball loss forbids it), so
        # the interval is built from the sorted pair rather than assuming order.
        low = np.minimum(lo, hi)
        high = np.maximum(lo, hi)
        cov, width, n = coverage_and_width(np.clip(low, 0.0, None), high, t)
        nominal = levels[-1] - levels[0]
        rows.append({
            "interval": "quantile_central",
            "nominal": nominal,
            "coverage": cov,
            "width": width,
            "coverage_gap": cov - nominal,
            "n": n,
            "crossing_fraction": float((lo > hi).mean()),
        })
    return rows


def error_uncertainty_auroc(
    error: np.ndarray, sigma: np.ndarray, quantile: float = 0.80
) -> tuple[float, int]:
    """AUROC of predicted ``sigma`` for detecting the largest errors.

    Rows whose absolute error is above the ``quantile`` of the error distribution
    are the positives. This asks the practical question directly: if I flag the
    predictions the model is least sure about, do I catch the ones it got wrong?

    Computed from the Mann-Whitney U statistic, which is exactly the AUROC and
    avoids a threshold sweep.

    Returns:
        ``(auroc, n_positive)``. NaN when either class is empty.
    """
    e = np.abs(np.asarray(error, dtype=np.float64))
    s = np.asarray(sigma, dtype=np.float64)
    m = np.isfinite(e) & np.isfinite(s)
    e, s = e[m], s[m]
    if e.size < 4:
        return (float("nan"), 0)
    thresh = np.quantile(e, quantile)
    pos = e > thresh
    if pos.all() or not pos.any():
        return (float("nan"), int(pos.sum()))
    u = stats.mannwhitneyu(s[pos], s[~pos], alternative="two-sided").statistic
    return (float(u / (pos.sum() * (~pos).sum())), int(pos.sum()))


def sparsification(
    error: np.ndarray, sigma: np.ndarray, n_steps: int = 20
) -> dict[str, float | list[float]]:
    """Sparsification curve and AUSE.

    Rows are discarded in order of decreasing predicted ``sigma`` and the mean
    absolute error of what remains is recorded. A useful uncertainty estimate
    makes that curve fall quickly. The **oracle** curve discards in order of
    decreasing actual error, giving the best achievable, and AUSE is the area
    between the two — a scale-free number where 0 is perfect.

    Returns:
        ``fractions``, ``curve``, ``oracle``, ``random``, ``ause`` and ``n``.
    """
    e = np.abs(np.asarray(error, dtype=np.float64))
    s = np.asarray(sigma, dtype=np.float64)
    m = np.isfinite(e) & np.isfinite(s)
    e, s = e[m], s[m]
    n = e.size
    if n < 4:
        return {"fractions": [], "curve": [], "oracle": [], "random": [],
                "ause": float("nan"), "n": int(n)}
    fr = np.linspace(0.0, 0.8, n_steps)
    by_sigma = np.argsort(-s, kind="stable")
    by_error = np.argsort(-e, kind="stable")
    curve, oracle = [], []
    for f in fr:
        drop = int(round(f * n))
        keep_s = by_sigma[drop:]
        keep_o = by_error[drop:]
        curve.append(float(e[keep_s].mean()) if keep_s.size else float("nan"))
        oracle.append(float(e[keep_o].mean()) if keep_o.size else float("nan"))
    base = curve[0] if curve[0] > 0 else 1.0
    c = np.asarray(curve) / base
    o = np.asarray(oracle) / base
    return {
        "fractions": fr.tolist(),
        "curve": curve,
        "oracle": oracle,
        "random": [float(e.mean())] * len(fr),
        "ause": float(np.trapezoid(c - o, fr) / max(fr[-1], 1e-9)),
        "n": int(n),
    }


def uncertainty_shift_report(
    mu: np.ndarray, sigma: np.ndarray, true: np.ndarray
) -> dict[str, float]:
    """Summary of how well ``sigma`` tracks error on one split.

    ``sigma_error_spearman`` is the headline: a positive value means the model's
    uncertainty rises where its error rises, which is the property that makes an
    out-of-distribution warning trustworthy. Reported as a rank correlation
    because the two quantities are on different scales and only the ordering is
    claimed.
    """
    err = np.abs(np.asarray(mu, dtype=np.float64) - np.asarray(true, dtype=np.float64))
    s = np.asarray(sigma, dtype=np.float64)
    m = np.isfinite(err) & np.isfinite(s)
    auroc, n_pos = error_uncertainty_auroc(err, s)
    sp = (
        float(stats.spearmanr(s[m], err[m]).statistic)
        if m.sum() >= 3 and not np.all(s[m] == s[m][0])
        else float("nan")
    )
    sparse = sparsification(err, s)
    return {
        "mean_sigma": float(s[m].mean()) if m.any() else float("nan"),
        "median_sigma": float(np.median(s[m])) if m.any() else float("nan"),
        "mean_abs_error": float(err[m].mean()) if m.any() else float("nan"),
        "sigma_error_spearman": sp,
        "error_detection_auroc": auroc,
        "n_high_error": n_pos,
        "ause": float(sparse["ause"]),
        "n": int(m.sum()),
    }
