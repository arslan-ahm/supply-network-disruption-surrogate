"""Statistical comparison of surrogate runs.

Reporting that one method reached MAE 0.0121 and another 0.0134 is not a result.
With a few thousand rows and a heavily zero-inflated target, that gap can be
entirely a matter of which seed the model was initialised with. This module
supplies the machinery for saying whether a difference is real, and one
distinction the rest of the repository leans on heavily:

**A paired per-row test conditions on one trained model per method.** It answers
"does this set of weights beat that set of weights consistently across rows?",
which is a real question with a real answer. It does *not* answer "is this method
better", because that claim treats the training run as the sampling unit and a
paired test over rows has an n of 1 in that unit. Both are computed here and kept
apart: :func:`compare` for the former, :func:`run_level_comparison` for the
latter.

**The noise scale.** Given the per-metric standard deviation ``sd`` across seeds
of an identical configuration, the run-to-run scale of a *difference* between two
single runs is ``sqrt(2) * sd``. Any improvement smaller than that is inside
noise, and :func:`noise_verdict` labels it so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class Interval:
    """A point estimate with a confidence interval."""

    estimate: float
    lower: float
    upper: float
    level: float = 0.95
    n: int = 0

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.lower:.4f}, {self.upper:.4f}]"

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class Comparison:
    """The outcome of comparing two methods on the same rows."""

    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    #: ``mean_a - mean_b``, with a bootstrap interval on the difference.
    difference: Interval
    #: Wilcoxon signed-rank p-value on the paired differences.
    p_value: float
    #: Paired Cohen's d — the mean difference in units of its own SD.
    effect_size: float
    n: int
    #: Set by :func:`holm_bonferroni`; ``None`` until corrected.
    p_adjusted: float | None = None

    @property
    def significant(self) -> bool:
        p = self.p_value if self.p_adjusted is None else self.p_adjusted
        return bool(np.isfinite(p) and p < 0.05)

    def summary(self) -> str:
        p = self.p_value if self.p_adjusted is None else self.p_adjusted
        marker = "*" if self.significant else " "
        return (
            f"{self.name_a} vs {self.name_b}: {self.mean_a:.4f} vs {self.mean_b:.4f}, "
            f"delta={self.difference}, p={p:.4g}{marker}, d={self.effect_size:.3f}"
        )

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "difference": self.difference.estimate,
            "ci_lower": self.difference.lower,
            "ci_upper": self.difference.upper,
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "effect_size": self.effect_size,
            "n": self.n,
            "significant": self.significant,
        }


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    statistic: str = "mean",
) -> Interval:
    """Percentile bootstrap interval for a summary of ``values``.

    Non-parametric, because none of the quantities here are close to normal: the
    per-row impact error is a spike at zero with a right tail, and Spearman
    correlations are bounded.

    Args:
        values: Per-item values. Non-finite entries are dropped, and ``n``
            records how many actually contributed.
        n_resamples: Bootstrap replicates.
        level: Coverage, e.g. 0.95.
        seed: RNG seed, so intervals are reproducible.
        statistic: ``"mean"`` or ``"median"``.
    """
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    reduce = np.mean if statistic == "mean" else np.median
    if data.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), level, 0)
    point = float(reduce(data))
    if data.size < 2:
        return Interval(point, point, point, level, int(data.size))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, data.size, size=(n_resamples, data.size))
    reps = reduce(data[picks], axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(reps, [alpha, 1.0 - alpha])
    return Interval(point, float(lo), float(hi), level, int(data.size))


def paired_bootstrap_difference(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 2000, level: float = 0.95, seed: int = 0
) -> Interval:
    """Bootstrap interval for the mean paired difference ``a - b``.

    Resamples row *indices*, not the two arrays independently, which preserves
    the pairing and gives the tighter interval the paired design earns.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"paired arrays must match in shape: {x.shape} vs {y.shape}")
    valid = np.isfinite(x) & np.isfinite(y)
    return bootstrap_ci((x - y)[valid], n_resamples, level, seed, "mean")


def compare(
    a: np.ndarray,
    b: np.ndarray,
    name_a: str = "a",
    name_b: str = "b",
    n_resamples: int = 2000,
    seed: int = 0,
) -> Comparison:
    """Full paired comparison of two methods on the same rows.

    Returns:
        A :class:`Comparison`. The p-value is NaN when every paired difference is
        exactly zero, where the signed-rank test is undefined — which is not the
        same as "not significant" and is not reported as such.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    diff = x - y
    if diff.size == 0 or np.allclose(diff, 0.0):
        p = float("nan")
    else:
        p = float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue)
    sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    effect = float(diff.mean() / sd) if sd > 0 else 0.0
    return Comparison(
        name_a=name_a,
        name_b=name_b,
        mean_a=float(x.mean()) if x.size else float("nan"),
        mean_b=float(y.mean()) if y.size else float("nan"),
        difference=paired_bootstrap_difference(x, y, n_resamples, seed=seed),
        p_value=p,
        effect_size=effect,
        n=int(diff.size),
    )


def holm_bonferroni(comparisons: list[Comparison], alpha: float = 0.05) -> list[Comparison]:
    """Apply the Holm-Bonferroni step-down correction in place.

    Sorts raw p-values ascending, multiplies the ``i``-th by ``m - i``, then
    enforces monotonicity. NaN p-values (undefined tests) are excluded from the
    family size, since they carry no evidence either way.
    """
    testable = [c for c in comparisons if np.isfinite(c.p_value)]
    m = len(testable)
    if m == 0:
        return comparisons
    order = sorted(range(m), key=lambda i: testable[i].p_value)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * testable[idx].p_value)
        running = max(running, adj)
        testable[idx].p_adjusted = running
    del alpha
    return comparisons


# --------------------------------------------------------------------------- #
# Run-level analysis
# --------------------------------------------------------------------------- #


@dataclass
class NoiseScale:
    """Seed-to-seed variability of one metric under a fixed configuration.

    Attributes:
        metric: Metric name.
        values: The per-seed values.
        sd: Sample standard deviation across seeds.
        spread: ``max - min``.
        diff_scale: ``sqrt(2) * sd`` — the scale on which a difference between
            two single runs is indistinguishable from noise.
    """

    metric: str
    values: list[float]
    sd: float
    spread: float
    diff_scale: float

    @property
    def n_seeds(self) -> int:
        return len(self.values)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "metric": self.metric,
            "n_seeds": self.n_seeds,
            "mean": float(np.mean(self.values)) if self.values else float("nan"),
            "sd": self.sd,
            "min": float(np.min(self.values)) if self.values else float("nan"),
            "max": float(np.max(self.values)) if self.values else float("nan"),
            "spread": self.spread,
            "diff_noise_scale": self.diff_scale,
        }


def noise_scale(metric: str, values: list[float] | np.ndarray) -> NoiseScale:
    """Compute the seed noise scale for one metric.

    Raises:
        ValueError: with fewer than two seeds. A single run has no measurable
            variability, and returning 0 would silently declare every difference
            significant.
    """
    v = [float(x) for x in np.asarray(values, dtype=np.float64) if np.isfinite(x)]
    if len(v) < 2:
        raise ValueError(f"need >= 2 seeds to estimate noise for {metric!r}, got {len(v)}")
    sd = float(np.std(v, ddof=1))
    return NoiseScale(metric, v, sd, float(max(v) - min(v)), float(np.sqrt(2.0) * sd))


def noise_verdict(gain: float, ns: NoiseScale, suggestive: float = 2.0) -> tuple[float, str]:
    """Place a measured gain against the run-to-run noise scale.

    Args:
        gain: The observed improvement, already signed so positive is better.
        ns: The noise scale for that metric.
        suggestive: Ratio above which a gain is called "survives" rather than
            "suggestive". 2.0 is roughly a two-sigma difference on the
            difference's own scale; it is a convention, stated rather than hidden.

    Returns:
        ``(ratio, verdict)`` with verdict one of ``"survives"``, ``"suggestive"``,
        ``"inside noise"``.
    """
    if not np.isfinite(gain) or ns.diff_scale <= 0:
        return (float("nan"), "not measured")
    ratio = float(gain / ns.diff_scale)
    if ratio >= suggestive:
        return (ratio, "survives")
    if ratio >= 1.0:
        return (ratio, "suggestive")
    return (ratio, "inside noise")


def run_level_comparison(
    a: list[float], b: list[float], name_a: str = "a", name_b: str = "b"
) -> dict[str, float | str]:
    """Compare two methods with the *training run* as the sampling unit.

    Uses Welch's t-test rather than a paired test: the two methods' runs are not
    paired by anything meaningful once the seed also changes initialisation, and
    Welch does not assume equal variances. With three seeds per side this test has
    very little power, which is exactly the point — the honest report is a wide
    interval and a weak p-value, not a paired test's confident-looking one.
    """
    x = np.asarray([v for v in a if np.isfinite(v)], dtype=np.float64)
    y = np.asarray([v for v in b if np.isfinite(v)], dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return {
            "name_a": name_a, "name_b": name_b, "n_a": int(x.size), "n_b": int(y.size),
            "mean_a": float(x.mean()) if x.size else float("nan"),
            "mean_b": float(y.mean()) if y.size else float("nan"),
            "difference": float("nan"), "p_value": float("nan"),
            "test": "not measured (need >= 2 runs per method)",
        }
    t = stats.ttest_ind(x, y, equal_var=False)
    pooled = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2.0)
    return {
        "name_a": name_a,
        "name_b": name_b,
        "n_a": int(x.size),
        "n_b": int(y.size),
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "difference": float(x.mean() - y.mean()),
        "p_value": float(t.pvalue),
        "cohens_d": float((x.mean() - y.mean()) / pooled) if pooled > 0 else float("nan"),
        "test": "Welch t-test, unit = training run",
    }


def summarize_metric(
    per_row: dict[str, np.ndarray],
    metric: str,
    baseline: str,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[dict[str, Interval], list[Comparison]]:
    """Intervals for every method and paired comparisons against a baseline.

    Returns:
        ``(intervals, comparisons)`` with Holm-Bonferroni already applied.

    Raises:
        KeyError: if ``baseline`` is not among the methods.
    """
    if baseline not in per_row:
        raise KeyError(f"baseline {baseline!r} not among methods {sorted(per_row)}")
    intervals = {
        name: bootstrap_ci(v, n_resamples, seed=seed) for name, v in per_row.items()
    }
    comparisons = [
        compare(v, per_row[baseline], f"{name}.{metric}", f"{baseline}.{metric}",
                n_resamples, seed)
        for name, v in per_row.items()
        if name != baseline
    ]
    return intervals, holm_bonferroni(comparisons)
