"""Wall-clock benchmarking, done properly.

Parameters and FLOPs are inputs to a performance story; wall-clock is the claim.
Two rules are enforced here because getting them wrong produces confidently wrong
numbers:

**Warm up enough.** The first several iterations of any torch model pay for lazy
kernel selection, allocator growth and cache population. The reference project
for this build standard measured a tiny model as 5x *slower* than reality by
timing it cold. The floor here is 8 warm-up iterations and 25 timed repeats, and
the config cannot go below it without saying so.

**Report the spread.** Median and interquartile range, not mean. On a shared
4-core machine another process can steal a core mid-measurement, which produces a
long right tail that a mean absorbs and a median does not.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


@dataclass
class Timing:
    """Result of a timed benchmark.

    Attributes:
        label: What was timed.
        median_ms: Median wall-clock per call.
        iqr_ms: Interquartile range, as a spread measure.
        min_ms: Fastest observed call.
        mean_ms: Mean, reported alongside the median so a reader can see skew.
        n_warmup: Warm-up iterations discarded.
        n_repeats: Timed iterations.
        per_item_ms: ``median_ms / items`` when the call processed a batch.
        items: Items per call.
    """

    label: str
    median_ms: float
    iqr_ms: float
    min_ms: float
    mean_ms: float
    n_warmup: int
    n_repeats: int
    per_item_ms: float
    items: int

    def to_dict(self) -> dict:
        return asdict(self)


def benchmark(
    fn: Callable[[], object],
    label: str = "call",
    warmup: int = 8,
    repeats: int = 25,
    items: int = 1,
) -> Timing:
    """Time ``fn`` with warm-up, and summarise robustly.

    Args:
        fn: Zero-argument callable. Any return value is discarded.
        label: Name for the results table.
        warmup: Discarded iterations. Raised to 8 if a smaller value is passed —
            silently accepting 1 would let a caller produce a wrong number that
            looks like a measurement.
        repeats: Timed iterations, floored at 5.
        items: Items processed per call, for the per-item column.

    Returns:
        A :class:`Timing`.
    """
    warmup = max(8, int(warmup))
    repeats = max(5, int(repeats))
    for _ in range(warmup):
        fn()
    samples = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - t0) * 1000.0
    med = float(np.median(samples))
    q1, q3 = np.quantile(samples, [0.25, 0.75])
    return Timing(
        label=label,
        median_ms=med,
        iqr_ms=float(q3 - q1),
        min_ms=float(samples.min()),
        mean_ms=float(samples.mean()),
        n_warmup=warmup,
        n_repeats=repeats,
        per_item_ms=med / max(items, 1),
        items=int(items),
    )


def break_even_scenarios(
    train_seconds: float,
    sim_ms_per_scenario: float,
    surrogate_ms_per_scenario: float,
    dataset_seconds: float = 0.0,
) -> dict[str, float]:
    """How many screened scenarios it takes for the surrogate to pay for itself.

    A surrogate is only worth building if the time it saves exceeds the time it
    cost to build, and "the time it cost" includes generating its training data
    *with the simulator it replaces*. Omitting that is the most common way this
    kind of comparison is inflated, so ``dataset_seconds`` is a required part of
    the accounting and appears in the returned dict.

    Returns:
        ``setup_seconds``, ``saving_per_scenario_ms``, ``break_even_scenarios``
        and ``speedup_per_scenario``. Break-even is ``inf`` when the surrogate is
        not actually faster per scenario, which is a real possible outcome and is
        reported rather than hidden.
    """
    setup = float(train_seconds + dataset_seconds)
    saving_ms = float(sim_ms_per_scenario - surrogate_ms_per_scenario)
    return {
        "train_seconds": float(train_seconds),
        "dataset_seconds": float(dataset_seconds),
        "setup_seconds": setup,
        "sim_ms_per_scenario": float(sim_ms_per_scenario),
        "surrogate_ms_per_scenario": float(surrogate_ms_per_scenario),
        "saving_per_scenario_ms": saving_ms,
        "break_even_scenarios": (setup * 1000.0 / saving_ms) if saving_ms > 0 else float("inf"),
        "speedup_per_scenario": (
            float(sim_ms_per_scenario / surrogate_ms_per_scenario)
            if surrogate_ms_per_scenario > 0
            else float("inf")
        ),
    }


def budget_curve(
    truth: np.ndarray,
    surrogate_scores: np.ndarray,
    sim_ms: float,
    surrogate_total_ms: float,
    budgets_s: tuple[float, ...],
    k: int = 10,
    seed: int = 0,
    n_repeats: int = 20,
) -> list[dict[str, float]]:
    """Recall@k of the true critical set versus compute budget, for both methods.

    This is the decision-quality-at-fixed-compute experiment, and it is the fair
    way to compare an exact-but-slow method with an approximate-but-fast one.

    The **simulator** strategy spends its budget evaluating as many candidates
    exactly as it can afford, chosen at random, and returns those it evaluated
    ranked by their exact impact. Candidates it could not afford to evaluate are
    ranked last. Randomly chosen because an ordering that told it which
    candidates to try first would already be the surrogate's contribution.
    Averaged over ``n_repeats`` random draws, since which candidates it happens
    to pick matters a great deal at small budgets.

    The **surrogate** strategy pays ``surrogate_total_ms`` once, screens every
    candidate, and returns its own ranking. Below that cost it returns nothing.

    Args:
        truth: Exact criticality per candidate.
        surrogate_scores: Predicted criticality per candidate.
        sim_ms: Measured milliseconds per exact candidate evaluation.
        surrogate_total_ms: Measured milliseconds to screen all candidates.
        budgets_s: Budgets in seconds.
        k: Top-k for the recall metric.
        seed: RNG seed for the simulator's random candidate choice.
        n_repeats: Random draws averaged per budget.

    Returns:
        One row per (budget, method).
    """
    from sndsur.metrics.ranking import recall_at_k, true_critical_set

    t = np.asarray(truth, dtype=np.float64)
    n = t.size
    target = true_critical_set(t, k)
    rows: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)

    for b in budgets_s:
        ms = b * 1000.0
        afford = int(min(n, ms // sim_ms)) if sim_ms > 0 else n
        if target.size == 0:
            recall_sim = float("nan")
        elif afford == 0:
            recall_sim = 0.0
        else:
            vals = []
            for _ in range(n_repeats):
                subset = rng.choice(n, size=afford, replace=False)
                # Score = exact impact for evaluated candidates, -inf otherwise,
                # so unevaluated candidates can never enter the shortlist.
                score = np.full(n, -np.inf)
                score[subset] = t[subset]
                vals.append(recall_at_k(score, t, k)[0])
            recall_sim = float(np.mean(vals))
        rows.append({
            "budget_s": float(b),
            "method": "simulator",
            "candidates_evaluated": float(afford),
            "recall_at_k": recall_sim,
            "k": int(k),
            "true_set_size": int(target.size),
        })

        if ms >= surrogate_total_ms:
            recall_sur = recall_at_k(np.asarray(surrogate_scores, dtype=np.float64), t, k)[0]
            evaluated = float(n)
        else:
            recall_sur = 0.0 if target.size else float("nan")
            evaluated = 0.0
        rows.append({
            "budget_s": float(b),
            "method": "surrogate",
            "candidates_evaluated": evaluated,
            "recall_at_k": recall_sur,
            "k": int(k),
            "true_set_size": int(target.size),
        })
    return rows
