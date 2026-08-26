"""Critical-set recall and decision regret.

These are the metrics that decide whether the surrogate is *useful*, as distinct
from whether it is accurate. The question a planner brings is "which links should
I look at first", and the answer is a short list. So the metrics are:

``recall_at_k``
    Of the ``k`` truly most critical nodes, how many appear in the surrogate's
    top ``k``? This is the screening metric. It is deliberately set-based rather
    than order-based inside the top-k: if the surrogate finds the right ten links
    in the wrong order, the planner still examines the right ten.
``decision_regret``
    Suppose the planner can protect ``k`` nodes and picks them from the
    surrogate's ranking. How much true impact do they fail to avert compared with
    picking from the true ranking? Reported both absolutely and as a fraction of
    the best achievable, because "0.004 of service level" means nothing without
    knowing that the best possible was 0.31.
``top_k_jaccard`` and ``kendall_top``
    Set overlap and rank agreement restricted to the union of the two top-k sets,
    which is where a screening user's attention actually goes.

**Ties.** The true criticality vector has a large mass at exactly zero, so "the
true top-10" can be ambiguous when fewer than ten nodes have any impact at all.
:func:`true_critical_set` therefore returns only nodes with impact above a
tolerance, and every metric reports the size of the set it actually used.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

#: Impacts at or below this are treated as no impact. Service-level loss is a
#: fraction in [0, 1], so 1e-6 is far below anything a planner would act on and
#: far above float noise from the simulator's material balance.
TOL = 1e-6


def true_critical_set(scores: np.ndarray, k: int, tol: float = TOL) -> np.ndarray:
    """Indices of the ``k`` highest-impact items, excluding zero-impact ones.

    Returns:
        Up to ``k`` indices, descending by score. Fewer than ``k`` when the
        network has fewer than ``k`` nodes with any downstream impact — a real
        and informative case, not an error.
    """
    s = np.asarray(scores, dtype=np.float64)
    live = np.flatnonzero(s > tol)
    if live.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = live[np.argsort(-s[live], kind="stable")]
    return order[:k]


def recall_at_k(
    pred_scores: np.ndarray, true_scores: np.ndarray, k: int, tol: float = TOL
) -> tuple[float, int]:
    """Fraction of the true top-``k`` recovered by the predicted top-``k``.

    Returns:
        ``(recall, size_of_true_set)``. NaN recall when the true set is empty,
        because there is nothing to recall — reporting 0.0 there would punish a
        model for a network that has no critical nodes.
    """
    truth = true_critical_set(true_scores, k, tol)
    if truth.size == 0:
        return (float("nan"), 0)
    p = np.asarray(pred_scores, dtype=np.float64)
    # The predicted set is the top-k by score with no tolerance filter: the
    # surrogate is entitled to nominate k candidates whatever it thinks of them,
    # and filtering its list would flatter it by shrinking the denominator.
    pred = np.argsort(-p, kind="stable")[: truth.size]
    return (len(set(pred.tolist()) & set(truth.tolist())) / truth.size, int(truth.size))


def precision_at_k(
    pred_scores: np.ndarray, true_scores: np.ndarray, k: int, tol: float = TOL
) -> tuple[float, int]:
    """Fraction of the predicted top-``k`` that have any true impact at all.

    A weaker but useful companion to recall: it catches a model that fills its
    shortlist with nodes that cannot possibly matter (zero downstream reach).
    """
    p = np.asarray(pred_scores, dtype=np.float64)
    t = np.asarray(true_scores, dtype=np.float64)
    kk = min(k, p.size)
    if kk == 0:
        return (float("nan"), 0)
    pred = np.argsort(-p, kind="stable")[:kk]
    return (float((t[pred] > tol).mean()), int(kk))


def decision_regret(
    pred_scores: np.ndarray, true_scores: np.ndarray, k: int
) -> dict[str, float]:
    """Loss from protecting the surrogate's top-``k`` rather than the true top-``k``.

    The decision model is deliberately simple and stated rather than dressed up:
    protecting a node averts exactly its own counterfactual impact, and impacts do
    not interact. Both assumptions are wrong in a network — protecting two nodes
    on the same path averts less than the sum — so this is an *upper bound* on
    achievable benefit and the regret is a lower bound on the true regret. It is
    reported anyway because it is monotone in ranking quality and because the
    alternative, re-simulating every subset, is combinatorial.

    Returns:
        ``regret`` (absolute impact not averted), ``regret_fraction`` (as a share
        of the oracle's averted impact), ``averted`` and ``oracle_averted``.
    """
    t = np.asarray(true_scores, dtype=np.float64)
    p = np.asarray(pred_scores, dtype=np.float64)
    kk = min(k, t.size)
    if kk == 0:
        return {"regret": float("nan"), "regret_fraction": float("nan"),
                "averted": float("nan"), "oracle_averted": float("nan"), "k": 0}
    oracle = float(np.sort(t)[::-1][:kk].sum())
    chosen = float(t[np.argsort(-p, kind="stable")[:kk]].sum())
    regret = oracle - chosen
    return {
        "regret": regret,
        "regret_fraction": (regret / oracle) if oracle > 0 else float("nan"),
        "averted": chosen,
        "oracle_averted": oracle,
        "k": int(kk),
    }


def top_k_jaccard(pred_scores: np.ndarray, true_scores: np.ndarray, k: int) -> float:
    """Jaccard overlap of the two top-``k`` sets."""
    p = np.asarray(pred_scores, dtype=np.float64)
    t = np.asarray(true_scores, dtype=np.float64)
    kk = min(k, p.size)
    if kk == 0:
        return float("nan")
    a = set(np.argsort(-p, kind="stable")[:kk].tolist())
    b = set(np.argsort(-t, kind="stable")[:kk].tolist())
    union = a | b
    return len(a & b) / len(union) if union else float("nan")


def kendall_on_union(pred_scores: np.ndarray, true_scores: np.ndarray, k: int) -> float:
    """Kendall tau restricted to the union of the two top-``k`` sets.

    Full-vector rank correlation is dominated by the long tail of zero-impact
    nodes, whose relative order nobody cares about and which are all tied in
    truth. Restricting to the union of the shortlists measures agreement where a
    user would notice it.
    """
    p = np.asarray(pred_scores, dtype=np.float64)
    t = np.asarray(true_scores, dtype=np.float64)
    kk = min(k, p.size)
    if kk == 0:
        return float("nan")
    idx = np.array(
        sorted(
            set(np.argsort(-p, kind="stable")[:kk].tolist())
            | set(np.argsort(-t, kind="stable")[:kk].tolist())
        )
    )
    if idx.size < 3 or np.all(t[idx] == t[idx][0]) or np.all(p[idx] == p[idx][0]):
        return float("nan")
    return float(stats.kendalltau(p[idx], t[idx], variant="b").statistic)


def ranking_report(
    pred_scores: np.ndarray,
    true_scores: np.ndarray,
    ks: tuple[int, ...] = (5, 10, 20),
    prefix: str = "",
) -> dict[str, float]:
    """Every ranking metric at every ``k``, plus the full-vector correlations."""
    out: dict[str, float] = {}
    p = np.asarray(pred_scores, dtype=np.float64)
    t = np.asarray(true_scores, dtype=np.float64)
    out[f"{prefix}n_candidates"] = int(t.size)
    out[f"{prefix}n_nonzero_true"] = int((t > TOL).sum())
    if t.size >= 3 and not np.all(t == t[0]) and not np.all(p == p[0]):
        out[f"{prefix}spearman_full"] = float(stats.spearmanr(p, t).statistic)
    else:
        out[f"{prefix}spearman_full"] = float("nan")
    for k in ks:
        r, n = recall_at_k(p, t, k)
        out[f"{prefix}recall@{k}"] = r
        out[f"{prefix}true_set_size@{k}"] = n
        pr, _ = precision_at_k(p, t, k)
        out[f"{prefix}precision@{k}"] = pr
        out[f"{prefix}jaccard@{k}"] = top_k_jaccard(p, t, k)
        out[f"{prefix}kendall_union@{k}"] = kendall_on_union(p, t, k)
        reg = decision_regret(p, t, k)
        out[f"{prefix}regret@{k}"] = reg["regret"]
        out[f"{prefix}regret_frac@{k}"] = reg["regret_fraction"]
        out[f"{prefix}oracle_averted@{k}"] = reg["oracle_averted"]
    return out
