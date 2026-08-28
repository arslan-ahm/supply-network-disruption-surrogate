"""Counterfactual criticality: ranking nodes and lanes by measured downstream impact.

This is the product. A reference-style pipeline outputs a risk score per supplier;
this outputs an answer to "if this node fails for eight periods next month, what
happens to my customers' service level, and which node should I look at first?"

Three rankings are produced for the same network and compared against the
simulator's exhaustive answer:

``simulator``
    The oracle: apply a standardised outage to each candidate in turn and measure
    total service loss. Exact, and ``O(n)`` simulator runs.
``surrogate``
    One batched forward pass over ``n`` synthetic scenarios. Approximate, and
    roughly ``O(1)`` in wall-clock terms at these sizes.
``tabular`` / ``heuristic``
    The reference-style feature score and the topology composite, ranking the same
    candidates so the disagreement can be located and then adjudicated by the
    simulator.

The disagreement experiment in :func:`find_disagreements` is the headline result
this repository is built around: it looks for candidate pairs where the
feature-based score and the counterfactual ranking give opposite verdicts, and
reports the simulator's answer for both. There is no ambiguity about who is right
in that comparison, which is the rare and valuable thing about having an oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sndsur.config import Config
from sndsur.data.disruptions import Disruption, DisruptionSet
from sndsur.data.features import edge_features, node_features, tabular_row_features
from sndsur.data.network import N_TIERS, SupplyNetwork
from sndsur.data.scenarios import NetworkBundle, Scenario
from sndsur.models.baselines import require_non_degenerate
from sndsur.sim.simulator import (
    SimConfig,
    counterfactual,
    demand_realisation,
    simulate,
)


@dataclass
class CriticalityResult:
    """Candidate rankings for one network under one standardised disruption.

    Attributes:
        network: Network name.
        candidates: Node ids (or edge indices) scored, ascending.
        truth: Total service loss from exhaustive simulation, aligned.
        scores: Method name to score vector, aligned with ``candidates``.
        sim_seconds: Wall-clock for the exhaustive sweep, including the one
            baseline run it shares across candidates.
        method_seconds: Wall-clock per method for producing its full ranking.
        kind: ``"node"`` or ``"edge"``.
    """

    network: str
    candidates: np.ndarray
    truth: np.ndarray
    scores: dict[str, np.ndarray] = field(default_factory=dict)
    sim_seconds: float = 0.0
    method_seconds: dict[str, float] = field(default_factory=dict)
    kind: str = "node"


def standard_outage(node: int, cfg: Config) -> DisruptionSet:
    """The standardised probe used for every candidate.

    Fixing start, duration and severity across candidates is what turns the
    resulting ranking into a statement about *network position*. If each candidate
    were probed with a randomly drawn disruption, a node could rank highly for
    having drawn a longer outage, and the ranking would no longer be a property of
    the topology.
    """
    return DisruptionSet(
        [
            Disruption(
                "supplier_outage",
                int(node),
                -1,
                cfg.eval.criticality_start,
                cfg.eval.criticality_duration,
                1.0,
            )
        ]
    )


def standard_lane_closure(u: int, v: int, cfg: Config) -> DisruptionSet:
    """The standardised probe for a transport lane."""
    return DisruptionSet(
        [
            Disruption(
                "lane_closure",
                int(u),
                int(v),
                cfg.eval.criticality_start,
                cfg.eval.criticality_duration,
                1.0,
            )
        ]
    )


def probe_scenarios(
    bundle: NetworkBundle,
    cfg: Config,
    sim: SimConfig,
    seed: int,
    candidates: np.ndarray,
    traj_periods: int,
    kind: str = "node",
    edges: list[tuple[int, int]] | None = None,
) -> list[Scenario]:
    """Build the surrogate's input scenarios for every candidate, without simulating.

    This is the whole efficiency argument in one function: the surrogate needs
    only the features, which are cheap, whereas the oracle needs a simulator run
    per candidate. The returned scenarios carry NaN targets — they exist to be
    predicted from, not learned from, and filling the targets with zeros would
    make it possible to accidentally train on them.
    """
    net = bundle.net
    n_dem = net.demand_nodes.size
    nan_v = np.full(n_dem, np.nan, dtype=np.float32)
    nan_t = np.full((n_dem, traj_periods), np.nan, dtype=np.float32)
    out: list[Scenario] = []
    for i, c in enumerate(candidates):
        if kind == "node":
            ds = standard_outage(int(c), cfg).bind(net)
        else:
            u, v = (edges or net.edges())[int(c)]
            ds = standard_lane_closure(u, v, cfg).bind(net)
        nf = node_features(net, ds, sim.horizon, static=bundle.static)
        out.append(
            Scenario(
                net_id=0,
                scenario_id=i,
                disruption=ds,
                seed=seed,
                node_feat=nf.astype(np.float32),
                edge_feat=edge_features(net, ds).astype(np.float32),
                tabular=tabular_row_features(
                    net, ds, sim.horizon, nf=nf, summary=bundle.summary
                ).astype(np.float32),
                service_loss=nan_v.copy(),
                unmet_extra=nan_v.copy(),
                time_to_impact=nan_v.copy(),
                recovery_time=nan_v.copy(),
                traj=nan_t.copy(),
                demand_rows=net.demand_nodes.copy(),
            )
        )
    return out


def simulate_truth(
    net: SupplyNetwork,
    cfg: Config,
    sim: SimConfig,
    seed: int,
    candidates: np.ndarray,
    kind: str = "node",
    edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Exhaustive ground-truth criticality for every candidate.

    The single baseline run is computed once and shared, which any real user would
    also do — the efficiency comparison would be dishonest against an
    artificially slowed oracle.
    """
    d = demand_realisation(net, sim, seed)
    base = simulate(net, sim, d, None)
    out = np.zeros(candidates.size, dtype=np.float64)
    for i, c in enumerate(candidates):
        if kind == "node":
            ds = standard_outage(int(c), cfg).bind(net)
        else:
            u, v = (edges or net.edges())[int(c)]
            ds = standard_lane_closure(u, v, cfg).bind(net)
        out[i] = counterfactual(net, sim, ds, seed, baseline=base).total_service_loss
    return out


def node_candidates(net: SupplyNetwork) -> np.ndarray:
    """Every non-demand node — the set a planner would actually screen.

    Demand points are excluded because "the customer stops wanting the product" is
    a demand-side event, not a supply failure, and including them would put a
    guaranteed-high-impact item into every top-k for a trivial reason.
    """
    return np.flatnonzero(net.tier < N_TIERS - 1)


def aggregate_surrogate_scores(
    impact: np.ndarray, n_candidates: int, n_demand: int
) -> np.ndarray:
    """Reduce per-(candidate, demand point) predictions to one score per candidate.

    Summed over demand points, matching the oracle's ``total_service_loss``. The
    sum rather than the mean because a node that starves four demand points is
    worse than one that starves one, and the mean would erase exactly the
    difference this project is about.
    """
    return impact.reshape(n_candidates, n_demand).sum(axis=1)


def tabular_scores_for_candidates(rows: np.ndarray, model, n_candidates: int,
                                  n_demand: int) -> np.ndarray:
    """Reference-style score per candidate, summed the same way as the surrogate's."""
    pred = model.predict(rows)
    return np.asarray(pred, dtype=np.float64).reshape(n_candidates, n_demand).sum(axis=1)


@dataclass
class Disagreement:
    """One pair where two rankings give opposite verdicts.

    Attributes:
        node_a: The candidate the feature-based method prefers to flag.
        node_b: The candidate the counterfactual method prefers to flag.
        rank_a_feature, rank_b_feature: 1-based ranks under the feature score.
        rank_a_counterfactual, rank_b_counterfactual: 1-based ranks under the
            counterfactual score.
        truth_a, truth_b: Simulated total service loss for each.
        winner: Which candidate the simulator says matters more.
        margin: ``|truth_b - truth_a|``, the service level at stake.
        explanation: The structural reason, read off the network.
    """

    node_a: int
    node_b: int
    rank_a_feature: int
    rank_b_feature: int
    rank_a_counterfactual: int
    rank_b_counterfactual: int
    truth_a: float
    truth_b: float
    winner: str
    margin: float
    explanation: str

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["winner"] = self.winner
        return d


def explain_node(net: SupplyNetwork, v: int) -> str:
    """A one-line structural description of why a node matters, or does not."""
    ss = net.sole_source_reach()[v]
    reach = net.downstream_reach()[v]
    thr = net.throughput if net.throughput.size == net.n_nodes else net.compute_throughput()
    finite = np.isfinite(net.capacity[v])
    slack = (
        net.capacity[v] / max(thr[v], 1e-9) - 1.0 if finite and thr[v] > 0 else float("nan")
    )
    n_cust = len(net.customers()[v])
    return (
        f"tier {net.tier[v]}, {n_cust} customers, reaches {int(reach)} demand points "
        f"({int(ss)} sole-sourced), throughput {thr[v]:.1f}, capacity slack {slack:.2f}"
    )


def find_disagreements(
    net: SupplyNetwork,
    candidates: np.ndarray,
    feature_scores: np.ndarray,
    counterfactual_scores: np.ndarray,
    truth: np.ndarray,
    top_k: int = 10,
    max_pairs: int = 6,
) -> list[Disagreement]:
    """Locate pairs the two rankings order oppositely, and adjudicate them.

    A pair qualifies when one candidate is inside the feature score's top-``k``
    and outside the counterfactual's, while the other is the reverse. Pairs are
    returned in descending order of the true service level at stake, so the first
    one is the most consequential disagreement rather than merely the first found.

    The simulator's ``truth`` decides the winner. This is the part that a purely
    observational study cannot do: there is no arguing about which ranking was
    right, because the counterfactual was actually run.

    Both score vectors must actually rank. A constant score has no ordering, and
    ``np.argsort`` of a constant returns index order, so a constant "feature
    score" silently produces ranks that are nothing but candidate-node ids. This
    repository shipped 36 adjudicated disagreements produced exactly that way —
    in every one of them ``rank_a_feature == node_a + 1`` — which made the
    headline experiment an adjudication of node numbering rather than of a
    ranking. Hence the check.

    Raises:
        DegenerateBaselineError: if either score vector is constant.
    """
    for label, sc in (("feature_scores", feature_scores), ("counterfactual_scores",
                                                           counterfactual_scores)):
        require_non_degenerate(f"find_disagreements:{label}", sc)
    f_order = np.argsort(-np.asarray(feature_scores, dtype=np.float64), kind="stable")
    c_order = np.argsort(-np.asarray(counterfactual_scores, dtype=np.float64), kind="stable")
    f_rank = np.empty(candidates.size, dtype=np.int64)
    c_rank = np.empty(candidates.size, dtype=np.int64)
    f_rank[f_order] = np.arange(1, candidates.size + 1)
    c_rank[c_order] = np.arange(1, candidates.size + 1)

    feature_only = [i for i in range(candidates.size) if f_rank[i] <= top_k < c_rank[i]]
    cf_only = [i for i in range(candidates.size) if c_rank[i] <= top_k < f_rank[i]]

    out: list[Disagreement] = []
    for ia in feature_only:
        for ib in cf_only:
            a, b = int(candidates[ia]), int(candidates[ib])
            ta, tb = float(truth[ia]), float(truth[ib])
            out.append(
                Disagreement(
                    node_a=a,
                    node_b=b,
                    rank_a_feature=int(f_rank[ia]),
                    rank_b_feature=int(f_rank[ib]),
                    rank_a_counterfactual=int(c_rank[ia]),
                    rank_b_counterfactual=int(c_rank[ib]),
                    truth_a=ta,
                    truth_b=tb,
                    winner="counterfactual" if tb > ta else ("feature" if ta > tb else "tie"),
                    margin=abs(tb - ta),
                    explanation=(
                        f"node {a}: {explain_node(net, a)} | "
                        f"node {b}: {explain_node(net, b)}"
                    ),
                )
            )
    out.sort(key=lambda d: -d.margin)
    return out[:max_pairs]
