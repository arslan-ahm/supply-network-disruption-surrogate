"""Baselines, all of them actually run.

Six, each answering a different objection:

``ConstantPredictor``
    The two trivial baselines: predict exactly ``0.0`` everywhere, or predict the
    training mean everywhere. They cost nothing to fit and they exist because the
    target is **92.5% exact zeros at the row level**, and on a distribution like
    that the number a results table most needs is what predicting nothing scores.
    Leaving them out is how a degenerate model gets mistaken for a good one: the
    L1-optimal constant on this target *is* zero, so any method whose MAE matches
    ``constant_zero`` to six decimal places has learned nothing, and without the
    column there is no way for a reader to notice. See ``docs/RESULTS.md`` §8.7 —
    this repository shipped exactly that mistake.

``TabularRiskModel``
    The reference approach. Gradient-boosted trees (and, as a second variant,
    ridge regression) on a flat feature row: attributes of the disrupted node,
    attributes of the demand point, and network summary statistics. No relational
    term connects the two, which is exactly the structural limitation this
    repository argues about. It is given the same targets, the same rows in the
    same order, and generous features — including three graph-derived columns it
    would not normally have — so that its failure, where it fails, is about the
    missing *relation* and not about missing information.

    Two loss variants are kept, and the second is kept *because* it fails.
    ``kind="gbt"`` uses squared error and is the working reference model.
    ``kind="gbt_l1"`` uses absolute error and is **degenerate by construction on
    this target**: the L1-optimal constant is the median, the median is 0, so the
    boosting initialisation is 0, every leaf's L1-optimal value is 0, early
    stopping fires within ~10 rounds and the model emits a single unique
    prediction. It is reported with that label rather than deleted, because a
    reference model that collapses under a plausible-sounding tuning choice is
    informative, and because dropping it would hide the bug that made it the
    repository's accidental headline baseline.

``TopologyHeuristic``
    No learning at all: a hand-weighted composite of the topology signals a
    resilience engineer would actually reach for — sole-source reach, downstream
    reach, betweenness, BOM depth, capacity slack. It exists because a graph
    neural network that cannot beat five lines of graph arithmetic is not worth
    its training cost, and because it produces a *ranking* without a magnitude,
    which is a fair description of what centrality analysis gives you.

``ScenarioRetrieval``
    Nearest neighbour in scenario-descriptor space, returning the neighbour's
    measured impact. This is the "just look it up" objection, and it is a strong
    baseline in-distribution because the training set contains many near-duplicate
    scenarios. Its behaviour under topology shift is the interesting part.

The remaining one — an MLP with the surrogate's own features but no message
passing — is not here: it is the surrogate with ``model.layers=0``, so it shares
the training loop, the loss, the heads and the parameter budget with the full
model. Running it as a separate implementation would have made it a different
model in more ways than the one being tested.

Every method that enters a comparison table goes through
:func:`prediction_diversity`, and every method not declared constant *by design*
goes through :func:`require_non_degenerate`. A predictor with fewer than two
unique values is not a model, and it must not be allowed into a results table
looking like one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sndsur.data.network import SupplyNetwork


@dataclass
class RowData:
    """Flat per-row view of a split, aligned with the surrogate's predictions.

    Attributes:
        x: ``(R, F)`` features.
        y: ``(R,)`` impact target.
        scenario_id: ``(R,)`` scenario each row came from.
        net_id: ``(R,)`` network each row came from.
        demand_node: ``(R,)`` node id of the demand point.
        tti: ``(R,)`` time-to-impact target, NaN where undefined.
        recovery: ``(R,)`` recovery-time target, NaN where undefined.
    """

    x: np.ndarray
    y: np.ndarray
    scenario_id: np.ndarray
    net_id: np.ndarray
    demand_node: np.ndarray
    tti: np.ndarray
    recovery: np.ndarray


def rows_from_split(ds, split: str) -> RowData:
    """Flatten a split into aligned arrays for the tabular baselines.

    Rows are emitted in scenario order and, within a scenario, in ascending
    demand-node order — the same order the batching module produces — so a
    tabular prediction can be compared row-for-row with a surrogate prediction
    without a join.
    """
    scs = ds.splits[split]
    if not scs:
        empty = np.zeros((0,), dtype=np.float32)
        return RowData(np.zeros((0, 0), np.float32), empty, empty.astype(np.int64),
                       empty.astype(np.int64), empty.astype(np.int64), empty, empty)
    return RowData(
        x=np.concatenate([s.tabular for s in scs], axis=0),
        y=np.concatenate([s.service_loss for s in scs]),
        scenario_id=np.concatenate([np.full(s.n_demand, s.scenario_id) for s in scs]),
        net_id=np.concatenate([np.full(s.n_demand, s.net_id) for s in scs]),
        demand_node=np.concatenate([s.demand_rows for s in scs]),
        tti=np.concatenate([s.time_to_impact for s in scs]),
        recovery=np.concatenate([s.recovery_time for s in scs]),
    )


class DegenerateBaselineError(RuntimeError):
    """A method's predictions carry no ordering information.

    Raised for a predictor whose output has fewer than two distinct values when
    that was not the declared intent. This exists because the repository shipped
    a comparison table in which the best-MAE "model" was a constant, and no
    metric in the table revealed it: MAE is minimised by predicting zero on a
    92.5%-zero target, rank correlations came out as ``NaN`` and were rendered
    as ``not measured``, and top-1 agreement scored 0.125 purely from
    ``argmax`` returning index 0 on an all-tied row. The only reliable detector
    is counting unique predictions, so that is now counted for every method.
    """


def prediction_diversity(pred: np.ndarray, decimals: int = 12) -> dict[str, float]:
    """How many distinct values a predictor actually emitted.

    Args:
        pred: Predictions for one split.
        decimals: Rounding applied before counting, so float noise at the 1e-15
            level is not mistaken for genuine variation.

    Returns:
        ``{"n_unique_predictions", "pred_range", "degenerate"}``. ``degenerate``
        is true when fewer than two distinct values were emitted — i.e. the
        predictor is a constant function and has no ranking at all.
    """
    v = np.asarray(pred, dtype=np.float64).ravel()
    finite = v[np.isfinite(v)]
    n_unique = int(np.unique(np.round(finite, decimals)).size)
    return {
        "n_unique_predictions": n_unique,
        "pred_range": float(np.ptp(finite)) if finite.size else float("nan"),
        "degenerate": bool(n_unique < 2),
    }


def require_non_degenerate(
    name: str, pred: np.ndarray, decimals: int = 12
) -> dict[str, float]:
    """:func:`prediction_diversity`, but a constant predictor is an error.

    Args:
        name: Method name, for the message.
        pred: Predictions for one split.
        decimals: Passed through.

    Returns:
        The diversity record, for recording alongside the method's metrics.

    Raises:
        DegenerateBaselineError: if fewer than two distinct values were emitted.
    """
    d = prediction_diversity(pred, decimals)
    if d["degenerate"]:
        raise DegenerateBaselineError(
            f"{name!r} emitted {d['n_unique_predictions']} unique prediction(s) over "
            f"{np.asarray(pred).size} rows: it is a constant function, not a model. "
            "Either fix it or declare it constant by design so the table says so."
        )
    return d


class ConstantPredictor:
    """The trivial baselines: predict zero, or predict the training mean.

    Not padding. On a target that is 92.5% exact zeros these are the two numbers
    a reader needs in order to interpret any error metric in the table at all:

    * ``kind="zero"`` is the **MAE-optimal constant** here, because the L1
      minimiser is the median and the median of this target is exactly 0. Any
      method that does not beat it on MAE has not demonstrated magnitude
      fidelity, and any method that *matches* it to six decimal places has
      collapsed to it.
    * ``kind="train_mean"`` is the **RMSE-optimal constant**, and it is the
      weaker of the two on MAE by roughly a factor of two on every split here.
      Reporting only the train-mean constant — the conventional choice — would
      have flattered every learned method in this repository.

    There is nothing to fit but one scalar, so these cost no compute and there is
    no excuse for their absence from a results table.
    """

    #: Both are constants by design; the degeneracy guard must not raise on them.
    KINDS = ("zero", "train_mean")

    def __init__(self, kind: str = "zero") -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown constant kind {kind!r}")
        self.kind = kind
        self.value: float | None = None

    @property
    def degenerate_by_construction(self) -> bool:
        """Always true. That is the entire point of this class."""
        return True

    def fit(self, x: np.ndarray, y: np.ndarray) -> ConstantPredictor:
        del x
        t = np.asarray(y, dtype=np.float64)
        t = t[np.isfinite(t)]
        if self.kind == "zero":
            self.value = 0.0
        else:
            self.value = float(t.mean()) if t.size else 0.0
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.value is None:
            raise RuntimeError("model used before fit")
        n = int(np.asarray(x).shape[0])
        return np.full(n, self.value, dtype=np.float64)


class TabularRiskModel:
    """Reference-style per-row regressor.

    Args:
        kind: ``"gbt"`` for histogram gradient boosting under squared error,
            ``"gbt_l1"`` for the same trees under absolute error (degenerate on
            this target — see :attr:`DEGENERATE_KINDS`), ``"ridge"`` for a linear
            model. All three are run: the linear one shows what the features
            alone support, the boosted one shows what a strong tabular learner
            extracts from them so a weak result cannot be blamed on model
            choice, and the L1 one shows what happens when the objective is
            chosen to match the surrogate's without checking the target's
            distribution.
        seed: RNG seed.
        max_iter: Boosting rounds. Capped so this baseline's training time stays
            in the same order of magnitude as the surrogate's — a baseline given
            ten times the compute is not a fair comparison in either direction.
    """

    #: Boosting loss per ``kind``. ``gbt_l1`` exists to be shown failing; see
    #: :attr:`DEGENERATE_KINDS` and the module docstring.
    GBT_LOSS = {"gbt": "squared_error", "gbt_l1": "absolute_error"}

    #: Kinds that cannot fit this target and are reported as degenerate rather
    #: than treated as models.
    DEGENERATE_KINDS = frozenset({"gbt_l1"})

    def __init__(self, kind: str = "gbt", seed: int = 0, max_iter: int = 300) -> None:
        self.kind = kind
        self.seed = seed
        self.max_iter = max_iter
        self.model = None
        self.n_params_ = 0

    @property
    def degenerate_by_construction(self) -> bool:
        """True for the L1 variant, whose collapse on this target is analytic."""
        return self.kind in self.DEGENERATE_KINDS

    def fit(self, x: np.ndarray, y: np.ndarray) -> TabularRiskModel:
        if self.kind in self.GBT_LOSS:
            from sklearn.ensemble import HistGradientBoostingRegressor

            self.model = HistGradientBoostingRegressor(
                max_iter=self.max_iter,
                learning_rate=0.08,
                max_depth=6,
                min_samples_leaf=20,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self.seed,
                # Squared error, which is what a boosted-tree regressor on this
                # target has to use to be a model at all.
                #
                # The original choice here was `absolute_error`, justified as
                # "matching the surrogate's L1 objective". That reasoning is
                # wrong on a target that is 92.5% exact zeros: L1's optimal
                # constant is the median, the median is 0, the boosting
                # initialisation is therefore 0, and every leaf's L1-optimal
                # value is 0 as well. The result was a model that emitted
                # 0.000000 for all 20,624 rows of the shipped comparison,
                # scored the best MAE in the table because the all-zero
                # constant is MAE-optimal here, and was then written up in the
                # README as a "reference GBT" the surrogate lost to. It was the
                # constant zero function wearing a boosted-tree costume.
                #
                # `kind="gbt_l1"` keeps that variant reachable and labelled, so
                # the failure is documented instead of quietly deleted.
                loss=self.GBT_LOSS[self.kind],
            )
        elif self.kind == "ridge":
            from sklearn.linear_model import Ridge

            self.model = Ridge(alpha=1.0, random_state=None)
        else:
            raise ValueError(f"unknown tabular kind {self.kind!r}")
        self.model.fit(x, y)
        self.n_params_ = self._count_params(x.shape[1])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model used before fit")
        # Clipped at zero for the same reason the surrogate's head is softplus:
        # service-level loss cannot be negative, and an unclipped tree ensemble
        # emits small negative values that scramble the bottom of the ranking.
        return np.clip(self.model.predict(x), 0.0, None)

    def _count_params(self, n_features: int) -> int:
        """Parameter count, for the efficiency table.

        For a tree ensemble the honest analogue of a parameter count is the total
        node count across all trees, which is what this returns. It is not
        comparable to a neural network's weight count in any deep sense, and the
        efficiency table says so rather than putting them in the same column.
        """
        if self.kind == "ridge":
            return int(n_features + 1)
        total = 0
        for preds in getattr(self.model, "_predictors", []):
            for p in preds:
                total += int(p.nodes.shape[0])
        return total


class TopologyHeuristic:
    """Learning-free criticality score from graph structure.

    The score is a weighted sum of standardised topology signals. Weights are set
    by hand from what the resilience literature treats as important
    (Simchi-Levi et al., 2015 on time-to-recover exposure; Ivanov & Dolgui, 2020
    on ripple effect) rather than fitted, because a fitted heuristic is just a
    linear model on graph features and that is already covered by the ridge
    variant of the tabular baseline. Reporting it unfitted keeps it honest about
    what centrality analysis alone delivers.

    It scores a *node*, not a (node, demand point) pair, so when a per-row
    prediction is needed the node's score is broadcast to every demand point that
    is reachable from it and zero elsewhere. That reachability mask is the one
    relational fact the heuristic gets, and it is a generous one.
    """

    #: Signal weights. Sole-source reach dominates because it is the only signal
    #: that identifies a hard single point of failure rather than a busy one.
    WEIGHTS = {
        "sole_source_reach": 1.00,
        "downstream_reach": 0.45,
        "betweenness": 0.35,
        "bom_depth": 0.15,
        "inv_slack": 0.25,
        "throughput": 0.20,
    }

    def signals(self, net: SupplyNetwork) -> dict[str, np.ndarray]:
        """Raw (unstandardised) topology signals per node."""
        n = net.n_nodes
        thr = net.throughput if net.throughput.size == n else net.compute_throughput()
        finite = np.isfinite(net.capacity)
        slack = np.zeros(n)
        slack[finite] = np.where(
            thr[finite] > 0, net.capacity[finite] / np.maximum(thr[finite], 1e-9) - 1.0, 0.0
        )
        n_dem = max(1, net.demand_nodes.size)
        return {
            "sole_source_reach": net.sole_source_reach() / n_dem,
            "downstream_reach": net.downstream_reach() / n_dem,
            "betweenness": path_betweenness(net),
            "bom_depth": net.bom_depth().astype(float),
            "inv_slack": 1.0 / (1.0 + np.maximum(slack, 0.0)),
            "throughput": np.log1p(thr),
        }

    def node_scores(self, net: SupplyNetwork) -> np.ndarray:
        """Composite criticality score per node, standardised then weighted."""
        sig = self.signals(net)
        score = np.zeros(net.n_nodes)
        for name, w in self.WEIGHTS.items():
            v = sig[name].astype(np.float64)
            sd = v.std()
            score += w * ((v - v.mean()) / sd if sd > 1e-12 else np.zeros_like(v))
        return score


def path_betweenness(net: SupplyNetwork) -> np.ndarray:
    """Fraction of supplier-to-demand-point paths passing through each node.

    Counted over *all* directed paths from tier-0 suppliers to demand points, with
    each source-target pair contributing weight 1 split across its paths. Exact
    rather than sampled: these networks have at most a few hundred nodes and five
    tiers, so the path counts are small integers and a dynamic program over the
    topological order is both exact and instant.

    Returns:
        ``(n_nodes,)`` in ``[0, 1]``.
    """
    n = net.n_nodes
    cust = net.customers()
    demand = set(net.demand_nodes.tolist())
    # paths_to[v] = number of distinct paths from v to any demand point
    paths_to = np.zeros(n, dtype=np.float64)
    for v in range(n - 1, -1, -1):
        paths_to[v] = 1.0 if v in demand else sum(paths_to[c] for c in cust[v])
    sources = np.flatnonzero(net.tier == 0)
    # paths_from[v] = number of distinct paths from any tier-0 source to v
    paths_from = np.zeros(n, dtype=np.float64)
    paths_from[sources] = 1.0
    for u in range(n):
        for c in cust[u]:
            paths_from[c] += paths_from[u]
    through = paths_from * paths_to
    total = through[sources].sum() if sources.size else 0.0
    return through / total if total > 0 else np.zeros(n)


class ScenarioRetrieval:
    """k-nearest-neighbour retrieval over scenario descriptors.

    The descriptor is the same tabular row the reference model sees, so the two
    differ only in how they use it: one fits a function, the other looks up
    neighbours. Distance is Euclidean on standardised features, and the
    prediction is the inverse-distance-weighted mean of the k neighbours'
    impacts.

    Implemented in NumPy rather than with a spatial index: at ~24k training rows
    a brute-force distance matrix in chunks is a few hundred milliseconds, and a
    tree structure would add a dependency and a build step to save time this
    project does not need saved. That cost *is* included in the efficiency
    accounting, because retrieval's cost scales with the training set and the
    surrogate's does not — which is one of the honest arguments for the surrogate.
    """

    def __init__(self, k: int = 8) -> None:
        self.k = k
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> ScenarioRetrieval:
        self.mean = x.mean(axis=0)
        sd = x.std(axis=0)
        self.std = np.where(sd < 1e-8, 1.0, sd)
        self.x = ((x - self.mean) / self.std).astype(np.float32)
        self.y = y.astype(np.float64)
        return self

    def predict(self, x: np.ndarray, chunk: int = 512) -> np.ndarray:
        if self.x is None or self.y is None:
            raise RuntimeError("model used before fit")
        q = ((x - self.mean) / self.std).astype(np.float32)
        k = min(self.k, self.x.shape[0])
        out = np.empty(q.shape[0], dtype=np.float64)
        for start in range(0, q.shape[0], chunk):
            block = q[start : start + chunk]
            d2 = (
                (block**2).sum(1)[:, None]
                - 2.0 * block @ self.x.T
                + (self.x**2).sum(1)[None, :]
            )
            idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
            take = np.take_along_axis(d2, idx, axis=1)
            w = 1.0 / np.sqrt(np.maximum(take, 1e-12))
            w /= w.sum(axis=1, keepdims=True)
            out[start : start + block.shape[0]] = (self.y[idx] * w).sum(axis=1)
        return np.clip(out, 0.0, None)

    def memory_floats(self) -> int:
        """Stored floats, for the efficiency table's memory column."""
        return 0 if self.x is None else int(self.x.size + self.y.size)  # type: ignore[union-attr]
