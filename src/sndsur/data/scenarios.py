"""Scenario dataset: run the simulator once, train on it many times.

A *scenario* is one (network, disruption set, demand seed) triple together with
the simulator's answer: per-demand-point service loss, extra unmet demand,
time-to-impact, recovery time, and the excess-unmet trajectory. Building the
dataset is the only place the simulator is expensive; everything afterwards is
tensor arithmetic.

Splits
------

The split structure *is* the generalisation experiment, so it is worth being
explicit about what each one changes relative to training:

============= ========================= ==================================
split         topology                  disruption
============= ========================= ==================================
``train``     pool A                    single-point, all kinds but the
                                        held-out one
``val``       pool A (same scenarios'   same
              network, disjoint
              scenarios)
``test_id``   pool A, disjoint scenarios same
``shift_topo``pool B (unseen networks)  same
``shift_size``pool C (unseen, ~2x nodes) same
``shift_type``pool B                    **only** the held-out kind
``shift_multi``pool B                   2-3 simultaneous disruptions
============= ========================= ==================================

Each shift split differs from ``test_id`` in exactly one respect, which is what
makes the gap attributable. ``shift_type`` and ``shift_multi`` deliberately reuse
pool B rather than pool A: a reader would otherwise be unable to tell whether
the gap came from the new mechanism or from the new topology, so the topology
shift is held constant and ``shift_topo`` is the reference point for both.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from sndsur.config import Config
from sndsur.data.disruptions import (
    DISRUPTION_TYPES,
    DisruptionSet,
    DisruptionSpec,
    sample_disruptions,
)
from sndsur.data.features import (
    edge_features,
    edge_index,
    network_summary,
    node_features,
    static_node_features,
    tabular_row_features,
)
from sndsur.data.network import NetworkSpec, SupplyNetwork, generate_network
from sndsur.sim.simulator import SimConfig, SimResult, counterfactual, demand_realisation, simulate
from sndsur.utils.seed import child_seed

#: Split names in a fixed order, so tables and figures agree everywhere.
SPLITS = ("train", "val", "test_id", "shift_topo", "shift_size", "shift_type", "shift_multi")
#: The distribution-shift splits, in reporting order.
SHIFT_SPLITS = ("shift_topo", "shift_size", "shift_type", "shift_multi")


@dataclass
class Scenario:
    """One simulated counterfactual, with everything a model needs as input.

    Attributes:
        net_id: Index into the dataset's network list.
        scenario_id: Unique id within the dataset.
        disruption: What failed.
        seed: Demand seed used for both the baseline and the disrupted run.
        node_feat: ``(n_nodes, N_NODE_FEATURES)``.
        edge_feat: ``(n_edges, N_EDGE_FEATURES)``.
        tabular: ``(n_demand, N_TABULAR_FEATURES)`` reference-style view.
        service_loss: ``(n_demand,)`` target — fill-rate loss vs baseline.
        unmet_extra: ``(n_demand,)`` extra unmet units.
        time_to_impact: ``(n_demand,)`` periods, NaN where never affected.
        recovery_time: ``(n_demand,)`` periods, NaN where undefined.
        traj: ``(n_demand, traj_periods)`` excess-unmet, normalised by mean demand.
        demand_rows: Node ids of the demand points, ascending.
    """

    net_id: int
    scenario_id: int
    disruption: DisruptionSet
    seed: int
    node_feat: np.ndarray
    edge_feat: np.ndarray
    tabular: np.ndarray
    service_loss: np.ndarray
    unmet_extra: np.ndarray
    time_to_impact: np.ndarray
    recovery_time: np.ndarray
    traj: np.ndarray
    demand_rows: np.ndarray

    @property
    def n_demand(self) -> int:
        return int(self.service_loss.shape[0])


@dataclass
class NetworkBundle:
    """A network plus everything about it that does not depend on the scenario."""

    net: SupplyNetwork
    static: np.ndarray
    edge_index: np.ndarray
    summary: np.ndarray
    #: Baseline runs keyed by demand seed, so a counterfactual costs one run.
    baselines: dict[int, SimResult]


@dataclass
class ScenarioDataset:
    """Networks plus scenarios, grouped by split."""

    networks: list[NetworkBundle]
    splits: dict[str, list[Scenario]]
    traj_periods: int

    def __len__(self) -> int:
        return sum(len(v) for v in self.splits.values())

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.splits.items()}

    def rows(self) -> dict[str, int]:
        """Per-split count of (scenario, demand point) prediction rows."""
        return {k: int(sum(s.n_demand for s in v)) for k, v in self.splits.items()}

    def save(self, path: str | Path) -> Path:
        """Pickle the dataset.

        Pickle rather than npz because a scenario set is a ragged collection of
        graphs of differing sizes; flattening it into arrays would need an index
        structure that is itself more code than this, with more ways to be wrong.
        The file is a cache, never a published artefact, and is gitignored.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return p

    @staticmethod
    def load(path: str | Path) -> ScenarioDataset:
        with Path(path).open("rb") as fh:
            return pickle.load(fh)  # noqa: S301 - our own cache file only


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def _net_spec(cfg: Config, scale: float = 1.0) -> NetworkSpec:
    g = cfg.netgen
    spec = NetworkSpec(
        n_per_tier=tuple(g.n_per_tier),  # type: ignore[arg-type]
        n_regions=g.n_regions,
        same_region_prob=g.same_region_prob,
        groups_per_node=tuple(g.groups_per_node),  # type: ignore[arg-type]
        sole_source_prob=g.sole_source_prob,
        max_suppliers_per_group=g.max_suppliers_per_group,
        preferential_exponent=g.preferential_exponent,
        lead_time_same_region=tuple(g.lead_time_same_region),  # type: ignore[arg-type]
        lead_time_cross_region=tuple(g.lead_time_cross_region),  # type: ignore[arg-type]
        capacity_slack=tuple(g.capacity_slack),  # type: ignore[arg-type]
        base_stock_periods=tuple(g.base_stock_periods),  # type: ignore[arg-type]
        input_cover_periods=tuple(g.input_cover_periods),  # type: ignore[arg-type]
        bom_coeff=tuple(g.bom_coeff),  # type: ignore[arg-type]
        demand_mean=tuple(g.demand_mean),  # type: ignore[arg-type]
        demand_cv=g.demand_cv,
    )
    return spec.scaled(scale) if scale != 1.0 else spec


def _sim_config(cfg: Config) -> SimConfig:
    s = cfg.sim
    return SimConfig(
        warmup=s.warmup,
        horizon=s.horizon,
        backlog=s.backlog,
        backlog_cap_periods=s.backlog_cap_periods,
        allow_resourcing=s.allow_resourcing,
        allocation=s.allocation,
        record_trajectory=s.record_trajectory,
    )


def _train_kinds(cfg: Config) -> tuple[str, ...]:
    """Disruption kinds available at training time (held-out kind removed)."""
    held = cfg.disgen.holdout_kind
    kinds = tuple(k for k in cfg.disgen.kinds if k != held)
    if not kinds:
        raise ValueError("holdout_kind removed every disruption kind")
    return kinds


def _dis_spec(cfg: Config, kinds: tuple[str, ...], n_points: tuple[int, int]) -> DisruptionSpec:
    d = cfg.disgen
    return DisruptionSpec(
        kinds=kinds,
        n_points=n_points,
        start=tuple(d.start),  # type: ignore[arg-type]
        duration=tuple(d.duration),  # type: ignore[arg-type]
        severity=tuple(d.severity),  # type: ignore[arg-type]
        lead_severity=tuple(d.lead_severity),  # type: ignore[arg-type]
        demand_severity=tuple(d.demand_severity),  # type: ignore[arg-type]
        upstream_bias=d.upstream_bias,
    )


def make_bundle(net: SupplyNetwork) -> NetworkBundle:
    """Precompute the scenario-independent views of a network."""
    net.compute_throughput()
    return NetworkBundle(
        net=net,
        static=static_node_features(net),
        edge_index=edge_index(net),
        summary=network_summary(net),
        baselines={},
    )


def _baseline(bundle: NetworkBundle, sim: SimConfig, seed: int) -> SimResult:
    """Cached undisrupted run for a (network, seed) pair."""
    if seed not in bundle.baselines:
        d = demand_realisation(bundle.net, sim, seed)
        bundle.baselines[seed] = simulate(bundle.net, sim, d, None)
    return bundle.baselines[seed]


def build_scenario(
    bundle: NetworkBundle,
    net_id: int,
    scenario_id: int,
    ds: DisruptionSet,
    seed: int,
    sim: SimConfig,
    traj_periods: int,
) -> Scenario:
    """Simulate one counterfactual and package it as a :class:`Scenario`."""
    net = bundle.net
    base = _baseline(bundle, sim, seed)
    imp = counterfactual(net, sim, ds, seed, baseline=base)

    nf = node_features(net, ds, sim.horizon, static=bundle.static)
    ef = edge_features(net, ds)
    tab = tabular_row_features(net, ds, sim.horizon, nf=nf, summary=bundle.summary)

    dn = net.demand_nodes
    mean_d = np.maximum(net.mean_demand[dn], 1e-9)
    # Normalising the trajectory by mean demand makes it comparable across demand
    # points and across networks with different demand scales; without it the
    # trajectory loss would be dominated by whichever network has the biggest
    # numbers, which is a units artefact rather than a modelling choice.
    traj = imp.unmet_traj[:, :traj_periods] / mean_d[:, None]
    if traj.shape[1] < traj_periods:
        traj = np.pad(traj, ((0, 0), (0, traj_periods - traj.shape[1])))

    return Scenario(
        net_id=net_id,
        scenario_id=scenario_id,
        disruption=ds,
        seed=seed,
        node_feat=nf.astype(np.float32),
        edge_feat=ef.astype(np.float32),
        tabular=tab.astype(np.float32),
        service_loss=imp.service_loss.astype(np.float32),
        unmet_extra=(imp.unmet_extra / mean_d).astype(np.float32),
        time_to_impact=imp.time_to_impact.astype(np.float32),
        recovery_time=imp.recovery_time.astype(np.float32),
        traj=np.clip(traj, 0.0, None).astype(np.float32),
        demand_rows=dn.copy(),
    )


def build_dataset(cfg: Config, progress=None) -> ScenarioDataset:
    """Generate every network and simulate every scenario.

    This is the only expensive step in the repository and it runs once. The
    ``train``/``val``/``test_id`` scenarios are drawn from one stream per network
    and then partitioned, so the three share a topology pool but no scenario.

    Args:
        cfg: Full configuration.
        progress: Optional ``callable(stage, done, total)`` for a progress line.

    Returns:
        A :class:`ScenarioDataset` with all seven splits populated.
    """
    sim = _sim_config(cfg)
    d = cfg.dataset
    base_seed = cfg.run.seed
    tp = d.traj_periods
    kinds_train = _train_kinds(cfg)
    held = (cfg.disgen.holdout_kind,)

    networks: list[NetworkBundle] = []
    pool_a: list[int] = []
    pool_b: list[int] = []
    pool_c: list[int] = []

    for i in range(d.n_train_networks):
        net = generate_network(_net_spec(cfg), child_seed(0, i, base=base_seed), f"train_{i}")
        pool_a.append(len(networks))
        networks.append(make_bundle(net))
    for i in range(d.n_shift_networks):
        net = generate_network(_net_spec(cfg), child_seed(1, i, base=base_seed), f"shift_{i}")
        pool_b.append(len(networks))
        networks.append(make_bundle(net))
    for i in range(d.n_large_networks):
        net = generate_network(
            _net_spec(cfg, cfg.netgen.large_scale), child_seed(2, i, base=base_seed), f"large_{i}"
        )
        pool_c.append(len(networks))
        networks.append(make_bundle(net))

    splits: dict[str, list[Scenario]] = {k: [] for k in SPLITS}
    sid = 0

    # ---- pool A: train / val / test_id ----
    spec_single = _dis_spec(cfg, kinds_train, (1, 1))
    n_per = d.scenarios_per_train_network
    n_val = max(1, int(round(n_per * d.val_fraction)))
    n_tid = max(1, int(round(n_per * d.test_id_fraction)))
    total_a = len(pool_a) * n_per
    done = 0
    for k, net_id in enumerate(pool_a):
        bundle = networks[net_id]
        rng = np.random.default_rng(child_seed(10, k, base=base_seed))
        for j in range(n_per):
            ds = sample_disruptions(bundle.net, spec_single, rng)
            seed = child_seed(11, k, j, base=base_seed)
            sc = build_scenario(bundle, net_id, sid, ds, seed, sim, tp)
            sid += 1
            if j < n_val:
                splits["val"].append(sc)
            elif j < n_val + n_tid:
                splits["test_id"].append(sc)
            else:
                splits["train"].append(sc)
            done += 1
            if progress and done % 200 == 0:
                progress("pool_a", done, total_a)

    # ---- pool B: shift_topo / shift_type / shift_multi ----
    spec_held = _dis_spec(cfg, held, (1, 1))
    spec_multi = _dis_spec(cfg, kinds_train, tuple(cfg.disgen.multi_points))  # type: ignore[arg-type]
    n_eval = d.scenarios_per_eval_network
    for k, net_id in enumerate(pool_b):
        bundle = networks[net_id]
        rng = np.random.default_rng(child_seed(20, k, base=base_seed))
        for name, spec in (
            ("shift_topo", spec_single),
            ("shift_type", spec_held),
            ("shift_multi", spec_multi),
        ):
            for j in range(n_eval):
                ds = sample_disruptions(bundle.net, spec, rng)
                seed = child_seed(21, k, j, base=base_seed)
                splits[name].append(build_scenario(bundle, net_id, sid, ds, seed, sim, tp))
                sid += 1
        if progress:
            progress("pool_b", k + 1, len(pool_b))

    # ---- pool C: shift_size ----
    for k, net_id in enumerate(pool_c):
        bundle = networks[net_id]
        rng = np.random.default_rng(child_seed(30, k, base=base_seed))
        for j in range(n_eval):
            ds = sample_disruptions(bundle.net, spec_single, rng)
            seed = child_seed(31, k, j, base=base_seed)
            splits["shift_size"].append(build_scenario(bundle, net_id, sid, ds, seed, sim, tp))
            sid += 1
        if progress:
            progress("pool_c", k + 1, len(pool_c))

    # Baseline caches hold two float arrays per (network, seed) and there are
    # thousands of seeds; dropping them shrinks the cache file by ~50x and they
    # are cheap to recompute if a caller needs one.
    for b in networks:
        b.baselines = {}

    return ScenarioDataset(networks=networks, splits=splits, traj_periods=tp)


def dataset_path(cfg: Config) -> Path:
    """Cache path keyed by the settings that change the dataset's contents.

    Keyed rather than fixed because a sweep that changes ``sole_source_prob``
    must not silently reuse a cache built with the old value. Anything not in the
    key provably cannot change the scenarios.
    """
    d = cfg.dataset
    key = "_".join(
        str(x)
        for x in (
            cfg.run.seed,
            *cfg.netgen.n_per_tier,
            cfg.netgen.sole_source_prob,
            cfg.netgen.large_scale,
            d.n_train_networks,
            d.n_shift_networks,
            d.n_large_networks,
            d.scenarios_per_train_network,
            d.scenarios_per_eval_network,
            d.traj_periods,
            cfg.sim.warmup,
            cfg.sim.horizon,
            cfg.sim.allow_resourcing,
            cfg.disgen.holdout_kind,
        )
    )
    return Path(d.cache_dir) / f"scenarios_{key}.pkl"


def load_or_build(cfg: Config, rebuild: bool = False, progress=None) -> ScenarioDataset:
    """Load the cached dataset, or build and cache it."""
    p = dataset_path(cfg)
    if p.exists() and not rebuild:
        return ScenarioDataset.load(p)
    ds = build_dataset(cfg, progress=progress)
    ds.save(p)
    return ds


def split_summary(ds: ScenarioDataset) -> list[dict[str, float]]:
    """Per-split descriptive statistics, for ``results/tables/dataset.csv``.

    Reports the fraction of scenarios with essentially zero impact as well as the
    mean, because a dataset where 80% of labels are 0 makes a good MAE trivial
    and the reader needs to know that before reading any error number.
    """
    rows = []
    for name in SPLITS:
        scs = ds.splits[name]
        if not scs:
            continue
        loss = np.concatenate([s.service_loss for s in scs])
        nodes = np.array([s.node_feat.shape[0] for s in scs], dtype=float)
        kinds = [d.kind for s in scs for d in s.disruption.items]
        rows.append(
            {
                "split": name,
                "scenarios": len(scs),
                "rows": int(loss.size),
                "networks": len({s.net_id for s in scs}),
                "mean_nodes": float(nodes.mean()),
                "mean_service_loss": float(loss.mean()),
                "p90_service_loss": float(np.quantile(loss, 0.90)),
                "max_service_loss": float(loss.max()),
                "frac_zero_impact": float((loss <= 1e-4).mean()),
                "mean_disruptions": float(len(kinds) / len(scs)),
                "n_kinds": len(set(kinds)),
            }
        )
    return rows


def kind_breakdown(ds: ScenarioDataset, split: str) -> dict[str, int]:
    """Count of each disruption kind in a split, for the dataset table."""
    out = dict.fromkeys(DISRUPTION_TYPES, 0)
    for s in ds.splits[split]:
        for d in s.disruption.items:
            out[d.kind] += 1
    return out


def with_sim_horizon(cfg: Config, horizon: int) -> Config:
    """A copy of ``cfg`` with a different simulator horizon."""
    return replace(cfg, sim=replace(cfg.sim, horizon=horizon))
