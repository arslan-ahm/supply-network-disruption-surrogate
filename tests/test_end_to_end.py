"""End-to-end tests: the real pipeline at tiny scale.

The specific failure these guard against is a pipeline that runs cleanly and
writes an **empty** results table — which is easy to ship and hard to notice,
because nothing raises. Every test here therefore asserts that the output is
populated and that the numbers in it are in range, not merely that the code
returned.

Anything that trains a model is marked ``slow``; ``pytest -m "not slow"`` skips
them and still covers the seams.
"""

from __future__ import annotations

import numpy as np
import pytest

from sndsur.analysis.counterfactual import (
    aggregate_surrogate_scores,
    explain_node,
    find_disagreements,
    node_candidates,
    probe_scenarios,
    simulate_truth,
    standard_outage,
)
from sndsur.config import load_config
from sndsur.data.scenarios import (
    SHIFT_SPLITS,
    SPLITS,
    ScenarioDataset,
    _sim_config,
    build_dataset,
    dataset_path,
    kind_breakdown,
    split_summary,
)
from sndsur.utils.bench import benchmark, break_even_scenarios, budget_curve
from sndsur.utils.seed import child_seed, seed_everything


@pytest.fixture(scope="module")
def tiny_cfg(tmp_path_factory):
    """Smoke config with the cache redirected into a temp directory."""
    cfg = load_config("configs/smoke.yaml")
    cfg.dataset.cache_dir = str(tmp_path_factory.mktemp("scenarios"))
    cfg.dataset.n_train_networks = 2
    cfg.dataset.n_shift_networks = 2
    cfg.dataset.n_large_networks = 1
    cfg.dataset.scenarios_per_train_network = 16
    cfg.dataset.scenarios_per_eval_network = 6
    cfg.train.epochs = 2
    cfg.model.ensemble = 2
    cfg.eval.bootstrap = 100
    return cfg


@pytest.fixture(scope="module")
def tiny_ds(tiny_cfg) -> ScenarioDataset:
    return build_dataset(tiny_cfg)


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #


def test_every_split_is_populated(tiny_ds):
    counts = tiny_ds.counts()
    for s in SPLITS:
        assert counts[s] > 0, f"split {s} is empty"


def test_split_scenarios_are_disjoint(tiny_ds):
    ids = [s.scenario_id for scs in tiny_ds.splits.values() for s in scs]
    assert len(ids) == len(set(ids))


def test_train_and_shift_pools_share_no_network(tiny_ds):
    train_nets = {s.net_id for s in tiny_ds.splits["train"]}
    for split in ("shift_topo", "shift_size", "shift_type", "shift_multi"):
        assert not (train_nets & {s.net_id for s in tiny_ds.splits[split]}), split


def test_val_and_test_id_reuse_the_training_topologies(tiny_ds):
    """test_id must measure interpolation, so it shares pool A."""
    train_nets = {s.net_id for s in tiny_ds.splits["train"]}
    for split in ("val", "test_id"):
        assert {s.net_id for s in tiny_ds.splits[split]} <= train_nets


def test_shift_type_contains_only_the_held_out_kind(tiny_ds, tiny_cfg):
    held = tiny_cfg.disgen.holdout_kind
    kinds = {d.kind for s in tiny_ds.splits["shift_type"] for d in s.disruption.items}
    assert kinds == {held}


def test_training_never_sees_the_held_out_kind(tiny_ds, tiny_cfg):
    held = tiny_cfg.disgen.holdout_kind
    for split in ("train", "val", "test_id", "shift_topo", "shift_size"):
        kinds = {d.kind for s in tiny_ds.splits[split] for d in s.disruption.items}
        assert held not in kinds, split


def test_shift_multi_has_more_disruptions_per_scenario(tiny_ds):
    single = np.mean([len(s.disruption.items) for s in tiny_ds.splits["train"]])
    multi = np.mean([len(s.disruption.items) for s in tiny_ds.splits["shift_multi"]])
    assert single == pytest.approx(1.0)
    assert multi > 1.5


def test_shift_size_networks_are_larger(tiny_ds):
    train_nodes = np.mean([s.node_feat.shape[0] for s in tiny_ds.splits["train"]])
    big_nodes = np.mean([s.node_feat.shape[0] for s in tiny_ds.splits["shift_size"]])
    assert big_nodes > 1.4 * train_nodes


def test_shift_type_and_shift_multi_reuse_the_topology_shift_pool(tiny_ds):
    """So a gap is attributable to the mechanism, not to a second topology change."""
    b = {s.net_id for s in tiny_ds.splits["shift_topo"]}
    assert {s.net_id for s in tiny_ds.splits["shift_type"]} <= b
    assert {s.net_id for s in tiny_ds.splits["shift_multi"]} <= b


def test_targets_are_in_range(tiny_ds):
    for scs in tiny_ds.splits.values():
        for s in scs:
            assert np.all(s.service_loss >= -1e-9)
            assert np.all(s.service_loss <= 1.0 + 1e-9)
            assert np.all(np.isfinite(s.service_loss))
            assert np.all(s.traj >= -1e-9)


def test_timing_targets_are_nan_or_non_negative(tiny_ds):
    for scs in tiny_ds.splits.values():
        for s in scs:
            for arr in (s.time_to_impact, s.recovery_time):
                finite = np.isfinite(arr)
                assert np.all(arr[finite] >= -1e-9)


def test_some_scenarios_actually_bite(tiny_ds):
    """A dataset of all-zero labels would make every metric meaningless."""
    loss = np.concatenate([s.service_loss for s in tiny_ds.splits["train"]])
    assert float((loss > 1e-4).mean()) > 0.01
    assert loss.max() > 0.05


def test_feature_shapes_are_consistent_within_a_scenario(tiny_ds):
    for scs in tiny_ds.splits.values():
        for s in scs:
            assert s.tabular.shape[0] == s.n_demand
            assert s.traj.shape[0] == s.n_demand
            assert s.edge_feat.shape[0] > 0


def test_split_summary_is_populated(tiny_ds):
    rows = split_summary(tiny_ds)
    assert len(rows) == len(SPLITS)
    for r in rows:
        assert r["scenarios"] > 0
        assert r["rows"] > 0
        assert np.isfinite(r["mean_service_loss"])


def test_kind_breakdown_sums_to_the_disruption_count(tiny_ds):
    total = sum(kind_breakdown(tiny_ds, "train").values())
    assert total == sum(len(s.disruption.items) for s in tiny_ds.splits["train"])


def test_dataset_cache_roundtrips(tiny_ds, tmp_path):
    p = tiny_ds.save(tmp_path / "d.pkl")
    again = ScenarioDataset.load(p)
    assert again.counts() == tiny_ds.counts()
    assert np.allclose(
        again.splits["train"][0].service_loss, tiny_ds.splits["train"][0].service_loss
    )


def test_dataset_path_key_reacts_to_meaningful_settings(tiny_cfg):
    import dataclasses

    a = dataset_path(tiny_cfg)
    b = dataset_path(
        dataclasses.replace(
            tiny_cfg, netgen=dataclasses.replace(tiny_cfg.netgen, sole_source_prob=0.9)
        )
    )
    assert a != b


def test_dataset_path_key_ignores_settings_that_cannot_change_the_data(tiny_cfg):
    import dataclasses

    a = dataset_path(tiny_cfg)
    b = dataset_path(
        dataclasses.replace(tiny_cfg, eval=dataclasses.replace(tiny_cfg.eval, bootstrap=99))
    )
    assert a == b


def test_dataset_generation_is_deterministic(tiny_cfg):
    a = build_dataset(tiny_cfg)
    b = build_dataset(tiny_cfg)
    for split in SPLITS:
        for x, y in zip(a.splits[split], b.splits[split], strict=True):
            assert float(np.abs(x.service_loss - y.service_loss).max()) == 0.0


# --------------------------------------------------------------------------- #
# Counterfactual analysis
# --------------------------------------------------------------------------- #


def test_node_candidates_excludes_demand_points(tiny_ds):
    net = tiny_ds.networks[0].net
    cands = node_candidates(net)
    assert not set(cands.tolist()) & set(net.demand_nodes.tolist())


def test_standard_outage_is_identical_for_every_candidate(tiny_cfg, tiny_ds):
    net = tiny_ds.networks[0].net
    a = standard_outage(1, tiny_cfg).items[0]
    b = standard_outage(2, tiny_cfg).items[0]
    assert (a.start, a.duration, a.severity) == (b.start, b.duration, b.severity)
    assert a.target != b.target
    del net


def test_probe_scenarios_carry_nan_targets(tiny_cfg, tiny_ds):
    """Probes exist to be predicted from; NaN makes it impossible to train on them."""
    bundle = tiny_ds.networks[0]
    cands = node_candidates(bundle.net)[:4]
    sim = _sim_config(tiny_cfg)
    probes = probe_scenarios(bundle, tiny_cfg, sim, 0, cands, 5)
    assert len(probes) == 4
    for p in probes:
        assert np.all(np.isnan(p.service_loss))
        assert np.all(np.isnan(p.traj))


def test_simulate_truth_is_non_negative_and_sized(tiny_cfg, tiny_ds):
    net = tiny_ds.networks[0].net
    cands = node_candidates(net)
    truth = simulate_truth(net, tiny_cfg, _sim_config(tiny_cfg), 0, cands)
    assert truth.shape == cands.shape
    assert np.all(truth >= -1e-12)


def test_simulate_truth_finds_at_least_one_critical_node(tiny_cfg, tiny_ds):
    """If every candidate had zero impact the ranking experiment would be vacuous."""
    net = tiny_ds.networks[0].net
    cands = node_candidates(net)
    truth = simulate_truth(net, tiny_cfg, _sim_config(tiny_cfg), 0, cands)
    assert float(truth.max()) > 1e-4


def test_aggregate_surrogate_scores_sums_over_demand_points():
    impact = np.arange(6.0)  # 3 candidates x 2 demand points
    out = aggregate_surrogate_scores(impact, 3, 2)
    assert out.tolist() == [1.0, 5.0, 9.0]


def test_find_disagreements_returns_nothing_when_rankings_agree():
    cands = np.arange(20)
    scores = np.linspace(1.0, 0.0, 20)
    from tests.conftest import make_sole_source_hub

    out = find_disagreements(make_sole_source_hub(), cands, scores, scores, scores, top_k=5)
    assert out == []


def test_find_disagreements_adjudicates_with_the_simulator():
    from tests.conftest import make_sole_source_hub

    net = make_sole_source_hub()
    cands = np.arange(6)
    feature = np.array([0.0, 0.0, 0.0, 9.0, 8.0, 7.0])  # likes 3,4,5
    counterf = np.array([9.0, 8.0, 7.0, 0.0, 0.0, 0.0])  # likes 0,1,2
    truth = np.array([5.0, 4.0, 3.0, 0.1, 0.1, 0.1])  # the counterfactual is right
    out = find_disagreements(net, cands, feature, counterf, truth, top_k=3, max_pairs=9)
    assert out
    assert all(d.winner == "counterfactual" for d in out)
    # Sorted by the service level actually at stake.
    assert out[0].margin >= out[-1].margin


def test_explain_node_mentions_the_structure(hub):
    text = explain_node(hub, 0)
    assert "tier" in text and "demand points" in text and "sole-sourced" in text


# --------------------------------------------------------------------------- #
# Benchmarking helpers
# --------------------------------------------------------------------------- #


def test_benchmark_enforces_the_warmup_floor():
    t = benchmark(lambda: sum(range(100)), "x", warmup=1, repeats=1)
    assert t.n_warmup >= 8
    assert t.n_repeats >= 5
    assert t.median_ms >= 0.0


def test_benchmark_per_item_divides_by_items():
    t = benchmark(lambda: sum(range(1000)), "x", items=10)
    assert t.per_item_ms == pytest.approx(t.median_ms / 10)


def test_break_even_is_infinite_when_the_surrogate_is_slower():
    out = break_even_scenarios(10.0, 1.0, 5.0, 0.0)
    assert out["break_even_scenarios"] == float("inf")
    assert out["speedup_per_scenario"] < 1.0


def test_break_even_matches_the_closed_form():
    # 100 s setup, saving 9 ms per scenario -> 100000/9 scenarios.
    out = break_even_scenarios(60.0, 10.0, 1.0, 40.0)
    assert out["setup_seconds"] == pytest.approx(100.0)
    assert out["saving_per_scenario_ms"] == pytest.approx(9.0)
    assert out["break_even_scenarios"] == pytest.approx(100_000 / 9)
    assert out["speedup_per_scenario"] == pytest.approx(10.0)


def test_break_even_includes_dataset_generation():
    """Charging only training time is the standard way this gets inflated."""
    with_data = break_even_scenarios(10.0, 10.0, 1.0, 90.0)
    without = break_even_scenarios(10.0, 10.0, 1.0, 0.0)
    assert with_data["break_even_scenarios"] > without["break_even_scenarios"]


def test_budget_curve_is_monotone_for_the_simulator():
    rng = np.random.default_rng(0)
    truth = np.clip(rng.random(60) - 0.5, 0, None)
    rows = budget_curve(truth, truth, 10.0, 5.0, (0.05, 0.2, 1.0, 5.0), k=5, n_repeats=8)
    sim = [r["recall_at_k"] for r in rows if r["method"] == "simulator"]
    assert all(b >= a - 1e-9 for a, b in zip(sim, sim[1:], strict=False))


def test_budget_curve_surrogate_is_zero_below_its_own_cost():
    truth = np.array([0.0] * 10 + [1.0, 0.9, 0.8])
    rows = budget_curve(truth, truth, 10.0, 1000.0, (0.1, 10.0), k=3)
    sur = [r for r in rows if r["method"] == "surrogate"]
    assert sur[0]["recall_at_k"] == pytest.approx(0.0)
    assert sur[1]["recall_at_k"] == pytest.approx(1.0)


def test_budget_curve_reports_candidates_evaluated():
    truth = np.clip(np.random.default_rng(1).random(40) - 0.5, 0, None)
    rows = budget_curve(truth, truth, 10.0, 5.0, (0.1,), k=5, n_repeats=4)
    sim = [r for r in rows if r["method"] == "simulator"][0]
    assert sim["candidates_evaluated"] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #


def test_child_seed_is_stable_and_distinct():
    assert child_seed(1, 2, base=7) == child_seed(1, 2, base=7)
    assert child_seed(1, 2, base=7) != child_seed(1, 3, base=7)
    assert child_seed(1, 2, base=7) != child_seed(1, 2, base=8)


def test_child_seed_is_a_valid_seed():
    s = child_seed(3, 4, base=5)
    assert 0 <= s < 2**63
    np.random.default_rng(s)


def test_seed_everything_makes_numpy_reproducible():
    seed_everything(123)
    a = np.random.rand(5)
    seed_everything(123)
    assert np.allclose(a, np.random.rand(5))


# --------------------------------------------------------------------------- #
# Full pipeline (slow)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_comparison_pipeline_populates_every_table(tiny_cfg, tmp_path, monkeypatch):
    """The guard against shipping an empty results table."""
    import sndsur.pipelines as P

    monkeypatch.setattr(P, "TABLES", tmp_path / "tables")
    tiny_cfg.run.out_dir = str(tmp_path / "runs")
    frames = P.run_comparison(tiny_cfg)

    methods = frames["methods"]
    assert not methods.empty
    assert methods["method"].nunique() >= 6
    assert set(methods["split"]) >= set(SPLITS)
    assert methods["mae"].notna().all()
    assert (methods["mae"] >= 0).all()

    assert not frames["statistics"].empty
    assert not frames["calibration"].empty
    assert not frames["per_row"].empty
    assert frames["per_row"]["truth"].notna().all()
    for s in SHIFT_SPLITS:
        assert f"{s}_gap" in frames["generalisation"].columns

    for name in ("method_comparison.csv", "statistical_tests.csv", "calibration.csv",
                 "generalisation.csv"):
        f = tmp_path / "tables" / name
        assert f.exists() and f.stat().st_size > 0, name


@pytest.mark.slow
def test_surrogate_beats_predicting_the_training_mean(tiny_cfg):
    """A model that cannot beat a constant has learned nothing at all."""
    import sndsur.pipelines as P
    from sndsur.data.features import N_EDGE_FEATURES, N_NODE_FEATURES
    from sndsur.engine.batching import iterate_batches
    from sndsur.engine.trainer import predict, train_surrogate
    from sndsur.metrics.fidelity import mae
    from sndsur.models.baselines import rows_from_split

    ds = P.prepare(tiny_cfg)
    model, _ = train_surrogate(
        tiny_cfg, ds, N_NODE_FEATURES, N_EDGE_FEATURES, log_every=0
    )
    b = iterate_batches(ds.splits["test_id"], ds, tiny_cfg.train.batch_graphs, False)
    p = predict(model, b, ds)
    rows = rows_from_split(ds, "test_id")
    constant = np.full_like(rows.y, rows_from_split(ds, "train").y.mean())
    assert mae(p.impact, rows.y) <= mae(constant, rows.y)


@pytest.mark.slow
def test_training_is_reproducible_from_its_seed(tiny_cfg):
    import sndsur.pipelines as P
    from sndsur.data.features import N_EDGE_FEATURES, N_NODE_FEATURES
    from sndsur.engine.batching import iterate_batches
    from sndsur.engine.trainer import predict, train_surrogate

    ds = P.prepare(tiny_cfg)
    b = iterate_batches(ds.splits["test_id"], ds, tiny_cfg.train.batch_graphs, False)
    outs = []
    for _ in range(2):
        m, _ = train_surrogate(
            tiny_cfg, ds, N_NODE_FEATURES, N_EDGE_FEATURES, seed=5, log_every=0
        )
        outs.append(predict(m, b, ds).impact)
    assert float(np.abs(outs[0] - outs[1]).max()) == 0.0


@pytest.mark.slow
def test_different_seeds_give_different_models(tiny_cfg):
    """A guard against an accidentally constant pipeline."""
    import sndsur.pipelines as P
    from sndsur.data.features import N_EDGE_FEATURES, N_NODE_FEATURES
    from sndsur.engine.batching import iterate_batches
    from sndsur.engine.trainer import predict, train_surrogate

    ds = P.prepare(tiny_cfg)
    b = iterate_batches(ds.splits["test_id"], ds, tiny_cfg.train.batch_graphs, False)
    outs = [
        predict(
            train_surrogate(
                tiny_cfg, ds, N_NODE_FEATURES, N_EDGE_FEATURES, seed=s, log_every=0
            )[0],
            b, ds,
        ).impact
        for s in (1, 2)
    ]
    assert float(np.abs(outs[0] - outs[1]).max()) > 0.0
