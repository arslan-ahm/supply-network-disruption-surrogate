"""Tests for features, disruption specs, batching and scenario construction.

The feature layer is where the fairness of the whole comparison is decided: the
graph model and the graph-free baselines must see the same information about a
node, differing only in whether they can see the network. Several tests below
exist purely to keep that true.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sndsur.config import load_config
from sndsur.data.disruptions import (
    DISRUPTION_TYPES,
    TYPE_INDEX,
    Disruption,
    DisruptionSet,
    DisruptionSpec,
    sample_disruptions,
)
from sndsur.data.features import (
    EDGE_FEATURE_NAMES,
    N_EDGE_FEATURES,
    N_NODE_FEATURES,
    NODE_FEATURE_NAMES,
    TABULAR_FEATURE_NAMES,
    disruption_node_features,
    edge_features,
    edge_index,
    network_summary,
    node_features,
    static_node_features,
    tabular_row_features,
)
from sndsur.data.scenarios import _sim_config, build_scenario, make_bundle
from sndsur.engine.batching import FeatureNormaliser, collate, iterate_batches
from sndsur.engine.losses import (
    LOGVAR_MIN,
    gaussian_nll,
    masked_l1,
    quantile_loss,
    trajectory_loss,
)

# --------------------------------------------------------------------------- #
# Disruption specs
# --------------------------------------------------------------------------- #


def test_disruption_rejects_a_negative_start():
    with pytest.raises(ValueError, match="start must be >= 0"):
        Disruption("supplier_outage", 0, -1, -1, 5, 1.0)


def test_disruption_rejects_a_zero_duration():
    with pytest.raises(ValueError, match="duration must be >= 1"):
        Disruption("supplier_outage", 0, -1, 0, 0, 1.0)


def test_disruption_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown disruption kind"):
        Disruption("meteor", 0, -1, 0, 1, 1.0)


def test_lane_closure_needs_both_endpoints():
    with pytest.raises(ValueError, match="both endpoints"):
        Disruption("lane_closure", 0, -1, 0, 1, 1.0)


def test_active_window_is_half_open():
    d = Disruption("supplier_outage", 0, -1, 3, 2, 1.0)
    assert not d.active(2)
    assert d.active(3)
    assert d.active(4)
    assert not d.active(5)
    assert d.end == 5


def test_disruption_dict_roundtrip():
    d = Disruption("lane_closure", 1, 2, 3, 4, 0.5)
    assert Disruption.from_dict(d.to_dict()) == d


def test_disruption_set_roundtrip():
    ds = DisruptionSet([Disruption("supplier_outage", 1, -1, 0, 3, 1.0)])
    again = DisruptionSet.from_list(ds.to_list())
    assert again.items == ds.items


def test_disruption_set_start_and_end_span_every_item():
    ds = DisruptionSet([
        Disruption("supplier_outage", 0, -1, 5, 2, 1.0),
        Disruption("capacity_reduction", 1, -1, 1, 3, 0.5),
    ])
    assert ds.start == 1
    assert ds.end == 7


def test_empty_disruption_set_is_inert():
    ds = DisruptionSet([])
    assert ds.capacity_multiplier(0, 0) == pytest.approx(1.0)
    assert ds.demand_multiplier(0, 0) == pytest.approx(1.0)
    assert ds.lane_open(0, 1, 0)
    assert ds.effective_lead_time(0, 1, 0, 3) == 3
    assert ds.describe() == ""


def test_lead_time_inflation_rounds_and_floors_at_one():
    ds = DisruptionSet([Disruption("lead_time_inflation", 0, -1, 0, 5, 2.0)])
    assert ds.effective_lead_time(0, 1, 1, 3) == 9
    assert ds.effective_lead_time(0, 1, 1, 1) == 3
    # Outside the window, unchanged.
    assert ds.effective_lead_time(0, 1, 9, 3) == 3


def test_lead_time_inflation_only_affects_the_targets_outbound_lanes():
    ds = DisruptionSet([Disruption("lead_time_inflation", 0, -1, 0, 5, 1.0)])
    assert ds.effective_lead_time(0, 1, 1, 2) == 4
    assert ds.effective_lead_time(5, 1, 1, 2) == 2


def test_demand_multiplier_composes_multiplicatively():
    ds = DisruptionSet([
        Disruption("demand_spike", 3, -1, 0, 5, 1.0),
        Disruption("demand_spike", 3, -1, 0, 5, 1.0),
    ])
    assert ds.demand_multiplier(3, 1) == pytest.approx(4.0)


def test_severity_one_gives_a_total_outage():
    ds = DisruptionSet([Disruption("supplier_outage", 2, -1, 0, 5, 1.0)])
    assert ds.capacity_multiplier(2, 1) == pytest.approx(0.0)


def test_describe_is_readable():
    ds = DisruptionSet([Disruption("lane_closure", 1, 2, 0, 3, 1.0)])
    text = ds.describe()
    assert "lane_closure" in text and "1->2" in text


@pytest.mark.parametrize("kind", DISRUPTION_TYPES)
def test_sampler_produces_only_the_requested_kind(kind, small_net):
    spec = DisruptionSpec(kinds=(kind,))
    rng = np.random.default_rng(0)
    for _ in range(10):
        ds = sample_disruptions(small_net, spec, rng)
        assert all(d.kind == kind for d in ds.items)
        assert ds.region is not None


def test_sampler_honours_the_multi_point_range(small_net):
    spec = DisruptionSpec(n_points=(2, 3))
    rng = np.random.default_rng(1)
    for _ in range(15):
        assert 2 <= len(sample_disruptions(small_net, spec, rng).items) <= 3


def test_sampler_never_targets_a_demand_point_with_a_supply_outage(small_net):
    spec = DisruptionSpec(kinds=("supplier_outage",))
    rng = np.random.default_rng(2)
    demand = set(small_net.demand_nodes.tolist())
    for _ in range(30):
        for d in sample_disruptions(small_net, spec, rng).items:
            assert d.target not in demand


def test_demand_spikes_only_target_demand_points(small_net):
    spec = DisruptionSpec(kinds=("demand_spike",))
    rng = np.random.default_rng(3)
    demand = set(small_net.demand_nodes.tolist())
    for _ in range(20):
        for d in sample_disruptions(small_net, spec, rng).items:
            assert d.target in demand


def test_lane_closures_target_real_edges(small_net):
    spec = DisruptionSpec(kinds=("lane_closure",))
    rng = np.random.default_rng(4)
    edges = set(small_net.edges())
    for _ in range(20):
        for d in sample_disruptions(small_net, spec, rng).items:
            assert (d.target, d.target2) in edges


def test_sampling_is_reproducible(small_net):
    spec = DisruptionSpec()
    a = sample_disruptions(small_net, spec, np.random.default_rng(7)).to_list()
    b = sample_disruptions(small_net, spec, np.random.default_rng(7)).to_list()
    assert a == b


def test_upstream_bias_actually_biases(small_net):
    rng = np.random.default_rng(5)
    spec = DisruptionSpec(kinds=("supplier_outage",), upstream_bias=0.95)
    tiers = [
        small_net.tier[sample_disruptions(small_net, spec, rng).items[0].target]
        for _ in range(80)
    ]
    assert float(np.mean(np.asarray(tiers) <= 1)) > 0.7


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def test_node_feature_width_matches_the_name_list(small_net):
    ds = DisruptionSet([]).bind(small_net)
    f = node_features(small_net, ds, 30)
    assert f.shape == (small_net.n_nodes, N_NODE_FEATURES)
    assert len(NODE_FEATURE_NAMES) == N_NODE_FEATURES


def test_edge_feature_width_matches_the_name_list(small_net):
    ds = DisruptionSet([]).bind(small_net)
    e = edge_features(small_net, ds)
    assert e.shape == (len(small_net.edges()), N_EDGE_FEATURES)
    assert len(EDGE_FEATURE_NAMES) == N_EDGE_FEATURES


def test_features_are_finite(small_net):
    ds = DisruptionSet([Disruption("supplier_outage", 0, -1, 1, 4, 1.0)]).bind(small_net)
    assert np.all(np.isfinite(node_features(small_net, ds, 30)))
    assert np.all(np.isfinite(edge_features(small_net, ds)))
    assert np.all(np.isfinite(network_summary(small_net)))


def test_demand_point_capacity_slack_does_not_become_inf(small_net):
    """Demand points have infinite capacity; the feature must stay finite."""
    f = static_node_features(small_net)
    assert np.all(np.isfinite(f))


def test_edge_index_matches_the_edge_list(small_net):
    ei = edge_index(small_net)
    assert ei.shape == (2, len(small_net.edges()))
    assert [tuple(c) for c in ei.T.tolist()] == small_net.edges()


def test_edge_index_of_an_edgeless_network():
    from tests.conftest import make_chain

    net = make_chain(n_tiers=2)
    net.groups[1] = []
    net.tier[1] = 0
    assert edge_index(net).shape == (2, 0)


def test_disruption_features_mark_the_right_node(small_net):
    ds = DisruptionSet([Disruption("supplier_outage", 2, -1, 1, 4, 1.0)]).bind(small_net)
    f = disruption_node_features(small_net, ds, 30)
    assert f[2, TYPE_INDEX["supplier_outage"]] == 1.0
    assert f[2, len(DISRUPTION_TYPES)] == pytest.approx(1.0)  # severity
    others = np.delete(np.arange(small_net.n_nodes), 2)
    assert np.all(f[others] == 0.0)


def test_regional_event_features_expand_over_the_region(small_net):
    ds = DisruptionSet([Disruption("regional_event", 0, -1, 0, 4, 0.5)]).bind(small_net)
    f = disruption_node_features(small_net, ds, 30)
    members = np.flatnonzero(small_net.region == 0)
    assert members.size > 1
    assert np.all(f[members, TYPE_INDEX["regional_event"]] == 1.0)
    assert np.all(f[np.flatnonzero(small_net.region != 0)].sum() == 0.0)


def test_lane_closure_features_mark_both_endpoints(small_net):
    u, v = small_net.edges()[0]
    ds = DisruptionSet([Disruption("lane_closure", u, v, 0, 4, 1.0)]).bind(small_net)
    f = disruption_node_features(small_net, ds, 30)
    n = len(DISRUPTION_TYPES)
    assert f[u, n + 4] == 1.0
    assert f[v, n + 5] == 1.0


def test_lane_closed_edge_feature_is_set(small_net):
    u, v = small_net.edges()[0]
    ds = DisruptionSet([Disruption("lane_closure", u, v, 0, 4, 1.0)]).bind(small_net)
    e = edge_features(small_net, ds)
    col = EDGE_FEATURE_NAMES.index("lane_closed")
    assert e[0, col] == 1.0
    assert e[1:, col].sum() == 0.0


def test_no_disruption_leaves_the_disruption_block_zero(small_net):
    f = disruption_node_features(small_net, DisruptionSet([]).bind(small_net), 30)
    assert np.all(f == 0.0)


def test_tabular_rows_are_one_per_demand_point(small_net):
    ds = DisruptionSet([Disruption("supplier_outage", 1, -1, 0, 4, 1.0)]).bind(small_net)
    rows = tabular_row_features(small_net, ds, 30)
    assert rows.shape == (small_net.demand_nodes.size, len(TABULAR_FEATURE_NAMES))


def test_tabular_rows_share_the_source_block(small_net):
    """Every row of a scenario describes the same disrupted node."""
    ds = DisruptionSet([Disruption("supplier_outage", 1, -1, 0, 4, 1.0)]).bind(small_net)
    rows = tabular_row_features(small_net, ds, 30)
    src_width = N_NODE_FEATURES
    assert np.allclose(rows[:, :src_width], rows[0, :src_width])


def test_tabular_rows_differ_in_the_destination_block(small_net):
    ds = DisruptionSet([Disruption("supplier_outage", 1, -1, 0, 4, 1.0)]).bind(small_net)
    rows = tabular_row_features(small_net, ds, 30)
    dst = rows[:, N_NODE_FEATURES : 2 * N_NODE_FEATURES]
    assert not np.allclose(dst, dst[0])


def test_tabular_view_contains_the_same_node_features_as_the_graph_view(small_net):
    """Fairness check: the baseline sees the identical node feature vector."""
    ds = DisruptionSet([Disruption("supplier_outage", 1, -1, 0, 4, 1.0)]).bind(small_net)
    nf = node_features(small_net, ds, 30)
    rows = tabular_row_features(small_net, ds, 30, nf=nf)
    assert np.allclose(rows[0, :N_NODE_FEATURES], nf[1])
    j = int(small_net.demand_nodes[0])
    assert np.allclose(rows[0, N_NODE_FEATURES : 2 * N_NODE_FEATURES], nf[j])


def test_static_features_do_not_depend_on_the_disruption(small_net):
    a = node_features(small_net, DisruptionSet([]).bind(small_net), 30)
    b = node_features(
        small_net,
        DisruptionSet([Disruption("supplier_outage", 0, -1, 0, 4, 1.0)]).bind(small_net),
        30,
    )
    static_width = N_NODE_FEATURES - (len(DISRUPTION_TYPES) + 6)
    assert np.allclose(a[:, :static_width], b[:, :static_width])


# --------------------------------------------------------------------------- #
# Batching and normalisation
# --------------------------------------------------------------------------- #


def _tiny_dataset():
    from sndsur.data.scenarios import ScenarioDataset

    cfg = load_config("configs/smoke.yaml")
    sim = _sim_config(cfg)
    from sndsur.data.network import NetworkSpec, generate_network

    net = generate_network(NetworkSpec(n_per_tier=(4, 3, 3, 2, 3), n_regions=2), seed=0)
    bundle = make_bundle(net)
    scs = []
    rng = np.random.default_rng(0)
    for i in range(6):
        ds = sample_disruptions(net, DisruptionSpec(), rng)
        scs.append(build_scenario(bundle, 0, i, ds, 1, sim, 5))
    return ScenarioDataset([bundle], {"train": scs, "val": scs[:2]}, 5)


def test_collate_offsets_edge_indices():
    ds = _tiny_dataset()
    b = collate(ds.splits["train"][:3], ds)
    n_per = ds.splits["train"][0].node_feat.shape[0]
    assert b.n_graphs == 3
    assert b.n_nodes == 3 * n_per
    assert int(b.edge_index.max()) < b.n_nodes
    assert int(b.demand_rows.max()) < b.n_nodes


def test_collate_preserves_row_order():
    ds = _tiny_dataset()
    scs = ds.splits["train"][:2]
    b = collate(scs, ds)
    expected = np.concatenate([s.service_loss for s in scs])
    assert np.allclose(b.impact.numpy(), expected)


def test_collate_graph_of_row_is_correct():
    ds = _tiny_dataset()
    scs = ds.splits["train"][:3]
    b = collate(scs, ds)
    counts = np.bincount(b.graph_of_row.numpy(), minlength=3)
    assert counts.tolist() == [s.n_demand for s in scs]


def test_iterate_batches_covers_every_scenario():
    ds = _tiny_dataset()
    bs = iterate_batches(ds.splits["train"], ds, 4)
    assert sum(b.n_graphs for b in bs) == len(ds.splits["train"])


def test_iterate_batches_shuffle_changes_order_not_content():
    ds = _tiny_dataset()
    a = iterate_batches(ds.splits["train"], ds, 2, shuffle=False)
    b = iterate_batches(ds.splits["train"], ds, 2, shuffle=True, seed=3)
    ids_a = sorted(int(x) for bb in a for x in bb.scenario_ids)
    ids_b = sorted(int(x) for bb in b for x in bb.scenario_ids)
    assert ids_a == ids_b


def test_normaliser_standardises():
    x = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])
    n = FeatureNormaliser().fit([x])
    out = n.transform(x)
    assert out[:, 0].mean() == pytest.approx(0.0, abs=1e-9)
    assert out[:, 0].std() == pytest.approx(1.0, abs=1e-6)


def test_normaliser_leaves_constant_columns_alone():
    """Dividing a zero-variance column by epsilon would amplify float noise."""
    x = np.array([[1.0, 7.0], [3.0, 7.0]])
    n = FeatureNormaliser().fit([x])
    assert np.all(np.isfinite(n.transform(x)))
    assert n.std[1] == pytest.approx(1.0)


def test_normaliser_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        FeatureNormaliser().transform(np.zeros((2, 2)))


def test_normaliser_state_dict_roundtrip():
    x = np.random.default_rng(0).normal(size=(20, 4))
    a = FeatureNormaliser().fit([x])
    b = FeatureNormaliser().load_state_dict(a.state_dict())
    assert np.allclose(a.transform(x), b.transform(x))


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #


def test_masked_l1_ignores_nan_targets_and_counts_the_rest():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, float("nan"), 5.0])
    loss, n = masked_l1(pred, target)
    assert n == 2
    assert loss.item() == pytest.approx(1.0)


def test_masked_l1_with_no_valid_rows_is_zero_and_differentiable():
    pred = torch.tensor([1.0, 2.0], requires_grad=True)
    loss, n = masked_l1(pred, torch.tensor([float("nan"), float("nan")]))
    assert n == 0
    loss.backward()
    assert pred.grad is not None


def test_gaussian_nll_is_minimised_at_the_true_variance():
    target = torch.tensor([0.0])
    pred = torch.tensor([0.0])
    losses = [
        gaussian_nll(pred, torch.tensor([lv]), target + 1.0).item()
        for lv in (-2.0, 0.0, 2.0)
    ]
    # With a residual of 1, the NLL is minimised at logvar = 0.
    assert losses[1] < losses[0]
    assert losses[1] < losses[2]


def test_gaussian_nll_clamps_extreme_logvar():
    v = gaussian_nll(
        torch.tensor([0.0]), torch.tensor([-1000.0]), torch.tensor([1.0])
    )
    assert torch.isfinite(v)
    assert v.item() <= 0.5 * (LOGVAR_MIN + np.exp(-LOGVAR_MIN)) + 1e-3


def test_quantile_loss_is_asymmetric():
    """An under-prediction costs a high quantile more than an over-prediction."""
    target = torch.tensor([1.0])
    under = quantile_loss(torch.tensor([[0.0]]), target, (0.9,))
    over = quantile_loss(torch.tensor([[2.0]]), target, (0.9,))
    assert under > over


def test_quantile_loss_is_symmetric_at_the_median():
    target = torch.tensor([1.0])
    a = quantile_loss(torch.tensor([[0.0]]), target, (0.5,))
    b = quantile_loss(torch.tensor([[2.0]]), target, (0.5,))
    assert a.item() == pytest.approx(b.item())


def test_trajectory_loss_weights_early_periods_more():
    true = torch.zeros(1, 8)
    early = torch.zeros(1, 8)
    early[0, 0] = 1.0
    late = torch.zeros(1, 8)
    late[0, 7] = 1.0
    assert trajectory_loss(early, true) > trajectory_loss(late, true)


def test_trajectory_loss_handles_a_length_mismatch():
    """The model may predict fewer periods than the simulator recorded."""
    pred = torch.zeros(2, 4)
    true = torch.zeros(2, 10)
    assert torch.isfinite(trajectory_loss(pred, true))


def test_trajectory_loss_of_a_perfect_prediction_is_zero():
    t = torch.rand(3, 6)
    assert trajectory_loss(t, t).item() == pytest.approx(0.0)
