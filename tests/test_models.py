"""Tests for the message passing, the surrogate and the baselines.

The invariants asserted here are the ones that make the batching correct:
permutation equivariance, no leakage between graphs in a batch, and identical
single-graph and batched inference. A violation of any of them would produce a
model that trains fine and screens wrongly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sndsur.data.features import N_EDGE_FEATURES, N_NODE_FEATURES
from sndsur.models.baselines import (
    ScenarioRetrieval,
    TabularRiskModel,
    TopologyHeuristic,
)
from sndsur.models.layers import MessagePassingLayer, mlp, scatter_max, scatter_mean
from sndsur.models.surrogate import SupplyGraphSurrogate

torch.manual_seed(0)


def make_model(**kw) -> SupplyGraphSurrogate:
    kw.setdefault("traj_periods", 6)
    kw.setdefault("hidden", 16)
    return SupplyGraphSurrogate(N_NODE_FEATURES, N_EDGE_FEATURES, **kw)


def make_graph(n=20, e=40, r=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    nf = torch.randn(n, N_NODE_FEATURES, generator=g)
    ei = torch.randint(0, n, (2, e), generator=g)
    ef = torch.randn(e, N_EDGE_FEATURES, generator=g)
    dr = torch.arange(n - r, n)
    return nf, ei, ef, dr


# --------------------------------------------------------------------------- #
# Scatter operations
# --------------------------------------------------------------------------- #


def test_scatter_mean_matches_a_manual_loop():
    src = torch.tensor([[1.0], [3.0], [5.0]])
    idx = torch.tensor([0, 0, 1])
    out = scatter_mean(src, idx, 3)
    assert out[0].item() == pytest.approx(2.0)
    assert out[1].item() == pytest.approx(5.0)
    assert out[2].item() == pytest.approx(0.0)


def test_scatter_max_matches_a_manual_loop():
    src = torch.tensor([[1.0], [3.0], [5.0]])
    idx = torch.tensor([0, 0, 1])
    out = scatter_max(src, idx, 3)
    assert out[0].item() == pytest.approx(3.0)
    assert out[1].item() == pytest.approx(5.0)


def test_empty_groups_are_zero_not_nan_or_inf():
    """A tier-0 supplier has no inputs and must not poison the representation."""
    src = torch.randn(4, 3)
    idx = torch.zeros(4, dtype=torch.long)
    for fn in (scatter_mean, scatter_max):
        out = fn(src, idx, 5)
        assert torch.isfinite(out).all()
        assert torch.allclose(out[1:], torch.zeros(4, 3))


def test_scatter_handles_no_edges_at_all():
    src = torch.zeros(0, 3)
    idx = torch.zeros(0, dtype=torch.long)
    for fn in (scatter_mean, scatter_max):
        out = fn(src, idx, 4)
        assert out.shape == (4, 3)
        assert torch.allclose(out, torch.zeros(4, 3))


def test_scatter_max_differs_from_mean_on_a_skewed_group():
    """The two aggregators must actually carry different information."""
    src = torch.tensor([[0.0], [0.0], [9.0]])
    idx = torch.zeros(3, dtype=torch.long)
    assert scatter_mean(src, idx, 1)[0].item() == pytest.approx(3.0)
    assert scatter_max(src, idx, 1)[0].item() == pytest.approx(9.0)


def test_scatter_ops_are_differentiable():
    src = torch.randn(6, 2, requires_grad=True)
    idx = torch.tensor([0, 0, 1, 1, 2, 2])
    for fn in (scatter_mean, scatter_max):
        out = fn(src, idx, 3)
        out.sum().backward()
        assert src.grad is not None
        src.grad = None


def test_mlp_shapes_and_no_trailing_activation():
    m = mlp([4, 8, 3])
    assert m(torch.randn(5, 4)).shape == (5, 3)
    assert isinstance(m[-1], torch.nn.Linear)


def test_mlp_with_out_act_ends_in_an_activation():
    m = mlp([4, 8, 3], out_act=True)
    assert not isinstance(m[-1], torch.nn.Linear)


# --------------------------------------------------------------------------- #
# Message passing
# --------------------------------------------------------------------------- #


def test_message_passing_preserves_shape():
    lay = MessagePassingLayer(16, 8)
    h = torch.randn(12, 16)
    ei = torch.randint(0, 12, (2, 30))
    assert lay(h, ei, torch.randn(30, 8)).shape == (12, 16)


def test_unidirectional_layer_has_fewer_parameters():
    a = MessagePassingLayer(16, 8, bidirectional=True)
    b = MessagePassingLayer(16, 8, bidirectional=False)
    assert sum(p.numel() for p in b.parameters()) < sum(p.numel() for p in a.parameters())


def test_message_passing_is_permutation_equivariant():
    """Relabelling nodes must permute the output, not change it."""
    torch.manual_seed(1)
    lay = MessagePassingLayer(8, 0, bidirectional=True).eval()
    n = 10
    h = torch.randn(n, 8)
    ei = torch.randint(0, n, (2, 25))
    perm = torch.randperm(n)
    inv = torch.argsort(perm)
    with torch.no_grad():
        a = lay(h, ei, None)
        b = lay(h[perm], inv[ei], None)
    assert torch.allclose(a[perm], b, atol=1e-5)


def test_message_passing_respects_direction():
    """With no reverse pass, an isolated sink cannot influence its source."""
    torch.manual_seed(2)
    lay = MessagePassingLayer(8, 0, bidirectional=False).eval()
    h = torch.randn(3, 8)
    ei = torch.tensor([[0], [1]])  # 0 supplies 1; node 2 is isolated
    with torch.no_grad():
        out_a = lay(h, ei, None)
        h2 = h.clone()
        h2[1] += 5.0  # perturb the customer
        out_b = lay(h2, ei, None)
    # The supplier's own update must be unchanged by its customer.
    assert torch.allclose(out_a[0], out_b[0], atol=1e-6)


def test_bidirectional_layer_does_propagate_upstream():
    torch.manual_seed(3)
    lay = MessagePassingLayer(8, 0, bidirectional=True).eval()
    h = torch.randn(3, 8)
    ei = torch.tensor([[0], [1]])
    with torch.no_grad():
        a = lay(h, ei, None)
        h2 = h.clone()
        h2[1] += 5.0
        b = lay(h2, ei, None)
    assert not torch.allclose(a[0], b[0], atol=1e-4)


# --------------------------------------------------------------------------- #
# Surrogate
# --------------------------------------------------------------------------- #


def test_forward_output_shapes():
    m = make_model().eval()
    nf, ei, ef, dr = make_graph()
    with torch.no_grad():
        o = m(nf, ei, ef, dr)
    assert o.impact.shape == (4,)
    assert o.traj.shape == (4, 6)
    assert o.logvar.shape == (4,)
    assert o.quantiles.shape == (4, 3)
    assert o.time_to_impact.shape == (4,)
    assert o.node_embedding.shape == (20, 16)


def test_impact_and_timing_are_non_negative():
    """Softplus heads: service loss and periods cannot be negative."""
    m = make_model().eval()
    nf, ei, ef, dr = make_graph()
    with torch.no_grad():
        o = m(nf, ei, ef, dr)
    assert (o.impact >= 0).all()
    assert (o.traj >= 0).all()
    assert (o.time_to_impact >= 0).all()
    assert (o.recovery_time >= 0).all()


def test_layers_zero_removes_all_neighbour_information():
    """The graph ablation must be exactly that: no node sees a neighbour."""
    m = make_model(layers=0, no_graph_blocks=1, no_graph_width=16).eval()
    nf, ei, ef, dr = make_graph()
    other = torch.randint(0, 20, (2, 40))
    with torch.no_grad():
        a = m(nf, ei, ef, dr).impact
        b = m(nf, other, ef, dr).impact
    assert torch.allclose(a, b)


def test_full_model_does_use_the_edges():
    m = make_model(layers=2).eval()
    nf, ei, ef, dr = make_graph()
    other = torch.randint(0, 20, (2, 40), generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        a = m(nf, ei, ef, dr).impact
        b = m(nf, other, ef, dr).impact
    assert not torch.allclose(a, b, atol=1e-5)


def test_edge_features_matter_when_enabled():
    m = make_model(layers=2, use_edge_features=True).eval()
    nf, ei, ef, dr = make_graph()
    with torch.no_grad():
        a = m(nf, ei, ef, dr).impact
        b = m(nf, ei, ef * 3.0 + 1.0, dr).impact
    assert not torch.allclose(a, b, atol=1e-5)


def test_edge_features_are_ignored_when_disabled():
    m = make_model(layers=2, use_edge_features=False).eval()
    nf, ei, ef, dr = make_graph()
    with torch.no_grad():
        a = m(nf, ei, ef, dr).impact
        b = m(nf, ei, torch.randn_like(ef), dr).impact
    assert torch.allclose(a, b)


def test_batching_two_graphs_matches_running_them_separately():
    """The single most important batching invariant: no leakage between graphs."""
    m = make_model(layers=2).eval()
    n1, e1, f1, d1 = make_graph(n=12, e=25, r=3, seed=1)
    n2, e2, f2, d2 = make_graph(n=15, e=30, r=4, seed=2)
    with torch.no_grad():
        a = m(n1, e1, f1, d1).impact
        b = m(n2, e2, f2, d2).impact
        cat = m(
            torch.cat([n1, n2]),
            torch.cat([e1, e2 + 12], dim=1),
            torch.cat([f1, f2]),
            torch.cat([d1, d2 + 12]),
        ).impact
    assert torch.allclose(cat[:3], a, atol=1e-5)
    assert torch.allclose(cat[3:], b, atol=1e-5)


def test_model_handles_a_graph_with_no_edges():
    m = make_model(layers=2).eval()
    nf, _, _, dr = make_graph(n=8, r=2)
    with torch.no_grad():
        o = m(nf, torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, N_EDGE_FEATURES), dr)
    assert torch.isfinite(o.impact).all()


def test_model_handles_a_single_demand_point():
    m = make_model(layers=2).eval()
    nf, ei, ef, _ = make_graph(n=10, e=20)
    with torch.no_grad():
        o = m(nf, ei, ef, torch.tensor([9]))
    assert o.impact.shape == (1,)


def test_no_heteroscedastic_head_gives_none():
    m = make_model(heteroscedastic=False).eval()
    nf, ei, ef, dr = make_graph()
    with torch.no_grad():
        assert m(nf, ei, ef, dr).logvar is None


def test_no_quantile_head_gives_none():
    m = make_model(n_quantiles=0).eval()
    nf, ei, ef, dr = make_graph()
    with torch.no_grad():
        assert m(nf, ei, ef, dr).quantiles is None


def test_linear_and_gru_decoders_both_give_the_right_shape():
    for temporal in (True, False):
        m = make_model(temporal_decoder=temporal).eval()
        nf, ei, ef, dr = make_graph()
        with torch.no_grad():
            assert m(nf, ei, ef, dr).traj.shape == (4, 6)


def test_no_graph_ablation_has_a_comparable_parameter_budget():
    """Otherwise the ablation confounds 'the graph helps' with 'more parameters help'."""
    full = SupplyGraphSurrogate(N_NODE_FEATURES, N_EDGE_FEATURES, 12, hidden=48, layers=3)
    nomp = SupplyGraphSurrogate(
        N_NODE_FEATURES, N_EDGE_FEATURES, 12, hidden=48, layers=0,
        no_graph_blocks=5, no_graph_width=256,
    )
    ratio = nomp.n_params() / full.n_params()
    assert 0.75 < ratio < 1.35, ratio


def test_gradients_reach_every_parameter():
    m = make_model(layers=2)
    nf, ei, ef, dr = make_graph()
    o = m(nf, ei, ef, dr)
    (o.impact.sum() + o.traj.sum() + o.time_to_impact.sum()
     + o.logvar.sum() + o.quantiles.sum()).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, missing


def test_dropout_is_inactive_in_eval_mode():
    m = make_model(layers=2, dropout=0.5).eval()
    nf, ei, ef, dr = make_graph()
    with torch.no_grad():
        assert torch.allclose(m(nf, ei, ef, dr).impact, m(nf, ei, ef, dr).impact)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #


def test_tabular_gbt_fits_and_predicts_non_negative():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 6))
    y = np.clip(x[:, 0] * 0.1, 0, None)
    m = TabularRiskModel("gbt", 0, max_iter=30).fit(x, y)
    p = m.predict(x)
    assert p.shape == (300,)
    assert np.all(p >= 0.0)
    assert m.n_params_ > 0


def test_tabular_ridge_recovers_a_linear_signal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 4))
    y = np.clip(2.0 * x[:, 0] + 1.0, 0, None)
    m = TabularRiskModel("ridge", 0).fit(x, y)
    assert np.corrcoef(m.predict(x), y)[0, 1] > 0.9
    assert m.n_params_ == 5


def test_unknown_tabular_kind_raises():
    with pytest.raises(ValueError, match="unknown tabular kind"):
        TabularRiskModel("nope").fit(np.zeros((4, 2)), np.zeros(4))


def test_tabular_used_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        TabularRiskModel().predict(np.zeros((2, 3)))


def test_retrieval_returns_the_neighbours_value():
    x = np.array([[0.0], [1.0], [10.0]])
    y = np.array([0.0, 0.0, 5.0])
    m = ScenarioRetrieval(k=1).fit(x, y)
    assert m.predict(np.array([[9.9]]))[0] == pytest.approx(5.0)


def test_retrieval_is_exact_on_its_own_training_points():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 3))
    y = rng.random(80)
    m = ScenarioRetrieval(k=1).fit(x, y)
    assert np.allclose(m.predict(x), y, atol=1e-6)


def test_retrieval_clips_at_zero_and_reports_memory():
    x = np.array([[0.0], [1.0]])
    m = ScenarioRetrieval(k=2).fit(x, np.array([-1.0, -2.0]))
    assert np.all(m.predict(x) >= 0.0)
    assert m.memory_floats() == 4


def test_retrieval_used_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        ScenarioRetrieval().predict(np.zeros((2, 2)))


def test_retrieval_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 4))
    y = rng.random(60)
    m = ScenarioRetrieval(k=3).fit(x, y)
    assert np.allclose(m.predict(x, chunk=7), m.predict(x, chunk=1000))


def test_heuristic_scores_the_sole_source_hub_above_its_hedged_peer(hub):
    s = TopologyHeuristic().node_scores(hub)
    assert s[0] > s[1]


def test_heuristic_signals_are_all_finite(small_net):
    for name, v in TopologyHeuristic().signals(small_net).items():
        assert np.all(np.isfinite(v)), name


def test_heuristic_is_deterministic(small_net):
    h = TopologyHeuristic()
    assert np.allclose(h.node_scores(small_net), h.node_scores(small_net))


def test_heuristic_handles_a_constant_signal(chain):
    """Standardising a zero-variance signal must not divide by zero."""
    assert np.all(np.isfinite(TopologyHeuristic().node_scores(chain)))
