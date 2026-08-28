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
    ConstantPredictor,
    DegenerateBaselineError,
    ScenarioRetrieval,
    TabularRiskModel,
    TopologyHeuristic,
    prediction_diversity,
    require_non_degenerate,
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
# The degeneracy guard
#
# These exist because this repository shipped a comparison table whose best-MAE
# entry was the constant zero function, and not one metric in the table revealed
# it. Counting unique predictions is the detector; these tests are the detector's
# tests.
# --------------------------------------------------------------------------- #


def zero_inflated(n=4000, zero_frac=0.925, seed=0):
    """Synthetic data with this project's zero-mass and a learnable signal.

    The signal is in feature 0 and is only present on the nonzero rows, exactly
    like the real target: most disruptions are absorbed, and when one is not, its
    size depends on the features.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8))
    y = np.zeros(n)
    k = int(round((1.0 - zero_frac) * n))
    idx = rng.choice(n, k, replace=False)
    y[idx] = np.clip(0.2 + 0.1 * x[idx, 0], 0.0, 1.0)
    return x, y


def test_prediction_diversity_counts_distinct_values():
    d = prediction_diversity(np.array([0.0, 0.0, 0.0]))
    assert d["n_unique_predictions"] == 1
    assert d["degenerate"] is True
    assert d["pred_range"] == pytest.approx(0.0)
    d = prediction_diversity(np.array([0.0, 1.0, 1.0, 2.0]))
    assert d["n_unique_predictions"] == 3
    assert d["degenerate"] is False
    assert d["pred_range"] == pytest.approx(2.0)


def test_prediction_diversity_ignores_float_noise():
    """1e-15 of jitter is not two models' worth of information."""
    p = np.full(50, 0.25) + np.linspace(0, 1e-15, 50)
    assert prediction_diversity(p)["degenerate"] is True


def test_prediction_diversity_skips_nan():
    d = prediction_diversity(np.array([np.nan, 1.0, 2.0]))
    assert d["n_unique_predictions"] == 2


def test_require_non_degenerate_raises_on_a_constant():
    with pytest.raises(DegenerateBaselineError, match="constant function"):
        require_non_degenerate("pretend_model", np.zeros(1000))


def test_require_non_degenerate_passes_a_real_predictor():
    d = require_non_degenerate("ok", np.linspace(0.0, 1.0, 20))
    assert d["n_unique_predictions"] == 20


def test_gbt_with_absolute_error_collapses_on_a_zero_inflated_target():
    """The bug, reproduced as a test rather than as a paragraph.

    L1's optimal constant is the median; the median of a 92.5%-zero target is 0;
    so the boosting initialisation is 0 and every leaf's L1-optimal value is 0.
    The model emits exactly one distinct prediction and the guard must catch it.
    """
    x, y = zero_inflated()
    m = TabularRiskModel("gbt_l1", 0).fit(x, y)
    p = m.predict(x)
    assert m.degenerate_by_construction
    assert prediction_diversity(p)["n_unique_predictions"] == 1
    assert float(np.abs(p).max()) == 0.0
    # And its MAE is indistinguishable from predicting nothing at all.
    assert np.abs(p - y).mean() == pytest.approx(np.abs(y).mean())
    with pytest.raises(DegenerateBaselineError):
        require_non_degenerate("tabular_gbt_l1", p)


def test_gbt_with_squared_error_actually_fits_the_same_target():
    """The control: same data, same trees, a loss that is not minimised by 0."""
    x, y = zero_inflated()
    m = TabularRiskModel("gbt", 0).fit(x, y)
    p = m.predict(x)
    assert not m.degenerate_by_construction
    assert prediction_diversity(p)["n_unique_predictions"] > 100
    require_non_degenerate("tabular_gbt", p)
    # It beats the all-zero constant where the truth is nonzero, which is the
    # thing the L1 variant cannot do at all.
    nz = y > 0
    assert np.abs(p[nz] - y[nz]).mean() < np.abs(y[nz]).mean()


def test_the_two_gbt_variants_use_the_losses_they_claim_to():
    x, y = zero_inflated(n=500)
    assert TabularRiskModel("gbt").fit(x, y).model.loss == "squared_error"
    assert TabularRiskModel("gbt_l1").fit(x, y).model.loss == "absolute_error"


# --------------------------------------------------------------------------- #
# Trivial baselines
# --------------------------------------------------------------------------- #


def test_constant_zero_predicts_exactly_zero():
    x, y = zero_inflated(n=200)
    m = ConstantPredictor("zero").fit(x, y)
    p = m.predict(x)
    assert p.shape == (200,)
    assert np.all(p == 0.0)
    assert m.degenerate_by_construction


def test_constant_train_mean_predicts_the_training_mean():
    x, y = zero_inflated(n=200)
    m = ConstantPredictor("train_mean").fit(x, y)
    assert m.value == pytest.approx(y.mean())
    assert np.all(m.predict(x[:7]) == pytest.approx(y.mean()))


def test_constant_zero_is_the_mae_optimal_constant_on_this_target():
    """Why the zero baseline is the one that matters, stated as an assertion."""
    _, y = zero_inflated(n=5000)
    zero_mae = np.abs(y - 0.0).mean()
    mean_mae = np.abs(y - y.mean()).mean()
    assert zero_mae < mean_mae
    # No constant does better on MAE than the median, which here is exactly 0.
    assert np.median(y) == 0.0
    for c in (0.001, 0.01, 0.05, 0.2):
        assert np.abs(y - c).mean() >= zero_mae


def test_constant_predictors_reject_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown constant kind"):
        ConstantPredictor("median")


def test_constant_predictor_used_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        ConstantPredictor("zero").predict(np.zeros((3, 2)))


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
