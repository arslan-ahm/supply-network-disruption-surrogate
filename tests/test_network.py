"""Tests for the network representation and generator.

The topology quantities tested here (sole-source reach, downstream reach,
betweenness, throughput propagation) are the inputs to the heuristic baseline and
several node features, so an error in any of them would quietly move both the
baseline and the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from sndsur.data.network import (
    N_TIERS,
    InputGroup,
    NetworkSpec,
    SupplyNetwork,
    generate_network,
)
from sndsur.models.baselines import path_betweenness
from tests.conftest import make_chain, make_diamond, make_sole_source_hub

# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def test_chain_edges_are_the_expected_pairs(chain):
    assert chain.edges() == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_customers_is_the_reverse_of_edges(small_net):
    cust = small_net.customers()
    pairs = {(u, v) for u, vs in enumerate(cust) for v in vs}
    assert pairs == set(small_net.edges())


def test_group_of_finds_the_right_group(hub):
    assert hub.group_of(3, 7) == 0
    assert hub.group_of(5, 7) == 1
    with pytest.raises(KeyError):
        hub.group_of(0, 7)


def test_share_sums_to_one_within_a_group(small_net):
    for v, gs in enumerate(small_net.groups):
        for g in gs:
            assert sum(g.shares) == pytest.approx(1.0)
            for u in g.suppliers:
                assert small_net.share(u, v) in [pytest.approx(s) for s in g.shares]


def test_sole_source_property(hub):
    assert hub.groups[3][0].sole_source
    assert not hub.groups[6][0].sole_source


def test_demand_nodes_are_the_last_tier(small_net):
    assert np.array_equal(
        small_net.demand_nodes, np.flatnonzero(small_net.tier == N_TIERS - 1)
    )


def test_node_ids_are_in_topological_order(small_net):
    for u, v in small_net.edges():
        assert u < v
        assert small_net.tier[u] < small_net.tier[v]


# --------------------------------------------------------------------------- #
# Derived quantities
# --------------------------------------------------------------------------- #


def test_throughput_propagates_exactly_along_a_chain():
    net = make_chain(n_tiers=5, demand=7.0)
    thr = net.compute_throughput()
    assert np.allclose(thr, 7.0)


def test_throughput_splits_by_sourcing_share(diamond):
    """A 50/50 group sends half the requirement to each supplier."""
    thr = diamond.compute_throughput()
    assert thr[4] == pytest.approx(10.0)
    assert thr[3] == pytest.approx(10.0)
    assert thr[2] == pytest.approx(10.0)
    assert thr[0] == pytest.approx(5.0)
    assert thr[1] == pytest.approx(5.0)


def test_throughput_scales_with_bom_coefficient():
    groups = [[], [InputGroup(3.0, [0], [1.0])]]
    net = SupplyNetwork(
        tier=np.array([0, N_TIERS - 1]),
        region=np.zeros(2, dtype=int),
        groups=groups,
        capacity=np.array([1e4, np.inf]),
        base_stock=np.zeros(2),
        input_cover=np.zeros(2),
        lead_time={(0, 1): 1},
        mean_demand=np.array([0.0, 4.0]),
        demand_cv=np.zeros(2),
    )
    net.validate()
    assert net.compute_throughput()[0] == pytest.approx(12.0)


def test_bom_depth_counts_hops_to_demand(chain):
    assert np.array_equal(chain.bom_depth(), np.array([4, 3, 2, 1, 0]))


def test_lead_time_to_demand_accumulates(chain):
    assert chain.lead_time_to_demand()[0] == pytest.approx(4.0)
    assert chain.min_lead_time_to_demand()[0] == pytest.approx(4.0)


def test_max_and_min_lead_time_differ_when_paths_differ(hub):
    net = make_sole_source_hub()
    net.lead_time[(0, 3)] = 5
    net.lead_time[(0, 4)] = 1
    net.lead_time[(0, 5)] = 1
    assert net.lead_time_to_demand()[0] > net.min_lead_time_to_demand()[0]


def test_downstream_reach_counts_demand_points(hub):
    reach = hub.downstream_reach()
    assert reach[8] == 1.0
    assert reach[0] == 1.0  # only one demand point exists in this fixture


def test_sole_source_reach_is_zero_through_a_dual_sourced_group(hub):
    """Node 1 reaches demand only via a dual-sourced group, so its SS reach is 0."""
    ss = hub.sole_source_reach()
    assert ss[1] == 0.0
    assert ss[2] == 0.0


def test_sole_source_reach_is_positive_through_a_sole_sourced_chain():
    """A pure sole-source chain gives every upstream node full sole-source reach."""
    net = make_chain(n_tiers=5)
    assert np.all(net.sole_source_reach()[:-1] == 1.0)


def test_sole_source_flags_mark_the_supplier_not_the_customer(hub):
    flags = hub.sole_source_flags()
    assert flags[0] == 1.0  # node 0 is the sole member of nodes 3, 4 and 5's group
    assert flags[1] == 0.0  # node 1 shares its group with node 2


def test_betweenness_is_normalised(small_net):
    b = path_betweenness(small_net)
    assert np.all(b >= 0.0)
    assert np.all(b <= 1.0 + 1e-9)


def test_betweenness_is_one_along_a_single_chain(chain):
    b = path_betweenness(chain)
    # Every source-to-demand path passes through every intermediate node.
    assert b[1] == pytest.approx(1.0)
    assert b[2] == pytest.approx(1.0)


def test_betweenness_splits_across_parallel_paths(diamond):
    b = path_betweenness(diamond)
    assert b[2] == pytest.approx(1.0)
    assert b[0] == pytest.approx(0.5)
    assert b[1] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_validate_rejects_a_tier0_node_with_inputs():
    net = make_chain()
    net.groups[0] = [InputGroup(1.0, [0], [1.0])]
    with pytest.raises(ValueError, match="tier-0"):
        net.validate()


def test_validate_rejects_shares_that_do_not_sum_to_one():
    net = make_diamond()
    net.groups[2][0].shares = [0.5, 0.4]
    with pytest.raises(ValueError, match="shares sum"):
        net.validate()


def test_validate_rejects_a_backwards_edge():
    net = make_chain()
    net.groups[1] = [InputGroup(1.0, [2], [1.0])]
    net.lead_time[(2, 1)] = 1
    with pytest.raises(ValueError, match="does not go up an echelon"):
        net.validate()


def test_validate_rejects_a_missing_lead_time():
    net = make_chain()
    del net.lead_time[(0, 1)]
    with pytest.raises(ValueError, match="missing lead time"):
        net.validate()


def test_validate_rejects_a_zero_lead_time():
    net = make_chain()
    net.lead_time[(0, 1)] = 0
    with pytest.raises(ValueError, match="must be >= 1 period"):
        net.validate()


def test_validate_rejects_demand_on_a_producer():
    net = make_chain()
    net.mean_demand[0] = 5.0
    with pytest.raises(ValueError, match="only demand points"):
        net.validate()


def test_validate_rejects_a_non_positive_bom_coefficient():
    net = make_chain()
    net.groups[1][0].coeff = 0.0
    with pytest.raises(ValueError, match="non-positive BOM"):
        net.validate()


def test_validate_rejects_an_empty_group():
    net = make_chain()
    net.groups[1][0].suppliers = []
    net.groups[1][0].shares = []
    with pytest.raises(ValueError, match="empty group"):
        net.validate()


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_generated_networks_validate(seed):
    generate_network(NetworkSpec(), seed=seed).validate()


def test_generation_is_deterministic():
    a = generate_network(NetworkSpec(), seed=11)
    b = generate_network(NetworkSpec(), seed=11)
    assert a.edges() == b.edges()
    assert np.allclose(a.capacity[np.isfinite(a.capacity)], b.capacity[np.isfinite(b.capacity)])
    assert a.lead_time == b.lead_time


def test_different_seeds_give_different_networks():
    a = generate_network(NetworkSpec(), seed=1)
    b = generate_network(NetworkSpec(), seed=2)
    assert a.edges() != b.edges()


def test_node_count_matches_the_spec():
    spec = NetworkSpec(n_per_tier=(7, 6, 5, 4, 3))
    net = generate_network(spec, seed=0)
    assert net.n_nodes == 25
    for t, c in enumerate(spec.n_per_tier):
        assert int((net.tier == t).sum()) == c


def test_scaled_spec_multiplies_every_tier():
    spec = NetworkSpec(n_per_tier=(10, 8, 6, 4, 4))
    big = spec.scaled(2.0)
    assert big.n_per_tier == (20, 16, 12, 8, 8)
    # and nothing else changed
    assert big.sole_source_prob == spec.sole_source_prob
    assert big.lead_time_cross_region == spec.lead_time_cross_region


def test_scaled_spec_keeps_at_least_two_nodes_per_tier():
    spec = NetworkSpec(n_per_tier=(4, 3, 2, 2, 2))
    assert min(spec.scaled(0.1).n_per_tier) >= 2


def test_capacity_is_sized_from_throughput_not_sampled():
    """Capacity slack must stay inside the configured band for every producer."""
    spec = NetworkSpec(capacity_slack=(0.2, 0.3))
    net = generate_network(spec, seed=5)
    thr = net.compute_throughput()
    prod = (net.tier < N_TIERS - 1) & (thr > 1e-9)
    slack = net.capacity[prod] / thr[prod] - 1.0
    assert np.all(slack >= 0.2 - 1e-9)
    assert np.all(slack <= 0.3 + 1e-9)


def test_demand_points_have_infinite_capacity(small_net):
    assert np.all(np.isinf(small_net.capacity[small_net.demand_nodes]))
    assert np.all(np.isfinite(small_net.capacity[small_net.tier < N_TIERS - 1]))


def test_sole_source_probability_is_approximately_honoured():
    """The knob has to actually control the sole-source fraction."""
    fracs = {}
    for p in (0.1, 0.8):
        counts = []
        for seed in range(6):
            net = generate_network(NetworkSpec(sole_source_prob=p), seed=seed)
            groups = [g for gs in net.groups for g in gs]
            counts.append(sum(g.sole_source for g in groups) / len(groups))
        fracs[p] = float(np.mean(counts))
    assert fracs[0.1] < 0.35
    assert fracs[0.8] > 0.65
    assert fracs[0.8] > fracs[0.1]


def test_dirichlet_shares_are_not_degenerate():
    """Dirichlet(4) must not produce many 97/3 'dual-sourced' groups."""
    net = generate_network(NetworkSpec(sole_source_prob=0.0), seed=2)
    multi = [g for gs in net.groups for g in gs if not g.sole_source]
    assert multi
    lopsided = sum(1 for g in multi if max(g.shares) > 0.9)
    assert lopsided / len(multi) < 0.15


def test_regional_clustering_beats_chance():
    """same_region_prob above 1/n_regions must produce real clustering."""
    spec = NetworkSpec(n_regions=4, same_region_prob=0.75)
    same, total = 0, 0
    for seed in range(6):
        net = generate_network(spec, seed=seed)
        for u, v in net.edges():
            same += int(net.region[u] == net.region[v])
            total += 1
    assert same / total > 0.30  # chance is 0.25


def test_demand_points_have_unit_bom_coefficient(small_net):
    for v in small_net.demand_nodes:
        for g in small_net.groups[int(v)]:
            assert g.coeff == pytest.approx(1.0)


def test_larger_networks_have_more_edges():
    small = generate_network(NetworkSpec(n_per_tier=(8, 7, 5, 3, 4)), seed=0)
    big = generate_network(NetworkSpec(n_per_tier=(16, 14, 10, 6, 8)), seed=0)
    assert len(big.edges()) > len(small.edges())


def test_input_group_to_dict_roundtrips():
    g = InputGroup(1.5, [1, 2], [0.4, 0.6])
    d = g.to_dict()
    assert d["coeff"] == pytest.approx(1.5)
    assert d["suppliers"] == [1, 2]
    assert sum(d["shares"]) == pytest.approx(1.0)
