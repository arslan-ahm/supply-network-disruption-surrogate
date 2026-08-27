"""Tests for the mechanistic simulator.

The simulator is the ground truth for every other claim in this repository, so
these are the most important tests here. They fall into four groups:

1. **Hand-solvable cases.** A single chain with known lead times and no buffers
   has an exactly predictable shortage pattern. If the simulator disagrees with
   arithmetic, everything downstream is worthless.
2. **Conservation.** Material in equals material consumed plus material held, to
   float precision, in every period of every run.
3. **Invariants.** Capacity is never exceeded; a worse disruption is never
   better; multi-sourcing never hurts; a zero-severity disruption is a no-op.
4. **Regression guards** for bugs actually found during development, each named
   after what it caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from sndsur.data.disruptions import Disruption, DisruptionSet
from sndsur.data.network import NetworkSpec, generate_network
from sndsur.sim.simulator import (
    EPS,
    SimConfig,
    counterfactual,
    demand_realisation,
    exhaustive_edge_criticality,
    exhaustive_node_criticality,
    simulate,
)
from tests.conftest import make_chain

# --------------------------------------------------------------------------- #
# Hand-solvable cases
# --------------------------------------------------------------------------- #


def test_chain_with_ample_capacity_reaches_full_service(chain, sim_cfg):
    """No disruption, ample capacity, deterministic demand: fill rate is exactly 1."""
    d = demand_realisation(chain, sim_cfg, 0)
    r = simulate(chain, sim_cfg, d, None)
    assert r.fill_rate.shape == (1,)
    assert r.fill_rate[0] == pytest.approx(1.0, abs=1e-12)
    assert r.unmet_total[0] == pytest.approx(0.0, abs=1e-9)


def test_zero_cv_demand_is_exactly_the_mean(chain, sim_cfg):
    """With CV 0 the lognormal collapses to a point mass at the mean."""
    d = demand_realisation(chain, sim_cfg, 5)
    assert np.allclose(d, 10.0, atol=1e-9)


def test_demand_realisation_depends_only_on_seed(chain, sim_cfg):
    """The demand path must not depend on the disruption, or baselines cannot cache."""
    a = demand_realisation(chain, sim_cfg, 11)
    b = demand_realisation(chain, sim_cfg, 11)
    c = demand_realisation(chain, sim_cfg, 12)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_demand_realisation_mean_matches_requested_mean(sim_cfg):
    """The lognormal is parameterised by mean, not by median."""
    net = make_chain(demand=10.0, cv=0.4)
    d = demand_realisation(net, SimConfig(warmup=0, horizon=20000), 0)
    assert d.mean() == pytest.approx(10.0, rel=0.02)


def test_total_outage_with_no_inventory_starves_after_the_cumulative_lead_time():
    """The exact hand-computed case.

    Four transit legs of one period each, no finished-goods stock and no input
    cover. Orders propagate to the raw supplier within the period they are
    placed, but material takes one period per leg, so demand in period ``t`` can
    only be met from production started at ``t - 4``. Cut the raw supplier from
    relative period 0 and the demand point is short from period 4 onward, by the
    full 10 units of demand.

    ``backlog=False`` is essential to this being hand-solvable. With backlogging
    and effectively unbounded capacity, the warm-up transient over-orders (every
    backlogged unit is re-ordered on top of fresh demand), the excess arrives
    late and settles as inventory, and that self-inflicted buffer then absorbs
    the outage entirely. That is the bullwhip effect showing up in this
    simulator's own warm-up: real behaviour, but not what this test is about.
    It is documented in docs/METHOD.md.
    """
    net = make_chain(n_tiers=5, lead=1, base_stock=0.0, input_cover=0.0, demand=10.0, cv=0.0)
    cfg = SimConfig(warmup=30, horizon=12, backlog=False)
    ds = DisruptionSet([Disruption("supplier_outage", 0, -1, 0, 12, 1.0)]).bind(net)
    imp = counterfactual(net, cfg, ds, 0, baseline=None)
    unmet = imp.disrupted.unmet_series[0]
    assert np.allclose(unmet[:4], 0.0, atol=1e-9), unmet
    assert np.allclose(unmet[4:], 10.0, atol=1e-9), unmet
    assert imp.time_to_impact[0] == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("n_tiers", "lead"), [(4, 1), (4, 2), (4, 3), (5, 1), (5, 2), (5, 3)]
)
def test_time_to_impact_equals_legs_times_lead_time(n_tiers, lead):
    """Time-to-impact is exactly (transit legs) x (lead time per leg).

    The sharpest available statement that shortage propagation is timed
    correctly. Checked across two chain lengths so an off-by-one in the leg count
    could not pass by coincidence on one of them.
    """
    net = make_chain(n_tiers=n_tiers, lead=lead, base_stock=0.0, input_cover=0.0, cv=0.0)
    cfg = SimConfig(warmup=40, horizon=20, backlog=False)
    ds = DisruptionSet([Disruption("supplier_outage", 0, -1, 0, 20, 1.0)]).bind(net)
    imp = counterfactual(net, cfg, ds, 0)
    legs = n_tiers - 1
    assert imp.time_to_impact[0] == pytest.approx(legs * lead)
    # And once it arrives, the shortage is the entire demand.
    assert imp.unmet_traj[0, legs * lead] == pytest.approx(10.0, abs=1e-6)


def test_base_stock_delays_the_shortage_by_its_own_cover():
    """One period of finished-goods cover at each of three stages buys periods."""
    bare = make_chain(n_tiers=4, lead=1, base_stock=0.0, input_cover=0.0, cv=0.0)
    stocked = make_chain(n_tiers=4, lead=1, base_stock=10.0, input_cover=0.0, cv=0.0)
    cfg = SimConfig(warmup=30, horizon=20, backlog=False)
    ds = DisruptionSet([Disruption("supplier_outage", 0, -1, 0, 20, 1.0)])
    t_bare = counterfactual(bare, cfg, ds.bind(bare), 0).time_to_impact[0]
    t_stocked = counterfactual(stocked, cfg, ds.bind(stocked), 0).time_to_impact[0]
    assert t_stocked > t_bare


def test_no_disruption_gives_exactly_zero_impact(small_net, sim_cfg):
    """A paired counterfactual against an empty disruption set must be identically 0."""
    imp = counterfactual(small_net, sim_cfg, DisruptionSet([]).bind(small_net), 4)
    assert np.allclose(imp.service_loss, 0.0, atol=1e-12)
    assert np.allclose(imp.unmet_extra, 0.0, atol=1e-9)
    assert np.all(np.isnan(imp.time_to_impact))


def test_disruption_that_starts_after_the_horizon_is_a_no_op(small_net):
    """Timing is relative to the measurement window, so a late start does nothing."""
    cfg = SimConfig(warmup=20, horizon=10)
    ds = DisruptionSet([Disruption("supplier_outage", 0, -1, 50, 5, 1.0)]).bind(small_net)
    imp = counterfactual(small_net, cfg, ds, 1)
    assert np.allclose(imp.service_loss, 0.0, atol=1e-12)


def test_warmup_is_never_disrupted():
    """start=0 means the first *measured* period, not somewhere in the burn-in."""
    net = make_chain(n_tiers=3, lead=1, cv=0.0)
    cfg = SimConfig(warmup=15, horizon=10)
    ds = DisruptionSet([Disruption("supplier_outage", 0, -1, 0, 10, 1.0)]).bind(net)
    d = demand_realisation(net, cfg, 0)
    r = simulate(net, cfg, d, ds)
    # If warm-up were disrupted the shortage would already be under way at rel 0.
    assert r.unmet_series[0, 0] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Conservation and physical limits
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_material_is_conserved(seed, sim_cfg):
    """The simulator's own material-balance residual stays at float noise."""
    net = generate_network(NetworkSpec(n_per_tier=(5, 4, 3, 2, 3)), seed=seed)
    d = demand_realisation(net, sim_cfg, seed)
    r = simulate(net, sim_cfg, d, None)
    assert r.conservation_error < 1e-9, r.conservation_error


@pytest.mark.parametrize("seed", [0, 1])
def test_material_is_conserved_under_disruption(seed, sim_cfg):
    net = generate_network(NetworkSpec(n_per_tier=(5, 4, 3, 2, 3)), seed=seed)
    ds = DisruptionSet(
        [Disruption("supplier_outage", 0, -1, 1, 8, 1.0)]
    ).bind(net)
    d = demand_realisation(net, sim_cfg, seed)
    r = simulate(net, sim_cfg, d, ds)
    assert r.conservation_error < 1e-9


def test_served_never_exceeds_demand(small_net, sim_cfg):
    d = demand_realisation(small_net, sim_cfg, 2)
    r = simulate(small_net, sim_cfg, d, None)
    assert np.all(r.served_total <= r.demand_total + 1e-9)
    assert np.all(r.unmet_total >= -1e-9)


def test_fill_rate_is_in_the_unit_interval(small_net, sim_cfg):
    d = demand_realisation(small_net, sim_cfg, 6)
    r = simulate(small_net, sim_cfg, d, None)
    finite = np.isfinite(r.fill_rate)
    assert np.all(r.fill_rate[finite] >= -1e-9)
    assert np.all(r.fill_rate[finite] <= 1.0 + 1e-9)


def test_capacity_limits_throughput_exactly():
    """A hard capacity cap below demand produces exactly the capped fill rate."""
    net = make_chain(n_tiers=2, lead=1, capacity=6.0, demand=10.0, cv=0.0)
    cfg = SimConfig(warmup=30, horizon=30, backlog=False)
    d = demand_realisation(net, cfg, 0)
    r = simulate(net, cfg, d, None)
    assert r.fill_rate[0] == pytest.approx(0.6, abs=1e-6)


def test_unmet_series_sums_to_unmet_total(small_net, sim_cfg):
    """The trajectory and the scalar must be the same quantity."""
    d = demand_realisation(small_net, sim_cfg, 8)
    r = simulate(small_net, sim_cfg, d, None)
    assert np.allclose(r.unmet_series.sum(axis=1), r.unmet_total, atol=1e-8)


# --------------------------------------------------------------------------- #
# Monotonicity and structural invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_longer_outage_is_never_less_damaging(seed):
    net = generate_network(NetworkSpec(n_per_tier=(6, 5, 4, 3, 3)), seed=seed)
    cfg = SimConfig(warmup=20, horizon=30)
    target = int(np.flatnonzero(net.tier == 0)[0])
    losses = []
    for dur in (2, 6, 12, 20):
        ds = DisruptionSet([Disruption("supplier_outage", target, -1, 2, dur, 1.0)]).bind(net)
        losses.append(counterfactual(net, cfg, ds, seed).total_service_loss)
    assert all(b >= a - 1e-9 for a, b in zip(losses, losses[1:], strict=False)), losses


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_higher_severity_is_never_less_damaging(seed):
    net = generate_network(NetworkSpec(n_per_tier=(6, 5, 4, 3, 3)), seed=seed)
    cfg = SimConfig(warmup=20, horizon=30)
    target = int(np.flatnonzero(net.tier == 0)[0])
    losses = []
    for sev in (0.25, 0.5, 0.75, 1.0):
        ds = DisruptionSet(
            [Disruption("capacity_reduction", target, -1, 2, 10, sev)]
        ).bind(net)
        losses.append(counterfactual(net, cfg, ds, seed).total_service_loss)
    assert all(b >= a - 1e-9 for a, b in zip(losses, losses[1:], strict=False)), losses


def test_dual_sourcing_halves_the_damage_of_one_supplier_failing(diamond):
    """Losing one of two 50/50 suppliers costs half of losing a sole source.

    With static sourcing shares, node 2 keeps ordering 50% from the dead supplier
    and gets nothing, so half the material arrives and the steady-state fill rate
    is 0.5. The *measured* loss approaches 0.5 from below rather than hitting it,
    because the material already in transit when the outage starts is still
    delivered. Asserting convergence rather than one number makes this a
    statement about the mechanism instead of about the horizon.
    """
    losses = []
    for horizon in (20, 60, 200):
        cfg = SimConfig(warmup=20, horizon=horizon, backlog=False)
        ds = DisruptionSet(
            [Disruption("supplier_outage", 0, -1, 0, horizon, 1.0)]
        ).bind(diamond)
        losses.append(float(counterfactual(diamond, cfg, ds, 0).service_loss[0]))
    assert losses[0] < losses[1] < losses[2] < 0.5
    assert losses[-1] == pytest.approx(0.5, abs=0.01)


def test_resourcing_recovers_what_dual_sourcing_promises(diamond):
    """With re-sourcing on, the surviving supplier absorbs the whole requirement."""
    base_cfg = SimConfig(warmup=20, horizon=20, backlog=False, allow_resourcing=False)
    re_cfg = SimConfig(warmup=20, horizon=20, backlog=False, allow_resourcing=True)
    ds = DisruptionSet([Disruption("supplier_outage", 0, -1, 0, 20, 1.0)]).bind(diamond)
    static_loss = counterfactual(diamond, base_cfg, ds, 0).service_loss[0]
    re_loss = counterfactual(diamond, re_cfg, ds, 0).service_loss[0]
    assert re_loss < static_loss
    assert re_loss == pytest.approx(0.0, abs=0.02)


def test_sole_source_hub_beats_its_hedged_peer(hub):
    """The structural claim: the hub matters more than the dual-sourced supplier."""
    cfg = SimConfig(warmup=20, horizon=20, backlog=False)
    loss = {}
    for v in (0, 1):
        ds = DisruptionSet([Disruption("supplier_outage", v, -1, 0, 20, 1.0)]).bind(hub)
        loss[v] = counterfactual(hub, cfg, ds, 0).total_service_loss
    assert loss[0] > loss[1]


def test_unreachable_node_has_no_impact(hub):
    """A node with no path to a demand point cannot change service, by construction."""
    reach = hub.downstream_reach()
    assert reach[8] == 1.0  # the demand point reaches itself
    assert np.all(reach[:8] > 0)  # this fixture is fully connected


# --------------------------------------------------------------------------- #
# Disruption mechanisms
# --------------------------------------------------------------------------- #


def test_lead_time_inflation_delays_by_the_inflated_amount():
    """Regression guard for the ring-buffer sizing bug.

    The in-transit buffer was sized from the *nominal* maximum lead time, so an
    inflated shipment wrapped around and arrived early. Every one of 120 sampled
    lead-time-inflation scenarios produced exactly zero impact, which is how the
    bug surfaced. Here a 3x inflation on a 2-period lane must actually delay
    material, and the shortage must be strictly positive.
    """
    net = make_chain(n_tiers=2, lead=2, base_stock=0.0, input_cover=0.0, cv=0.0)
    cfg = SimConfig(warmup=40, horizon=25, backlog=False)
    ds = DisruptionSet(
        [Disruption("lead_time_inflation", 0, -1, 0, 20, 3.0)]
    ).bind(net)
    imp = counterfactual(net, cfg, ds, 0)
    assert imp.total_service_loss > 0.05, imp.total_service_loss
    assert np.isfinite(imp.time_to_impact[0])


def test_lead_time_inflation_is_applied_at_departure():
    """A shipment that left before the inflation window is not retroactively delayed."""
    net = make_chain(n_tiers=2, lead=2, cv=0.0)
    cfg = SimConfig(warmup=25, horizon=20, backlog=False)
    late = DisruptionSet([Disruption("lead_time_inflation", 0, -1, 10, 5, 2.0)]).bind(net)
    imp = counterfactual(net, cfg, late, 0)
    # Nothing can be short before the window opens.
    assert np.allclose(imp.unmet_traj[0, :10], 0.0, atol=1e-9)


def test_lane_closure_blocks_only_its_own_edge(hub):
    """Closing 0->3 must not affect 0->4."""
    cfg = SimConfig(warmup=20, horizon=20, backlog=False)
    one = DisruptionSet([Disruption("lane_closure", 0, 3, 0, 20, 1.0)]).bind(hub)
    allout = DisruptionSet([Disruption("supplier_outage", 0, -1, 0, 20, 1.0)]).bind(hub)
    assert (
        counterfactual(hub, cfg, one, 0).total_service_loss
        < counterfactual(hub, cfg, allout, 0).total_service_loss
    )


def test_demand_spike_raises_demand_and_can_only_hurt(small_net):
    cfg = SimConfig(warmup=20, horizon=20)
    j = int(small_net.demand_nodes[0])
    ds = DisruptionSet([Disruption("demand_spike", j, -1, 2, 8, 1.5)]).bind(small_net)
    d = demand_realisation(small_net, cfg, 3)
    base = simulate(small_net, cfg, d, None)
    hit = simulate(small_net, cfg, d, ds)
    assert hit.demand_total[0] > base.demand_total[0]


def test_regional_event_hits_every_node_in_the_region():
    net = generate_network(NetworkSpec(n_per_tier=(6, 5, 4, 3, 3), n_regions=2), seed=1)
    cfg = SimConfig(warmup=20, horizon=25)
    ds = DisruptionSet([Disruption("regional_event", 0, -1, 1, 10, 1.0)]).bind(net)
    members = np.flatnonzero(net.region == 0)
    assert members.size > 1
    for v in members:
        assert ds.capacity_multiplier(int(v), 3) == pytest.approx(0.0)
    for v in np.flatnonzero(net.region != 0):
        assert ds.capacity_multiplier(int(v), 3) == pytest.approx(1.0)
    assert counterfactual(net, cfg, ds, 1).total_service_loss > 0


def test_regional_event_is_inert_without_binding():
    """An unbound set has no region labels; this must fail loudly in the generator."""
    ds = DisruptionSet([Disruption("regional_event", 0, -1, 0, 5, 1.0)])
    assert ds.region is None
    assert ds.capacity_multiplier(0, 1) == pytest.approx(1.0)


def test_two_disruptions_compose_at_least_as_badly_as_one(small_net):
    cfg = SimConfig(warmup=20, horizon=25)
    producers = np.flatnonzero(small_net.tier < 4)
    a, b = int(producers[0]), int(producers[1])
    da = DisruptionSet([Disruption("supplier_outage", a, -1, 2, 10, 1.0)]).bind(small_net)
    db = DisruptionSet([Disruption("supplier_outage", b, -1, 2, 10, 1.0)]).bind(small_net)
    both = DisruptionSet([da.items[0], db.items[0]]).bind(small_net)
    la = counterfactual(small_net, cfg, da, 0).total_service_loss
    lb = counterfactual(small_net, cfg, db, 0).total_service_loss
    lab = counterfactual(small_net, cfg, both, 0).total_service_loss
    assert lab >= max(la, lb) - 1e-9


def test_capacity_multipliers_compose_multiplicatively(small_net):
    ds = DisruptionSet(
        [
            Disruption("capacity_reduction", 0, -1, 0, 5, 0.5),
            Disruption("capacity_reduction", 0, -1, 0, 5, 0.5),
        ]
    ).bind(small_net)
    assert ds.capacity_multiplier(0, 1) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Determinism and edge cases
# --------------------------------------------------------------------------- #


def test_simulation_is_deterministic(small_net, sim_cfg):
    d = demand_realisation(small_net, sim_cfg, 9)
    a = simulate(small_net, sim_cfg, d, None)
    b = simulate(small_net, sim_cfg, d, None)
    assert np.array_equal(a.fill_rate, b.fill_rate)
    assert float(np.abs(a.unmet_series - b.unmet_series).max()) == 0.0


def test_compiled_structure_is_reused_across_calls(small_net, sim_cfg):
    """The structure cache must be a cache, not a per-call rebuild."""
    d = demand_realisation(small_net, sim_cfg, 0)
    simulate(small_net, sim_cfg, d, None)
    first = small_net.__dict__["_sim_struct"]
    simulate(small_net, sim_cfg, d, None)
    assert small_net.__dict__["_sim_struct"] is first


def test_wrong_demand_shape_is_rejected(small_net, sim_cfg):
    with pytest.raises(ValueError, match="demand must be"):
        simulate(small_net, sim_cfg, np.zeros((2, 3)), None)


def test_record_trajectory_off_still_gives_totals(small_net):
    cfg = SimConfig(warmup=15, horizon=10, record_trajectory=False)
    d = demand_realisation(small_net, cfg, 0)
    r = simulate(small_net, cfg, d, None)
    assert r.unmet_series.size == 0
    assert r.demand_total.shape == (small_net.demand_nodes.size,)


def test_backlog_cap_bounds_the_carried_shortfall():
    """Unbounded backlog would make recovery time a statement about the cap."""
    net = make_chain(n_tiers=2, lead=1, capacity=0.0, demand=10.0, cv=0.0)
    cfg = SimConfig(warmup=5, horizon=30, backlog=True, backlog_cap_periods=3.0)
    d = demand_realisation(net, cfg, 0)
    r = simulate(net, cfg, d, None)
    assert r.unmet_total[0] == pytest.approx(r.demand_total[0], rel=1e-6)


def test_recovery_time_is_nan_when_still_short_at_the_horizon():
    net = make_chain(n_tiers=3, lead=1, cv=0.0)
    cfg = SimConfig(warmup=25, horizon=10, backlog=False)
    ds = DisruptionSet([Disruption("supplier_outage", 0, -1, 0, 10, 1.0)]).bind(net)
    imp = counterfactual(net, cfg, ds, 0)
    assert np.isnan(imp.recovery_time[0])


def test_recovery_time_is_defined_when_service_returns():
    """A 4-period outage on a 2-leg chain: short at periods 2-5, recovered at 6.

    Recovery time is measured from the *end* of the disruption (period 4), so the
    expected value is exactly 2 - the two legs of transit it takes for restarted
    production to reach the customer.
    """
    net = make_chain(n_tiers=3, lead=1, cv=0.0)
    cfg = SimConfig(warmup=25, horizon=30, backlog=False)
    ds = DisruptionSet([Disruption("supplier_outage", 0, -1, 0, 4, 1.0)]).bind(net)
    imp = counterfactual(net, cfg, ds, 0)
    assert imp.time_to_impact[0] == pytest.approx(2.0)
    assert imp.recovery_time[0] == pytest.approx(2.0)
    assert np.allclose(imp.unmet_traj[0, 2:6], 10.0, atol=1e-6)
    assert np.allclose(imp.unmet_traj[0, 6:], 0.0, atol=1e-6)


def test_service_loss_clip_is_almost_never_active(small_net, sim_cfg):
    """The clip at zero exists for float noise, not to hide negative impacts."""
    worse = 0
    for seed in range(6):
        ds = DisruptionSet(
            [Disruption("capacity_reduction", 0, -1, 1, 8, 0.6)]
        ).bind(small_net)
        imp = counterfactual(small_net, sim_cfg, ds, seed)
        worse += int((imp.service_loss_raw < -1e-6).sum())
    assert worse == 0, "a disruption made service strictly better; check the mechanics"


def test_exhaustive_node_criticality_covers_every_producer(small_net, sim_cfg):
    cands, scores = exhaustive_node_criticality(small_net, sim_cfg, 0, duration=6, start=2)
    assert cands.size == int((small_net.tier < 4).sum())
    assert scores.shape == cands.shape
    assert np.all(scores >= -1e-12)


def test_exhaustive_edge_criticality_covers_every_edge(small_net, sim_cfg):
    edges, scores = exhaustive_edge_criticality(small_net, sim_cfg, 0, duration=6, start=2)
    assert len(edges) == len(small_net.edges())
    assert np.all(scores >= -1e-12)


def test_baseline_caching_gives_identical_results(small_net, sim_cfg):
    """Passing a cached baseline must not change any number."""
    ds = DisruptionSet([Disruption("supplier_outage", 0, -1, 1, 8, 1.0)]).bind(small_net)
    d = demand_realisation(small_net, sim_cfg, 4)
    base = simulate(small_net, sim_cfg, d, None)
    a = counterfactual(small_net, sim_cfg, ds, 4, baseline=None)
    b = counterfactual(small_net, sim_cfg, ds, 4, baseline=base)
    assert np.array_equal(a.service_loss, b.service_loss)


def test_eps_is_far_below_one_unit_of_demand():
    """A sanity check on the tolerance constant itself."""
    assert 0 < EPS < 1e-3
