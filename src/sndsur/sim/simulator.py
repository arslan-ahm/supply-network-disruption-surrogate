"""A discrete-time multi-echelon supply-network simulator.

This module is the ground truth for the entire repository. The learned surrogate
is trained to imitate it, all criticality rankings are defined by it, and every
"is the surrogate right?" question is answered by running it. That makes its
correctness the load-bearing property of the project, so the mechanics are
specified exactly here and tested against hand-solvable cases in
``tests/test_simulator.py``.

Period mechanics, in this order
-------------------------------

Let ``t`` index periods. Within one period:

1. **Arrivals.** Material shipped on edge ``(u, v)`` at period ``t - L_uv``
   lands in ``v``'s input stock for the group that ``u`` serves.
2. **Demand.** Each demand point realises ``d_j(t)``, drawn from a seeded
   lognormal with the node's mean and CV, scaled by any demand-spike
   disruption active at ``t``.
3. **Requirement propagation**, in reverse topological order (demand points
   first). A demand point requires ``d_j(t) + backlog_j``. A producing node
   requires the total it has been asked for this period, plus whatever it needs
   to restore its finished-goods base stock. It then places orders on its
   suppliers: the material implied by its own requirement, plus an input-side
   order-up-to term, split across the group's suppliers by their sourcing shares.
4. **Production and shipment**, in forward topological order. A node produces
   ``min(requirement, capacity, material)`` where ``material`` is the Leontief
   minimum over groups of ``stock_g / coeff_g``. It consumes inputs, adds to
   finished goods, and ships against its customers' orders — fully if it can,
   pro rata if it cannot.
5. **Fulfilment.** Each demand point serves backlog first, then current-period
   demand, from whatever has arrived. What is left unserved is recorded and (if
   backlogging is on) carried, up to a cap.

Why this ordering
-----------------

Requirements move up the network within a single period and material moves down
across periods (that is what a lead time *is*). Putting both passes in one period
means an order is seen immediately but the goods are not, which is the behaviour
that produces the delayed, amplified shortage wave the surrogate has to learn. A
simulator that resolved material in the same period as the order would make every
disruption instantaneous and the time-to-impact target meaningless.

Two deliberate simplifications, stated rather than hidden:

* **Static sourcing shares.** By default a node keeps ordering from a failed
  supplier at its nominal share. Setting ``allow_resourcing`` re-normalises
  shares over suppliers with positive capacity, which is the optimistic case.
  Both are supported because the difference between them is exactly the value of
  qualified alternates, and reporting one without the other would be a choice
  dressed up as a fact.
* **No cost model, no ordering optimisation.** Base-stock levels are given, not
  optimised. This is a *propagation* model, not an inventory-optimisation model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sndsur.data.disruptions import Disruption, DisruptionSet
from sndsur.data.network import N_TIERS, SupplyNetwork

#: Quantities below this are treated as zero when deciding whether a demand
#: point was short. Chosen far above float noise but far below one unit of
#: demand, so it never masks a real shortage.
EPS = 1e-6


@dataclass
class SimConfig:
    """Simulation horizon and policy switches.

    Attributes:
        warmup: Periods run before the measurement window, with no disruption.
            Inventory starts at its base-stock level, but the pipeline starts
            empty, so a warm-up is needed before service levels are meaningful.
        horizon: Measured periods after warm-up. All reported metrics are over
            this window.
        backlog: Carry unmet demand forward (True) or lose the sale (False).
        backlog_cap_periods: Maximum backlog, in periods of mean demand.
            Unbounded backlog lets a long outage produce arbitrarily large
            recovery times that say more about the cap than the network.
        allow_resourcing: Re-normalise sourcing shares over suppliers with
            positive capacity each period.
        allocation: ``"proportional"`` (fair share) or ``"priority"`` (customers
            served in ascending node id).
        record_trajectory: Keep the per-period unmet series. Off in the hot loop
            of the exhaustive counterfactual sweep, where only totals are needed.
    """

    warmup: int = 12
    horizon: int = 36
    backlog: bool = True
    backlog_cap_periods: float = 6.0
    allow_resourcing: bool = False
    allocation: str = "proportional"
    record_trajectory: bool = True


@dataclass
class SimResult:
    """Per-demand-point outcome of one simulation run.

    Attributes:
        demand_nodes: Node ids, ascending, matching every array's first axis.
        demand_total: Total demand over the measurement window.
        served_total: Total current-period demand served in the window.
        unmet_total: ``demand_total - served_total``.
        fill_rate: ``served_total / demand_total`` per demand point.
        unmet_series: ``(n_demand, horizon)`` unmet units per period, or an
            empty array when ``record_trajectory`` is off.
        demand_series: ``(n_demand, horizon)`` realised demand per period.
        conservation_error: Largest absolute material-balance residual seen over
            the whole run. Should be at float-noise level; asserted in tests and
            surfaced here so a caller can check cheaply.
    """

    demand_nodes: np.ndarray
    demand_total: np.ndarray
    served_total: np.ndarray
    unmet_total: np.ndarray
    fill_rate: np.ndarray
    unmet_series: np.ndarray
    demand_series: np.ndarray
    conservation_error: float = 0.0

    @property
    def mean_fill_rate(self) -> float:
        """Demand-weighted overall service level."""
        d = self.demand_total.sum()
        return float(self.served_total.sum() / d) if d > 0 else float("nan")


@dataclass
class ImpactResult:
    """A counterfactual: the disrupted run measured against its own baseline.

    Both runs use the *same* demand realisation, so the difference isolates the
    disruption rather than mixing in demand noise. This pairing is what makes
    the impact target learnable at this scale; with independent demand draws the
    label noise swamps the effect of a mild disruption.

    Attributes:
        service_loss: ``baseline_fill_rate - disrupted_fill_rate`` per demand
            point, clipped at 0 below. Non-negative by construction of the
            clip; the raw value is kept in ``service_loss_raw``.
        service_loss_raw: Unclipped loss, so a caller can check how often the
            clip fires (it should be rare and tiny — reported in RESULTS.md).
        unmet_extra: Extra unmet units caused by the disruption, per demand point.
        time_to_impact: Periods from the disruption's start to the first period
            in which that demand point is short by more than the baseline. NaN
            when the demand point is never affected.
        recovery_time: Periods from the disruption's *end* until the demand point
            stops being short relative to baseline for the rest of the window.
            NaN when it never recovers inside the horizon, and 0 when the
            shortage ends before the disruption does.
        unmet_traj: ``(n_demand, horizon)`` excess-unmet trajectory.
        baseline: The undisrupted run.
        disrupted: The disrupted run.
    """

    demand_nodes: np.ndarray
    service_loss: np.ndarray
    service_loss_raw: np.ndarray
    unmet_extra: np.ndarray
    time_to_impact: np.ndarray
    recovery_time: np.ndarray
    unmet_traj: np.ndarray
    baseline: SimResult = field(repr=False, default=None)  # type: ignore[assignment]
    disrupted: SimResult = field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def total_service_loss(self) -> float:
        """Sum of per-demand-point service loss — the scalar criticality score."""
        return float(np.nansum(self.service_loss))


# --------------------------------------------------------------------------- #
# Demand realisation
# --------------------------------------------------------------------------- #


def demand_realisation(net: SupplyNetwork, cfg: SimConfig, seed: int) -> np.ndarray:
    """Draw the demand path for every demand point.

    Returns:
        ``(n_demand, warmup + horizon)`` non-negative demand.

    The path depends only on ``(net, cfg, seed)``, never on the disruption, so a
    baseline run can be cached and reused across every counterfactual on the same
    network. In the exhaustive sweep that halves the simulator's cost, and the
    efficiency comparison would be dishonest if it did not exploit an
    optimisation any real user would make.
    """
    rng = np.random.default_rng(seed)
    dn = net.demand_nodes
    T = cfg.warmup + cfg.horizon
    mu = net.mean_demand[dn][:, None]
    cv = net.demand_cv[dn][:, None]
    # Lognormal parameterised by mean and CV: demand is positive and right-skewed,
    # which a truncated normal would not reproduce.
    sigma2 = np.log1p(np.maximum(cv, 1e-9) ** 2)
    normal = rng.standard_normal(size=(dn.size, T))
    return mu * np.exp(np.sqrt(sigma2) * normal - 0.5 * sigma2)


# --------------------------------------------------------------------------- #
# Core simulation
# --------------------------------------------------------------------------- #


def simulate(
    net: SupplyNetwork,
    cfg: SimConfig,
    demand: np.ndarray,
    disruptions: DisruptionSet | None = None,
) -> SimResult:
    """Run the network forward for ``warmup + horizon`` periods.

    Args:
        net: The network. Must satisfy :meth:`SupplyNetwork.validate`.
        cfg: Horizon and policy switches.
        demand: ``(n_demand, warmup + horizon)`` demand path, from
            :func:`demand_realisation`.
        disruptions: Active disruptions, or None for the undisrupted baseline.
            Disruption periods are given relative to the start of the
            *measurement* window, so ``start=0`` means the first measured period
            and negative starts are rejected by :class:`Disruption`.

    Returns:
        A :class:`SimResult` covering the measurement window only.
    """
    n = net.n_nodes
    T = cfg.warmup + cfg.horizon
    dn = net.demand_nodes
    n_demand = dn.size
    if demand.shape != (n_demand, T):
        raise ValueError(f"demand must be {(n_demand, T)}, got {demand.shape}")

    ds = disruptions if disruptions is not None else DisruptionSet([])
    demand_row = {int(v): i for i, v in enumerate(dn)}
    is_demand = net.tier == N_TIERS - 1

    # ---- static structure, flattened once so the period loop is arithmetic ----
    groups = net.groups
    n_groups = [len(g) for g in groups]
    coeff = [np.array([g.coeff for g in gs], dtype=np.float64) for gs in groups]
    # supplier_edges[v] = list of (group_index, supplier, nominal_share)
    supplier_edges: list[list[tuple[int, int, float]]] = []
    for v in range(n):
        rows = []
        for gi, g in enumerate(groups[v]):
            for u, w in zip(g.suppliers, g.shares, strict=True):
                rows.append((gi, int(u), float(w)))
        supplier_edges.append(rows)
    customers = net.customers()
    group_index = {(u, v): net.group_of(u, v) for (u, v) in net.edges()}

    # ---- state ----
    raw = [np.zeros(k, dtype=np.float64) for k in n_groups]
    fg = net.base_stock.copy()
    fg[is_demand] = 0.0
    backlog = np.zeros(n, dtype=np.float64)
    thr = net.throughput if net.throughput.size == n else net.compute_throughput()

    # Input stock starts at its order-up-to level: the network begins in a
    # plausible steady state rather than empty, which would make the first
    # several periods of every run a stockout artefact.
    for v in range(n):
        if n_groups[v]:
            raw[v][:] = coeff[v] * thr[v] * net.input_cover[v]

    max_lead = max(net.lead_time.values()) if net.lead_time else 1
    # pipeline[(u, v)] is a ring buffer of arrivals indexed by (t % span).
    span = max_lead + 2
    pipeline = {e: np.zeros(span, dtype=np.float64) for e in net.lead_time}

    backlog_cap = np.zeros(n, dtype=np.float64)
    backlog_cap[dn] = cfg.backlog_cap_periods * net.mean_demand[dn]

    unmet_series = (
        np.zeros((n_demand, cfg.horizon), dtype=np.float64)
        if cfg.record_trajectory
        else np.zeros((0, 0))
    )
    served_total = np.zeros(n_demand, dtype=np.float64)
    demand_total = np.zeros(n_demand, dtype=np.float64)
    cons_err = 0.0

    orders_in = np.zeros(n, dtype=np.float64)
    req = np.zeros(n, dtype=np.float64)
    # order_book[(u, v)] is what v asked u for this period.
    order_book = dict.fromkeys(net.lead_time, 0.0)

    proportional = cfg.allocation == "proportional"

    for t in range(T):
        rel = t - cfg.warmup  # period index relative to the measurement window
        slot = t % span

        # ---- 1. arrivals ----
        for (u, v), buf in pipeline.items():
            q = buf[slot]
            if q:
                raw[v][group_index[(u, v)]] += q
                buf[slot] = 0.0

        # ---- 2. demand ----
        d_now = demand[:, t] * np.array(
            [ds.demand_multiplier(int(v), rel) for v in dn], dtype=np.float64
        )

        # ---- 3. requirements, reverse topological order ----
        orders_in[:] = 0.0
        for e in order_book:
            order_book[e] = 0.0
        for v in range(n - 1, -1, -1):
            if is_demand[v]:
                req[v] = d_now[demand_row[v]] + backlog[v]
            else:
                req[v] = orders_in[v] + max(0.0, net.base_stock[v] - fg[v])
                cap = net.capacity[v] * ds.capacity_multiplier(int(v), rel)
                # A node does not order material it has no capacity to convert.
                # Without this the shortage signal would propagate upstream
                # through a dead node, which is not how a plant behaves.
                req[v] = min(req[v], cap)
            if req[v] < 0.0:
                req[v] = 0.0
            if not n_groups[v]:
                continue

            # Order-up-to on the input side, per group.
            in_transit = np.zeros(n_groups[v], dtype=np.float64)
            for gi, u, _w in supplier_edges[v]:
                in_transit[gi] += pipeline[(u, v)].sum()
            target = coeff[v] * thr[v] * net.input_cover[v]
            group_order = coeff[v] * req[v] + np.maximum(
                0.0, target - raw[v] - in_transit
            )

            for gi in range(n_groups[v]):
                rows = [(u, w) for g2, u, w in supplier_edges[v] if g2 == gi]
                if cfg.allow_resourcing:
                    live = [
                        (u, w)
                        for u, w in rows
                        if net.capacity[u] * ds.capacity_multiplier(u, rel) > EPS
                        and ds.lane_open(u, v, rel)
                    ]
                    if live:
                        rows = live
                tot_w = sum(w for _u, w in rows)
                if tot_w <= 0:
                    continue
                for u, w in rows:
                    order_book[(u, v)] += group_order[gi] * w / tot_w
                    orders_in[u] += group_order[gi] * w / tot_w

        # ---- 4. production and shipment, forward topological order ----
        for v in range(n):
            if is_demand[v]:
                continue
            cap = net.capacity[v] * ds.capacity_multiplier(int(v), rel)
            if n_groups[v]:
                material = float(np.min(raw[v] / coeff[v]))
            else:
                material = np.inf
            prod = min(req[v], cap, material)
            if prod < 0.0:
                prod = 0.0
            before = raw[v].sum() if n_groups[v] else 0.0
            if n_groups[v]:
                raw[v] -= prod * coeff[v]
                consumed = prod * coeff[v].sum()
                cons_err = max(cons_err, abs(before - raw[v].sum() - consumed))
                # Floating-point subtraction can leave a stock at -1e-16, which
                # would then look like a shortage for the rest of the run.
                np.maximum(raw[v], 0.0, out=raw[v])
            fg[v] += prod

            wanted = [(c, order_book[(v, c)]) for c in customers[v]]
            total_want = sum(q for _c, q in wanted)
            if total_want <= EPS or fg[v] <= EPS:
                continue
            avail = fg[v]
            if proportional:
                frac = min(1.0, avail / total_want)
                allocation = [(c, q * frac) for c, q in wanted]
            else:
                allocation = []
                left = avail
                for c, q in sorted(wanted):
                    give = min(q, left)
                    left -= give
                    allocation.append((c, give))
            shipped = 0.0
            for c, q in allocation:
                if q <= 0.0:
                    continue
                if not ds.lane_open(int(v), int(c), rel):
                    continue  # a closed lane blocks the shipment entirely
                lead = ds.effective_lead_time(int(v), int(c), rel, net.lead_time[(v, c)])
                pipeline[(v, c)][(t + lead) % span] += q
                shipped += q
            fg[v] -= shipped
            if fg[v] < 0.0:
                cons_err = max(cons_err, -fg[v])
                fg[v] = 0.0

        # ---- 5. fulfilment at demand points ----
        for i, v in enumerate(dn):
            v = int(v)
            avail = raw[v].sum()
            serve_bl = min(avail, backlog[v])
            avail -= serve_bl
            backlog[v] -= serve_bl
            d = d_now[i]
            serve_now = min(avail, d)
            avail -= serve_now
            short = d - serve_now
            if cfg.backlog:
                backlog[v] = min(backlog[v] + short, backlog_cap[v])
            taken = serve_bl + serve_now
            if taken > 0:
                total = raw[v].sum()
                if total > 0:
                    raw[v] *= max(0.0, (total - taken) / total)
            if rel >= 0:
                demand_total[i] += d
                served_total[i] += serve_now
                if cfg.record_trajectory:
                    unmet_series[i, rel] = short

    unmet_total = demand_total - served_total
    with np.errstate(invalid="ignore", divide="ignore"):
        fill = np.where(demand_total > 0, served_total / demand_total, np.nan)

    return SimResult(
        demand_nodes=dn.copy(),
        demand_total=demand_total,
        served_total=served_total,
        unmet_total=unmet_total,
        fill_rate=fill,
        unmet_series=unmet_series,
        demand_series=demand[:, cfg.warmup :].copy() if cfg.record_trajectory else np.zeros((0, 0)),
        conservation_error=float(cons_err),
    )


def counterfactual(
    net: SupplyNetwork,
    cfg: SimConfig,
    disruptions: DisruptionSet,
    seed: int,
    baseline: SimResult | None = None,
) -> ImpactResult:
    """Impact of ``disruptions``, measured against a paired undisrupted run.

    Args:
        net: The network.
        cfg: Simulation configuration.
        disruptions: What fails, when, and how hard.
        seed: Demand seed. Both runs use the same realisation.
        baseline: A cached undisrupted run for this ``(net, cfg, seed)``. Passing
            it halves the cost, and it is always safe to pass because the
            baseline cannot depend on the disruption.

    Returns:
        An :class:`ImpactResult`.
    """
    d = demand_realisation(net, cfg, seed)
    base = baseline if baseline is not None else simulate(net, cfg, d, None)
    dis = simulate(net, cfg, d, disruptions)

    loss_raw = base.fill_rate - dis.fill_rate
    loss = np.clip(loss_raw, 0.0, None)
    extra = dis.unmet_total - base.unmet_total

    start = disruptions.start if disruptions.items else 0
    end = disruptions.end if disruptions.items else 0
    if base.unmet_series.size and dis.unmet_series.size:
        traj = dis.unmet_series - base.unmet_series
        tti, rec = _timing(traj, start, end, net.mean_demand[net.demand_nodes])
    else:
        traj = np.zeros((net.demand_nodes.size, 0))
        tti = np.full(net.demand_nodes.size, np.nan)
        rec = np.full(net.demand_nodes.size, np.nan)

    return ImpactResult(
        demand_nodes=base.demand_nodes,
        service_loss=loss,
        service_loss_raw=loss_raw,
        unmet_extra=extra,
        time_to_impact=tti,
        recovery_time=rec,
        unmet_traj=traj,
        baseline=base,
        disrupted=dis,
    )


def _timing(
    traj: np.ndarray, start: int, end: int, mean_demand: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Time-to-impact and recovery time from an excess-unmet trajectory.

    ``traj`` is disrupted minus baseline unmet, per demand point per period. A
    period counts as affected when the excess exceeds 1% of that demand point's
    mean demand — an absolute float tolerance would flag rounding noise at large
    demand and miss real shortages at small demand.

    Returns:
        ``(time_to_impact, recovery_time)``, both NaN where undefined:
        time-to-impact when the point is never affected, recovery time when it is
        still short at the end of the horizon.
    """
    n, T = traj.shape
    thresh = np.maximum(0.01 * mean_demand, EPS)[:, None]
    affected = traj > thresh
    tti = np.full(n, np.nan)
    rec = np.full(n, np.nan)
    for i in range(n):
        idx = np.flatnonzero(affected[i, max(start, 0) :])
        if idx.size == 0:
            continue
        tti[i] = float(idx[0])
        last = int(np.flatnonzero(affected[i])[-1])
        if last < T - 1:
            rec[i] = float(max(0, last + 1 - end))
    return tti, rec


def exhaustive_node_criticality(
    net: SupplyNetwork,
    cfg: SimConfig,
    seed: int,
    duration: int = 8,
    start: int = 4,
    severity: float = 1.0,
    candidates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """True criticality of every candidate node, by simulating each removal.

    This is the oracle the surrogate is screening for. A standardised outage —
    same start, same duration, same severity for every candidate — is applied to
    one node at a time and the resulting total service loss is recorded. Holding
    the disruption fixed is what makes the ranking a statement about *position in
    the network* rather than about which node happened to draw a nastier event.

    Args:
        net: The network.
        cfg: Simulation configuration.
        seed: Demand seed, shared by the baseline and every counterfactual.
        duration: Outage length in periods.
        start: Outage start, relative to the measurement window.
        severity: 1.0 is a total outage.
        candidates: Node ids to test. Defaults to every non-demand node.

    Returns:
        ``(candidates, total_service_loss)``, aligned.
    """
    if candidates is None:
        candidates = np.flatnonzero(net.tier < N_TIERS - 1)
    d = demand_realisation(net, cfg, seed)
    base = simulate(net, cfg, d, None)
    scores = np.zeros(candidates.size, dtype=np.float64)
    for i, v in enumerate(candidates):
        ds = DisruptionSet(
            [Disruption("supplier_outage", int(v), -1, start, duration, severity)]
        )
        res = counterfactual(net, cfg, ds, seed, baseline=base)
        scores[i] = res.total_service_loss
    return candidates, scores


def exhaustive_edge_criticality(
    net: SupplyNetwork,
    cfg: SimConfig,
    seed: int,
    duration: int = 8,
    start: int = 4,
    edges: list[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """True criticality of every transport lane, by closing each in turn."""
    if edges is None:
        edges = net.edges()
    d = demand_realisation(net, cfg, seed)
    base = simulate(net, cfg, d, None)
    scores = np.zeros(len(edges), dtype=np.float64)
    for i, (u, v) in enumerate(edges):
        ds = DisruptionSet([Disruption("lane_closure", int(u), int(v), start, duration, 1.0)])
        res = counterfactual(net, cfg, ds, seed, baseline=base)
        scores[i] = res.total_service_loss
    return edges, scores
