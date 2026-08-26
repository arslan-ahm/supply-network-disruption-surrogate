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

    warmup: int = 22
    horizon: int = 30
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


class _Structure:
    """Flattened, disruption-independent view of a network.

    The period loop below runs tens of thousands of times while a dataset is
    built. A first version of this module held every per-node quantity in a NumPy
    array of length 1-3 and re-derived which supplier served which group inside
    the loop; profiling put most of its time in NumPy call overhead rather than
    arithmetic. Everything static is therefore compiled once into plain Python
    lists and cached on the network object, which cut per-counterfactual cost by
    more than an order of magnitude with bit-identical outputs. The committed
    per-run cost is measured in ``results/tables/efficiency.csv``; the earlier
    implementation is not in the repository, so no speed-up factor for that
    rewrite is claimed as a result.

    Plain lists rather than NumPy arrays is the right choice *here* specifically
    because every inner quantity is a scalar or a length-1-to-3 vector, where
    NumPy's per-call overhead dwarfs its vectorisation benefit.
    """

    __slots__ = (
        "n", "n_edges", "is_demand", "demand_rows", "demand_row_of", "prod_order",
        "n_groups", "grp_coeff", "grp_rows", "in_group_edges", "e_u", "e_v",
        "e_grp", "e_lead", "cust_edges", "max_lead",
    )

    def __init__(self, net: SupplyNetwork) -> None:
        n = net.n_nodes
        self.n = n
        edges = net.edges()
        self.n_edges = len(edges)
        self.is_demand = [bool(t == N_TIERS - 1) for t in net.tier.tolist()]
        self.demand_rows = [int(v) for v in net.demand_nodes]
        self.demand_row_of = {v: i for i, v in enumerate(self.demand_rows)}
        self.prod_order = [v for v in range(n) if not self.is_demand[v]]

        self.e_u = [int(u) for u, _v in edges]
        self.e_v = [int(v) for _u, v in edges]
        self.e_grp = [net.group_of(u, v) for u, v in edges]
        self.e_lead = [int(net.lead_time[(u, v)]) for u, v in edges]
        self.max_lead = max(self.e_lead) if self.e_lead else 1

        self.n_groups = [len(g) for g in net.groups]
        self.grp_coeff = [[float(g.coeff) for g in gs] for gs in net.groups]
        # grp_rows[v][gi] = [(edge_id, supplier, nominal_share), ...]
        self.grp_rows = [[[] for _ in gs] for gs in net.groups]
        self.in_group_edges = [[[] for _ in gs] for gs in net.groups]
        self.cust_edges = [[] for _ in range(n)]
        for e, (u, v) in enumerate(edges):
            gi = self.e_grp[e]
            self.grp_rows[v][gi].append((e, u, net.share(u, v)))
            self.in_group_edges[v][gi].append(e)
            self.cust_edges[u].append((e, v))


def _structure(net: SupplyNetwork) -> _Structure:
    """Compiled structure for ``net``, cached on the instance.

    The cache key is the edge count plus the node count: the generator never
    mutates a network in place, and the CSV loader builds a fresh object, so a
    structural change always arrives as a new object. The key exists only to
    catch a caller who edits ``groups`` by hand, which would otherwise produce
    a silently stale structure.
    """
    key = (net.n_nodes, len(net.lead_time))
    cached = net.__dict__.get("_sim_struct")
    if cached is not None and net.__dict__.get("_sim_struct_key") == key:
        return cached
    st = _Structure(net)
    net.__dict__["_sim_struct"] = st
    net.__dict__["_sim_struct_key"] = key
    return st


def _disruption_tables(
    net: SupplyNetwork, st: _Structure, cfg: SimConfig, ds: DisruptionSet, T: int
) -> tuple[list, list, list, list]:
    """Expand a disruption set into per-period lookup tables.

    Returns ``(cap_mult, dem_mult, lane_open, lead_eff)``, each indexed by
    absolute period ``t`` then by node / demand row / edge. Warm-up periods are
    always undisrupted, which is what makes ``start=0`` mean "the first measured
    period" rather than "somewhere in the burn-in".

    Building the tables costs O(T * (n + E)) once per scenario and removes a
    method call per node per period from the loop. When there is no disruption
    the constant rows are shared, so the baseline run allocates nothing.
    """
    n, n_e = st.n, st.n_edges
    n_d = len(st.demand_rows)
    if not ds.items:
        one_n = [1.0] * n
        one_d = [1.0] * n_d
        open_e = [True] * n_e
        lead0 = list(st.e_lead)
        return ([one_n] * T, [one_d] * T, [open_e] * T, [lead0] * T)

    cap = [[1.0] * n for _ in range(T)]
    dem = [[1.0] * n_d for _ in range(T)]
    lane = [[True] * n_e for _ in range(T)]
    lead = [list(st.e_lead) for _ in range(T)]
    lead_add = [[0.0] * n for _ in range(T)]
    region = net.region.tolist()
    warm = cfg.warmup

    for d in ds.items:
        lo = warm + d.start
        hi = min(T, warm + d.end)
        if lo >= T:
            continue
        if d.kind in ("supplier_outage", "capacity_reduction"):
            f = max(0.0, 1.0 - d.severity)
            v = int(d.target)
            for t in range(lo, hi):
                cap[t][v] *= f
        elif d.kind == "regional_event":
            f = max(0.0, 1.0 - d.severity)
            members = [v for v in range(n) if region[v] == d.target]
            for t in range(lo, hi):
                row = cap[t]
                for v in members:
                    row[v] *= f
        elif d.kind == "demand_spike":
            i = st.demand_row_of.get(int(d.target))
            if i is not None:
                f = 1.0 + d.severity
                for t in range(lo, hi):
                    dem[t][i] *= f
        elif d.kind == "lane_closure":
            ids = [
                e
                for e in range(n_e)
                if st.e_u[e] == int(d.target) and st.e_v[e] == int(d.target2)
            ]
            for t in range(lo, hi):
                for e in ids:
                    lane[t][e] = False
        elif d.kind == "lead_time_inflation":
            v = int(d.target)
            for t in range(lo, hi):
                lead_add[t][v] += d.severity

    for t in range(T):
        row = lead_add[t]
        if not any(row):
            continue
        out = lead[t]
        for e in range(n_e):
            add = row[st.e_u[e]]
            if add:
                out[e] = max(1, int(round(st.e_lead[e] * (1.0 + add))))
    return cap, dem, lane, lead


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
    st = _structure(net)
    n = st.n
    T = cfg.warmup + cfg.horizon
    dn = net.demand_nodes
    n_demand = dn.size
    if demand.shape != (n_demand, T):
        raise ValueError(f"demand must be {(n_demand, T)}, got {demand.shape}")

    ds = disruptions if disruptions is not None else DisruptionSet([])
    cap_tab, dem_tab, lane_tab, lead_tab = _disruption_tables(net, st, cfg, ds, T)

    is_demand = st.is_demand
    n_groups = st.n_groups
    grp_coeff = st.grp_coeff
    grp_rows = st.grp_rows
    in_group_edges = st.in_group_edges
    cust_edges = st.cust_edges
    e_v, e_grp = st.e_v, st.e_grp

    thr = (net.throughput if net.throughput.size == n else net.compute_throughput()).tolist()
    capacity = net.capacity.tolist()
    base_stock = net.base_stock.tolist()
    cover = net.input_cover.tolist()
    demand_l = demand.tolist()

    # ---- state ----
    raw = [[grp_coeff[v][g] * thr[v] * cover[v] for g in range(n_groups[v])] for v in range(n)]
    grp_target = [list(raw[v]) for v in range(n)]
    fg = [0.0 if is_demand[v] else base_stock[v] for v in range(n)]
    backlog = [0.0] * n
    # The in-transit ring buffer must be long enough for the longest lead time
    # that can actually occur, which is the *inflated* one, not the nominal one.
    # Sizing it from st.max_lead made every lead_time_inflation disruption wrap
    # around and land early: 120 of 120 sampled inflation scenarios produced
    # exactly zero impact, which is how the bug was found. Tested by
    # test_simulator.py::test_lead_time_inflation_delays_by_the_inflated_amount.
    span = max(max(row) for row in lead_tab) + 2 if lead_tab else st.max_lead + 2
    pipe = [[0.0] * span for _ in range(st.n_edges)]
    pipe_tot = [0.0] * st.n_edges
    order_book = [0.0] * st.n_edges
    orders_in = [0.0] * n
    req = [0.0] * n

    backlog_cap = [0.0] * n
    md = net.mean_demand.tolist()
    for v in st.demand_rows:
        backlog_cap[v] = cfg.backlog_cap_periods * md[v]

    record = cfg.record_trajectory
    unmet_series = (
        np.zeros((n_demand, cfg.horizon), dtype=np.float64) if record else np.zeros((0, 0))
    )
    unmet_rows = unmet_series.tolist() if record else []
    served_total = [0.0] * n_demand
    demand_total = [0.0] * n_demand
    cons_err = 0.0
    proportional = cfg.allocation == "proportional"
    resource = cfg.allow_resourcing
    warm = cfg.warmup

    for t in range(T):
        rel = t - warm
        slot = t % span
        cap_row = cap_tab[t]
        lane_row = lane_tab[t]
        lead_row = lead_tab[t]
        dem_row = dem_tab[t]

        # ---- 1. arrivals ----
        for e in range(st.n_edges):
            buf = pipe[e]
            q = buf[slot]
            if q:
                raw[e_v[e]][e_grp[e]] += q
                pipe_tot[e] -= q
                buf[slot] = 0.0

        # ---- 2. demand ----
        d_now = [demand_l[i][t] * dem_row[i] for i in range(n_demand)]

        # ---- 3. requirements, reverse topological order ----
        for v in range(n):
            orders_in[v] = 0.0
        for e in range(st.n_edges):
            order_book[e] = 0.0

        for v in range(n - 1, -1, -1):
            if is_demand[v]:
                r = d_now[st.demand_row_of[v]] + backlog[v]
            else:
                r = orders_in[v] + base_stock[v] - fg[v]
                if r < orders_in[v]:
                    r = orders_in[v]
                c = capacity[v] * cap_row[v]
                if r > c:
                    r = c
            if r < 0.0:
                r = 0.0
            req[v] = r
            ng = n_groups[v]
            if not ng:
                continue

            coeffs = grp_coeff[v]
            rawv = raw[v]
            targets = grp_target[v]
            for gi in range(ng):
                transit = 0.0
                for e in in_group_edges[v][gi]:
                    transit += pipe_tot[e]
                short = targets[gi] - rawv[gi] - transit
                order = coeffs[gi] * r + (short if short > 0.0 else 0.0)
                if order <= 0.0:
                    continue
                rows = grp_rows[v][gi]
                if resource:
                    live = [
                        (e, u, w)
                        for (e, u, w) in rows
                        if capacity[u] * cap_row[u] > EPS and lane_row[e]
                    ]
                    if live:
                        rows = live
                tot_w = 0.0
                for _e, _u, w in rows:
                    tot_w += w
                if tot_w <= 0.0:
                    continue
                for e, u, w in rows:
                    q = order * w / tot_w
                    order_book[e] += q
                    orders_in[u] += q

        # ---- 4. production and shipment, forward topological order ----
        for v in st.prod_order:
            c = capacity[v] * cap_row[v]
            ng = n_groups[v]
            prod = req[v]
            if prod > c:
                prod = c
            if ng:
                rawv = raw[v]
                coeffs = grp_coeff[v]
                for gi in range(ng):
                    m = rawv[gi] / coeffs[gi]
                    if m < prod:
                        prod = m
            if prod <= 0.0:
                prod = 0.0
            elif ng:
                rawv = raw[v]
                coeffs = grp_coeff[v]
                for gi in range(ng):
                    val = rawv[gi] - prod * coeffs[gi]
                    if val < 0.0:
                        if -val > cons_err:
                            cons_err = -val
                        val = 0.0
                    rawv[gi] = val
            fg[v] += prod

            ces = cust_edges[v]
            if not ces:
                continue
            total_want = 0.0
            for e, _c in ces:
                total_want += order_book[e]
            avail = fg[v]
            if total_want <= EPS or avail <= EPS:
                continue
            if proportional:
                frac = avail / total_want
                if frac > 1.0:
                    frac = 1.0
                shipped = 0.0
                for e, _cnode in ces:
                    q = order_book[e] * frac
                    if q <= 0.0 or not lane_row[e]:
                        continue
                    pipe[e][(t + lead_row[e]) % span] += q
                    pipe_tot[e] += q
                    shipped += q
            else:
                left = avail
                shipped = 0.0
                for e, _cnode in sorted(ces, key=lambda x: x[1]):
                    q = order_book[e]
                    if q > left:
                        q = left
                    left -= q
                    if q <= 0.0 or not lane_row[e]:
                        continue
                    pipe[e][(t + lead_row[e]) % span] += q
                    pipe_tot[e] += q
                    shipped += q
            fg[v] = avail - shipped
            if fg[v] < 0.0:
                if -fg[v] > cons_err:
                    cons_err = -fg[v]
                fg[v] = 0.0

        # ---- 5. fulfilment at demand points ----
        for i, v in enumerate(st.demand_rows):
            rawv = raw[v]
            avail = 0.0
            for q in rawv:
                avail += q
            bl = backlog[v]
            serve_bl = bl if bl < avail else avail
            avail -= serve_bl
            bl -= serve_bl
            d = d_now[i]
            serve_now = d if d < avail else avail
            avail -= serve_now
            short = d - serve_now
            if cfg.backlog:
                bl += short
                if bl > backlog_cap[v]:
                    bl = backlog_cap[v]
            backlog[v] = bl
            taken = serve_bl + serve_now
            if taken > 0.0:
                total = 0.0
                for q in rawv:
                    total += q
                if total > 0.0:
                    f = (total - taken) / total
                    if f < 0.0:
                        f = 0.0
                    for gi in range(len(rawv)):
                        rawv[gi] *= f
            if rel >= 0:
                demand_total[i] += d
                served_total[i] += serve_now
                if record:
                    unmet_rows[i][rel] = short

    served_arr = np.asarray(served_total, dtype=np.float64)
    demand_arr = np.asarray(demand_total, dtype=np.float64)
    unmet_total = demand_arr - served_arr
    with np.errstate(invalid="ignore", divide="ignore"):
        fill = np.where(demand_arr > 0, served_arr / demand_arr, np.nan)

    return SimResult(
        demand_nodes=dn.copy(),
        demand_total=demand_arr,
        served_total=served_arr,
        unmet_total=unmet_total,
        fill_rate=fill,
        unmet_series=np.asarray(unmet_rows, dtype=np.float64) if record else unmet_series,
        demand_series=demand[:, warm:].copy() if record else np.zeros((0, 0)),
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
