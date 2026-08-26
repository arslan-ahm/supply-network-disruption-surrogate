"""Feature construction for the surrogate and for every baseline.

Three feature views are built here, and keeping them in one module is
deliberate: the fairness of the whole comparison rests on the three views
containing the *same information* about a node, differing only in whether the
model can see the network.

``node_features``
    Per-node structural and inventory attributes plus the disruption encoding.
    This is the GNN's input and, unchanged, the no-message-passing MLP's input.
``edge_features``
    Per-edge lead time, BOM coefficient, sourcing share, sole-source flag and
    lane-closure flag. Only the GNN consumes these.
``tabular_row_features``
    The reference-style view: attributes of the disrupted node, attributes of the
    demand point, and network-level summary statistics — with no relational term
    connecting the two. This is what a per-supplier risk model has, and the
    experiment in ``docs/RESULTS.md`` is about what that omission costs.

The topology columns marked *oracle* below (downstream reach, sole-source reach,
lead time to demand) are given to the **graph-free** baselines as well. That is
generous to the baselines on purpose: without them, the MLP ablation would be
measuring "does the model know anything about the graph" rather than "does
message passing help", and the reference-style comparison would be a straw man.
"""

from __future__ import annotations

import numpy as np

from sndsur.data.disruptions import DISRUPTION_TYPES, TYPE_INDEX, DisruptionSet
from sndsur.data.network import N_TIERS, SupplyNetwork

#: Column names of :func:`node_features`, in order.
NODE_FEATURE_NAMES: tuple[str, ...] = (
    *(f"tier_{t}" for t in range(N_TIERS)),
    "log_throughput",
    "capacity_slack",
    "base_stock_cover",
    "input_cover",
    "in_degree",
    "out_degree",
    "n_groups",
    "sole_source_out",
    "min_group_size",
    "bom_depth",
    "lead_to_demand_max",
    "lead_to_demand_min",
    "downstream_reach_frac",
    "sole_source_reach_frac",
    "is_demand",
    *(f"dis_{k}" for k in DISRUPTION_TYPES),
    "dis_severity",
    "dis_duration",
    "dis_start",
    "dis_any",
    "dis_lane_out",
    "dis_lane_in",
)
N_NODE_FEATURES = len(NODE_FEATURE_NAMES)

#: Column names of :func:`edge_features`, in order.
EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "log_lead_time",
    "bom_coeff",
    "share",
    "group_sole_source",
    "group_size_inv",
    "same_region",
    "lane_closed",
    "lead_inflated",
)
N_EDGE_FEATURES = len(EDGE_FEATURE_NAMES)


def _log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def static_node_features(net: SupplyNetwork) -> np.ndarray:
    """The disruption-independent part of :func:`node_features`.

    Split out because it is identical for every scenario on a given network, and
    caching it turns dataset construction from O(scenarios x nodes^2) topology
    work into O(networks x nodes^2).

    Returns:
        ``(n_nodes, N_NODE_FEATURES - n_disruption_columns)``.
    """
    n = net.n_nodes
    thr = net.throughput if net.throughput.size == n else net.compute_throughput()
    edges = net.edges()
    in_deg = np.zeros(n)
    out_deg = np.zeros(n)
    for u, v in edges:
        out_deg[u] += 1
        in_deg[v] += 1

    n_groups = np.array([len(g) for g in net.groups], dtype=np.float64)
    min_group = np.array(
        [min((len(g.suppliers) for g in gs), default=0) for gs in net.groups],
        dtype=np.float64,
    )
    depth = net.bom_depth().astype(np.float64)
    lead_max = net.lead_time_to_demand()
    lead_min = net.min_lead_time_to_demand()
    reach = net.downstream_reach()
    ss_reach = net.sole_source_reach()
    n_dem = max(1, net.demand_nodes.size)

    cap = net.capacity.copy()
    finite = np.isfinite(cap)
    slack = np.zeros(n)
    slack[finite] = np.where(
        thr[finite] > 0, cap[finite] / np.maximum(thr[finite], 1e-9) - 1.0, 0.0
    )
    cover = np.where(thr > 0, net.base_stock / np.maximum(thr, 1e-9), 0.0)

    tier_oh = np.zeros((n, N_TIERS))
    tier_oh[np.arange(n), net.tier] = 1.0

    cols = [
        tier_oh,
        _log1p(thr)[:, None],
        np.clip(slack, 0.0, 5.0)[:, None],
        np.clip(cover, 0.0, 10.0)[:, None],
        net.input_cover[:, None],
        _log1p(in_deg)[:, None],
        _log1p(out_deg)[:, None],
        n_groups[:, None],
        net.sole_source_flags()[:, None],
        min_group[:, None],
        depth[:, None],
        lead_max[:, None] / 10.0,
        lead_min[:, None] / 10.0,
        (reach / n_dem)[:, None],
        (ss_reach / n_dem)[:, None],
        (net.tier == N_TIERS - 1).astype(np.float64)[:, None],
    ]
    return np.concatenate(cols, axis=1)


#: Number of columns produced by the disruption encoding.
N_DISRUPTION_COLUMNS = len(DISRUPTION_TYPES) + 6


def disruption_node_features(
    net: SupplyNetwork, ds: DisruptionSet, horizon: int
) -> np.ndarray:
    """Per-node encoding of the active disruption set.

    A ``regional_event`` is expanded onto every node in the region, and a
    ``lane_closure`` onto both endpoints via the two lane columns. That expansion
    is why the surrogate can represent a correlated event at all: the alternative
    (a graph-level disruption vector) would force the model to rediscover which
    nodes are in the region from a scalar.

    Returns:
        ``(n_nodes, N_DISRUPTION_COLUMNS)``.
    """
    n = net.n_nodes
    out = np.zeros((n, N_DISRUPTION_COLUMNS))
    n_types = len(DISRUPTION_TYPES)
    for d in ds.items:
        if d.kind == "regional_event":
            targets = np.flatnonzero(net.region == d.target)
        elif d.kind == "lane_closure":
            targets = np.array([d.target, d.target2])
        else:
            targets = np.array([d.target])
        for v in targets:
            v = int(v)
            out[v, TYPE_INDEX[d.kind]] = 1.0
            out[v, n_types] = max(out[v, n_types], d.severity)
            out[v, n_types + 1] = max(out[v, n_types + 1], d.duration / max(horizon, 1))
            out[v, n_types + 2] = d.start / max(horizon, 1)
            out[v, n_types + 3] = 1.0
        if d.kind == "lane_closure":
            out[d.target, n_types + 4] = 1.0
            out[d.target2, n_types + 5] = 1.0
    return out


def node_features(
    net: SupplyNetwork,
    ds: DisruptionSet,
    horizon: int,
    static: np.ndarray | None = None,
) -> np.ndarray:
    """Full per-node input: static structure concatenated with the disruption.

    Returns:
        ``(n_nodes, N_NODE_FEATURES)``.
    """
    s = static if static is not None else static_node_features(net)
    return np.concatenate([s, disruption_node_features(net, ds, horizon)], axis=1)


def edge_index(net: SupplyNetwork) -> np.ndarray:
    """``(2, n_edges)`` array of ``[supplier; customer]`` node ids."""
    e = net.edges()
    if not e:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(e, dtype=np.int64).T


def edge_features(net: SupplyNetwork, ds: DisruptionSet) -> np.ndarray:
    """Per-edge features aligned with :func:`edge_index`.

    Returns:
        ``(n_edges, N_EDGE_FEATURES)``.
    """
    e = net.edges()
    out = np.zeros((len(e), N_EDGE_FEATURES))
    for i, (u, v) in enumerate(e):
        gi = net.group_of(u, v)
        g = net.groups[v][gi]
        lead = net.lead_time[(u, v)]
        out[i] = (
            np.log1p(lead),
            g.coeff,
            net.share(u, v),
            1.0 if g.sole_source else 0.0,
            1.0 / len(g.suppliers),
            1.0 if net.region[u] == net.region[v] else 0.0,
            0.0 if ds.lane_open(u, v, ds.start) else 1.0,
            1.0 if ds.effective_lead_time(u, v, ds.start, lead) > lead else 0.0,
        )
    return out


# --------------------------------------------------------------------------- #
# Reference-style tabular view
# --------------------------------------------------------------------------- #

#: Column names of :func:`tabular_row_features`, in order.
TABULAR_FEATURE_NAMES: tuple[str, ...] = (
    # attributes of the disrupted node ("the supplier being scored")
    *(f"src_{c}" for c in NODE_FEATURE_NAMES),
    # attributes of the demand point whose service level is being predicted
    *(f"dst_{c}" for c in NODE_FEATURE_NAMES),
    # network-level summary
    "net_n_nodes",
    "net_n_edges",
    "net_n_demand",
    "net_mean_slack",
    "net_sole_source_frac",
    "net_mean_lead",
    "net_max_depth",
    "net_total_demand",
)
N_TABULAR_FEATURES = len(TABULAR_FEATURE_NAMES)


def network_summary(net: SupplyNetwork) -> np.ndarray:
    """Eight graph-level scalars, shared by every row of a scenario."""
    edges = net.edges()
    thr = net.throughput if net.throughput.size == net.n_nodes else net.compute_throughput()
    finite = np.isfinite(net.capacity)
    slack = np.where(
        thr[finite] > 0, net.capacity[finite] / np.maximum(thr[finite], 1e-9) - 1.0, 0.0
    )
    n_groups = sum(len(g) for g in net.groups)
    n_sole = sum(1 for gs in net.groups for g in gs if g.sole_source)
    return np.array(
        [
            net.n_nodes / 100.0,
            len(edges) / 100.0,
            net.demand_nodes.size / 10.0,
            float(np.mean(slack)) if slack.size else 0.0,
            n_sole / max(n_groups, 1),
            float(np.mean(list(net.lead_time.values()))) if net.lead_time else 0.0,
            float(net.bom_depth().max()),
            float(np.log1p(net.mean_demand.sum())),
        ]
    )


def tabular_row_features(
    net: SupplyNetwork,
    ds: DisruptionSet,
    horizon: int,
    nf: np.ndarray | None = None,
    summary: np.ndarray | None = None,
) -> np.ndarray:
    """The reference approach's view: one row per (disruption, demand point).

    The "source" block is the primary disrupted node. For a multi-point scenario
    the node with the largest severity is used, with ties broken by lowest tier —
    a single-node risk score has no way to represent two simultaneous failures,
    and pretending otherwise would flatter it. This limitation is stated in
    ``docs/METHOD.md`` and is part of what the multi-point generalisation split
    measures.

    Returns:
        ``(n_demand, N_TABULAR_FEATURES)``.
    """
    f = nf if nf is not None else node_features(net, ds, horizon)
    s = summary if summary is not None else network_summary(net)
    dn = net.demand_nodes

    if ds.items:
        primary = max(ds.items, key=lambda d: (d.severity, -net.tier[d.target]))
        src = int(primary.target) if primary.kind != "regional_event" else int(
            np.flatnonzero(net.region == primary.target)[0]
        )
    else:
        src = int(dn[0])

    src_row = f[src]
    rows = np.empty((dn.size, N_TABULAR_FEATURES))
    for i, v in enumerate(dn):
        rows[i] = np.concatenate([src_row, f[int(v)], s])
    return rows
