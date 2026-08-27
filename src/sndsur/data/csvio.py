"""Optional loader for a user-supplied network from CSV.

This is the hook for pointing the analysis at a real topology. It is optional and
nothing in the shipped results uses it — every committed number comes from a
generated network — but it is the difference between a closed demo and something
a reader can try on their own data.

**What this does and does not give you.** It replaces the *topology*: nodes,
echelons, regions, input groups, sourcing shares, BOM coefficients, lead times,
capacities, inventory targets and demand. It does **not** replace the
*mechanics*: your network is still simulated by this module's base-stock,
Leontief, pro-rata-allocation model. Results transfer only insofar as those
mechanics match how your network actually behaves. That caveat is repeated in
``docs/METHOD.md`` because it is the single most important limitation of the
whole repository.

Format
------

``nodes.csv`` — one row per node, ordered so tiers are non-decreasing:

===============  ==========================================================
column           meaning
===============  ==========================================================
``node_id``      integer, 0-based, ascending
``tier``         0 raw, 1 component, 2 assembly, 3 distribution, 4 demand
``region``       integer region label
``capacity``     units per period; blank or ``inf`` for demand points
``base_stock``   finished-goods order-up-to level
``input_cover``  periods of input demand held as safety stock
``mean_demand``  demand per period; must be > 0 for tier 4, 0 otherwise
``demand_cv``    coefficient of variation of demand; 0 elsewhere
===============  ==========================================================

``edges.csv`` — one row per supplier/customer link:

================  =========================================================
column            meaning
================  =========================================================
``supplier``      node id of the supplier
``customer``      node id of the customer
``group``         input-group index within the customer, 0-based
``coeff``         BOM coefficient of that group (constant within a group)
``share``         sourcing share; shares within a group must sum to 1
``lead_time``     transit periods, integer >= 1
================  =========================================================

Suppliers sharing a ``(customer, group)`` are substitutable. A group with one row
is sole-sourced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sndsur.data.network import N_TIERS, InputGroup, SupplyNetwork

NODE_COLUMNS = (
    "node_id", "tier", "region", "capacity", "base_stock",
    "input_cover", "mean_demand", "demand_cv",
)
EDGE_COLUMNS = ("supplier", "customer", "group", "coeff", "share", "lead_time")


def load_network_from_csv(
    nodes_csv: str | Path,
    edges_csv: str | Path,
    name: str = "user_network",
    normalise_shares: bool = True,
) -> SupplyNetwork:
    """Build a :class:`SupplyNetwork` from two CSV files.

    Args:
        nodes_csv: Path to the node table.
        edges_csv: Path to the edge table.
        name: Identifier carried into result tables.
        normalise_shares: Rescale each group's shares to sum to 1. On by default
            because hand-written data rarely sums to exactly 1, and a silent
            0.999 would fail validation for a reason that is not the user's real
            problem. Set False to have the mismatch raise instead.

    Returns:
        A validated network.

    Raises:
        ValueError: on a missing column, an unknown node id, an inconsistent BOM
            coefficient within a group, or any structural violation. Loading is
            strict on purpose: a malformed network otherwise produces quietly
            wrong service levels rather than an error.
    """
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    missing = set(NODE_COLUMNS) - set(nodes.columns)
    if missing:
        raise ValueError(f"nodes.csv is missing column(s) {sorted(missing)}")
    missing = set(EDGE_COLUMNS) - set(edges.columns)
    if missing:
        raise ValueError(f"edges.csv is missing column(s) {sorted(missing)}")

    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    n = len(nodes)
    if nodes["node_id"].tolist() != list(range(n)):
        raise ValueError("node_id must be 0-based and contiguous")

    tier = nodes["tier"].to_numpy(dtype=np.int64)
    if tier.max() >= N_TIERS or tier.min() < 0:
        raise ValueError(f"tier must be in [0, {N_TIERS})")

    capacity = nodes["capacity"].to_numpy(dtype=np.float64)
    capacity = np.where(np.isnan(capacity), np.inf, capacity)
    capacity[tier == N_TIERS - 1] = np.inf

    groups: list[list[InputGroup]] = [[] for _ in range(n)]
    lead_time: dict[tuple[int, int], int] = {}
    by_group: dict[tuple[int, int], list[tuple[int, float, float]]] = {}

    for row in edges.itertuples(index=False):
        u, v = int(row.supplier), int(row.customer)
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge {u}->{v} references an unknown node id")
        lead = int(row.lead_time)
        if lead < 1:
            raise ValueError(f"edge {u}->{v} has lead_time {lead}; must be >= 1")
        lead_time[(u, v)] = lead
        by_group.setdefault((v, int(row.group)), []).append(
            (u, float(row.coeff), float(row.share))
        )

    for (v, gi), rows in sorted(by_group.items()):
        coeffs = {round(c, 9) for _u, c, _s in rows}
        if len(coeffs) > 1:
            raise ValueError(
                f"node {v} group {gi} has inconsistent BOM coefficients {sorted(coeffs)}; "
                "the coefficient is a property of the group, not of the supplier"
            )
        suppliers = [u for u, _c, _s in rows]
        shares = np.array([s for _u, _c, s in rows], dtype=np.float64)
        if np.any(shares <= 0):
            raise ValueError(f"node {v} group {gi} has a non-positive share")
        total = float(shares.sum())
        if abs(total - 1.0) > 1e-9:
            if not normalise_shares:
                raise ValueError(f"node {v} group {gi} shares sum to {total}, not 1")
            shares = shares / total
        # Groups are stored in ascending group index so the on-disk order is the
        # in-memory order, which keeps edge features reproducible.
        while len(groups[v]) <= gi:
            groups[v].append(None)  # type: ignore[arg-type]
        groups[v][gi] = InputGroup(
            coeff=float(next(iter(coeffs))),
            suppliers=suppliers,
            shares=[float(s) for s in shares],
        )

    for v in range(n):
        if any(g is None for g in groups[v]):
            raise ValueError(f"node {v} has a gap in its group indices")

    net = SupplyNetwork(
        tier=tier,
        region=nodes["region"].to_numpy(dtype=np.int64),
        groups=groups,
        capacity=capacity,
        base_stock=nodes["base_stock"].to_numpy(dtype=np.float64),
        input_cover=nodes["input_cover"].to_numpy(dtype=np.float64),
        lead_time=lead_time,
        mean_demand=nodes["mean_demand"].to_numpy(dtype=np.float64),
        demand_cv=nodes["demand_cv"].to_numpy(dtype=np.float64),
        name=name,
    )
    net.validate()
    net.compute_throughput()
    return net


def write_network_to_csv(net: SupplyNetwork, nodes_csv: str | Path,
                         edges_csv: str | Path) -> tuple[Path, Path]:
    """Write a network out in the loader's format. Useful for round-trip checks."""
    thr = net.throughput if net.throughput.size == net.n_nodes else net.compute_throughput()
    del thr
    nodes = pd.DataFrame({
        "node_id": np.arange(net.n_nodes),
        "tier": net.tier,
        "region": net.region,
        "capacity": np.where(np.isfinite(net.capacity), net.capacity, np.nan),
        "base_stock": net.base_stock,
        "input_cover": net.input_cover,
        "mean_demand": net.mean_demand,
        "demand_cv": net.demand_cv,
    })
    rows = []
    for v, gs in enumerate(net.groups):
        for gi, g in enumerate(gs):
            for u, s in zip(g.suppliers, g.shares, strict=True):
                rows.append({
                    "supplier": u, "customer": v, "group": gi,
                    "coeff": g.coeff, "share": s, "lead_time": net.lead_time[(u, v)],
                })
    np_ = Path(nodes_csv)
    ep = Path(edges_csv)
    np_.parent.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(np_, index=False, lineterminator="\n")
    pd.DataFrame(rows, columns=list(EDGE_COLUMNS)).to_csv(
        ep, index=False, lineterminator="\n"
    )
    return np_, ep


def write_example_csvs(directory: str | Path) -> tuple[Path, Path]:
    """Write a minimal worked example: one dual-sourced group, one sole source.

    Six nodes: two raw suppliers feeding one dual-sourced group at a component
    maker, which sole-sources an assembly plant, a DC and a demand point.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    nodes = pd.DataFrame({
        "node_id": [0, 1, 2, 3, 4, 5],
        "tier": [0, 0, 1, 2, 3, 4],
        "region": [0, 1, 0, 0, 0, 0],
        "capacity": [60.0, 60.0, 60.0, 60.0, 60.0, np.nan],
        "base_stock": [0.0, 0.0, 10.0, 10.0, 10.0, 0.0],
        "input_cover": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "mean_demand": [0.0, 0.0, 0.0, 0.0, 0.0, 40.0],
        "demand_cv": [0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    })
    edges = pd.DataFrame([
        {"supplier": 0, "customer": 2, "group": 0, "coeff": 1.0, "share": 0.6, "lead_time": 1},
        {"supplier": 1, "customer": 2, "group": 0, "coeff": 1.0, "share": 0.4, "lead_time": 3},
        {"supplier": 2, "customer": 3, "group": 0, "coeff": 1.0, "share": 1.0, "lead_time": 2},
        {"supplier": 3, "customer": 4, "group": 0, "coeff": 1.0, "share": 1.0, "lead_time": 1},
        {"supplier": 4, "customer": 5, "group": 0, "coeff": 1.0, "share": 1.0, "lead_time": 1},
    ])
    npath, epath = d / "nodes.csv", d / "edges.csv"
    nodes.to_csv(npath, index=False, lineterminator="\n")
    edges.to_csv(epath, index=False, lineterminator="\n")
    return npath, epath
