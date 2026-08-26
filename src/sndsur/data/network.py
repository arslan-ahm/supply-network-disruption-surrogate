"""Supply-network representation and a generator for realistic topologies.

The central design decision in this module is the **input group**. A node does not
simply have a list of suppliers; it has a list of *groups*, and

* suppliers **within** a group are substitutable (multi-sourcing, with sourcing
  shares that sum to one), and
* groups are **complementary** — every group must deliver material or the node
  cannot produce anything (a Leontief production function).

That distinction is the whole reason this repository exists. A per-supplier risk
score cannot express "this supplier is the only member of its group, and that
group feeds four assembly plants", because that fact is not a property of the
supplier's features — it is a property of the group structure around it. A
sole-source group is a hard single point of failure; a three-supplier group is a
soft one. Both look identical in a table of supplier attributes.

Structure of a generated network (five tiers, following the standard
raw → component → assembly → distribution → demand echelon convention):

    tier 0  raw suppliers      (no inputs; capacity-constrained producers)
    tier 1  component makers
    tier 2  assembly plants
    tier 3  distribution centres
    tier 4  demand points      (no production; consume and report service level)

Geography enters through a region label per node. Lead times and sourcing
preference both depend on whether supplier and customer share a region, which is
what produces correlated exposure to a regional event — the failure mode that
motivated most of the post-2020 supply-chain resilience literature
(Ivanov & Dolgui, 2020).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Echelon names by tier index. Used for feature encoding and for reporting.
TIER_NAMES = ("raw", "component", "assembly", "distribution", "demand")
N_TIERS = len(TIER_NAMES)


@dataclass
class InputGroup:
    """One complementary material requirement of a node.

    Attributes:
        coeff: Units of this group's material consumed per unit of the node's own
            output (the BOM coefficient ``q``).
        suppliers: Node indices that can serve this group. Length 1 means the
            group is **sole-sourced**.
        shares: Sourcing split across ``suppliers``, summing to 1.
    """

    coeff: float
    suppliers: list[int]
    shares: list[float]

    @property
    def sole_source(self) -> bool:
        return len(self.suppliers) == 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "coeff": float(self.coeff),
            "suppliers": [int(s) for s in self.suppliers],
            "shares": [float(s) for s in self.shares],
        }


@dataclass
class SupplyNetwork:
    """A directed multi-echelon supply network with a bill of materials.

    Every array is indexed by node id. Node ids are assigned tier by tier in
    ascending order, so ``tier`` is non-decreasing and a plain ``range(n_nodes)``
    loop is already a valid topological order — a property the simulator relies
    on and :meth:`validate` enforces.

    Attributes:
        tier: Echelon index per node, in ``[0, N_TIERS)``.
        region: Geographic region label per node.
        groups: Input groups per node. Empty for tier-0 suppliers.
        capacity: Units of own output producible per period. ``inf`` for demand
            points, which do not produce.
        base_stock: Finished-goods order-up-to level ``S`` per node.
        input_cover: Periods of input demand held as an input safety stock,
            per node. Multiplied by the group's material rate to get ``S_in``.
        lead_time: ``(u, v) -> periods`` transit time, one entry per edge.
        mean_demand: Expected demand per period at each demand point; 0 elsewhere.
        demand_cv: Coefficient of variation of demand at each demand point.
        throughput: Expected units per period, propagated up the BOM from demand.
            Derived, not sampled — see :meth:`compute_throughput`.
        name: Identifier used in result tables.
    """

    tier: np.ndarray
    region: np.ndarray
    groups: list[list[InputGroup]]
    capacity: np.ndarray
    base_stock: np.ndarray
    input_cover: np.ndarray
    lead_time: dict[tuple[int, int], int]
    mean_demand: np.ndarray
    demand_cv: np.ndarray
    throughput: np.ndarray = field(default_factory=lambda: np.zeros(0))
    name: str = "network"

    # ------------------------------------------------------------------ #
    # Basic structure
    # ------------------------------------------------------------------ #

    @property
    def n_nodes(self) -> int:
        return int(self.tier.shape[0])

    @property
    def demand_nodes(self) -> np.ndarray:
        """Indices of tier-4 nodes, in ascending order."""
        return np.flatnonzero(self.tier == N_TIERS - 1)

    @property
    def n_regions(self) -> int:
        return int(self.region.max()) + 1 if self.n_nodes else 0

    def edges(self) -> list[tuple[int, int]]:
        """All ``(supplier, customer)`` pairs, deduplicated and sorted."""
        seen: set[tuple[int, int]] = set()
        for v, gs in enumerate(self.groups):
            for g in gs:
                for u in g.suppliers:
                    seen.add((u, v))
        return sorted(seen)

    def customers(self) -> list[list[int]]:
        """``customers[u]`` lists the nodes that ``u`` supplies."""
        out: list[list[int]] = [[] for _ in range(self.n_nodes)]
        for u, v in self.edges():
            out[u].append(v)
        return out

    def group_of(self, u: int, v: int) -> int:
        """Index of the group of ``v`` that ``u`` serves.

        Raises:
            KeyError: if ``u`` does not supply ``v``.
        """
        for gi, g in enumerate(self.groups[v]):
            if u in g.suppliers:
                return gi
        raise KeyError(f"{u} does not supply {v}")

    def share(self, u: int, v: int) -> float:
        """Sourcing share of ``u`` inside the group of ``v`` that it serves."""
        gi = self.group_of(u, v)
        g = self.groups[v][gi]
        return float(g.shares[g.suppliers.index(u)])

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #

    def compute_throughput(self) -> np.ndarray:
        """Expected units per period at every node, propagated up the BOM.

        Demand points contribute their mean demand; every other node's
        throughput is the sum over its customers of
        ``coeff * share * customer_throughput``. Because node ids are in
        topological order, one reverse sweep suffices.

        This is the quantity that sizes capacity and inventory, so it must be
        computed the same way at generation time and at feature time — hence one
        method rather than two ad-hoc loops.
        """
        thr = np.zeros(self.n_nodes, dtype=np.float64)
        thr[self.demand_nodes] = self.mean_demand[self.demand_nodes]
        for v in range(self.n_nodes - 1, -1, -1):
            for g in self.groups[v]:
                for u, w in zip(g.suppliers, g.shares, strict=True):
                    thr[u] += g.coeff * w * thr[v]
        self.throughput = thr
        return thr

    def bom_depth(self) -> np.ndarray:
        """Longest path in periods-agnostic hops from each node to a demand point.

        Depth 0 for demand points. Used as a topology feature and as one of the
        heuristic baseline's inputs.
        """
        depth = np.zeros(self.n_nodes, dtype=np.int64)
        cust = self.customers()
        for v in range(self.n_nodes - 1, -1, -1):
            if cust[v]:
                depth[v] = 1 + max(depth[c] for c in cust[v])
        return depth

    def lead_time_to_demand(self) -> np.ndarray:
        """Maximum cumulative transit time from each node to any demand point.

        The physically meaningful "how long before a failure here is felt"
        quantity is the *minimum* over paths, but the quantity that governs how
        long the shortage lasts is the maximum. Both are computed; this returns
        the max, and :meth:`min_lead_time_to_demand` the min.
        """
        return self._lead_accumulate(reduce=max)

    def min_lead_time_to_demand(self) -> np.ndarray:
        """Minimum cumulative transit time from each node to any demand point."""
        return self._lead_accumulate(reduce=min)

    def _lead_accumulate(self, reduce) -> np.ndarray:
        out = np.zeros(self.n_nodes, dtype=np.float64)
        cust = self.customers()
        for v in range(self.n_nodes - 1, -1, -1):
            if cust[v]:
                out[v] = reduce(self.lead_time[(v, c)] + out[c] for c in cust[v])
        return out

    def sole_source_flags(self) -> np.ndarray:
        """1.0 where a node is the only member of at least one customer's group."""
        flag = np.zeros(self.n_nodes, dtype=np.float64)
        for gs in self.groups:
            for g in gs:
                if g.sole_source:
                    flag[g.suppliers[0]] = 1.0
        return flag

    def downstream_reach(self) -> np.ndarray:
        """Number of demand points reachable from each node.

        A node with reach 0 cannot affect service at all; a node with high reach
        is structurally important regardless of its own attributes.
        """
        cust = self.customers()
        demand = set(self.demand_nodes.tolist())
        reach: list[set[int]] = [set() for _ in range(self.n_nodes)]
        for v in range(self.n_nodes - 1, -1, -1):
            if v in demand:
                reach[v] = {v}
            else:
                acc: set[int] = set()
                for c in cust[v]:
                    acc |= reach[c]
                reach[v] = acc
        return np.array([len(r) for r in reach], dtype=np.float64)

    def sole_source_reach(self) -> np.ndarray:
        """Demand points reachable from a node *only* through sole-source groups.

        This is the honest structural definition of a single point of failure:
        if every group on every path is sole-sourced, removing the node removes
        the material entirely. It is the quantity the heuristic baseline is built
        around, and the one the reference-style feature model has no access to.
        """
        cust = self.customers()
        demand = set(self.demand_nodes.tolist())
        reach: list[set[int]] = [set() for _ in range(self.n_nodes)]
        for v in range(self.n_nodes - 1, -1, -1):
            if v in demand:
                reach[v] = {v}
                continue
            acc: set[int] = set()
            for c in cust[v]:
                gi = self.group_of(v, c)
                if self.groups[c][gi].sole_source:
                    acc |= reach[c]
            reach[v] = acc
        return np.array([len(r) for r in reach], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Assert the invariants the simulator depends on.

        Raises:
            ValueError: on any structural violation. Called by the generator and
                by the CSV loader, so a malformed user network fails loudly at
                load time rather than producing quietly wrong service levels.
        """
        n = self.n_nodes
        if not np.all(np.diff(self.tier) >= 0):
            raise ValueError("node ids must be in non-decreasing tier order")
        for v, gs in enumerate(self.groups):
            if self.tier[v] == 0 and gs:
                raise ValueError(f"tier-0 node {v} must have no input groups")
            if self.tier[v] > 0 and not gs:
                raise ValueError(f"node {v} in tier {self.tier[v]} has no input groups")
            for g in gs:
                if not g.suppliers:
                    raise ValueError(f"node {v} has an empty group")
                if abs(sum(g.shares) - 1.0) > 1e-9:
                    raise ValueError(f"node {v} group shares sum to {sum(g.shares)}")
                if g.coeff <= 0:
                    raise ValueError(f"node {v} has non-positive BOM coefficient")
                for u in g.suppliers:
                    if not 0 <= u < n:
                        raise ValueError(f"node {v} references out-of-range supplier {u}")
                    if self.tier[u] >= self.tier[v]:
                        raise ValueError(
                            f"edge {u}->{v} does not go up an echelon "
                            f"({self.tier[u]} -> {self.tier[v]})"
                        )
        for e in self.edges():
            if e not in self.lead_time:
                raise ValueError(f"missing lead time for edge {e}")
            if self.lead_time[e] < 1:
                raise ValueError(f"lead time for {e} must be >= 1 period")
        if np.any(self.mean_demand[self.demand_nodes] <= 0):
            raise ValueError("every demand point needs positive mean demand")
        if np.any(self.mean_demand[self.tier < N_TIERS - 1] != 0):
            raise ValueError("only demand points may carry external demand")
        if self.demand_nodes.size == 0:
            raise ValueError("network has no demand points")


# ---------------------------------------------------------------------- #
# Generator
# ---------------------------------------------------------------------- #


@dataclass
class NetworkSpec:
    """Controls for :func:`generate_network`.

    Attributes:
        n_per_tier: Node counts for the five tiers.
        n_regions: Number of geographic regions.
        same_region_prob: Probability that a supplier is drawn from the
            customer's own region. Above 1/n_regions this creates geographic
            clustering and therefore correlated regional exposure.
        groups_per_node: Inclusive range of input groups per producing node.
        sole_source_prob: Probability that a group is sole-sourced. This is the
            single most consequential structural knob in the repository.
        max_suppliers_per_group: Cap on multi-sourced group size.
        preferential_exponent: Exponent on existing out-degree when choosing
            suppliers. 0 gives uniform attachment, 1 gives a scale-free-like
            degree distribution where a few suppliers serve very many customers.
        lead_time_same_region: Inclusive lead-time range within a region.
        lead_time_cross_region: Inclusive lead-time range across regions.
        capacity_slack: Inclusive range of capacity above expected throughput.
        base_stock_periods: Inclusive range of finished-goods cover in periods.
        input_cover_periods: Inclusive range of input-side cover in periods.
        bom_coeff: Inclusive range of BOM coefficients.
        demand_mean: Inclusive range of mean demand per demand point.
        demand_cv: Coefficient of variation of demand.
    """

    n_per_tier: tuple[int, int, int, int, int] = (16, 14, 10, 6, 8)
    n_regions: int = 4
    same_region_prob: float = 0.55
    groups_per_node: tuple[int, int] = (1, 3)
    sole_source_prob: float = 0.40
    max_suppliers_per_group: int = 3
    preferential_exponent: float = 1.0
    lead_time_same_region: tuple[int, int] = (1, 2)
    lead_time_cross_region: tuple[int, int] = (2, 5)
    capacity_slack: tuple[float, float] = (0.10, 0.60)
    base_stock_periods: tuple[float, float] = (0.0, 2.0)
    input_cover_periods: tuple[float, float] = (0.5, 2.5)
    bom_coeff: tuple[float, float] = (0.8, 1.6)
    demand_mean: tuple[float, float] = (8.0, 20.0)
    demand_cv: float = 0.20

    def scaled(self, factor: float) -> NetworkSpec:
        """A copy with every tier scaled by ``factor`` (min 2 nodes per tier).

        Used to build the larger-network generalisation split without changing
        any other structural parameter, so the shift is *size only*.
        """
        n = tuple(max(2, int(round(c * factor))) for c in self.n_per_tier)
        from dataclasses import replace

        return replace(self, n_per_tier=n)  # type: ignore[arg-type]


def generate_network(spec: NetworkSpec, seed: int, name: str = "network") -> SupplyNetwork:
    """Sample a supply network from ``spec``.

    Suppliers are chosen with preferential attachment biased towards the
    customer's own region, which produces the two features that make real
    supply networks fragile in ways a node-level model cannot see: a heavy-tailed
    out-degree distribution (a few hub suppliers) and geographic concentration
    (a regional event hits many nodes at once).

    Capacity and inventory are sized from the *derived* expected throughput
    rather than sampled independently, so a large hub supplier automatically has
    large capacity. Without this, hubs would be trivially and unrealistically
    fragile and every experiment would be too easy.

    Args:
        spec: Structural controls.
        seed: RNG seed. The same seed gives the same network.
        name: Identifier carried into result tables.

    Returns:
        A validated :class:`SupplyNetwork`.
    """
    rng = np.random.default_rng(seed)
    counts = list(spec.n_per_tier)
    n = sum(counts)

    tier = np.concatenate([np.full(c, t, dtype=np.int64) for t, c in enumerate(counts)])
    region = rng.integers(0, spec.n_regions, size=n)
    tier_nodes = [np.flatnonzero(tier == t) for t in range(N_TIERS)]

    groups: list[list[InputGroup]] = [[] for _ in range(n)]
    out_degree = np.zeros(n, dtype=np.float64)

    for t in range(1, N_TIERS):
        pool = tier_nodes[t - 1]
        for v in tier_nodes[t]:
            n_groups = int(rng.integers(spec.groups_per_node[0], spec.groups_per_node[1] + 1))
            n_groups = min(n_groups, len(pool))
            used: set[int] = set()
            for _ in range(n_groups):
                sole = rng.random() < spec.sole_source_prob
                want = 1 if sole else int(rng.integers(2, spec.max_suppliers_per_group + 1))
                picks = _pick_suppliers(
                    rng, pool, want, region[v], region, out_degree, spec, exclude=used
                )
                if not picks:
                    continue
                used.update(picks)
                out_degree[picks] += 1.0
                shares = rng.dirichlet(np.full(len(picks), 4.0))
                # Dirichlet(4) keeps splits reasonably balanced; a concentration
                # of 1 produced many "nominally dual-sourced" groups with a 97/3
                # split, which are sole-source in all but name and made the
                # sole_source_prob knob meaningless.
                groups[v].append(
                    InputGroup(
                        coeff=float(rng.uniform(*spec.bom_coeff)),
                        suppliers=list(picks),
                        shares=[float(s) for s in shares],
                    )
                )
            if not groups[v]:
                u = int(rng.choice(pool))
                out_degree[u] += 1.0
                groups[v].append(InputGroup(float(rng.uniform(*spec.bom_coeff)), [u], [1.0]))

    # Demand points consume finished product one-for-one; a BOM coefficient
    # other than 1 there would just rescale the reported service level.
    for v in tier_nodes[N_TIERS - 1]:
        for g in groups[v]:
            g.coeff = 1.0

    mean_demand = np.zeros(n, dtype=np.float64)
    dnodes = tier_nodes[N_TIERS - 1]
    mean_demand[dnodes] = rng.uniform(*spec.demand_mean, size=len(dnodes))
    demand_cv = np.zeros(n, dtype=np.float64)
    demand_cv[dnodes] = spec.demand_cv

    net = SupplyNetwork(
        tier=tier,
        region=region,
        groups=groups,
        capacity=np.zeros(n),
        base_stock=np.zeros(n),
        input_cover=np.zeros(n),
        lead_time={},
        mean_demand=mean_demand,
        demand_cv=demand_cv,
        name=name,
    )

    for u, v in net.edges():
        lo, hi = (
            spec.lead_time_same_region
            if region[u] == region[v]
            else spec.lead_time_cross_region
        )
        net.lead_time[(u, v)] = int(rng.integers(lo, hi + 1))

    thr = net.compute_throughput()
    slack = rng.uniform(*spec.capacity_slack, size=n)
    net.capacity = thr * (1.0 + slack)
    net.capacity[dnodes] = np.inf  # demand points do not produce
    net.base_stock = thr * rng.uniform(*spec.base_stock_periods, size=n)
    net.base_stock[dnodes] = 0.0
    net.input_cover = rng.uniform(*spec.input_cover_periods, size=n)

    net.validate()
    return net


def _pick_suppliers(
    rng: np.random.Generator,
    pool: np.ndarray,
    want: int,
    cust_region: int,
    region: np.ndarray,
    out_degree: np.ndarray,
    spec: NetworkSpec,
    exclude: set[int],
) -> list[int]:
    """Sample ``want`` distinct suppliers with regional and degree preference."""
    cand = np.array([u for u in pool if u not in exclude], dtype=np.int64)
    if cand.size == 0:
        return []
    want = min(want, cand.size)

    weight = (1.0 + out_degree[cand]) ** spec.preferential_exponent
    same = region[cand] == cust_region
    # Convert same_region_prob into a multiplicative bias. With R regions the
    # unbiased same-region mass is 1/R, so the factor below reproduces the
    # requested marginal probability in expectation for a uniform pool.
    p = float(np.clip(spec.same_region_prob, 1e-6, 1 - 1e-6))
    r = spec.n_regions
    bias = (p * (r - 1.0)) / max(1.0 - p, 1e-9)
    weight = weight * np.where(same, bias, 1.0)
    weight = weight / weight.sum()
    return [int(x) for x in rng.choice(cand, size=want, replace=False, p=weight)]
