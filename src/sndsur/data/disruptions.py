"""Disruption specifications and a generator for scenario sampling.

A disruption is a *counterfactual intervention*, not a risk attribute. It names
what fails, when, for how long, and how hard — and the simulator answers what
happens to service. Six mechanisms are supported, chosen because they are the
ones that behave differently in a network rather than because there are six of
them:

``supplier_outage``
    A node's capacity goes to zero. The canonical single point of failure test.
``capacity_reduction``
    A node's capacity is scaled by ``1 - severity``. Partial, and therefore
    interesting: whether it hurts depends on how much slack the node had.
``lead_time_inflation``
    Every outbound lane of a node has its transit time multiplied by
    ``1 + severity``. Adds no shortage of material, only of *timing* — the
    failure mode inventory buffers absorb and capacity slack does not.
``demand_spike``
    A demand point's demand is multiplied by ``1 + severity``. The disruption
    that travels *up* the network, so a model that only propagates downstream
    cannot represent it.
``lane_closure``
    One edge stops carrying material. Distinguishable from a supplier outage
    only if the model has edge-level structure — which is precisely why it is
    the held-out type in the unseen-mechanism generalisation split.
``regional_event``
    Every node in one region takes a capacity reduction. Correlated, and the
    reason geographic clustering is in the network generator at all.

Timing is expressed **relative to the start of the measurement window**, so
``start=0`` is the first measured period. Warm-up is always undisrupted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sndsur.data.network import N_TIERS, SupplyNetwork

#: Canonical order. The index in this tuple is the one-hot position in every
#: feature vector, so appending is safe and reordering is not.
DISRUPTION_TYPES = (
    "supplier_outage",
    "capacity_reduction",
    "lead_time_inflation",
    "demand_spike",
    "lane_closure",
    "regional_event",
)
TYPE_INDEX = {t: i for i, t in enumerate(DISRUPTION_TYPES)}


@dataclass(frozen=True)
class Disruption:
    """One intervention.

    Attributes:
        kind: One of :data:`DISRUPTION_TYPES`.
        target: Node id for node-scoped kinds; region id for ``regional_event``;
            the tail node for ``lane_closure``.
        target2: Head node for ``lane_closure``, otherwise ``-1``.
        start: First affected period, relative to the measurement window. Must
            be >= 0 so warm-up is always clean.
        duration: Number of affected periods. Must be >= 1.
        severity: In ``(0, 1]`` for capacity kinds (1 = total outage); a positive
            multiplier increment for lead-time and demand kinds.
    """

    kind: str
    target: int
    target2: int = -1
    start: int = 0
    duration: int = 1
    severity: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in TYPE_INDEX:
            raise ValueError(f"unknown disruption kind {self.kind!r}")
        if self.start < 0:
            raise ValueError("start must be >= 0 (warm-up is always undisrupted)")
        if self.duration < 1:
            raise ValueError("duration must be >= 1 period")
        if self.severity <= 0:
            raise ValueError("severity must be positive")
        if self.kind == "lane_closure" and self.target2 < 0:
            raise ValueError("lane_closure needs both endpoints")

    @property
    def end(self) -> int:
        """One past the last affected period."""
        return self.start + self.duration

    def active(self, rel: int) -> bool:
        return self.start <= rel < self.end

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": int(self.target),
            "target2": int(self.target2),
            "start": int(self.start),
            "duration": int(self.duration),
            "severity": float(self.severity),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Disruption:
        return Disruption(
            kind=str(d["kind"]),
            target=int(d["target"]),
            target2=int(d.get("target2", -1)),
            start=int(d["start"]),
            duration=int(d["duration"]),
            severity=float(d["severity"]),
        )


@dataclass
class DisruptionSet:
    """A set of simultaneous disruptions, with the simulator's query interface.

    Multiple disruptions of the same kind on the same target compose
    multiplicatively for capacity and demand and additively for lead time, which
    is the only composition rule that keeps a double hit at least as bad as a
    single one. That monotonicity is asserted in the tests.
    """

    items: list[Disruption]
    #: Region label per node, needed to resolve ``regional_event``. Set by
    #: :meth:`bind`; without it a regional event is inert and a warning-free
    #: silent no-op, so :meth:`bind` is called by every generator path.
    region: np.ndarray | None = None

    def bind(self, net: SupplyNetwork) -> DisruptionSet:
        """Attach the network's region labels. Returns self for chaining."""
        self.region = net.region
        return self

    # -- timing ------------------------------------------------------------ #

    @property
    def start(self) -> int:
        return min((d.start for d in self.items), default=0)

    @property
    def end(self) -> int:
        return max((d.end for d in self.items), default=0)

    # -- simulator queries ------------------------------------------------- #

    def capacity_multiplier(self, node: int, rel: int) -> float:
        """Factor on ``node``'s per-period capacity at relative period ``rel``."""
        m = 1.0
        for d in self.items:
            if not d.active(rel):
                continue
            node_scoped = (
                d.kind in ("supplier_outage", "capacity_reduction") and d.target == node
            )
            regional = (
                d.kind == "regional_event"
                and self.region is not None
                and int(self.region[node]) == d.target
            )
            if node_scoped or regional:
                m *= max(0.0, 1.0 - d.severity)
        return m

    def demand_multiplier(self, node: int, rel: int) -> float:
        """Factor on ``node``'s external demand at relative period ``rel``."""
        m = 1.0
        for d in self.items:
            if d.active(rel) and d.kind == "demand_spike" and d.target == node:
                m *= 1.0 + d.severity
        return m

    def effective_lead_time(self, u: int, v: int, rel: int, base: int) -> int:
        """Transit time on ``(u, v)`` for a shipment leaving at ``rel``.

        Inflation is applied at *departure*, which is the physically correct
        reading: a shipment that leaves during a port closure is slow, and one
        that left before it is not retroactively delayed. The alternative
        (inflating in-transit shipments) would require re-timing the pipeline
        every period for no additional realism.
        """
        add = 0.0
        for d in self.items:
            if d.active(rel) and d.kind == "lead_time_inflation" and d.target == u:
                add += d.severity
        if add == 0.0:
            return base
        return max(1, int(round(base * (1.0 + add))))

    def lane_open(self, u: int, v: int, rel: int) -> bool:
        """False when edge ``(u, v)`` is closed at relative period ``rel``."""
        for d in self.items:
            if (
                d.active(rel)
                and d.kind == "lane_closure"
                and d.target == u
                and d.target2 == v
            ):
                return False
        return True

    # -- serialisation ----------------------------------------------------- #

    def to_list(self) -> list[dict[str, Any]]:
        return [d.to_dict() for d in self.items]

    @staticmethod
    def from_list(rows: list[dict[str, Any]]) -> DisruptionSet:
        return DisruptionSet([Disruption.from_dict(r) for r in rows])

    def describe(self) -> str:
        return "; ".join(
            f"{d.kind}@{d.target}"
            + (f"->{d.target2}" if d.target2 >= 0 else "")
            + f" t={d.start}+{d.duration} s={d.severity:.2f}"
            for d in self.items
        )


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


@dataclass
class DisruptionSpec:
    """Controls for :func:`sample_disruptions`.

    Attributes:
        kinds: Which mechanisms may be sampled. Restricting this is how the
            held-out-mechanism generalisation split is built.
        n_points: Inclusive range of simultaneous disruptions per scenario.
        start: Inclusive range of start periods.
        duration: Inclusive range of durations in periods.
        severity: Inclusive range of severities for capacity kinds.
        lead_severity: Inclusive range for ``lead_time_inflation``.
        demand_severity: Inclusive range for ``demand_spike``.
        upstream_bias: Probability of drawing a node target from tiers 0-1
            rather than uniformly. Real disruption reporting is concentrated
            upstream, and an unbiased sample spends most of its scenarios on
            distribution centres where the answer is trivially "yes it hurts".
    """

    kinds: tuple[str, ...] = DISRUPTION_TYPES
    n_points: tuple[int, int] = (1, 1)
    start: tuple[int, int] = (2, 8)
    duration: tuple[int, int] = (4, 18)
    severity: tuple[float, float] = (0.40, 1.0)
    lead_severity: tuple[float, float] = (0.5, 3.0)
    demand_severity: tuple[float, float] = (0.4, 2.0)
    upstream_bias: float = 0.6


def sample_disruptions(
    net: SupplyNetwork, spec: DisruptionSpec, rng: np.random.Generator
) -> DisruptionSet:
    """Draw one scenario's disruption set from ``spec``.

    Returns:
        A :class:`DisruptionSet`, already bound to ``net``.
    """
    k = int(rng.integers(spec.n_points[0], spec.n_points[1] + 1))
    items: list[Disruption] = []
    for _ in range(k):
        kind = str(rng.choice(np.array(spec.kinds, dtype=object)))
        start = int(rng.integers(spec.start[0], spec.start[1] + 1))
        duration = int(rng.integers(spec.duration[0], spec.duration[1] + 1))

        if kind == "demand_spike":
            target = int(rng.choice(net.demand_nodes))
            sev = float(rng.uniform(*spec.demand_severity))
            items.append(Disruption(kind, target, -1, start, duration, sev))
        elif kind == "lead_time_inflation":
            target = _pick_producer(net, spec, rng)
            sev = float(rng.uniform(*spec.lead_severity))
            items.append(Disruption(kind, target, -1, start, duration, sev))
        elif kind == "lane_closure":
            edges = net.edges()
            u, v = edges[int(rng.integers(0, len(edges)))]
            items.append(Disruption(kind, int(u), int(v), start, duration, 1.0))
        elif kind == "regional_event":
            target = int(rng.integers(0, net.n_regions))
            sev = float(rng.uniform(*spec.severity))
            items.append(Disruption(kind, target, -1, start, duration, sev))
        else:  # supplier_outage | capacity_reduction
            target = _pick_producer(net, spec, rng)
            sev = 1.0 if kind == "supplier_outage" else float(rng.uniform(*spec.severity))
            items.append(Disruption(kind, target, -1, start, duration, sev))
    return DisruptionSet(items).bind(net)


def _pick_producer(
    net: SupplyNetwork, spec: DisruptionSpec, rng: np.random.Generator
) -> int:
    """A non-demand node, biased upstream per ``spec.upstream_bias``."""
    upstream = np.flatnonzero(net.tier <= 1)
    allp = np.flatnonzero(net.tier < N_TIERS - 1)
    if upstream.size and rng.random() < spec.upstream_bias:
        return int(rng.choice(upstream))
    return int(rng.choice(allp))
