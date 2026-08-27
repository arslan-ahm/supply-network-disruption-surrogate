"""Shared fixtures.

The hand-built networks here are the backbone of the simulator tests. They are
small enough that the correct answer can be worked out on paper, which is the
only way to test a simulator that is itself the ground truth for everything else.
"""

from __future__ import annotations

import numpy as np
import pytest

from sndsur.config import load_config
from sndsur.data.network import (
    N_TIERS,
    InputGroup,
    NetworkSpec,
    SupplyNetwork,
    generate_network,
)
from sndsur.sim.simulator import SimConfig


def make_chain(
    n_tiers: int = 5,
    lead: int = 1,
    capacity: float = 1e4,
    base_stock: float = 0.0,
    input_cover: float = 0.0,
    demand: float = 10.0,
    cv: float = 0.0,
) -> SupplyNetwork:
    """A single linear chain, one node per tier, deterministic demand.

    Every quantity is exact, so shortages appear at arithmetically predictable
    periods. This is the network the hand-solvable tests use.
    """
    n = n_tiers
    groups: list[list[InputGroup]] = [[]]
    for v in range(1, n):
        groups.append([InputGroup(1.0, [v - 1], [1.0])])
    cap = np.full(n, capacity, dtype=float)
    cap[-1] = np.inf
    md = np.zeros(n)
    md[-1] = demand
    cvs = np.zeros(n)
    cvs[-1] = cv
    # The demand point must sit at the last echelon whatever the chain length,
    # so a 3-node chain is raw -> component -> demand, not raw -> component ->
    # assembly. The validator enforces this and it is what makes a short chain a
    # legal network rather than a special case.
    tier = np.arange(n)
    tier[-1] = N_TIERS - 1
    net = SupplyNetwork(
        tier=tier,
        region=np.zeros(n, dtype=int),
        groups=groups,
        capacity=cap,
        base_stock=np.full(n, base_stock),
        input_cover=np.full(n, input_cover),
        lead_time={(v - 1, v): lead for v in range(1, n)},
        mean_demand=md,
        demand_cv=cvs,
        name="chain",
    )
    net.validate()
    net.compute_throughput()
    return net


def make_diamond() -> SupplyNetwork:
    """Two tier-0 suppliers feeding one dual-sourced group, then a chain.

    Node 0 and node 1 are substitutable within node 2's single group at a 50/50
    split. Used to test that multi-sourcing genuinely halves exposure and that
    ``allow_resourcing`` recovers the rest.
    """
    groups: list[list[InputGroup]] = [
        [],
        [],
        [InputGroup(1.0, [0, 1], [0.5, 0.5])],
        [InputGroup(1.0, [2], [1.0])],
        [InputGroup(1.0, [3], [1.0])],
    ]
    cap = np.array([1e4, 1e4, 1e4, 1e4, np.inf])
    net = SupplyNetwork(
        tier=np.array([0, 0, 1, 2, 4]),
        region=np.zeros(5, dtype=int),
        groups=groups,
        capacity=cap,
        base_stock=np.zeros(5),
        input_cover=np.zeros(5),
        lead_time={(0, 2): 1, (1, 2): 1, (2, 3): 1, (3, 4): 1},
        mean_demand=np.array([0.0, 0, 0, 0, 10.0]),
        demand_cv=np.zeros(5),
        name="diamond",
    )
    net.validate()
    net.compute_throughput()
    return net


def make_sole_source_hub() -> SupplyNetwork:
    """One tier-0 hub sole-sourcing three plants, beside a well-hedged supplier.

    This is the structure the headline argument is about. Node 0 is the hub: it
    is the only member of the input group of nodes 3, 4 and 5. Node 1 and node 2
    jointly supply node 6, so neither is a single point of failure. Attribute-wise
    node 1 can look worse than node 0 (bigger, more customers by raw count) while
    being far less critical.
    """
    groups: list[list[InputGroup]] = [
        [], [], [],
        [InputGroup(1.0, [0], [1.0])],
        [InputGroup(1.0, [0], [1.0])],
        [InputGroup(1.0, [0], [1.0])],
        [InputGroup(1.0, [1, 2], [0.5, 0.5])],
        [InputGroup(1.0, [3, 4], [0.5, 0.5]), InputGroup(1.0, [5, 6], [0.5, 0.5])],
        [InputGroup(1.0, [7], [1.0])],
    ]
    cap = np.full(9, 1e4)
    cap[-1] = np.inf
    net = SupplyNetwork(
        tier=np.array([0, 0, 0, 1, 1, 1, 1, 3, 4]),
        region=np.zeros(9, dtype=int),
        groups=groups,
        capacity=cap,
        base_stock=np.zeros(9),
        input_cover=np.zeros(9),
        lead_time={
            (0, 3): 1, (0, 4): 1, (0, 5): 1, (1, 6): 1, (2, 6): 1,
            (3, 7): 1, (4, 7): 1, (5, 7): 1, (6, 7): 1, (7, 8): 1,
        },
        mean_demand=np.array([0.0, 0, 0, 0, 0, 0, 0, 0, 12.0]),
        demand_cv=np.zeros(9),
        name="hub",
    )
    net.validate()
    net.compute_throughput()
    return net


@pytest.fixture
def chain() -> SupplyNetwork:
    return make_chain()


@pytest.fixture
def diamond() -> SupplyNetwork:
    return make_diamond()


@pytest.fixture
def hub() -> SupplyNetwork:
    return make_sole_source_hub()


@pytest.fixture
def small_net() -> SupplyNetwork:
    """A generated network small enough for fast tests but structurally real."""
    return generate_network(NetworkSpec(n_per_tier=(6, 5, 4, 3, 4), n_regions=2), seed=3)


@pytest.fixture
def sim_cfg() -> SimConfig:
    """Warm-up comfortably exceeds the cumulative lead time of the fixtures."""
    return SimConfig(warmup=20, horizon=20)


@pytest.fixture
def smoke_cfg():
    """The smoke configuration, with the cache redirected under tmp at call sites."""
    return load_config("configs/smoke.yaml")
