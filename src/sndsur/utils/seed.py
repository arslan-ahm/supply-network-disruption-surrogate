"""One-call seeding for Python, NumPy and PyTorch.

Every stochastic decision in this repository is a pure function of a seed and an
index (network id, scenario id, ensemble member), never of iteration order. That
is what makes the determinism check in ``docs/REPRODUCIBILITY.md`` pass on a
different machine with a different number of cores.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed the global RNGs.

    Args:
        seed: The seed.
        deterministic: Also request deterministic kernels and disable the cuDNN
            autotuner. Costs a little speed and buys reproducibility, which for
            a repository whose point is traceable numbers is the right trade.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass


def limit_threads(threads: int = 2) -> None:
    """Cap intra-op parallelism, for both torch and the BLAS underneath it.

    This machine runs several projects concurrently on four cores. The env vars
    must be set before NumPy/torch import their BLAS to take effect, so entry
    points call this first; calling it later still caps torch, which is where
    most of the time goes.
    """
    threads = max(1, int(threads))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        # set_num_interop_threads raises if the pool is already initialised,
        # which is harmless and means someone already called this.
        pass


def child_seed(*parts: int, base: int = 0) -> int:
    """A stable 63-bit seed derived from ``base`` and an index tuple.

    Used everywhere an inner loop needs its own RNG: ``child_seed(net_id,
    scenario_id, base=cfg.run.seed)``. Deriving rather than incrementing means
    adding a scenario to one network does not shift every other network's draws.
    """
    ss = np.random.SeedSequence([base, *[int(p) for p in parts]])
    return int(ss.generate_state(1, dtype=np.uint64)[0] >> 1)
