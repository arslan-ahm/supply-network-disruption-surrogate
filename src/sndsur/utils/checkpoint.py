"""Ensemble checkpoint cache.

Three pipeline stages need the same trained ensemble: the method comparison, the
criticality experiment and the efficiency benchmark. Training it three times
would triple the project's compute budget for no scientific gain, so it is
trained once and cached, keyed by every config field that can change the weights.

The key is deliberately explicit rather than a hash of the whole config: a change
to ``eval.bootstrap`` cannot change the weights, and invalidating the cache for it
would be waste, while a change to ``model.hidden`` must invalidate it. Getting
that list wrong is a real hazard, so the key is written out in full and the
cached file records the config it came from for cross-checking.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from sndsur.config import Config
from sndsur.utils.logging import get_logger

LOG = get_logger()
CKPT_DIR = Path("checkpoints")


def ensemble_key(cfg: Config) -> str:
    """Cache key covering everything that affects the trained weights."""
    d = cfg.dataset
    m = cfg.model
    t = cfg.train
    parts = [
        cfg.run.seed,
        *cfg.netgen.n_per_tier,
        cfg.netgen.sole_source_prob,
        cfg.netgen.large_scale,
        d.n_train_networks,
        d.n_shift_networks,
        d.n_large_networks,
        d.scenarios_per_train_network,
        d.scenarios_per_eval_network,
        d.traj_periods,
        d.max_train_scenarios,
        d.max_eval_scenarios,
        m.traj_horizon,
        cfg.sim.warmup,
        cfg.sim.horizon,
        cfg.sim.allow_resourcing,
        cfg.disgen.holdout_kind,
        m.hidden, m.layers, m.dropout, m.bidirectional, m.use_edge_features,
        m.temporal_decoder, m.heteroscedastic, len(m.quantiles), m.ensemble,
        m.no_graph_blocks, m.no_graph_width,
        t.lr, t.weight_decay, t.epochs, t.batch_graphs, t.grad_clip,
        t.scheduler, t.warmup_epochs, t.w_impact, t.w_traj, t.w_timing,
        t.w_nll, t.w_quantile, t.patience,
    ]
    return "_".join(str(p) for p in parts)


def ensemble_path(cfg: Config) -> Path:
    """Cache file for this configuration's ensemble.

    Hashed with SHA-256 rather than the builtin ``hash()``. Python randomises
    string hashing per process unless ``PYTHONHASHSEED`` is fixed *before*
    interpreter start, so the builtin produced a different filename in every
    process and the cache never hit across runs - the criticality and efficiency
    stages silently retrained the ensemble they were supposed to reuse, which is
    exactly the compute this cache exists to save.
    """
    digest = hashlib.sha256(ensemble_key(cfg).encode("utf-8")).hexdigest()[:16]
    return CKPT_DIR / f"ensemble_{digest}.pt"


def save_ensemble(cfg: Config, models: list, records: list[dict]) -> Path:
    """Persist an ensemble's weights and its training records."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    p = ensemble_path(cfg)
    torch.save(
        {
            "key": ensemble_key(cfg),
            "config": json.loads(str(cfg)),
            "state_dicts": [m.state_dict() for m in models],
            "records": records,
        },
        p,
    )
    return p


def load_ensemble(cfg: Config, node_dim: int, edge_dim: int, traj_periods: int):
    """Rebuild a cached ensemble, or return ``(None, None)`` if there is no hit.

    A key mismatch inside the file is treated as a miss rather than an error: the
    only way it can happen is a Python hash collision, and silently training
    fresh weights is the safe response.
    """
    p = ensemble_path(cfg)
    if not p.exists():
        return None, None
    blob = torch.load(p, map_location="cpu", weights_only=False)
    if blob.get("key") != ensemble_key(cfg):
        LOG.warning("checkpoint key mismatch at %s; retraining", p)
        return None, None
    from sndsur.models.surrogate import build_surrogate

    models = []
    for sd in blob["state_dicts"]:
        m = build_surrogate(cfg, node_dim, edge_dim, traj_periods)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    LOG.info("loaded cached ensemble (%d members) from %s", len(models), p.name)
    return models, blob["records"]


def train_or_load_ensemble(cfg: Config, ds, node_dim: int, edge_dim: int, run=None):
    """Load the cached ensemble if present, otherwise train it and cache it.

    Returns:
        ``(models, records)``. ``records`` carries ``cached: True`` when loaded,
        so the efficiency table never reports a cached load as a training time.
    """
    models, records = load_ensemble(cfg, node_dim, edge_dim, ds.traj_periods)
    if models is not None:
        for r in records:
            r["cached"] = True
        return models, records
    from sndsur.engine.trainer import train_ensemble

    models, records = train_ensemble(cfg, ds, node_dim, edge_dim, run=run)
    for r in records:
        r["cached"] = False
    save_ensemble(cfg, models, records)
    return models, records
