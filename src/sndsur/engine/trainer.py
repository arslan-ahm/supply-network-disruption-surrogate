"""Training loop and prediction collection for the surrogate.

One loop trains every variant. What differs between the surrogate, the
no-message-passing ablation and each component ablation is a config switch, never
a code path — which is what makes an ablation attributable to the mechanism it
names rather than to an incidental difference in how it was trained.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from sndsur.config import Config
from sndsur.data.scenarios import ScenarioDataset
from sndsur.engine.batching import Batch, iterate_batches
from sndsur.engine.losses import surrogate_loss
from sndsur.models.surrogate import SupplyGraphSurrogate, build_surrogate
from sndsur.utils.logging import get_logger
from sndsur.utils.seed import seed_everything

LOG = get_logger()


@dataclass
class Predictions:
    """Per-row predictions and targets for one split.

    Every array is aligned and length ``R`` (or ``(R, P)`` for trajectories), so a
    metric function never has to re-derive the pairing. ``scenario_id`` and
    ``net_id`` let a caller group rows by scenario or by network, which the
    method-level statistics need.
    """

    impact: np.ndarray
    impact_true: np.ndarray
    sigma: np.ndarray
    quantiles: np.ndarray
    traj: np.ndarray
    traj_true: np.ndarray
    tti: np.ndarray
    tti_true: np.ndarray
    recovery: np.ndarray
    recovery_true: np.ndarray
    scenario_id: np.ndarray
    net_id: np.ndarray
    demand_node: np.ndarray

    def __len__(self) -> int:
        return int(self.impact.shape[0])


def _lr_at(epoch: int, cfg: Config) -> float:
    """Cosine schedule with a linear warm-up, in units of the base LR."""
    tc = cfg.train
    if tc.warmup_epochs > 0 and epoch < tc.warmup_epochs:
        return (epoch + 1) / tc.warmup_epochs
    if tc.scheduler != "cosine":
        return 1.0
    span = max(1, tc.epochs - tc.warmup_epochs)
    p = (epoch - tc.warmup_epochs) / span
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def predict(
    model: SupplyGraphSurrogate,
    batches: list[Batch],
    ds: ScenarioDataset,
    device: str = "cpu",
) -> Predictions:
    """Run the model over pre-collated batches and gather aligned arrays."""
    model.eval()
    acc: dict[str, list[np.ndarray]] = {k: [] for k in (
        "impact", "impact_true", "sigma", "quantiles", "traj", "traj_true",
        "tti", "tti_true", "recovery", "recovery_true", "scenario_id", "net_id",
        "demand_node",
    )}
    net_of_scenario = {
        s.scenario_id: s.net_id for scs in ds.splits.values() for s in scs
    }
    node_of = {s.scenario_id: s.demand_rows for scs in ds.splits.values() for s in scs}

    for b in batches:
        b = b.to(device)
        out = model(b.node_feat, b.edge_index, b.edge_feat, b.demand_rows)
        acc["impact"].append(out.impact.cpu().numpy())
        acc["impact_true"].append(b.impact.cpu().numpy())
        if out.logvar is not None:
            acc["sigma"].append(
                torch.exp(0.5 * out.logvar.clamp(-8.0, 4.0)).cpu().numpy()
            )
        else:
            acc["sigma"].append(np.full(b.impact.shape[0], np.nan, dtype=np.float32))
        if out.quantiles is not None:
            acc["quantiles"].append(out.quantiles.cpu().numpy())
        else:
            acc["quantiles"].append(np.full((b.impact.shape[0], 0), np.nan, dtype=np.float32))
        acc["traj"].append(out.traj.cpu().numpy())
        acc["traj_true"].append(b.traj.cpu().numpy())
        acc["tti"].append(out.time_to_impact.cpu().numpy())
        acc["tti_true"].append(b.tti.cpu().numpy())
        acc["recovery"].append(out.recovery_time.cpu().numpy())
        acc["recovery_true"].append(b.recovery.cpu().numpy())
        gor = b.graph_of_row.cpu().numpy()
        sids = b.scenario_ids.cpu().numpy()[gor]
        acc["scenario_id"].append(sids)
        acc["net_id"].append(np.array([net_of_scenario[int(s)] for s in sids]))
        acc["demand_node"].append(
            np.concatenate([node_of[int(s)] for s in b.scenario_ids.cpu().numpy()])
        )

    def cat(key: str) -> np.ndarray:
        return np.concatenate(acc[key], axis=0)

    return Predictions(**{k: cat(k) for k in acc})


def train_surrogate(
    cfg: Config,
    ds: ScenarioDataset,
    node_dim: int,
    edge_dim: int,
    seed: int | None = None,
    log_every: int = 5,
    run=None,
) -> tuple[SupplyGraphSurrogate, dict]:
    """Train one surrogate and return it with its training record.

    Args:
        cfg: Full configuration.
        ds: The scenario dataset, already normalised.
        node_dim: Node feature width.
        edge_dim: Edge feature width.
        seed: Overrides ``cfg.run.seed``, so ensemble members and the seed study
            differ only in this number.
        log_every: Epoch interval for the console line.
        run: Optional :class:`sndsur.utils.logging.RunDir` for ``history.jsonl``.

    Returns:
        ``(model, record)`` where ``record`` holds the best epoch, the best
        validation loss, wall-clock seconds and the parameter count.
    """
    s = cfg.run.seed if seed is None else seed
    seed_everything(s, cfg.run.deterministic)
    device = cfg.run.device
    model = build_surrogate(cfg, node_dim, edge_dim, ds.traj_periods).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)

    train_b = iterate_batches(ds.splits["train"], ds, cfg.train.batch_graphs, True, s)
    val_b = iterate_batches(ds.splits["val"], ds, cfg.train.batch_graphs, False)

    best = math.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = -1
    t0 = time.perf_counter()

    for epoch in range(cfg.train.epochs):
        model.train()
        scale = _lr_at(epoch, cfg)
        for g in opt.param_groups:
            g["lr"] = cfg.train.lr * scale
        # Re-shuffling the batch *order* each epoch, rather than re-collating,
        # keeps the epoch cost dominated by the forward pass. The graph
        # composition of each batch is fixed, which is a mild loss of stochastic
        # regularisation and a large saving in collation time.
        order = np.random.default_rng(s + epoch).permutation(len(train_b))
        agg: dict[str, float] = {}
        for i in order:
            b = train_b[i].to(device)
            out = model(b.node_feat, b.edge_index, b.edge_feat, b.demand_rows)
            loss, parts = surrogate_loss(out, b, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v)
        n_steps = max(1, len(order))
        record = {k: v / n_steps for k, v in agg.items()}

        val = _val_loss(model, val_b, cfg, device)
        record.update({"epoch": epoch, "lr": cfg.train.lr * scale, "val_loss": val})
        if run is not None:
            run.log_epoch(record)
        if val < best - 1e-9:
            best = val
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if log_every and (epoch % log_every == 0 or epoch == cfg.train.epochs - 1):
            LOG.info(
                "epoch %3d loss %.5f val %.5f (best %.5f @ %d)",
                epoch, record.get("loss", float("nan")), val, best, best_epoch,
            )
        if cfg.train.patience and epoch - best_epoch >= cfg.train.patience:
            LOG.info("early stop at epoch %d (no val improvement for %d)",
                     epoch, cfg.train.patience)
            break

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_loss": best,
        "train_seconds": time.perf_counter() - t0,
        "params": model.n_params(),
        "epochs_run": epoch + 1,
        "seed": s,
    }


@torch.no_grad()
def _val_loss(model, batches: list[Batch], cfg: Config, device: str) -> float:
    """Mean total loss over the validation batches."""
    model.eval()
    tot, n = 0.0, 0
    for b in batches:
        b = b.to(device)
        out = model(b.node_feat, b.edge_index, b.edge_feat, b.demand_rows)
        _, parts = surrogate_loss(out, b, cfg)
        tot += parts["loss"]
        n += 1
    return tot / max(n, 1)


def train_ensemble(
    cfg: Config, ds: ScenarioDataset, node_dim: int, edge_dim: int, run=None
) -> tuple[list[SupplyGraphSurrogate], list[dict]]:
    """Train ``cfg.model.ensemble`` independently initialised members.

    Deep ensembles (Lakshminarayanan et al., 2017) differ only in initialisation
    and batch order here — no bagging. Bagging would also shrink each member's
    training set, confounding "ensembling helps" with "less data hurts", and at
    this scale the initialisation seed already produces ample diversity.
    """
    models, records = [], []
    for m in range(max(1, cfg.model.ensemble)):
        LOG.info("ensemble member %d/%d", m + 1, cfg.model.ensemble)
        model, rec = train_surrogate(
            cfg, ds, node_dim, edge_dim, seed=cfg.run.seed + 1000 * m, run=run
        )
        rec["member"] = m
        models.append(model)
        records.append(rec)
    return models, records


@torch.no_grad()
def predict_ensemble(
    models: list[SupplyGraphSurrogate],
    batches: list[Batch],
    ds: ScenarioDataset,
    device: str = "cpu",
) -> Predictions:
    """Ensemble mean prediction with the total predictive standard deviation.

    The reported ``sigma`` combines both sources the way the deep-ensembles paper
    does: the mean of each member's predicted variance (aleatoric) plus the
    variance of their means (epistemic). Reporting only the second would
    understate uncertainty on easy-but-noisy rows; reporting only the first would
    miss the shifted-topology disagreement, which is the whole point.
    """
    per = [predict(m, batches, ds, device) for m in models]
    mu = np.stack([p.impact for p in per], axis=0)
    var_a = np.stack([np.where(np.isfinite(p.sigma), p.sigma, 0.0) ** 2 for p in per], axis=0)
    mean = mu.mean(axis=0)
    total_var = var_a.mean(axis=0) + mu.var(axis=0)
    base = per[0]
    q = (
        np.stack([p.quantiles for p in per], axis=0).mean(axis=0)
        if base.quantiles.size
        else base.quantiles
    )
    return Predictions(
        impact=mean,
        impact_true=base.impact_true,
        sigma=np.sqrt(total_var),
        quantiles=q,
        traj=np.stack([p.traj for p in per], axis=0).mean(axis=0),
        traj_true=base.traj_true,
        tti=np.stack([p.tti for p in per], axis=0).mean(axis=0),
        tti_true=base.tti_true,
        recovery=np.stack([p.recovery for p in per], axis=0).mean(axis=0),
        recovery_true=base.recovery_true,
        scenario_id=base.scenario_id,
        net_id=base.net_id,
        demand_node=base.demand_node,
    )


def ensemble_component_sigmas(
    models: list[SupplyGraphSurrogate],
    batches: list[Batch],
    ds: ScenarioDataset,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Aleatoric and epistemic standard deviations, separately.

    Returned as a pair so the calibration table can show which component tracks
    the distribution shift. If the aleatoric term rose on shifted topologies but
    the epistemic term did not, the ensemble would be adding nothing and should
    not be claimed to.
    """
    per = [predict(m, batches, ds, device) for m in models]
    mu = np.stack([p.impact for p in per], axis=0)
    var_a = np.stack([np.where(np.isfinite(p.sigma), p.sigma, 0.0) ** 2 for p in per], axis=0)
    return np.sqrt(var_a.mean(axis=0)), np.sqrt(mu.var(axis=0))


def as_tensor(x: np.ndarray, device: str = "cpu") -> Tensor:
    """NumPy to float32 torch tensor on ``device``."""
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device)
