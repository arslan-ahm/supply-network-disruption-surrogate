"""Collating scenarios into one disconnected graph per batch.

A batch is built by concatenating node features and offsetting edge indices by
the running node count. Nothing is padded, so a batch of a 54-node network and a
110-node network costs exactly what those two graphs cost — which matters here
because the larger-network shift split has roughly twice the nodes and must be
evaluated with the same code path as training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from sndsur.data.scenarios import Scenario, ScenarioDataset


@dataclass
class Batch:
    """One collated batch of scenarios.

    Attributes:
        node_feat: ``(N, F)``.
        edge_index: ``(2, E)`` in the batch's global node numbering.
        edge_feat: ``(E, Fe)``.
        demand_rows: ``(R,)`` global node indices of demand points.
        graph_of_row: ``(R,)`` which graph in the batch each demand row belongs to.
        impact, unmet, tti, recovery: ``(R,)`` targets. ``tti`` and ``recovery``
            carry NaN where the simulator left them undefined, and the loss masks
            them rather than substituting a value — an unaffected demand point
            does not have a time-to-impact of zero, it has none.
        traj: ``(R, P)`` target trajectory.
        tabular: ``(R, Ft)`` reference-style features, carried so the tabular
            baseline trains on exactly the same rows in the same order.
        n_nodes: Total nodes.
        n_graphs: Scenarios in the batch.
        scenario_ids: ``(G,)`` ids, for joining predictions back to scenarios.
    """

    node_feat: Tensor
    edge_index: Tensor
    edge_feat: Tensor
    demand_rows: Tensor
    graph_of_row: Tensor
    impact: Tensor
    unmet: Tensor
    tti: Tensor
    recovery: Tensor
    traj: Tensor
    tabular: Tensor
    n_nodes: int
    n_graphs: int
    scenario_ids: Tensor

    def to(self, device: str) -> Batch:
        if device == "cpu":
            return self
        moved = {
            k: (v.to(device) if isinstance(v, Tensor) else v) for k, v in self.__dict__.items()
        }
        return Batch(**moved)


def collate(scenarios: list[Scenario], ds: ScenarioDataset) -> Batch:
    """Concatenate scenarios into one :class:`Batch`.

    Args:
        scenarios: The scenarios to batch.
        ds: The dataset, for the per-network edge index.

    Returns:
        A :class:`Batch` on CPU.
    """
    nf, ef, ei, dr, gor, tab = [], [], [], [], [], []
    imp, unm, tti, rec, tr, sids = [], [], [], [], [], []
    offset = 0
    for gi, s in enumerate(scenarios):
        n = s.node_feat.shape[0]
        nf.append(s.node_feat)
        ef.append(s.edge_feat)
        ei.append(ds.networks[s.net_id].edge_index + offset)
        dr.append(s.demand_rows + offset)
        gor.append(np.full(s.n_demand, gi, dtype=np.int64))
        imp.append(s.service_loss)
        unm.append(s.unmet_extra)
        tti.append(s.time_to_impact)
        rec.append(s.recovery_time)
        tr.append(s.traj)
        tab.append(s.tabular)
        sids.append(s.scenario_id)
        offset += n

    def cat(xs, dtype=torch.float32):
        return torch.as_tensor(np.concatenate(xs, axis=0), dtype=dtype)

    return Batch(
        node_feat=cat(nf),
        edge_index=torch.as_tensor(np.concatenate(ei, axis=1), dtype=torch.long),
        edge_feat=cat(ef),
        demand_rows=cat(dr, torch.long),
        graph_of_row=cat(gor, torch.long),
        impact=cat(imp),
        unmet=cat(unm),
        tti=cat(tti),
        recovery=cat(rec),
        traj=cat(tr),
        tabular=cat(tab),
        n_nodes=offset,
        n_graphs=len(scenarios),
        scenario_ids=torch.as_tensor(np.array(sids), dtype=torch.long),
    )


def iterate_batches(
    scenarios: list[Scenario],
    ds: ScenarioDataset,
    batch_graphs: int,
    shuffle: bool = False,
    seed: int = 0,
) -> list[Batch]:
    """Split ``scenarios`` into batches of at most ``batch_graphs`` graphs.

    Returns a materialised list rather than a generator: the whole dataset is a
    few hundred megabytes at most at this scale, batches are reused across
    epochs, and re-collating every epoch turned out to cost more than the
    forward pass.
    """
    idx = np.arange(len(scenarios))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    out = []
    for start in range(0, len(idx), batch_graphs):
        chunk = [scenarios[i] for i in idx[start : start + batch_graphs]]
        if chunk:
            out.append(collate(chunk, ds))
    return out


class FeatureNormaliser:
    """Per-column standardisation fitted on the training split only.

    Fitted on train and applied everywhere, which is the only version of this
    that does not leak. Columns with zero variance in training (a one-hot tier
    that never appears, say) are left alone rather than divided by epsilon, which
    would otherwise amplify float noise into a large input.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, blocks: list[np.ndarray]) -> FeatureNormaliser:
        x = np.concatenate(blocks, axis=0).astype(np.float64)
        self.mean = x.mean(axis=0)
        sd = x.std(axis=0)
        self.std = np.where(sd < 1e-8, 1.0, sd)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("normaliser used before fit")
        return ((x.astype(np.float64) - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> dict[str, list[float]]:
        return {
            "mean": [] if self.mean is None else self.mean.tolist(),
            "std": [] if self.std is None else self.std.tolist(),
        }

    def load_state_dict(self, d: dict[str, list[float]]) -> FeatureNormaliser:
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.std = np.asarray(d["std"], dtype=np.float64)
        return self


def fit_normalisers(
    ds: ScenarioDataset, split: str = "train"
) -> tuple[FeatureNormaliser, FeatureNormaliser, FeatureNormaliser]:
    """Fit node, edge and tabular normalisers on one split.

    Returns:
        ``(node, edge, tabular)`` normalisers.
    """
    scs = ds.splits[split]
    node = FeatureNormaliser().fit([s.node_feat for s in scs])
    edge = FeatureNormaliser().fit([s.edge_feat for s in scs])
    tab = FeatureNormaliser().fit([s.tabular for s in scs])
    return node, edge, tab


def apply_normalisers(
    ds: ScenarioDataset,
    node: FeatureNormaliser,
    edge: FeatureNormaliser,
    tab: FeatureNormaliser,
) -> ScenarioDataset:
    """Standardise every scenario's features in place and return the dataset."""
    for scs in ds.splits.values():
        for s in scs:
            s.node_feat = node.transform(s.node_feat)
            s.edge_feat = edge.transform(s.edge_feat)
            s.tabular = tab.transform(s.tabular)
    return ds
