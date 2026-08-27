"""The temporal graph surrogate and its prediction heads.

The model maps ``(network state, disruption specification)`` to a counterfactual:
for every demand point, how much service is lost, when the shortage starts, how
long recovery takes, and the shape of the shortage over time. It is a learned
simulator in the sense of Sanchez-Gonzalez et al. (2020) and Pfaff et al. (2021)
— an encoder-processor-decoder over a graph, trained on trajectories from a
mechanistic model — applied to a supply network rather than to particles or a
mesh, following Kosasih & Brintrup (2021) in treating the supply network itself
as the graph.

Four heads, and each exists for a stated reason:

``impact``
    Scalar service-level loss per demand point. The headline target and the one
    the criticality ranking is built on.
``logvar``
    Per-demand-point predictive log-variance, trained by Gaussian NLL. This is
    what lets the surrogate say "I do not know" on a shifted topology, and the
    claim it must support is that error grows with predicted variance.
``traj``
    The excess-unmet trajectory. Auxiliary: forcing the representation to
    reproduce *when* the shortage happens is a much stronger constraint than the
    scalar alone, and the same embedding then answers time-to-impact.
``timing``
    Time-to-impact and recovery time, in periods. Both are undefined for an
    unaffected demand point, so both are trained only on rows where the
    simulator defined them and are reported with their contributing counts.

The trajectory decoder is a GRU rolled over periods rather than a single linear
projection. A linear head predicts 24 numbers independently and has no way to
express "the shortage starts, persists, then decays", which is the only shape the
simulator ever produces. ``temporal_decoder=False`` swaps in the linear head and
is the temporal-modelling ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sndsur.models.layers import MessagePassingLayer, mlp


@dataclass
class SurrogateOutput:
    """Model predictions for one batch.

    All tensors are indexed by *demand row*: the concatenation over graphs of
    each graph's demand points, in ascending node id.

    Attributes:
        impact: ``(R,)`` predicted service-level loss.
        logvar: ``(R,)`` predicted log-variance, or None when the head is off.
        traj: ``(R, P)`` predicted excess-unmet trajectory.
        time_to_impact: ``(R,)`` predicted periods to first shortage.
        recovery_time: ``(R,)`` predicted periods to recovery.
        quantiles: ``(R, Q)`` predicted quantiles of impact, or None.
        node_embedding: ``(N, H)`` final node states, for analysis and for the
            retrieval baseline's feature space.
    """

    impact: Tensor
    logvar: Tensor | None
    traj: Tensor
    time_to_impact: Tensor
    recovery_time: Tensor
    quantiles: Tensor | None
    node_embedding: Tensor


class SupplyGraphSurrogate(nn.Module):
    """Encoder-processor-decoder graph surrogate for disruption impact.

    Args:
        node_dim: Node feature width.
        edge_dim: Edge feature width. Ignored when ``use_edge_features`` is False.
        traj_periods: Number of trajectory periods to predict.
        hidden: Embedding width.
        layers: Message-passing rounds. ``0`` removes the processor entirely,
            which is the graph ablation: the model becomes a per-node MLP with
            exactly the same inputs and roughly the same parameter count.
        dropout: Dropout in every MLP.
        bidirectional: Two directed message functions rather than one.
        use_edge_features: Condition messages on edge features.
        temporal_decoder: GRU trajectory decoder rather than a linear projection.
        heteroscedastic: Predict a log-variance for ``impact``.
        n_quantiles: Number of quantile outputs; 0 disables the head.
        no_graph_blocks: Node-only trunk blocks used when ``layers == 0``.
        no_graph_width: Hidden width of those blocks. Together these two are set
            so the no-graph ablation matches the graph model's parameter count.

    The impact head ends in ``softplus`` because service-level loss is
    non-negative by construction. A linear head spent its early epochs learning
    that floor and predicted small negative losses on easy scenarios, which then
    corrupted the criticality ranking at the bottom of the list — where most of
    the nodes are.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        traj_periods: int,
        hidden: int = 64,
        layers: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = True,
        use_edge_features: bool = True,
        temporal_decoder: bool = True,
        heteroscedastic: bool = True,
        n_quantiles: int = 3,
        no_graph_blocks: int = 4,
        no_graph_width: int = 160,
    ) -> None:
        super().__init__()
        self.traj_periods = traj_periods
        self.hidden = hidden
        self.n_layers = layers
        self.use_edge_features = use_edge_features and edge_dim > 0
        self.temporal_decoder = temporal_decoder
        self.heteroscedastic = heteroscedastic
        self.n_quantiles = n_quantiles

        self.encoder = mlp([node_dim, hidden, hidden], dropout, out_act=True)
        self.edge_encoder = (
            mlp([edge_dim, hidden // 2, hidden // 2], dropout, out_act=True)
            if self.use_edge_features
            else None
        )
        edge_hidden = hidden // 2 if self.use_edge_features else 0
        self.layers = nn.ModuleList(
            MessagePassingLayer(hidden, edge_hidden, dropout, bidirectional)
            for _ in range(layers)
        )
        if layers == 0:
            # Message passing is removed, but the parameters are not: the trunk
            # is a stack of node-only blocks widened so the no-graph model has a
            # comparable budget. The first version of this ablation simply
            # dropped the processor and came out 3.6x smaller, which would have
            # confounded "the graph helps" with "more parameters help". The
            # committed widths land the two within a few per cent; both counts
            # are printed in results/tables/ablations.csv so the reader can check.
            blocks = []
            for _ in range(max(1, no_graph_blocks)):
                blocks.append(mlp([hidden, no_graph_width, hidden], dropout, out_act=True))
            self.no_graph_trunk = nn.Sequential(*blocks)
        else:
            self.no_graph_trunk = None

        self.impact_head = nn.Sequential(mlp([hidden, hidden], dropout, out_act=True),
                                         nn.Linear(hidden, 1), nn.Softplus())
        self.logvar_head = (
            nn.Sequential(mlp([hidden, hidden], dropout, out_act=True), nn.Linear(hidden, 1))
            if heteroscedastic
            else None
        )
        self.quantile_head = (
            nn.Sequential(mlp([hidden, hidden], dropout, out_act=True),
                          nn.Linear(hidden, n_quantiles))
            if n_quantiles > 0
            else None
        )
        # Timing outputs are periods, so softplus keeps them non-negative; a
        # negative time-to-impact is not a thing.
        self.timing_head = nn.Sequential(
            mlp([hidden, hidden], dropout, out_act=True), nn.Linear(hidden, 2), nn.Softplus()
        )
        if temporal_decoder:
            self.gru = nn.GRUCell(1 + hidden, hidden)
            self.traj_out = nn.Linear(hidden, 1)
            self.traj_linear = None
        else:
            self.gru = None
            self.traj_out = None
            self.traj_linear = nn.Sequential(
                mlp([hidden, hidden], dropout, out_act=True), nn.Linear(hidden, traj_periods)
            )

    # ------------------------------------------------------------------ #

    def encode(
        self, node_feat: Tensor, edge_index: Tensor, edge_feat: Tensor | None
    ) -> Tensor:
        """Node features to final node embeddings.

        Returns:
            ``(N, hidden)``.
        """
        h = self.encoder(node_feat)
        if self.n_layers == 0:
            return self.no_graph_trunk(h)  # type: ignore[misc]
        ea = (
            self.edge_encoder(edge_feat)
            if (self.edge_encoder is not None and edge_feat is not None)
            else None
        )
        for layer in self.layers:
            h = layer(h, edge_index, ea)
        return h

    def decode_traj(self, z: Tensor) -> Tensor:
        """Trajectory from demand-point embeddings.

        Returns:
            ``(R, traj_periods)``, non-negative.
        """
        if self.traj_linear is not None:
            return torch.nn.functional.softplus(self.traj_linear(z))
        r = z.shape[0]
        state = z
        prev = z.new_zeros((r, 1))
        outs = []
        for _ in range(self.traj_periods):
            state = self.gru(torch.cat([prev, z], dim=1), state)  # type: ignore[misc]
            step = torch.nn.functional.softplus(self.traj_out(state))  # type: ignore[misc]
            outs.append(step)
            # Teacher forcing is deliberately not used: at screening time there
            # is no ground truth to feed, so training the decoder on its own
            # output keeps train and inference behaviour identical.
            prev = step
        return torch.cat(outs, dim=1)

    def forward(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        edge_feat: Tensor | None,
        demand_rows: Tensor,
    ) -> SurrogateOutput:
        """Args:
            node_feat: ``(N, node_dim)``.
            edge_index: ``(2, E)``, supplier row then customer row.
            edge_feat: ``(E, edge_dim)`` or None.
            demand_rows: ``(R,)`` node indices of the demand points, in the
                batch's global numbering.

        Returns:
            A :class:`SurrogateOutput`.
        """
        h = self.encode(node_feat, edge_index, edge_feat)
        z = h[demand_rows]
        timing = self.timing_head(z)
        return SurrogateOutput(
            impact=self.impact_head(z).squeeze(-1),
            logvar=self.logvar_head(z).squeeze(-1) if self.logvar_head is not None else None,
            traj=self.decode_traj(z),
            time_to_impact=timing[:, 0],
            recovery_time=timing[:, 1],
            quantiles=self.quantile_head(z) if self.quantile_head is not None else None,
            node_embedding=h,
        )

    def n_params(self) -> int:
        """Trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_surrogate(cfg, node_dim: int, edge_dim: int, traj_periods: int) -> SupplyGraphSurrogate:
    """Construct a surrogate from a :class:`sndsur.config.Config`."""
    m = cfg.model
    return SupplyGraphSurrogate(
        node_dim=node_dim,
        edge_dim=edge_dim,
        traj_periods=traj_periods,
        hidden=m.hidden,
        layers=m.layers,
        dropout=m.dropout,
        bidirectional=m.bidirectional,
        use_edge_features=m.use_edge_features,
        temporal_decoder=m.temporal_decoder,
        heteroscedastic=m.heteroscedastic,
        n_quantiles=len(m.quantiles),
        no_graph_blocks=m.no_graph_blocks,
        no_graph_width=m.no_graph_width,
    )
