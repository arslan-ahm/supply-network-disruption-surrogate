"""Scatter-based message passing in plain PyTorch.

**Why not torch-geometric or DGL.** Both are excellent libraries and both are
install-fragile: they compile against a specific torch/CUDA pair, and a reader
who runs ``uv sync`` on a different torch build gets a linker error rather than a
result. The message passing this project needs is a gather, an MLP, and an
``index_add_`` — about forty lines. Trading forty lines of code for a
reproducibility guarantee is a good trade for a repository whose point is that
its numbers can be re-derived, so it is made deliberately and stated in the docs.

**Batching.** A batch of graphs is one big disconnected graph: node features are
concatenated and edge indices are offset by the running node count. No padding,
no masking, and the per-node compute is exactly proportional to the real work.

**Directionality.** Supply networks are directed and the two directions carry
different physics. Material and therefore *shortage* flows supplier to customer;
orders and therefore *requirement* flow customer to supplier. A single symmetric
message function would have to encode both in one set of weights, so by default
this module runs two: a downstream pass and an upstream pass, whose outputs are
concatenated. ``bidirectional=False`` is the ablation that removes the upstream
pass and is expected to hurt demand-spike scenarios specifically, since a demand
spike is an intervention that only travels upstream.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def scatter_mean(src: Tensor, index: Tensor, n: int) -> Tensor:
    """Mean of ``src`` rows grouped by ``index`` into ``n`` output rows.

    Args:
        src: ``(E, F)`` messages.
        index: ``(E,)`` destination row per message.
        n: Number of output rows.

    Returns:
        ``(n, F)``. Rows with no incoming message are exactly zero, not NaN — a
        tier-0 supplier has no inputs and must not poison the representation
        with a 0/0.
    """
    out = src.new_zeros((n, src.shape[1]))
    if src.numel() == 0:
        return out
    out.index_add_(0, index, src)
    count = src.new_zeros(n)
    count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / count.clamp(min=1.0).unsqueeze(1)


def scatter_max(src: Tensor, index: Tensor, n: int) -> Tensor:
    """Elementwise maximum of ``src`` rows grouped by ``index``.

    Included alongside the mean because a shortage is a *bottleneck*: the binding
    constraint on a node's production is the worst of its inputs, not their
    average. Aggregating with both and letting the update MLP combine them lets
    the model express a min/max-like rule, which mean pooling alone cannot
    (Gilmer et al., 2017 discuss exactly this aggregator-expressivity trade).

    Empty groups return 0 rather than ``-inf``, for the same reason as above.
    """
    out = src.new_zeros((n, src.shape[1]))
    if src.numel() == 0:
        return out
    out = out.fill_(float("-inf"))
    out = out.index_reduce_(0, index, src, "amax", include_self=True)
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


def mlp(sizes: list[int], dropout: float = 0.0, out_act: bool = False) -> nn.Sequential:
    """A SiLU MLP with LayerNorm on the hidden layers.

    LayerNorm rather than BatchNorm: a batch here is a variable-size set of
    graphs whose node count changes every step, and batch statistics computed
    over "all nodes in this batch" mix tier-0 suppliers with demand points, whose
    feature distributions are nothing alike. LayerNorm normalises per node and is
    invariant to how the graphs were batched — which also keeps single-graph
    inference identical to batched inference, a property the latency benchmark
    and the counterfactual screening both rely on.
    """
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        last = i == len(sizes) - 2
        if not last or out_act:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MessagePassingLayer(nn.Module):
    """One round of directed message passing with a residual node update.

    The message function sees ``[h_src, h_dst, edge_features]`` — the full
    relational triple of Gilmer et al. (2017) rather than the ``h_src``-only form
    of a plain GCN (Kipf & Welling, 2017). Edge features matter here because two
    otherwise identical suppliers differ by lead time, BOM coefficient and
    sourcing share, and those are properties of the *relationship*, not of either
    node.

    Args:
        hidden: Node embedding width.
        edge_dim: Edge feature width. 0 disables edge conditioning, which is the
            edge-feature ablation.
        dropout: Dropout inside the message and update MLPs.
        bidirectional: Run an upstream pass in addition to the downstream one.
    """

    def __init__(
        self,
        hidden: int,
        edge_dim: int,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.edge_dim = edge_dim
        msg_in = 2 * hidden + edge_dim
        self.msg_down = mlp([msg_in, hidden, hidden], dropout, out_act=True)
        self.msg_up = (
            mlp([msg_in, hidden, hidden], dropout, out_act=True) if bidirectional else None
        )
        # Each direction contributes a mean and a max, hence 2 aggregates per
        # direction plus the node's own state.
        n_agg = 2 * (2 if bidirectional else 1)
        self.update = mlp([hidden * (1 + n_agg), hidden, hidden], dropout, out_act=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor | None) -> Tensor:
        """Args:
            h: ``(N, hidden)`` node states.
            edge_index: ``(2, E)`` with row 0 the supplier and row 1 the customer.
            edge_attr: ``(E, edge_dim)`` or None.

        Returns:
            ``(N, hidden)`` updated node states.
        """
        n = h.shape[0]
        src, dst = edge_index[0], edge_index[1]
        parts = [h]

        def pack(a: Tensor, b: Tensor) -> Tensor:
            xs = [a, b]
            if self.edge_dim and edge_attr is not None:
                xs.append(edge_attr)
            return torch.cat(xs, dim=1)

        m_down = self.msg_down(pack(h[src], h[dst]))
        parts += [scatter_mean(m_down, dst, n), scatter_max(m_down, dst, n)]

        if self.msg_up is not None:
            m_up = self.msg_up(pack(h[dst], h[src]))
            parts += [scatter_mean(m_up, src, n), scatter_max(m_up, src, n)]

        # Residual: with three or four rounds the un-residualised stack lost the
        # node's own disruption flags by the final layer, and the model predicted
        # a network-average impact for every demand point.
        return self.norm(h + self.update(torch.cat(parts, dim=1)))
