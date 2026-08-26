"""Figures, drawn from the committed CSVs rather than from live objects.

Every figure reads ``results/tables/*.csv``. That is deliberate: it means a
figure cannot disagree with a table, and a reader can regenerate every plot
without re-running an experiment.

The matplotlib backend is **not** forced at import time. Setting ``Agg``
unconditionally makes every notebook plot render blank, which is a silent
failure; the backend is only switched when this module is imported outside a
kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if "ipykernel" not in sys.modules and "IPython" not in sys.modules:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

TABLES = Path("results/tables")
FIGURES = Path("results/figures")

#: One colour per method, held fixed across every figure so a reader does not
#: have to re-read the legend.
COLORS = {
    "surrogate": "#1b4f9c",
    "surrogate_ensemble": "#1b4f9c",
    "surrogate_single": "#5b8dd6",
    "mlp_no_message_passing": "#e07b39",
    "tabular_gbt": "#b02a2a",
    "tabular_ridge": "#d4757a",
    "retrieval_knn": "#7a5c99",
    "topology_heuristic": "#3f8f52",
    "simulator": "#222222",
}
SHIFTS = ("test_id", "shift_topo", "shift_size", "shift_type", "shift_multi")


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    p = FIGURES / name
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _read(name: str) -> pd.DataFrame | None:
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None


def figure_generalisation() -> Path | None:
    """MAE and rank correlation per method across the distribution-shift splits.

    Two panels rather than one number, because the whole argument of this
    repository is that the two can move in opposite directions.
    """
    df = _read("method_comparison.csv")
    if df is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    methods = [m for m in COLORS if m in set(df["method"])]
    x = np.arange(len(SHIFTS))
    w = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        sub = df[df["method"] == m].set_index("split")
        mae = [sub.loc[s, "mae"] if s in sub.index else np.nan for s in SHIFTS]
        sp = [
            sub.loc[s, "spearman_within_scenario"] if s in sub.index else np.nan
            for s in SHIFTS
        ]
        axes[0].bar(x + i * w, mae, w, label=m, color=COLORS[m])
        axes[1].bar(x + i * w, sp, w, label=m, color=COLORS[m])
    for ax, t, yl in (
        (axes[0], "Magnitude error (lower is better)", "MAE of service-level loss"),
        (axes[1], "Rank fidelity (higher is better)", "within-scenario Spearman"),
    ):
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(SHIFTS, rotation=20, ha="right")
        _style(ax, t, "", yl)
    axes[1].legend(fontsize=7, ncol=2, frameon=False)
    return _save(fig, "generalisation.png")


def figure_budget_curve() -> Path | None:
    """Recall@10 of the true critical set versus compute budget."""
    df = _read("budget_curve.csv")
    if df is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, grp in df.groupby("method"):
        agg = grp.groupby("budget_s")["recall_at_k"].agg(["mean", "std"]).reset_index()
        ax.plot(
            agg["budget_s"], agg["mean"], "o-", label=str(method),
            color=COLORS.get(str(method), "#666"), linewidth=1.8, markersize=4,
        )
        ax.fill_between(
            agg["budget_s"],
            agg["mean"] - agg["std"].fillna(0),
            agg["mean"] + agg["std"].fillna(0),
            alpha=0.15,
            color=COLORS.get(str(method), "#666"),
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    _style(
        ax,
        "Decision quality at a fixed compute budget",
        "compute budget (s, log scale)",
        "recall@10 of the true critical set",
    )
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    return _save(fig, "budget_curve.png")


def figure_calibration() -> Path | None:
    """Reliability of the predictive intervals, in-distribution and shifted."""
    df = _read("calibration.csv")
    if df is None or "nominal" not in df.columns:
        return None
    gauss = df[df["interval"].isin(["gaussian", "quantile_central"])].dropna(subset=["nominal"])
    if gauss.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
    for split in SHIFTS:
        sub = gauss[(gauss["split"] == split) & (gauss["interval"] == "gaussian")]
        if sub.empty:
            continue
        sub = sub.sort_values("nominal")
        axes[0].plot(sub["nominal"], sub["coverage"], "o-", label=split, markersize=4)
    _style(axes[0], "Interval coverage", "nominal coverage", "empirical coverage")
    axes[0].legend(fontsize=7, frameon=False)

    shift = df[df["interval"] == "uncertainty_shift"]
    if not shift.empty:
        sub = shift.set_index("split")
        order = [s for s in SHIFTS if s in sub.index]
        ax = axes[1]
        ax.bar(
            np.arange(len(order)) - 0.2,
            [sub.loc[s, "mean_sigma"] for s in order],
            0.4, label="mean predicted sigma", color="#1b4f9c",
        )
        ax.bar(
            np.arange(len(order)) + 0.2,
            [sub.loc[s, "mean_abs_error"] for s in order],
            0.4, label="mean abs error", color="#b02a2a",
        )
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=20, ha="right")
        _style(ax, "Does uncertainty track error under shift?", "", "service-level units")
        ax.legend(fontsize=7, frameon=False)
    return _save(fig, "calibration.png")


def figure_criticality_scatter() -> Path | None:
    """Predicted versus true criticality, and where the reference score disagrees.

    The right panel is the headline: nodes are placed by their reference-style
    feature score against their true counterfactual impact. Points in the lower
    right are nodes the feature score flags that do not matter; points in the
    upper left are the single points of failure it misses.
    """
    df = _read("criticality_detail.csv")
    if df is None or df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, col, title in (
        (axes[0], "surrogate", "Surrogate vs simulator truth"),
        (axes[1], "tabular_gbt", "Reference-style feature score vs truth"),
    ):
        sole = df["sole_source_reach"] > 0
        ax.scatter(
            df.loc[~sole, col], df.loc[~sole, "truth"], s=14, alpha=0.55,
            color="#7590b5", label="multi-sourced downstream",
        )
        ax.scatter(
            df.loc[sole, col], df.loc[sole, "truth"], s=26, alpha=0.85,
            color="#b02a2a", marker="^", label="sole-source reach > 0",
        )
        _style(ax, title, f"{col} score", "true total service loss")
    axes[0].legend(fontsize=7, frameon=False)
    return _save(fig, "criticality_scatter.png")


def figure_seed_noise() -> Path | None:
    """Per-metric seed spread against the gains claimed elsewhere."""
    df = _read("seed_variance.csv")
    if df is None or df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = df[df["split"] == "test_id"] if "split" in df.columns else df
    if sub.empty:
        sub = df
    x = np.arange(len(sub))
    ax.bar(x, sub["diff_noise_scale"], 0.55, color="#888", label="run-to-run noise scale")
    ax.errorbar(
        x, sub["mean"], yerr=sub["sd"], fmt="o", color="#1b4f9c", markersize=4,
        capsize=3, label="metric mean +- sd",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sub["metric"], rotation=25, ha="right")
    _style(ax, "Seed variance: what a difference has to beat", "", "value")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, "seed_noise.png")


def figure_trajectory_examples() -> Path | None:
    """Predicted versus simulated excess-unmet trajectories for a few rows."""
    p = Path("results/runs")
    files = sorted(p.glob("*/per_item.csv"))
    if not files:
        return None
    df = pd.read_csv(files[0])
    sub = df[df["split"] == "shift_topo"] if "split" in df.columns else df
    if sub.empty:
        return None
    top = sub.nlargest(6, "truth")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(top["pred_surrogate_ensemble"], top["truth"], s=40, color="#1b4f9c")
    lims = [
        0,
        float(max(top["truth"].max(), top["pred_surrogate_ensemble"].max())) * 1.1 + 1e-6,
    ]
    ax.plot(lims, lims, "k--", linewidth=0.8)
    _style(
        ax,
        "Six largest true impacts on unseen topologies",
        "predicted service-level loss",
        "true service-level loss",
    )
    return _save(fig, "worst_case_agreement.png")


def figure_ablations() -> Path | None:
    """Ablation MAE per split, one bar group per variant."""
    df = _read("ablations.csv")
    if df is None or df.empty:
        return None
    variants = list(dict.fromkeys(df["variant"]))
    splits = [s for s in SHIFTS if s in set(df["split"])]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(variants))
    w = 0.8 / max(len(splits), 1)
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(splits):
        sub = df[df["split"] == s].set_index("variant")
        vals = [sub.loc[v, "mae"] if v in sub.index else np.nan for v in variants]
        ax.bar(x + i * w, vals, w, label=s, color=cmap(i / max(len(splits) - 1, 1)))
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(variants, rotation=25, ha="right")
    _style(ax, "Ablations: one mechanism removed at a time", "", "MAE")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    return _save(fig, "ablations.png")


def figure_efficiency() -> Path | None:
    """Measured wall-clock per scenario, simulator against surrogate."""
    df = _read("efficiency.csv")
    if df is None or df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4))
    comps = list(dict.fromkeys(df["component"]))
    x = np.arange(len(comps))
    med = [df[df["component"] == c]["per_item_ms"].median() for c in comps]
    iqr = [df[df["component"] == c]["iqr_ms"].median() for c in comps]
    ax.bar(x, med, 0.55, yerr=iqr, capsize=4,
           color=["#222222" if "sim" in c else "#1b4f9c" for c in comps])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(comps, rotation=20, ha="right")
    _style(
        ax,
        "Wall-clock per scenario (median, IQR bars, log scale)",
        "",
        "ms per scenario",
    )
    for xi, v in zip(x, med, strict=True):
        if np.isfinite(v):
            ax.text(xi, v * 1.15, f"{v:.3g}", ha="center", fontsize=7)
    return _save(fig, "efficiency.png")


def figure_network_example() -> Path | None:
    """A generated network drawn by tier, with sole-source edges highlighted.

    Regenerated from the config rather than read from a table, since a picture of
    the data generator is the one figure that cannot come from a CSV.
    """
    from sndsur.data.network import N_TIERS, NetworkSpec, generate_network

    net = generate_network(NetworkSpec(), seed=0, name="example")
    net.compute_throughput()
    fig, ax = plt.subplots(figsize=(9, 5))
    pos = {}
    for t in range(N_TIERS):
        nodes = np.flatnonzero(net.tier == t)
        for i, v in enumerate(nodes):
            pos[int(v)] = (t, i - len(nodes) / 2.0)
    for u, v in net.edges():
        gi = net.group_of(u, v)
        sole = net.groups[v][gi].sole_source
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot(
            [x0, x1], [y0, y1],
            color="#b02a2a" if sole else "#c9d2de",
            linewidth=1.4 if sole else 0.6,
            alpha=0.9 if sole else 0.5,
            zorder=1,
        )
    thr = net.throughput
    sizes = 20 + 180 * thr / max(thr.max(), 1e-9)
    for t in range(N_TIERS):
        nodes = np.flatnonzero(net.tier == t)
        ax.scatter(
            [pos[int(v)][0] for v in nodes],
            [pos[int(v)][1] for v in nodes],
            s=sizes[nodes], zorder=2,
            color=plt.get_cmap("viridis")(t / (N_TIERS - 1)),
            edgecolors="white", linewidths=0.6,
        )
    ax.set_xticks(range(N_TIERS))
    ax.set_xticklabels(["raw", "component", "assembly", "distribution", "demand"])
    ax.set_yticks([])
    _style(
        ax,
        "Generated network: node size = throughput, red = sole-source edge",
        "echelon",
        "",
    )
    return _save(fig, "network_example.png")


def make_all_figures() -> list[Path]:
    """Draw every figure whose input table exists. Returns the written paths."""
    out = []
    for fn in (
        figure_network_example,
        figure_generalisation,
        figure_budget_curve,
        figure_calibration,
        figure_criticality_scatter,
        figure_seed_noise,
        figure_ablations,
        figure_efficiency,
        figure_trajectory_examples,
    ):
        p = fn()
        if p is not None:
            out.append(p)
    return out
