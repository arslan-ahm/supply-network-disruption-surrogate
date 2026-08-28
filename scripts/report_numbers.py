"""Print the documentation's tables directly from the committed CSVs.

The reference project for this build standard shipped three wrong tables on its
first pass, all from transcribing numbers by hand. This script exists so that no
number in `README.md` or `docs/RESULTS.md` is ever typed: it renders them as
markdown from `results/tables/*.csv`, and the docs are updated by pasting its
output.

It also applies the noise-scale verdict to every claimed gain, so a table cannot
report an improvement without also reporting whether that improvement survives
the seed study.

Usage:
    python scripts/report_numbers.py [--section all|methods|shift|ablation|...]
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Windows consoles default to cp1252, which cannot encode the maths symbols used
# in the table headers. Reconfiguring rather than avoiding them keeps the rendered
# markdown readable; the fallback keeps this working on a stream that cannot be
# reconfigured (a pipe on some platforms).
with contextlib.suppress(AttributeError, ValueError):
    sys.stdout.reconfigure(encoding="utf-8")

TABLES = Path("results/tables")
SHIFTS = ("test_id", "shift_topo", "shift_size", "shift_type", "shift_multi")
#: Every split, in dataset order. The trivial-baseline table uses all seven,
#: because "the surrogate loses to the constant zero on 6 of 7 splits" is a claim
#: about all of them and quoting only the five evaluation splits would soften it.
ALL_SPLITS = ("train", "val", "test_id", "shift_topo", "shift_size", "shift_type",
              "shift_multi")
#: Print order for method tables.
METHOD_ORDER = ["surrogate_ensemble", "surrogate_single", "mlp_no_message_passing",
                "tabular_gbt", "tabular_gbt_l1", "tabular_ridge", "retrieval_knn",
                "topology_heuristic", "constant_zero", "constant_train_mean"]


def read(name: str) -> pd.DataFrame | None:
    p = TABLES / name
    if not p.exists():
        print(f"<!-- {name} missing -->")
        return None
    return pd.read_csv(p)


def fmt(x, nd: int = 4) -> str:
    """Format a cell, showing 'not measured' rather than inventing a value."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "not measured"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}"


def md_table(frame: pd.DataFrame, cols: list[str], headers: list[str] | None = None,
             nd: int = 4) -> str:
    head = headers or cols
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for _, r in frame.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(v if isinstance(v, str) else fmt(v, nd))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def section_dataset() -> None:
    d = read("dataset.csv")
    if d is None:
        return
    print("### Dataset (`results/tables/dataset.csv`)\n")
    print(md_table(
        d,
        ["split", "scenarios", "rows", "networks", "mean_nodes", "mean_service_loss",
         "p90_service_loss", "max_service_loss", "frac_zero_impact", "mean_disruptions"],
        ["split", "scenarios", "rows", "networks", "nodes", "mean loss", "p90", "max",
         "zero-impact rows", "disruptions/scenario"],
    ))
    if "dataset_generation_seconds" in d.columns:
        print(f"\nGeneration: {fmt(d['dataset_generation_seconds'].iloc[0], 1)} s for "
              f"{int(d.get('generated_scenarios', pd.Series([np.nan])).iloc[0])} scenarios "
              "(the experiment uses a deterministic subsample of those).")
    print()


def section_methods() -> None:
    m = read("method_comparison.csv")
    if m is None:
        return
    print("### Method comparison, in-distribution "
          "(`results/tables/method_comparison.csv`, split `test_id`)\n")
    sub = m[m.split == "test_id"].copy()
    sub["o"] = sub.method.map({k: i for i, k in enumerate(METHOD_ORDER)}).fillna(99)
    sub = sub.sort_values("o")
    spec = [
        ("method", "method"),
        ("mae", "MAE (low=good)"),
        ("mae_nonzero_truth", "MAE on nonzero truth (low=good)"),
        ("rmse", "RMSE (low=good)"),
        ("bias", "bias"),
        ("spearman_pooled", "Spearman pooled (high=good)"),
        ("spearman_within_scenario", "Spearman within-scenario (high=good)"),
        ("spearman_groups", "scenarios contributing"),
        ("top1_agreement", "top-1 agree (high=good)"),
        ("top1_groups", "scenarios contributing"),
        ("n_unique_predictions", "unique predictions"),
        ("n", "rows"),
    ]
    spec = [(c, h) for c, h in spec if c in sub.columns]
    print(md_table(sub, [c for c, _ in spec], [h for _, h in spec]))
    print()
    print("`unique predictions` is the degeneracy check: a method with 1 unique "
          "prediction is a constant function and has no ranking at all, whatever "
          "its MAE says. `MAE on nonzero truth` is the column a constant cannot "
          "win.")
    print()


def section_shift() -> None:
    m = read("method_comparison.csv")
    if m is None:
        return
    for metric, label, nd in (
        ("mae", "MAE (lower better)", 4),
        ("mae_nonzero_truth", "MAE on nonzero-truth rows only (lower better)", 4),
        ("spearman_within_scenario", "within-scenario Spearman (higher better)", 4),
    ):
        if metric not in m.columns:
            continue
        print(f"### {label} across every split "
              f"(`results/tables/method_comparison.csv`)\n")
        piv = m.pivot_table(index="method", columns="split", values=metric)
        piv = piv[[c for c in SHIFTS if c in piv.columns]]
        rows = piv.reset_index()
        print(md_table(rows, list(rows.columns), nd=nd))
        print()


def section_trivial() -> None:
    """The trivial baselines, on every split, next to the learned model.

    This is the table whose absence let a constant be reported as the best model
    in this repository. It is printed on all seven splits rather than the five
    evaluation ones because the claim it supports is about all seven.
    """
    m = read("method_comparison.csv")
    if m is None or "constant_zero" not in set(m.method):
        return
    keep = ["constant_zero", "constant_train_mean", "surrogate_single",
            "surrogate_ensemble", "tabular_gbt", "tabular_gbt_l1"]
    for metric, label in (
        ("mae", "MAE (lower better) — the metric a constant can win"),
        ("mae_nonzero_truth",
         "MAE on nonzero-truth rows (lower better) — the metric a constant cannot win"),
    ):
        if metric not in m.columns:
            continue
        print(f"### {label}, every split "
              f"(`results/tables/method_comparison.csv`)\n")
        piv = m.pivot_table(index="method", columns="split", values=metric)
        piv = piv.reindex([k for k in keep if k in piv.index])
        piv = piv[[c for c in ALL_SPLITS if c in piv.columns]]
        rows = piv.reset_index()
        print(md_table(rows, list(rows.columns), nd=6))
        print()
    piv = m.pivot_table(index="method", columns="split", values="mae")
    if "constant_zero" in piv.index and "surrogate_single" in piv.index:
        cols = [c for c in ALL_SPLITS if c in piv.columns]
        lost = [c for c in cols if piv.loc["surrogate_single", c] > piv.loc["constant_zero", c]]
        print(f"`surrogate_single` has a **worse** MAE than `constant_zero` on "
              f"{len(lost)} of {len(cols)} splits ({', '.join(lost)}).\n")
        if "constant_train_mean" in piv.index:
            won = [c for c in cols
                   if piv.loc["surrogate_single", c] < piv.loc["constant_train_mean", c]]
            print(f"It beats `constant_train_mean` on {len(won)} of {len(cols)} "
                  f"splits ({', '.join(won)}).\n")
    if "n_unique_predictions" in m.columns:
        deg = m[m.n_unique_predictions < 2]
        if len(deg):
            names = sorted(set(deg.method))
            print("Methods emitting a single distinct prediction on at least one "
                  "split (constant functions, not models): "
                  + ", ".join(f"`{x}`" for x in names) + ".\n")


def section_seeds() -> None:
    n = read("seed_variance.csv")
    r = read("seed_runs.csv")
    if n is None:
        return
    print("### Seed variance (`results/tables/seed_variance.csv`)\n")
    for split in ("test_id", "shift_topo"):
        sub = n[n.split == split]
        if sub.empty:
            continue
        print(f"**{split}**\n")
        print(md_table(
            sub,
            ["metric", "n_seeds", "mean", "sd", "min", "max", "spread", "diff_noise_scale"],
            ["metric", "seeds", "mean", "sd", "min", "max", "range",
             "noise scale (sqrt2*sd)"],
            nd=5,
        ))
        print()
    if r is not None:
        print("Per-seed values (`results/tables/seed_runs.csv`, split `test_id`):\n")
        piv = r[r.split == "test_id"].pivot_table(
            index="metric", columns="seed", values="value"
        ).reset_index()
        piv.columns = [str(c) for c in piv.columns]
        print(md_table(piv, list(piv.columns), nd=5))
        print()


def _noise_for(metric: str, split: str = "test_id") -> float:
    n = read("seed_variance.csv")
    if n is None:
        return float("nan")
    sub = n[(n.split == split) & (n.metric == metric)]
    return float(sub.diff_noise_scale.iloc[0]) if len(sub) else float("nan")


def _verdict(gain: float, scale: float) -> str:
    """Place a signed gain against the noise scale.

    A large *negative* ratio is not "inside noise" - it means the surrogate is
    decisively worse, which the first version of this function mislabelled and
    would have quietly hidden a defeat behind a reassuring phrase.
    """
    if not np.isfinite(gain) or not np.isfinite(scale) or scale <= 0:
        return "not measured"
    r = gain / scale
    if r >= 2.0:
        return f"{r:+.2f} -> survives"
    if r >= 1.0:
        return f"{r:+.2f} -> suggestive"
    if r > -1.0:
        return f"{r:+.2f} -> inside noise"
    if r > -2.0:
        return f"{r:+.2f} -> worse (suggestive)"
    return f"{r:+.2f} -> **worse, outside noise**"


def section_verdicts() -> None:
    """The headline claims, each placed against the seed noise scale."""
    m = read("method_comparison.csv")
    if m is None:
        return
    print("### Every claimed gain against the run-to-run noise scale\n")
    print("A difference between two single runs is inside noise unless it exceeds "
          "`sqrt(2) * sd` for that metric, from `seed_variance.csv`.\n")
    print("The surrogate is compared against **four** references: the reference-style "
          "feature model it is meant to replace, whichever baseline is actually "
          "strongest on that split, and both trivial constants. Comparing only "
          "against the reference would flatter it, and on a 92.5%-zero target "
          "omitting the constants hides the only comparison that establishes "
          "whether anything was learned at all.\n")
    rows = []
    for split in SHIFTS:
        sub = m[m.split == split].set_index("method")
        if "surrogate_ensemble" not in sub.index:
            continue
        for metric, better_is_lower in (("mae", True),
                                        ("mae_nonzero_truth", True),
                                        ("spearman_within_scenario", False)):
            if metric not in sub.columns:
                continue
            ours = float(sub.loc["surrogate_ensemble", metric])
            scale = _noise_for(metric, split)
            # "Baseline" excludes every surrogate variant: comparing the ensemble
            # against its own single member is not a baseline comparison.
            others = {
                b: float(sub.loc[b, metric])
                for b in sub.index
                if not str(b).startswith("surrogate") and np.isfinite(sub.loc[b, metric])
            }
            if not others:
                continue
            # A constant is not eligible to be the "best baseline": calling the
            # all-zero function the strongest baseline would be true on MAE and
            # completely uninformative. It gets its own explicit rows instead.
            ranked = {k: v for k, v in others.items() if not k.startswith("constant_")}
            best = (min if better_is_lower else max)(ranked or others, key=others.get)
            for label, ref_name in (("vs reference", "tabular_gbt"),
                                    ("vs best baseline", best),
                                    ("vs constant zero", "constant_zero"),
                                    ("vs constant train-mean", "constant_train_mean")):
                if ref_name not in others:
                    continue
                ref = others[ref_name]
                gain = (ref - ours) if better_is_lower else (ours - ref)
                rows.append({
                    "split": split, "metric": metric, "comparison": label,
                    "baseline": ref_name,
                    "surrogate": fmt(ours, 5), "baseline_value": fmt(ref, 5),
                    "gain": fmt(gain, 5), "noise_scale": fmt(scale, 5),
                    "verdict": _verdict(gain, scale),
                })
    if rows:
        print(md_table(pd.DataFrame(rows), list(rows[0].keys())))
    print()


def section_stats() -> None:
    s = read("statistical_tests.csv")
    if s is None:
        return
    print("### Paired per-row tests vs the reference model "
          "(`results/tables/statistical_tests.csv`)\n")
    print("Unit of analysis is a **row**, so this compares two sets of weights, "
          "not two methods.\n")
    sub = s[s.split == "test_id"]
    print(md_table(
        sub,
        ["name_a", "mean_a", "mean_b", "difference", "ci_lower", "ci_upper",
         "p_adjusted", "effect_size", "n"],
        ["method (abs error)", "mean", "ref mean", "delta", "CI low", "CI high",
         "p (Holm)", "Cohen's d", "n"],
        nd=5,
    ))
    print()


def section_ablation() -> None:
    a = read("ablations.csv")
    if a is None:
        return
    print("### Ablations (`results/tables/ablations.csv`, split `test_id`)\n")
    sub = a[a.split == "test_id"].copy()
    full = sub[sub.variant == "full"]
    base = float(full.mae.iloc[0]) if len(full) else float("nan")
    scale = _noise_for("mae")
    sub["delta_vs_full"] = sub.mae - base
    sub["verdict"] = [_verdict(d, scale) for d in sub.delta_vs_full]
    print(md_table(
        sub,
        ["variant", "params", "mae", "delta_vs_full", "verdict",
         "spearman_within_scenario", "train_seconds"],
        ["variant", "params", "MAE", "delta vs full", "vs noise",
         "Spearman (within)", "train s"],
        nd=5,
    ))
    print("\nA positive delta means removing that mechanism made the model worse.\n")


def section_criticality() -> None:
    c = read("criticality_summary.csv")
    if c is None:
        return
    print("### Counterfactual criticality vs the simulator oracle "
          "(`results/tables/criticality_summary.csv`)\n")
    cols = ["method"] + [x for x in c.columns if x.startswith(("recall@", "precision@",
                                                               "regret_frac@"))]
    cols += [x for x in ("spearman_full", "score_std", "distinct_scores",
                         "method_seconds", "sim_seconds",
                         "speedup_vs_simulator") if x in c.columns]
    print(md_table(c, cols, nd=4))
    print()
    if "score_std" in c.columns and (c.score_std < 1e-9).any():
        dead = ", ".join(f"`{m}`" for m in c.loc[c.score_std < 1e-9, "method"])
        print(f"**{dead} produced a constant score for every candidate**, i.e. "
              "one distinct score across all candidates in all networks. That is "
              "not a fact about tabular features or about topology; it is a fact "
              "about an L1 objective on a 92.5%-zero target, where the "
              "loss-minimising constant is the median and the median is exactly "
              "0. Its recall@k is therefore whatever a tie-broken arbitrary "
              "ordering scores. It is reported here, labelled, because this "
              "repository previously shipped it *as* `tabular_gbt` - the "
              "reference model the surrogate was said to lose to - and the "
              "collapse was invisible in every metric except this one. The "
              "squared-error variant next to it is the boosted-tree reference "
              "that actually ranks.\n")
    d = read("disagreements.csv")
    if d is not None and len(d):
        print("### Where the feature score and the counterfactual disagree "
              "(`results/tables/disagreements.csv`)\n")
        print(f"{len(d)} disagreeing pairs found. Simulator verdict: "
              + ", ".join(f"**{k}** {v}" for k, v in d.winner.value_counts().items())
              + ".\n")
        top = d.nlargest(min(5, len(d)), "margin")
        print(md_table(
            top,
            ["network", "node_a", "rank_a_feature", "rank_a_counterfactual", "truth_a",
             "node_b", "rank_b_feature", "rank_b_counterfactual", "truth_b",
             "winner", "margin"],
            ["network", "A", "A rank (feat)", "A rank (cf)", "A true loss",
             "B", "B rank (feat)", "B rank (cf)", "B true loss", "winner", "margin"],
        ))
        print()
        print("Worked example of the largest disagreement:\n")
        print("> " + str(top.explanation.iloc[0]).replace(" | ", "\n> \n> "))
        print()


def section_calibration() -> None:
    c = read("calibration.csv")
    if c is None:
        return
    print("### Predictive-interval calibration (`results/tables/calibration.csv`)\n")
    g = c[(c.interval == "gaussian")]
    if len(g):
        piv = g.pivot_table(index="split", columns="nominal",
                            values=["coverage", "width"]).reset_index()
        piv.columns = [f"{a}@{b}" if b != "" else a for a, b in piv.columns]
        print("**Deep-ensemble Gaussian intervals**\n")
        print(md_table(piv, list(piv.columns)))
        print()
    q = c[c.interval == "quantile_central"]
    if len(q):
        print("**Quantile-head central interval** (nominal "
              f"{fmt(q.nominal.iloc[0], 2)})\n")
        print(md_table(q, ["split", "coverage", "width", "coverage_gap", "n"],
                       ["split", "coverage", "width", "gap", "n"]))
        print()
    s = c[c.interval == "uncertainty_shift"]
    if len(s):
        print("**Does uncertainty track error?**\n")
        print(md_table(
            s,
            ["split", "mean_sigma", "mean_abs_error", "sigma_error_spearman",
             "error_detection_auroc", "ause", "mean_sigma_aleatoric",
             "mean_sigma_epistemic"],
            ["split", "mean sigma", "mean |err|", "sigma-vs-err Spearman", "err-detect AUROC",
             "AUSE", "sigma aleatoric", "sigma epistemic"],
        ))
        print()


def section_efficiency() -> None:
    e = read("efficiency.csv")
    if e is not None:
        print("### Measured wall-clock (`results/tables/efficiency.csv`)\n")
        agg = e.groupby(["component", "variant"]).agg(
            n_nodes=("n_nodes", "median"),
            median_ms=("median_ms", "median"),
            iqr_ms=("iqr_ms", "median"),
            per_item_ms=("per_item_ms", "median"),
            params=("params", "median"),
        ).reset_index()
        print(md_table(agg, list(agg.columns), nd=4))
        print("\nWarm-up ≥ 8 iterations, ≥ 25 timed repeats, median and IQR reported.\n")
    b = read("break_even.csv")
    if b is not None:
        print("### Break-even accounting (`results/tables/break_even.csv`)\n")
        r = b.iloc[0]
        keys = ["sim_ms_per_scenario", "surrogate_ms_per_scenario",
                "speedup_per_scenario", "train_seconds", "ensemble_train_seconds",
                "dataset_seconds", "setup_seconds", "break_even_scenarios",
                "break_even_scenarios_ensemble"]
        print("| quantity | value |")
        print("|---|---|")
        for k in keys:
            if k in b.columns:
                print(f"| `{k}` | {fmt(r[k], 3)} |")
        print("\nSetup cost includes the dataset generation that used the very simulator "
              "being replaced.\n")
    c = read("budget_curve.csv")
    if c is not None:
        print("### Decision quality at a fixed compute budget "
              "(`results/tables/budget_curve.csv`)\n")
        piv = c.groupby(["budget_s", "method"]).recall_at_k.mean().unstack().reset_index()
        print(md_table(piv, list(piv.columns), nd=3))
        print("\nRecall@10 of the true critical set, averaged over the evaluated "
              "networks.\n")


SECTIONS = {
    "dataset": section_dataset,
    "methods": section_methods,
    "trivial": section_trivial,
    "shift": section_shift,
    "seeds": section_seeds,
    "verdicts": section_verdicts,
    "stats": section_stats,
    "ablation": section_ablation,
    "criticality": section_criticality,
    "calibration": section_calibration,
    "efficiency": section_efficiency,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", default="all", help="|".join(["all", *SECTIONS]))
    args = ap.parse_args(argv)
    names = list(SECTIONS) if args.section == "all" else args.section.split(",")
    for n in names:
        if n not in SECTIONS:
            raise SystemExit(f"unknown section {n!r}")
        SECTIONS[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
