"""Cross-check every hand-written number in README.md and docs/RESULTS.md.

Usage:
    python scripts/check_docs_against_csvs.py     # exits non-zero on any mismatch


The auto-generated tables cannot be wrong (they are rendered from the CSVs), but
the prose around them is typed, and that is exactly where the reference project
shipped three wrong tables. Every claim below is asserted against its source CSV.
"""
import pathlib
import sys

import pandas as pd

R = pathlib.Path(__file__).resolve().parent.parent
T = R / "results/tables"

m = pd.read_csv(T / "method_comparison.csv")
a = pd.read_csv(T / "ablations.csv")
c = pd.read_csv(T / "criticality_summary.csv").set_index("method")
d = pd.read_csv(T / "disagreements.csv")
b = pd.read_csv(T / "break_even.csv").iloc[0]
ds = pd.read_csv(T / "dataset.csv").set_index("split")
bc = pd.read_csv(T / "budget_curve.csv")
sv = pd.read_csv(T / "seed_variance.csv")
cal = pd.read_csv(T / "calibration.csv")

mt = m[m.split == "test_id"].set_index("method")
at = a[a.split == "test_id"].set_index("variant")
readme = (R / "README.md").read_text(encoding="utf-8")
results = (R / "docs/RESULTS.md").read_text(encoding="utf-8")
method = (R / "docs/METHOD.md").read_text(encoding="utf-8")
both = readme + results

checks: list[tuple[str, bool, str]] = []


def chk(label, cond, detail=""):
    checks.append((label, bool(cond), detail))


# --- disagreement headline ---
chk("36 disagreement pairs", len(d) == 36 and "36 of 36" in readme, f"len={len(d)}")
chk("all won by counterfactual", (d.winner == "counterfactual").all())
chk("truth_a mean and max are exactly 0", d.truth_a.max() == 0.0 and "exactly 0.0000" in readme)
chk("truth_b mean 0.4288", abs(d.truth_b.mean() - 0.4288) < 5e-5 and "0.4288" in readme)

# --- criticality ---
chk("heuristic recall@10 = 0.633",
    abs(c.loc["topology_heuristic", "recall@10"] - 0.6333) < 5e-4 and "0.633" in readme)
chk("surrogate recall@10 = 0.433",
    abs(c.loc["surrogate", "recall@10"] - 0.4333) < 5e-4 and "0.433" in readme)
chk("heuristic regret_frac@10 = 0.161",
    abs(c.loc["topology_heuristic", "regret_frac@10"] - 0.1613) < 5e-4 and "0.161" in readme)
chk("surrogate regret_frac@10 = 0.470",
    abs(c.loc["surrogate", "regret_frac@10"] - 0.4701) < 5e-4 and "0.470" in readme)
chk("gbt is constant (1 distinct score)", c.loc["tabular_gbt", "distinct_scores"] == 1.0)
chk("gbt recall@10 = 0.05", abs(c.loc["tabular_gbt", "recall@10"] - 0.05) < 1e-6
    and "0.05" in readme)
chk("gbt regret 0.98", abs(c.loc["tabular_gbt", "regret_frac@10"] - 0.9792) < 5e-4
    and "0.98" in readme)
chk("surrogate score spread 0.0007",
    abs(c.loc["surrogate", "score_std"] - 0.0007) < 5e-5 and "0.0007" in readme)
chk("surrogate spearman_full 0.21 vs heuristic 0.57",
    abs(c.loc["surrogate", "spearman_full"] - 0.2071) < 5e-4
    and abs(c.loc["topology_heuristic", "spearman_full"] - 0.5683) < 5e-4)

# --- ablations ---
full_mae = at.loc["full", "mae"]
nomp_mae = at.loc["no_message_passing", "mae"]
noise_mae = float(sv[(sv.split == "test_id") & (sv.metric == "mae")].diff_noise_scale.iloc[0])
ratio = (nomp_mae - full_mae) / noise_mae
chk("no_message_passing ratio 4.11", abs(ratio - 4.11) < 0.02 and "+4.11" in readme,
    f"ratio={ratio:.3f}")
chk("no_mp MAE delta 0.0028", abs((nomp_mae - full_mae) - 0.00282) < 5e-5
    and "0.0028" in readme)
chk("ablation spearman 0.582 -> 0.402",
    abs(at.loc["full", "spearman_within_scenario"] - 0.5824) < 5e-4
    and abs(at.loc["no_message_passing", "spearman_within_scenario"] - 0.4018) < 5e-4
    and "0.582 to 0.402" in readme)
chk("parameter ratio 1.025",
    abs(at.loc["no_message_passing", "params"] / at.loc["full", "params"] - 1.025) < 0.002
    and "1.025x" in readme)
chk("linear_traj_decoder inside noise",
    abs(at.loc["linear_traj_decoder", "mae"] - full_mae) / noise_mae < 1.0)

# --- efficiency ---
chk("speedup 11.4x", abs(b.speedup_per_scenario - 11.41) < 0.02 and "11.4x" in readme)
chk("sim 50.3 ms", abs(b.sim_ms_per_scenario - 50.328) < 0.01 and "50.3 ms" in readme)
chk("surrogate 4.41 ms", abs(b.surrogate_ms_per_scenario - 4.411) < 0.01
    and "4.41 ms" in readme)
chk("break-even 13,270", abs(b.break_even_scenarios - 13270.2) < 1.0 and "13,270" in readme)
chk("break-even ensemble 26,167",
    abs(b.break_even_scenarios_ensemble - 26166.6) < 1.0 and "26,167" in readme)
chk("dataset_seconds is the generation time, not a cache load",
    abs(b.dataset_seconds - 313.247) < 0.01)

# --- budget curve ---
piv = bc.groupby(["budget_s", "method"]).recall_at_k.mean().unstack()
chk("surrogate ties at 1 s (0.367 vs 0.369)",
    abs(piv.loc[1.0, "surrogate"] - 0.367) < 5e-4
    and abs(piv.loc[1.0, "simulator"] - 0.369) < 5e-4 and "0.367" in readme)
chk("simulator reaches 1.0 by 5 s", piv.loc[5.0, "simulator"] == 1.0)
chk("surrogate plateaus at 0.433", abs(piv.loc[10.0, "surrogate"] - 0.433) < 5e-4
    and "0.433" in readme)
chk("surrogate never beats simulator",
    all(piv.loc[x, "surrogate"] <= piv.loc[x, "simulator"] + 1e-9 for x in piv.index)
    and "never win" in readme)

# --- dataset ---
chk("93.5% zero rows in train",
    abs(ds.loc["train", "frac_zero_impact"] - 0.9347) < 5e-4
    and "93.5%" in readme and "93.5%" in method)
chk("5140 generated scenarios", ds.loc["train", "generated_scenarios"] == 5140)
chk("313 s generation", abs(ds.loc["train", "dataset_generation_seconds"] - 313.247) < 0.01
    and "313 s" in results)

# --- retrieval / shift verdicts ---
noise_sp = float(
    sv[(sv.split == "test_id") & (sv.metric == "spearman_within_scenario")]
    .diff_noise_scale.iloc[0]
)
gap = mt.loc["surrogate_ensemble", "spearman_within_scenario"] - mt.loc[
    "retrieval_knn", "spearman_within_scenario"]
chk("retrieval beats surrogate in-distribution by 3.86x noise",
    abs(gap / noise_sp + 3.86) < 0.05 and "3.86x" in results, f"{gap/noise_sp:.3f}")

ms = m[m.split == "shift_size"].set_index("method")
nz = float(
    sv[(sv.split == "shift_size") & (sv.metric == "spearman_within_scenario")]
    .diff_noise_scale.iloc[0]
)
g2 = ms.loc["surrogate_ensemble", "spearman_within_scenario"] - ms.loc[
    "retrieval_knn", "spearman_within_scenario"]
chk("surrogate wins shift_size by 2.50x noise",
    abs(g2 / nz - 2.50) < 0.05 and "2.50x" in results, f"{g2/nz:.3f}")

# --- calibration ---
g = cal[(cal.interval == "gaussian") & (cal.split == "test_id")].set_index("nominal")
chk("nominal 0.50 covers ~0.97", abs(g.loc[0.50, "coverage"] - 0.967) < 0.01
    and "97%" in results)
sh = cal[(cal.interval == "uncertainty_shift") & (cal.split == "test_id")].iloc[0]
chk("error-detection AUROC 0.99", sh.error_detection_auroc > 0.98 and "0.99" in readme)
chk("epistemic is ~4% of total sigma",
    sh.mean_sigma_epistemic / sh.mean_sigma_aleatoric < 0.06 and "4%" in results)

ok = sum(1 for _, p, _ in checks if p)
for label, passed, detail in checks:
    print(f"{'PASS' if passed else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
print(f"\n{ok}/{len(checks)} cross-checks passed")
sys.exit(0 if ok == len(checks) else 1)
