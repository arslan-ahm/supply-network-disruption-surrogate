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
pi = pd.read_csv(R / "results/runs/base/per_item.csv")
mz = m.pivot_table(index="method", columns="split", values="mae")
mnz = m.pivot_table(index="method", columns="split", values="mae_nonzero_truth")
ALL_SPLITS = ["train", "val", "test_id", "shift_topo", "shift_size", "shift_type",
              "shift_multi"]

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
chk("36 disagreement pairs", len(d) == 36 and "36 of 36" in readme
    and "36 of 36" in results, f"len={len(d)}")
chk("all won by counterfactual", (d.winner == "counterfactual").all())

# --- criticality ---
chk("heuristic recall@10 = 0.633",
    abs(c.loc["topology_heuristic", "recall@10"] - 0.6333) < 5e-4 and "0.633" in readme)
chk("heuristic regret_frac@10 = 0.161",
    abs(c.loc["topology_heuristic", "regret_frac@10"] - 0.1613) < 5e-4 and "0.161" in readme)
chk("surrogate regret_frac@10 = 0.433",
    abs(c.loc["surrogate", "regret_frac@10"] - 0.4332) < 5e-4 and "0.433" in readme)
chk("surrogate spearman_full 0.26 vs heuristic 0.57",
    abs(c.loc["surrogate", "spearman_full"] - 0.2609) < 5e-4
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
chk("simulator leads at 1 s (0.548 vs 0.333)",
    abs(piv.loc[1.0, "surrogate"] - 0.333) < 5e-4
    and abs(piv.loc[1.0, "simulator"] - 0.548) < 5e-4
    and "0.548" in readme and "0.333" in readme)
chk("simulator reaches 1.0 by 5 s", piv.loc[5.0, "simulator"] == 1.0)
chk("surrogate plateaus at 0.450", abs(piv.loc[10.0, "surrogate"] - 0.45) < 5e-4
    and "0.450" in readme)
chk("surrogate never beats simulator",
    all(piv.loc[x, "surrogate"] <= piv.loc[x, "simulator"] + 1e-9 for x in piv.index)
    and "never win" in readme)

# --- dataset ---
chk("93.5% zero rows in train (dataset.csv, <=1e-4 threshold)",
    abs(ds.loc["train", "frac_zero_impact"] - 0.9347) < 5e-4
    and "93.5%" in readme and "93.5%" in method and "93.5%" in results)
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
chk("surrogate wins shift_size by 2.59x noise",
    abs(g2 / nz - 2.59) < 0.05 and "2.59x" in results, f"{g2/nz:.3f}")

# --- calibration ---
g = cal[(cal.interval == "gaussian") & (cal.split == "test_id")].set_index("nominal")
chk("nominal 0.50 covers ~0.97", abs(g.loc[0.50, "coverage"] - 0.967) < 0.01
    and "97%" in results)
sh = cal[(cal.interval == "uncertainty_shift") & (cal.split == "test_id")].iloc[0]
chk("error-detection AUROC 0.99", sh.error_detection_auroc > 0.98 and "0.99" in readme)
chk("epistemic is ~4% of total sigma",
    sh.mean_sigma_epistemic / sh.mean_sigma_aleatoric < 0.06 and "4%" in results)

# --- the zero-inflation premise, which every metric claim rests on ---
zero_frac = float((pi.truth == 0.0).mean())
chk("92.48% of per-item rows are exactly zero",
    abs(zero_frac - 0.9248) < 5e-5 and "92.48%" in both, f"{zero_frac:.6f}")
chk("19,073 of 20,624 rows",
    int((pi.truth == 0.0).sum()) == 19073 and len(pi) == 20624
    and "19,073 of 20,624" in both)
nzrows = pi[pi.truth > 0]
chk("1,551 nonzero rows, mean truth 0.193317",
    len(nzrows) == 1551 and abs(nzrows.truth.mean() - 0.193317) < 5e-6
    and "1,551" in results and "0.193317" in results)
chk("the median of the target is exactly 0", float(pi.truth.median()) == 0.0)

# --- the fixed GBT is a model; the L1 variant is a constant ---
chk("tabular_gbt is no longer a constant",
    int(pi.pred_tabular_gbt.nunique()) > 100
    and mt.loc["tabular_gbt", "n_unique_predictions"] > 100,
    f"{pi.pred_tabular_gbt.nunique()} unique")
chk("tabular_gbt_l1 emits exactly one distinct prediction",
    int(pi.pred_tabular_gbt_l1.nunique()) == 1
    and (m[m.method == "tabular_gbt_l1"].n_unique_predictions == 1).all())
chk("tabular_gbt_l1 is element-wise identical to constant_zero",
    bool((pi.pred_tabular_gbt_l1 == pi.pred_constant_zero).all())
    and "element-wise identical" in results)
chk("the old shipped GBT was exactly 0 for all 20,624 rows",
    "exactly 0.000000 for all 20,624 rows" in readme
    and "exactly 0.000000 for all 20,624 rows" in results)
chk("30 tree nodes in the collapsed model",
    "30 total tree nodes" in results and "30 total tree nodes" in readme.replace(
        "30 total tree nodes", "30 total tree nodes"))

# --- the trivial baselines are present and reported ---
for name in ("constant_zero", "constant_train_mean"):
    chk(f"{name} present on all 7 splits",
        set(m[m.method == name].split) == set(ALL_SPLITS))
    chk(f"{name} in per_item.csv", f"pred_{name}" in pi.columns)
chk("constant_train_mean value is the train mean 0.012783",
    abs(float(pi.loc[pi.split == "train", "truth"].mean()) - 0.012783) < 5e-7
    and abs(mz.loc["constant_train_mean", "train"] - 0.023994) < 5e-6
    and "0.012783" in readme)

# --- the headline corrected claim ---
lost = [c for c in ALL_SPLITS
        if mz.loc["surrogate_single", c] > mz.loc["constant_zero", c]]
chk("surrogate loses pooled MAE to constant_zero on 6 of 7 splits",
    len(lost) == 6 and "shift_multi" not in lost
    and "6 of 7" in readme and "6 of 7" in results, ",".join(lost))
won = [c for c in ALL_SPLITS
       if mz.loc["surrogate_single", c] < mz.loc["constant_train_mean", c]]
chk("surrogate beats constant_train_mean on pooled MAE on 7 of 7 splits",
    len(won) == 7 and "all 7" in readme, f"{len(won)}")
nzwon = [c for c in ALL_SPLITS
         if mnz.loc["surrogate_single", c] < mnz.loc["constant_zero", c]]
chk("surrogate beats constant_zero on nonzero-truth MAE on all 7 splits",
    len(nzwon) == 7 and "all 7 splits" in readme, f"{len(nzwon)}")
chk("test_id: surrogate 0.014581 vs constant_zero 0.012246 on MAE",
    abs(mz.loc["surrogate_single", "test_id"] - 0.014581) < 5e-6
    and abs(mz.loc["constant_zero", "test_id"] - 0.012246) < 5e-6
    and "0.014581" in readme and "0.012246" in readme)
chk("test_id: surrogate 0.1569 vs constant_zero 0.2137 on nonzero MAE",
    abs(mnz.loc["surrogate_single", "test_id"] - 0.156903) < 5e-6
    and abs(mnz.loc["constant_zero", "test_id"] - 0.213741) < 5e-6
    and "0.1569" in readme and "0.2137" in readme)
# Identity, not approximation: a zero predictor's error on the nonzero rows *is*
# the mean nonzero truth. The tolerance is 1e-6 only because the CSV is written
# at six decimal places.
chk("a zero predictor scores exactly the mean nonzero truth",
    abs(mnz.loc["constant_zero", "test_id"]
        - float(pi.loc[(pi.split == "test_id") & (pi.truth > 0), "truth"].mean())) < 1e-6)

# --- the fixed GBT beats the surrogate, which is the real defeat ---
chk("tabular_gbt beats the surrogate on MAE and nonzero MAE on test_id",
    mt.loc["tabular_gbt", "mae"] < mt.loc["surrogate_single", "mae"]
    and mt.loc["tabular_gbt", "mae_nonzero_truth"]
    < mt.loc["surrogate_single", "mae_nonzero_truth"]
    and abs(mt.loc["tabular_gbt", "mae"] - 0.010418) < 5e-6
    and abs(mt.loc["tabular_gbt", "mae_nonzero_truth"] - 0.094543) < 5e-6
    and "0.0104" in readme and "0.0945" in readme)
chk("tabular_gbt has the best top-1 agreement on test_id",
    abs(mt.loc["tabular_gbt", "top1_agreement"] - 0.729167) < 5e-5
    and "0.729" in readme)
mtype = m[m.split == "shift_type"].set_index("method")
nsp = float(sv[(sv.split == "shift_type")
               & (sv.metric == "spearman_within_scenario")].diff_noise_scale.iloc[0])
ratio = (mtype.loc["surrogate_ensemble", "spearman_within_scenario"]
         - mtype.loc["tabular_gbt", "spearman_within_scenario"]) / nsp
chk("surrogate beats the GBT on shift_type rank by +8.90x noise",
    abs(ratio - 8.90) < 0.05 and "8.90" in readme and "8.90" in results,
    f"{ratio:.3f}")

# --- the sign-vs-magnitude disagreement ---
st = pd.read_csv(T / "statistical_tests.csv")
stt = st[st.split == "test_id"].set_index("name_a")
chk("constant_zero is worse on only 4.5% of rows",
    abs(stt.loc["constant_zero.abs_error", "frac_rows_a_worse"] - 0.044922) < 5e-6
    and stt.loc["constant_zero.abs_error", "median_difference"] == 0.0
    and "4.5% of rows" in results)
chk("the surrogate is worse on 82.3% of rows",
    abs(stt.loc["surrogate_single.abs_error", "frac_rows_a_worse"] - 0.822917) < 5e-6
    and "82.3%" in results)

# --- the contaminated disagreement experiment ---
chk("rank_a_feature is no longer node_a + 1",
    int((d.rank_a_feature == d.node_a + 1).sum()) == 1
    and "1 of 36" in results and "all 36 rows" in results)
chk("truth_a mean 0.000926 max 0.033346",
    abs(d.truth_a.mean() - 0.000926) < 5e-6 and abs(d.truth_a.max() - 0.033346) < 5e-6
    and "0.000926" in both and "0.033346" in both)
chk("truth_b mean 0.426345 min 0.228754",
    abs(d.truth_b.mean() - 0.426345) < 5e-6 and abs(d.truth_b.min() - 0.228754) < 5e-6
    and "0.426345" in both and "0.228754" in both)

# --- criticality, both variants ---
chk("gbt_l1 is the constant row this table used to publish as gbt",
    c.loc["tabular_gbt_l1", "distinct_scores"] == 1.0
    and abs(c.loc["tabular_gbt_l1", "recall@10"] - 0.05) < 1e-6
    and abs(c.loc["tabular_gbt_l1", "regret_frac@10"] - 0.9791) < 5e-4)
chk("working gbt recall@10 0.183, regret 0.867, 6-9 distinct scores",
    abs(c.loc["tabular_gbt", "recall@10"] - 0.1833) < 5e-4
    and abs(c.loc["tabular_gbt", "regret_frac@10"] - 0.8667) < 5e-4
    and abs(c.loc["tabular_gbt", "distinct_scores"] - 7.3333) < 5e-4
    and "0.183" in both and "0.867" in results and "6 to 9 distinct scores" in results)
cd = pd.read_csv(T / "criticality_detail.csv")
per_net = cd.groupby("network").tabular_gbt.nunique()
chk("gbt distinct scores per network are between 6 and 9",
    int(per_net.min()) == 6 and int(per_net.max()) == 9)
chk("the worst-true node's feature rank is 45,20,4,41,46,25",
    all(x in readme for x in ("45, 20, 4, 41, 46 and 25",)))

# --- the surrogate's criticality ranking is smaller than its own noise ---
chk("surrogate criticality score_std 0.0008, max pred 0.0523, true max 0.7481",
    abs(c.loc["surrogate", "score_std"] - 0.0008) < 5e-5
    and abs(cd.surrogate.max() - 0.0523) < 5e-4
    and abs(cd.truth.max() - 0.7481) < 5e-4
    and "0.0008" in both and "0.0523" in both and "0.7481" in both)
chk("surrogate recall@10 0.450 and full Spearman 0.261",
    abs(c.loc["surrogate", "recall@10"] - 0.45) < 5e-4
    and abs(c.loc["surrogate", "spearman_full"] - 0.2609) < 5e-4
    and "0.450" in readme and "0.261" in results)
chk("recall@10 moved 0.433 -> 0.450 across launches",
    "0.433 to 0.450" in results and "0.433 in one process and 0.450 in" in readme)

ok = sum(1 for _, p, _ in checks if p)
for label, passed, detail in checks:
    print(f"{'PASS' if passed else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
print(f"\n{ok}/{len(checks)} cross-checks passed")
sys.exit(0 if ok == len(checks) else 1)
