![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13%20CPU-red)
![Tests](https://img.shields.io/badge/tests-375%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

# Counterfactual Impact, Not a Risk Score

**A learned temporal graph surrogate for a multi-echelon supply-network
simulator — validated against that simulator as an exact oracle.**

The reference approach scores suppliers: features about a supplier go into a
classifier, out comes a risk number. This repository argues that is the wrong
object, replaces it with a **counterfactual** — *if this node fails for eight
periods, what happens to service level at each demand point, when does it bite,
and how long until recovery* — and, because the simulator gives the true answer
for every candidate, adjudicates the two approaches against ground truth instead
of against a proxy.

**Efficiency axis: per-scenario evaluation cost.** One disruption scenario costs
**50.33 ms** in the simulator against **4.41 ms** batched through the surrogate — a
measured **11.41×** (`results/tables/efficiency.csv`, warm-up 8, 25 repeats, median
and IQR). **What that costs:** the saving only repays the training and
dataset-generation budget after **13,270 scenarios** (**26,167** if the ensemble is
charged), so at any realistic sweep size the simulator is simply the better tool.
The speed factor is real; the fixed-compute experiment is one the surrogate loses.

> **Correction first, because it invalidates the previous version of this
> section.** An earlier release of this repository retracted its
> magnitude-fidelity claim on the grounds that the surrogate "loses to a
> reference GBT on 4 of 5 splits (−4.5σ to −22.2σ)". **There was no reference
> GBT.** `TabularRiskModel(kind="gbt")` was configured with
> `loss="absolute_error"` to "match the surrogate's L1 objective"; the target is
> **92.48% exact zeros** (19,073 of 20,624 rows), so the L1-optimal constant is
> the median and the median is exactly 0. The boosting initialisation was 0,
> every leaf's L1-optimal value was 0, early stopping fired after 10 rounds, and
> the model emitted **exactly 0.000000 for all 20,624 rows** — one unique
> prediction, in 30 total tree nodes. Its prediction vector is *element-wise
> identical* to an all-zero predictor's. **What beat the surrogate was the
> constant zero function**, and pooled MAE on a target like this is minimised by
> predicting zero, so that "defeat" was a property of the metric rather than of
> a rival model. It is kept in the tables as `tabular_gbt_l1`, labelled
> degenerate by construction, because deleting it would hide how this happened.
>
> **The corrected statement is worse, not better, on two counts.**
>
> *One.* **The surrogate is beaten on pooled MAE by the constant zero function on
> 6 of 7 splits** (0.014581 against 0.012246 on `test_id`); it wins only
> `shift_multi`, the split with the least zero mass. It does beat the constant
> *train mean* on all 7. And on the one magnitude metric a constant cannot win —
> MAE restricted to the rows where a disruption actually bit — it beats the
> all-zero constant on **all 7 splits** (0.1569 against 0.2137 on `test_id`,
> +2.25σ to +6.20σ). So the model has learned something about magnitude; it
> spends that skill being wrong on the 92.5% of rows whose answer is zero, and
> pooled MAE weighs those rows 12 to 1. Both numbers are in the tables; neither
> alone is the answer.
>
> *Two.* Fixing the loss produced a genuinely strong baseline, and it beats the
> surrogate outright. `tabular_gbt` under squared error now has the best MAE
> (0.0104 against 0.0146), the best MAE on nonzero truth (0.0945 against 0.1569)
> and the best top-1 agreement (0.729 against 0.583) on `test_id`. **The
> magnitude claim stays retracted — now against a model instead of a constant.**
>
> **The same bug contaminated the headline experiment.** `find_disagreements`
> ranks candidates by `argsort` of the feature score, and `argsort` of a constant
> returns index order, so all 36 adjudicated disagreements had
> `rank_a_feature == node_a + 1`: the "node the feature score ranks first" was
> simply the lowest-numbered candidate. Re-run against a GBT that ranks, **the
> conclusion survives — 36 pairs, 36 of 36 to the counterfactual** — and now on
> real ranks. The evidence was invalid; the claim was not.
>
> **One thing the fix gained.** With a baseline whose rank correlation is defined
> at all, a new result appears that was previously unmeasurable: on the held-out
> disruption mechanism the surrogate's within-scenario Spearman is **0.503
> against the GBT's 0.288, +8.90σ** — the largest surviving margin anywhere in
> this repository. The tabular model interpolates the mechanisms it was trained
> on; the surrogate transfers to one it has never seen.
>
> Full account: [Results §8.7](docs/RESULTS.md). A guard now fails the run if any
> method not declared constant-by-design emits fewer than two distinct
> predictions, and `constant_zero` / `constant_train_mean` are permanent rows in
> every comparison table — because the reason this went unnoticed for 43 commits
> is that **there was no row saying what predicting nothing scores.**

---

> **Result, up front — and the flattering half is not the whole story.**
>
> **The argument holds, and it holds against a baseline that works.** Across six
> unseen networks there are 36 candidate pairs that the reference-style feature
> score and the counterfactual ranking order oppositely. The simulator resolves
> **36 of 36 in favour of the counterfactual**. The nodes the feature score
> prefers average a true service loss of **0.000926** (maximum 0.033346); the
> nodes it passes over average **0.426345** (minimum 0.228754). In `shift_4`
> every one of the feature score's top five candidates has a true loss of
> **exactly 0.0000**, while the node it ranks **46th of 46** has the largest true
> loss in the network at **0.7430**.
>
> **Retracted from the previous release:** "in every one of the six networks the
> node ranked 44th–46th of 46 by features is the worst real single point of
> failure". That was an artefact of the constant baseline described above — with
> a constant score, `argsort` returns node-id order, so "ranked 46th" meant
> "highest node id". Against a working feature model the worst-true node's
> feature rank is **45, 20, 4, 41, 46 and 25** of 46 across the six networks
> (`results/tables/criticality_detail.csv`). It lands in the bottom third in
> three of six, and in `shift_2` the feature score actually ranks it **4th** —
> i.e. gets it nearly right. The feature approach is a poor criticality ranker on
> average (recall@10 0.183, regret 0.867), not a uniformly inverted one, and the
> stronger version of that claim does not survive.
>
> **The learned model is not what wins it.** A hand-weighted composite of six
> topology signals beats the trained graph network on every criticality metric
> (recall@10 **0.633 vs 0.450**, regret fraction **0.161 vs 0.433**). The
> surrogate is a poor criticality ranker: its score spread across candidates is
> 0.0008 against a true spread of 0.748, and its full-vector rank correlation with
> the truth is **0.26**, against **0.57** for the heuristic, and it under-predicts the worst impacts
> by more than an order of magnitude (max predicted 0.0523 against max true
> 0.7481). Worse, that spread is *smaller than its own floating-point noise*: the
> same weights on the same oracle gave recall@10 0.433 in one process and 0.450 in
> another. **That retracts this project's headline claim** — see
> [Results §5](docs/RESULTS.md) and [§8.8](docs/RESULTS.md).
>
> **What the graph does earn.** With identical features, loss, schedule and a
> parameter-matched budget (1.025x), removing message passing is the **only**
> ablation whose damage exceeds the seed-noise scale: MAE +0.0028 (**+4.11x** the
> noise scale, survives) and within-scenario Spearman 0.582 to 0.402. Every other
> mechanism — bidirectional messages, edge features, one hop versus three — lands
> at 1.8x the noise scale, which this repository calls *suggestive* and not more.
> The GRU trajectory decoder is **inside noise** and is not claimed to help.
>
> **Efficiency is real but does not pay for itself here.** **11.4x** per screened
> scenario, measured (50.3 ms simulator against 4.41 ms surrogate). But once
> training *and* the dataset generation that used the very simulator being
> replaced are charged, break-even is **13,270 screened scenarios** for a single
> model and **26,167** for the shipped ensemble — roughly 290 and 570 full
> 46-candidate network sweeps. And in the fixed-compute experiment the surrogate
> **never wins at any budget, and never even ties**: at 1 s the simulator reaches
> recall 0.548 against the surrogate's 0.333, by 2 s it is at 0.952, and by 5 s it
> is exhaustive at 1.000 while the surrogate plateaus at 0.450. (The previous
> release recorded a tie at 1 s, 0.367 against 0.369. This curve is built from
> *measured* simulator throughput, which varies by up to 3x with machine load —
> the oracle sweep took 1.80 s this run against 2.62 s before — so the crossover
> point is not stable. The conclusion that the surrogate never wins is stable
> across both runs.)

---

## The one-paragraph argument

Risk in a supply network is **not a property of a node**. Take a supplier with
excellent financials, ample capacity and a clean audit history, which happens to
be the only qualified source of a component feeding four assembly plants. It is
more dangerous than a shaky supplier with three alternates. No amount of feature
engineering on the supplier's own attributes recovers that, because the relevant
fact — *no one else can make this part* — is a statement about the **group** of
suppliers around it, not about the supplier.

So this repository models input groups explicitly: suppliers **within** a group
are substitutable, groups are **complementary**. A one-member group is a hard
single point of failure; a three-member group is a soft one. In a flat supplier
table those look identical. That distinction gives a precise definition of the
thing everyone gestures at:

> **Sole-source reach** of a node: the number of demand points reachable from it
> along paths where *every* group traversed is sole-sourced.

The headline disagreement is exactly this quantity doing its work. In `shift_4`
the node the feature model ranks **last of 46** is a tier-3 distribution point
with modest throughput (63.5) and middling degree — and sole-source reach 3. It
is the worst real single point of failure in that network, at a true service loss
of 0.7430.

## Why a surrogate at all, and where that argument breaks

The simulator answers the planner's question exactly, at one run per candidate.
Screening a 54-node network is ~46 runs; every transport lane is ~150. The
surrogate turns that into one batched forward pass. The honest test is therefore
**decision quality at a fixed compute budget**: given N seconds, the simulator
evaluates a few candidates exactly while the surrogate screens all of them
approximately — which finds the true top-10 more reliably?

| compute budget (s) | simulator recall@k | surrogate recall@k |
|---|---|---|
| 0.05 | 0.069 | 0.000 |
| 0.1 | 0.099 | 0.000 |
| 0.25 | 0.156 | 0.000 |
| 0.5 | 0.271 | 0.083 |
| 1 | 0.548 | 0.333 |
| 2 | 0.952 | 0.333 |
| 5 | 1.000 | 0.450 |
| 10 | 1.000 | 0.450 |

**The surrogate does not win at any budget.** A 46-candidate sweep costs the
oracle about 1.8 s, so there is no regime at this network size where the exact
answer is unaffordable — and below the surrogate's own fixed cost it returns
nothing at all. This curve is built from *measured* wall-clock, which on this
shared machine varies by up to 3x with load, so the exact numbers move between
runs (a previous run had the two tying at 1 s, 0.367 against 0.369); the ordering
does not. **A surrogate for this task earns its keep only when the candidate
set is large enough that the oracle is genuinely out of reach, and this repository
does not demonstrate such a regime.** That is the single most important limitation
here, and it is a limitation of the experiment's scale as much as of the model.

## The distinction that governs every number here

The target is **92.48% exact zeros pooled over all 20,624 evaluated rows**
(`results/runs/base/per_item.csv`; 93.5% of training rows at the `<= 1e-4`
threshold `dataset.csv` uses) — most disruptions are absorbed, and most demand
points are unaffected by any given failure. That single fact means:

- **A good MAE is cheap, and this repository proved it the hard way.** The
  MAE-optimal constant here is **zero**, and for 43 commits the best-MAE entry in
  the results table was a boosted-tree model that had collapsed to exactly that.
  See the correction above and [Results §8.7](docs/RESULTS.md).
- **Every comparison table therefore carries `constant_zero` and
  `constant_train_mean` as first-class methods**, and a run now fails outright if
  any other method emits fewer than two distinct predictions.
- **MAE is never quoted alone.** It appears next to **MAE restricted to
  nonzero-truth rows**, which a constant cannot win — an all-zero predictor scores
  exactly the mean nonzero truth there, the worst value in the column.
- **Rank fidelity is the real test**, because the planner's question is "which
  link do I look at first", and a constant has no ranking at all (its Spearman is
  undefined, which is precisely how the collapse hid: `not measured` reads like a
  missing measurement).

Which metrics a constant can and cannot win is tabulated in
[Results §0](docs/RESULTS.md). Both families are reported, unaggregated, for every
method on every split, so a reader can see which one was achieved.

## Measured results

Every table below is rendered from a CSV by `python scripts/report_numbers.py`. Full analysis, including every retraction, is in [docs/RESULTS.md](docs/RESULTS.md).

### Method comparison, in-distribution (`results/tables/method_comparison.csv`, split `test_id`)

| method | MAE (low=good) | MAE on nonzero truth (low=good) | RMSE (low=good) | bias | Spearman pooled (high=good) | Spearman within-scenario (high=good) | scenarios contributing | top-1 agree (high=good) | scenarios contributing | unique predictions | rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| surrogate_ensemble | 0.0153 | 0.1649 | 0.0554 | -0.0030 | 0.3536 | 0.5715 | 48 | 0.5208 | 48 | 1077 | 1536 |
| surrogate_single | 0.0146 | 0.1569 | 0.0531 | -0.0025 | 0.3567 | 0.5824 | 48 | 0.5833 | 48 | 982 | 1536 |
| mlp_no_message_passing | 0.0174 | 0.2071 | 0.0685 | -0.0063 | 0.2671 | 0.4018 | 48 | 0.2292 | 48 | 206 | 1536 |
| tabular_gbt | 0.0104 | 0.0945 | 0.0366 | 0.0026 | 0.4115 | 0.6867 | 48 | 0.7292 | 48 | 766 | 1536 |
| tabular_gbt_l1 | 0.0122 | 0.2137 | 0.0696 | -0.0122 | not measured | not measured | 0 | 0.1250 | 48 | 1 | 1536 |
| tabular_ridge | 0.0246 | 0.1676 | 0.0616 | 0.0065 | 0.3526 | 0.4866 | 48 | 0.4583 | 48 | 978 | 1536 |
| retrieval_knn | 0.0106 | 0.1093 | 0.0412 | 0.0002 | 0.5078 | 0.7368 | 44 | 0.6458 | 48 | 275 | 1536 |
| topology_heuristic | 0.0240 | 0.1896 | 0.0668 | 0.0025 | 0.2321 | 0.3486 | 4 | 0.1250 | 48 | 132 | 1536 |
| constant_zero | 0.0122 | 0.2137 | 0.0696 | -0.0122 | not measured | not measured | 0 | 0.1250 | 48 | 1 | 1536 |
| constant_train_mean | 0.0236 | 0.2022 | 0.0685 | 0.0005 | not measured | not measured | 0 | 0.1250 | 48 | 1 | 1536 |

`unique predictions` is the degeneracy check: a method with 1 unique prediction is a constant function and has no ranking at all, whatever its MAE says. `MAE on nonzero truth` is the column a constant cannot win.

### MAE (lower better) — the metric a constant can win, every split (`results/tables/method_comparison.csv`)

| method | train | val | test_id | shift_topo | shift_size | shift_type | shift_multi |
|---|---|---|---|---|---|---|---|
| constant_zero | 0.012783 | 0.014508 | 0.012246 | 0.014161 | 0.014851 | 0.005691 | 0.036020 |
| constant_train_mean | 0.023994 | 0.025569 | 0.023635 | 0.024823 | 0.025871 | 0.017068 | 0.044952 |
| surrogate_single | 0.015797 | 0.017105 | 0.014581 | 0.017242 | 0.018288 | 0.009764 | 0.034292 |
| surrogate_ensemble | 0.016358 | 0.017891 | 0.015338 | 0.017647 | 0.018639 | 0.010275 | 0.035632 |
| tabular_gbt | 0.007314 | 0.010283 | 0.010418 | 0.016844 | 0.016356 | 0.007341 | 0.032162 |
| tabular_gbt_l1 | 0.012783 | 0.014508 | 0.012246 | 0.014161 | 0.014851 | 0.005691 | 0.036020 |

### MAE on nonzero-truth rows (lower better) — the metric a constant cannot win, every split (`results/tables/method_comparison.csv`)

| method | train | val | test_id | shift_topo | shift_size | shift_type | shift_multi |
|---|---|---|---|---|---|---|---|
| constant_zero | 0.195408 | 0.206342 | 0.213741 | 0.159089 | 0.202225 | 0.087517 | 0.230064 |
| constant_train_mean | 0.184167 | 0.194623 | 0.202197 | 0.148047 | 0.191006 | 0.078678 | 0.218248 |
| surrogate_single | 0.152904 | 0.169176 | 0.156903 | 0.130552 | 0.160015 | 0.081104 | 0.176389 |
| surrogate_ensemble | 0.158817 | 0.174418 | 0.164948 | 0.134284 | 0.162454 | 0.080625 | 0.184142 |
| tabular_gbt | 0.060959 | 0.083832 | 0.094543 | 0.111176 | 0.118720 | 0.073561 | 0.163343 |
| tabular_gbt_l1 | 0.195408 | 0.206342 | 0.213741 | 0.159089 | 0.202225 | 0.087517 | 0.230064 |

`surrogate_single` has a **worse** MAE than `constant_zero` on 6 of 7 splits (train, val, test_id, shift_topo, shift_size, shift_type).

It beats `constant_train_mean` on 7 of 7 splits (train, val, test_id, shift_topo, shift_size, shift_type, shift_multi).

Methods emitting a single distinct prediction on at least one split (constant functions, not models): `constant_train_mean`, `constant_zero`, `tabular_gbt_l1`.

### MAE (lower better) across every split (`results/tables/method_comparison.csv`)

| method | test_id | shift_topo | shift_size | shift_type | shift_multi |
|---|---|---|---|---|---|
| constant_train_mean | 0.0236 | 0.0248 | 0.0259 | 0.0171 | 0.0450 |
| constant_zero | 0.0122 | 0.0142 | 0.0149 | 0.0057 | 0.0360 |
| mlp_no_message_passing | 0.0174 | 0.0190 | 0.0196 | 0.0109 | 0.0402 |
| retrieval_knn | 0.0106 | 0.0179 | 0.0177 | 0.0090 | 0.0337 |
| surrogate_ensemble | 0.0153 | 0.0176 | 0.0186 | 0.0103 | 0.0356 |
| surrogate_single | 0.0146 | 0.0172 | 0.0183 | 0.0098 | 0.0343 |
| tabular_gbt | 0.0104 | 0.0168 | 0.0164 | 0.0073 | 0.0322 |
| tabular_gbt_l1 | 0.0122 | 0.0142 | 0.0149 | 0.0057 | 0.0360 |
| tabular_ridge | 0.0246 | 0.0243 | 0.0148 | 0.0368 | 0.0411 |
| topology_heuristic | 0.0240 | 0.0259 | 0.0249 | 0.0067 | 0.0434 |

### MAE on nonzero-truth rows only (lower better) across every split (`results/tables/method_comparison.csv`)

| method | test_id | shift_topo | shift_size | shift_type | shift_multi |
|---|---|---|---|---|---|
| constant_train_mean | 0.2022 | 0.1480 | 0.1910 | 0.0787 | 0.2182 |
| constant_zero | 0.2137 | 0.1591 | 0.2022 | 0.0875 | 0.2301 |
| mlp_no_message_passing | 0.2071 | 0.1529 | 0.1953 | 0.0802 | 0.2231 |
| retrieval_knn | 0.1093 | 0.1272 | 0.1721 | 0.0718 | 0.1902 |
| surrogate_ensemble | 0.1649 | 0.1343 | 0.1625 | 0.0806 | 0.1841 |
| surrogate_single | 0.1569 | 0.1306 | 0.1600 | 0.0811 | 0.1764 |
| tabular_gbt | 0.0945 | 0.1112 | 0.1187 | 0.0736 | 0.1633 |
| tabular_gbt_l1 | 0.2137 | 0.1591 | 0.2022 | 0.0875 | 0.2301 |
| tabular_ridge | 0.1676 | 0.1219 | 0.1904 | 0.0816 | 0.1934 |
| topology_heuristic | 0.1896 | 0.1396 | 0.1692 | 0.0833 | 0.2171 |

### within-scenario Spearman (higher better) across every split (`results/tables/method_comparison.csv`)

| method | test_id | shift_topo | shift_size | shift_type | shift_multi |
|---|---|---|---|---|---|
| mlp_no_message_passing | 0.4018 | 0.5008 | 0.3703 | 0.4853 | 0.4841 |
| retrieval_knn | 0.7368 | 0.5879 | 0.4527 | 0.5166 | 0.5283 |
| surrogate_ensemble | 0.5715 | 0.6340 | 0.5059 | 0.5029 | 0.6269 |
| surrogate_single | 0.5824 | 0.6629 | 0.5141 | 0.5289 | 0.6536 |
| tabular_gbt | 0.6867 | 0.5941 | 0.5182 | 0.2883 | 0.5818 |
| tabular_ridge | 0.4866 | 0.5634 | 0.3656 | 0.4914 | 0.5911 |
| topology_heuristic | 0.3486 | 0.2179 | 0.2402 | 0.5047 | -0.4245 |

### Every claimed gain against the run-to-run noise scale

A difference between two single runs is inside noise unless it exceeds `sqrt(2) * sd` for that metric, from `seed_variance.csv`.

The surrogate is compared against **four** references: the reference-style feature model it is meant to replace, whichever baseline is actually strongest on that split, and both trivial constants. Comparing only against the reference would flatter it, and on a 92.5%-zero target omitting the constants hides the only comparison that establishes whether anything was learned at all.

| split | metric | comparison | baseline | surrogate | baseline_value | gain | noise_scale | verdict |
|---|---|---|---|---|---|---|---|---|
| test_id | mae | vs reference | tabular_gbt | 0.01534 | 0.01042 | -0.00492 | 0.00069 | -7.17 -> **worse, outside noise** |
| test_id | mae | vs best baseline | tabular_gbt | 0.01534 | 0.01042 | -0.00492 | 0.00069 | -7.17 -> **worse, outside noise** |
| test_id | mae | vs constant zero | constant_zero | 0.01534 | 0.01225 | -0.00309 | 0.00069 | -4.51 -> **worse, outside noise** |
| test_id | mae | vs constant train-mean | constant_train_mean | 0.01534 | 0.02363 | 0.00830 | 0.00069 | +12.10 -> survives |
| test_id | mae_nonzero_truth | vs reference | tabular_gbt | 0.16495 | 0.09454 | -0.07041 | 0.02170 | -3.24 -> **worse, outside noise** |
| test_id | mae_nonzero_truth | vs best baseline | tabular_gbt | 0.16495 | 0.09454 | -0.07041 | 0.02170 | -3.24 -> **worse, outside noise** |
| test_id | mae_nonzero_truth | vs constant zero | constant_zero | 0.16495 | 0.21374 | 0.04879 | 0.02170 | +2.25 -> survives |
| test_id | mae_nonzero_truth | vs constant train-mean | constant_train_mean | 0.16495 | 0.20220 | 0.03725 | 0.02170 | +1.72 -> suggestive |
| test_id | spearman_within_scenario | vs reference | tabular_gbt | 0.57147 | 0.68668 | -0.11522 | 0.04288 | -2.69 -> **worse, outside noise** |
| test_id | spearman_within_scenario | vs best baseline | retrieval_knn | 0.57147 | 0.73677 | -0.16530 | 0.04288 | -3.86 -> **worse, outside noise** |
| shift_topo | mae | vs reference | tabular_gbt | 0.01765 | 0.01684 | -0.00080 | 0.00052 | -1.54 -> worse (suggestive) |
| shift_topo | mae | vs best baseline | tabular_gbt | 0.01765 | 0.01684 | -0.00080 | 0.00052 | -1.54 -> worse (suggestive) |
| shift_topo | mae | vs constant zero | constant_zero | 0.01765 | 0.01416 | -0.00349 | 0.00052 | -6.69 -> **worse, outside noise** |
| shift_topo | mae | vs constant train-mean | constant_train_mean | 0.01765 | 0.02482 | 0.00718 | 0.00052 | +13.77 -> survives |
| shift_topo | mae_nonzero_truth | vs reference | tabular_gbt | 0.13428 | 0.11118 | -0.02311 | 0.00954 | -2.42 -> **worse, outside noise** |
| shift_topo | mae_nonzero_truth | vs best baseline | tabular_gbt | 0.13428 | 0.11118 | -0.02311 | 0.00954 | -2.42 -> **worse, outside noise** |
| shift_topo | mae_nonzero_truth | vs constant zero | constant_zero | 0.13428 | 0.15909 | 0.02481 | 0.00954 | +2.60 -> survives |
| shift_topo | mae_nonzero_truth | vs constant train-mean | constant_train_mean | 0.13428 | 0.14805 | 0.01376 | 0.00954 | +1.44 -> suggestive |
| shift_topo | spearman_within_scenario | vs reference | tabular_gbt | 0.63398 | 0.59410 | 0.03988 | 0.03430 | +1.16 -> suggestive |
| shift_topo | spearman_within_scenario | vs best baseline | tabular_gbt | 0.63398 | 0.59410 | 0.03988 | 0.03430 | +1.16 -> suggestive |
| shift_size | mae | vs reference | tabular_gbt | 0.01864 | 0.01636 | -0.00228 | 0.00018 | -13.03 -> **worse, outside noise** |
| shift_size | mae | vs best baseline | tabular_ridge | 0.01864 | 0.01478 | -0.00386 | 0.00018 | -22.00 -> **worse, outside noise** |
| shift_size | mae | vs constant zero | constant_zero | 0.01864 | 0.01485 | -0.00379 | 0.00018 | -21.61 -> **worse, outside noise** |
| shift_size | mae | vs constant train-mean | constant_train_mean | 0.01864 | 0.02587 | 0.00723 | 0.00018 | +41.26 -> survives |
| shift_size | mae_nonzero_truth | vs reference | tabular_gbt | 0.16245 | 0.11872 | -0.04373 | 0.01510 | -2.90 -> **worse, outside noise** |
| shift_size | mae_nonzero_truth | vs best baseline | tabular_gbt | 0.16245 | 0.11872 | -0.04373 | 0.01510 | -2.90 -> **worse, outside noise** |
| shift_size | mae_nonzero_truth | vs constant zero | constant_zero | 0.16245 | 0.20222 | 0.03977 | 0.01510 | +2.63 -> survives |
| shift_size | mae_nonzero_truth | vs constant train-mean | constant_train_mean | 0.16245 | 0.19101 | 0.02855 | 0.01510 | +1.89 -> suggestive |
| shift_size | spearman_within_scenario | vs reference | tabular_gbt | 0.50594 | 0.51816 | -0.01222 | 0.02057 | -0.59 -> inside noise |
| shift_size | spearman_within_scenario | vs best baseline | tabular_gbt | 0.50594 | 0.51816 | -0.01222 | 0.02057 | -0.59 -> inside noise |
| shift_type | mae | vs reference | tabular_gbt | 0.01027 | 0.00734 | -0.00293 | 0.00044 | -6.73 -> **worse, outside noise** |
| shift_type | mae | vs best baseline | topology_heuristic | 0.01027 | 0.00669 | -0.00359 | 0.00044 | -8.23 -> **worse, outside noise** |
| shift_type | mae | vs constant zero | constant_zero | 0.01027 | 0.00569 | -0.00458 | 0.00044 | -10.51 -> **worse, outside noise** |
| shift_type | mae | vs constant train-mean | constant_train_mean | 0.01027 | 0.01707 | 0.00679 | 0.00044 | +15.58 -> survives |
| shift_type | mae_nonzero_truth | vs reference | tabular_gbt | 0.08063 | 0.07356 | -0.00706 | 0.00111 | -6.35 -> **worse, outside noise** |
| shift_type | mae_nonzero_truth | vs best baseline | retrieval_knn | 0.08063 | 0.07178 | -0.00885 | 0.00111 | -7.95 -> **worse, outside noise** |
| shift_type | mae_nonzero_truth | vs constant zero | constant_zero | 0.08063 | 0.08752 | 0.00689 | 0.00111 | +6.20 -> survives |
| shift_type | mae_nonzero_truth | vs constant train-mean | constant_train_mean | 0.08063 | 0.07868 | -0.00195 | 0.00111 | -1.75 -> worse (suggestive) |
| shift_type | spearman_within_scenario | vs reference | tabular_gbt | 0.50293 | 0.28829 | 0.21464 | 0.02411 | +8.90 -> survives |
| shift_type | spearman_within_scenario | vs best baseline | retrieval_knn | 0.50293 | 0.51658 | -0.01364 | 0.02411 | -0.57 -> inside noise |
| shift_multi | mae | vs reference | tabular_gbt | 0.03563 | 0.03216 | -0.00347 | 0.00254 | -1.37 -> worse (suggestive) |
| shift_multi | mae | vs best baseline | tabular_gbt | 0.03563 | 0.03216 | -0.00347 | 0.00254 | -1.37 -> worse (suggestive) |
| shift_multi | mae | vs constant zero | constant_zero | 0.03563 | 0.03602 | 0.00039 | 0.00254 | +0.15 -> inside noise |
| shift_multi | mae | vs constant train-mean | constant_train_mean | 0.03563 | 0.04495 | 0.00932 | 0.00254 | +3.67 -> survives |
| shift_multi | mae_nonzero_truth | vs reference | tabular_gbt | 0.18414 | 0.16334 | -0.02080 | 0.01948 | -1.07 -> worse (suggestive) |
| shift_multi | mae_nonzero_truth | vs best baseline | tabular_gbt | 0.18414 | 0.16334 | -0.02080 | 0.01948 | -1.07 -> worse (suggestive) |
| shift_multi | mae_nonzero_truth | vs constant zero | constant_zero | 0.18414 | 0.23006 | 0.04592 | 0.01948 | +2.36 -> survives |
| shift_multi | mae_nonzero_truth | vs constant train-mean | constant_train_mean | 0.18414 | 0.21825 | 0.03411 | 0.01948 | +1.75 -> suggestive |
| shift_multi | spearman_within_scenario | vs reference | tabular_gbt | 0.62692 | 0.58178 | 0.04514 | 0.05526 | +0.82 -> inside noise |
| shift_multi | spearman_within_scenario | vs best baseline | tabular_ridge | 0.62692 | 0.59108 | 0.03583 | 0.05526 | +0.65 -> inside noise |

### Counterfactual criticality vs the simulator oracle (`results/tables/criticality_summary.csv`)

| method | recall@5 | precision@5 | regret_frac@5 | recall@10 | precision@10 | regret_frac@10 | recall@20 | precision@20 | regret_frac@20 | spearman_full | score_std | distinct_scores | method_seconds | sim_seconds | speedup_vs_simulator |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| surrogate | 0.3667 | 0.6000 | 0.4875 | 0.4500 | 0.5500 | 0.4332 | 0.4914 | 0.4333 | 0.3173 | 0.2609 | 0.0008 | 23.8333 | 0.9241 | 1.8048 | 2.6587 |
| tabular_gbt | 0.1000 | 0.3667 | 0.8551 | 0.1833 | 0.2333 | 0.8667 | 0.2353 | 0.2583 | 0.7886 | 0.0355 | 0.0496 | 7.3333 | 0.0534 | 1.8048 | 35.3687 |
| tabular_gbt_l1 | 0.0000 | 0.1000 | 0.9943 | 0.0500 | 0.1667 | 0.9791 | 0.1660 | 0.2583 | 0.7724 | not measured | 0.0000 | 1.0000 | 0.0024 | 1.8048 | 733.1260 |
| tabular_ridge | 0.2000 | 0.6667 | 0.7029 | 0.4167 | 0.6167 | 0.5457 | 0.5749 | 0.5083 | 0.3495 | 0.4324 | 0.0231 | 37.1667 | 0.0007 | 1.8048 | 2499.7800 |
| topology_heuristic | 0.4333 | 0.7667 | 0.4248 | 0.6333 | 0.7500 | 0.1613 | 0.7056 | 0.5667 | 0.0932 | 0.5683 | 1.4484 | 41.3333 | 0.0029 | 1.8048 | 638.0260 |

**`tabular_gbt_l1` produced a constant score for every candidate**, i.e. one distinct score across all candidates in all networks. That is not a fact about tabular features or about topology; it is a fact about an L1 objective on a 92.5%-zero target, where the loss-minimising constant is the median and the median is exactly 0. Its recall@k is therefore whatever a tie-broken arbitrary ordering scores. It is reported here, labelled, because this repository previously shipped it *as* `tabular_gbt` - the reference model the surrogate was said to lose to - and the collapse was invisible in every metric except this one. The squared-error variant next to it is the boosted-tree reference that actually ranks.

### Where the feature score and the counterfactual disagree (`results/tables/disagreements.csv`)

36 disagreeing pairs found. Simulator verdict: **counterfactual** 36.

| network | A | A rank (feat) | A rank (cf) | A true loss | B | B rank (feat) | B rank (cf) | B true loss | winner | margin |
|---|---|---|---|---|---|---|---|---|---|---|
| shift_4 | 0 | 10 | 16 | 0.0000 | 45 | 46 | 1 | 0.7430 | counterfactual | 0.7430 |
| shift_4 | 9 | 9 | 25 | 0.0000 | 45 | 46 | 1 | 0.7430 | counterfactual | 0.7430 |
| shift_4 | 12 | 3 | 27 | 0.0000 | 45 | 46 | 1 | 0.7430 | counterfactual | 0.7430 |
| shift_4 | 16 | 8 | 13 | 0.0000 | 45 | 46 | 1 | 0.7430 | counterfactual | 0.7430 |
| shift_4 | 18 | 1 | 31 | 0.0000 | 45 | 46 | 1 | 0.7430 | counterfactual | 0.7430 |

Worked example of the largest disagreement:

> node 0: tier 0, 4 customers, reaches 6 demand points (0 sole-sourced), throughput 158.9, capacity slack 0.15
> 
> node 45: tier 3, 7 customers, reaches 7 demand points (3 sole-sourced), throughput 63.5, capacity slack 0.41

### Measured wall-clock (`results/tables/efficiency.csv`)

| component | variant | n_nodes | median_ms | iqr_ms | per_item_ms | params |
|---|---|---|---|---|---|---|
| simulator | shift_size | 108.0000 | 108.4670 | 9.9628 | 108.4670 | not measured |
| simulator | shift_topo | 54.0000 | 50.3284 | 5.7636 | 50.3284 | not measured |
| surrogate_batched_all_candidates | shift_size | 108.0000 | 589.8935 | 105.2922 | 6.4119 | 55752.0000 |
| surrogate_batched_all_candidates | shift_topo | 54.0000 | 202.9000 | 20.3435 | 4.4109 | 55752.0000 |
| surrogate_single | shift_size | 108.0000 | 51.8127 | 30.3842 | 51.8127 | 55752.0000 |
| surrogate_single | shift_topo | 54.0000 | 27.8575 | 5.8990 | 27.8575 | 55752.0000 |

Warm-up ≥ 8 iterations, ≥ 25 timed repeats, median and IQR reported.

### Break-even accounting (`results/tables/break_even.csv`)

| quantity | value |
|---|---|
| `sim_ms_per_scenario` | 50.328 |
| `surrogate_ms_per_scenario` | 4.411 |
| `speedup_per_scenario` | 11.410 |
| `train_seconds` | 296.085 |
| `ensemble_train_seconds` | 888.256 |
| `dataset_seconds` | 313.247 |
| `setup_seconds` | 609.332 |
| `break_even_scenarios` | 13270.200 |
| `break_even_scenarios_ensemble` | 26166.600 |

Setup cost includes the dataset generation that used the very simulator being replaced.

### Decision quality at a fixed compute budget (`results/tables/budget_curve.csv`)

| budget_s | simulator | surrogate |
|---|---|---|
| 0.050 | 0.069 | 0.000 |
| 0.100 | 0.099 | 0.000 |
| 0.250 | 0.156 | 0.000 |
| 0.500 | 0.271 | 0.083 |
| 1.000 | 0.548 | 0.333 |
| 2.000 | 0.952 | 0.333 |
| 5.000 | 1.000 | 0.450 |
| 10.000 | 1.000 | 0.450 |

Recall@10 of the true critical set, averaged over the evaluated networks.

## What this is not

- **Not validated on a real supply network.** Everything here is synthetic by
  design. That is a real strength — the simulator *is* the data source, so labels
  are exact, the counterfactual is available for every candidate rather than only
  for events that happened, and the distribution can be shifted deliberately along
  named axes. It is also a real limitation: results transfer to a real network
  only insofar as this simulator's mechanics are right. No real-world validation
  is claimed anywhere in this repository.
- **Not an inventory optimiser.** Base-stock levels are given, not optimised. This
  is a propagation model.
- **Not a probability model.** Following Simchi-Levi et al. (2015), disruptions are
  *scenarios*, not events with estimated likelihoods. The output is "if this fails,
  here is what happens", not "this is how likely it is to fail".

## Install

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python ./.venv/Scripts/python.exe \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch
uv pip install --python ./.venv/Scripts/python.exe \
  numpy scipy pandas pyyaml matplotlib scikit-learn pytest ruff
uv pip install --python ./.venv/Scripts/python.exe -e . --no-deps
```

or `make setup`. Python 3.12 specifically — the 3.14 torch wheels in this
environment fail to import on a missing bundled `torchgen`.

**No network access at run time. No API keys. No dataset downloads.** The
simulator generates everything.

## 60-second quickstart

```bash
make smoke          # the entire pipeline end to end on a tiny config
```

```python
from sndsur.config import load_config
from sndsur.data.network import NetworkSpec, generate_network
from sndsur.data.disruptions import Disruption, DisruptionSet
from sndsur.sim.simulator import SimConfig, counterfactual

net = generate_network(NetworkSpec(), seed=0)
net.compute_throughput()

# Which node is the worst single point of failure?
ss = net.sole_source_reach()
target = int(ss.argmax())

# What actually happens if it goes down for 8 periods?
cfg = SimConfig(warmup=22, horizon=30)
ds = DisruptionSet([Disruption("supplier_outage", target, -1, 4, 8, 1.0)]).bind(net)
imp = counterfactual(net, cfg, ds, seed=1)

print(f"total service loss   {imp.total_service_loss:.4f}")
print(f"per demand point     {imp.service_loss.round(3)}")
print(f"time to impact       {imp.time_to_impact}")     # NaN where never affected
print(f"recovery time        {imp.recovery_time}")
```

Full experiment matrix:

```bash
make all            # data -> seeds -> compare -> ablate -> criticality -> efficiency -> figures
```

Order matters: `compare` trains and caches the ensemble that `criticality` and
`efficiency` reuse, and **`seeds` should be read before any comparison** because
it produces the noise scale that every claimed improvement is measured against.

## Repository map

```
src/sndsur/
  config.py              typed config, YAML `_base_` inheritance, --set overrides
  data/
    network.py           SupplyNetwork, input groups, topology generator
    disruptions.py       six mechanisms + the scenario sampler
    features.py          node / edge / reference-style-tabular feature views
    scenarios.py         scenario dataset and the seven splits
    csvio.py             optional loader for a user-supplied network
  sim/simulator.py       THE GROUND TRUTH: discrete-time multi-echelon simulator
  models/
    layers.py            hand-rolled scatter message passing (no torch-geometric)
    surrogate.py         encoder-processor-decoder + four heads
    baselines.py         tabular GBT/ridge, topology heuristic, kNN retrieval
  metrics/
    fidelity.py          magnitude vs rank fidelity, kept separate on purpose
    ranking.py           critical-set recall@k, decision regret
    calibration.py       coverage, sharpness, sparsification, AUSE
    stats.py             bootstrap, Wilcoxon, Holm-Bonferroni, noise scale
  engine/                batching, losses, training loop
  analysis/              counterfactual criticality and the disagreement experiment
  utils/                 seeding, logging, benchmarking, checkpoint cache
  pipelines.py           the orchestration every script wraps
configs/                 base.yaml + smoke.yaml + one per variant
scripts/                 thin argparse wrappers; report_numbers.py renders the docs
tests/                   simulator, network, metrics, config, models, data, e2e
docs/                    METHOD.md, RESULTS.md, REPRODUCIBILITY.md
results/tables/          every number in the docs comes from one of these CSVs
results/figures/         regenerated from those CSVs by `make figures`
notebooks/               5 notebooks, executed, outputs committed
```

## Design decisions worth arguing with

**Hand-rolled message passing rather than torch-geometric or DGL.** Both compile
against a specific torch/CUDA pair, and a reader running `uv sync` on a different
torch build gets a linker error instead of a result. What this project needs is a
gather, an MLP and an `index_add_` — about forty lines. Forty lines for a
reproducibility guarantee is a good trade for a repository whose whole point is
that its numbers can be re-derived.

**Two message functions, not one.** Material and therefore shortage flows
supplier → customer; orders and therefore requirement flow customer → supplier.
Different physics, different weights. This is also why `demand_spike` is the
held-out mechanism: it is the only intervention that travels purely upstream.

**Mean and max aggregation.** A shortage is a bottleneck — the binding constraint
on production is the worst input, not the average one. Mean pooling alone cannot
express that.

**The no-message-passing ablation is parameter-matched.** The first version simply
deleted the processor and came out 3.6× smaller, which would have confounded "the
graph helps" with "more parameters help". The node-only trunk is widened to match.

**The heuristic baseline gets a two-parameter affine calibration** fitted on train
only, because a sum of standardised topology signals has an arbitrary scale and
its uncalibrated MAE (2.4 against a target bounded near 1) said nothing about the
heuristic and everything about its units.

**The tabular baseline is treated generously — and the one place that generosity
was implemented wrongly cost this repository its headline evidence.** It gets three
graph-derived columns it would not normally have. It *used* to be trained with
absolute error "to match the surrogate's objective", on the reasoning that
otherwise the comparison would be about loss functions rather than
representations. That reasoning is sound in general and wrong on a 92.5%-zero
target, where L1's optimal constant is the median and the median is 0: the model
collapsed to an exact constant. The working GBT uses squared error; the
absolute-error variant is kept as `tabular_gbt_l1`, labelled degenerate by
construction. See [Results §8.7](docs/RESULTS.md).

The interesting asymmetry is that the *same* objective on the *same* target only
blunts the neural network's global scale (see §5) while it destroys the trees
outright. A gradient step on a shared parameter vector still moves the whole
function; a tree leaf takes the exact L1 minimiser of the samples that land in it,
and that minimiser is 0.

**Both trivial constants are permanent rows in the results tables.** Not padding:
on a target this zero-inflated, a table that does not say what predicting nothing
scores is not interpretable, and the absence of that row is the single reason the
collapse above survived 43 commits. A run now fails outright if any method not
declared constant-by-design emits fewer than two distinct predictions.

## Citations

- Kipf & Welling (2017), *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR.
- Veličković et al. (2018), *Graph Attention Networks*, ICLR.
- Gilmer et al. (2017), *Neural Message Passing for Quantum Chemistry*, ICML.
- Battaglia et al. (2018), *Relational Inductive Biases, Deep Learning, and Graph Networks*, arXiv:1806.01261.
- Sanchez-Gonzalez et al. (2020), *Learning to Simulate Complex Physics with Graph Networks*, ICML.
- Pfaff et al. (2021), *Learning Mesh-Based Simulation with Graph Networks*, ICLR.
- Kosasih & Brintrup (2021), *A Machine Learning Approach for Predicting Hidden Links in Supply Chain with Graph Neural Networks*, IJPR.
- Ivanov & Dolgui (2020), *Viability of Intertwined Supply Networks*, IJPR 58(10).
- Simchi-Levi, Schmidt, Wei et al. (2015), *Identifying Risks and Mitigating Disruptions in the Automotive Supply Chain*, Interfaces 45(5).
- Lakshminarayanan, Pritzel & Blundell (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, NeurIPS.
- Nix & Weigend (1994), *Estimating the Mean and Variance of the Target Probability Distribution*, IEEE ICNN.

## License

MIT. See [LICENSE](LICENSE).
