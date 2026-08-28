# Results

Every table below is rendered from a CSV in `results/tables/` by
`scripts/report_numbers.py`. No number here was typed by hand. Where a cell reads
`not measured`, no run stands behind it and none was invented.

**Read section 2 before section 3.** The seed study is what makes the comparison
interpretable; without it a table of differences is just a table of differences.

---

## Summary of what held and what did not

| claim | verdict |
|---|---|
| Counterfactual ranking beats feature-based risk scoring at finding true single points of failure | **holds, decisively** — 36 of 36 adjudicated disagreements go to the counterfactual |
| Message passing is doing the work | **holds** — the only ablation whose damage clears the noise scale (+4.11x); within-scenario Spearman 0.582 to 0.402 |
| The surrogate generalises better than nearest-neighbour retrieval under topology shift | **one axis survives** — retrieval wins in-distribution (-3.86x, worse outside noise); the surrogate wins on `shift_size` (+2.59x, survives); `shift_topo`, `shift_type` and `shift_multi` do not clear the noise scale |
| The surrogate predicts impact *magnitude* well | **retracted, and re-retracted for a better reason** — see §8.7. The earlier retraction credited a "reference GBT" that was the constant zero function. With a working GBT the surrogate is worse on MAE on all five evaluation splits (-1.4x to -13.0x the noise scale) *and* worse on MAE restricted to rows where a disruption bit (-1.1x to -6.4x) |
| The surrogate is better than predicting nothing | **half** — it is beaten on pooled MAE by the **constant zero function** on 6 of 7 splits, because on a 92.5%-zero target zero is the MAE-optimal constant. On MAE restricted to nonzero-truth rows, which a constant cannot win, it beats the all-zero constant on all 7 splits (+2.25x to +6.20x the noise scale) and the train-mean constant on 4 of 5 evaluation splits |
| The surrogate transfers better than the tabular model to an unseen disruption mechanism | **survives, and was previously unmeasurable** — on `shift_type` its within-scenario Spearman is 0.503 against the GBT's 0.288, **+8.90x** the noise scale. The old table could not report this at all: the broken baseline's rank correlation was undefined |
| The learned surrogate is the best criticality ranker | **retracted** — a five-line topology heuristic beats it on recall@k, regret and rank correlation |
| The surrogate's predictive intervals are calibrated | **fails** — grossly over-covered (0.97 empirical at nominal 0.50) |
| Uncertainty tracks error | **holds within a split** (error-detection AUROC 0.99), **fails across splits** (mean sigma barely moves under shift) |
| The deep ensemble contributes epistemic uncertainty | **fails** — the epistemic term is ~4% of total predictive sigma |
| The surrogate wins on decision quality at a fixed compute budget | **fails at every budget tested, and does not even tie** — at 1 s the simulator reaches 0.548 against 0.333, and by 5 s exhaustive search is at 1.000 while the surrogate plateaus at 0.450. The crossover point is load-dependent (a previous run tied at 1 s); the ordering is not |
| The surrogate is faster per scenario | **holds** — 11.4x measured, but break-even is 13,270 screened scenarios (26,167 for the ensemble) once dataset generation is charged |
| The surrogate's criticality ranking is reproducible | **fails** — its criticality score spread is 0.0008 while its per-row predictions move by up to 1.3e-03 between process launches (parallel scatter reductions), so recall@10 moved from 0.433 to 0.450 with identical weights and a bit-identical oracle. See §8.8 |

---

## 0. Which metrics a constant can win, and which it cannot

The target is **92.48% exact zeros pooled over all 20,624 evaluated rows**
(19,073 of them; `results/runs/base/per_item.csv`), and 93.5% of training rows at
the `<= 1e-4` threshold `dataset.csv` uses. Most disruptions are absorbed and most
demand points are unaffected by any given failure. That single fact governs how
every error number in this document must be read, and getting it wrong is how this
repository shipped a constant as its best-performing baseline (§8.7).

So the table below comes first, before any result. Every metric here was checked
against the two trivial predictors — **`constant_zero`**, which emits 0.0 for
every row, and **`constant_train_mean`**, which emits 0.012783 — both of which are
run as first-class methods on every split.

| metric | can a constant win it? | why |
|---|---|---|
| **MAE** (pooled) | **yes, and `constant_zero` does** | The L1-optimal constant is the median, and the median of this target is exactly 0. `constant_zero` beats the surrogate on 6 of 7 splits. |
| **RMSE** | no | The L2-optimal constant is the mean; `constant_train_mean` scores 0.0685 on `test_id` against the surrogate's 0.0531. |
| **MAE on nonzero-truth rows** | **no** | A zero predictor scores exactly the mean nonzero truth (0.2137 on `test_id`), the worst value in the column. This is the magnitude metric to read. |
| **Spearman** (pooled or within-scenario) | no — it is *undefined* | A constant has no ordering. Reported as `not measured`, which is why the collapse hid for so long: it looks like a missing measurement. |
| **top-1 agreement** | **partly** | A constant scores 0.1250 on `test_id` purely because `np.argmax` returns index 0 on an all-tied row, and demand point 0 is worst-hit 12.5% of the time. Not zero, and not evidence of anything. |
| **paired Wilcoxon on absolute errors** | **yes** | It counts rows, not magnitudes. `constant_zero` is worse than the working GBT on only **4.5% of rows** with a median paired difference of exactly 0, while being clearly worse on the mean. See §3. |
| **recall@k / regret** on the criticality sweep | **partly** | A constant score gets whatever an arbitrary tie-broken ordering scores: `tabular_gbt_l1` posts recall@10 = 0.05 and regret 0.979 that way. |

Two consequences for how the rest of this document is written:

* **A good pooled MAE is cheap and is never quoted alone.** It is always shown
  next to `mae_nonzero_truth` and next to the rank metrics.
* **Rank fidelity is the real test**, because the planner's question is "which link
  should I look at first". A constant cannot win it at all.

Both families are reported for every method on every split, unaggregated, so the
reader can see which one was achieved. A single "accuracy" number would hide the
distinction, and in this task the distinction is the whole point.

`spearman_within_scenario` is computed **within** each scenario and then averaged,
never pooled. Pooling lets the easy between-scenario signal (big disruptions hurt
more than small ones) carry the number; the question of interest is which demand
point inside a given scenario is worst hit. The "scenarios contributing" columns
matter for the same reason: a within-scenario correlation is undefined when every
demand point in a scenario is unaffected, and most scenarios are.

---

## 1. The data

### Dataset (`results/tables/dataset.csv`)

| split | scenarios | rows | networks | nodes | mean loss | p90 | max | zero-impact rows | disruptions/scenario |
|---|---|---|---|---|---|---|---|---|---|
| train | 1200 | 9600 | 12 | 54 | 0.0128 | 0.0000 | 0.8016 | 0.9347 | 1.0000 |
| val | 192 | 1536 | 12 | 54 | 0.0145 | 0.0000 | 0.8257 | 0.9297 | 1.0000 |
| test_id | 192 | 1536 | 12 | 54 | 0.0122 | 0.0000 | 0.7786 | 0.9427 | 1.0000 |
| shift_topo | 198 | 1584 | 6 | 54 | 0.0142 | 0.0000 | 0.7358 | 0.9129 | 1.0000 |
| shift_size | 200 | 3200 | 4 | 108 | 0.0149 | 0.0000 | 0.7338 | 0.9266 | 1.0000 |
| shift_type | 198 | 1584 | 6 | 54 | 0.0057 | 0.0000 | 0.7477 | 0.9356 | 1.0000 |
| shift_multi | 198 | 1584 | 6 | 54 | 0.0360 | 0.1146 | 0.8428 | 0.8434 | 2.5101 |

Generation: 313.2 s for 5140 scenarios (the experiment uses a deterministic subsample of those).

The zero-inflation is not a defect of the generator; it is what a supply network
does. Capacity slack, inventory buffers and multi-sourcing absorb most single
disruptions, and the ones that get through are the interesting ones. `shift_multi`
has visibly more impact (mean 0.0360 against 0.0122 for
`test_id`) because two or three simultaneous failures are much harder to absorb
than one.

---

## 2. Seed variance — the noise scale

Three training runs of the **identical** configuration, differing only in seed.
The run-to-run scale of a difference between two single runs is `sqrt(2) * sd`;
any "improvement" smaller than that is noise.

### Seed variance (`results/tables/seed_variance.csv`)

**test_id**

| metric | seeds | mean | sd | min | max | range | noise scale (sqrt2*sd) |
|---|---|---|---|---|---|---|---|
| mae | 3 | 0.01509 | 0.00048 | 0.01461 | 0.01558 | 0.00097 | 0.00068 |
| rmse | 3 | 0.05850 | 0.00479 | 0.05354 | 0.06311 | 0.00957 | 0.00678 |
| spearman_pooled | 3 | 0.35550 | 0.00198 | 0.35330 | 0.35714 | 0.00384 | 0.00280 |
| spearman_within_scenario | 3 | 0.58025 | 0.03032 | 0.54911 | 0.60968 | 0.06057 | 0.04288 |
| top1_agreement | 3 | 0.52778 | 0.09845 | 0.41667 | 0.60417 | 0.18750 | 0.13924 |

**shift_topo**

| metric | seeds | mean | sd | min | max | range | noise scale (sqrt2*sd) |
|---|---|---|---|---|---|---|---|
| mae | 3 | 0.01750 | 0.00037 | 0.01722 | 0.01792 | 0.00070 | 0.00052 |
| rmse | 3 | 0.05858 | 0.00251 | 0.05619 | 0.06121 | 0.00501 | 0.00356 |
| spearman_pooled | 3 | 0.40523 | 0.01129 | 0.39744 | 0.41817 | 0.02074 | 0.01596 |
| spearman_within_scenario | 3 | 0.63799 | 0.02305 | 0.61437 | 0.66041 | 0.04605 | 0.03259 |
| top1_agreement | 3 | 0.70417 | 0.03819 | 0.66250 | 0.73750 | 0.07500 | 0.05401 |

Per-seed values (`results/tables/seed_runs.csv`, split `test_id`):

| metric | 0 | 7 | 1337 |
|---|---|---|---|
| mae | 0.01461 | 0.01509 | 0.01558 |
| rmse | 0.05354 | 0.05884 | 0.06311 |
| spearman_pooled | 0.35605 | 0.35714 | 0.35330 |
| spearman_within_scenario | 0.58196 | 0.60968 | 0.54911 |
| top1_agreement | 0.56250 | 0.60417 | 0.41667 |

---

## 3. Method comparison

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

### Generalisation along each shift axis

Each split differs from `test_id` in exactly one respect.

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

### What this table says, in order of importance

**The reference model beats the surrogate on magnitude, and this is the version of
that finding you should believe.** With `squared_error` in place of the collapsed
`absolute_error` (§8.7), `tabular_gbt` has the best MAE of any method on
`test_id` (0.0104 against the surrogate's 0.0146) **and** the best MAE on the rows
where a disruption actually bit (0.0945 against 0.1569). It also has the best
top-1 agreement (0.729 against 0.583). The previous release retracted the
magnitude claim on the strength of a comparison against a constant; the claim
stays retracted, now against a model.

**The surrogate loses to the constant zero function on pooled MAE on 6 of 7
splits.** `constant_zero` scores 0.012246 on `test_id` against the surrogate's
0.014581, and beats it on `train`, `val`, `test_id`, `shift_topo`, `shift_size`
and `shift_type`. The surrogate wins only `shift_multi` (0.034292 against
0.036020), the split with the least zero mass (84.3% zeros rather than ~93%). This
is stated plainly rather than buried because a reader is entitled to know that the
headline error metric on this task is minimised by refusing to predict.

**On the metric a constant cannot win, the surrogate does beat predicting
nothing, everywhere.** Restricted to nonzero-truth rows, it is below the all-zero
constant on all 7 splits — 0.1569 against 0.2137 on `test_id` — and the margin
clears the noise scale on all five evaluation splits (+2.25x to +6.20x, see the
verdict table). It also beats `constant_train_mean` on pooled MAE on all 7 splits.
So the model has learned something about magnitude; it just spends that skill
being wrong on the 92.5% of rows where the answer is zero, and pooled MAE weighs
those rows 12 to 1.

**`tabular_gbt_l1` and `constant_zero` are the same function.** Not similar: their
prediction vectors are element-wise identical across all 20,624 rows, so every
metric in every table matches exactly (MAE 0.012246, MAE-on-nonzero 0.213741,
RMSE 0.069612, 1 unique prediction). That row is left in the table as the standing
proof of §8.7.

**A new result the broken baseline had hidden.** The old table could not compare
rank fidelity against the reference model at all, because a constant's Spearman is
undefined. It can now, and the answer is interesting: the GBT wins
within-scenario Spearman in-distribution (0.687 against 0.571) but the surrogate
wins on the **held-out disruption mechanism** by 0.503 against 0.288 — **+8.90x
the noise scale**, the largest surviving margin anywhere in this document — and
suggestively on unseen topologies (+1.16x). The tabular model interpolates the
mechanisms it was trained on; the surrogate transfers to one it has never seen.
That is the shape of result this project was looking for, and it was invisible
until the baseline worked.

**The retrieval result is still the interesting rival.** Nearest-neighbour
scenario retrieval has the best within-scenario Spearman in-distribution (0.7368),
beating the surrogate by 0.165 = 3.86x the noise scale, so that defeat is real and
is reported as one. It then degrades under shift while the surrogate holds. Its
training MAE of 0.000062 says why: it is memorising, and `train` is a lookup.

**The no-message-passing result is unaffected and remains the strongest thing
here.** With the same features, loss, schedule and a parameter budget matched to
1.025x, removing message passing costs 0.18 of within-scenario Spearman (0.582 to
0.402) and is the *only* ablation whose MAE damage exceeds the seed-noise scale
(+4.11x). Nothing in the baseline fix touches it: that comparison is the surrogate
against itself.

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

### Paired per-row tests vs the reference model (`results/tables/statistical_tests.csv`)

Unit of analysis is a **row**, so this compares two sets of weights, not two methods.

| method (abs error) | mean | ref mean | delta | CI low | CI high | p (Holm) | Cohen's d | median delta | rows worse | n |
|---|---|---|---|---|---|---|---|---|---|---|
| surrogate_ensemble.abs_error | 0.01534 | 0.01042 | 0.00492 | 0.00299 | 0.00706 | 0.00000 | 0.11819 | 0.00372 | 0.83529 | 1536 |
| surrogate_single.abs_error | 0.01458 | 0.01042 | 0.00416 | 0.00227 | 0.00617 | 0.00000 | 0.10412 | 0.00330 | 0.82292 | 1536 |
| mlp_no_message_passing.abs_error | 0.01740 | 0.01042 | 0.00698 | 0.00452 | 0.00981 | 0.00000 | 0.13121 | 0.00539 | 0.83984 | 1536 |
| tabular_gbt_l1.abs_error | 0.01225 | 0.01042 | 0.00183 | -0.00072 | 0.00473 | 0.00000 | 0.03326 | 0.00000 | 0.04492 | 1536 |
| tabular_ridge.abs_error | 0.02461 | 0.01042 | 0.01419 | 0.01220 | 0.01645 | 0.00000 | 0.33476 | 0.00314 | 0.55990 | 1536 |
| retrieval_knn.abs_error | 0.01055 | 0.01042 | 0.00013 | -0.00114 | 0.00140 | 0.00000 | 0.00526 | 0.00000 | 0.12240 | 1536 |
| topology_heuristic.abs_error | 0.02399 | 0.01042 | 0.01358 | 0.01128 | 0.01623 | 0.00000 | 0.26974 | 0.00858 | 0.73698 | 1536 |
| constant_zero.abs_error | 0.01225 | 0.01042 | 0.00183 | -0.00072 | 0.00473 | 0.00000 | 0.03326 | 0.00000 | 0.04492 | 1536 |
| constant_train_mean.abs_error | 0.02363 | 0.01042 | 0.01322 | 0.01081 | 0.01597 | 0.00000 | 0.25438 | 0.01278 | 0.89453 | 1536 |

`median delta` and `rows worse` are there because the Wilcoxon p-value counts rows while the bootstrap interval weighs them, and on this target the two point in **opposite directions**. `constant_zero` is worse than the reference on the mean (+0.00183) but worse on only **4.5% of rows**, with a median paired difference of exactly 0: predicting zero is exactly right on 92.5% of rows and wrong only on the few that matter. The surrogate, by contrast, is worse on 82.3% of rows. A paired sign test on absolute errors is therefore one more thing a constant can win on this target, and it is reported with its direction rather than as a bare asterisk.

**Every one of these p-values clears any threshold you like, and that is the
point.** With ~1,500 paired rows and a tight pairing, a mean absolute-error
difference of 0.001 — about a tenth of the seed-to-seed spread — comes out at
p < 1e-5. The test is not wrong; it is answering "is this difference consistent
across rows", and the answer is genuinely yes. It simply is not the question
anyone cares about, and reading these stars as evidence that one *method* beats
another would be a mistake. The noise-scale table above is the honest version.

**What these tests do and do not answer.** A paired per-row test conditions on
**one trained model per method**. It correctly answers "are these weights better
than those weights, consistently across rows". It does *not* answer "is this
method better", because that claim treats the **training run** as the sampling
unit and there is one run per method here. No number of test rows fixes an n of 1
in that unit. That is exactly why section 2 exists, and why the table above
reports the noise ratio next to every gain.

---

## 4. Ablations

One mechanism removed at a time, everything else held fixed: same data, same
seed, same loss, same schedule, same training loop.

### Ablations (`results/tables/ablations.csv`, split `test_id`)

| variant | params | MAE | delta vs full | vs noise | Spearman (within) | train s |
|---|---|---|---|---|---|---|
| full | 55752 | 0.01458 | 0.00000 | +0.00 -> inside noise | 0.58241 | 144.80700 |
| no_message_passing | 57160 | 0.01740 | 0.00282 | +4.11 -> survives | 0.40183 | 49.07230 |
| one_hop | 27848 | 0.01585 | 0.00127 | +1.85 -> suggestive | 0.53539 | 62.02050 |
| unidirectional | 38280 | 0.01585 | 0.00127 | +1.85 -> suggestive | 0.58009 | 73.86840 |
| no_edge_features | 52200 | 0.01584 | 0.00126 | +1.84 -> suggestive | 0.58646 | 117.43000 |
| linear_traj_decoder | 50671 | 0.01437 | -0.00021 | -0.31 -> inside noise | 0.56779 | 124.80700 |

A positive delta means removing that mechanism made the model worse.

---

## 5. Counterfactual criticality — the headline experiment, and where it fails

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

### Reading this table honestly

Three things are true at once and all three belong in the summary.

**The counterfactual framing wins the argument this repository set out to make,
and now it wins it on evidence.** The previous release reported 36 of 36
adjudicated disagreements going to the counterfactual — but the feature score
driving that experiment was a constant, and `np.argsort` of a constant returns
index order, so `rank_a_feature` was `node_a + 1` in **all 36 rows** and the
experiment was adjudicating node numbering (§8.7). Re-run against a GBT that
actually ranks, the count is unchanged — **36 disagreeing pairs, 36 of 36 resolved
in favour of the counterfactual** — and now the feature ranks are real. In
`shift_4` the GBT's top five candidates are nodes **18, 32, 12, 31, 42** (scores
0.2164, 0.2164, 0.2017, 0.1946, 0.1867), not nodes 0-4, and
`rank_a_feature == node_a + 1` now holds in 1 of 36 rows, which is roughly what
chance gives at 46 candidates.

The argument survives in one line, and it is a stronger line than before: in
`shift_4` **every one of the feature score's top five nodes has a true service
loss of exactly 0.0000**, while the node it ranks **46th of 46** — node 45, a
tier-3 distribution point that is the sole source for three demand points — has
the largest true loss in the network at **0.7430**. Across all 36 adjudicated
pairs the nodes the feature score prefers average a true loss of **0.000926**
(maximum 0.033346) and the nodes it passes over average **0.426345** (minimum
0.228754).

**The feature-based approach is still a poor criticality ranker, and the reason is
now diagnosable rather than a bug.** The working GBT scores recall@10 of 0.183 and
a regret fraction of 0.867, against 0.633 and 0.161 for the topology heuristic. It
emits only **6 to 9 distinct scores across the 46 candidates** in each network
(`distinct_scores` 7.33 on average), because every probe carries the same
standardised outage, so the only inputs that vary are the disrupted node's own
attributes and the trees bucket those coarsely. Its full-vector rank correlation
with the truth is 0.036. That is a real structural limitation of scoring a node by
its own features — which is the thing this repository set out to argue — and it is
now supported by a model rather than by an artefact.

**`tabular_gbt_l1` is the row this table used to publish as `tabular_gbt`.** One
distinct score, recall@10 0.05, regret 0.979. Its numbers are unchanged from the
previous release, which is the point: the previous release's "reference model" is
this row.

**The learned surrogate is not the thing that wins it.** A hand-weighted composite
of six topology signals — sole-source reach, downstream reach, path betweenness,
BOM depth, inverse capacity slack, throughput — beats the trained graph network on
every criticality metric: recall@10 0.633 against 0.450, regret fraction 0.161
against 0.433, full-vector Spearman 0.568 against 0.261. The heuristic costs
milliseconds and no training at all.

**Why the surrogate underperforms here, specifically.** Its score spread across
candidates is 0.0008, against a true spread of 0.748 — it is *nearly constant* on
the criticality probes, and it under-predicts the largest impacts by more than an
order of magnitude (max predicted 0.0523 against max true 0.7481). Its
within-scenario ranking is good (Spearman ~0.58) but criticality ranking is a
purely **cross-probe** comparison, and its pooled cross-scenario correlation is
only ~0.35. The model learned "which demand point suffers most in this scenario"
much better than it learned "how bad is this scenario compared to that one" — and
only the second is what a criticality sweep needs.

That points at the zero-inflated target: an L1 objective on a distribution that is
92.5% zeros rewards shrinking every prediction toward zero, which preserves local
ordering and destroys global scale. Note the asymmetry with §8.7, because it is
the interesting part: the *same* objective on the *same* target merely blunted the
neural network's global scale, while it reduced the boosted trees to an exact
constant. A gradient step on a shared parameter vector still moves the whole
function; a tree leaf takes the exact L1 minimiser of the samples that fall in it,
and that minimiser is 0.

**And the surrogate's criticality ranking is not reproducible.** Its score spread
(0.0008) is *smaller* than the run-to-run drift in its own per-row predictions
(up to 1.3e-03 between process launches, from parallel scatter reductions). Its
recall@10 moved from 0.433 to 0.450 between two runs with the same seed, the same
cached weights and a bit-identical oracle. That is not a separate failure — it is
the same near-constancy measured a second way. See §8.8.



---

## 6. Uncertainty

Coverage without sharpness is meaningless — a `[0, 1]` interval covers everything —
so both are reported. Out of distribution, the property that matters is whether
error grows *with* the predicted uncertainty.

### Predictive-interval calibration (`results/tables/calibration.csv`)

**Deep-ensemble Gaussian intervals**

| split | coverage@0.5 | coverage@0.8 | coverage@0.9 | coverage@0.95 | width@0.5 | width@0.8 | width@0.9 | width@0.95 |
|---|---|---|---|---|---|---|---|---|
| shift_multi | 0.9078 | 0.9381 | 0.9520 | 0.9646 | 0.0612 | 0.1022 | 0.1268 | 0.1480 |
| shift_size | 0.9609 | 0.9747 | 0.9816 | 0.9844 | 0.0497 | 0.0849 | 0.1059 | 0.1241 |
| shift_topo | 0.9514 | 0.9716 | 0.9798 | 0.9874 | 0.0442 | 0.0757 | 0.0946 | 0.1109 |
| shift_type | 0.9773 | 0.9848 | 0.9874 | 0.9880 | 0.0355 | 0.0623 | 0.0784 | 0.0923 |
| test_id | 0.9674 | 0.9831 | 0.9876 | 0.9902 | 0.0442 | 0.0757 | 0.0945 | 0.1108 |
| train | 0.9636 | 0.9752 | 0.9812 | 0.9857 | 0.0437 | 0.0748 | 0.0934 | 0.1096 |
| val | 0.9551 | 0.9694 | 0.9772 | 0.9831 | 0.0434 | 0.0745 | 0.0931 | 0.1092 |

**Quantile-head central interval** (nominal 0.90)

| split | coverage | width | gap | n |
|---|---|---|---|---|
| train | 0.9854 | 0.1201 | 0.0854 | 9600 |
| val | 0.9831 | 0.1208 | 0.0831 | 1536 |
| test_id | 0.9857 | 0.1213 | 0.0857 | 1536 |
| shift_topo | 0.9848 | 0.1231 | 0.0848 | 1584 |
| shift_size | 0.9838 | 0.1368 | 0.0838 | 3200 |
| shift_type | 0.9912 | 0.1125 | 0.0912 | 1584 |
| shift_multi | 0.9564 | 0.1456 | 0.0564 | 1584 |

**Does uncertainty track error?**

| split | mean sigma | mean |err| | sigma-vs-err Spearman | err-detect AUROC | AUSE | sigma aleatoric | sigma epistemic |
|---|---|---|---|---|---|---|---|
| train | 0.0513 | 0.0164 | 0.9293 | 0.9915 | 0.0463 | 0.0511 | 0.0023 |
| val | 0.0512 | 0.0179 | 0.9244 | 0.9913 | 0.0467 | 0.0511 | 0.0021 |
| test_id | 0.0518 | 0.0153 | 0.9406 | 0.9922 | 0.0323 | 0.0517 | 0.0023 |
| shift_topo | 0.0519 | 0.0176 | 0.8941 | 0.9819 | 0.0580 | 0.0518 | 0.0024 |
| shift_size | 0.0579 | 0.0186 | 0.9369 | 0.9902 | 0.0504 | 0.0577 | 0.0027 |
| shift_type | 0.0442 | 0.0103 | 0.9099 | 0.9841 | 0.0608 | 0.0442 | 0.0010 |
| shift_multi | 0.0676 | 0.0356 | 0.9079 | 0.9648 | 0.0851 | 0.0673 | 0.0047 |

### What survives here and what does not

**The intervals are not calibrated.** A nominal 50% Gaussian interval covers 97%
of the truth on `test_id`, and a nominal 95% one covers 99%. The predicted
variance is far too large almost everywhere, which is what a Gaussian
log-likelihood does when it is fitted to a target that is a spike at zero plus a
right tail: the single variance that best explains both the mass at zero and the
tail is much wider than the mass at zero needs. `docs/METHOD.md` predicted the
Gaussian route would be the worse-calibrated of the two, and it is — but the
quantile head is over-covered too (0.986 at a nominal 0.90), so the failure is not
purely a symmetry artefact.

**The uncertainty *ordering* is excellent.** Error-detection AUROC is 0.99 and the
sigma-to-error rank correlation is 0.94 on `test_id`. If a planner asks "which of
these predictions should I not trust", the model answers that question very well
even though its interval widths are meaningless. Those are different claims and
only the second one is supported.

**The deep ensemble contributes almost nothing.** The epistemic term (the variance
of the members' means) is around 0.0023 against an aleatoric term of ~0.0517 — 4%
of the total. Three members initialised differently converge to nearly the same
function. Whatever the ensemble is buying here, it is not the epistemic signal it
was included for, and the cost is 3x the training time.

**Out-of-distribution detection between splits is weak.** Mean predicted sigma is
0.0518 on `test_id` and 0.0519 on `shift_topo`, while mean absolute error rises
from 0.0153 to 0.0176. The model's uncertainty does not move when the topology
changes. The within-split ordering claim above therefore does **not** extend to
"the surrogate knows when it is out of distribution", and this repository does not
make that claim.

---

## 7. Efficiency

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

### The honest reading

**The per-scenario speed-up is real**: 50.3 ms for one simulator counterfactual
against 4.41 ms for one screened scenario in a batched surrogate pass, a measured
**11.4x**, both with 8 warm-up iterations and 25 timed repeats on the same machine
state and reported as median with IQR.

**The decision-quality experiment does not favour the surrogate at any budget
tested.** The budget curve is the honest test and the surrogate loses it outright:
below its own fixed cost it returns nothing, at 1 s the simulator is already ahead
(0.548 against 0.333), by 2 s it is at 0.952, and by 5 s exhaustive search is
exhaustive at 1.000 while the surrogate plateaus at 0.450 because its own ranking
quality caps it. The reason is arithmetic rather than modelling — a 46-candidate
sweep costs the oracle about 1.8 s, so at this network size there is simply no
regime in which the exact answer is unaffordable.

**This curve is the least stable table in this document, and the instability is
one-sided.** It is built from measured wall-clock, and §8.4 records that the same
batch timed at 600 ms and 2870 ms minutes apart on this shared machine. The
previous release's run had the oracle sweep at 2.62 s and the two methods tying at
1 s (0.367 against 0.369); this run had it at 1.80 s and the simulator clearly
ahead. The *ordering* — the surrogate never wins at any budget — held in both runs,
and it is the only thing claimed from this table. A surrogate for this task earns its keep only when
the candidate set is large enough that the oracle is out of reach, and this
repository does not demonstrate such a regime. That is a limitation of the
experiment's scale as much as of the model, and it is stated rather than buried.

**Break-even is brutal once the accounting is honest.** Charging the surrogate for
its training *and* for the dataset generation that used the very simulator it
replaces, a single model must screen **13,270** scenarios before it has paid for
itself, and the shipped three-member ensemble **26,167** — roughly 290 and 570 full
46-candidate network sweeps respectively. Had the dataset-generation term been
omitted, as it commonly is, the figure would have looked roughly 250x better; see
the retraction in §8.6.

---

## 8. What failed, what was retracted, and what this project cannot support

This section is not decoration. Everything below is a thing that went wrong or a
claim that did not survive contact with a measurement.

### 8.1 Bugs that silently produced wrong results

**Lead-time inflation was a no-op for the first half of this project.** The
in-transit ring buffer was sized from the *nominal* maximum lead time, so an
inflated shipment wrapped around it and arrived **early**. All 120 sampled
`lead_time_inflation` scenarios produced exactly `0.0000` impact. It was found by
tabulating mean impact per disruption kind and noticing a column of exact zeros —
not by a test, because the test did not exist yet. After the fix only that row
moved (0.0000 → 0.0017) and the other five kinds were unchanged to four decimal
places, which is what makes the fix credible rather than merely different.
Regression test: `test_lead_time_inflation_delays_by_the_inflated_amount`.

**The headline target was effectively untrained in the first configuration.**
Time-to-impact and recovery time are in periods (0–30); service loss is a fraction
(0–1). Unscaled, the timing L1 came out around **12** against an impact L1 around
**0.05**, so even at `w_timing = 0.2` more than 98% of the gradient went to an
auxiliary head. Both terms are now divided by the horizon.

**Config type coercion never ran.** `from __future__ import annotations` makes
every `dataclasses.Field.type` a *string*, so `getattr(t, "__origin__")` was always
`None` and `_coerce` returned its input unchanged. YAML lists stayed lists instead
of becoming tuples, and `--set model.bidirectional=0` stored the integer `0`
rather than `False`. Caught by a test that asserted the coerced *type*, not just
the value. Fixed with `typing.get_type_hints`.

### 8.2 Comparisons that were unfair until they were fixed

**The no-message-passing ablation was 3.6x smaller than the full model.** Deleting
the processor also deletes its parameters, so the first version of this ablation
would have confounded "the graph helps" with "more parameters help". The node-only
trunk is now widened to match; the two counts are printed in `ablations.csv`.

**The topology heuristic's MAE was meaningless.** It is a sum of standardised
signals, so its scale is arbitrary, and it reported an MAE of ~2.4 against a
target bounded near 1. A two-parameter affine map is now fitted **on the training
split only**, which is the least that makes its error interpretable without
quietly turning it into a linear model.

**~~The boosted-tree reference model collapses to an exact constant on the
criticality probes... a real property of the reference approach on this task
rather than a tuning failure.~~** **This paragraph was wrong in the most important
way a paragraph in this section can be wrong: it looked at a bug and concluded it
was a property of the world.** It was a tuning failure, it was diagnosable in one
line, and the collapse was not confined to the criticality probes — the same model
emitted 0.000000 for all 20,624 rows of the method comparison. Retracted in full;
see §8.7. What survives from it is only the mechanism sketch, which was correct:
every probe carries the same standardised disruption, so the only varying inputs
are the disrupted node's own attributes. With a working loss the GBT emits 6 to 9
distinct scores over 46 candidates rather than 1, and its recall@10 is 0.183
rather than 0.05 — still poor, and now poor for the structural reason this
repository actually wanted to demonstrate.

### 8.3 Experimental-setup errors found by testing

**The warm-up was shorter than the pipeline fill time.** The simulator starts with
an empty in-transit pipeline, so a short warm-up measures a fill transient rather
than a steady state — the "undisrupted baseline" had a fill rate of 0.956 instead
of ~1.0, which would have contaminated every impact label. Measured across
warm-up lengths on networks whose maximum cumulative lead time to demand is 18.8
periods: 0.9557 at 10, 0.9748 at 16, 0.9944 at 20, 0.9973 at 26, 0.9997 at 32.

**Three of my own test expectations were wrong and the simulator was right.**
Dual sourcing under static shares gives a steady-state loss of 0.5, but the
*measured* loss over a finite horizon approaches it from below (0.425, 0.475,
0.4925 at horizons 20, 60, 200) because material already in transit is still
delivered. I had asserted 0.5 and had to correct the test, not the simulator. The
same happened with a Gaussian interval width (I forgot the non-negativity clip)
and with a tie-handling case in top-1 agreement.

**Backlogging plus unbounded capacity builds its own buffer.** With backlogging on,
each backlogged unit is re-ordered on top of fresh demand; with effectively
infinite capacity the chain over-produces during warm-up and the excess settles as
inventory that then absorbs a later outage *entirely*. This is the bullwhip effect
appearing in the simulator's own warm-up. It is real behaviour, and a mild
validation that the mechanics are non-trivial, but it means the exact arithmetic
tests must use lost sales rather than backlogging.

### 8.4 What the compute budget cost, and what that forbids claiming

This ran on 4 shared CPU cores with **no GPU**, alongside several other projects.
Two things went wrong that are worth recording because they shaped every number
here.

**The machine ran out of memory, not CPU.** The full scenario set expands from an
83 MB pickle to roughly 800 MB of live Python objects per process. With free
physical memory under 1 GB, training stopped being compute-bound and became
page-fault bound: one process accumulated **23 seconds of CPU time in ten minutes
of wall-clock**. The evaluation splits are now capped, which roughly halves the
footprint.

**Contention makes wall-clock unreliable by a factor of three.** The identical
batch was timed at 600 ms and at 2870 ms minutes apart. This is why every
benchmark reports median and IQR, and why the efficiency conclusions are stated
as ratios measured back-to-back rather than as absolute throughput.

The scale reductions this forced, relative to what the design intends:

| setting | intended | shipped | why |
|---|---|---|---|
| hidden width | 64 | **32** | epoch time |
| message-passing rounds | 3 | 3 | kept — it is the contribution |
| training epochs | 30 | **8** | epoch time |
| ensemble members | 5 | **3** | five runs did not fit |
| training scenarios | 2,736 | **1,200** | memory |
| evaluation scenarios per split | 420 | **~198** | memory |
| predicted trajectory periods | 24 | **8** | the GRU costs one dispatch per period, regardless of batch size |
| ablation variants | 8 | **6** | each is a full training run |

The scenario cache itself was generated at full size (5,140 scenarios over 22
networks, 313 s); the caps above are applied at *load* time, so the evaluation
splits are deterministic stratified subsamples of the full ones rather than a
differently-generated dataset.

**What that forbids.** Any claim of the form "this architecture is better than
that one" is weak here, because every model is under-trained relative to its
capacity and the seed spread is correspondingly large. The results that survive
are the ones where the *effect is structural* rather than a matter of fit quality.
Where a difference is smaller than the seed noise scale, this document says
"inside noise" and does not argue with it.

### 8.5 Claims this project does **not** make

- **No real-world validation.** Everything is synthetic. Results transfer to a real
  network only insofar as this simulator's mechanics are right. The CSV loader
  lets a reader substitute their own topology, but not their own mechanics.
- **No probability of disruption.** Following Simchi-Levi et al. (2015), impact is
  computed conditional on a scenario. Nothing here estimates how likely a node is
  to fail.
- **No claim that the decision-regret model is realistic.** Protecting a node is
  assumed to avert exactly its own counterfactual impact and impacts are assumed
  not to interact. Both are wrong in a network, so the reported regret is a lower
  bound on true regret. It is used because it is monotone in ranking quality, not
  because it is a cost model.
- **No claim about optimal inventory policy.** Base-stock levels are given.

### 8.6 Three more bugs, found late

**The criticality log line reported the wrong method's recall.** It indexed
`rank_rows[-3]` — correct when there were three methods, silently wrong the moment
a fourth was added. For a while it printed the *constant* baseline's recall (0.00)
as though it were the surrogate's, which briefly looked like a catastrophic
failure of the surrogate. The tables were always right; only the log line was
wrong. It now looks the method up by name. Lesson: positional indexing into a
list of results is a bug waiting for the next row.

**Parameter matching silently drifted.** The no-message-passing trunk was tuned to
match the graph model at `hidden = 48`. When the compute budget forced `hidden`
down to 32, the graph model shrank and the trunk did not: the "parameter-matched"
ablation had **1.80x** the parameters of the model it was being compared against.
Retuned to 1.025x, and `test_no_graph_ablation_has_a_comparable_parameter_budget`
now guards the ratio so the same drift cannot recur silently.

**Dataset generation time was overwritten by a cache hit.** The efficiency
break-even charges the surrogate for the simulator time that produced its training
set. The data stage recorded its own wall-clock, so once the scenario cache
existed it recorded a 1.2-second pickle load instead of the 313 seconds of
simulation — which would have understated the surrogate's true setup cost by more
than two orders of magnitude and made the break-even look ~250x better than it is.
Generation time and stage time are now separate columns, and a cache hit carries
the generation time forward instead of overwriting it.

### 8.7 The worst bug in this repository: the reference baseline was a constant

**`tabular_gbt` was the constant zero function, and it was the model this
project's magnitude claim had been retracted in favour of.**

`TabularRiskModel(kind="gbt")` was constructed with `loss="absolute_error"`, and
the code comment justified it: absolute error "to match the surrogate's L1
objective", so that the comparison would be about representations rather than loss
functions. The reasoning is defensible in general and fatal here. The row-level
target is **exactly 0.0 for 92.48% of rows** (19,073 of 20,624 in
`results/runs/base/per_item.csv`). The constant that minimises absolute error is
the median. The median of this target is exactly 0. So:

* `HistGradientBoostingRegressor` initialises its L1 baseline prediction at the
  median — 0;
* the L1-optimal value of every leaf is the median of that leaf's residuals, which
  is also 0;
* the validation loss never improves, early stopping fires after 10 rounds, and the
  shipped model came out at **30 total tree nodes** across those rounds — a root
  and two leaves per round, all predicting 0
  (`results/runs/base/summary.json`, `tabular_gbt_l1_tree_nodes`).

**Measured consequence.** `pred_tabular_gbt` in the previously committed
`per_item.csv` was **exactly 0.000000 for all 20,624 rows**: one unique value, min
0, max 0. On the 1,551 rows where a disruption actually bit (mean truth 0.193317)
it predicted 0. The variant is retained as `tabular_gbt_l1`, and its prediction
vector is **element-wise identical** to `constant_zero`'s on all 20,624 rows —
which is why those two rows of `method_comparison.csv` agree in every digit of
every metric on every split.

**Why nothing caught it**, one metric at a time:

* **MAE rewarded it.** On a 92.48%-zero target the all-zero constant is the
  MAE-optimal predictor, so the collapse presented as the *best* MAE in the table
  on four of five evaluation splits.
* **RMSE hinted, and was not read.** Its RMSE was the worst of any tabular method
  (0.0696 against 0.0554 for the ensemble). That was in the table all along.
* **Spearman was `NaN`** — undefined for a constant — and the renderer printed
  `not measured`, which reads as a missing measurement rather than a dead model.
* **Top-1 agreement scored 0.1250**, which looks like a weak-but-real result. It is
  what `np.argmax` returns on an all-tied row: index 0, correct whenever demand
  point 0 happens to be worst-hit.
* **The paired Wilcoxon favoured it.** It counts rows, and the constant is exactly
  right on 92.5% of them: it is worse than the working GBT on only 4.5% of rows,
  with a median paired difference of exactly 0.
* **No trivial baseline was in the table.** With `constant_zero` present, the two
  columns agreeing to six decimal places would have been the first thing anyone
  saw. This is the omission that made all of the above possible.

**What it corrupted.**

1. *The magnitude retraction was mis-attributed.* The README and this document
   retracted magnitude fidelity because the surrogate "loses to a reference GBT on
   4 of 5 splits (-4.5x to -22.2x the noise scale)". Those ratios were correct
   arithmetic against a column that was a constant. The honest statement is harder,
   not softer: **the surrogate is beaten on pooled MAE by the constant zero
   function on 6 of 7 splits**, and pooled MAE on this target is minimised by
   predicting nothing, so that defeat is as much a property of the metric as of the
   model. Separately, and this is the real defeat: a **working** GBT beats the
   surrogate on MAE and on MAE-restricted-to-nonzero-truth on all five evaluation
   splits.
2. *The headline experiment was adjudicating node numbering.* `find_disagreements`
   ranks with `np.argsort(-scores, kind="stable")`, and `argsort` of a constant
   returns index order. Every one of the 36 shipped disagreements had
   `rank_a_feature == node_a + 1`, and `node_b` was always the highest-numbered
   candidate in its network. "The node the feature score ranks first has a true
   service loss of exactly 0.0000" was a statement about the lowest-numbered
   candidate. Re-run against a GBT that ranks, the result holds — 36 of 36 — so the
   conclusion was right and the evidence for it was not.
3. *§8.2 drew the wrong lesson.* It recorded the collapse as "a real property of
   the reference approach on this task, not a tuning failure". Exactly backwards.

**What was changed.**

* `kind="gbt"` uses `squared_error`. `kind="gbt_l1"` keeps the absolute-error
  variant under that name, labelled degenerate by construction, because deleting it
  would hide how the failure happened and because the contrast is the cleanest
  demonstration of the point.
* `constant_zero` and `constant_train_mean` are first-class methods, run and
  reported on every split, in `method_comparison.csv` and in `per_item.csv`.
* `prediction_diversity` / `require_non_degenerate` / `DegenerateBaselineError`:
  every method's distinct-prediction count is recorded, and any method not declared
  constant by design that emits fewer than two distinct values **fails the run**
  rather than entering the table. Guarded by
  `test_require_non_degenerate_raises_on_a_constant` and
  `test_gbt_with_absolute_error_collapses_on_a_zero_inflated_target`, which
  reproduces the whole mechanism on synthetic data with this project's zero-mass.
* `find_disagreements` refuses a constant score vector outright.
* `mae_nonzero_truth` is reported next to pooled MAE everywhere, and
  `median_difference` / `frac_rows_a_worse` next to every p-value.

**The general lesson is not "check your loss function".** It is that a results
table on a heavily zero-inflated target is uninterpretable without the trivial
predictors in it. Every guard above is downstream of a single omission: there was
no row saying what predicting nothing scores, so nothing in the table could reveal
that the best-scoring entry *was* predicting nothing. The loss-function bug was
ordinary. The missing baseline is what turned it into a published claim.

### 8.8 Cross-process determinism is five decimal places, not exact

Found while re-running the comparison for §8.7, and it changes what
`docs/REPRODUCIBILITY.md` is entitled to claim.

Re-running `compare` at the same seed on the same machine reproduces the training
trace exactly (`epoch 0 loss 0.07373 val -0.28690`, `epoch 5 loss -1.26929 val
-1.38481`, `best_val_loss -1.49757` — identical to the committed
`results/runs/base/summary.json`). But the **per-row predictions** are not
bit-identical across process launches:

| quantity | max abs difference across two launches |
|---|---|
| simulator truth | **0.0** |
| `topology_heuristic` | **0.0** |
| `mlp_no_message_passing` (retrained from scratch) | **0.0** |
| `surrogate_single` | 4.1e-06 |
| `surrogate_ensemble` | 9.4e-04 |
| `tabular_ridge` | 2.2e-04 |
| `retrieval_knn` | 1.0e-02 |

The pattern is informative. Everything that is pure NumPy and Python arithmetic —
the simulator, the topology heuristic — is exact. So is the `layers = 0` MLP, which
has no scatter operations. The full surrogate drifts, and it is the one model with
parallel `index_add_` reductions over a two-thread pool; `retrieval_knn` drifts
most because its distances come from a BLAS matmul and `argpartition` then breaks
near-ties differently.

Two consequences, and the second one matters:

1. `test_training_is_reproducible_from_its_seed` asserts max abs difference exactly
   0.0 and passes — but it is a **within-process** guarantee, not a cross-process
   one. `docs/REPRODUCIBILITY.md` previously described it as a within-*machine*
   guarantee, which was too strong.
2. **The surrogate's criticality ranking is not reproducible.** Its criticality
   score spread is 0.0008, which is *smaller* than the 1.3e-03 drift in its own
   criticality scores between launches. Its recall@10 moved from 0.433 to 0.450
   with the same seed, the same cached weights and a bit-identical oracle truth.
   That is not a new failure; it is §5's "nearly constant on the criticality
   probes" measured a second way, and it is the sharpest available statement of it:
   **the model's ranking signal on this task is the same size as its
   floating-point noise.**

Separately: exporting `OMP_NUM_THREADS=2` into the environment *before* launch —
rather than leaving it to the in-process `limit_threads` call, which uses
`os.environ.setdefault` and therefore runs after NumPy has already bound its BLAS —
changes the BLAS thread count and diverges the training trace by the fourth decimal
by epoch 5 (`-1.26665` against `-1.26929`). The committed numbers all come from
launches with those variables unset. Determinism here is conditional on the launch
environment, and that condition was undocumented.

### 8.9 What was *not* re-run for §8.7, and why

`ablations.csv` and `ablation_statistics.csv` were **not** regenerated. Every
ablation compares the surrogate against itself with `full` as the baseline; none of
them touches a tabular model, so the baseline fix cannot move them. The seed study
was re-run and reproduced its own noise scales to six decimals (`test_id` MAE
0.000685 -> 0.000686), so the ablation verdicts against the noise scale are
unchanged: `no_message_passing` is +4.11x and still the only ablation that clears
it. Re-running six training runs would have perturbed every ablation number in the
fifth decimal via the drift in §8.8 for no scientific gain.

`efficiency.csv` and `break_even.csv` were not regenerated either: they measure
wall-clock and charge training plus dataset-generation time, none of which the
baseline fix touches.

Consequence worth stating: `ablations.csv` predates the addition of
`mae_nonzero_truth` to `fidelity_report`, so it does not carry that column while
`method_comparison.csv` does. The column is additive and no table in this document
reads it from `ablations.csv`; a `make ablate` will populate it.
