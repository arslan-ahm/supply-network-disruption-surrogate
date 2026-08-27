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
| The surrogate generalises better than nearest-neighbour retrieval under topology shift | **one axis survives** — retrieval wins in-distribution (-3.86x, worse outside noise); the surrogate wins on `shift_size` (+2.50x, survives) and `shift_topo` (+1.41x, suggestive); `shift_type` and `shift_multi` are inside noise |
| The surrogate predicts impact *magnitude* well | **retracted** — worse than the reference model on MAE on four of five splits, by -4.5x to -22.2x the noise scale |
| The learned surrogate is the best criticality ranker | **retracted** — a five-line topology heuristic beats it on recall@k, regret and rank correlation |
| The surrogate's predictive intervals are calibrated | **fails** — grossly over-covered (0.97 empirical at nominal 0.50) |
| Uncertainty tracks error | **holds within a split** (error-detection AUROC 0.99), **fails across splits** (mean sigma barely moves under shift) |
| The deep ensemble contributes epistemic uncertainty | **fails** — the epistemic term is ~4% of total predictive sigma |
| The surrogate wins on decision quality at a fixed compute budget | **fails at every budget tested** — it ties at 1 s and is beaten from 2 s onward, where exhaustive search reaches recall 1.0 and the surrogate plateaus at 0.433 |
| The surrogate is faster per scenario | **holds** — 11.4x measured, but break-even is 13,270 screened scenarios (26,167 for the ensemble) once dataset generation is charged |

---

## 0. How to read the two fidelity families

The target is **93.5% exact zeros at the row level** — most disruptions are
absorbed and most demand points are unaffected. That single fact governs how every
error number here should be read:

* **A good MAE is cheap.** Predicting approximately zero everywhere scores well
  and is useless for screening. The boosted-tree reference model demonstrates this
  literally: it has the *best* MAE of any method on four of five splits **and**
  produces an exactly constant score on the criticality probes.
* **Rank fidelity is the real test.** The planner's question is "which link should
  I look at first", so what matters is whether the ordering is right.

Both are reported for every method on every split, unaggregated, so the reader can
see which one was achieved. A single "accuracy" number would hide the distinction,
and in this task the distinction is the whole point.

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

| method | MAE (low=good) | RMSE (low=good) | bias | Spearman pooled (high=good) | Spearman within-scenario (high=good) | scenarios contributing | top-1 agree (high=good) | scenarios contributing | rows |
|---|---|---|---|---|---|---|---|---|---|
| surrogate_ensemble | 0.0153 | 0.0554 | -0.0030 | 0.3536 | 0.5715 | 48 | 0.5208 | 48 | 1536 |
| surrogate_single | 0.0146 | 0.0531 | -0.0025 | 0.3567 | 0.5824 | 48 | 0.5833 | 48 | 1536 |
| mlp_no_message_passing | 0.0174 | 0.0685 | -0.0063 | 0.2671 | 0.4018 | 48 | 0.2292 | 48 | 1536 |
| tabular_gbt | 0.0122 | 0.0696 | -0.0122 | not measured | not measured | 0 | 0.1250 | 48 | 1536 |
| tabular_ridge | 0.0246 | 0.0616 | 0.0065 | 0.3526 | 0.4866 | 48 | 0.4583 | 48 | 1536 |
| retrieval_knn | 0.0106 | 0.0412 | 0.0002 | 0.5078 | 0.7368 | 44 | 0.6458 | 48 | 1536 |
| topology_heuristic | 0.0240 | 0.0668 | 0.0025 | 0.2321 | 0.3486 | 4 | 0.1250 | 48 | 1536 |

### Generalisation along each shift axis

Each split differs from `test_id` in exactly one respect.

### MAE (lower better) across every split (`results/tables/method_comparison.csv`)

| method | test_id | shift_topo | shift_size | shift_type | shift_multi |
|---|---|---|---|---|---|
| mlp_no_message_passing | 0.0174 | 0.0190 | 0.0196 | 0.0109 | 0.0402 |
| retrieval_knn | 0.0106 | 0.0179 | 0.0177 | 0.0090 | 0.0337 |
| surrogate_ensemble | 0.0153 | 0.0176 | 0.0186 | 0.0103 | 0.0356 |
| surrogate_single | 0.0146 | 0.0172 | 0.0183 | 0.0098 | 0.0343 |
| tabular_gbt | 0.0122 | 0.0142 | 0.0149 | 0.0057 | 0.0360 |
| tabular_ridge | 0.0246 | 0.0243 | 0.0148 | 0.0369 | 0.0411 |
| topology_heuristic | 0.0240 | 0.0259 | 0.0249 | 0.0067 | 0.0434 |

### within-scenario Spearman (higher better) across every split (`results/tables/method_comparison.csv`)

| method | test_id | shift_topo | shift_size | shift_type | shift_multi |
|---|---|---|---|---|---|
| mlp_no_message_passing | 0.4018 | 0.5008 | 0.3703 | 0.4853 | 0.4841 |
| retrieval_knn | 0.7368 | 0.5879 | 0.4527 | 0.5166 | 0.5283 |
| surrogate_ensemble | 0.5715 | 0.6340 | 0.5059 | 0.5010 | 0.6269 |
| surrogate_single | 0.5824 | 0.6629 | 0.5141 | 0.5289 | 0.6536 |
| tabular_ridge | 0.4866 | 0.5634 | 0.3656 | 0.4914 | 0.5911 |
| topology_heuristic | 0.3486 | 0.2179 | 0.2402 | 0.5047 | -0.4245 |

**The retrieval result is the interesting one.** Nearest-neighbour scenario
retrieval is the *best* method in-distribution on both magnitude and rank — it
beats the surrogate's within-scenario Spearman by 0.165, which is 3.86x the noise
scale, so that defeat is real and is reported as one. It then degrades under
shift while the surrogate holds: on the larger-network split the surrogate is
ahead by 2.50x the noise scale (survives) and on unseen topologies by 1.41x
(suggestive). That is exactly the behaviour the two approaches should have —
retrieval interpolates within its training set, the surrogate has learned
something transferable — and it is the clearest evidence here that the surrogate
is not merely memorising. It is also, honestly, a *one-axis* win: the unseen-
mechanism and multi-point splits are both inside noise.

**The no-message-passing result is the strongest.** With the same features, the
same loss, the same schedule and a parameter budget matched to 1.025x, removing
message passing costs 0.18 of within-scenario Spearman (0.582 to 0.402) and is the
*only* ablation whose MAE damage exceeds the seed-noise scale. A model that can
see a node's own attributes but not its neighbours is measurably worse at ordering
demand points, and that is the cleanest demonstration this setup can produce that
topology carries signal the node features do not.

### Every claimed gain, placed against the noise scale

### Every claimed gain against the run-to-run noise scale

A difference between two single runs is inside noise unless it exceeds `sqrt(2) * sd` for that metric, from `seed_variance.csv`.

The surrogate is compared against **two** baselines: the reference-style feature model it is meant to replace, and whichever baseline is actually strongest on that split. Comparing only against the reference would flatter it.

| split | metric | comparison | baseline | surrogate | baseline_value | gain | noise_scale | verdict |
|---|---|---|---|---|---|---|---|---|
| test_id | mae | vs reference | tabular_gbt | 0.01534 | 0.01225 | -0.00309 | 0.00068 | -4.52 -> **worse, outside noise** |
| test_id | mae | vs best baseline | retrieval_knn | 0.01534 | 0.01055 | -0.00479 | 0.00068 | -6.99 -> **worse, outside noise** |
| test_id | spearman_within_scenario | vs best baseline | retrieval_knn | 0.57147 | 0.73677 | -0.16530 | 0.04288 | -3.86 -> **worse, outside noise** |
| shift_topo | mae | vs reference | tabular_gbt | 0.01765 | 0.01416 | -0.00349 | 0.00052 | -6.68 -> **worse, outside noise** |
| shift_topo | mae | vs best baseline | tabular_gbt | 0.01765 | 0.01416 | -0.00349 | 0.00052 | -6.68 -> **worse, outside noise** |
| shift_topo | spearman_within_scenario | vs best baseline | retrieval_knn | 0.63398 | 0.58789 | 0.04609 | 0.03259 | +1.41 -> suggestive |
| shift_size | mae | vs reference | tabular_gbt | 0.01864 | 0.01485 | -0.00379 | 0.00017 | -22.21 -> **worse, outside noise** |
| shift_size | mae | vs best baseline | tabular_ridge | 0.01864 | 0.01478 | -0.00386 | 0.00017 | -22.62 -> **worse, outside noise** |
| shift_size | spearman_within_scenario | vs best baseline | retrieval_knn | 0.50594 | 0.45273 | 0.05321 | 0.02126 | +2.50 -> survives |
| shift_type | mae | vs reference | tabular_gbt | 0.01028 | 0.00569 | -0.00459 | 0.00044 | -10.50 -> **worse, outside noise** |
| shift_type | mae | vs best baseline | tabular_gbt | 0.01028 | 0.00569 | -0.00459 | 0.00044 | -10.50 -> **worse, outside noise** |
| shift_type | spearman_within_scenario | vs best baseline | retrieval_knn | 0.50097 | 0.51658 | -0.01561 | 0.02411 | -0.65 -> inside noise |
| shift_multi | mae | vs reference | tabular_gbt | 0.03563 | 0.03602 | 0.00039 | 0.00253 | +0.15 -> inside noise |
| shift_multi | mae | vs best baseline | retrieval_knn | 0.03563 | 0.03371 | -0.00192 | 0.00253 | -0.76 -> inside noise |
| shift_multi | spearman_within_scenario | vs best baseline | tabular_ridge | 0.62692 | 0.59108 | 0.03583 | 0.05526 | +0.65 -> inside noise |

### Paired per-row significance tests

### Paired per-row tests vs the reference model (`results/tables/statistical_tests.csv`)

Unit of analysis is a **row**, so this compares two sets of weights, not two methods.

| method (abs error) | mean | ref mean | delta | CI low | CI high | p (Holm) | Cohen's d | n |
|---|---|---|---|---|---|---|---|---|
| surrogate_ensemble.abs_error | 0.01534 | 0.01225 | 0.00309 | 0.00193 | 0.00416 | 0.00000 | 0.13652 | 1536 |
| surrogate_single.abs_error | 0.01458 | 0.01225 | 0.00234 | 0.00098 | 0.00363 | 0.00000 | 0.08696 | 1536 |
| mlp_no_message_passing.abs_error | 0.01740 | 0.01225 | 0.00515 | 0.00497 | 0.00532 | 0.00000 | 1.45799 | 1536 |
| tabular_ridge.abs_error | 0.02461 | 0.01225 | 0.01236 | 0.01099 | 0.01375 | 0.00000 | 0.45870 | 1536 |
| retrieval_knn.abs_error | 0.01055 | 0.01225 | -0.00169 | -0.00423 | 0.00046 | 0.00001 | -0.03535 | 1536 |
| topology_heuristic.abs_error | 0.02399 | 0.01225 | 0.01175 | 0.01094 | 0.01257 | 0.00000 | 0.73508 | 1536 |

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
| surrogate | 0.3333 | 0.5333 | 0.5546 | 0.4333 | 0.5167 | 0.4701 | 0.5045 | 0.4250 | 0.3378 | 0.2071 | 0.0007 | 24.1667 | 0.8179 | 2.6186 | 3.3960 |
| tabular_gbt | 0.0000 | 0.1000 | 0.9943 | 0.0500 | 0.1667 | 0.9791 | 0.1660 | 0.2583 | 0.7724 | not measured | 0.0000 | 1.0000 | 0.0071 | 2.6186 | 600.9970 |
| tabular_ridge | 0.2000 | 0.6667 | 0.7029 | 0.4167 | 0.6167 | 0.5457 | 0.5749 | 0.5083 | 0.3495 | 0.4324 | 0.0231 | 37.1667 | 0.0010 | 2.6186 | 2734.6600 |
| topology_heuristic | 0.4333 | 0.7667 | 0.4248 | 0.6333 | 0.7500 | 0.1613 | 0.7056 | 0.5667 | 0.0932 | 0.5683 | 1.4484 | 41.3333 | 0.0038 | 2.6186 | 698.6680 |

**`tabular_gbt` produced a constant score for every candidate.** Its recall is therefore 0 for a reason that has nothing to do with topology: on a target that is ~94% exact zeros, the constant that minimises absolute error is 0, and every probe shares the same standardised disruption so the only varying inputs are the disrupted node's own attributes. This is a real property of the reference approach on this task, not a tuning failure - but the linear variant is reported alongside it precisely so the reader can see whether the collapse is specific to the boosted trees.

### Where the feature score and the counterfactual disagree (`results/tables/disagreements.csv`)

36 disagreeing pairs found. Simulator verdict: **counterfactual** 36.

| network | A | A rank (feat) | A rank (cf) | A true loss | B | B rank (feat) | B rank (cf) | B true loss | winner | margin |
|---|---|---|---|---|---|---|---|---|---|---|
| shift_4 | 0 | 1 | 12 | 0 | 45 | 46 | 1 | 0.7430 | counterfactual | 0.7430 |
| shift_4 | 1 | 2 | 13 | 0 | 45 | 46 | 1 | 0.7430 | counterfactual | 0.7430 |
| shift_4 | 2 | 3 | 14 | 0 | 45 | 46 | 1 | 0.7430 | counterfactual | 0.7430 |
| shift_4 | 4 | 5 | 16 | 0 | 45 | 46 | 1 | 0.7430 | counterfactual | 0.7430 |
| shift_4 | 5 | 6 | 17 | 0 | 45 | 46 | 1 | 0.7430 | counterfactual | 0.7430 |

Worked example of the largest disagreement:

> node 0: tier 0, 4 customers, reaches 6 demand points (0 sole-sourced), throughput 158.9, capacity slack 0.15
> 
> node 45: tier 3, 7 customers, reaches 7 demand points (3 sole-sourced), throughput 63.5, capacity slack 0.41

### Reading this table honestly

Two things are true at once and both belong in the summary.

**The counterfactual framing wins the argument this repository set out to make.**
The reference-style feature score is not merely worse, it is close to useless for
this task: recall@10 of 0.05 and a regret fraction of 0.98, meaning a planner
acting on it averts 2% of the impact an oracle would have averted. Every one of
the 36 pairs where the feature ranking and the counterfactual ranking disagree is
resolved by the simulator in favour of the counterfactual. The worked example is
the argument in one line: the node the feature score ranks **first** has a true
service loss of **exactly zero**, and the node it ranks **last of 46** has the
largest true loss in the network, because that node is a sole source for three
demand points and the feature score has no way to see that.

**The learned surrogate is not the thing that wins it.** A hand-weighted composite
of six topology signals — sole-source reach, downstream reach, path betweenness,
BOM depth, inverse capacity slack, throughput — beats the trained graph network on
every criticality metric: recall@10 0.633 against 0.433, regret fraction 0.161
against 0.470, full-vector Spearman 0.568 against 0.207. The heuristic costs
milliseconds and no training at all.

**Why the surrogate underperforms here, specifically.** Its score spread across
candidates is 0.0007, against a true spread of 0.13 — it is *nearly constant* on
the criticality probes, and it under-predicts the largest impacts by more than an
order of magnitude (max predicted 0.052 against max true 0.748). Its
within-scenario ranking is good (Spearman ~0.58) but criticality ranking is a
purely **cross-probe** comparison, and its pooled cross-scenario correlation is
only ~0.35. The model learned "which demand point suffers most in this scenario"
much better than it learned "how bad is this scenario compared to that one" — and
only the second is what a criticality sweep needs.

That is a specific, actionable diagnosis rather than a shrug, and it points at the
zero-inflated target: an L1 objective on a distribution that is 93.5% zeros
rewards shrinking every prediction toward zero, which preserves local ordering and
destroys global scale.

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
| 0.050 | 0.000 | 0.000 |
| 0.100 | 0.074 | 0.000 |
| 0.250 | 0.126 | 0.000 |
| 0.500 | 0.201 | 0.000 |
| 1.000 | 0.369 | 0.367 |
| 2.000 | 0.752 | 0.433 |
| 5.000 | 1.000 | 0.433 |
| 10.000 | 1.000 | 0.433 |

Recall@10 of the true critical set, averaged over the evaluated networks.

### The honest reading

**The per-scenario speed-up is real**: 50.3 ms for one simulator counterfactual
against 4.41 ms for one screened scenario in a batched surrogate pass, a measured
**11.4x**, both with 8 warm-up iterations and 25 timed repeats on the same machine
state and reported as median with IQR.

**The decision-quality experiment does not favour the surrogate at any budget
tested.** The budget curve is the honest test and the surrogate loses it outright:
below its own fixed cost it returns nothing, at 1 s it ties the simulator
(0.367 against 0.369), and from 2 s onward exhaustive search pulls away to recall
1.0 while the surrogate plateaus at 0.433 because its own ranking quality caps it.
The reason is arithmetic rather than modelling — a 46-candidate sweep costs the
oracle about 2.3 s, so at this network size there is simply no regime in which the
exact answer is unaffordable. A surrogate for this task earns its keep only when
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

**The boosted-tree reference model collapses to an exact constant on the
criticality probes.** Its recall@k is therefore 0 for a reason that has nothing to
do with topology: on a target that is ~94% exact zeros the constant minimising
absolute error is 0, and every probe carries the same standardised disruption so
the only varying inputs are the disrupted node's own attributes. This is a real
property of the reference *approach* on this task rather than a tuning failure,
but presenting it without qualification would be a straw man. The linear variant
is reported alongside it, and every method's score spread is now in the table so
a degenerate ranking is visible rather than inferred.

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

