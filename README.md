![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13%20CPU-red)
![Tests](https://img.shields.io/badge/tests-345%20passing-brightgreen)
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

> **Result, up front — and the flattering half is not the whole story.**
>
> **The argument holds, decisively.** Across six unseen networks there are 36
> candidate pairs that the reference-style feature score and the counterfactual
> ranking order oppositely. The simulator resolves **36 of 36 in favour of the
> counterfactual**. The nodes the feature score prefers have a mean *and maximum*
> true service loss of **exactly 0.0000**; the nodes it passes over average
> **0.4288**. In every one of the six networks the node ranked 44th–46th of 46 by
> features is the node the counterfactual puts 1st–3rd, and it is the worst real
> single point of failure in that network.
>
> **The learned model is not what wins it.** A hand-weighted composite of six
> topology signals beats the trained graph network on every criticality metric
> (recall@10 **0.633 vs 0.433**, regret fraction **0.161 vs 0.470**). The
> surrogate is a poor criticality ranker: its score spread across candidates is
> 0.0007 against a true spread of 0.130, and its full-vector rank correlation with
> the truth is **0.21**, against **0.57** for the heuristic, and it under-predicts the worst impacts
> by more than an order of magnitude. **That retracts this project's headline
> claim** — see [Results §5](docs/RESULTS.md).
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
> **never wins at any budget**: it ties the simulator at 1 s (0.367 against 0.369)
> and is beaten from 2 s onward, where exhaustive search reaches recall 1.0 and
> the surrogate plateaus at 0.433.

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

The headline disagreement is exactly this quantity doing its work. The node the
feature model ranks last is a distribution centre with modest throughput and
middling degree — and sole-source reach 3.

## Why a surrogate at all, and where that argument breaks

The simulator answers the planner's question exactly, at one run per candidate.
Screening a 54-node network is ~46 runs; every transport lane is ~150. The
surrogate turns that into one batched forward pass. The honest test is therefore
**decision quality at a fixed compute budget**: given N seconds, the simulator
evaluates a few candidates exactly while the surrogate screens all of them
approximately — which finds the true top-10 more reliably?

| compute budget (s) | simulator recall@k | surrogate recall@k |
|---|---|---|
| 0.05 | 0.000 | 0.000 |
| 0.1 | 0.074 | 0.000 |
| 0.25 | 0.126 | 0.000 |
| 0.5 | 0.201 | 0.000 |
| 1 | 0.369 | 0.367 |
| 2 | 0.752 | 0.433 |
| 5 | 1.000 | 0.433 |
| 10 | 1.000 | 0.433 |

**The surrogate does not win at any budget.** A 46-candidate sweep costs the
oracle about 2.3 s, so there is no regime at this network size where the exact
answer is unaffordable — and below the surrogate's own fixed cost it returns
nothing at all. **A surrogate for this task earns its keep only when the candidate
set is large enough that the oracle is genuinely out of reach, and this repository
does not demonstrate such a regime.** That is the single most important limitation
here, and it is a limitation of the experiment's scale as much as of the model.

## The distinction that governs every number here

The target is **93.5% exact zeros at the row level** — most disruptions are
absorbed, and most demand points are unaffected by any given failure. That single
fact means:

- **A good MAE is cheap.** The boosted-tree reference model has the *best* MAE of
  any method on four of five splits, and produces an exactly constant score on the
  criticality probes. Its recall@10 is 0.05 and its regret fraction is 0.98: a
  planner acting on it averts 2% of what an oracle would.
- **Rank fidelity is the real test**, because the planner's question is "which
  link do I look at first".

Both are reported, unaggregated, for every method on every split, so a reader can
see which one was achieved.

## Measured results

Every table below is rendered from a CSV by `python scripts/report_numbers.py`. Full analysis, including every retraction, is in [docs/RESULTS.md](docs/RESULTS.md).

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

**The tabular baseline is treated generously.** It gets three graph-derived
columns it would not normally have, and its GBT is trained with absolute error to
match the surrogate's objective. Otherwise the comparison would be about loss
functions rather than about representations.

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
