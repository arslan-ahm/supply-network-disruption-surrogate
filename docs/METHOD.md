# Method

This document specifies the simulator, the surrogate, the baselines and the
counterfactual analysis exactly enough to reimplement them, and gives the
reasoning behind the design decisions that constrain everything else.

---

## 1. The argument

The reference approach this project replaces scores suppliers. Features about a
supplier go into a classifier, and out comes a risk number. Two things are wrong
with that, and they are structural rather than a matter of model quality.

**Risk in a supply network is not a property of a node.** It is a property of the
node's position. A supplier with excellent financials, ample capacity and a clean
audit history that happens to be the *only* qualified source for a component
feeding four assembly plants is more dangerous than a shaky supplier with three
alternates. No amount of feature engineering on the supplier's own attributes
recovers this, because the relevant fact — "no one else can make this part" — is
a statement about the *group* of suppliers around it.

**A score does not answer the planner's question.** The question is "what happens
to customer service level if this link fails next month, when does it bite, and
how long until we recover". That is a counterfactual, and a counterfactual has a
well-defined true answer that can be obtained by running the mechanism forward.

So: **the model is a learned surrogate for a mechanistic simulator, and the
output is a counterfactual, not a score.** The simulator provides an exact
ground-truth oracle, which is a rare luxury — every claim about ranking quality
in this repository is checked against the right answer rather than against a
proxy.

### The methodological trade this makes

Everything here is synthetic. That is a genuine strength and a genuine limitation
and both should be stated plainly.

*Strength.* The simulator **is** the data source, so labels are exact rather than
annotated, the counterfactual is available for every candidate rather than only
for events that happened, and the distribution can be shifted deliberately along
named axes. There is no label noise, no annotator disagreement, and no survivor
bias in which disruptions got recorded. For studying *whether a surrogate can
learn to imitate a propagation mechanism*, this is the cleanest possible setup.

*Limitation.* Results transfer to real supply networks exactly insofar as the
simulator's mechanics are right. **No real-world validation is claimed anywhere
in this repository.** A reader should treat every number as a statement about
this simulator, not about any particular company's network. The optional CSV
loader exists so that a user can substitute their own topology, but their own
*mechanics* would still be these mechanics.

---

## 2. The network model

### 2.1 Structure

A network is a directed acyclic graph over five echelons:

```
tier 0  raw suppliers      (no inputs; capacity-constrained producers)
tier 1  component makers
tier 2  assembly plants
tier 3  distribution centres
tier 4  demand points      (no production; consume and report service level)
```

Node ids are assigned in ascending tier order, so `range(n)` is already a
topological order. The simulator relies on this and `SupplyNetwork.validate()`
enforces it.

### 2.2 Input groups — the central modelling decision

A node does not have a flat list of suppliers. It has a list of **input groups**,
and

- suppliers **within** a group are substitutable, with sourcing shares
  $w_{uv} \ge 0$ summing to one over the group;
- groups are **complementary** — every group must deliver or the node cannot
  produce (a Leontief production function).

Group $g$ of node $v$ carries a BOM coefficient $q_{v,g}$: units of that group's
material consumed per unit of $v$'s output.

This distinction is the whole reason the repository exists. A group with one
member is a **hard single point of failure**. A group with three members is a
soft one. In a flat supplier table these look identical, because the difference
is not an attribute of any supplier — it is the structure around them.

It also gives a precise definition of the thing everybody gestures at:

> **Sole-source reach** of node $v$: the number of demand points reachable from
> $v$ along paths where *every* group traversed is sole-sourced.

If that number is large, removing $v$ removes the material entirely. This is
computed exactly by `SupplyNetwork.sole_source_reach()` in one reverse sweep, and
it is the single strongest signal available to the topology heuristic — and the
one the reference-style feature model cannot see.

### 2.3 Generation

`generate_network` samples suppliers with **preferential attachment** biased
towards the customer's own **region**. This produces the two properties that make
real supply networks fragile in ways node-level models cannot see:

- a heavy-tailed out-degree distribution (a few hub suppliers serve very many
  customers), and
- geographic concentration, so a regional event hits many nodes at once —
  the correlated failure mode that motivated most of the post-2020 resilience
  literature (Ivanov & Dolgui, 2020).

Sourcing shares are drawn from `Dirichlet(4)` rather than `Dirichlet(1)`. With
concentration 1, many "nominally dual-sourced" groups came out with a 97/3 split,
which is sole-sourced in all but name and made the `sole_source_prob` knob
meaningless.

**Capacity and inventory are derived, not sampled.** Expected throughput is
propagated up the BOM from demand:

$$\text{thr}(u) = \sum_{v \in \text{customers}(u)} q_{v,g(u,v)} \, w_{uv} \, \text{thr}(v)$$

with $\text{thr}(j) = \bar d_j$ at demand points. Then capacity is
$\text{thr}(1+\text{slack})$ and base stock is $\text{thr} \times$ cover periods.
Sampling capacity independently would make hub suppliers trivially and
unrealistically fragile, and every experiment would be too easy.

---

## 3. The simulator

Discrete time. Within period $t$, in this order:

1. **Arrivals.** Material shipped on edge $(u,v)$ at $t - L_{uv}$ lands in $v$'s
   input stock for the group $u$ serves.
2. **Demand.** Each demand point realises $d_j(t)$, lognormal with mean
   $\bar d_j$ and coefficient of variation $c_j$, scaled by any active demand
   spike. Lognormal rather than truncated normal because demand is positive and
   right-skewed.
3. **Requirement propagation**, in reverse topological order. A demand point
   requires $d_j(t) + \text{backlog}_j$. A producing node requires

   $$r_v = \min\Big(\underbrace{O_v}_{\text{orders received}} + \max(0, S_v - \text{fg}_v),\ \ \text{cap}_v(t)\Big)$$

   and then orders from each group $g$

   $$\text{ord}_{v,g} = q_{v,g} r_v + \max\big(0,\ S^{\text{in}}_{v,g} - \text{raw}_{v,g} - \text{in-transit}_{v,g}\big)$$

   split across the group's suppliers by their shares.
4. **Production and shipment**, in forward topological order:

   $$p_v = \min\Big(r_v,\ \ \text{cap}_v(t),\ \ \min_g \tfrac{\text{raw}_{v,g}}{q_{v,g}}\Big)$$

   Inputs are consumed, finished goods increase, and the node ships against its
   customers' orders — fully if it can, **pro rata** if it cannot.
5. **Fulfilment.** Each demand point serves backlog first, then current demand.
   The shortfall is recorded and (if backlogging is on) carried, up to a cap.

### 3.1 Why requirements move up within a period and material moves down across periods

That asymmetry is what a lead time *is*. An order is seen immediately; the goods
are not. This produces the delayed, amplified shortage wave the surrogate has to
learn. A simulator that resolved material in the same period as the order would
make every disruption instantaneous and the time-to-impact target meaningless.

### 3.2 The clamp on ordering

`r_v` is capped at the node's disrupted capacity. Without this, a dead plant would
keep passing requirement upstream, which is not how a plant behaves — if you
cannot convert the material you do not order it.

### 3.3 Deliberate simplifications, stated rather than hidden

**Static sourcing shares (default).** A node keeps ordering from a failed supplier
at its nominal share. `allow_resourcing: true` re-normalises over suppliers with
positive capacity. Both are supported because the difference between them is
exactly the value of qualified alternates. Reporting one without the other would
be a choice dressed up as a fact. The measured difference on the diamond fixture:
static gives a 0.5 steady-state service loss, re-sourcing gives 0.0.

**No cost model, no order optimisation.** Base-stock levels are given, not
optimised. This is a *propagation* model, not an inventory-optimisation model.

**One item per node.** A node produces a single item. Multi-item plants would add
an index everywhere and change no conclusion here.

### 3.4 Two things found by testing that constrain the experimental setup

**Warm-up must exceed the cumulative lead time to a demand point.** The pipeline
starts empty, so a short warm-up measures a pipeline-fill transient rather than a
steady state. Measured on four generated networks whose maximum cumulative lead
time to demand is 18.8 periods:

| warm-up | 10 | 16 | 20 | 26 | 32 |
|---|---|---|---|---|---|
| baseline fill rate | 0.9557 | 0.9748 | 0.9944 | 0.9973 | 0.9997 |

The shipped configuration uses `warmup: 22` with lead times capped so the maximum
cumulative lead time is 16.

**Backlogging plus unbounded capacity produces a bullwhip buffer.** With
backlogging on, every backlogged unit is re-ordered on top of fresh demand; with
effectively infinite capacity the chain over-produces during the warm-up
transient, the excess arrives late, and it settles as inventory that then absorbs
a subsequent outage completely. This is the bullwhip effect arising in the
simulator's own warm-up — real behaviour, and a mild validation that the mechanics
are doing something non-trivial, but it makes hand-solvable tests non-hand-solvable.
The exact arithmetic tests therefore use `backlog=False`; the shipped experiments
use `backlog=True` with a cap of 6 periods of mean demand, since unbounded backlog
would make recovery time a statement about the cap.

### 3.5 What the tests pin down

The simulator is the ground truth, so it is tested against arithmetic:

- Single chain, ample capacity, no disruption → fill rate exactly 1.0.
- Total outage with no inventory on an $n$-leg chain with lead $L$ → shortage
  begins at exactly period $nL$ and equals the entire demand, verified for
  $n \in \{3,4\}$ and $L \in \{1,2,3\}$.
- A 4-period outage on a 2-leg chain → short at periods 2–5, recovered at 6,
  recovery time exactly 2.
- Dual sourcing at 50/50 with static shares → service loss converges to exactly
  0.5 as the horizon grows (measured 0.425, 0.475, 0.4925 at horizons 20, 60, 200).
- Material conservation residual below 1e-9 in every period of every run.
- Monotonicity in both duration and severity.

---

## 4. Disruptions

Six mechanisms, chosen because they behave *differently in a network*, not to
make six:

| kind | what it does | why it is here |
|---|---|---|
| `supplier_outage` | capacity → 0 | the canonical single-point-of-failure probe |
| `capacity_reduction` | capacity × (1−s) | partial; whether it hurts depends on slack |
| `lead_time_inflation` | outbound lanes × (1+s) | no material lost, only timing — the failure mode inventory absorbs and capacity slack does not |
| `demand_spike` | demand × (1+s) | the only mechanism that propagates **upstream** |
| `lane_closure` | one edge carries nothing | distinguishable from an outage only with edge-level structure |
| `regional_event` | every node in a region loses capacity | correlated; the reason geography is in the generator |

Timing is relative to the **measurement window**, so `start=0` is the first
measured period and warm-up is always clean.

Composition: capacity and demand multiply, lead-time inflation adds. That is the
only rule under which a double hit is guaranteed at least as bad as a single one,
and the monotonicity is asserted in the tests.

Lead-time inflation applies at **departure**: a shipment that left before a port
closure is not retroactively delayed.

### 4.1 A bug this found

The in-transit ring buffer was originally sized from the *nominal* maximum lead
time. An inflated shipment then wrapped around the buffer and arrived **early**.
All 120 sampled `lead_time_inflation` scenarios produced exactly 0.0000 impact,
which is how it surfaced. The buffer is now sized from the maximum lead time that
can actually occur. Regression test:
`test_lead_time_inflation_delays_by_the_inflated_amount`.

Measured mean total service loss per demand point, 120 scenarios per kind on four
generated networks, before and after the fix — note only the inflation row moves,
which is what makes the fix credible:

| kind | before | after |
|---|---|---|
| supplier_outage | 0.0102 | 0.0102 |
| capacity_reduction | 0.0031 | 0.0031 |
| **lead_time_inflation** | **0.0000** | **0.0017** |
| demand_spike | 0.0110 | 0.0110 |
| lane_closure | 0.0018 | 0.0018 |
| regional_event | 0.0500 | 0.0500 |

---

## 5. Targets

Every scenario is a **paired** counterfactual: the disrupted run and its baseline
use the *same* demand realisation, so the difference isolates the disruption
rather than mixing in demand noise. With independent draws the label noise swamps
the effect of a mild disruption. Because the baseline cannot depend on the
disruption, it is computed once per (network, seed) and reused — which also
halves the oracle's cost in the efficiency comparison, as any real user would do.

Per demand point:

- **service loss** = baseline fill rate − disrupted fill rate, clipped at 0. The
  clip exists for float noise; a test asserts it is never active for a real
  negative.
- **extra unmet demand**, normalised by mean demand.
- **time to impact**: periods from the disruption's start to the first period the
  point is short by more than 1% of its mean demand. **NaN if never affected.**
- **recovery time**: periods from the disruption's *end* until the point stops
  being short for the rest of the window. **NaN if it never recovers.**
- **trajectory**: excess unmet per period, normalised by mean demand.

**Undefined is NaN, never zero.** An unaffected demand point does not have a
time-to-impact of zero; it has none. Every timing metric reports how many rows
contributed, because in this dataset only a minority do.

---

## 6. The surrogate

Encoder–processor–decoder over the graph, in the learned-simulator tradition of
Sanchez-Gonzalez et al. (2020) and Pfaff et al. (2021), applied to a supply
network following Kosasih & Brintrup (2021).

### 6.1 Message passing, hand-rolled

**No torch-geometric, no DGL.** Both are excellent and both compile against a
specific torch/CUDA pair; a reader running `uv sync` on a different torch build
gets a linker error instead of a result. The message passing needed here is a
gather, an MLP and an `index_add_` — about forty lines. Forty lines of code for a
reproducibility guarantee is a good trade for a repository whose entire point is
that its numbers can be re-derived.

Batching is the standard trick: a batch of graphs is one big disconnected graph,
node features concatenated and edge indices offset. No padding, no masking, and
per-node cost proportional to real work — which matters because the
larger-network split has roughly twice the nodes and must run through the same
code path.

### 6.2 Two directions

Material (and therefore shortage) flows supplier → customer. Orders (and therefore
requirement) flow customer → supplier. These are different physics, so the default
runs **two** message functions and concatenates their outputs.
`bidirectional: false` removes the upstream pass and is the ablation.

This is also why `demand_spike` is the held-out mechanism: it is the only
intervention that travels purely upstream, so holding it out tests the
bidirectional design directly rather than testing memorisation.

### 6.3 Two aggregators

Mean **and** elementwise max. A shortage is a *bottleneck*: the binding constraint
on production is the worst input, not the average one. Aggregating with both lets
the update MLP express a min/max-like rule, which mean pooling alone cannot —
the aggregator-expressivity point of Gilmer et al. (2017). Nodes with no incoming
messages get exactly zero from both, not NaN and not −inf.

### 6.4 Edge conditioning

Messages see `[h_src, h_dst, edge_features]` — the full relational triple of
Gilmer et al. (2017) rather than the `h_src`-only form of a GCN
(Kipf & Welling, 2017). Two otherwise identical suppliers differ by lead time,
BOM coefficient and sourcing share, and those are properties of the
*relationship*.

### 6.5 Design decisions that constrain the rest

**LayerNorm, not BatchNorm.** A batch is a variable-size set of graphs whose node
count changes every step, and batch statistics over "all nodes in this batch" mix
tier-0 suppliers with demand points, whose feature distributions are nothing
alike. LayerNorm is also what makes single-graph inference numerically identical
to batched inference — a property both the latency benchmark and the
counterfactual screening rely on.

**Residual updates.** Without them, at three or four rounds the stack lost the
node's own disruption flags by the final layer and the model predicted a
network-average impact for every demand point.

**Softplus on the impact head.** Service loss is non-negative by construction. A
linear head spent its early epochs learning that floor and emitted small negative
values on easy scenarios, which corrupted the bottom of the criticality ranking —
where most of the nodes are.

**No teacher forcing in the trajectory decoder.** At screening time there is no
ground truth to feed, so the decoder is trained on its own output and train and
inference behaviour stay identical.

### 6.6 Heads

| head | output | trained by |
|---|---|---|
| impact | scalar service loss | L1 + Gaussian NLL |
| logvar | predictive log-variance | Gaussian NLL |
| quantiles | 5th / 50th / 95th percentile | pinball |
| traj | 8-period excess-unmet (`model.traj_horizon`) | tapered L1 |
| timing | time-to-impact, recovery time | masked L1 |

### 6.7 Loss design

**L1, not L2, on impact.** The label distribution is a large mass at exactly zero
(most disruptions are absorbed — 93.5% of training *rows* at the `<= 1e-4`
threshold in `dataset.csv`, and 92.48% of all evaluated rows at exact equality,
from `per_item.csv`) with a long right tail.
Squared error puts nearly all its gradient on the tail; the first version trained
that way had a decent MAE on big events and ranked the bottom two-thirds of nodes
at random, which is useless, because separating small from zero *is* the screening
job.

**This choice is defensible for the network and indefensible for a tree, and §7
records what happened when it was copied across.** L1 on a 92.5%-zero target pulls
the network's predictions toward zero — visible as the under-prediction of the tail
in `docs/RESULTS.md` §5 — but it does not stop it learning, because every gradient
step moves a shared function that must also account for the nonzero rows. A
regression tree has no shared function to move: each leaf takes the exact L1
minimiser of its own samples, which is 0. Transplanting "use L1 to match the
surrogate" into the boosted-tree baseline therefore did not make the comparison
fairer, it deleted the baseline.

**Timing terms are divided by the horizon.** Timing targets are in periods (0–30)
while impact is a fraction (0–1). Left unscaled, the timing L1 came out around 12
against an impact L1 around 0.05, so even at `w_timing = 0.2` more than 98% of the
gradient went to an auxiliary head and the headline target was effectively
untrained. The head still emits periods, so predictions stay readable.

**Trajectory loss is tapered by $1/\sqrt{1+t}$.** Early periods are what a planner
acts on. Untapered, capacity went to the long flat tail where the target is almost
always zero.

**Both L1 and NLL on impact.** Pure NLL lets the model cut its loss by inflating
variance on hard rows instead of fitting them, and the point estimate is what the
ranking uses.

**Masking, not imputation.** Timing terms are masked to rows where the simulator
defined the target, and the contributing count is logged every epoch.

### 6.8 Uncertainty

Both routes are implemented and both are measured:

- **Gaussian**, from the heteroscedastic head, combined across a deep ensemble
  (Lakshminarayanan et al., 2017) as $\bar\sigma^2_{\text{alea}} + \mathrm{Var}(\mu)$.
  Reporting only the second would understate uncertainty on easy-but-noisy rows;
  only the first would miss shifted-topology disagreement.
- **Quantile**, from the pinball head, which assumes nothing about shape.

The Gaussian route assumes symmetry, which a spike-at-zero-plus-right-tail target
does not have, so it is *expected* to be the worse-calibrated of the two. Whether
it is, is reported in `docs/RESULTS.md` rather than assumed.

Ensemble members differ only in initialisation and batch order — no bagging.
Bagging would shrink each member's training set and confound "ensembling helps"
with "less data hurts".

---

## 7. Baselines

All four are run, on the same rows, in the same order, with the same targets.

**`tabular_gbt` / `tabular_gbt_l1` / `tabular_ridge` — the reference approach.** A
flat feature row per (disrupted node, demand point): attributes of the disrupted
node, attributes of the demand point, and eight network-level summary statistics.
**No relational term connects the two.** That omission is the thing under test.

It is treated generously on purpose: it receives three graph-derived columns
(downstream reach, sole-source reach, cumulative lead time to demand) that a
per-supplier risk model would not normally have. Its structural limitation
remains: for a multi-point scenario it can only describe one "primary" disrupted
node, because a single-node risk score has no way to represent two simultaneous
failures.

**The GBT is trained with squared error, and the reason is a corrected mistake.**
The first version used `absolute_error`, chosen so the baseline would optimise the
same thing as the surrogate — otherwise the comparison is about loss functions
rather than about representations. That argument is right in general and wrong
here. On a target that is 92.5% exact zeros, L1's minimising constant is the
median, the median is exactly 0, and a boosted-tree regressor therefore
initialises at 0, finds every leaf's L1-optimal value to be 0, early-stops after
10 rounds and emits a single distinct prediction. It did: exactly 0.000000 for all
20,624 rows of the shipped comparison. The variant is kept as `tabular_gbt_l1`,
labelled degenerate by construction, and reported next to the working one, because
the contrast is a cleaner illustration of the zero-inflation problem than any
prose. See `docs/RESULTS.md` §8.7.

The asymmetry with §6.7 is the part worth understanding. The surrogate uses L1 on
the same target and does *not* collapse — it merely under-predicts the tail
(§5). The difference is where the minimisation happens. Gradient descent on a
shared parameter vector takes a step whose direction is the sign of the residual
but whose effect is spread across a function that must also fit the nonzero rows;
a regression tree assigns each leaf the **exact** minimiser of the loss over the
samples that fall in it, and on this target that minimiser is 0 for essentially
every leaf. Same objective, same data, categorically different failure.

**Both trivial constants are baselines.** `constant_zero` (emit 0.0) and
`constant_train_mean` (emit 0.012783) are run and reported on every split. On a
target this zero-inflated they are not padding: `constant_zero` is the MAE-optimal
predictor, so no MAE in this repository is interpretable without it, and its
absence is the reason the collapse above went unnoticed. Every method's
distinct-prediction count is recorded, and a method that is not declared
constant-by-design and emits fewer than two distinct values raises
`DegenerateBaselineError` rather than entering a results table.

**`mlp_no_message_passing`.** The surrogate with `layers = 0`: identical features,
loss, training loop and heads; no node ever sees a neighbour. The node-only trunk
is widened so the parameter count matches (57,160 vs 55,752 at the shipped width —
within 2.5%),
because the first version of this ablation simply deleted the processor and came
out 3.6× smaller, which would have confounded "the graph helps" with "more
parameters help".

**`topology_heuristic`.** No learning: a hand-weighted composite of sole-source
reach, downstream reach, path betweenness, BOM depth, inverse capacity slack and
throughput. Weights are set from what the literature treats as important
(Simchi-Levi et al., 2015; Ivanov & Dolgui, 2020) rather than fitted — a fitted
heuristic is just a linear model on graph features, which the ridge baseline
already covers. Because its raw scale is arbitrary (a sum of standardised
signals), a **two-parameter affine map is fitted on the training split only** so
its MAE is in service-level units and therefore interpretable; without this it
reported an MAE of 2.4 against a target bounded near 1, which says nothing about
the heuristic and everything about its units.

**`retrieval_knn`.** Inverse-distance-weighted 8-nearest-neighbour over the same
tabular descriptor. The "just look it up" objection. Strong in-distribution
because the training set contains many near-duplicate scenarios; its behaviour
under topology shift is the interesting part. Its cost scales with the training
set and the surrogate's does not, which is one of the honest arguments for the
surrogate, and that cost is included in the efficiency accounting.

---

## 8. Generalisation splits

Each shift split differs from `test_id` in **exactly one** respect, which is what
makes the gap attributable.

| split | topology | disruption |
|---|---|---|
| `train` / `val` / `test_id` | pool A (12 networks) | single-point, all kinds but the held-out one |
| `shift_topo` | pool B (6 unseen networks) | same |
| `shift_size` | pool C (4 unseen, ~2× nodes) | same |
| `shift_type` | pool B | **only** the held-out kind (`demand_spike`) |
| `shift_multi` | pool B | 2–3 simultaneous |

`shift_type` and `shift_multi` reuse pool B rather than pool A deliberately: a
reader would otherwise be unable to tell whether a gap came from the new mechanism
or from the new topology. The topology shift is held constant and `shift_topo` is
the reference point for both.

`demand_spike` was chosen as the held-out mechanism over `lane_closure` on
measured grounds — demand spikes leave 55% of scenarios with no impact at all,
against 81% for lane closures and 91% for lead-time inflation, so a lane-closure
holdout would mostly have tested whether the model can predict zero.

---

## 9. Counterfactual criticality

For each candidate node, a **standardised** probe: same start, same duration, same
severity. Fixing the probe is what turns the ranking into a statement about
network position; with a randomly drawn disruption per candidate, a node could
rank highly for having drawn a longer outage.

- **Oracle**: one simulator run per candidate (sharing one baseline).
- **Surrogate**: one batched forward pass over all candidates.
- **Tabular / heuristic**: the same candidates scored their way.

Scores are **summed** over demand points, not averaged: a node that starves four
demand points is worse than one that starves one, and the mean would erase exactly
the difference this project is about.

### 9.1 The disagreement experiment

A pair of candidates qualifies when one is inside the feature score's top-10 and
outside the counterfactual's, and the other is the reverse. The simulator then
adjudicates. Pairs are ranked by the true service level at stake, so the first is
the most consequential disagreement rather than merely the first found.

There is no arguing about who is right in that comparison, because the
counterfactual was actually run. This is what an oracle buys.

### 9.2 Decision regret

The decision model is deliberately simple and stated rather than dressed up:
protecting a node averts exactly its own counterfactual impact, and impacts do not
interact. Both are wrong in a network — protecting two nodes on the same path
averts less than the sum — so this is an **upper bound** on achievable benefit and
the reported regret is a **lower bound** on true regret. It is used anyway because
it is monotone in ranking quality, and the alternative (re-simulating every
subset) is combinatorial.

---

## 10. Efficiency accounting

The framing is **decision quality at a fixed compute budget**: given $N$ seconds,
the simulator can evaluate $k$ candidates exactly while the surrogate screens all
of them approximately. Which finds the true top-10 more reliably?

The simulator strategy spends its budget on randomly chosen candidates and ranks
those it evaluated by their exact impact; candidates it could not afford rank
last. Random because an ordering telling it which to try first would already be
the surrogate's contribution. Averaged over 20 draws, since which candidates it
happens to pick matters a great deal at small budgets.

**Break-even charges the surrogate honestly.** Setup cost includes training *and*
the dataset generation that used the very simulator being replaced. Omitting the
latter is the standard way this comparison gets inflated. A surrogate that takes
an hour to train to save a minute of screening is a bad trade, and the break-even
scenario count says where the line is.

Benchmarks use ≥8 warm-up iterations and ≥25 timed repeats, and report **median
and IQR** rather than mean — on a shared 4-core machine another process can steal
a core mid-measurement, which produces a long right tail that a mean absorbs and a
median does not.

---

## 11. References

- Kipf, T. & Welling, M. (2017). *Semi-Supervised Classification with Graph
  Convolutional Networks.* ICLR.
- Veličković, P. et al. (2018). *Graph Attention Networks.* ICLR.
- Gilmer, J. et al. (2017). *Neural Message Passing for Quantum Chemistry.* ICML.
- Battaglia, P. W. et al. (2018). *Relational Inductive Biases, Deep Learning, and
  Graph Networks.* arXiv:1806.01261.
- Sanchez-Gonzalez, A. et al. (2020). *Learning to Simulate Complex Physics with
  Graph Networks.* ICML.
- Pfaff, T. et al. (2021). *Learning Mesh-Based Simulation with Graph Networks.*
  ICLR.
- Kosasih, E. E. & Brintrup, A. (2021). *A Machine Learning Approach for Predicting
  Hidden Links in Supply Chain with Graph Neural Networks.* International Journal
  of Production Research.
- Ivanov, D. & Dolgui, A. (2020). *Viability of Intertwined Supply Networks:
  Extending the Supply Chain Resilience Angles towards Survivability.*
  International Journal of Production Research 58(10).
- Simchi-Levi, D., Schmidt, W., Wei, Y. et al. (2015). *Identifying Risks and
  Mitigating Disruptions in the Automotive Supply Chain.* Interfaces 45(5).
  (The Ford "time-to-recover" work: risk exposure without probability estimates.)
- Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). *Simple and Scalable
  Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS.
- Nix, D. A. & Weigend, A. S. (1994). *Estimating the Mean and Variance of the
  Target Probability Distribution.* IEEE ICNN. (Heteroscedastic NLL.)
