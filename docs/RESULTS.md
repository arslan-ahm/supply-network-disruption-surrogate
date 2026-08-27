# Results

Every table below is rendered from a CSV in `results/tables/` by
`scripts/report_numbers.py`. No number here was typed by hand. Where a cell reads
`not measured`, no run stands behind it and none was invented.

**Read section 2 before section 3.** The seed study is what makes the comparison
interpretable; without it a table of differences is just a table of differences.

---

## 0. How to read the two fidelity families

The target is **93.5% exact zeros at the row level** — most disruptions are
absorbed and most demand points are unaffected. That single fact governs how every
error number here should be read:

* **A good MAE is cheap.** Predicting approximately zero everywhere scores well
  and is useless for screening.
* **Rank fidelity is the real test.** The planner's question is "which link should
  I look at first", so what matters is whether the ordering is right.

Both are reported for every method on every split, unaggregated, so the reader can
see which one was achieved. A single "accuracy" number would hide the distinction,
and in this task the distinction is the whole point.

`spearman_within_scenario` is computed **within** each scenario and then averaged,
never pooled. Pooling lets the easy between-scenario signal (big disruptions hurt
more than small ones) carry the number; the question of interest is which demand
point inside a given scenario is worst hit.

---

## 1. The data

{{DATASET}}

The zero-inflation is not a defect of the generator; it is what a supply network
does. Capacity slack, inventory buffers and multi-sourcing absorb most single
disruptions, and the ones that get through are the interesting ones. `shift_multi`
has visibly more impact (mean {{MULTI_MEAN}} vs {{TESTID_MEAN}}) because two or
three simultaneous failures are much harder to absorb than one.

---

## 2. Seed variance — the noise scale

Three training runs of the **identical** configuration, differing only in seed.
The run-to-run scale of a difference between two single runs is `√2 · sd`; any
"improvement" smaller than that is noise.

{{SEEDS}}

---

## 3. Method comparison

{{METHODS}}

### Generalisation along each shift axis

Each split differs from `test_id` in exactly one respect.

{{SHIFT}}

### Every claimed gain, placed against the noise scale

{{VERDICTS}}

### Paired per-row significance tests

{{STATS}}

**What these tests do and do not answer.** A paired per-row test conditions on
**one trained model per method**. It correctly answers "are these weights better
than those weights, consistently across rows". It does *not* answer "is this
method better", because that claim treats the **training run** as the sampling
unit and there is one run per method here. No number of test rows fixes an n of 1
in that unit. That is exactly why section 2 exists.

---

## 4. Ablations

One mechanism removed at a time, everything else held fixed: same data, same
seed, same loss, same schedule, same training loop.

{{ABLATION}}

---

## 5. Counterfactual criticality — the headline experiment

{{CRITICALITY}}

---

## 6. Uncertainty

Coverage without sharpness is meaningless — a `[0, 1]` interval covers everything —
so both are reported. Out of distribution, the property that matters is whether
error grows *with* the predicted uncertainty.

{{CALIBRATION}}

---

## 7. Efficiency

{{EFFICIENCY}}

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
| hidden width | 64 | 40 | epoch time |
| message-passing rounds | 3 | 3 | kept — it is the contribution |
| training epochs | 30 | 8 | epoch time |
| ensemble members | 5 | 3 | 5 runs did not fit |
| training scenarios | 2,736 | 1,200 | memory |
| evaluation scenarios per split | 420 | 240 | memory |
| predicted trajectory periods | 24 | 12 | the GRU costs one dispatch per period |

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

