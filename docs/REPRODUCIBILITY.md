# Reproducibility

Every number in this repository is produced by a command in this file. Nothing is
transcribed from a notebook that no longer exists, and every table in
`docs/RESULTS.md` names the CSV it came from.

---

## Environment

The project pins Python 3.12 and installs with [uv](https://docs.astral.sh/uv/).

```bash
uv python install 3.12
uv venv --python 3.12 .venv

# CPU wheels - what the committed results were produced with
uv pip install --python ./.venv/Scripts/python.exe \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch numpy scipy pandas pyyaml matplotlib scikit-learn

uv pip install --python ./.venv/Scripts/python.exe -e . --no-deps
```

Or simply `make setup`.

**Python 3.12 specifically.** The 3.14 torch wheels available in this environment
fail to import on a missing bundled `torchgen`, and the `torchgen` package on
PyPI is an unrelated stub that does not fix it. Use 3.12.

Verified on:

```
python  3.12.13
torch   2.13.0+cpu
numpy   2.5.2
scipy   1.18.1
pandas  2.x
scikit-learn 1.x
```

**No network access at run time.** There are no API keys, no dataset downloads
and no pretrained weights. The simulator generates everything.

---

## The machine the committed numbers came from

Windows 10, 4 CPU cores, 16 GB RAM, **no GPU**, and — importantly for reading the
wall-clock numbers — **shared with several other concurrently running projects**.
Every entry point calls `sndsur.utils.seed.limit_threads(cfg.run.threads)`, which
caps torch to `run.threads` (2 by default, 1 for runs launched alongside another)
and sets `OMP_NUM_THREADS` to match.

That contention is visible in the measurements: the same batch was timed at
600 ms and at 2870 ms minutes apart, purely from other processes' load. This is
why every benchmark reports **median and IQR** rather than a mean, and why the
efficiency conclusions are stated as *ratios measured back-to-back on the same
machine state* rather than as absolute throughput figures.

---

## Reproducing everything

```bash
make all
```

which runs, in order:

Stages that were **not** re-run for the baseline correction in
`docs/RESULTS.md` §8.7, and why: `ablate` (every ablation compares the surrogate
against itself, so no tabular model enters it) and `efficiency` (wall-clock and
setup cost only). See §8.10 of that document.

| stage | command | what it writes |
|---|---|---|
| data | `python scripts/build_dataset.py --config configs/base.yaml` | `results/tables/dataset.csv`, `data/scenarios/*.pkl` |
| seeds | `python scripts/run_seed_study.py --config configs/base.yaml --seeds 0 7 1337` | `seed_runs.csv`, `seed_variance.csv` |
| compare | `python scripts/compare_methods.py --config configs/base.yaml` | `method_comparison.csv`, `statistical_tests.csv`, `calibration.csv`, `generalisation.csv`, `results/runs/<name>/` |
| ablate | `python scripts/run_ablations.py --config configs/base.yaml` | `ablations.csv`, `ablation_statistics.csv` |
| criticality | `python scripts/run_criticality.py --config configs/base.yaml` | `criticality_ranking.csv`, `criticality_summary.csv`, `criticality_detail.csv`, `disagreements.csv`, `budget_curve.csv` |
| efficiency | `python scripts/benchmark_efficiency.py --config configs/base.yaml` | `efficiency.csv`, `break_even.csv` |
| figures | `python scripts/make_figures.py` | `results/figures/*.png` |

**Run the seed study before reading any comparison.** It produces the noise scale
that every claimed improvement is measured against, and without it a table of
differences is not interpretable.

### Order matters in two places

1. `compare` trains the surrogate ensemble and **caches it** under `checkpoints/`,
   keyed by every config field that can change the weights. `criticality` and
   `efficiency` then reuse that cache rather than retraining, which is the only
   reason the whole matrix fits the compute budget. The cached record carries
   `cached: true` so the efficiency table never reports a cache hit as a
   training time.
2. `efficiency` reads `dataset.csv` for the dataset-generation time it charges to
   the surrogate in the break-even calculation. Run `data` first or that column
   is `not measured`.

### The 60-second version

```bash
make smoke      # the entire pipeline on a tiny config
```

---

## Measured wall-clock on this machine

Measured under real conditions, i.e. with other projects competing for the same
four cores. Treat these as an order of magnitude, not a benchmark.

| stage | wall-clock | notes |
|---|---|---|
| dataset generation (5,140 scenarios, 22 networks) | **313 s** | one-time; cached afterwards |
| loading + normalising the cached dataset | ~38 s | 83 MB pickle |
| one training run (10 epochs) | see `results/tables/ablations.csv`, `train_seconds` | varies by a factor of 3 with machine load |
| exhaustive oracle sweep, 46 candidates | ~1.0 s | `results/tables/criticality_ranking.csv`, `sim_seconds` |
| fast test suite (366 tests) | ~90 s | `pytest -m "not slow"` |
| full test suite (375 tests, 9 slow) | **138 s** measured | a bare `pytest tests` -> `373 passed, 2 xfailed`; the 2 are `xfail(strict)` and are *expected* to fail |

The per-scenario simulator and surrogate costs that the efficiency claim rests on
are in `results/tables/efficiency.csv`, measured with 8 warm-up iterations and 25
timed repeats.

---

## Determinism

`sndsur.utils.seed.seed_everything` seeds Python, NumPy and torch in one call.
Every stochastic decision is a pure function of a seed and an index, never of
iteration order: `child_seed(net_id, scenario_id, base=cfg.run.seed)` derives a
seed from a `SeedSequence`, so adding scenarios to one network does not shift any
other network's draws.

Three properties are asserted by tests rather than by assertion in prose:

```bash
# The dataset is byte-identical when regenerated.
pytest tests/test_end_to_end.py::test_dataset_generation_is_deterministic

# The same seed gives max abs difference 0.0 across per-row predictions.
pytest tests/test_end_to_end.py::test_training_is_reproducible_from_its_seed -m slow

# A different seed gives a different model - the guard against a pipeline
# that is accidentally constant.
pytest tests/test_end_to_end.py::test_different_seeds_give_different_models -m slow

# The simulator itself is deterministic.
pytest tests/test_simulator.py::test_simulation_is_deterministic
```

`test_training_is_reproducible_from_its_seed` asserts
`np.abs(run_a - run_b).max() == 0.0` exactly, not approximately.

### Observed in the wild, not only in a test

The ensemble was trained twice from separate process launches, hours apart, on a
machine whose load differed substantially between the two. The training traces are
identical to five decimal places:

```
run 1   epoch 0  loss 0.07373  val -0.28690      run 2   epoch 0  loss 0.07373  val -0.28690
        epoch 5  loss -1.26929 val -1.38481              epoch 5  loss -1.26929 val -1.38481
```

Both appear in the committed stage logs. This is the property that lets the
checkpoint cache be trusted: reusing a cached ensemble is equivalent to retraining
it, so the criticality and efficiency stages measure the same weights the method
comparison evaluated.

### What is *not* deterministic — measured, and narrower than previously claimed

Wall-clock timings, obviously. But the exact-zero determinism check is a
**within-process** guarantee, not a within-machine one. This document previously
said within-machine; that was too strong, and `docs/RESULTS.md` §8.8 has the
measurements. Re-running `compare` at the same seed on the same machine reproduces
the *training trace* exactly (`epoch 5 loss -1.26929 val -1.38481`,
`best_val_loss -1.49757`) but not the per-row predictions:

| quantity | max abs difference across two launches |
|---|---|
| simulator truth | **0.0** |
| `topology_heuristic` | **0.0** |
| `mlp_no_message_passing` (`layers = 0`, retrained) | **0.0** |
| `surrogate_single` | 4.1e-06 |
| `surrogate_ensemble` | 9.4e-04 |
| `tabular_ridge` | 2.2e-04 |
| `retrieval_knn` | 1.0e-02 |

Pure NumPy and Python arithmetic is exact, so the dataset and the topology
heuristic reproduce bit-for-bit across machines. What drifts is anything whose
reduction order depends on a thread pool: the surrogate's parallel `index_add_`
scatter, the ridge solve, and the BLAS matmul behind kNN's distances (whose
`argpartition` then breaks near-ties differently, which is why it drifts most).

**This is not a footnote for one result.** The surrogate's criticality score spread
is 0.0008, smaller than the 1.3e-03 drift in those same scores, so its recall@10
moved from 0.433 to 0.450 between two launches with identical weights and a
bit-identical oracle. See `docs/RESULTS.md` §8.8.

### The launch environment is part of the seed

**Do not export `OMP_NUM_THREADS` before launching a stage.** `limit_threads` sets
it with `os.environ.setdefault` and runs *after* NumPy has bound its BLAS, so
exporting it beforehand actually changes the BLAS thread count. Doing so diverges
the training trace by the fourth decimal by epoch 5 (`-1.26665` against
`-1.26929`). Every committed number comes from a launch with `OMP_NUM_THREADS`,
`MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` **unset**; torch is still capped to
`run.threads` in-process, which is where the time goes. Reproduce with:

```bash
# correct
python scripts/compare_methods.py --config configs/base.yaml

# NOT this - it changes BLAS reduction order and the trace drifts
OMP_NUM_THREADS=2 python scripts/compare_methods.py --config configs/base.yaml
```

---

## Testing

```bash
make test        # fast suite, ~90 s
make test-all    # includes the training tests
make lint        # ruff, must be clean
```

Markers: `slow` covers anything that trains a model. `pytest -m "not slow"` is
what CI runs on both Ubuntu and Windows.

**Run the full suite locally, not just `-m "not slow"`.** The previous release was
only ever checked with `-m "not slow"` and with `python -m pytest`, and both a
permanently failing slow test and a collection bug survived that way. A bare
`uv run pytest tests` collects 375 tests: 373 pass and **2 are
`xfail(strict=True)`** — deliberately recorded expectations that the surrogate
does not meet, namely that it beats the all-zero constant on pooled MAE at fixture
scale and at real scale. `strict=True` means those two would *fail the suite* if
they ever started passing, which is the point: they are claims under measurement,
not skipped tests.

---

## Notebooks

Notebooks are **generated** from `scripts/make_notebooks.py` (with explicit cell
ids so the JSON stays diffable), then executed and committed with outputs:

```bash
python scripts/make_notebooks.py
python -m nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

They read the committed CSVs rather than recomputing, so a figure in a notebook
cannot disagree with a table in the docs.

`sndsur.viz` deliberately does **not** force the matplotlib backend at import
time; it only switches to `Agg` when no IPython kernel is present. Forcing `Agg`
unconditionally makes every notebook plot render blank, which is a silent failure.

---

## Cross-checking the documentation against the data

Every table in `README.md` and `docs/RESULTS.md` names its source CSV. To verify
a specific number, read that CSV. For example:

```bash
python -c "import pandas as pd; d=pd.read_csv('results/tables/method_comparison.csv'); \
print(d[d.split=='test_id'][['method','mae','spearman_within_scenario']].round(4).to_string(index=False))"
```

If a table cell reads `not measured`, no run stands behind it and none was
invented to fill it.
