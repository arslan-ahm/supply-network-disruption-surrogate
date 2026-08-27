# Development and experiment entry points.
# Every committed number in results/ comes from one of these targets.

PY := ./.venv/Scripts/python.exe
ifeq ($(OS),)
PY := ./.venv/bin/python
endif

CFG := configs/base.yaml
SEEDS := 0 7 1337

.PHONY: help setup test test-all lint fmt smoke data seeds compare ablate \
        criticality efficiency figures notebooks all clean clean-results

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup:  ## Create the environment with uv and install the package
	uv python install 3.12
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) --index-url https://download.pytorch.org/whl/cpu \
	  --extra-index-url https://pypi.org/simple torch
	uv pip install --python $(PY) numpy scipy pandas pyyaml matplotlib scikit-learn
	uv pip install --python $(PY) pytest ruff nbformat nbconvert jupyter
	uv pip install --python $(PY) -e . --no-deps

test:  ## Fast unit tests (~90 s)
	$(PY) -m pytest tests -q -m "not slow"

test-all:  ## Everything, including the training tests (minutes)
	$(PY) -m pytest tests -q

lint:  ## Static checks
	$(PY) -m ruff check src scripts tests

fmt:  ## Auto-fix what ruff can
	$(PY) -m ruff check --fix src scripts tests

smoke:  ## The whole pipeline end to end on a tiny config (~1 min)
	$(PY) scripts/compare_methods.py --config configs/smoke.yaml --rebuild

data:  ## Generate every network and simulate every scenario (~5 min, cached after)
	$(PY) scripts/build_dataset.py --config $(CFG)

seeds:  ## Seed-variance study. RUN THIS BEFORE READING ANY COMPARISON.
	$(PY) scripts/run_seed_study.py --config $(CFG) --seeds $(SEEDS)

compare:  ## Surrogate vs every baseline on all seven splits; caches the ensemble
	$(PY) scripts/compare_methods.py --config $(CFG)

ablate:  ## One mechanism removed at a time, with significance tests
	$(PY) scripts/run_ablations.py --config $(CFG)

criticality:  ## Exhaustive oracle sweep, ranking comparison, budget curve
	$(PY) scripts/run_criticality.py --config $(CFG)

efficiency:  ## Wall-clock, parameters and the break-even accounting
	$(PY) scripts/benchmark_efficiency.py --config $(CFG)

figures:  ## Redraw every figure from the committed CSVs
	$(PY) scripts/make_figures.py

notebooks:  ## Regenerate and execute the notebooks
	$(PY) scripts/make_notebooks.py
	$(PY) -m nbconvert --to notebook --execute --inplace notebooks/*.ipynb

# `compare` must precede `criticality` and `efficiency`: it trains and caches the
# ensemble those two reuse. `data` must precede `efficiency`, which reads the
# dataset-generation time it charges to the surrogate's break-even.
all: data seeds compare ablate criticality efficiency figures  ## The full matrix

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-results:  ## Remove run artefacts and the scenario cache (keeps tables/figures)
	rm -rf results/runs data/scenarios checkpoints/*.pt
