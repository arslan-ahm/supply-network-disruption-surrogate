"""Generate the notebooks programmatically with nbformat.

Notebooks are generated rather than hand-authored so their code cannot drift from
the package, and every cell gets an explicit id so the JSON is stable across
regenerations and therefore diffable.

After generating, execute them with:

    python -m nbconvert --to notebook --execute --inplace notebooks/*.ipynb

Usage:
    python scripts/make_notebooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

OUT = Path("notebooks")

HEADER = """import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
sys.path.insert(0, "../src")
import numpy as np, pandas as pd, torch
torch.set_num_threads(2)
import matplotlib.pyplot as plt
pd.set_option("display.width", 130)
"""


def nb(cells: list) -> nbf.NotebookNode:
    n = nbf.v4.new_notebook()
    n.cells = cells
    n.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return n


def md(text: str, i: str) -> nbf.NotebookNode:
    c = nbf.v4.new_markdown_cell(text)
    c.id = i
    return c


def code(text: str, i: str) -> nbf.NotebookNode:
    c = nbf.v4.new_code_cell(text)
    c.id = i
    return c


def notebook_01() -> nbf.NotebookNode:
    """The network, the simulator, and why topology beats attributes."""
    return nb([
        md(
            "# 1. The network, the simulator, and the core idea\n\n"
            "This notebook shows what the data *is*: a generated multi-echelon supply\n"
            "network, a mechanistic simulator over it, and the one structural fact that\n"
            "a per-supplier risk table cannot express.\n\n"
            "**The claim being illustrated:** risk is a property of a node's position,\n"
            "not of its attributes.",
            "n1-title",
        ),
        code(HEADER, "n1-setup"),
        code(
            "from sndsur.data.network import NetworkSpec, generate_network, N_TIERS\n"
            "from sndsur.config import load_config\n"
            "cfg = load_config('../configs/base.yaml')\n"
            "from sndsur.data.scenarios import _net_spec, _sim_config\n"
            "spec, sim = _net_spec(cfg), _sim_config(cfg)\n"
            "net = generate_network(spec, seed=0, name='demo')\n"
            "net.compute_throughput()\n"
            "print(f'{net.n_nodes} nodes, {len(net.edges())} edges, "
            "{net.demand_nodes.size} demand points, {net.n_regions} regions')\n"
            "groups = [g for gs in net.groups for g in gs]\n"
            "print(f'{len(groups)} input groups, "
            "{sum(g.sole_source for g in groups)/len(groups):.0%} sole-sourced')",
            "n1-gen",
        ),
        md(
            "## The network, drawn by echelon\n\n"
            "Node size is expected throughput. **Red edges are sole-source links** — the\n"
            "only member of a customer's input group. Those are the hard single points of\n"
            "failure, and nothing about the supplier's own attributes reveals them.",
            "n1-md-draw",
        ),
        code(
            "from sndsur.viz import figure_network_example\n"
            "import os\n"
            "os.chdir('..')\n"
            "p = figure_network_example()\n"
            "os.chdir('notebooks')\n"
            "from IPython.display import Image, display\n"
            "display(Image(filename='../' + str(p)))",
            "n1-draw",
        ),
        md(
            "## Attributes versus position\n\n"
            "Below, every node's *own* attributes sit next to its **sole-source reach** —\n"
            "the number of demand points it can starve outright.\n\n"
            "**The correlations are not small, and that is the subtle part.** Throughput and\n"
            "customer count both correlate around +0.65 with sole-source reach on this\n"
            "network: big suppliers really do tend to be sole sources. The naive story --\n"
            "'attributes tell you nothing about position' -- is therefore *false*, and this\n"
            "notebook prints the numbers rather than quietly picking a friendlier example.\n\n"
            "The failure is finer than that. A monotone correlation across the bulk of a\n"
            "distribution does not make the *top* of the ranking right, and the top is the\n"
            "only part a planner reads. In `docs/RESULTS.md` the feature-based score ranks\n"
            "the single worst point of failure **44th to 46th of 46** in every one of six\n"
            "networks -- precisely because those nodes are downstream distribution centres\n"
            "with modest throughput, which is the tail where the correlation breaks down.",
            "n1-md-attr",
        ),
        code(
            "import pandas as pd, numpy as np\n"
            "thr = net.throughput\n"
            "slack = np.where(np.isfinite(net.capacity), net.capacity/np.maximum(thr,1e-9)-1, np.nan)\n"
            "df = pd.DataFrame({\n"
            "    'node': np.arange(net.n_nodes), 'tier': net.tier,\n"
            "    'throughput': thr.round(1), 'capacity_slack': slack.round(2),\n"
            "    'n_customers': [len(c) for c in net.customers()],\n"
            "    'downstream_reach': net.downstream_reach().astype(int),\n"
            "    'sole_source_reach': net.sole_source_reach().astype(int),\n"
            "})\n"
            "prod = df[df.tier < N_TIERS-1]\n"
            "print('correlation of each attribute with sole-source reach:')\n"
            "for c in ('throughput','capacity_slack','n_customers'):\n"
            "    print(f'  {c:18s} {prod[c].corr(prod.sole_source_reach):+.3f}')\n"
            "prod.sort_values('sole_source_reach', ascending=False).head(10)",
            "n1-attr",
        ),
        md(
            "## The simulator, and a hand-checkable case\n\n"
            "A single chain with no inventory: an outage at the raw supplier must produce a\n"
            "shortage after exactly the cumulative lead time, and it must be the entire\n"
            "demand. This is the arithmetic the simulator is tested against.",
            "n1-md-sim",
        ),
        code(
            "from sndsur.sim.simulator import SimConfig, simulate, counterfactual, demand_realisation\n"
            "from sndsur.data.disruptions import Disruption, DisruptionSet\n"
            "import sys; sys.path.insert(0, '../tests')\n"
            "from conftest import make_chain\n"
            "chain = make_chain(n_tiers=5, lead=1, base_stock=0., input_cover=0., cv=0.)\n"
            "c = SimConfig(warmup=30, horizon=12, backlog=False)\n"
            "print('baseline fill rate:', simulate(chain, c, demand_realisation(chain, c, 0), None).fill_rate)\n"
            "ds = DisruptionSet([Disruption('supplier_outage', 0, -1, 0, 12, 1.0)]).bind(chain)\n"
            "imp = counterfactual(chain, c, ds, 0)\n"
            "print('excess unmet per period:', imp.unmet_traj[0].round(2))\n"
            "print('time to impact:', imp.time_to_impact[0], '(4 legs x lead 1 = 4)')",
            "n1-sim",
        ),
        md(
            "## What a disruption actually does on a real network\n\n"
            "The trajectory is delayed, then amplified, then decays. That shape is what the\n"
            "surrogate has to learn — and it is why the trajectory decoder is a GRU rather\n"
            "than 12 independent linear outputs.",
            "n1-md-traj",
        ),
        code(
            "d = demand_realisation(net, sim, 3)\n"
            "base = simulate(net, sim, d, None)\n"
            "print(f'undisrupted fill rate {np.nanmean(base.fill_rate):.4f}')\n"
            "ss = net.sole_source_reach(); target = int(np.argmax(ss))\n"
            "ds = DisruptionSet([Disruption('supplier_outage', target, -1, 4, 10, 1.0)]).bind(net)\n"
            "imp = counterfactual(net, sim, ds, 3, baseline=base)\n"
            "print(f'outage at node {target} (sole-source reach {ss[target]:.0f}): "
            "total service loss {imp.total_service_loss:.4f}')\n"
            "fig, ax = plt.subplots(figsize=(8,3.5))\n"
            "for i, v in enumerate(net.demand_nodes):\n"
            "    if imp.unmet_traj[i].sum() > 1e-6:\n"
            "        ax.plot(imp.unmet_traj[i], label=f'demand {v}')\n"
            "ax.axvspan(4, 14, alpha=.12, color='red', label='outage window')\n"
            "ax.set_xlabel('period'); ax.set_ylabel('excess unmet units'); ax.legend(fontsize=8)\n"
            "ax.set_title('Shortage propagates with a delay, then recovers'); plt.show()",
            "n1-traj",
        ),
        md(
            "## The cost of the oracle\n\n"
            "Exhaustive counterfactual analysis means one simulator run per candidate. That\n"
            "is exact and it is the thing the surrogate is trying to replace.",
            "n1-md-cost",
        ),
        code(
            "import time\n"
            "from sndsur.sim.simulator import exhaustive_node_criticality\n"
            "t0 = time.perf_counter()\n"
            "cands, scores = exhaustive_node_criticality(net, sim, 3, duration=8, start=4)\n"
            "el = time.perf_counter()-t0\n"
            "print(f'{cands.size} candidates in {el:.2f}s ({el/cands.size*1000:.1f} ms each)')\n"
            "order = np.argsort(-scores)[:8]\n"
            "pd.DataFrame({'node': cands[order], 'tier': net.tier[cands[order]],\n"
            "              'true_service_loss': scores[order].round(4),\n"
            "              'sole_source_reach': ss[cands[order]].astype(int)})",
            "n1-cost",
        ),
    ])


def notebook_02() -> nbf.NotebookNode:
    """Train and compare against the baselines."""
    return nb([
        md(
            "# 2. Train the surrogate and compare it with the baselines\n\n"
            "Reads the committed results. Nothing here recomputes a number that appears in\n"
            "the documentation — the tables *are* the evidence, and this notebook reads\n"
            "them so a figure can never disagree with a table.\n\n"
            "Regenerate with `python scripts/compare_methods.py --config configs/base.yaml`.",
            "n2-title",
        ),
        code(HEADER, "n2-setup"),
        code(
            "T = '../results/tables/'\n"
            "methods = pd.read_csv(T+'method_comparison.csv')\n"
            "print(f\"{methods.method.nunique()} methods x {methods.split.nunique()} splits\")\n"
            "cols = ['split','method','mae','rmse','spearman_pooled','spearman_within_scenario','top1_agreement']\n"
            "methods.loc[methods.split=='test_id', cols].round(4).to_string(index=False)",
            "n2-load",
        ),
        md(
            "## Magnitude error and rank fidelity are different questions\n\n"
            "The target is **92.48% exact zeros**, so the MAE-optimal predictor is the\n"
            "constant **0** — and a model that collapses to it posts the best MAE in the\n"
            "table while being useless for screening. That is not hypothetical: this\n"
            "repository shipped it for 43 commits. `constant_zero` and\n"
            "`constant_train_mean` are therefore run as first-class methods, and\n"
            "`tabular_gbt_l1` — the same trees under `absolute_error` — is numerically\n"
            "identical to `constant_zero` on every metric on every split. See\n"
            "`docs/RESULTS.md` §8.7.\n\n"
            "Rank fidelity is the metric that matters for the actual task, and a constant\n"
            "cannot win it at all (its Spearman is undefined). The two columns below can\n"
            "disagree sharply.",
            "n2-md-two",
        ),
        code(
            "sub = methods[methods.split=='test_id'].set_index('method')\n"
            "fig, axes = plt.subplots(1,2, figsize=(12,4))\n"
            "sub['mae'].sort_values().plot.barh(ax=axes[0], color='#b02a2a')\n"
            "axes[0].set_title('MAE (lower better)')\n"
            "sub['spearman_within_scenario'].sort_values().plot.barh(ax=axes[1], color='#1b4f9c')\n"
            "axes[1].set_title('within-scenario Spearman (higher better)')\n"
            "for a in axes: a.grid(alpha=.25)\n"
            "plt.tight_layout(); plt.show()",
            "n2-two",
        ),
        md(
            "## Generalisation along each shift axis\n\n"
            "Each split changes exactly one thing relative to `test_id`, which is what makes\n"
            "a gap attributable.",
            "n2-md-gen",
        ),
        code(
            "gen = pd.read_csv(T+'generalisation.csv')\n"
            "gen.round(4).to_string(index=False)",
            "n2-gen",
        ),
        code(
            "from IPython.display import Image, display\n"
            "import os\n"
            "if os.path.exists('../results/figures/generalisation.png'):\n"
            "    display(Image(filename='../results/figures/generalisation.png'))",
            "n2-genfig",
        ),
        md(
            "## Is any difference real?\n\n"
            "Paired per-row Wilcoxon with Holm-Bonferroni, against the reference-style\n"
            "tabular model. **This conditions on one trained model per method** — it answers\n"
            "a question about two sets of weights, not about two methods. The method-level\n"
            "question needs the seed study in notebook 3.",
            "n2-md-stat",
        ),
        code(
            "st = pd.read_csv(T+'statistical_tests.csv')\n"
            "s = st[st.split=='test_id'][['name_a','mean_a','mean_b','difference','ci_lower','ci_upper','p_adjusted','effect_size','significant']]\n"
            "s.round(5).to_string(index=False)",
            "n2-stat",
        ),
    ])


def notebook_03() -> nbf.NotebookNode:
    """Seed variance and ablations."""
    return nb([
        md(
            "# 3. Seed variance and ablations\n\n"
            "**Read this before believing any improvement in notebook 2.** The same\n"
            "configuration trained at three seeds gives a spread; a difference between two\n"
            "single runs is inside noise unless it exceeds `sqrt(2) x sd`.",
            "n3-title",
        ),
        code(HEADER, "n3-setup"),
        code(
            "T = '../results/tables/'\n"
            "noise = pd.read_csv(T+'seed_variance.csv')\n"
            "noise[noise.split=='test_id'].round(5).to_string(index=False)",
            "n3-noise",
        ),
        md(
            "The `diff_noise_scale` column is the bar every claimed gain has to clear.",
            "n3-md-bar",
        ),
        code(
            "runs = pd.read_csv(T+'seed_runs.csv')\n"
            "piv = runs[runs.split=='test_id'].pivot(index='metric', columns='seed', values='value')\n"
            "piv.round(5)",
            "n3-runs",
        ),
        code(
            "from IPython.display import Image, display\n"
            "import os\n"
            "if os.path.exists('../results/figures/seed_noise.png'):\n"
            "    display(Image(filename='../results/figures/seed_noise.png'))",
            "n3-noisefig",
        ),
        md(
            "## Ablations: one mechanism removed at a time\n\n"
            "`no_message_passing` is the one that isolates the graph. Its parameter count is\n"
            "matched to the full model on purpose, so the comparison is about message\n"
            "passing rather than about capacity.",
            "n3-md-abl",
        ),
        code(
            "abl = pd.read_csv(T+'ablations.csv')\n"
            "p = abl[abl.split=='test_id'].set_index('variant')[['params','mae','spearman_within_scenario','train_seconds']]\n"
            "p.round(4)",
            "n3-abl",
        ),
        code(
            "st = pd.read_csv(T+'ablation_statistics.csv')\n"
            "st[st.split=='test_id'][['name_a','mean_a','mean_b','difference','p_adjusted','significant']].round(5).to_string(index=False)",
            "n3-ablstat",
        ),
        md(
            "### Placing each ablation against the noise scale\n\n"
            "A significant paired p-value is not enough: the delta also has to be large\n"
            "relative to seed-to-seed variation of the identical configuration.",
            "n3-md-verdict",
        ),
        code(
            "from sndsur.metrics.stats import noise_scale, noise_verdict\n"
            "ns = noise_scale('mae', runs[(runs.split=='test_id')&(runs.metric=='mae')].value.values)\n"
            "full = float(abl[(abl.split=='test_id')&(abl.variant=='full')].mae.iloc[0])\n"
            "rows = []\n"
            "for _, r in abl[abl.split=='test_id'].iterrows():\n"
            "    if r.variant == 'full': continue\n"
            "    gain = float(r.mae) - full  # positive = removing it hurt\n"
            "    ratio, verdict = noise_verdict(gain, ns)\n"
            "    rows.append({'variant': r.variant, 'mae': round(float(r.mae),5),\n"
            "                 'delta_vs_full': round(gain,5), 'ratio_to_noise': round(ratio,2),\n"
            "                 'verdict': verdict})\n"
            "pd.DataFrame(rows)",
            "n3-verdict",
        ),
    ])


def notebook_04() -> nbf.NotebookNode:
    """Counterfactual ranking, disagreement, uncertainty, efficiency."""
    return nb([
        md(
            "# 4. Counterfactual criticality, calibration and the compute trade\n\n"
            "The product: rank nodes by *measured* downstream impact rather than by\n"
            "intrinsic risk features, show where the two disagree, and let the simulator\n"
            "settle it.",
            "n4-title",
        ),
        code(HEADER, "n4-setup"),
        code(
            "T = '../results/tables/'\n"
            "rank = pd.read_csv(T+'criticality_summary.csv')\n"
            "rank.round(4).to_string(index=False)",
            "n4-rank",
        ),
        md(
            "## Where the feature score and the counterfactual disagree\n\n"
            "Each row is a pair the two rankings order oppositely. `winner` is the\n"
            "simulator's verdict — there is no arguing about it, because the counterfactual\n"
            "was actually run.",
            "n4-md-dis",
        ),
        code(
            "dis = pd.read_csv(T+'disagreements.csv')\n"
            "if len(dis):\n"
            "    print('winner counts:'); print(dis.winner.value_counts().to_string())\n"
            "    display(dis.nlargest(6,'margin')[['network','node_a','node_b','rank_a_feature',"
            "'rank_b_feature','rank_a_counterfactual','rank_b_counterfactual','truth_a','truth_b','winner','margin']].round(4))\n"
            "    print()\n"
            "    print(dis.nlargest(1,'margin').explanation.iloc[0])\n"
            "else:\n"
            "    print('no disagreements found')",
            "n4-dis",
        ),
        code(
            "from IPython.display import Image, display\n"
            "import os\n"
            "for f in ('criticality_scatter.png','budget_curve.png'):\n"
            "    if os.path.exists('../results/figures/'+f): display(Image(filename='../results/figures/'+f))",
            "n4-figs",
        ),
        md(
            "## Does the surrogate know when it does not know?\n\n"
            "Coverage without sharpness is meaningless (a `[0,1]` interval covers\n"
            "everything), so both are reported. The property that matters out of\n"
            "distribution is whether error *grows with* predicted uncertainty.",
            "n4-md-cal",
        ),
        code(
            "cal = pd.read_csv(T+'calibration.csv')\n"
            "g = cal[(cal.interval=='gaussian')]\n"
            "g.pivot_table(index='split', columns='nominal', values=['coverage','width']).round(4)",
            "n4-cal",
        ),
        code(
            "sh = cal[cal.interval=='uncertainty_shift']\n"
            "sh[['split','mean_sigma','mean_abs_error','sigma_error_spearman',"
            "'error_detection_auroc','ause','mean_sigma_aleatoric','mean_sigma_epistemic']].round(4).to_string(index=False)",
            "n4-shift",
        ),
        md(
            "## Decision quality at a fixed compute budget\n\n"
            "The honest framing. Given N seconds, the simulator evaluates a few candidates\n"
            "exactly; the surrogate screens all of them approximately. Which finds the true\n"
            "top-10 more reliably?",
            "n4-md-budget",
        ),
        code(
            "bud = pd.read_csv(T+'budget_curve.csv')\n"
            "bud.groupby(['budget_s','method']).recall_at_k.mean().unstack().round(3)",
            "n4-budget",
        ),
        code(
            "eff = pd.read_csv(T+'efficiency.csv')\n"
            "display(eff.groupby(['component','variant'])[['median_ms','iqr_ms','per_item_ms']].median().round(4))\n"
            "be = pd.read_csv(T+'break_even.csv')\n"
            "be.T",
            "n4-eff",
        ),
        md(
            "The break-even row is the honest part: it charges the surrogate for **both** its\n"
            "training time and the dataset generation that used the very simulator it\n"
            "replaces. Below that many screened scenarios, running the simulator directly is\n"
            "the better trade.",
            "n4-md-be",
        ),
    ])


def notebook_05() -> nbf.NotebookNode:
    """Colab/GPU notebook at larger scale."""
    return nb([
        md(
            "# 5. Larger networks on a GPU (Colab)\n\n"
            "The committed results are CPU-scale by necessity: ~54-node networks, 10 epochs,\n"
            "a 3-member ensemble, all on two threads of a shared 4-core machine. This\n"
            "notebook runs the **same code** at a scale the CPU budget cannot reach.\n\n"
            "Nothing here is required to reproduce a documented number. It exists so a\n"
            "reader with a GPU can check whether the conclusions survive more capacity —\n"
            "and the honest expectation is that the *rank* results improve more than the\n"
            "magnitude ones.",
            "n5-title",
        ),
        md(
            "## Setup\n\n"
            "On Colab, uncomment the clone cell. Locally, this runs from the repo root.",
            "n5-md-setup",
        ),
        code(
            "# !git clone https://github.com/HabibaSajid321/supply-network-disruption-surrogate.git\n"
            "# %cd supply-network-disruption-surrogate\n"
            "# !pip install -q torch numpy scipy pandas pyyaml matplotlib scikit-learn\n"
            "import os, sys\n"
            "sys.path.insert(0, '../src' if os.path.basename(os.getcwd())=='notebooks' else 'src')\n"
            "import torch\n"
            "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "print('device:', DEVICE, '| torch', torch.__version__)\n"
            "if DEVICE == 'cpu':\n"
            "    print('No GPU: the cells below will still run but use the small settings.')",
            "n5-setup",
        ),
        md(
            "## Scale up\n\n"
            "Larger networks, more scenarios, a wider model, more epochs, a bigger ensemble.\n"
            "Note that **dataset generation is CPU-bound** — the simulator is pure Python —\n"
            "so a GPU speeds up training but not data generation. That asymmetry is exactly\n"
            "why the break-even accounting in notebook 4 charges dataset time separately.",
            "n5-md-scale",
        ),
        code(
            "from sndsur.config import load_config\n"
            "big = ['netgen.n_per_tier=[40,34,24,14,18]',\n"
            "       'dataset.n_train_networks=20',\n"
            "       'dataset.scenarios_per_train_network=500',\n"
            "       'dataset.max_train_scenarios=0',\n"
            "       'model.hidden=128', 'model.layers=4', 'model.traj_horizon=24',\n"
            "       'model.ensemble=5', 'train.epochs=60', 'train.batch_graphs=32',\n"
            "       f'run.device={DEVICE}', 'run.threads=8', 'run.name=gpu_full']\n"
            "cfg = load_config('../configs/base.yaml' if os.path.basename(os.getcwd())=='notebooks' else 'configs/base.yaml', big)\n"
            "print(cfg.netgen.n_per_tier, '->', sum(cfg.netgen.n_per_tier), 'nodes per network')\n"
            "print('hidden', cfg.model.hidden, '| layers', cfg.model.layers, '| ensemble', cfg.model.ensemble)",
            "n5-scale",
        ),
        code(
            "# Dataset generation is the slow part (CPU-bound simulator). Expect ~20-40 min.\n"
            "# from sndsur.pipelines import build_data\n"
            "# build_data(cfg, rebuild=True)",
            "n5-data",
        ),
        code(
            "# from sndsur.pipelines import run_comparison, run_criticality, run_efficiency\n"
            "# frames = run_comparison(cfg)\n"
            "# frames['methods'][['split','method','mae','spearman_within_scenario']].round(4)",
            "n5-train",
        ),
        md(
            "## Bring your own network\n\n"
            "`sndsur.data.csvio` loads a user-supplied topology from CSV. The **mechanics**\n"
            "are still this simulator's mechanics, so results transfer only insofar as those\n"
            "mechanics match reality. No real-world validation is claimed anywhere in this\n"
            "repository.",
            "n5-md-csv",
        ),
        code(
            "from sndsur.data.csvio import write_example_csvs, load_network_from_csv\n"
            "import tempfile, pathlib\n"
            "d = pathlib.Path(tempfile.mkdtemp())\n"
            "write_example_csvs(d)\n"
            "print(open(d/'nodes.csv').read())\n"
            "print(open(d/'edges.csv').read())\n"
            "net = load_network_from_csv(d/'nodes.csv', d/'edges.csv')\n"
            "net.validate()\n"
            "print('loaded:', net.n_nodes, 'nodes,', len(net.edges()), 'edges')",
            "n5-csv",
        ),
    ])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    books = {
        "01_network_and_simulator.ipynb": notebook_01(),
        "02_train_and_compare.ipynb": notebook_02(),
        "03_seed_variance_and_ablations.ipynb": notebook_03(),
        "04_criticality_calibration_efficiency.ipynb": notebook_04(),
        "05_colab_gpu_full_scale.ipynb": notebook_05(),
    }
    for name, book in books.items():
        p = OUT / name
        p.write_text(nbf.writes(book), encoding="utf-8", newline="\n")
        print(f"wrote {p} ({len(book.cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
