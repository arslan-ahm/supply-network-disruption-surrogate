"""Experiment orchestration. Scripts are thin wrappers over these functions.

Everything that produces a committed number lives here, so a test can call it and
a script cannot diverge from what was tested. The functions are:

:func:`build_data`
    Generate networks, simulate scenarios, write the dataset table.
:func:`run_comparison`
    Train the surrogate and every baseline on the same rows; evaluate on all
    seven splits; write the method-comparison and statistics tables.
:func:`run_seed_study`
    The same configuration at several seeds, to establish the noise scale that
    every claim is then placed against.
:func:`run_ablations`
    One switch at a time, with significance tests.
:func:`run_criticality`
    The exhaustive counterfactual sweep, the ranking comparison, the
    disagreement experiment and the compute-budget curve.
:func:`run_efficiency`
    Parameters, latency and the break-even accounting.
"""

from __future__ import annotations

import gc
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from sndsur.analysis.counterfactual import (
    aggregate_surrogate_scores,
    find_disagreements,
    node_candidates,
    probe_scenarios,
    simulate_truth,
)
from sndsur.config import Config
from sndsur.data.features import N_EDGE_FEATURES, N_NODE_FEATURES
from sndsur.data.scenarios import (
    SHIFT_SPLITS,
    SPLITS,
    ScenarioDataset,
    _sim_config,
    kind_breakdown,
    load_or_build,
    split_summary,
)
from sndsur.engine.batching import apply_normalisers, collate, fit_normalisers, iterate_batches
from sndsur.engine.trainer import (
    Predictions,
    ensemble_component_sigmas,
    predict,
    predict_ensemble,
    train_surrogate,
)
from sndsur.metrics import calibration as CAL
from sndsur.metrics import fidelity as FID
from sndsur.metrics import ranking as RANK
from sndsur.metrics import stats as ST
from sndsur.models.baselines import (
    ScenarioRetrieval,
    TabularRiskModel,
    TopologyHeuristic,
    rows_from_split,
)
from sndsur.utils.bench import benchmark, break_even_scenarios, budget_curve
from sndsur.utils.checkpoint import train_or_load_ensemble
from sndsur.utils.logging import RunDir, get_logger, write_csv
from sndsur.utils.seed import limit_threads, seed_everything

LOG = get_logger()
TABLES = Path("results/tables")


def prepare(cfg: Config, rebuild: bool = False) -> ScenarioDataset:
    """Load or build the dataset and standardise its features.

    Normalisers are fitted on ``train`` only and applied to every split, so no
    statistic from an evaluation split ever touches the model's inputs.
    """
    limit_threads(cfg.run.threads)
    seed_everything(cfg.run.seed, cfg.run.deterministic)
    t0 = time.perf_counter()
    ds = load_or_build(
        cfg,
        rebuild=rebuild,
        progress=lambda stage, d, t: LOG.info("  data %s %d/%d", stage, d, t),
    )
    LOG.info(
        "dataset ready in %.1fs: %s", time.perf_counter() - t0, ds.counts()
    )
    # Truncate the predicted trajectory horizon. The simulator recorded more
    # periods than the surrogate predicts; slicing here keeps the cache reusable
    # across trajectory-horizon settings instead of forcing a regeneration.
    keep = min(ds.traj_periods, cfg.model.traj_horizon)
    if keep < ds.traj_periods:
        for scs in ds.splits.values():
            for s in scs:
                s.traj = s.traj[:, :keep]
        ds.traj_periods = keep

    # Deterministic subsampling, stratified by network so a smaller budget never
    # silently drops a whole topology.
    def _cap(split: str, cap: int, salt: int) -> None:
        scs = ds.splits[split]
        if not cap or cap >= len(scs):
            return
        rng = np.random.default_rng(cfg.run.seed + salt)
        by_net: dict[int, list[int]] = {}
        for i, s in enumerate(scs):
            by_net.setdefault(s.net_id, []).append(i)
        per = max(1, cap // max(len(by_net), 1))
        keep: list[int] = []
        for _net, idxs in sorted(by_net.items()):
            take = min(per, len(idxs))
            keep += list(rng.permutation(idxs)[:take])
        ds.splits[split] = [scs[i] for i in sorted(keep)]
        LOG.info("%s capped at %d scenarios", split, len(ds.splits[split]))

    _cap("train", cfg.dataset.max_train_scenarios, 0)
    for i, split in enumerate(("val", "test_id", *SHIFT_SPLITS)):
        _cap(split, cfg.dataset.max_eval_scenarios, i + 1)
    gc.collect()

    node, edge, tab = fit_normalisers(ds, "train")
    return apply_normalisers(ds, node, edge, tab)


def build_data(cfg: Config, rebuild: bool = True) -> pd.DataFrame:
    """Generate the dataset and write ``results/tables/dataset.csv``."""
    t0 = time.perf_counter()
    ds = prepare(cfg, rebuild=rebuild)
    seconds = time.perf_counter() - t0
    rows = split_summary(ds)
    for r in rows:
        r.update({f"kind_{k}": v for k, v in kind_breakdown(ds, str(r["split"])).items()})
    frame = pd.DataFrame(rows)
    frame["build_seconds_total"] = seconds
    write_csv(frame, TABLES / "dataset.csv")
    LOG.info("dataset table written (%.1fs total build)", seconds)
    return frame


# --------------------------------------------------------------------------- #
# Method comparison
# --------------------------------------------------------------------------- #


def _batches(ds: ScenarioDataset, cfg: Config) -> dict[str, list]:
    """Pre-collated evaluation batches for every split."""
    return {
        s: iterate_batches(ds.splits[s], ds, cfg.train.batch_graphs, False)
        for s in SPLITS
        if ds.splits[s]
    }


def _fit_heuristic_scale(train_scores: np.ndarray, train_truth: np.ndarray) -> tuple[float, float]:
    """Least-squares affine map from heuristic score to service-loss units.

    The composite heuristic is a weighted sum of standardised topology signals,
    so its raw scale is arbitrary and its unscaled MAE is meaningless — the first
    run of this comparison reported the heuristic at MAE 2.4 against a target
    bounded by the number of demand points, which says nothing about the
    heuristic and everything about its units.

    The scale is fitted on the **training** split only and applied unchanged
    everywhere else, exactly like the learned models' parameters. Two numbers is
    the least the heuristic can be given to make its error interpretable, and
    fitting more than that would quietly turn it into a linear model — which is
    already covered by the ridge variant of the tabular baseline.
    """
    x = np.asarray(train_scores, dtype=np.float64)
    y = np.asarray(train_truth, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.allclose(x[m], x[m][0]):
        return (0.0, float(np.nanmean(y)) if m.any() else 0.0)
    a, b = np.polyfit(x[m], y[m], 1)
    return (float(a), float(b))


def _heuristic_rows(ds: ScenarioDataset, split: str) -> np.ndarray:
    """Per-row heuristic score, aligned with :func:`rows_from_split`.

    The heuristic scores a node; the row asks about a (disrupted node, demand
    point) pair. The node's score is broadcast to demand points reachable from
    it and set to zero elsewhere, which gives the heuristic the reachability
    relation for free — the most generous fair treatment available to it.
    """
    heur = TopologyHeuristic()
    out: list[np.ndarray] = []
    cache: dict[int, tuple[np.ndarray, dict[int, set[int]]]] = {}
    for s in ds.splits[split]:
        if s.net_id not in cache:
            net = ds.networks[s.net_id].net
            scores = heur.node_scores(net)
            cust = net.customers()
            demand = set(net.demand_nodes.tolist())
            reach: dict[int, set[int]] = {}
            for v in range(net.n_nodes - 1, -1, -1):
                reach[v] = {v} if v in demand else set().union(
                    *[reach[c] for c in cust[v]]
                ) if cust[v] else set()
            cache[s.net_id] = (scores, reach)
        scores, reach = cache[s.net_id]
        if s.disruption.items:
            primary = max(s.disruption.items, key=lambda d: d.severity)
            src = int(primary.target)
        else:
            src = int(s.demand_rows[0])
        base = float(scores[src]) if 0 <= src < scores.shape[0] else 0.0
        # Shifted to be non-negative: the composite is standardised and so has
        # negative values, and a negative "criticality" ranks below an
        # unreachable pair, which is the wrong ordering.
        base = base - float(scores.min())
        hit = reach.get(src, set())
        out.append(
            np.array([base if int(v) in hit else 0.0 for v in s.demand_rows], dtype=np.float64)
        )
    return np.concatenate(out) if out else np.zeros(0)


def run_comparison(cfg: Config, rebuild: bool = False) -> dict[str, pd.DataFrame]:
    """Train every method on the same data and evaluate on every split.

    Returns:
        ``{"methods": frame, "statistics": frame, "calibration": frame,
        "per_row": frame}``, all also written to ``results/tables/``.
    """
    ds = prepare(cfg, rebuild=rebuild)
    batches = _batches(ds, cfg)
    run = RunDir(cfg.run.out_dir, cfg.run.name)
    run.clear_history()
    cfg.to_yaml(run.path / "config.yaml")

    LOG.info("training surrogate ensemble (%d members)", cfg.model.ensemble)
    models, records = train_or_load_ensemble(
        cfg, ds, N_NODE_FEATURES, N_EDGE_FEATURES, run=run
    )
    train_seconds = float(np.sum([r["train_seconds"] for r in records]))
    LOG.info("ensemble trained in %.1fs", train_seconds)

    train_rows = rows_from_split(ds, "train")
    LOG.info("fitting baselines on %d training rows", train_rows.y.size)
    tab_gbt = TabularRiskModel("gbt", cfg.run.seed).fit(train_rows.x, train_rows.y)
    tab_ridge = TabularRiskModel("ridge", cfg.run.seed).fit(train_rows.x, train_rows.y)
    retrieval = ScenarioRetrieval(k=8).fit(train_rows.x, train_rows.y)
    heur_a, heur_b = _fit_heuristic_scale(_heuristic_rows(ds, "train"), train_rows.y)
    LOG.info("heuristic affine calibration fitted on train: %.5f * s + %.5f", heur_a, heur_b)

    LOG.info("training no-message-passing MLP ablation")
    cfg_nograph = replace(cfg, model=replace(cfg.model, layers=0, ensemble=1))
    mlp_model, mlp_rec = train_surrogate(
        cfg_nograph, ds, N_NODE_FEATURES, N_EDGE_FEATURES, log_every=0
    )

    method_rows: list[dict] = []
    per_row_frames: list[pd.DataFrame] = []
    cal_rows: list[dict] = []
    stat_rows: list[dict] = []
    preds_by_split: dict[str, dict[str, np.ndarray]] = {}

    for split in SPLITS:
        if not ds.splits[split]:
            continue
        rows = rows_from_split(ds, split)
        ens: Predictions = predict_ensemble(models, batches[split], ds, cfg.run.device)
        single: Predictions = predict(models[0], batches[split], ds, cfg.run.device)
        mlp_pred: Predictions = predict(mlp_model, batches[split], ds, cfg.run.device)

        # Every method predicts the same rows in the same order; assert it rather
        # than trust it, because a silent misalignment would make every number in
        # this table wrong in a way no metric would reveal.
        if not np.array_equal(ens.scenario_id, rows.scenario_id):
            raise RuntimeError(f"row alignment broken on split {split}")

        preds = {
            "surrogate_ensemble": ens.impact,
            "surrogate_single": single.impact,
            "mlp_no_message_passing": mlp_pred.impact,
            "tabular_gbt": tab_gbt.predict(rows.x),
            "tabular_ridge": tab_ridge.predict(rows.x),
            "retrieval_knn": retrieval.predict(rows.x),
            "topology_heuristic": np.clip(
                heur_a * _heuristic_rows(ds, split) + heur_b, 0.0, None
            ),
        }
        preds_by_split[split] = preds
        truth = rows.y

        for name, p in preds.items():
            rec = {"split": split, "method": name}
            rec.update(FID.fidelity_report(p, truth, rows.scenario_id))
            if name == "surrogate_ensemble":
                rec.update(FID.trajectory_metrics(ens.traj, ens.traj_true))
                rec.update(
                    FID.timing_metrics(ens.tti, ens.tti_true, ens.recovery, ens.recovery_true)
                )
            method_rows.append(rec)

        # Calibration is only defined for the methods that emit an uncertainty.
        cal_rows += [
            {"split": split, "method": "surrogate_ensemble", **r}
            for r in CAL.calibration_table(
                ens.impact, ens.sigma, truth, tuple(cfg.eval.coverage_levels)
            )
        ]
        cal_rows += [
            {"split": split, "method": "surrogate_single_heteroscedastic", **r}
            for r in CAL.calibration_table(
                single.impact, single.sigma, truth, tuple(cfg.eval.coverage_levels),
                label="gaussian_single",
            )
        ]
        if single.quantiles.size:
            cal_rows += [
                {"split": split, "method": "surrogate_quantile", **r}
                for r in CAL.quantile_calibration(
                    single.quantiles, tuple(cfg.model.quantiles), truth
                )
            ]
        alea, epi = ensemble_component_sigmas(models, batches[split], ds, cfg.run.device)
        shift = CAL.uncertainty_shift_report(ens.impact, ens.sigma, truth)
        shift.update({
            "split": split,
            "method": "surrogate_ensemble",
            "mean_sigma_aleatoric": float(np.mean(alea)),
            "mean_sigma_epistemic": float(np.mean(epi)),
        })
        cal_rows.append({**shift, "interval": "uncertainty_shift", "nominal": float("nan")})

        per_row_frames.append(
            pd.DataFrame({
                "split": split,
                "scenario_id": rows.scenario_id,
                "net_id": rows.net_id,
                "demand_node": rows.demand_node,
                "truth": truth,
                "sigma": ens.sigma,
                **{f"pred_{k}": v for k, v in preds.items()},
            })
        )

        # Paired per-row tests against the reference approach. The unit here is a
        # row, so this answers "do these weights beat that model consistently
        # across rows", not "is the method better" — see docs/RESULTS.md.
        err = {k: np.abs(v - truth) for k, v in preds.items()}
        _, comps = ST.summarize_metric(
            err, "abs_error", "tabular_gbt", cfg.eval.bootstrap, cfg.run.seed
        )
        for c in comps:
            stat_rows.append({"split": split, "metric": "abs_error", **c.to_dict()})

    frames = {
        "methods": pd.DataFrame(method_rows),
        "statistics": pd.DataFrame(stat_rows),
        "calibration": pd.DataFrame(cal_rows),
        "per_row": pd.concat(per_row_frames, ignore_index=True),
    }
    write_csv(frames["methods"], TABLES / "method_comparison.csv")
    write_csv(frames["statistics"], TABLES / "statistical_tests.csv")
    write_csv(frames["calibration"], TABLES / "calibration.csv")
    write_csv(frames["per_row"], run.path / "per_item.csv")

    gen = pd.DataFrame([
        {
            "method": m,
            "test_id_mae": _pick(frames["methods"], "test_id", m, "mae"),
            **{
                f"{s}_mae": _pick(frames["methods"], s, m, "mae") for s in SHIFT_SPLITS
            },
            **{
                f"{s}_gap": _pick(frames["methods"], s, m, "mae")
                - _pick(frames["methods"], "test_id", m, "mae")
                for s in SHIFT_SPLITS
            },
            "test_id_spearman": _pick(frames["methods"], "test_id", m, "spearman_pooled"),
            **{
                f"{s}_spearman": _pick(frames["methods"], s, m, "spearman_pooled")
                for s in SHIFT_SPLITS
            },
        }
        for m in frames["methods"]["method"].unique()
    ])
    write_csv(gen, TABLES / "generalisation.csv")
    frames["generalisation"] = gen

    run.write_summary({
        "train_seconds": train_seconds,
        "ensemble": records,
        "mlp_no_message_passing": mlp_rec,
        "params_surrogate": records[0]["params"],
        "params_mlp_no_mp": mlp_rec["params"],
        "tabular_gbt_tree_nodes": tab_gbt.n_params_,
        "retrieval_stored_floats": retrieval.memory_floats(),
        "splits": ds.counts(),
        "rows": ds.rows(),
    })
    LOG.info("comparison written to %s", TABLES)
    return frames


def _pick(frame: pd.DataFrame, split: str, method: str, col: str) -> float:
    """One cell from the long-format method table, or NaN if absent."""
    m = (frame["split"] == split) & (frame["method"] == method)
    if not m.any() or col not in frame.columns:
        return float("nan")
    return float(frame.loc[m, col].iloc[0])


# --------------------------------------------------------------------------- #
# Seed study
# --------------------------------------------------------------------------- #

#: Metrics the seed study tracks. Chosen to cover both fidelity families and the
#: uncertainty story, because the noise scale differs enormously between them.
SEED_METRICS = (
    "mae",
    "rmse",
    "spearman_pooled",
    "spearman_within_scenario",
    "top1_agreement",
)


def run_seed_study(cfg: Config, seeds: tuple[int, ...] = (0, 7, 1337)) -> pd.DataFrame:
    """Train the identical configuration at several seeds and measure the spread.

    This runs *before* any claim is written. Its output is the denominator for
    every "improvement" in the results: a difference between two single runs is
    inside noise unless it exceeds ``sqrt(2) * sd`` for that metric.

    A single ensemble member is used per seed, not the full ensemble: the claim
    being calibrated is about the effect of the training seed, and averaging five
    members per seed would report the ensemble's (much smaller) variance instead.

    Returns:
        A long frame with one row per (seed, split, metric) plus the derived
        noise-scale rows.
    """
    ds = prepare(cfg)
    batches = _batches(ds, cfg)
    cfg_one = replace(cfg, model=replace(cfg.model, ensemble=1))
    per_seed: list[dict] = []

    for s in seeds:
        LOG.info("seed study: seed %d", s)
        model, rec = train_surrogate(
            cfg_one, ds, N_NODE_FEATURES, N_EDGE_FEATURES, seed=s, log_every=0
        )
        for split in ("test_id", *SHIFT_SPLITS):
            if split not in batches:
                continue
            p = predict(model, batches[split], ds, cfg.run.device)
            rows = rows_from_split(ds, split)
            rep = FID.fidelity_report(p.impact, rows.y, rows.scenario_id)
            for m in SEED_METRICS:
                per_seed.append({
                    "seed": s, "split": split, "metric": m,
                    "value": rep.get(m, float("nan")),
                    "train_seconds": rec["train_seconds"],
                })

    frame = pd.DataFrame(per_seed)
    noise_rows = []
    for (split, metric), grp in frame.groupby(["split", "metric"]):
        try:
            ns = ST.noise_scale(str(metric), grp["value"].to_numpy())
        except ValueError:
            continue
        noise_rows.append({"split": split, **ns.to_dict()})
    noise = pd.DataFrame(noise_rows)
    write_csv(frame, TABLES / "seed_runs.csv")
    write_csv(noise, TABLES / "seed_variance.csv")
    LOG.info("seed study written: %d runs", len(seeds))
    return noise


# --------------------------------------------------------------------------- #
# Ablations
# --------------------------------------------------------------------------- #

#: Each entry disables exactly one mechanism. Anything that changes two things at
#: once attributes nothing and is not an ablation.
ABLATIONS: dict[str, dict] = {
    "full": {},
    "no_message_passing": {"layers": 0},
    "one_hop": {"layers": 1},
    "two_hop": {"layers": 2},
    "unidirectional": {"bidirectional": False},
    "no_edge_features": {"use_edge_features": False},
    "linear_traj_decoder": {"temporal_decoder": False},
}


def run_ablations(cfg: Config, variants: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Train one model per ablation and test each against the full model.

    Every variant trains a single member (not an ensemble) so the comparison is
    architecture against architecture rather than ensemble against ensemble, and
    every variant shares the seed, the data, the loss and the schedule.

    Ablations ship with significance tests. A raw delta on its own attributes
    nothing, and the seed-variance table says how big a delta has to be before it
    means anything.
    """
    ds = prepare(cfg)
    batches = _batches(ds, cfg)
    names = variants or tuple(ABLATIONS)
    preds: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict] = []

    for name in names:
        over = ABLATIONS[name]
        m = replace(cfg.model, ensemble=1, **over)
        vcfg = replace(cfg, model=m)
        LOG.info("ablation %-22s %s", name, over or "(baseline)")
        model, rec = train_surrogate(
            vcfg, ds, N_NODE_FEATURES, N_EDGE_FEATURES, log_every=0
        )
        preds[name] = {}
        for split in ("test_id", *SHIFT_SPLITS):
            if split not in batches:
                continue
            p = predict(model, batches[split], ds, cfg.run.device)
            r = rows_from_split(ds, split)
            preds[name][split] = np.abs(p.impact - r.y)
            rep = FID.fidelity_report(p.impact, r.y, r.scenario_id)
            rows.append({
                "variant": name, "split": split, "params": rec["params"],
                "train_seconds": rec["train_seconds"], "best_epoch": rec["best_epoch"],
                **rep,
            })

    frame = pd.DataFrame(rows)
    write_csv(frame, TABLES / "ablations.csv")

    stat_rows = []
    for split in ("test_id", *SHIFT_SPLITS):
        errs = {n: preds[n][split] for n in names if split in preds[n]}
        if "full" not in errs or len(errs) < 2:
            continue
        _, comps = ST.summarize_metric(
            errs, "abs_error", "full", cfg.eval.bootstrap, cfg.run.seed
        )
        for c in comps:
            stat_rows.append({"split": split, **c.to_dict()})
    stats_frame = pd.DataFrame(stat_rows)
    write_csv(stats_frame, TABLES / "ablation_statistics.csv")
    LOG.info("ablations written: %d variants", len(names))
    return frame


# --------------------------------------------------------------------------- #
# Counterfactual criticality — the headline experiment
# --------------------------------------------------------------------------- #


def run_criticality(cfg: Config) -> dict[str, pd.DataFrame]:
    """Exhaustive counterfactual sweep, ranking comparison and budget curve.

    For each of ``cfg.eval.n_criticality_networks`` unseen networks:

    1. Run the simulator once per candidate node to get the **true** criticality
       ranking (the oracle).
    2. Score the same candidates with the surrogate, the reference-style tabular
       model and the topology heuristic.
    3. Report recall@k, regret and rank agreement of each against the oracle.
    4. Find pairs the tabular model and the counterfactual ranking order
       oppositely, and let the simulator adjudicate.
    5. Measure the decision-quality-versus-compute-budget curve.

    Networks are drawn from the shift pool, so none of them was seen in training.
    Evaluating criticality on a training topology would be measuring memorisation.
    """
    ds = prepare(cfg)
    sim = _sim_config(cfg)
    train_rows = rows_from_split(ds, "train")
    tab_gbt = TabularRiskModel("gbt", cfg.run.seed).fit(train_rows.x, train_rows.y)
    tab_ridge = TabularRiskModel("ridge", cfg.run.seed).fit(train_rows.x, train_rows.y)
    models, _ = train_or_load_ensemble(cfg, ds, N_NODE_FEATURES, N_EDGE_FEATURES)
    heur = TopologyHeuristic()

    shift_net_ids = sorted({s.net_id for s in ds.splits["shift_topo"]})
    chosen = shift_net_ids[: cfg.eval.n_criticality_networks]
    # The k used for the headline log line and the budget curve. Taken from the
    # config rather than hardcoded: the smoke config uses recall_k = [3, 5] and a
    # hardcoded 10 crashed the stage with a KeyError.
    ks = tuple(cfg.eval.recall_k)
    k_report = ks[1] if len(ks) > 1 else ks[0]
    node_norm, edge_norm, tab_norm = fit_normalisers(ds, "train")

    rank_rows: list[dict] = []
    disagree_rows: list[dict] = []
    budget_rows: list[dict] = []
    detail_rows: list[dict] = []

    for net_id in chosen:
        bundle = ds.networks[net_id]
        net = bundle.net
        cands = node_candidates(net)
        n_dem = net.demand_nodes.size
        seed = cfg.run.seed + 5000 + net_id

        t0 = time.perf_counter()
        truth = simulate_truth(net, cfg, sim, seed, cands)
        sim_seconds = time.perf_counter() - t0
        sim_ms_each = sim_seconds * 1000.0 / max(cands.size, 1)

        probes = probe_scenarios(bundle, cfg, sim, seed, cands, ds.traj_periods)
        for p in probes:
            p.node_feat = node_norm.transform(p.node_feat)
            p.edge_feat = edge_norm.transform(p.edge_feat)
            p.tabular = tab_norm.transform(p.tabular)

        probe_ds = ScenarioDataset(
            networks=[bundle], splits={"probe": probes}, traj_periods=ds.traj_periods
        )
        t0 = time.perf_counter()
        pb = iterate_batches(probes, probe_ds, cfg.train.batch_graphs, False)
        sur = predict_ensemble(models, pb, probe_ds, cfg.run.device)
        sur_seconds = time.perf_counter() - t0
        sur_scores = aggregate_surrogate_scores(sur.impact, cands.size, n_dem)

        t0 = time.perf_counter()
        tab_x = np.concatenate([p.tabular for p in probes], axis=0)
        tab_scores = tab_gbt.predict(tab_x).reshape(cands.size, n_dem).sum(axis=1)
        tab_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        ridge_scores = tab_ridge.predict(tab_x).reshape(cands.size, n_dem).sum(axis=1)
        ridge_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        heur_all = heur.node_scores(net)
        heur_scores = heur_all[cands] - heur_all.min()
        heur_seconds = time.perf_counter() - t0

        scores = {
            "surrogate": sur_scores,
            "tabular_gbt": tab_scores,
            "tabular_ridge": ridge_scores,
            "topology_heuristic": heur_scores,
        }
        secs = {
            "surrogate": sur_seconds,
            "tabular_gbt": tab_seconds,
            "tabular_ridge": ridge_seconds,
            "topology_heuristic": heur_seconds,
        }
        for name, sc in scores.items():
            rank_rows.append(
                {
                    "network": net.name,
                    "net_id": net_id,
                    "method": name,
                    "n_candidates": int(cands.size),
                    "sim_seconds": sim_seconds,
                    "sim_ms_per_candidate": sim_ms_each,
                    "method_seconds": secs[name],
                    "speedup_vs_simulator": sim_seconds / max(secs[name], 1e-9),
                    # A method that emits the same score for every candidate has
                    # no ranking at all; without these columns that shows up only
                    # as a suspiciously round recall.
                    "score_std": float(np.std(sc)),
                    "score_range": float(np.max(sc) - np.min(sc)),
                    "distinct_scores": int(np.unique(np.round(sc, 9)).size),
                    "is_constant": bool(np.ptp(sc) < 1e-12),
                    **RANK.ranking_report(sc, truth, tuple(cfg.eval.recall_k)),
                }
            )

        ss_reach = net.sole_source_reach()
        d_reach = net.downstream_reach()
        for i, c in enumerate(cands):
            detail_rows.append(
                {
                    "network": net.name,
                    "node": int(c),
                    "tier": int(net.tier[c]),
                    "truth": float(truth[i]),
                    "surrogate": float(sur_scores[i]),
                    "tabular_gbt": float(tab_scores[i]),
                    "tabular_ridge": float(ridge_scores[i]),
                    "topology_heuristic": float(heur_scores[i]),
                    "sole_source_reach": float(ss_reach[c]),
                    "downstream_reach": float(d_reach[c]),
                }
            )

        for d in find_disagreements(net, cands, tab_scores, sur_scores, truth, top_k=10):
            disagree_rows.append({"network": net.name, **d.to_dict()})

        budgets = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
        for r in budget_curve(
            truth,
            sur_scores,
            sim_ms_each,
            sur_seconds * 1000.0,
            budgets,
            k=k_report,
            seed=cfg.run.seed,
        ):
            budget_rows.append({"network": net.name, **r})
        LOG.info(
            "criticality %-10s %d candidates: oracle %.2fs, surrogate %.3fs, recall@%d %.2f",
            net.name,
            cands.size,
            sim_seconds,
            sur_seconds,
            k_report,
            rank_rows[-3][f"recall@{k_report}"],
        )

    frames = {
        "ranking": pd.DataFrame(rank_rows),
        "disagreements": pd.DataFrame(disagree_rows),
        "budget": pd.DataFrame(budget_rows),
        "detail": pd.DataFrame(detail_rows),
    }
    write_csv(frames["ranking"], TABLES / "criticality_ranking.csv")
    write_csv(frames["disagreements"], TABLES / "disagreements.csv")
    write_csv(frames["budget"], TABLES / "budget_curve.csv")
    write_csv(frames["detail"], TABLES / "criticality_detail.csv")

    metric_cols = [
        c
        for c in frames["ranking"].columns
        if c.startswith(("recall@", "regret_frac@", "spearman", "precision@", "jaccard@"))
    ] + [
        "speedup_vs_simulator", "method_seconds", "sim_seconds",
        "score_std", "score_range", "distinct_scores",
    ]
    agg = frames["ranking"].groupby("method")[metric_cols].mean().reset_index()
    write_csv(agg, TABLES / "criticality_summary.csv")
    frames["summary"] = agg
    LOG.info("criticality written for %d networks", len(chosen))
    return frames


# --------------------------------------------------------------------------- #
# Efficiency
# --------------------------------------------------------------------------- #


def run_efficiency(cfg: Config) -> pd.DataFrame:
    """Measure latency, parameters and the honest break-even accounting.

    Three things are measured, not computed:

    * simulator wall-clock per counterfactual, at the network sizes used;
    * surrogate wall-clock per scenario, single and batched;
    * training and dataset-generation cost, which the break-even calculation
      charges against the surrogate.

    The last point is the one usually omitted. A surrogate that takes an hour to
    train to save a minute of screening is a bad trade, and the break-even
    scenario count says where the line is.
    """
    import torch

    from sndsur.analysis.counterfactual import standard_outage
    from sndsur.sim.simulator import counterfactual as cf_run
    from sndsur.sim.simulator import demand_realisation
    from sndsur.sim.simulator import simulate as sim_run

    ds = prepare(cfg)
    sim = _sim_config(cfg)
    rows: list[dict] = []

    models, records = train_or_load_ensemble(cfg, ds, N_NODE_FEATURES, N_EDGE_FEATURES)
    # Training seconds come from the records, never from the wall-clock of this
    # call: a cache hit would otherwise be reported as a near-zero training cost
    # and inflate the break-even number into nonsense.
    single_train_seconds = float(np.mean([r["train_seconds"] for r in records]))
    ensemble_train_seconds = float(np.sum([r["train_seconds"] for r in records]))

    dataset_seconds = float("nan")
    dpath = TABLES / "dataset.csv"
    if dpath.exists():
        dframe = pd.read_csv(dpath)
        if "build_seconds_total" in dframe.columns:
            dataset_seconds = float(dframe["build_seconds_total"].iloc[0])

    pools = (
        ("shift_topo", sorted({s.net_id for s in ds.splits["shift_topo"]})[:2]),
        ("shift_size", sorted({s.net_id for s in ds.splits["shift_size"]})[:2]),
    )

    # --- simulator, per network size ---
    for label, net_ids in pools:
        for net_id in net_ids:
            net = ds.networks[net_id].net
            cands = node_candidates(net)
            seed = cfg.run.seed + 9000 + net_id
            d = demand_realisation(net, sim, seed)
            base = sim_run(net, sim, d, None)
            probe = standard_outage(int(cands[0]), cfg).bind(net)
            t = benchmark(
                lambda n=net, p=probe, s=seed, b=base: cf_run(n, sim, p, s, baseline=b),
                f"simulator_{net.name}",
                cfg.eval.bench_warmup,
                cfg.eval.bench_repeats,
            )
            rows.append(
                {
                    "component": "simulator",
                    "variant": label,
                    "network": net.name,
                    "n_nodes": net.n_nodes,
                    "n_edges": len(net.edges()),
                    "n_candidates": int(cands.size),
                    "median_ms": t.median_ms,
                    "iqr_ms": t.iqr_ms,
                    "per_item_ms": t.per_item_ms,
                    "n_warmup": t.n_warmup,
                    "n_repeats": t.n_repeats,
                    "params": float("nan"),
                }
            )

    # --- surrogate ---
    model = models[0]
    model.eval()
    node_norm, edge_norm, tab_norm = fit_normalisers(ds, "train")
    for label, net_ids in pools:
        for net_id in net_ids:
            bundle = ds.networks[net_id]
            net = bundle.net
            cands = node_candidates(net)
            probes = probe_scenarios(bundle, cfg, sim, cfg.run.seed, cands, ds.traj_periods)
            for p in probes:
                p.node_feat = node_norm.transform(p.node_feat)
                p.edge_feat = edge_norm.transform(p.edge_feat)
                p.tabular = tab_norm.transform(p.tabular)
            pds = ScenarioDataset([bundle], {"probe": probes}, ds.traj_periods)
            single = collate(probes[:1], pds)
            allb = collate(probes, pds)

            for name, batch, items in (
                ("surrogate_single", single, 1),
                ("surrogate_batched_all_candidates", allb, int(cands.size)),
            ):

                def call(b=batch):
                    with torch.no_grad():
                        model(b.node_feat, b.edge_index, b.edge_feat, b.demand_rows)

                t = benchmark(
                    call,
                    f"{name}_{net.name}",
                    cfg.eval.bench_warmup,
                    cfg.eval.bench_repeats,
                    items=items,
                )
                rows.append(
                    {
                        "component": name,
                        "variant": label,
                        "network": net.name,
                        "n_nodes": net.n_nodes,
                        "n_edges": len(net.edges()),
                        "n_candidates": int(cands.size),
                        "median_ms": t.median_ms,
                        "iqr_ms": t.iqr_ms,
                        "per_item_ms": t.per_item_ms,
                        "n_warmup": t.n_warmup,
                        "n_repeats": t.n_repeats,
                        "params": float(model.n_params()),
                    }
                )

    frame = pd.DataFrame(rows)
    write_csv(frame, TABLES / "efficiency.csv")

    sim_ms = float(
        frame[(frame["component"] == "simulator") & (frame["variant"] == "shift_topo")][
            "median_ms"
        ].median()
    )
    sur_ms = float(
        frame[
            (frame["component"] == "surrogate_batched_all_candidates")
            & (frame["variant"] == "shift_topo")
        ]["per_item_ms"].median()
    )
    be = break_even_scenarios(single_train_seconds, sim_ms, sur_ms, dataset_seconds)
    be_ens = break_even_scenarios(ensemble_train_seconds, sim_ms, sur_ms, dataset_seconds)
    be.update(
        {
            "ensemble_train_seconds": ensemble_train_seconds,
            "ensemble_members": cfg.model.ensemble,
            "params_per_member": records[0]["params"],
            "break_even_scenarios_ensemble": be_ens["break_even_scenarios"],
            "note": (
                "break-even charges one member training run plus full dataset "
                "generation; the ensemble column charges all members"
            ),
        }
    )
    write_csv(pd.DataFrame([be]), TABLES / "break_even.csv")
    LOG.info(
        "efficiency: simulator %.1f ms, surrogate %.3f ms/scenario, %.0fx, break-even %.0f",
        sim_ms,
        sur_ms,
        be["speedup_per_scenario"],
        be["break_even_scenarios"],
    )
    return frame
