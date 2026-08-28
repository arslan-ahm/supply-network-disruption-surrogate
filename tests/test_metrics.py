"""Tests for fidelity, ranking, calibration and statistics.

These are checked against closed forms wherever one exists, and against
hand-constructed cases where it does not. A metric bug is the most dangerous kind
of bug in a repository like this, because it produces plausible numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from sndsur.metrics import calibration as CAL
from sndsur.metrics import fidelity as FID
from sndsur.metrics import ranking as RANK
from sndsur.metrics import stats as ST

# --------------------------------------------------------------------------- #
# Fidelity
# --------------------------------------------------------------------------- #


def test_mae_of_a_perfect_prediction_is_zero():
    x = np.array([0.0, 0.5, 1.0])
    assert FID.mae(x, x) == pytest.approx(0.0)
    assert FID.rmse(x, x) == pytest.approx(0.0)
    assert FID.bias(x, x) == pytest.approx(0.0)


def test_mae_matches_the_closed_form():
    p = np.array([1.0, 2.0, 3.0])
    t = np.array([1.5, 2.5, 1.0])
    assert FID.mae(p, t) == pytest.approx((0.5 + 0.5 + 2.0) / 3)
    assert FID.rmse(p, t) == pytest.approx(np.sqrt((0.25 + 0.25 + 4.0) / 3))


def test_bias_sign_says_which_way_the_model_errs():
    assert FID.bias(np.array([2.0, 2.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)
    assert FID.bias(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# MAE on nonzero truth — the metric a constant predictor cannot win
# --------------------------------------------------------------------------- #


def test_mae_nonzero_truth_restricts_to_biting_rows():
    t = np.array([0.0, 0.0, 0.4, 0.6])
    p = np.array([0.9, 0.9, 0.5, 0.5])
    val, n = FID.mae_nonzero_truth(p, t)
    assert n == 2
    assert val == pytest.approx((0.1 + 0.1) / 2)


def test_a_zero_predictor_scores_the_mean_nonzero_truth():
    """The whole reason this metric exists: zero cannot win it."""
    rng = np.random.default_rng(0)
    t = np.zeros(1000)
    idx = rng.choice(1000, 75, replace=False)
    t[idx] = rng.uniform(0.05, 0.9, 75)
    zero_pooled = FID.mae(np.zeros(1000), t)
    zero_nz, n = FID.mae_nonzero_truth(np.zeros(1000), t)
    assert n == 75
    assert zero_nz == pytest.approx(t[idx].mean())
    # Pooled MAE flatters the constant by the zero fraction; the restricted one
    # does not. This factor is why the two must be reported side by side.
    assert zero_nz > 10.0 * zero_pooled


def test_a_zero_predictor_has_the_best_possible_pooled_mae_here():
    """States the trap directly: on a 92.5%-zero target, zero wins MAE."""
    rng = np.random.default_rng(1)
    t = np.zeros(2000)
    idx = rng.choice(2000, 150, replace=False)
    t[idx] = rng.uniform(0.05, 0.9, 150)
    zero = FID.mae(np.zeros(2000), t)
    for c in (t.mean(), 0.05, 0.1, 0.5):
        assert FID.mae(np.full(2000, c), t) >= zero


def test_mae_nonzero_truth_is_nan_when_nothing_bit():
    val, n = FID.mae_nonzero_truth(np.ones(5), np.zeros(5))
    assert n == 0
    assert np.isnan(val)


def test_mae_nonzero_truth_ignores_non_finite_rows():
    t = np.array([0.0, 0.5, np.nan, 0.5])
    p = np.array([0.0, 0.4, 9.0, np.nan])
    val, n = FID.mae_nonzero_truth(p, t)
    assert n == 1
    assert val == pytest.approx(0.1)


def test_fidelity_report_carries_the_nonzero_columns():
    t = np.array([0.0, 0.0, 0.0, 0.5])
    p = np.array([0.0, 0.0, 0.0, 0.3])
    rep = FID.fidelity_report(p, t)
    assert rep["n_nonzero_truth"] == 1
    assert rep["mae_nonzero_truth"] == pytest.approx(0.2)
    assert rep["mae"] == pytest.approx(0.05)


def test_metrics_ignore_non_finite_entries():
    p = np.array([1.0, np.nan, 3.0])
    t = np.array([1.0, 5.0, 4.0])
    assert FID.mae(p, t) == pytest.approx(0.5)


def test_metrics_on_empty_input_are_nan():
    e = np.zeros(0)
    assert np.isnan(FID.mae(e, e))
    assert np.isnan(FID.rmse(e, e))
    assert np.isnan(FID.r2(e, e))


def test_r2_is_one_for_a_perfect_fit():
    t = np.array([1.0, 2.0, 4.0, 8.0])
    assert FID.r2(t, t) == pytest.approx(1.0)


def test_r2_is_nan_when_truth_has_no_variance():
    """A split where nothing was disrupted has an undefined R^2, not zero."""
    t = np.zeros(5)
    assert np.isnan(FID.r2(np.arange(5.0), t))


def test_spearman_is_one_under_a_monotone_transform():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert FID.spearman(np.exp(t), t) == pytest.approx(1.0)
    assert FID.spearman(-t, t) == pytest.approx(-1.0)


def test_spearman_is_nan_when_a_side_is_constant():
    t = np.array([1.0, 2.0, 3.0])
    assert np.isnan(FID.spearman(np.ones(3), t))


def test_kendall_matches_scipy():
    rng = np.random.default_rng(0)
    p, t = rng.normal(size=30), rng.normal(size=30)
    assert FID.kendall_tau(p, t) == pytest.approx(
        stats.kendalltau(p, t, variant="b").statistic
    )


def test_grouped_spearman_skips_small_groups():
    pred = np.arange(8.0)
    true = np.arange(8.0)
    group = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    mean, n, vals = FID.grouped_spearman(pred, true, group, min_size=3)
    assert n == 2  # group 1 has only two members
    assert mean == pytest.approx(1.0)
    assert vals.size == 2


def test_grouped_spearman_differs_from_pooled():
    """Pooling can be strongly positive while every within-group correlation is -1.

    This is the reason rank fidelity is reported within scenario rather than
    pooled: the easy between-scenario signal (big disruptions hurt more than
    small ones) would otherwise carry the number on its own.
    """
    true = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    pred = np.array([2.0, 1.0, 0.0, 12.0, 11.0, 10.0])
    group = np.array([0, 0, 0, 1, 1, 1])
    assert FID.spearman(pred, true) > 0.5
    mean, n, _ = FID.grouped_spearman(pred, true, group, min_size=3)
    assert n == 2
    assert mean == pytest.approx(-1.0)


def test_top1_agreement_counts_only_decidable_groups():
    """A group whose truth is entirely tied has no correct answer and is skipped."""
    pred = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    true = np.array([5.0, 1.0, 2.0, 2.0, 7.0, 3.0])
    group = np.array([0, 0, 1, 1, 2, 2])
    frac, n = FID.top1_agreement(pred, true, group)
    # group 0: pred picks index 0, truth's max is index 0 -> hit
    # group 1: truth is entirely tied -> excluded
    # group 2: pred picks the second element, truth's max is the first -> miss
    assert n == 2
    assert frac == pytest.approx(0.5)


def test_top1_agreement_excludes_all_tied_groups():
    pred = np.array([1.0, 0.0])
    true = np.array([0.0, 0.0])
    frac, n = FID.top1_agreement(pred, true, np.array([0, 0]))
    assert n == 0
    assert np.isnan(frac)


def test_trajectory_metrics_report_active_rows():
    pred = np.zeros((3, 4))
    true = np.zeros((3, 4))
    true[0, 2] = 5.0
    out = FID.trajectory_metrics(pred, true)
    assert out["traj_rows"] == 3
    assert out["traj_active_rows"] == 1
    assert out["traj_mae"] == pytest.approx(5.0 / 12)


def test_peak_period_error_is_zero_when_peaks_align():
    true = np.array([[0.0, 3.0, 1.0, 0.0]])
    pred = np.array([[0.0, 9.0, 2.0, 0.0]])
    assert FID.trajectory_metrics(pred, true)["peak_period_error"] == pytest.approx(0.0)


def test_trajectory_metrics_are_nan_with_no_active_rows():
    z = np.zeros((2, 3))
    out = FID.trajectory_metrics(z, z)
    assert np.isnan(out["peak_period_error"])
    assert np.isnan(out["traj_mae_active"])


def test_timing_metrics_count_defined_rows_only():
    out = FID.timing_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, np.nan, 4.0]),
        np.array([1.0, 1.0]),
        np.array([np.nan, np.nan]),
    )
    assert out["time_to_impact_n"] == 2
    assert out["time_to_impact_mae"] == pytest.approx(0.5)
    assert out["recovery_time_n"] == 0
    assert np.isnan(out["recovery_time_mae"])


def test_fidelity_report_has_every_key():
    rng = np.random.default_rng(1)
    t = rng.random(40)
    p = t + rng.normal(0, 0.05, 40)
    g = np.repeat(np.arange(8), 5)
    rep = FID.fidelity_report(p, t, g)
    for k in ("mae", "rmse", "bias", "r2", "spearman_pooled", "kendall_pooled",
              "spearman_within_scenario", "top1_agreement", "n"):
        assert k in rep


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def test_true_critical_set_excludes_zero_impact_nodes():
    t = np.array([0.0, 0.0, 0.5, 0.2, 0.0])
    s = RANK.true_critical_set(t, 4)
    assert s.tolist() == [2, 3]


def test_true_critical_set_is_ordered_by_impact():
    t = np.array([0.1, 0.9, 0.5])
    assert RANK.true_critical_set(t, 3).tolist() == [1, 2, 0]


def test_recall_is_one_for_a_perfect_ranking():
    t = np.array([0.5, 0.4, 0.3, 0.0, 0.0])
    r, n = RANK.recall_at_k(t, t, 3)
    assert r == pytest.approx(1.0)
    assert n == 3


def test_recall_is_nan_when_nothing_is_critical():
    t = np.zeros(5)
    r, n = RANK.recall_at_k(np.arange(5.0), t, 3)
    assert np.isnan(r)
    assert n == 0


def test_recall_is_zero_for_an_exactly_reversed_ranking():
    t = np.array([1.0, 0.9, 0.8, 0.1, 0.05, 0.01])
    r, _ = RANK.recall_at_k(-t, t, 2)
    assert r == pytest.approx(0.0)


def test_recall_denominator_shrinks_with_few_critical_nodes():
    """With only two non-zero nodes, recall@10 is out of 2, not out of 10."""
    t = np.array([0.0] * 8 + [0.3, 0.4])
    r, n = RANK.recall_at_k(t, t, 10)
    assert n == 2
    assert r == pytest.approx(1.0)


def test_precision_at_k_penalises_a_shortlist_of_irrelevant_nodes():
    t = np.array([0.0, 0.0, 0.0, 0.5])
    pred = np.array([1.0, 0.9, 0.8, 0.0])
    p, k = RANK.precision_at_k(pred, t, 3)
    assert p == pytest.approx(0.0)
    assert k == 3


def test_decision_regret_is_zero_for_the_oracle_ranking():
    t = np.array([0.5, 0.3, 0.1, 0.0])
    out = RANK.decision_regret(t, t, 2)
    assert out["regret"] == pytest.approx(0.0)
    assert out["regret_fraction"] == pytest.approx(0.0)
    assert out["oracle_averted"] == pytest.approx(0.8)


def test_decision_regret_matches_the_closed_form():
    t = np.array([1.0, 0.6, 0.2])
    pred = np.array([0.0, 0.0, 1.0])  # picks the worst node first
    out = RANK.decision_regret(pred, t, 1)
    assert out["oracle_averted"] == pytest.approx(1.0)
    assert out["averted"] == pytest.approx(0.2)
    assert out["regret"] == pytest.approx(0.8)
    assert out["regret_fraction"] == pytest.approx(0.8)


def test_decision_regret_fraction_is_nan_when_nothing_can_be_averted():
    t = np.zeros(4)
    assert np.isnan(RANK.decision_regret(np.arange(4.0), t, 2)["regret_fraction"])


def test_jaccard_is_one_for_identical_shortlists():
    t = np.array([3.0, 2.0, 1.0, 0.0])
    assert RANK.top_k_jaccard(t, t, 2) == pytest.approx(1.0)


def test_jaccard_is_zero_for_disjoint_shortlists():
    t = np.array([3.0, 2.0, 1.0, 0.0])
    p = np.array([0.0, 1.0, 2.0, 3.0])
    assert RANK.top_k_jaccard(p, t, 2) == pytest.approx(0.0)


def test_kendall_on_union_ignores_the_tied_tail():
    t = np.array([5.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p = t.copy()
    assert RANK.kendall_on_union(p, t, 3) == pytest.approx(1.0)


def test_ranking_report_has_every_k():
    rng = np.random.default_rng(2)
    t = np.clip(rng.random(40) - 0.6, 0, None)
    p = t + rng.normal(0, 0.02, 40)
    rep = RANK.ranking_report(p, t, (5, 10))
    for k in (5, 10):
        for key in ("recall@", "precision@", "jaccard@", "regret@", "regret_frac@"):
            assert f"{key}{k}" in rep
    assert rep["n_candidates"] == 40


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_gaussian_interval_width_matches_the_normal_quantile():
    """Away from the zero floor the width is exactly 2 z sigma."""
    mu = np.array([100.0])
    sigma = np.array([2.0])
    lo, hi = CAL.gaussian_interval(mu, sigma, 0.95)
    assert (hi - lo)[0] == pytest.approx(2 * stats.norm.ppf(0.975) * 2.0)


def test_gaussian_interval_width_is_reduced_by_the_zero_clip():
    """Near zero the interval is one-sided, and the reported width says so.

    Service loss cannot be negative, so counting the sub-zero half of a Gaussian
    interval as width would flatter every sharpness number in the calibration
    table.
    """
    lo, hi = CAL.gaussian_interval(np.array([1.0]), np.array([2.0]), 0.95)
    assert lo[0] == pytest.approx(0.0)
    assert (hi - lo)[0] == pytest.approx(1.0 + stats.norm.ppf(0.975) * 2.0)


def test_gaussian_interval_is_clipped_at_zero():
    lo, _ = CAL.gaussian_interval(np.array([0.01]), np.array([1.0]), 0.95)
    assert lo[0] == pytest.approx(0.0)


def test_coverage_of_a_well_specified_gaussian_is_near_nominal():
    rng = np.random.default_rng(0)
    mu = np.full(20000, 5.0)
    sigma = np.full(20000, 1.0)
    true = rng.normal(5.0, 1.0, 20000)
    lo, hi = CAL.gaussian_interval(mu, sigma, 0.90)
    cov, width, n = CAL.coverage_and_width(lo, hi, true)
    assert cov == pytest.approx(0.90, abs=0.02)
    assert n == 20000
    assert width == pytest.approx(2 * stats.norm.ppf(0.95), rel=0.01)


def test_coverage_is_one_for_an_absurdly_wide_interval():
    t = np.array([0.1, 0.5])
    cov, _, _ = CAL.coverage_and_width(np.zeros(2), np.full(2, 1e6), t)
    assert cov == pytest.approx(1.0)


def test_calibration_table_has_one_row_per_level():
    rows = CAL.calibration_table(
        np.zeros(10), np.ones(10), np.zeros(10), (0.5, 0.9)
    )
    assert len(rows) == 2
    assert {r["nominal"] for r in rows} == {0.5, 0.9}
    assert all("coverage_gap" in r for r in rows)


def test_quantile_calibration_reports_crossing():
    q = np.array([[1.0, 0.5, 0.2], [0.0, 0.5, 1.0]])  # first row crosses
    rows = CAL.quantile_calibration(q, (0.05, 0.5, 0.95), np.array([0.5, 0.5]))
    central = [r for r in rows if r["interval"] == "quantile_central"]
    assert central and central[0]["crossing_fraction"] == pytest.approx(0.5)


def test_quantile_calibration_returns_nothing_on_a_shape_mismatch():
    assert CAL.quantile_calibration(np.zeros((2, 2)), (0.1, 0.5, 0.9), np.zeros(2)) == []


def test_error_detection_auroc_is_one_when_sigma_is_the_error():
    rng = np.random.default_rng(3)
    err = rng.random(200)
    auroc, n_pos = CAL.error_uncertainty_auroc(err, err, 0.8)
    assert auroc == pytest.approx(1.0)
    assert n_pos == pytest.approx(40, abs=3)


def test_error_detection_auroc_is_half_for_random_sigma():
    rng = np.random.default_rng(4)
    err = rng.random(4000)
    sigma = rng.random(4000)
    auroc, _ = CAL.error_uncertainty_auroc(err, sigma, 0.8)
    assert auroc == pytest.approx(0.5, abs=0.05)


def test_ause_is_near_zero_for_a_perfect_uncertainty_estimate():
    rng = np.random.default_rng(5)
    err = rng.random(500)
    out = CAL.sparsification(err, err)
    assert out["ause"] == pytest.approx(0.0, abs=1e-6)
    assert out["n"] == 500


def test_ause_is_positive_for_useless_uncertainty():
    rng = np.random.default_rng(6)
    err = rng.random(500)
    out = CAL.sparsification(err, rng.random(500))
    assert out["ause"] > 0.05


def test_sparsification_handles_tiny_input():
    out = CAL.sparsification(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert np.isnan(out["ause"])
    assert out["n"] == 2


def test_uncertainty_shift_report_keys():
    rng = np.random.default_rng(7)
    t = rng.random(100)
    mu = t + rng.normal(0, 0.1, 100)
    rep = CAL.uncertainty_shift_report(mu, np.abs(mu - t) + 1e-3, t)
    for k in ("mean_sigma", "mean_abs_error", "sigma_error_spearman",
              "error_detection_auroc", "ause", "n"):
        assert k in rep
    assert rep["sigma_error_spearman"] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(8)
    v = rng.normal(3.0, 1.0, 500)
    ci = ST.bootstrap_ci(v, 1000, seed=0)
    assert ci.lower < ci.estimate < ci.upper
    assert ci.estimate == pytest.approx(v.mean())
    assert ci.n == 500


def test_bootstrap_ci_is_reproducible():
    v = np.arange(50.0)
    a = ST.bootstrap_ci(v, 500, seed=1)
    b = ST.bootstrap_ci(v, 500, seed=1)
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_bootstrap_ci_degenerates_gracefully():
    one = ST.bootstrap_ci(np.array([2.0]))
    assert one.estimate == one.lower == one.upper == pytest.approx(2.0)
    empty = ST.bootstrap_ci(np.zeros(0))
    assert np.isnan(empty.estimate) and empty.n == 0


def test_paired_bootstrap_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must match in shape"):
        ST.paired_bootstrap_difference(np.zeros(3), np.zeros(4))


def test_compare_detects_a_consistent_difference():
    rng = np.random.default_rng(9)
    b = rng.normal(0.0, 1.0, 200)
    a = b + 0.5
    c = ST.compare(a, b, "a", "b", 500, seed=0)
    assert c.p_value < 0.001
    assert c.difference.estimate == pytest.approx(0.5, abs=1e-9)
    assert c.significant


def test_compare_p_value_is_nan_for_identical_inputs():
    v = np.arange(20.0)
    c = ST.compare(v, v)
    assert np.isnan(c.p_value)
    assert not c.significant


def test_holm_bonferroni_is_monotone_and_conservative():
    rng = np.random.default_rng(10)
    comps = []
    for i in range(5):
        b = rng.normal(size=60)
        comps.append(ST.compare(b + 0.05 * i, b, f"m{i}", "base", 200, seed=0))
    ST.holm_bonferroni(comps)
    adj = [c.p_adjusted for c in comps if c.p_adjusted is not None]
    raw = [c.p_value for c in comps if np.isfinite(c.p_value)]
    assert len(adj) == len(raw)
    for a, r in zip(sorted(adj), sorted(raw), strict=True):
        assert a >= r - 1e-12


def test_holm_bonferroni_excludes_nan_from_the_family():
    rng = np.random.default_rng(11)
    b = rng.normal(size=50)
    good = ST.compare(b + 1.0, b, "good", "base", 200)
    tied = ST.compare(b, b, "tied", "base", 200)
    ST.holm_bonferroni([good, tied])
    # Family size is 1, so the adjusted p-value equals the raw one.
    assert good.p_adjusted == pytest.approx(good.p_value)
    assert tied.p_adjusted is None


def test_noise_scale_is_sqrt2_times_sd():
    ns = ST.noise_scale("m", [1.0, 2.0, 3.0])
    assert ns.sd == pytest.approx(1.0)
    assert ns.diff_scale == pytest.approx(np.sqrt(2.0))
    assert ns.spread == pytest.approx(2.0)
    assert ns.n_seeds == 3


def test_noise_scale_needs_at_least_two_seeds():
    with pytest.raises(ValueError, match="need >= 2 seeds"):
        ST.noise_scale("m", [1.0])


def test_noise_verdict_thresholds():
    ns = ST.noise_scale("m", [1.0, 2.0, 3.0])  # diff_scale = 1.414
    assert ST.noise_verdict(3.0, ns)[1] == "survives"
    assert ST.noise_verdict(1.6, ns)[1] == "suggestive"
    assert ST.noise_verdict(0.5, ns)[1] == "inside noise"
    assert ST.noise_verdict(float("nan"), ns)[1] == "not measured"


def test_run_level_comparison_uses_the_run_as_the_unit():
    out = ST.run_level_comparison([1.0, 1.1, 1.2], [2.0, 2.1, 2.2], "a", "b")
    assert out["n_a"] == 3
    assert out["difference"] == pytest.approx(-1.0, abs=1e-9)
    assert "Welch" in str(out["test"])


def test_run_level_comparison_refuses_with_one_run():
    out = ST.run_level_comparison([1.0], [2.0])
    assert np.isnan(out["p_value"])
    assert "not measured" in str(out["test"])


def test_summarize_metric_rejects_a_missing_baseline():
    with pytest.raises(KeyError, match="not among methods"):
        ST.summarize_metric({"a": np.zeros(3)}, "m", "missing")


def test_summarize_metric_returns_one_comparison_per_non_baseline():
    rng = np.random.default_rng(12)
    d = {n: rng.normal(size=40) for n in ("a", "b", "c")}
    intervals, comps = ST.summarize_metric(d, "m", "a", 200)
    assert set(intervals) == {"a", "b", "c"}
    assert len(comps) == 2


def test_comparison_to_dict_is_flat_and_complete():
    rng = np.random.default_rng(13)
    b = rng.normal(size=30)
    d = ST.compare(b + 0.2, b, "x", "y", 200).to_dict()
    for k in ("name_a", "mean_a", "difference", "ci_lower", "p_value", "effect_size", "n"):
        assert k in d
