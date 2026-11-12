"""Etapa 8: the statistics match hand calculations, the bootstrap is paired and reproducible, the re-weighting
reduces to the unweighted metrics, and the pre-registered rules for H1 and H2 give the expected verdicts."""
import numpy as np
import polars as pl
import pytest
from scipy import stats
from sklearn.metrics import matthews_corrcoef

from src.evaluation.evaluate import hypotheses, load_oof, metrics_by_window, recall_by_group
from src.evaluation.metrics import binary_metrics
from src.evaluation.stats import (bootstrap_counts, corrected_resampled_ttest, holm, mcc_from_counts,
                                  percentile_interval, projected_ppv, wilson)
from src.utils.io import write_csv_gz

LEVEL = 0.95
# one label per category: (label34, category8, window) — names as in src/data/labels.py
WEB, BRUTE, RECON, DDOS = (("Backdoor_Malware", "Web", 10), ("DictionaryBruteForce", "BruteForce", 10),
                           ("Recon-HostDiscovery", "Recon", 10), ("DDoS-ACK_Fragmentation", "DDoS", 100))


def test_corrected_resampled_ttest_by_hand():
    a, b = [0.80, 0.82, 0.81, 0.83, 0.79], [0.78, 0.80, 0.80, 0.80, 0.78]
    # d = (0.02, 0.02, 0.01, 0.03, 0.01): mean 0.018, sample variance 7e-5; n_test / n_train = 0.25
    t_hand = 0.018 / np.sqrt((1 / 5 + 0.25) * 7e-5)
    t, p = corrected_resampled_ttest(a, b, n_train=400, n_test=100)
    assert t == pytest.approx(t_hand) and p == pytest.approx(2 * stats.t.sf(t_hand, df=4))
    assert all(np.isnan(corrected_resampled_ttest([0.8] * 5, [0.7] * 5, 400, 100)))   # no variance: no test


def test_holm_and_wilson_reference_values():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(holm([0.5, 0.6]), [1.0, 1.0])                          # capped and monotone
    low, high = wilson(5, 10, LEVEL)
    assert (low, high) == (pytest.approx(0.236593, abs=1e-6), pytest.approx(0.763407, abs=1e-6))
    assert wilson(0, 10, LEVEL)[0] == pytest.approx(0.0, abs=1e-12)


def test_mcc_from_counts_matches_scikit_learn():
    rng = np.random.default_rng(0)
    y, pred = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
    tn, fp, fn, tp = (int(((y == a) & (pred == b)).sum()) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)))
    assert float(mcc_from_counts(tn, fp, fn, tp)) == pytest.approx(matthews_corrcoef(y, pred))
    assert float(mcc_from_counts(10, 0, 10, 0)) == 0.0                                # empty margin, as sklearn


def test_bootstrap_is_paired_reproducible_and_stratified():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], [60, 40])
    strata = np.where(y == 1, np.where(np.arange(100) % 2 == 0, "a", "b"), "benign")
    model = rng.integers(0, 2, 100)
    preds = np.vstack([model, model, rng.integers(0, 2, 100)])
    counts = bootstrap_counts(y, preds, strata, n_resamples=50, seed=7)
    np.testing.assert_array_equal(counts, bootstrap_counts(y, preds, strata, n_resamples=50, seed=7))
    mcc = mcc_from_counts(counts[..., 0], counts[..., 1], counts[..., 2], counts[..., 3])
    assert percentile_interval(mcc[:, 0] - mcc[:, 1], LEVEL) == (0.0, 0.0)          # identical models: no difference
    assert (counts.sum(axis=2) == 100).all()
    assert (counts[..., 2] + counts[..., 3] == 40).all()                              # strata keep the 40 attacks


def test_projected_ppv():
    # balanced classes: projected precision equals the precision of the confusion matrix
    assert float(projected_ppv(0.9, 0.95, 0.5)) == pytest.approx(0.9 / (0.9 + 0.05))
    assert float(projected_ppv(0.885, 0.945, 0.01)) == pytest.approx(0.14, abs=0.005)


def test_unit_weights_give_the_unweighted_metrics():
    rng = np.random.default_rng(2)
    y, pred, score = rng.integers(0, 2, 300), rng.integers(0, 2, 300), rng.random(300)
    plain, weighted = binary_metrics(y, pred, score), binary_metrics(y, pred, score, sample_weight=np.ones(300))
    assert all(isinstance(plain[c], int) for c in ("tp", "fn", "tn", "fp"))
    for key, value in plain.items():
        assert weighted[key] == pytest.approx(value)


def tiny_oof(missed: dict) -> pl.DataFrame:
    """10 attacks of each of four types and 10 benign; `missed` = false negatives per type."""
    rows = []
    for label, category, window in (WEB, BRUTE, RECON, DDOS):
        rows += [(label, category, window, 1, int(i >= missed.get(category, 0))) for i in range(10)]
    rows += [("BenignTraffic", "Benign", 10, 0, 0)] * 10
    return pl.DataFrame(rows, schema=["label34", "category8", "window", "y", "pred"], orient="row")


@pytest.mark.parametrize("missed, verdict", [
    ({"Web": 4, "BruteForce": 4, "Recon": 4}, "confirmed"),
    ({"Web": 6, "DDoS": 4}, "partially confirmed"),
    ({"DDoS": 6}, "refuted"),
])
def test_h2_rule(missed, verdict):
    recall = recall_by_group(tiny_oof(missed), "m", LEVEL)
    for lv in ("window", "category8", "label34"):
        part = recall.filter(pl.col("level") == lv)
        assert part["missed_share"].sum() == pytest.approx(1.0) and part["instance_share"].sum() == pytest.approx(1.0)
    assert ((recall["ci_low"] <= recall["sensitivity"]) & (recall["sensitivity"] <= recall["ci_high"])).all()
    summary = pl.DataFrame({"model": ["m", "decision_tree"], "mcc_mean": [0.8, 0.7]})
    tests = pl.DataFrame({"model_a": ["m"], "model_b": ["decision_tree"], "diff_mean": [0.1], "reject_h0": [True],
                          "ci_low": [0.05]})
    confusion = pl.DataFrame({"model": ["m", "decision_tree"], "fn": [10, 15], "fp": [5, 5]})
    hyp = hypotheses(summary, tests, confusion, recall, "m", ["Web", "BruteForce", "Recon"])
    assert hyp["H2"]["verdict"] == verdict
    assert hyp["H1"]["first_half_ensembles_have_highest_mcc"] == "confirmed"
    assert hyp["H1"]["second_half_numbers"][0]["tree_extra_errors_relative"] == pytest.approx(20 / 15 - 1)


def test_h1_first_half_needs_every_condition():
    summary = pl.DataFrame({"model": ["m", "decision_tree"], "mcc_mean": [0.8, 0.79]})
    confusion = pl.DataFrame({"model": ["m", "decision_tree"], "fn": [10, 11], "fp": [5, 5]})
    recall = recall_by_group(tiny_oof({}), "m", LEVEL)
    for reject, ci_low in ((False, 0.01), (True, -0.01)):
        tests = pl.DataFrame({"model_a": ["m"], "model_b": ["decision_tree"], "diff_mean": [0.01],
                              "reject_h0": [reject], "ci_low": [ci_low]})
        hyp = hypotheses(summary, tests, confusion, recall, "m", ["Web"])
        assert hyp["H1"]["first_half_ensembles_have_highest_mcc"] == "not confirmed"


def test_metrics_by_window_keep_all_benign_and_one_window_of_attacks():
    oof = tiny_oof({"Web": 5}).with_columns(pl.Series("outer_fold", [0, 1] * 25), pl.lit(0.5).alias("score"))
    table = metrics_by_window({"m": oof}, ["m"]).filter(pl.col("statistic") == "pooled")
    w10, w100 = (table.filter(pl.col("attack_window") == w).row(0, named=True) for w in (10, 100))
    assert (w10["n_attack"], w10["n_benign"], w100["n_attack"], w100["n_benign"]) == (30, 10, 10, 10)
    assert w10["sensitivity"] == pytest.approx(25 / 30) and w100["sensitivity"] == 1.0


def test_load_oof_rejects_incomplete_predictions(tmp_path):
    labels = pl.DataFrame({"position": [0, 1, 2, 3], "label34": ["BenignTraffic", WEB[0], WEB[0], "BenignTraffic"],
                           "outer": [0, 0, 0, 0], "category8": ["Benign", "Web", "Web", "Benign"],
                           "window": [10, 10, 10, 10]})
    pred = pl.DataFrame({"position": [0, 1, 2, 3], "y": [0, 1, 1, 0], "pred": [0, 1, 0, 0], "score": [.1, .9, .4, .2]})
    write_csv_gz(pred, tmp_path / "m" / "fold0_predictions.csv.gz")
    assert load_oof(tmp_path, "m", 1, labels).height == 4
    write_csv_gz(pred.head(3), tmp_path / "m" / "fold0_predictions.csv.gz")
    with pytest.raises(ValueError, match="exactly once"):
        load_oof(tmp_path, "m", 1, labels)
    write_csv_gz(pred.with_columns(pl.Series("y", [0, 0, 1, 0])), tmp_path / "m" / "fold0_predictions.csv.gz")
    with pytest.raises(ValueError, match="disagree"):
        load_oof(tmp_path, "m", 1, labels)
