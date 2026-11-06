"""Etapa 6: the metrics match their definitions, every model is scored the way it predicts, and the tie rule works."""
import warnings

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.evaluation.metrics import binary_metrics, predictions_and_scores
from src.experiments.spotcheck import evaluate_fold, select, summarize
from src.models.pipelines import auxiliary_columns, build_pipeline, feature_names, prepare
from src.models.registry import BASELINE, FACTORIES
from src.utils.config import load_config
from synthetic import sample

CFG, FEATURE_CFG, SEARCH_CFG = load_config(), load_config("features"), load_config("search_spaces")


def synthetic(n: int, seed: int):
    df = sample(n, seed)
    features = feature_names(df.columns, FEATURE_CFG, "full", auxiliary_columns(CFG))
    return prepare(df, features, FEATURE_CFG), df["y"].to_numpy(), features


def test_binary_metrics_from_a_known_confusion_matrix():
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0])            # TP 3, FN 2, FP 1, TN 4
    score = np.linspace(1.0, 0.0, 10)                              # ranks every attack above every benign
    m = binary_metrics(y_true, y_pred, score)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 2, 1, 4)
    assert m["sensitivity"] == pytest.approx(0.6) and m["specificity"] == pytest.approx(0.8)
    assert m["fpr"] == pytest.approx(1 - m["specificity"]) and m["balanced_accuracy"] == pytest.approx(0.7)
    assert m["mcc"] == pytest.approx((3 * 4 - 1 * 2) / np.sqrt(4 * 5 * 5 * 6))
    assert m["f1_attack"] == pytest.approx(2 * 3 / (2 * 3 + 1 + 2)) and m["f1_benign"] == pytest.approx(8 / 11)
    assert m["roc_auc"] == 1.0 and m["auprc"] == 1.0


def test_predictions_and_scores_follow_each_model_predict():
    X, y, features = synthetic(300, seed=10)
    for name in FACTORIES:
        pipe = build_pipeline(name, features, FEATURE_CFG, SEARCH_CFG, seed=0, n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)    # random data: nothing to converge to
            pipe.fit(X.iloc[:200], y[:200])
        pred, score = predictions_and_scores(pipe, X.iloc[200:])
        np.testing.assert_array_equal(pred, pipe.predict(X.iloc[200:]), err_msg=name)
        assert score.shape == (100,) and np.isfinite(score).all(), name
    svm = build_pipeline("svm", features, FEATURE_CFG, SEARCH_CFG, seed=0, n_jobs=1).fit(X.iloc[:200], y[:200])
    assert not hasattr(svm, "predict_proba")                       # no Platt scaling: the score is the margin
    np.testing.assert_array_equal(predictions_and_scores(svm, X.iloc[200:])[1], svm.decision_function(X.iloc[200:]))


def test_evaluate_fold_reports_the_training_mcc():
    X, y, features = synthetic(400, seed=11)
    train, val = np.arange(300), np.arange(300, 400)
    tree = build_pipeline("decision_tree", features, FEATURE_CFG, SEARCH_CFG, seed=0, n_jobs=1)
    metrics, timing = evaluate_fold(tree, X, y, train, val)
    assert metrics["mcc_train"] == 1.0 and abs(metrics["mcc"]) < 0.3  # random labels: memorised, not learned (P4)
    assert timing["fit_s"] >= 0 and timing["predict_s"] >= 0 and metrics["convergence_warnings"] == 0
    dummy = build_pipeline(BASELINE, features, FEATURE_CFG, SEARCH_CFG, seed=0, n_jobs=1)
    metrics, _ = evaluate_fold(dummy, X, y, train, val)
    assert metrics["mcc"] == 0.0 and metrics["mcc_train"] == 0.0 and metrics["roc_auc"] == 0.5


def per_fold(means: dict[str, list[float]]) -> pl.DataFrame:
    rows = []
    for name, values in means.items():
        for fold, v in enumerate(values):
            rows.append({"algorithm": name, "fold": fold, "convergence_warnings": 0,
                         **{m: v for m in ("mcc", "mcc_train", "f1_attack", "f1_benign", "balanced_accuracy",
                                           "sensitivity", "specificity", "fpr", "auprc", "roc_auc")}})
    return pl.DataFrame(rows)


def test_selection_reports_a_tie_at_the_top():
    folds = {"lightgbm": [0.99, 0.97], "random_forest": [0.99, 0.975], "decision_tree": [0.975, 0.97],
             "knn": [0.97, 0.975], "logreg": [0.80, 0.82], BASELINE: [0.0, 0.0]}
    out = select(summarize(per_fold(folds)), 3, BASELINE)
    assert out["ranking"][0]["algorithm"] == "random_forest" and out["baseline"]["mcc_mean"] == 0.0
    assert set(out["tie_group"]) == {"random_forest", "lightgbm", "decision_tree", "knn"}
    assert not out["discriminates"] and len(out["top_by_mean"]) == 3 and BASELINE not in out["tie_group"]


def test_selection_keeps_the_top_three_when_they_stand_out():
    folds = {"lightgbm": [0.99, 0.98], "random_forest": [0.98, 0.985], "mlp": [0.97, 0.99],
             "knn": [0.90, 0.91], BASELINE: [0.0, 0.0]}
    out = select(summarize(per_fold(folds)), 3, BASELINE)
    assert out["discriminates"] and out["top_by_mean"] == ["lightgbm", "random_forest", "mlp"]
