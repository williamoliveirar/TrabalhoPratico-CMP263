"""Etapa 9: rule functions of the interpretation, SHAP additivity per original attribute, and the guard that a refitted
fold model must reproduce the recorded predictions before it is interpreted."""
import json

import numpy as np
import polars as pl
import pytest

from src.evaluation.metrics import predictions_and_scores
from src.interpretation.common import original_feature, ranking_stability, typical
from src.interpretation.fold_models import FoldData, fold_model
from src.interpretation.shap_analysis import explain
from src.models.pipelines import auxiliary_columns, build_pipeline, feature_names, prepare
from src.utils.config import load_config
from src.utils.io import write_csv_gz
from synthetic import sample

CFG, FEATURE_CFG, SEARCH_CFG = load_config(), load_config("features"), load_config("search_spaces")
SMALL = {"n_estimators": 20, "num_leaves": 7, "min_child_samples": 5}


def test_original_feature_merges_one_hot_columns():
    assert original_feature("attributes__IAT", ["Protocol Type"]) == "IAT"
    assert original_feature("onehot__Protocol Type_6.0", ["Protocol Type"]) == "Protocol Type"
    assert original_feature("Protocol Type", ["Protocol Type"]) == "Protocol Type"


def test_ranking_stability():
    feats = [f"f{i}" for i in range(12)]
    same = pl.DataFrame({"fold": np.repeat([0, 1], 12), "feature": feats * 2, "v": np.tile(np.arange(12.0), 2)})
    s = ranking_stability(same, "v", top_k=10)
    assert (s["spearman_min"], s["top_k_overlap_min"], s["pairs"]) == (pytest.approx(1.0), 1.0, 1)
    flipped = same.with_columns(pl.when(pl.col("fold") == 1).then(11 - pl.col("v")).otherwise(pl.col("v")).alias("v"))
    s = ranking_stability(flipped, "v", top_k=10)
    assert s["spearman_mean"] == pytest.approx(-1.0) and s["top_k_overlap_mean"] == pytest.approx(8 / 10)


def test_typical_takes_scores_closest_to_the_median():
    c = pl.DataFrame({"position": [9, 3, 5, 7, 1], "score": [0.1, 0.3, 0.3, 0.45, 0.2]})   # median 0.3
    assert typical(c, 1)["position"].to_list() == [3]                                    # tie: lowest position
    assert typical(c, 3)["position"].to_list() == [3, 5, 1]


def synthetic_data(n: int, seed: int):
    df = sample(n, seed)
    features = feature_names(df.columns, FEATURE_CFG, "full", auxiliary_columns(CFG))
    return df, features, prepare(df, features, FEATURE_CFG), df["y"].to_numpy()


def test_shap_adds_up_and_merges_protocol_type():
    df, features, X, y = synthetic_data(300, seed=3)
    model = build_pipeline("lightgbm", features, FEATURE_CFG, SEARCH_CFG, seed=0, n_jobs=1)
    model.set_params(**{f"model__{k}": v for k, v in SMALL.items()}).fit(X, y)
    sv, base, attributes = explain(model, X.iloc[:40], FEATURE_CFG["categorical"])
    assert sorted(attributes) == sorted(features) and sv.shape == (40, len(features))  # one value per attribute
    raw = model[-1].predict(model[:-1].transform(X.iloc[:40]), raw_score=True)
    np.testing.assert_allclose(base + sv.sum(axis=1), raw, atol=1e-6)


def test_fold_model_refuses_a_model_that_does_not_reproduce_the_predictions(tmp_path):
    df, features, X, y = synthetic_data(240, seed=4)
    train, test = np.arange(180), np.arange(180, 240)
    data = FoldData(df, X, y, df["label34"].to_numpy(), features, [(train, test)])
    reference = build_pipeline("lightgbm", features, FEATURE_CFG, SEARCH_CFG, int(CFG["seed"]), n_jobs=1)
    reference.set_params(**{f"model__{k}": v for k, v in SMALL.items()}).fit(X.iloc[train], y[train])
    pred, score = predictions_and_scores(reference, X.iloc[test])
    (tmp_path / "lightgbm").mkdir()
    (tmp_path / "lightgbm" / "fold0.json").write_text(json.dumps({"best_params": SMALL}), "utf-8")
    recorded = pl.DataFrame({"position": test, "y": y[test], "pred": pred, "score": score})
    write_csv_gz(recorded, tmp_path / "lightgbm" / "fold0_predictions.csv.gz")
    model = fold_model(CFG, FEATURE_CFG, SEARCH_CFG, data, "lightgbm", 0, tmp_path, tmp_path / "cache")
    assert (tmp_path / "cache" / "lightgbm_fold0.joblib").exists()
    np.testing.assert_array_equal(predictions_and_scores(model, X.iloc[test])[0], pred)
    write_csv_gz(recorded.with_columns((1 - pl.col("pred")).alias("pred")),
                 tmp_path / "lightgbm" / "fold0_predictions.csv.gz")
    with pytest.raises(RuntimeError, match="does not reproduce"):
        fold_model(CFG, FEATURE_CFG, SEARCH_CFG, data, "lightgbm", 0, tmp_path, tmp_path / "cache")
