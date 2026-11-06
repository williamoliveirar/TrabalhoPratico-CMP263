"""Etapa 4: the fixed pre-processing is row-wise and the fitted pre-processing only sees training rows."""
import warnings

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.models.pipelines import auxiliary_columns, build_pipeline, feature_names, prepare
from src.models.registry import BASELINE, FACTORIES, INDUCTIVE_BIAS, TRANSFORMED
from src.utils.config import load_config
from synthetic import COLUMNS, sample

CFG = load_config()
FEATURE_CFG = load_config("features")
SEARCH_CFG = load_config("search_spaces")
AUX = auxiliary_columns(CFG)


def fitted_standard(pipe, name: str):
    step = pipe.named_steps["preprocess"].named_transformers_[name]
    return step.named_steps["standard"] if name == "log1p_standard" else step


def test_feature_sets_follow_features_yaml():
    full = feature_names(COLUMNS, FEATURE_CFG, "full", AUX)
    dropped = {item["name"] for item in FEATURE_CFG["drop"]}
    assert len(full) == 27 and not dropped & set(full) and not set(AUX) & set(full)
    assert set(full) - set(feature_names(COLUMNS, FEATURE_CFG, "no_window", AUX)) == {"Number"}
    assert set(full) - set(feature_names(COLUMNS, FEATURE_CFG, "no_shortcut", AUX)) == {"Number", "Time_To_Live"}
    with pytest.raises(KeyError):
        feature_names(COLUMNS, FEATURE_CFG, "unknown", AUX)


def test_fixed_preprocessing_is_row_wise_and_fills_std():
    df = sample(200, seed=0)
    features = feature_names(df.columns, FEATURE_CFG, "full", AUX)
    X = prepare(df, features, FEATURE_CFG)
    assert list(X.columns) == features and (X.loc[df["Std"].is_null().to_numpy(), "Std"] == 0.0).all()
    # the output of each row does not depend on the other rows, so it can run before the split
    assert X.iloc[150:].reset_index(drop=True).equals(prepare(df[150:], features, FEATURE_CFG))


def test_fixed_preprocessing_fails_on_values_it_cannot_handle():
    df = sample(50, seed=1).with_columns(pl.when(pl.col("row_idx") == 3).then(np.inf).otherwise(pl.col("IAT"))
                                         .alias("IAT"))
    with pytest.raises(ValueError, match="IAT"):
        prepare(df, feature_names(df.columns, FEATURE_CFG, "full", AUX), FEATURE_CFG)


def test_fitted_parameters_depend_only_on_training_rows():
    df = sample(400, seed=2)
    features = feature_names(df.columns, FEATURE_CFG, "full", AUX)
    X, y = prepare(df, features, FEATURE_CFG), df["y"].to_numpy()
    train, test = np.arange(300), np.arange(300, 400)
    pipe = build_pipeline("logreg", features, FEATURE_CFG, SEARCH_CFG, seed=0, n_jobs=1)
    pipe.fit(X.iloc[train], y[train])
    logged, linear = FEATURE_CFG["log1p"], [c for c in features if c not in FEATURE_CFG["log1p"] + ["Protocol Type"]]
    np.testing.assert_allclose(fitted_standard(pipe, "log1p_standard").mean_, np.log1p(X.iloc[train][logged]).mean())
    np.testing.assert_allclose(fitted_standard(pipe, "standard").mean_, X.iloc[train][linear].mean())
    np.testing.assert_allclose(fitted_standard(pipe, "standard").scale_, X.iloc[train][linear].std(ddof=0))
    # changing the test rows changes nothing that was fitted
    X_shifted = X.copy()
    X_shifted.iloc[test] *= 1000.0
    again = build_pipeline("logreg", features, FEATURE_CFG, SEARCH_CFG, seed=0, n_jobs=1)
    again.fit(X_shifted.iloc[train], y[train])
    np.testing.assert_array_equal(fitted_standard(again, "standard").mean_, fitted_standard(pipe, "standard").mean_)
    np.testing.assert_array_equal(again.named_steps["model"].coef_, pipe.named_steps["model"].coef_)


def test_one_hot_categories_come_from_training_rows():
    df = sample(300, seed=3)
    features = feature_names(df.columns, FEATURE_CFG, "full", AUX)
    X, y = prepare(df, features, FEATURE_CFG), df["y"].to_numpy()
    seen = X["Protocol Type"].isin([6.0, 17.0]).to_numpy()
    pipe = build_pipeline("knn", features, FEATURE_CFG, SEARCH_CFG, seed=0, n_jobs=1).fit(X[seen], y[seen])
    encoder = pipe.named_steps["preprocess"].named_transformers_["onehot"]
    assert [list(c) for c in encoder.categories_] == [[6.0, 17.0]]
    unseen = pipe.named_steps["preprocess"].transform(X[~seen])  # a protocol code never seen in training
    assert unseen[:, -2:].sum() == 0.0


def test_cross_validation_refits_the_preprocessing_in_every_fold():
    df = sample(500, seed=4)
    features = feature_names(df.columns, FEATURE_CFG, "full", AUX)
    X, y = prepare(df, features, FEATURE_CFG), df["y"].to_numpy()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pipe = build_pipeline("logreg", features, FEATURE_CFG, SEARCH_CFG, seed=0, n_jobs=1)
    out = cross_validate(pipe, X, y, cv=folds, return_estimator=True, return_indices=True)
    linear = [c for c in features if c not in FEATURE_CFG["log1p"] + ["Protocol Type"]]
    for estimator, train in zip(out["estimator"], out["indices"]["train"]):
        np.testing.assert_allclose(fitted_standard(estimator, "standard").mean_, X.iloc[train][linear].mean())


def test_tree_pipelines_get_numeric_attributes_unchanged_and_one_hot_protocol():
    df = sample(100, seed=5)
    features = feature_names(df.columns, FEATURE_CFG, "no_shortcut", AUX)
    X, y = prepare(df, features, FEATURE_CFG), df["y"].to_numpy()
    pipe = build_pipeline("decision_tree", features, FEATURE_CFG, SEARCH_CFG, seed=0, n_jobs=1).fit(X, y)
    out = pipe.named_steps["preprocess"].transform(X)
    numeric = [c for c in features if c not in FEATURE_CFG["categorical"]]
    np.testing.assert_array_equal(out[:, :len(numeric)], X[numeric].to_numpy())   # no scaling, no log1p
    codes = sorted(X["Protocol Type"].unique())                                       # P6: one column per code
    np.testing.assert_array_equal(out[:, len(numeric):], (X[["Protocol Type"]].to_numpy() == codes).astype(float))


def test_every_algorithm_builds_fits_and_predicts():
    assert set(SEARCH_CFG) == set(FACTORIES) == set(INDUCTIVE_BIAS)
    assert TRANSFORMED <= set(FACTORIES) and BASELINE in FACTORIES
    df = sample(300, seed=6)
    features = feature_names(df.columns, FEATURE_CFG, "full", AUX)
    X, y = prepare(df, features, FEATURE_CFG), df["y"].to_numpy()
    for name in FACTORIES:
        pipe = build_pipeline(name, features, FEATURE_CFG, SEARCH_CFG, seed=0, n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)  # random data: nothing to converge to
            pred = pipe.fit(X.iloc[:200], y[:200]).predict(X.iloc[200:])
        assert pred.shape == (100,) and set(pred) <= {0, 1}, name
