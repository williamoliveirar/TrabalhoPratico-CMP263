"""Etapa 7: the search spaces are valid, and the outer test set is used only once, after the inner-loop search."""
import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.experiments.nested_cv import param_distributions, plain, tune_and_test
from src.models.pipelines import auxiliary_columns, build_pipeline, feature_names, prepare
from src.models.registry import BASELINE, FACTORIES
from src.utils.config import load_config
from synthetic import sample

CFG, FEATURE_CFG, SEARCH_CFG = load_config(), load_config("features"), load_config("search_spaces")
FITTED_ROWS: list[np.ndarray] = []  # row labels seen by every fit of the spy below


class FitSpy(BaseEstimator, TransformerMixin):
    """Pass-through step that records which rows (DataFrame index labels) each fit receives."""

    def fit(self, X, y=None):
        FITTED_ROWS.append(np.asarray(X.index))
        return self

    def transform(self, X):
        return X


def synthetic(n: int, seed: int):
    df = sample(n, seed)
    features = feature_names(df.columns, FEATURE_CFG, "full", auxiliary_columns(CFG))
    return prepare(df, features, FEATURE_CFG), df["y"].to_numpy(), features


def test_every_model_of_etapa7_has_a_valid_search_space():
    models = [*CFG["search"]["finalist_models"], *CFG["search"]["reference_models"]]
    assert set(models) <= set(FACTORIES) and BASELINE in models and "decision_tree" in models
    _, _, features = synthetic(50, seed=0)
    for name, spec in SEARCH_CFG.items():
        distributions = param_distributions(spec.get("params", {}))
        if not distributions:                                           # the dummy baseline is never tuned
            continue
        pipe = build_pipeline(name, features, FEATURE_CFG, SEARCH_CFG, seed=0, n_jobs=1)
        for params in ParameterSampler(distributions, n_iter=40, random_state=0):
            pipe.set_params(**params)                                   # every key reaches the estimator
            for key, value in params.items():
                d = spec["params"][key.removeprefix("model__")]
                if d["dist"] in ("loguniform", "randint", "uniform"):
                    assert d["low"] <= value <= d["high"], (name, key, value)
                else:
                    assert plain(value) in d["values"], (name, key, value)
    assert param_distributions({}) == {}


def test_outer_test_rows_are_used_only_after_the_search():
    X, y, features = synthetic(240, seed=1)
    train, test = np.arange(180), np.arange(180, 240)
    inner = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(np.zeros((180, 1)), y[train]))
    pipe = build_pipeline("decision_tree", features, FEATURE_CFG, SEARCH_CFG, seed=0, n_jobs=1)
    spied = Pipeline([("spy", FitSpy()), *pipe.steps])
    FITTED_ROWS.clear()
    out = tune_and_test(spied, param_distributions(SEARCH_CFG["decision_tree"]["params"]), X, y, train, test,
                        inner, n_iter=4, seed=0, n_jobs=1)
    assert len(FITTED_ROWS) == 4 * 3 + 1                                   # 4 configurations × 3 inner folds + refit
    for rows, (tr, _) in zip(FITTED_ROWS, [fold for _ in range(4) for fold in inner]):
        np.testing.assert_array_equal(np.sort(rows), np.sort(train[tr]))  # inner training sets only
    np.testing.assert_array_equal(np.sort(FITTED_ROWS[-1]), train)        # refit on the whole outer training set
    assert not any(np.isin(rows, test).any() for rows in FITTED_ROWS)
    assert out["pred"].shape == (60,) and set(out["best_params"]) == set(SEARCH_CFG["decision_tree"]["params"])
    assert out["search"].height == 4 and -1 <= out["inner_validation_mcc"] <= 1
    assert out["test"]["tp"] + out["test"]["fn"] + out["test"]["tn"] + out["test"]["fp"] == 60


def test_the_baseline_is_fitted_without_a_search():
    X, y, features = synthetic(200, seed=2)
    train, test = np.arange(150), np.arange(150, 200)
    pipe = build_pipeline(BASELINE, features, FEATURE_CFG, SEARCH_CFG, seed=0, n_jobs=1)
    out = tune_and_test(pipe, param_distributions(SEARCH_CFG[BASELINE].get("params", {})), X, y, train, test,
                        inner=[], n_iter=30, seed=0, n_jobs=1)
    assert "search" not in out and out["test"]["mcc"] == 0.0 and out["test"]["roc_auc"] == 0.5


@pytest.mark.parametrize("spec, expected", [
    ({"dist": "randint", "low": 3, "high": 3}, {3}),
    ({"dist": "choice", "values": [[32], [64, 32]]}, {(32,), (64, 32)}),
])
def test_param_distributions_bounds_and_tuples(spec, expected):
    dist = param_distributions({"x": spec})["model__x"]
    values = {p["model__x"] for p in ParameterSampler({"model__x": dist}, n_iter=2, random_state=0)}
    assert values <= expected
