"""Etapa 9: the model of each outer fold of the nested CV, refitted from Etapa 7 and checked before any use.

The interpretation explains the model trained in each outer fold on the test fold it never saw (PLANO, Etapa 9). Etapa 7
kept only the predictions, so each fold model is refitted on the outer training set with the configuration chosen in
the inner loop (`best_params` of results/nested/<model>/fold<k>.json), single-threaded as in Etapa 7, and accepted
only if it reproduces the recorded test predictions exactly (classes) and to 1e-9 (scores). Fitted models are cached
in models/etapa9/ (not versioned) and re-checked when loaded.
"""
from dataclasses import dataclass

import joblib
import numpy as np
import polars as pl

from src.evaluation.metrics import predictions_and_scores
from src.experiments.make_splits import NESTED, load_assignment, outer_folds
from src.models.pipelines import auxiliary_columns, build_pipeline, feature_names, prepare
from src.utils.config import path

SCORE_TOLERANCE = 1e-9  # recorded scores went through CSV; classes must match exactly


@dataclass
class FoldData:
    """Sample M prepared once: features, target, label of every row, and the outer folds."""
    M: pl.DataFrame
    X: object  # pandas DataFrame from prepare()
    y: np.ndarray
    labels: np.ndarray
    features: list[str]
    folds: list[tuple[np.ndarray, np.ndarray]]


def fold_data(cfg: dict, feature_cfg: dict, M: pl.DataFrame, feature_set: str) -> FoldData:
    features = feature_names(M.columns, feature_cfg, feature_set, auxiliary_columns(cfg))
    return FoldData(M, prepare(M, features, feature_cfg), M[cfg["columns"]["target"]].to_numpy(),
                    M[cfg["columns"]["label"]].to_numpy(), features, outer_folds(load_assignment(cfg, NESTED, M)))


def _check(model, data: FoldData, test: np.ndarray, recorded: pl.DataFrame) -> bool:
    pred, score = predictions_and_scores(model, data.X.iloc[test])
    return (np.array_equal(recorded["position"].to_numpy(), test) and np.array_equal(pred, recorded["pred"].to_numpy())
            and float(np.max(np.abs(score - recorded["score"].to_numpy()))) <= SCORE_TOLERANCE)


def fold_model(cfg: dict, feature_cfg: dict, search_cfg: dict, data: FoldData, model_name: str, k: int,
               results_root, cache_dir):
    """Fitted pipeline of outer fold k, verified against the predictions recorded under results_root/model_name."""
    train, test = data.folds[k]
    recorded = pl.read_csv(results_root / model_name / f"fold{k}_predictions.csv.gz")
    cache = cache_dir / f"{model_name}_fold{k}.joblib"
    if cache.exists():
        model = joblib.load(cache)
        if _check(model, data, test, recorded):
            return model
    params = __import__("json").loads((results_root / model_name / f"fold{k}.json").read_text("utf-8"))["best_params"]
    model = build_pipeline(model_name, data.features, feature_cfg, search_cfg, int(cfg["seed"]), n_jobs=1)
    model.set_params(**{f"model__{p}": tuple(v) if isinstance(v, list) else v for p, v in params.items()})
    model.fit(data.X.iloc[train], data.y[train])
    if not _check(model, data, test, recorded):
        raise RuntimeError(f"{model_name} fold {k}: the refitted model does not reproduce the Etapa 7 predictions")
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cache)
    return model


def models_dir(cfg: dict):
    return path(cfg, "models_dir") / "etapa9"
