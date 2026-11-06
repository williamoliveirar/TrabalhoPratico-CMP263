"""Etapa 4: pre-processing without leakage (PLANO §7, Etapa 4; D13 in configs/features.yaml).

Two kinds of pre-processing, as in "Quando transformar os dados?" (Aula 13-14):
- before the split (`prepare`): operations that remove attributes or fill a value by a fixed rule, with no
  parameter estimated from the data. The redundant and quasi-constant columns are dropped, `Std` null becomes 0
  (one-packet windows have no dispersion), and only the predictive attributes of the feature set are kept. The
  output of a row depends on that row alone, so applying it to the whole sample leaks nothing.
- inside the Pipeline (`build_pipeline`): operations whose parameters are estimated from data, fitted on the
  training rows of each fold only and applied unchanged to its validation or test rows. Every pipeline one-hot
  encodes the nominal attribute, with the categories seen in training (P6). The TRANSFORMED algorithms also get
  log1p, which has no parameter, followed by standardisation (means and deviations of the training rows); the
  other algorithms get the numeric attributes as they are.

Hyperparameters of the estimator are reached as `model__<name>` (Etapa 7).
"""
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.models.registry import TRANSFORMED, make_estimator


def auxiliary_columns(cfg: dict) -> list[str]:
    """Label, category, target and provenance columns: never predictive attributes."""
    cols = cfg["columns"]
    return [cols["label"], cols["category"], cols["target"], *cols["provenance"]]


def feature_names(columns: list[str], feature_cfg: dict, feature_set: str, auxiliary: list[str]) -> list[str]:
    """Predictive attributes of `feature_set`, in the order of `columns`."""
    sets = feature_cfg["feature_sets"]
    if feature_set not in sets:
        raise KeyError(f"unknown feature set {feature_set!r}; known: {sorted(sets)}")
    dropped = {item["name"] for item in feature_cfg["drop"]}
    spec = sets[feature_set]
    excluded = set() if spec == "all" else set(spec["exclude"])
    missing = (dropped | excluded) - set(columns)
    if missing:
        raise KeyError(f"features.yaml names columns absent from the data: {sorted(missing)}")
    return [c for c in columns if c not in set(auxiliary) | dropped | excluded]


def prepare(df: pl.DataFrame, features: list[str], feature_cfg: dict) -> pd.DataFrame:
    """Fixed pre-processing before the split: keep `features` and fill nulls by the rules of features.yaml."""
    fills = {c: v for c, v in feature_cfg["fill_null"].items() if c in features}
    out = df.select(features).with_columns(pl.col(c).fill_null(v) for c, v in fills.items())
    values = out.to_numpy().astype(float)
    if not np.isfinite(values).all():
        bad = [c for c, ok in zip(features, np.isfinite(values).all(axis=0)) if not ok]
        raise ValueError(f"null or infinite values left after the fixed pre-processing: {bad}")
    return pd.DataFrame(values, columns=features)


def preprocessor(model_name: str, features: list[str], feature_cfg: dict) -> ColumnTransformer:
    categorical = [c for c in features if c in feature_cfg["categorical"]]
    numeric = [c for c in features if c not in categorical]
    onehot = ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
    if model_name not in TRANSFORMED:
        return ColumnTransformer([("attributes", "passthrough", numeric), onehot])
    logged = [c for c in numeric if c in feature_cfg["log1p"]]
    linear = [c for c in numeric if c not in logged]
    return ColumnTransformer([
        ("log1p_standard", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                     ("standard", StandardScaler())]), logged),
        ("standard", StandardScaler(), linear),
        onehot,
    ])


def build_pipeline(model_name: str, features: list[str], feature_cfg: dict, search_cfg: dict, seed: int,
                   n_jobs: int) -> Pipeline:
    """Pre-processing fitted on the training rows only, followed by the default estimator of `model_name`."""
    return Pipeline([("preprocess", preprocessor(model_name, features, feature_cfg)),
                     ("model", make_estimator(model_name, seed, n_jobs, search_cfg))])
