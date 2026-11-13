"""Shared pieces of the Etapa 9 interpretation: the best model, the fold data and small rule functions."""
import json

import numpy as np
import polars as pl
from scipy import stats

from src.utils.config import path


def best_model(cfg: dict) -> str:
    """Best model of Etapa 8 (pre-registered rule: highest mean outer-fold test MCC)."""
    source = path(cfg, "results_dir") / "evaluation" / "hypotheses.json"
    if not source.exists():
        raise SystemExit(f"{source} not found; run `make evaluate` first")
    return json.loads(source.read_text("utf-8"))["best_model"]


def original_feature(transformed: str, categorical: list[str]) -> str:
    """Original attribute of a column of the fitted ColumnTransformer ('attributes__IAT' -> 'IAT',
    'onehot__Protocol Type_6.0' -> 'Protocol Type')."""
    name = transformed.split("__", 1)[-1]
    for c in categorical:
        if name == c or name.startswith(f"{c}_"):
            return c
    return name


def ranking_stability(per_fold: pl.DataFrame, value: str, top_k: int) -> dict:
    """Spearman correlation of `value` across features between every pair of folds, and the overlap of the top-k
    features (ties broken by feature name)."""
    wide = per_fold.pivot(on="fold", index="feature", values=value).sort("feature")
    cols = [c for c in wide.columns if c != "feature"]
    rho, overlap = [], []
    tops = {c: set(wide.sort([c, "feature"], descending=[True, False])["feature"].head(top_k)) for c in cols}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rho.append(float(stats.spearmanr(wide[a], wide[b]).statistic))
            overlap.append(len(tops[a] & tops[b]) / top_k)
    return {"pairs": len(rho), "spearman_mean": float(np.mean(rho)), "spearman_min": float(np.min(rho)),
            "top_k": top_k, "top_k_overlap_mean": float(np.mean(overlap)), "top_k_overlap_min": float(np.min(overlap))}


def typical(candidates: pl.DataFrame, n: int) -> pl.DataFrame:
    """The n rows whose score is closest to the median score of `candidates` (ties: lowest position)."""
    median = candidates["score"].median()
    return (candidates.with_columns((pl.col("score") - median).abs().alias("distance"))
            .sort("distance", "position").head(n).drop("distance"))


def summarise_by_feature(per_fold: pl.DataFrame, value: str) -> pl.DataFrame:
    """Mean and std over folds of `value` per feature, ranked (1 = most important; ties by name)."""
    return (per_fold.group_by("feature").agg(pl.col(value).mean().alias("mean"), pl.col(value).std().alias("std"))
            .sort(["mean", "feature"], descending=[True, False]).with_row_index("rank", offset=1))
