"""Etapa 10, task 1 — the final model. Run early, as the PLANO allows, because the leave-capture-out of Etapa 9 uses
its configuration.

The best algorithm of Etapa 8 is tuned on the whole of sample M with the search of Etapa 7 (same space, number of
configurations and seed, so the candidates are the same 30 as in every outer fold), scored by MCC over
`final.cv_splits` folds stratified by label34, and refitted on all of M. This model is NOT evaluated again: its
performance estimate is the one of the nested cross-validation (Etapa 8). Picking the model of the best outer fold
instead would be a methodological error (PLANO, Etapa 10, task 1).

Outputs: models/final_model.joblib (not versioned; its SHA-256 is recorded), results/final/final_model.json
(configuration, cross-validated MCC of the search, data hash, features), results/final/final_search.csv and
results/final/final_model.run.json.

Usage: python -m src.experiments.final_model   (make final)
"""
import json

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.experiments.nested_cv import load_sample_M, param_distributions, plain
from src.interpretation.common import best_model
from src.models.pipelines import auxiliary_columns, build_pipeline, feature_names, prepare
from src.utils.config import load_config, path
from src.utils.io import rel, sha256_file, write_json
from src.utils.run_metadata import recorded_run


def search_table(res: dict) -> pl.DataFrame:
    return pl.DataFrame({
        "candidate": np.arange(len(res["params"])),
        "params": [json.dumps({k.removeprefix("model__"): plain(v) for k, v in p.items()}, sort_keys=True)
                   for p in res["params"]],
        "validation_mcc_mean": res["mean_test_score"], "validation_mcc_std": res["std_test_score"],
        "training_mcc_mean": res["mean_train_score"], "rank": res["rank_test_score"]})


def main() -> None:
    cfg, feature_cfg, search_cfg = load_config(), load_config("features"), load_config("search_spaces")
    seed, model_name = int(cfg["seed"]), best_model(cfg)
    out_dir, model_file = path(cfg, "results_dir") / "final", path(cfg, "models_dir") / "final_model.joblib"
    out_dir.mkdir(parents=True, exist_ok=True)
    with recorded_run(cfg, out_dir, "final_model") as run:
        M = load_sample_M(cfg)
        features = feature_names(M.columns, feature_cfg, cfg["nested"]["feature_set"], auxiliary_columns(cfg))
        X, y = prepare(M, features, feature_cfg), M[cfg["columns"]["target"]].to_numpy()
        labels = M[cfg["columns"]["label"]].to_numpy()
        cv = list(StratifiedKFold(n_splits=int(cfg["final"]["cv_splits"]), shuffle=True, random_state=seed)
                  .split(np.zeros(labels.size), labels))
        search = RandomizedSearchCV(build_pipeline(model_name, features, feature_cfg, search_cfg, seed, n_jobs=1),
                                    param_distributions(search_cfg[model_name].get("params", {})),
                                    n_iter=int(cfg["search"]["n_iter"]), scoring=make_scorer(matthews_corrcoef),
                                    cv=cv, refit=True, return_train_score=True, random_state=seed,
                                    n_jobs=int(cfg["n_jobs"]), error_score="raise")
        search.fit(X, y)
        table = search_table(search.cv_results_)
        table.write_csv(out_dir / "final_search.csv")
        model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(search.best_estimator_, model_file)
        res = search.cv_results_
        summary = {
            "model": model_name, "candidate": int(search.best_index_),
            "best_params": {k.removeprefix("model__"): plain(v) for k, v in search.best_params_.items()},
            "cv_splits": int(cfg["final"]["cv_splits"]), "cv_mcc_mean": float(search.best_score_),
            "cv_mcc_std": float(res["std_test_score"][search.best_index_]),
            "cv_training_mcc": float(res["mean_train_score"][search.best_index_]),
            "n_rows": int(y.size), "features": features, "feature_set": cfg["nested"]["feature_set"],
            "sample_M_sha256": sha256_file(path(cfg, "sample_M")), "model_file": rel(model_file),
            "model_file_sha256": sha256_file(model_file),
            "note": "Not evaluated again: the performance estimate is the nested cross-validation of Etapa 8."}
        write_json(summary, out_dir / "final_model.json")
        run["outputs"] = {rel(out_dir / n): sha256_file(out_dir / n) for n in ("final_search.csv", "final_model.json")}
        run["model_file_sha256"] = summary["model_file_sha256"]
    print(json.dumps({k: summary[k] for k in ("model", "candidate", "best_params", "cv_mcc_mean", "cv_mcc_std")},
                     indent=1))


if __name__ == "__main__":
    main()
