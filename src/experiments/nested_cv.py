"""Etapa 7: hyperparameter tuning and performance estimation with nested cross-validation on sample M.

For every model of `search.finalist_models` and `search.reference_models` and every outer fold k of
data/splits/nested_M.csv.gz (Etapa 5):
- inner loop (training × validation): RandomizedSearchCV draws `search.n_iter` hyperparameter configurations from
  configs/search_spaces.yaml and scores each by its mean MCC over the 3 inner validation folds drawn on the outer
  training set of fold k (`inner_folds(assign, k)`). The best configuration is refitted on the whole outer training
  set (refit=True). return_train_score=True records the training MCC of every configuration (P4). The search uses
  the same random_state in every outer fold, so the 30 candidate configurations are the same in all folds and only
  the choice among them depends on the data.
- outer loop (training × test): only after the search is the test set of fold k used, once, to predict and compute
  the metrics of src/evaluation/metrics.py. No decision depends on it.
The dummy baseline has no hyperparameters: it is fitted on the outer training set and evaluated the same way.

Inside the search, each estimator runs single-threaded and the search runs `n_jobs` fits in parallel, so the
results do not depend on how the work is spread over the cores.

Outputs in results/nested/<model>/, written as soon as each fold finishes. An interrupted run resumes where it
stopped: a fold whose fold<k>.json exists is skipped.
- fold<k>.json: best configuration, its inner validation and training MCC, test metrics and the training MCC of the
  refitted model on the outer training set (P4);
- fold<k>_search.csv: every configuration tried, with mean and std of the validation MCC and mean training MCC;
- fold<k>_predictions.csv.gz: position in M, true class, predicted class and score of every test instance
  (the out-of-fold predictions used in Etapa 8);
- fold<k>_timing.json: wall-clock times, the only output that varies between runs (and includes any time the
  machine spent suspended).
Once every model has every outer fold, results/nested/folds.csv gathers one row per model and fold. Each
invocation also writes its own run record (nested_cv[_<scope>].run.json).

Usage: python -m src.experiments.nested_cv [--models lightgbm ...] [--folds 0 1 ...]
"""
import argparse
import json
import time

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV

from src.evaluation.metrics import binary_metrics, predictions_and_scores
from src.experiments.make_splits import NESTED, inner_folds, load_assignment, outer_folds
from src.models.pipelines import auxiliary_columns, build_pipeline, feature_names, prepare
from src.utils.config import load_config, path
from src.utils.io import rel, sha256_file, write_csv_gz, write_json
from src.utils.run_metadata import recorded_run


def param_distributions(spec: dict) -> dict:
    """The `params` of a model in configs/search_spaces.yaml as RandomizedSearchCV distributions (model__ keys)."""
    out = {}
    for name, d in spec.items():
        for bound in ("low", "high"):
            if bound in d and not isinstance(d[bound], (int, float)):
                raise TypeError(f"{name}.{bound} = {d[bound]!r} is not a number; YAML reads 1.0e2 as a string, "
                                "write 1.0e+2 or 100.0")
        if d["dist"] == "loguniform":
            dist = stats.loguniform(d["low"], d["high"])
        elif d["dist"] == "randint":
            dist = stats.randint(d["low"], d["high"] + 1)  # inclusive upper bound, as written in the YAML
        elif d["dist"] == "uniform":
            dist = stats.uniform(d["low"], d["high"] - d["low"])
        elif d["dist"] == "choice":
            dist = [tuple(v) if isinstance(v, list) else v for v in d["values"]]  # e.g. hidden_layer_sizes
        else:
            raise ValueError(f"unknown distribution {d['dist']!r} for {name}")
        out[f"model__{name}"] = dist
    return out


def plain(value):
    """JSON-friendly copy of a hyperparameter value (numpy scalars, tuples)."""
    if isinstance(value, tuple):
        return [plain(v) for v in value]
    return value.item() if hasattr(value, "item") else value


def tune_and_test(pipe, distributions: dict, X, y: np.ndarray, train: np.ndarray, test: np.ndarray,
                  inner: list, n_iter: int, seed: int, n_jobs: int) -> dict:
    """One outer fold: search on the training set (inner loop), then a single evaluation on the test set."""
    start = time.perf_counter()
    search = None
    if distributions:
        search = RandomizedSearchCV(pipe, distributions, n_iter=n_iter, scoring=make_scorer(matthews_corrcoef),
                                    cv=inner, refit=True, return_train_score=True, random_state=seed,
                                    n_jobs=n_jobs, error_score="raise")
        search.fit(X.iloc[train], y[train])
        model = search.best_estimator_
    else:
        model = pipe.fit(X.iloc[train], y[train])
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    pred, score = predictions_and_scores(model, X.iloc[test])  # the only use of the outer test set
    predict_s = time.perf_counter() - start
    train_pred, _ = predictions_and_scores(model, X.iloc[train])  # training MCC of the refitted model (P4)
    out = {"model": model, "pred": pred, "score": score, "test": binary_metrics(y[test], pred, score),
           "outer_training_mcc": matthews_corrcoef(y[train], train_pred),
           "timing": {"search_and_refit_s": round(fit_s, 1), "predict_test_s": round(predict_s, 2)}}
    if search is not None:
        res = search.cv_results_
        out |= {"best_params": {k.removeprefix("model__"): plain(v) for k, v in search.best_params_.items()},
                "inner_validation_mcc": float(search.best_score_),
                "inner_validation_mcc_std": float(res["std_test_score"][search.best_index_]),
                "inner_training_mcc": float(res["mean_train_score"][search.best_index_]),
                "search": pl.DataFrame({
                    "candidate": np.arange(len(res["params"])),
                    "params": [json.dumps({k.removeprefix("model__"): plain(v) for k, v in p.items()},
                                          sort_keys=True) for p in res["params"]],
                    "validation_mcc_mean": res["mean_test_score"], "validation_mcc_std": res["std_test_score"],
                    "training_mcc_mean": res["mean_train_score"], "rank": res["rank_test_score"]})}
    return out


def run_fold(name: str, k: int, X, y: np.ndarray, assign: pl.DataFrame, cfg: dict, feature_cfg: dict,
             search_cfg: dict, features: list[str], out_dir) -> dict:
    """Tune and test `name` on outer fold k and write its files; returns the timing."""
    train, test = outer_folds(assign)[k]
    pipe = build_pipeline(name, features, feature_cfg, search_cfg, int(cfg["seed"]), n_jobs=1)
    result = tune_and_test(pipe, param_distributions(search_cfg[name].get("params", {})), X, y, train, test,
                           inner_folds(assign, k), int(cfg["search"]["n_iter"]), int(cfg["seed"]),
                           int(cfg["n_jobs"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    if "search" in result:
        result["search"].write_csv(out_dir / f"fold{k}_search.csv")
    write_csv_gz(pl.DataFrame({"position": test, "y": y[test], "pred": result["pred"], "score": result["score"]}),
                 out_dir / f"fold{k}_predictions.csv.gz")
    write_json(result["timing"], out_dir / f"fold{k}_timing.json")
    summary = {"model": name, "outer_fold": k, "n_train": int(train.size), "n_test": int(test.size)}
    summary |= {key: result[key] for key in ("best_params", "inner_validation_mcc", "inner_validation_mcc_std",
                                             "inner_training_mcc") if key in result}
    summary |= {"outer_training_mcc": result["outer_training_mcc"], "test": result["test"]}
    write_json(summary, out_dir / f"fold{k}.json")  # written last: its presence marks the fold as complete
    return result["timing"]


def collect(root, models: list[str], n_outer: int) -> pl.DataFrame | None:
    """One row per model and outer fold (inner-loop and outer-loop results), or None while a fold is missing."""
    rows = []
    for name in models:
        for k in range(n_outer):
            f = root / name / f"fold{k}.json"
            if not f.exists():
                return None
            d = json.loads(f.read_text("utf-8"))
            rows.append({"model": name, "outer_fold": k, "n_train": d["n_train"], "n_test": d["n_test"],
                         "inner_validation_mcc": d.get("inner_validation_mcc"),
                         "inner_validation_mcc_std": d.get("inner_validation_mcc_std"),
                         "inner_training_mcc": d.get("inner_training_mcc"),
                         "outer_training_mcc": d["outer_training_mcc"],
                         **{f"test_{m}": v for m, v in d["test"].items()},
                         "best_params": json.dumps(d["best_params"], sort_keys=True) if "best_params" in d else None})
    return pl.DataFrame(rows, infer_schema_length=None)


def run_nested(cfg: dict, feature_cfg: dict, search_cfg: dict, M: pl.DataFrame, feature_set: str, root,
               models: list[str], run_models: list[str], run_folds: list[int], run_name: str) -> None:
    """Nested CV of `run_models` × `run_folds` on sample M with `feature_set`, written under `root`; folds already
    done are skipped. Once every model of `models` has every outer fold, root/folds.csv gathers them."""
    n_outer = int(cfg["validation"]["outer"]["n_splits"])
    features = feature_names(M.columns, feature_cfg, feature_set, auxiliary_columns(cfg))
    X, y = prepare(M, features, feature_cfg), M[cfg["columns"]["target"]].to_numpy()
    assign = load_assignment(cfg, NESTED, M)
    root.mkdir(parents=True, exist_ok=True)
    with recorded_run(cfg, root, run_name) as run:
        run["feature_set"], run["features"] = feature_set, features
        done, outputs = [], {}
        for k in run_folds:
            for name in run_models:
                out_dir = root / name
                if (out_dir / f"fold{k}.json").exists():
                    print(f"{name} fold {k}: already done, skipped", flush=True)
                    continue
                timing = run_fold(name, k, X, y, assign, cfg, feature_cfg, search_cfg, features, out_dir)
                score = json.loads((out_dir / f"fold{k}.json").read_text("utf-8"))["test"]["mcc"]
                print(f"{feature_set} {name} fold {k}: test MCC {score:.4f} ({timing['search_and_refit_s']:.0f} s)",
                      flush=True)
                done.append(f"{name}/fold{k}")
                outputs |= {rel(f): sha256_file(f) for f in sorted(out_dir.glob(f"fold{k}*"))
                            if not f.name.endswith("_timing.json")}
        table = collect(root, models, n_outer)
        if table is not None:  # every model has every outer fold: consolidated table for Etapas 8 and 9
            table.write_csv(root / "folds.csv")
            outputs[rel(root / "folds.csv")] = sha256_file(root / "folds.csv")
            print(f"wrote {rel(root / 'folds.csv')} ({table.height} rows)")
        run["computed"] = done
        run["outputs"] = outputs


def load_sample_M(cfg: dict) -> pl.DataFrame:
    """Sample M in its stored row order, the order the fold assignments refer to; exits if it does not exist yet."""
    source = path(cfg, "sample_M")
    if not source.exists():
        raise SystemExit(f"{source} not found; run `make sample` first")
    return pl.read_parquet(source)


def main() -> None:
    """Nested CV of the requested models and outer folds on sample M, with the `nested.feature_set` features."""
    cfg, feature_cfg, search_cfg = load_config(), load_config("features"), load_config("search_spaces")
    models = [*cfg["search"]["finalist_models"], *cfg["search"]["reference_models"]]
    n_outer = int(cfg["validation"]["outer"]["n_splits"])
    parser = argparse.ArgumentParser(description="Etapa 7: nested cross-validation on sample M")
    parser.add_argument("--models", nargs="+", default=models, choices=models)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(n_outer)), choices=range(n_outer))
    args = parser.parse_args()
    # one run record per invocation scope, so resuming in several invocations keeps the provenance of each part
    scope = ([] if args.models == models else args.models) + \
            ([] if args.folds == list(range(n_outer)) else [f"fold{k}" for k in args.folds])
    run_nested(cfg, feature_cfg, search_cfg, load_sample_M(cfg), cfg["nested"]["feature_set"],
               path(cfg, "results_dir") / "nested", models, args.models, args.folds, "_".join(["nested_cv", *scope]))


if __name__ == "__main__":
    main()
