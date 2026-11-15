"""Etapa 6: spot-checking on sample E, every algorithm of D7 with default hyperparameters on the same folds.

Spot-checking is a quick comparison of algorithms with different inductive biases, not tuning (assignment). Each
algorithm is the pipeline of Etapa 4 (`build_pipeline`, feature set `spotcheck.feature_set`) with its default
hyperparameters plus the `fixed` entries of configs/search_spaces.yaml. It is trained on the training set and
evaluated on the validation set of each of the 10 folds of data/splits/spotcheck_E.csv.gz (Etapa 5). These folds
play the role of validation sets: they choose the finalists (model selection), so no performance estimate comes
from E.

Per algorithm and fold: the metrics of src/evaluation/metrics.py on the validation set, the MCC on the training
set (P4: training far above validation means overfitting, both low means underfitting), the number of convergence
warnings, and the training and prediction times.

Selection (PLANO §7, Etapa 6, task 4): the ranking by mean validation MCC and the tie group, the algorithms whose
mean is within one standard deviation (of the best algorithm) of the best mean. If the tie group is larger than
`search.finalists`, the spot-checking did not discriminate among its members, and the finalists are chosen among
them by diversity of inductive bias (INDUCTIVE_BIAS). The choice is a decision recorded in
configs/base.yaml; this script only reports.

Outputs (results/spotcheck/): metrics_per_fold.csv, summary.csv, selection.json and fig_spotcheck_mcc.png, which are
byte-identical across runs; timings_per_fold.csv, with wall-clock times that vary between runs; spotcheck.run.json.

Usage: python -m src.experiments.spotcheck
"""
import time
import warnings

import numpy as np
import polars as pl
from matplotlib.transforms import ScaledTranslation
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import matthews_corrcoef

from src.evaluation.metrics import binary_metrics, predictions_and_scores
from src.experiments.make_splits import SPOTCHECK, load_assignment, spotcheck_folds
from src.models.pipelines import auxiliary_columns, build_pipeline, feature_names, prepare
from src.models.registry import BASELINE, FACTORIES, INDUCTIVE_BIAS
from src.reporting.style import INK, INK_MUTED, INK_SECONDARY, SERIES, apply_style, plt, save
from src.utils.config import load_config, path
from src.utils.io import rel, sha256_file, write_json
from src.utils.run_metadata import recorded_run

SUMMARY_METRICS = ["mcc", "mcc_train", "f1_attack", "f1_benign", "balanced_accuracy", "sensitivity", "specificity",
                   "fpr", "auprc", "roc_auc"]
VALIDATION_COLOR = SERIES[6]  # slots 0 and 1 are reserved for benign and attack (src/reporting/style.py)


def evaluate_fold(pipe, X, y: np.ndarray, train: np.ndarray, val: np.ndarray) -> tuple[dict, dict]:
    """Fit on the training set, evaluate on the validation set; MCC on the training set too (P4)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        start = time.perf_counter()
        pipe.fit(X.iloc[train], y[train])
        fit_s = time.perf_counter() - start
    start = time.perf_counter()
    pred, score = predictions_and_scores(pipe, X.iloc[val])
    predict_s = time.perf_counter() - start
    train_pred, _ = predictions_and_scores(pipe, X.iloc[train])
    n_convergence = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    # only convergence warnings are counted and silenced; any other warning is re-emitted
    for w in caught:
        if not issubclass(w.category, ConvergenceWarning):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    metrics = (binary_metrics(y[val], pred, score)
               | {"mcc_train": matthews_corrcoef(y[train], train_pred), "convergence_warnings": n_convergence})
    return metrics, {"fit_s": round(fit_s, 3), "predict_s": round(predict_s, 3)}


def summarize(per_fold: pl.DataFrame) -> pl.DataFrame:
    """Mean and standard deviation over the folds, best mean validation MCC first."""
    stats = [expr for c in SUMMARY_METRICS for expr in (pl.col(c).mean().alias(f"{c}_mean"),
                                                         pl.col(c).std().alias(f"{c}_std"))]
    return (per_fold.group_by("algorithm")
            .agg(pl.len().alias("n_folds"), *stats, pl.col("convergence_warnings").sum())
            .with_columns(pl.col("algorithm").replace_strict(INDUCTIVE_BIAS).alias("inductive_bias"),
                          (pl.col("mcc_train_mean") - pl.col("mcc_mean")).alias("mcc_train_minus_validation"))
            .sort(["mcc_mean", "algorithm"], descending=[True, False])
            .select("algorithm", "inductive_bias", "n_folds", "mcc_mean", "mcc_std", "mcc_train_mean",
                    "mcc_train_minus_validation", pl.exclude("algorithm", "inductive_bias", "n_folds", "mcc_mean",
                                                             "mcc_std", "mcc_train_mean",
                                                             "mcc_train_minus_validation")))


def select(summary: pl.DataFrame, n_finalists: int, baseline: str) -> dict:
    """Ranking, the top `n_finalists` by mean MCC and the tie group (within one std of the best mean)."""
    ranked = summary.filter(pl.col("algorithm") != baseline)
    best = ranked.row(0, named=True)
    floor = best["mcc_mean"] - best["mcc_std"]
    tie = ranked.filter(pl.col("mcc_mean") >= floor)["algorithm"].to_list()
    return {
        "rule": f"top {n_finalists} by mean validation MCC; if more than {n_finalists} algorithms are within one "
                "standard deviation of the best mean (tie group), the spot-checking does not discriminate among "
                "them and the finalists are chosen among them by diversity of inductive bias",
        "ranking": [{"algorithm": r["algorithm"], "inductive_bias": r["inductive_bias"],
                     "mcc_mean": r["mcc_mean"], "mcc_std": r["mcc_std"]} for r in ranked.iter_rows(named=True)],
        "top_by_mean": ranked["algorithm"][:n_finalists].to_list(),
        "tie_floor": floor,
        "tie_group": tie,
        "discriminates": len(tie) <= n_finalists,
        "baseline": {"algorithm": baseline,
                     "mcc_mean": summary.filter(pl.col("algorithm") == baseline)["mcc_mean"].item()},
    }


def figure(per_fold: pl.DataFrame, summary: pl.DataFrame, baseline: str, target) -> None:
    """Validation MCC of every fold (dots), its mean (bar) and the mean training MCC (hollow diamond)."""
    apply_style()
    order = summary.filter(pl.col("algorithm") != baseline)["algorithm"].to_list()[::-1]  # best at the top
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(order) + 1.1))
    for i, name in enumerate(order):
        values = per_fold.filter(pl.col("algorithm") == name)["mcc"].to_numpy()
        row = summary.filter(pl.col("algorithm") == name).row(0, named=True)
        offsets = np.linspace(-0.18, 0.18, values.size)
        ax.scatter(values, i + offsets, s=14, color=VALIDATION_COLOR, alpha=0.75, linewidths=0, zorder=3,
                   label="validation MCC, one dot per fold" if i == 0 else None)
        ax.plot([row["mcc_mean"]] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=2, zorder=4,
                label="mean validation MCC" if i == 0 else None)
        ax.scatter(row["mcc_train_mean"], i, s=34, marker="D", facecolors="none", edgecolors=INK_MUTED,
                   linewidths=1.2, zorder=4, label="mean training MCC (P4)" if i == 0 else None)
        ax.annotate(f"{row['mcc_mean']:.3f}", (1.0, i), xycoords=("axes fraction", "data"), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(range(len(order)), order)
    ax.grid(axis="y", visible=False)
    base = summary.filter(pl.col("algorithm") == baseline)["mcc_mean"].item()
    ax.set_xlabel(f"MCC (dummy baseline: mean {base:.3f})")
    ax.set_title("Spot-checking on sample E: validation MCC, 5 folds × 2 repeats", loc="left")
    below_xlabel = ax.transAxes + ScaledTranslation(0, -0.42, fig.dpi_scale_trans)  # 0.42 inch under the axes
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0), bbox_transform=below_xlabel, ncol=3, fontsize=7.5,
              handletextpad=0.3, columnspacing=1.2, borderaxespad=0)
    save(fig, target)


def main() -> None:
    """Run every algorithm on every spot-checking fold of E, summarise, report the selection and draw the figure."""
    cfg, feature_cfg, search_cfg = load_config(), load_config("features"), load_config("search_spaces")
    seed, n_jobs, n_splits = int(cfg["seed"]), int(cfg["n_jobs"]), int(cfg["validation"]["spotcheck"]["n_splits"])
    source = path(cfg, "sample_E")
    if not source.exists():
        raise SystemExit(f"{source} not found; run `make sample` first")
    E = pl.read_parquet(source)
    features = feature_names(E.columns, feature_cfg, cfg["spotcheck"]["feature_set"], auxiliary_columns(cfg))
    X, y = prepare(E, features, feature_cfg), E[cfg["columns"]["target"]].to_numpy()
    folds = spotcheck_folds(load_assignment(cfg, SPOTCHECK, E))
    out_dir = path(cfg, "results_dir") / "spotcheck"

    with recorded_run(cfg, out_dir, "spotcheck") as run:
        rows, times = [], []
        for name in FACTORIES:
            start = time.perf_counter()
            for i, (train, val) in enumerate(folds):
                pipe = build_pipeline(name, features, feature_cfg, search_cfg, seed, n_jobs)
                metrics, timing = evaluate_fold(pipe, X, y, train, val)
                key = {"algorithm": name, "repeat": i // n_splits, "fold": i % n_splits}
                rows.append(key | metrics)
                times.append(key | timing)
            print(f"{name}: mean MCC {np.mean([r['mcc'] for r in rows[-len(folds):]]):.4f} "
                  f"({time.perf_counter() - start:.0f} s)", flush=True)
        per_fold, timings = pl.DataFrame(rows), pl.DataFrame(times)
        summary = summarize(per_fold)
        selection = select(summary, int(cfg["search"]["finalists"]), BASELINE)

        out_dir.mkdir(parents=True, exist_ok=True)
        per_fold.write_csv(out_dir / "metrics_per_fold.csv")
        summary.write_csv(out_dir / "summary.csv")
        timings.write_csv(out_dir / "timings_per_fold.csv")
        write_json({"source": {rel(source): sha256_file(source),
                               rel(path(cfg, "splits_dir") / f"{SPOTCHECK}.csv.gz"):
                                   sha256_file(path(cfg, "splits_dir") / f"{SPOTCHECK}.csv.gz")},
                    "feature_set": cfg["spotcheck"]["feature_set"], "n_features": len(features),
                    **selection}, out_dir / "selection.json")
        figure(per_fold, summary, BASELINE, out_dir / "fig_spotcheck_mcc.png")
        run["outputs"] = {rel(out_dir / f): sha256_file(out_dir / f)
                          for f in ("metrics_per_fold.csv", "summary.csv", "timings_per_fold.csv",
                                    "selection.json", "fig_spotcheck_mcc.png")}
        run["time_per_algorithm_s"] = dict(timings.group_by("algorithm", maintain_order=True)
                                           .agg((pl.col("fit_s") + pl.col("predict_s")).sum().round(1)).iter_rows())
    print(f"tie group: {selection['tie_group']}; top by mean: {selection['top_by_mean']}")


if __name__ == "__main__":
    main()
