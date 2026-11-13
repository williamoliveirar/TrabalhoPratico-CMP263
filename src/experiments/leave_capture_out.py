"""Etapa 9, task 6 (D8 (a)): does the performance depend on seeing, in training, windows of the capture under test?

The best algorithm, with the configuration of the final model (results/final/final_model.json, Etapa 10 task 1), is
trained in the 5 outer folds of the nested CV (`nested_M`, split by row) and in the 5 capture folds (`capture_M`:
for the 16 attack types with at least 5 captures, every capture is whole in one fold; the other labels keep their
outer fold; Etapa 5). The per-type sensitivity on the out-of-fold predictions of the two designs is compared for the
grouped types with the pre-registered collapse rule (`ablation.type_collapse` with non-overlapping Wilson intervals).
The configuration was chosen with the whole of M, which includes the test folds of both designs: this affects both
sides alike, and what is read is the difference between them, not the level.

Outputs in results/capture/: predictions.csv.gz, folds.csv, recall_by_type.csv, summary.json,
fig_capture_recall.png, leave_capture_out.run.json.

Usage: python -m src.experiments.leave_capture_out   (make final)
"""
import json

import numpy as np
import polars as pl

from src.data.labels import BENIGN, window_of
from src.evaluation.evaluate import MODEL_COLOR, MODEL_LABEL
from src.evaluation.metrics import binary_metrics, predictions_and_scores
from src.evaluation.stats import wilson
from src.experiments.make_splits import CAPTURE, NESTED, folds, load_assignment, outer_folds
from src.experiments.nested_cv import load_sample_M
from src.models.pipelines import auxiliary_columns, build_pipeline, feature_names, prepare
from src.reporting.style import INK_MUTED, INK_SECONDARY, apply_style, plt, save
from src.utils.config import load_config, path
from src.utils.io import rel, sha256_file, write_csv_gz, write_json
from src.utils.run_metadata import recorded_run

DESIGNS = ("nested_M", "capture_M")  # by row (reference) and by capture


def recall_by_type(pred: pl.DataFrame, level: float) -> pl.DataFrame:
    """Sensitivity per attack type for each design, with Wilson intervals."""
    att = pred.filter(pl.col("label34") != BENIGN)
    t = att.group_by("design", "label34").agg(pl.len().alias("n"), pl.col("pred").sum().alias("detected")).sort(
        "design", "label34")
    lo, hi = wilson(t["detected"].to_numpy(), t["n"].to_numpy(), level)
    return t.with_columns((pl.col("detected") / pl.col("n")).alias("sensitivity"), pl.Series("ci_low", lo),
                          pl.Series("ci_high", hi))


def compare(recall: pl.DataFrame, grouped: set[str], collapse: float) -> pl.DataFrame:
    """Row design vs capture design per type; a type collapses if the drop is at least `collapse` and the Wilson
    intervals do not overlap (pre-registered rule)."""
    a = recall.filter(pl.col("design") == DESIGNS[0]).drop("design")
    b = recall.filter(pl.col("design") == DESIGNS[1]).drop("design")
    t = a.join(b, on="label34", suffix="_capture")
    return (t.with_columns(pl.col("label34").is_in(sorted(grouped)).alias("grouped"),
                           pl.col("label34").map_elements(window_of, return_dtype=pl.Int64).alias("window"),
                           (pl.col("sensitivity") - pl.col("sensitivity_capture")).alias("drop"))
            .with_columns(((pl.col("drop") >= collapse) & (pl.col("ci_high_capture") < pl.col("ci_low")))
                          .alias("collapses"))
            .sort(pl.col("grouped").not_(), "label34"))


def figure(table: pl.DataFrame, model_name: str, target) -> None:
    g = table.filter(pl.col("grouped")).sort("sensitivity")
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    y = np.arange(g.height)[::-1]
    colour = MODEL_COLOR[model_name]
    for yi, a, b in zip(y, g["sensitivity"], g["sensitivity_capture"]):
        ax.plot([a, b], [yi, yi], color=INK_MUTED, linewidth=1, zorder=1)
    ax.scatter(g["sensitivity"], y, s=34, color=colour, edgecolors="white", linewidths=1, zorder=3,
               label="split by row (outer folds)")
    ax.scatter(g["sensitivity_capture"], y, s=34, facecolors="white", edgecolors=colour, linewidths=1.5, zorder=3,
               label="split by capture (leave-capture-out)")
    names = [f"{n} *" if c else n for n, c in zip(g["label34"], g["collapses"])]
    ax.set_yticks(y, names, fontsize=7.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Sensitivity on the out-of-fold predictions (* = collapse by the pre-registered rule)")
    ax.set_title(f"Leave-capture-out — {MODEL_LABEL[model_name]}, 16 types with ≥ 5 captures", loc="left")
    ax.legend(loc="lower left", fontsize=8, frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    ax.text(0, -0.13, "Same configuration (final model) in both designs; only the assignment of the test rows differs.",
            transform=ax.transAxes, fontsize=8, color=INK_SECONDARY)
    save(fig, target)


def main() -> None:
    cfg, feature_cfg, search_cfg = load_config(), load_config("features"), load_config("search_spaces")
    final = json.loads((path(cfg, "results_dir") / "final" / "final_model.json").read_text("utf-8"))
    model_name, params = final["model"], final["best_params"]
    level, collapse = float(cfg["evaluation"]["confidence_level"]), float(cfg["ablation"]["type_collapse"])
    out_dir = path(cfg, "results_dir") / "capture"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    with recorded_run(cfg, out_dir, "leave_capture_out") as run:
        M = load_sample_M(cfg)
        features = feature_names(M.columns, feature_cfg, final["feature_set"], auxiliary_columns(cfg))
        X, y = prepare(M, features, feature_cfg), M[cfg["columns"]["target"]].to_numpy()
        labels = M[cfg["columns"]["label"]].to_numpy()
        capture = load_assignment(cfg, CAPTURE, M)
        grouped = set(capture.filter(pl.col("grouped"))["label34"].unique().to_list())
        designs = {DESIGNS[0]: outer_folds(load_assignment(cfg, NESTED, M)),
                   DESIGNS[1]: folds(capture["capture_outer"].to_numpy())}
        parts, fold_rows = [], []
        for design, splits in designs.items():
            for k, (train, test) in enumerate(splits):
                model = build_pipeline(model_name, features, feature_cfg, search_cfg, int(cfg["seed"]),
                                       n_jobs=int(cfg["n_jobs"]))
                model.set_params(**{f"model__{p}": tuple(v) if isinstance(v, list) else v for p, v in params.items()})
                model.fit(X.iloc[train], y[train])
                pred, score = predictions_and_scores(model, X.iloc[test])
                parts.append(pl.DataFrame({"design": design, "fold": k, "position": test, "y": y[test],
                                           "label34": labels[test], "pred": pred, "score": score}))
                fold_rows.append({"design": design, "fold": k, "n_train": int(train.size), "n_test": int(test.size),
                                  **{f"test_{m}": v for m, v in binary_metrics(y[test], pred, score).items()}})
                print(f"{design} fold {k}: MCC {fold_rows[-1]['test_mcc']:.4f}", flush=True)
        pred = pl.concat(parts).sort("design", "position")
        fold_table = pl.DataFrame(fold_rows)
        table = compare(recall_by_type(pred, level), grouped, collapse)
        mcc = fold_table.group_by("design").agg(pl.col("test_mcc").mean().alias("mean"),
                                               pl.col("test_mcc").std().alias("std")).sort("design")
        g = table.filter(pl.col("grouped"))
        summary = {"model": model_name, "configuration": "final model (results/final/final_model.json)",
                   "grouped_types": sorted(grouped), "collapse_rule": f"drop >= {collapse} and non-overlapping "
                                                                       f"{level:.0%} Wilson intervals",
                   "collapsed_types": g.filter(pl.col("collapses"))["label34"].to_list(),
                   "grouped_mean_sensitivity": {"by_row": float(g["sensitivity"].mean()),
                                                "by_capture": float(g["sensitivity_capture"].mean())},
                   "mcc": {r["design"]: {"mean": r["mean"], "std": r["std"]} for r in mcc.iter_rows(named=True)}}
        write_csv_gz(pred.drop("label34"), out_dir / "predictions.csv.gz")
        fold_table.write_csv(out_dir / "folds.csv")
        table.write_csv(out_dir / "recall_by_type.csv")
        write_json(summary, out_dir / "summary.json")
        figure(table, model_name, out_dir / "fig_capture_recall.png")
        names = ("predictions.csv.gz", "folds.csv", "recall_by_type.csv", "summary.json", "fig_capture_recall.png")
        run["outputs"] = {rel(out_dir / n): sha256_file(out_dir / n) for n in names}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
