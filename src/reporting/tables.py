"""Etapa 10 — summary tables gathered from the recorded results (make report).

Nothing is recomputed: every table is read from the result files of Etapas 2 to 9, rounded for reading and written
to results/tables/ as CSV and Markdown. results/tables/index.csv lists each table, its title, the files it was built
from and the SHA-256 of its CSV.

Usage: python -m src.reporting.tables   (make report)
"""
import json

import polars as pl

from src.utils.config import load_config, path
from src.utils.io import rel, sha256_file
from src.utils.run_metadata import recorded_run

DIGITS = 4          # decimals kept in the tables; the result files keep full precision
P_VALUE_DIGITS = 2  # significant digits of p-values, which can be far below 10^-DIGITS
P_VALUES = ("p", "p_holm")


def round_floats(df: pl.DataFrame) -> pl.DataFrame:
    """Float columns rounded to DIGITS decimals, except p-values, which keep P_VALUE_DIGITS significant digits."""
    p_cols = [c for c in P_VALUES if c in df.columns]
    df = df.with_columns(pl.col(pl.Float64).exclude(p_cols).round(DIGITS))
    return df.with_columns(pl.col(c).map_elements(lambda v: float(f"{v:.{P_VALUE_DIGITS}g}"), return_dtype=pl.Float64)
                           for c in p_cols)


def markdown(df: pl.DataFrame) -> str:
    """The table as a Markdown pipe table, header row first."""
    def cell(v):
        """Cell text: empty for null, with `|` escaped so it does not split the column."""
        return "" if v is None else str(v).replace("|", "\\|")
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * df.width]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.iter_rows()]
    return "\n".join(lines) + "\n"


def mean_std(df: pl.DataFrame, metrics: list[str]) -> pl.DataFrame:
    """Columns <m>_mean and <m>_std side by side, in the order of `metrics`."""
    return df.select([c for m in metrics for c in (f"{m}_mean", f"{m}_std") if c in df.columns])


def build(cfg: dict, feature_cfg: dict) -> dict[str, tuple[str, list, pl.DataFrame]]:
    """Every summary table by name, as (title, source files, table), read from the recorded results."""
    res = path(cfg, "results_dir")
    best = json.loads((res / "evaluation" / "hypotheses.json").read_text(encoding="utf-8"))["best_model"]
    tables: dict[str, tuple[str, list, pl.DataFrame]] = {}
    sc, outer = cfg["validation"]["spotcheck"], int(cfg["validation"]["outer"]["n_splits"])

    src = res / "sampling" / "sample_counts.csv"
    counts = pl.read_csv(src)
    tables["samples"] = ("Samples E and M by category (after deduplication)", [src], counts.group_by("category8").agg(
        pl.len().alias("types"), pl.col("n_unique").sum(), pl.col("n_E").sum(), pl.col("n_M").sum())
        .sort("category8"))

    tables["dropped_features"] = ("Attributes removed before any split, and why", ["configs/features.yaml"],
                                  pl.DataFrame([{"attribute": d["name"], "reason": d["reason"]}
                                                for d in feature_cfg["drop"]]))

    src = res / "spotcheck" / "summary.csv"
    s = pl.read_csv(src).sort("mcc_mean", descending=True)
    tables["spotcheck"] = (f"Spot-checking on sample E: default hyperparameters, {sc['n_splits']} folds x "
                           f"{sc['n_repeats']} repetitions", [src],
                           s.select("algorithm", "inductive_bias", "mcc_mean", "mcc_std", "mcc_train_mean",
                                    "mcc_train_minus_validation", "sensitivity_mean", "specificity_mean"))

    src, ci = res / "evaluation" / "summary.csv", res / "evaluation" / "bootstrap_ci.csv"
    ev = pl.read_csv(src)
    metrics = ["mcc", "f1_attack", "f1_benign", "balanced_accuracy", "auprc", "roc_auc", "sensitivity",
               "specificity", "fpr"]
    tables["nested_cv"] = (f"Outer test folds of the nested cross-validation on sample M (mean and std over {outer} folds)",
                           [src, ci], pl.concat([ev.select("model"), mean_std(ev, metrics)], how="horizontal")
                           .join(pl.read_csv(ci).rename({"ci_low": "mcc_ci_low", "ci_high": "mcc_ci_high"}),
                                 on="model", how="left"))

    src = res / "evaluation" / "tests.csv"
    tables["model_tests"] = ("Corrected resampled t-test with Holm's correction, and bootstrap 95% interval of the "
                             "MCC difference", [src], pl.read_csv(src).select(
                                 "model_a", "model_b", "diff_mean", "folds_a_better", "t", "p_holm", "ci_low",
                                 "ci_high"))

    src = res / "evaluation" / "recall_by_group.csv"
    tables["recall_by_group"] = (f"Sensitivity of the best model ({best}) by window group and category, with Wilson "
                                 "95% intervals", [src], pl.read_csv(src).filter(
                                     pl.col("model") == best, pl.col("level").is_in(["all", "window", "category8"]))
                                 .select("level", "group", "n", "missed", "sensitivity", "ci_low", "ci_high",
                                         "missed_share", "instance_share"))

    src = res / "evaluation" / "metrics_by_window.csv"
    w = pl.read_csv(src)
    tables["metrics_by_window"] = ("MCC with benign traffic and the attacks of one window group only (mean and std "
                                   f"over {outer} folds)", [src], w.filter(pl.col("statistic") == "fold_mean")
                                   .select("model", "attack_window", pl.col("mcc").alias("mcc_mean"))
                                   .join(w.filter(pl.col("statistic") == "fold_std")
                                         .select("model", "attack_window", pl.col("mcc").alias("mcc_std")),
                                         on=["model", "attack_window"]))

    src, ppv = res / "evaluation" / "reweighted.csv", res / "evaluation" / "ppv_projection.csv"
    tables["reweighted"] = (f"Metrics of {best} re-weighted to the original distribution of the dataset", [src],
                            pl.read_csv(src).filter(pl.col("model") == best).drop("model"))
    tables["ppv_projection"] = (f"Precision of {best} projected to other attack prevalences", [ppv],
                                pl.read_csv(ppv).filter(pl.col("model") == best).drop("model"))

    perm, shap = res / "interpretation" / "permutation_summary.csv", res / "interpretation" / "shap_summary.csv"
    top = int(cfg["interpretation"]["top_k"])
    p = pl.read_csv(perm).head(top).rename({"feature": "permutation_feature", "mean": "mcc_drop_mean",
                                            "std": "mcc_drop_std"})
    sv = pl.read_csv(shap).head(top).rename({"feature": "shap_feature", "mean": "mean_abs_shap",
                                             "std": "mean_abs_shap_std"})
    tables["interpretation_top"] = (f"Top {top} attributes of {best}: permutation importance and mean |SHAP|",
                                    [perm, shap], p.join(sv, on="rank"))

    src = res / "shortcuts" / "single_feature_summary.csv"
    tables["single_feature_models"] = ("One-attribute trees on the outer folds of sample M, with the values on "
                                       "sample E", [src], pl.read_csv(src))

    tests, win = res / "shortcuts" / "ablation_tests.csv", res / "shortcuts" / "ablation_by_window.csv"
    bw = pl.read_csv(win).filter(pl.col("statistic") == "pooled")
    by_window = (bw.filter(pl.col("attack_window") == 10).select("feature_set", pl.col("mcc").alias("mcc_window10"),
                                                                pl.col("sensitivity").alias("sensitivity_window10"),
                                                                "fpr")
                 .join(bw.filter(pl.col("attack_window") == 100).select(
                     "feature_set", pl.col("mcc").alias("mcc_window100"),
                     pl.col("sensitivity").alias("sensitivity_window100")), on="feature_set"))
    abl = pl.read_csv(tests).select("feature_set", "mcc_reference", "mcc_ablated", "drop_mean", "folds_with_drop",
                                    "t", "p_holm", "ci_low", "ci_high", "reading")
    tables["ablation"] = (f"Shortcut ablation of {best}: MCC drop against the full attribute set, and metrics by "
                          "window group", [tests, win], abl.join(by_window, on="feature_set", how="full",
                                                                 coalesce=True)
                          .sort(pl.col("feature_set") != cfg["ablation"]["reference"]))

    src = res / "capture" / "summary.json"
    cap = json.loads(src.read_text(encoding="utf-8"))
    tables["leave_capture_out"] = ("Leave-capture-out: row-wise folds against folds grouped by capture", [src],
                                   pl.DataFrame([{"design": d, "mcc_mean": cap["mcc"][d]["mean"],
                                                  "mcc_std": cap["mcc"][d]["std"],
                                                  "grouped_types": len(cap["grouped_types"]),
                                                  "collapsed_types": len(cap["collapsed_types"])}
                                                 for d in ("nested_M", "capture_M")]))

    rows, sources = [], []
    for model in cfg["evaluation"]["models"]:
        for f in sorted((res / "nested" / model).glob("fold*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            if f.stem.removeprefix("fold").isdigit() and "best_params" in d:  # the dummy is never tuned
                rows.append({"model": model, "outer_fold": d["outer_fold"],
                             "best_params": json.dumps(d["best_params"], sort_keys=True)})
                sources.append(f)
    final = res / "final" / "final_model.json"
    fm = json.loads(final.read_text(encoding="utf-8"))
    rows.append({"model": fm["model"], "outer_fold": None, "best_params": json.dumps(fm["best_params"],
                                                                                      sort_keys=True)})
    tables["chosen_hyperparameters"] = ("Hyperparameters chosen in each outer fold, and those of the final model "
                                        "(outer_fold empty)", [*sources, final], pl.DataFrame(rows))
    return tables


def main() -> None:
    """Write each table as CSV and Markdown, plus results/tables/index.csv, inside a recorded run."""
    cfg, feature_cfg = load_config(), load_config("features")
    out_dir = path(cfg, "results_dir") / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    with recorded_run(cfg, out_dir, "tables") as run:
        index, outputs, inputs = [], {}, {}
        for name, (title, sources, df) in build(cfg, feature_cfg).items():
            df = round_floats(df)
            csv, md = out_dir / f"{name}.csv", out_dir / f"{name}.md"
            df.write_csv(csv)
            md.write_bytes((f"**{title}**\n\n" + markdown(df)).encode("utf-8"))
            # a string source is a repository path cited as is (configs/features.yaml): listed, not hashed
            names = [rel(s) if not isinstance(s, str) else s for s in sources]
            index.append({"table": name, "title": title, "sources": "; ".join(names), "csv_sha256": sha256_file(csv)})
            outputs |= {rel(csv): sha256_file(csv), rel(md): sha256_file(md)}
            inputs |= {n: sha256_file(s) for n, s in zip(names, sources) if not isinstance(s, str)}
            print(f"{name}: {df.height} rows")
        pl.DataFrame(index).write_csv(out_dir / "index.csv")
        outputs[rel(out_dir / "index.csv")] = sha256_file(out_dir / "index.csv")
        run["inputs"], run["outputs"] = inputs, outputs


if __name__ == "__main__":
    main()
