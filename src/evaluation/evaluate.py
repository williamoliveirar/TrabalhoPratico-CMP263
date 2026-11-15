"""Etapa 8: performance estimation and model comparison on the out-of-fold predictions of the nested CV (sample M).

Inputs (all versioned; no raw data or sample parquet needed):
- results/nested/folds.csv and results/nested/<model>/fold<k>_predictions.csv.gz (Etapa 7);
- data/splits/nested_M.csv.gz: label34 and outer fold of every row of M, in the row order of M (Etapa 5);
- results/inspection/label_counts_total.csv (Etapa 1) and results/sampling/sample_counts.csv (Etapa 2), for the
  weights that restore the original distribution of the dataset (D16, P1).

Every choice below was pre-registered in configs/base.yaml (`evaluation`) on 2026-11-12, before this script
computed any test:
- summary.csv: mean ± std over the 5 outer test folds of the metrics of D5 and D6 (main table);
- tests.csv: corrected resampled t-test for each pre-registered comparison, Holm-adjusted over the family, with the
  paired bootstrap interval of the MCC difference (P5);
- bootstrap_ci.csv: MCC of each model over the pooled out-of-fold predictions, with its bootstrap interval;
- confusion_matrix.csv: pooled confusion matrix of each model (the best one is flagged);
- recall_by_group.csv and benign_fpr.csv: sensitivity by attack type, category, window group and overall, and the
  false positive rate on benign traffic, with Wilson intervals (Q2, H2, D12);
- reweighted.csv: the main metrics re-weighted to the original distribution of the dataset (P1 a);
- metrics_by_window.csv: the main metrics on benign traffic plus the attacks of one window group (D12, decided on
  2026-10-25: the 10-packet group is the comparison a window-size shortcut cannot win);
- ppv_projection.csv: precision projected to other attack prevalences (P1 b);
- hypotheses.json: best model, H1 and H2 evaluated with the pre-registered rules;
- fig_mcc_per_fold.png, fig_recall_by_type.png, fig_pr_roc.png; evaluate.run.json.
Nothing here tunes a threshold, refits a model or recomputes a prediction.

Usage: python -m src.evaluation.evaluate   (make evaluate)
"""
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data.labels import BENIGN, CATEGORY, window_of
from src.evaluation.metrics import binary_metrics
from src.evaluation.stats import (bootstrap_counts, corrected_resampled_ttest, holm, mcc_from_counts,
                                  percentile_interval, projected_ppv, wilson)
from src.experiments.make_splits import NESTED
from src.models.registry import BASELINE
from src.reporting.style import INK, INK_MUTED, INK_SECONDARY, SERIES, apply_style, plt, save
from src.utils.config import load_config, path
from src.utils.io import rel, sha256_file, write_json
from src.utils.run_metadata import recorded_run

METRICS = ["mcc", "f1_attack", "f1_benign", "balanced_accuracy", "auprc", "roc_auc", "sensitivity", "specificity",
           "fpr"]
CELLS = ["tp", "fn", "tn", "fp"]
H1_REFERENCE = "decision_tree"  # H1 compares the tree ensembles with a single decision tree (P7)
LEVELS = ["all", "window", "category8", "label34"]
# slots 0 and 1 of SERIES are reserved for benign and attack (src/reporting/style.py); colour follows the model
MODEL_COLOR = {"lightgbm": SERIES[2], "random_forest": SERIES[3], "mlp": SERIES[4], "decision_tree": SERIES[6],
               BASELINE: INK_MUTED}  # SERIES[5] skipped: a second green next to LightGBM's
MODEL_LABEL = {"lightgbm": "LightGBM", "random_forest": "Random forest", "mlp": "MLP",
               "decision_tree": "Decision tree", BASELINE: "Dummy"}


def label_table(cfg: dict) -> pl.DataFrame:
    """position (row of M), label34, outer fold, category8 and window size of every instance of M."""
    windows = {label: window_of(label) for label in CATEGORY}
    return (pl.read_csv(path(cfg, "splits_dir") / f"{NESTED}.csv.gz", columns=["label34", "outer"])
            .with_row_index("position")
            .with_columns(pl.col("position").cast(pl.Int64),
                          pl.col("label34").replace_strict(CATEGORY).alias("category8"),
                          pl.col("label34").replace_strict(windows, return_dtype=pl.Int64).alias("window")))


def load_oof(root, model: str, n_outer: int, labels: pl.DataFrame) -> pl.DataFrame:
    """Out-of-fold predictions of one model, one row per instance of M in position order, with its labels."""
    parts = [pl.read_csv(root / model / f"fold{k}_predictions.csv.gz").with_columns(pl.lit(k).alias("outer_fold"))
             for k in range(n_outer)]
    oof = pl.concat(parts).join(labels, on="position", how="left").sort("position")
    if oof.height != labels.height or oof["position"].n_unique() != labels.height:
        raise ValueError(f"{model}: the out-of-fold predictions do not cover sample M exactly once")
    if not ((oof["outer_fold"] == oof["outer"]).all() and ((oof["label34"] != BENIGN).cast(pl.Int64) == oof["y"]).all()):
        raise ValueError(f"{model}: predictions disagree with data/splits ({NESTED}) on the fold or the class")
    return oof


def type_weights(cfg: dict, labels: pl.DataFrame) -> pl.DataFrame:
    """Weight of each label34: its fraction in the dataset (official CSV rows, before deduplication) divided by its
    fraction in M (P1 a). Fails if M's counts disagree with results/sampling/sample_counts.csv."""
    results = path(cfg, "results_dir")
    dataset = (pl.read_csv(results / "inspection" / "label_counts_total.csv")
               .filter(pl.col("level") == "label34").select(pl.col("value").alias("label34"),
                                                             pl.col("fraction").alias("dataset_fraction")))
    in_m = labels.group_by("label34").agg(pl.len().alias("n_M"))
    recorded = pl.read_csv(results / "sampling" / "sample_counts.csv").select("label34", pl.col("n_M").alias("recorded"))
    check = in_m.join(recorded, on="label34", how="full", coalesce=True)
    if check.null_count().sum_horizontal().item() or not (check["n_M"] == check["recorded"]).all():
        raise ValueError("label counts of M disagree with results/sampling/sample_counts.csv")
    table = in_m.join(dataset, on="label34", how="full", coalesce=True)
    if table.null_count().sum_horizontal().item():
        raise ValueError("label34 values of M and of label_counts_total.csv differ")
    return (table.with_columns((pl.col("n_M") / pl.col("n_M").sum()).alias("m_fraction"))
            .with_columns((pl.col("dataset_fraction") / pl.col("m_fraction")).alias("weight")).sort("label34"))


def summary_table(folds: pl.DataFrame, models: list[str]) -> pl.DataFrame:
    """Mean and std over the outer folds of every test metric (main table, PLANO Etapa 8 task 1)."""
    return (folds.filter(pl.col("model").is_in(models))
            .group_by("model").agg(pl.len().alias("n_folds"),
                                   *[pl.col(f"test_{m}").mean().alias(f"{m}_mean") for m in METRICS],
                                   *[pl.col(f"test_{m}").std().alias(f"{m}_std") for m in METRICS])
            .with_columns(pl.col("model").replace_strict({m: i for i, m in enumerate(models)}).alias("order"))
            .sort("order").drop("order"))


def comparison_table(folds: pl.DataFrame, comparisons: list[list[str]], boot_mcc: np.ndarray, pooled_mcc: dict,
                     models: list[str], level: float, alpha: float) -> pl.DataFrame:
    """Corrected resampled t-test per comparison, Holm over the family, and the paired bootstrap interval of the
    pooled MCC difference."""
    rows = []
    for a, b in comparisons:
        fa = folds.filter(pl.col("model") == a).sort("outer_fold")
        fb = folds.filter(pl.col("model") == b).sort("outer_fold")
        n_train, n_test = fa["n_train"].unique(), fa["n_test"].unique()
        if n_train.len() != 1 or n_test.len() != 1:
            raise ValueError("the outer folds must have equal training and test sizes for the corrected t-test")
        t, p = corrected_resampled_ttest(fa["test_mcc"], fb["test_mcc"], int(n_train[0]), int(n_test[0]))
        diff = boot_mcc[:, models.index(a)] - boot_mcc[:, models.index(b)]  # paired: same resamples for both
        low, high = percentile_interval(diff, level)
        d = fa["test_mcc"].to_numpy() - fb["test_mcc"].to_numpy()
        rows.append({"model_a": a, "model_b": b, "mcc_a_mean": fa["test_mcc"].mean(), "mcc_b_mean": fb["test_mcc"].mean(),
                     "diff_mean": float(d.mean()), "diff_std": float(d.std(ddof=1)), "folds_a_better": int((d > 0).sum()),
                     "n_train": int(n_train[0]), "n_test": int(n_test[0]), "t": t, "p": p,
                     "diff_pooled": pooled_mcc[a] - pooled_mcc[b], "ci_low": low, "ci_high": high})
    table = pl.DataFrame(rows)
    p_holm = holm(table["p"].to_numpy())
    return table.with_columns(pl.Series("p_holm", p_holm),
                              pl.Series("reject_h0", p_holm < alpha)).select(
        "model_a", "model_b", "mcc_a_mean", "mcc_b_mean", "diff_mean", "diff_std", "folds_a_better", "n_train",
        "n_test", "t", "p", "p_holm", "reject_h0", "diff_pooled", "ci_low", "ci_high")


def recall_by_group(oof: pl.DataFrame, model: str, level: float) -> pl.DataFrame:
    """Sensitivity of one model by window group, attack category and attack type, and over all attacks, with Wilson
    intervals, the share of the false negatives and the share of the attack instances of each group."""
    attacks = oof.filter(pl.col("y") == 1)
    total, missed_total = attacks.height, attacks.height - int(attacks["pred"].sum())
    meta = attacks.select("label34", "category8", "window").unique()
    parts = [pl.DataFrame({"level": ["all"], "group": ["all attacks"], "n": [total],
                           "detected": [int(attacks["pred"].sum())]})]
    for col in LEVELS[1:]:
        parts.append(attacks.group_by(col).agg(pl.len().alias("n"), pl.col("pred").sum().alias("detected"))
                     .select(pl.lit(col).alias("level"), pl.col(col).cast(pl.String).alias("group"), "n", "detected"))
    table = pl.concat([p.with_columns(pl.col("n").cast(pl.Int64), pl.col("detected").cast(pl.Int64)) for p in parts])
    lo, hi = wilson(table["detected"].to_numpy(), table["n"].to_numpy(), level)
    by_type = meta.select(pl.col("label34").alias("group"), "category8", "window")
    by_category = meta.group_by("category8").agg(pl.col("window").first()).select(pl.col("category8").alias("group"),
                                                                                   "category8", "window")
    context = pl.concat([by_type.with_columns(pl.lit("label34").alias("level")),
                         by_category.with_columns(pl.lit("category8").alias("level"))], how="diagonal")
    return (table.with_columns(pl.lit(model).alias("model"),
                               (pl.col("n") - pl.col("detected")).alias("missed"),
                               (pl.col("detected") / pl.col("n")).alias("sensitivity"),
                               pl.Series("ci_low", lo), pl.Series("ci_high", hi))
            .with_columns((pl.col("missed") / missed_total if missed_total else pl.lit(None)).alias("missed_share"),
                          (pl.col("n") / total).alias("instance_share"))
            .join(context, on=["level", "group"], how="left")
            .with_columns(pl.when(pl.col("level") == "window").then(pl.col("group").cast(pl.Int64))
                          .otherwise(pl.col("window")).alias("window"))
            .with_columns(pl.col("level").replace_strict({lv: i for i, lv in enumerate(LEVELS)}).alias("order"))
            .sort("order", "sensitivity", "group").drop("order")  # group breaks ties: byte-stable output
            .select("model", "level", "group", "category8", "window", "n", "detected", "missed", "sensitivity",
                    "ci_low", "ci_high", "missed_share", "instance_share"))


def benign_fpr(oof: pl.DataFrame, model: str, level: float) -> dict:
    """False positive rate of one model on the pooled out-of-fold benign instances, with its Wilson interval."""
    benign = oof.filter(pl.col("y") == 0)
    fp, n = int(benign["pred"].sum()), benign.height
    lo, hi = wilson(fp, n, level)
    return {"model": model, "n": n, "false_positives": fp, "fpr": fp / n, "ci_low": float(lo), "ci_high": float(hi)}


def reweighted(oofs: dict, models: list[str], weights: pl.DataFrame) -> pl.DataFrame:
    """Main metrics re-weighted to the original distribution (P1 a): mean and std over folds, and pooled."""
    rows = []
    for m in models:
        o = oofs[m].join(weights.select("label34", "weight"), on="label34", how="left")
        per_fold = [binary_metrics(f["y"].to_numpy(), f["pred"].to_numpy(), f["score"].to_numpy(),
                                   sample_weight=f["weight"].to_numpy())
                    for _, f in o.group_by("outer_fold", maintain_order=True)]
        pooled = binary_metrics(o["y"].to_numpy(), o["pred"].to_numpy(), o["score"].to_numpy(),
                                sample_weight=o["weight"].to_numpy())
        rows.append({"model": m, "statistic": "fold_mean", **{k: float(np.mean([f[k] for f in per_fold])) for k in METRICS}})
        rows.append({"model": m, "statistic": "fold_std", **{k: float(np.std([f[k] for f in per_fold], ddof=1)) for k in METRICS}})
        rows.append({"model": m, "statistic": "pooled", **{k: float(pooled[k]) for k in METRICS}})
    return pl.DataFrame(rows)


def metrics_by_window(oofs: dict, models: list[str]) -> pl.DataFrame:
    """Main metrics on benign traffic plus the attacks of one window group only (D12): with the 10-packet attacks,
    benign and attack share the window size, the comparison that a window-size shortcut cannot win."""
    rows = []
    for m in models:
        for w in sorted(oofs[m]["window"].unique().to_list()):
            sub = oofs[m].filter((pl.col("y") == 0) | (pl.col("window") == w))
            per_fold = [binary_metrics(f["y"].to_numpy(), f["pred"].to_numpy(), f["score"].to_numpy())
                        for _, f in sub.group_by("outer_fold", maintain_order=True)]
            pooled = binary_metrics(sub["y"].to_numpy(), sub["pred"].to_numpy(), sub["score"].to_numpy())
            base = {"model": m, "attack_window": w, "n_attack": int(sub["y"].sum()), "n_benign": int((sub["y"] == 0).sum())}
            rows.append({**base, "statistic": "fold_mean", **{k: float(np.mean([f[k] for f in per_fold])) for k in METRICS}})
            rows.append({**base, "statistic": "fold_std", **{k: float(np.std([f[k] for f in per_fold], ddof=1)) for k in METRICS}})
            rows.append({**base, "statistic": "pooled", **{k: float(pooled[k]) for k in METRICS}})
    return pl.DataFrame(rows)


def ppv_table(oofs: dict, models: list[str], weights: pl.DataFrame, prevalences: list[float],
              dataset_prevalence: float) -> pl.DataFrame:
    """Precision projected to other attack prevalences (P1 b), with the attack mix of M and with the original mix."""
    rows = []
    for m in models:
        o = oofs[m].join(weights.select("label34", "weight"), on="label34", how="left")
        att, ben = o.filter(pl.col("y") == 1), o.filter(pl.col("y") == 0)
        sens = float(att["pred"].mean())
        # each attack type weighted to its share in the original dataset instead of its share in M
        sens_original = float((att["pred"] * att["weight"]).sum() / att["weight"].sum())
        spec = 1 - float(ben["pred"].mean())
        for prevalence, source in [*((p, "config") for p in prevalences), (dataset_prevalence, "dataset")]:
            rows.append({"model": m, "prevalence": prevalence, "prevalence_source": source,
                         "sensitivity_m_mix": sens, "sensitivity_original_mix": sens_original, "specificity": spec,
                         "ppv_m_mix": float(projected_ppv(sens, spec, prevalence)),
                         "ppv_original_mix": float(projected_ppv(sens_original, spec, prevalence))})
    return pl.DataFrame(rows)


def hypotheses(summary: pl.DataFrame, tests: pl.DataFrame, confusion: pl.DataFrame, recall: pl.DataFrame,
               best: str, h2_categories: list[str]) -> dict:
    """Best model, H1 and H2 evaluated with the rules pre-registered on 2026-11-12."""
    h1 = tests.filter(pl.col("model_b") == H1_REFERENCE)
    first_half = bool(h1.height > 0 and (h1["diff_mean"] > 0).all() and h1["reject_h0"].all() and (h1["ci_low"] > 0).all())
    errors = {r["model"]: r["fn"] + r["fp"] for r in confusion.iter_rows(named=True)}
    mcc = {r["model"]: r["mcc_mean"] for r in summary.iter_rows(named=True)}
    second_half = [{"ensemble": r["model_a"], "mcc_difference": r["diff_mean"],
                    "errors_ensemble": errors[r["model_a"]], "errors_tree": errors[H1_REFERENCE],
                    "tree_extra_errors_relative": errors[H1_REFERENCE] / errors[r["model_a"]] - 1,
                    "tree_mcc_over_ensemble_mcc": mcc[H1_REFERENCE] / mcc[r["model_a"]]} for r in h1.iter_rows(named=True)]
    cats = recall.filter((pl.col("model") == best) & (pl.col("level") == "category8"))
    per_category = [{"category": r["group"], "missed_share": r["missed_share"], "instance_share": r["instance_share"],
                     "sensitivity": r["sensitivity"],
                     # no false negatives at all: no category concentrates them
                     "concentrates_false_negatives": r["missed_share"] is not None
                     and r["missed_share"] > r["instance_share"],
                     "in_h2": r["group"] in h2_categories} for r in cats.iter_rows(named=True)]
    hits = sum(c["concentrates_false_negatives"] for c in per_category if c["in_h2"])
    verdict = "confirmed" if hits == len(h2_categories) else "partially confirmed" if hits else "refuted"
    return {"best_model": best, "best_model_rule": "highest mean outer-fold test MCC",
            "H1": {"first_half_ensembles_have_highest_mcc": "confirmed" if first_half else "not confirmed",
                   "second_half_difference_is_small": "no verdict (pre-registered: no threshold was set)",
                   "second_half_numbers": second_half},
            "H2": {"verdict": verdict, "categories_in_h2_that_concentrate": hits, "categories": per_category}}


def figure_mcc_per_fold(folds: pl.DataFrame, models: list[str], target) -> None:
    """Test MCC of every tuned model in each outer fold; grey lines join the same fold (paired design)."""
    shown = [m for m in models if m != BASELINE]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    wide = folds.pivot(on="model", index="outer_fold", values="test_mcc").sort("outer_fold")
    for row in wide.iter_rows(named=True):
        ax.plot(range(len(shown)), [row[m] for m in shown], color=INK_MUTED, linewidth=0.8, alpha=0.6, zorder=1)
    for i, m in enumerate(shown):
        values = wide[m].to_numpy()
        ax.scatter([i] * values.size, values, s=30, color=MODEL_COLOR[m], edgecolors="white", linewidths=1, zorder=3)
        ax.plot([i - 0.18, i + 0.18], [values.mean()] * 2, color=INK, linewidth=2, zorder=4)
        ax.annotate(f"{values.mean():.3f}", (i + 0.22, values.mean()), va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(range(len(shown)), [MODEL_LABEL[m] for m in shown])
    ax.set_xlim(-0.5, len(shown) - 0.2)
    ax.set_ylabel("Test MCC (outer fold)")
    ax.set_title("Test MCC per outer fold (nested CV, sample M)", loc="left")
    ax.text(0, -0.2, "Each grey line is one outer fold; black bar = mean. The dummy baseline scored 0 in every fold.",
            transform=ax.transAxes, fontsize=8, color=INK_SECONDARY)
    save(fig, target)


def figure_recall_by_type(recall: pl.DataFrame, fpr: dict, best: str, target) -> None:
    """Sensitivity of the best model per attack type with Wilson intervals, worst first; marker = window size."""
    types = recall.filter((pl.col("model") == best) & (pl.col("level") == "label34")).sort("sensitivity")
    overall = recall.filter((pl.col("model") == best) & (pl.col("level") == "all"))["sensitivity"][0]
    fig, ax = plt.subplots(figsize=(6.4, 8.2))
    y = np.arange(types.height)[::-1]
    sens = types["sensitivity"].to_numpy()
    err = np.vstack([sens - types["ci_low"].to_numpy(), types["ci_high"].to_numpy() - sens])
    ax.errorbar(sens, y, xerr=err, fmt="none", ecolor=INK_MUTED, elinewidth=1, capsize=0, zorder=2)
    color = MODEL_COLOR[best]
    w10 = types["window"].to_numpy() == 10
    ax.scatter(sens[w10], y[w10], s=36, marker="o", color=color, edgecolors="white", linewidths=1, zorder=3)
    ax.scatter(sens[~w10], y[~w10], s=36, marker="D", facecolors="white", edgecolors=color, linewidths=1.5, zorder=3)
    ax.axvline(overall, color=INK_SECONDARY, linewidth=1, linestyle="--", zorder=1)
    ax.set_yticks(y, types["group"].to_list(), fontsize=7.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Sensitivity (recall) on the pooled out-of-fold predictions, 95% Wilson interval\n"
                  f"False positive rate on benign traffic: {fpr['fpr']:.4f} "
                  f"({fpr['false_positives']:,} of {fpr['n']:,})")
    ax.set_title(f"Sensitivity per attack type — {MODEL_LABEL[best]}", loc="left")
    handles = [Line2D([], [], marker="o", linestyle="", color=color, label="10-packet window"),
               Line2D([], [], marker="D", linestyle="", markerfacecolor="white", markeredgecolor=color,
                      label="100-packet window"),
               Line2D([], [], color=INK_SECONDARY, linestyle="--", label=f"all attacks ({overall:.3f})")]
    # upper right is empty: the worst types sit there at sensitivity below 0.6
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True, facecolor="white", edgecolor="none",
              framealpha=1)
    save(fig, target)


def figure_pr_roc(oof: pl.DataFrame, summary: pl.DataFrame, best: str, target) -> None:
    """PR and ROC curves of the best model: one thin line per outer fold and the pooled curve."""
    fig, (pr, roc) = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for _, f in oof.group_by("outer_fold", maintain_order=True):
        p, r, _ = precision_recall_curve(f["y"], f["score"])
        fpr, tpr, _ = roc_curve(f["y"], f["score"])
        pr.plot(r, p, color=INK_MUTED, linewidth=0.7, alpha=0.7)
        roc.plot(fpr, tpr, color=INK_MUTED, linewidth=0.7, alpha=0.7)
    p, r, _ = precision_recall_curve(oof["y"], oof["score"])
    fpr, tpr, _ = roc_curve(oof["y"], oof["score"])
    row = summary.filter(pl.col("model") == best).row(0, named=True)
    pr.plot(r, p, color=MODEL_COLOR[best], linewidth=2)
    roc.plot(fpr, tpr, color=MODEL_COLOR[best], linewidth=2)
    roc.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=0.8, linestyle=":")
    pr.set(xlabel="Recall (sensitivity)", ylabel="Precision", xlim=(0, 1.01), ylim=(0.45, 1.01))
    roc.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1.01), ylim=(0, 1.01))
    pr.set_title(f"PR — AUPRC {row['auprc_mean']:.4f} ± {row['auprc_std']:.4f}", loc="left")
    roc.set_title(f"ROC — AUC {row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}", loc="left")
    fig.suptitle(f"{MODEL_LABEL[best]}: thick = pooled out-of-fold; thin grey = each outer fold (hidden under the "
                 "pooled curve); AUC mean ± std over folds", x=0.01, ha="left", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout()
    save(fig, target)


def main() -> None:
    """Check the recorded folds and predictions, then write every Etapa 8 table, test and figure listed above."""
    cfg = load_config()
    ev = cfg["evaluation"]
    models, level, alpha = ev["models"], float(ev["confidence_level"]), float(ev["alpha"])
    n_outer = int(cfg["validation"]["outer"]["n_splits"])
    root = path(cfg, "results_dir") / "nested"
    out_dir = path(cfg, "results_dir") / "evaluation"
    folds = pl.read_csv(root / "folds.csv")
    missing = {(m, k) for m in models for k in range(n_outer)} - set(zip(folds["model"], folds["outer_fold"]))
    if missing:
        raise SystemExit(f"results/nested/folds.csv lacks {sorted(missing)}; run `make tune` first")
    unknown = {m for pair in ev["comparisons"] for m in pair} - set(models)
    if unknown:
        raise SystemExit(f"evaluation.comparisons names models outside evaluation.models: {sorted(unknown)}")
    folds = folds.filter(pl.col("model").is_in(models))
    tuned = [m for m in models if m != BASELINE]
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    with recorded_run(cfg, out_dir, "evaluate") as run:
        labels = label_table(cfg)
        oofs = {m: load_oof(root, m, n_outer, labels) for m in models}
        weights = type_weights(cfg, labels)
        # attack prevalence of the original dataset (official CSV rows), added to the configured prevalences
        dataset_prevalence = 1 - float(weights.filter(pl.col("label34") == BENIGN)["dataset_fraction"][0])

        summary = summary_table(folds, models)
        best = summary.filter(pl.col("model").is_in(tuned)).sort("mcc_mean", descending=True)["model"][0]
        confusion = (folds.group_by("model").agg(*[pl.col(f"test_{c}").sum().alias(c) for c in CELLS])
                     .with_columns((pl.col("model") == best).alias("best"),
                                   pl.col("model").replace_strict({m: i for i, m in enumerate(models)}).alias("order"))
                     .sort("order").drop("order"))
        for r in confusion.iter_rows(named=True):  # pooled predictions and per-fold cells must agree
            o = oofs[r["model"]]
            tp = int(((o["y"] == 1) & (o["pred"] == 1)).sum())
            fp = int(((o["y"] == 0) & (o["pred"] == 1)).sum())
            if (tp, fp) != (r["tp"], r["fp"]):
                raise ValueError(f"{r['model']}: folds.csv and the predictions disagree")

        y = oofs[models[0]]["y"].to_numpy()
        preds = np.vstack([oofs[m]["pred"].to_numpy() for m in models])
        strata = None
        if ev["bootstrap"]["stratify"]:
            strata = oofs[models[0]][ev["bootstrap"]["stratify"]].to_numpy()
        counts = bootstrap_counts(y, preds, strata, int(ev["bootstrap"]["n_resamples"]), int(cfg["seed"]))
        boot_mcc = mcc_from_counts(counts[..., 0], counts[..., 1], counts[..., 2], counts[..., 3])
        pooled_mcc = {m: float(mcc_from_counts(*(int(confusion.filter(pl.col("model") == m)[c][0])
                                                 for c in ("tn", "fp", "fn", "tp")))) for m in models}
        boot = pl.DataFrame([{"model": m, "mcc_pooled": pooled_mcc[m],
                              **dict(zip(("ci_low", "ci_high"), percentile_interval(boot_mcc[:, i], level)))}
                             for i, m in enumerate(models)])
        tests = comparison_table(folds, ev["comparisons"], boot_mcc, pooled_mcc, models, level, alpha)

        recall = pl.concat([recall_by_group(oofs[m], m, level) for m in tuned])
        fpr = pl.DataFrame([benign_fpr(oofs[m], m, level) for m in tuned])
        rew = reweighted(oofs, models, weights)
        by_window = metrics_by_window(oofs, tuned)
        ppv = ppv_table(oofs, tuned, weights, [float(p) for p in ev["prevalences"]], dataset_prevalence)
        hyp = hypotheses(summary, tests, confusion, recall, best, ev["h2_categories"])

        summary.write_csv(out_dir / "summary.csv")
        tests.write_csv(out_dir / "tests.csv")
        boot.write_csv(out_dir / "bootstrap_ci.csv")
        confusion.write_csv(out_dir / "confusion_matrix.csv")
        recall.write_csv(out_dir / "recall_by_group.csv")
        fpr.write_csv(out_dir / "benign_fpr.csv")
        rew.write_csv(out_dir / "reweighted.csv")
        by_window.write_csv(out_dir / "metrics_by_window.csv")
        weights.write_csv(out_dir / "type_weights.csv")
        ppv.write_csv(out_dir / "ppv_projection.csv")
        write_json(hyp, out_dir / "hypotheses.json")
        figure_mcc_per_fold(folds, models, out_dir / "fig_mcc_per_fold.png")
        figure_recall_by_type(recall, fpr.filter(pl.col("model") == best).row(0, named=True), best,
                              out_dir / "fig_recall_by_type.png")
        figure_pr_roc(oofs[best], summary, best, out_dir / "fig_pr_roc.png")

        run["best_model"] = best
        run["outputs"] = {rel(f): sha256_file(f) for f in sorted(out_dir.glob("*"))
                          if f.is_file() and not f.name.endswith(".run.json")}
    print(f"best model: {best}")
    print(summary.select("model", "mcc_mean", "mcc_std"))
    print(tests.select("model_a", "model_b", "diff_mean", "t", "p_holm", "ci_low", "ci_high"))
    print(f"H1 first half: {hyp['H1']['first_half_ensembles_have_highest_mcc']}; H2: {hyp['H2']['verdict']}")


if __name__ == "__main__":
    main()
