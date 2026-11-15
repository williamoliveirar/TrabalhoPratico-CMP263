"""Etapa 9 synthesis (Q3, H3, D12): one-feature models on M and the comparison of the shortcut ablation.

Every rule here was pre-registered on 2026-11-12 (diary, "Etapa 9 — Pré-registro da análise"; configs/base.yaml,
`ablation`):
- one-feature trees (depth `eda.single_feature_depth`) on Number, Time_To_Live and IAT, trained on each outer
  training set of M and tested on the outer test fold, compared with the same models on E (Etapa 3);
- the best model with `ablation.reference` (results/nested) against each ablated feature set (results/ablation):
  corrected resampled t-test on the outer-fold MCC, Holm over the comparisons available, paired bootstrap interval of
  the pooled MCC difference stratified by label34; a significant drop is small, moderate or large by
  `ablation.mcc_drop`; per-window metrics; per-type sensitivity, with the collapse rule `ablation.type_collapse`;
- the answer to Q3 and the verdict on H3, plus the agreement between the permutation and SHAP rankings.

Outputs in results/shortcuts/: single_feature_M.csv, single_feature_summary.csv, ablation_tests.csv,
ablation_by_window.csv, ablation_recall_by_type.csv, q3_h3.json, fig_ablation_recall_by_type.png, shortcuts.run.json.

Usage: python -m src.experiments.shortcuts   (make ablation, after make interpret)
"""
import json

import numpy as np
import polars as pl
from scipy import stats
from sklearn.tree import DecisionTreeClassifier

from src.evaluation.evaluate import MODEL_COLOR, MODEL_LABEL, label_table, load_oof, metrics_by_window, recall_by_group
from src.evaluation.metrics import binary_metrics, predictions_and_scores
from src.evaluation.stats import bootstrap_counts, corrected_resampled_ttest, holm, mcc_from_counts, percentile_interval
from src.experiments.make_splits import NESTED, load_assignment, outer_folds
from src.experiments.nested_cv import load_sample_M
from src.interpretation.common import best_model
from src.models.pipelines import auxiliary_columns, feature_names, prepare
from src.reporting.style import INK_MUTED, INK_SECONDARY, apply_style, plt, save
from src.utils.config import load_config, path
from src.utils.io import rel, sha256_file, write_json
from src.utils.run_metadata import recorded_run

TTL = "Time_To_Live"
# H3, first part: the pattern of E (Etapa 3) is repeated on M if the TTL-only model has an MCC within 0.02 of E's,
# a high sensitivity on 100-packet attacks and a false positive rate of at least 0.20 (pre-registered); "high" is
# made operational as at least 0.90 (E: 0.955), fixed here before this script first ran.


def single_feature_models(cfg: dict, feature_cfg: dict) -> pl.DataFrame:
    """One-feature trees on M: per outer fold, MCC and the other metrics, sensitivity per window group and FPR."""
    M = load_sample_M(cfg)
    features = feature_names(M.columns, feature_cfg, "full", auxiliary_columns(cfg))
    X, y = prepare(M, features, feature_cfg), M[cfg["columns"]["target"]].to_numpy()
    windows = label_table(cfg)["window"].to_numpy()
    rows = []
    for f in cfg["ablation"]["single_features"]:
        for k, (train, test) in enumerate(outer_folds(load_assignment(cfg, NESTED, M))):
            tree = DecisionTreeClassifier(max_depth=int(cfg["eda"]["single_feature_depth"]),
                                          random_state=int(cfg["seed"])).fit(X[[f]].iloc[train], y[train])
            pred, score = predictions_and_scores(tree, X[[f]].iloc[test])
            yt, wt = y[test], windows[test]
            rows.append({"feature": f, "fold": k, **binary_metrics(yt, pred, score),
                         "tpr_window10": float(pred[(yt == 1) & (wt == 10)].mean()),
                         "tpr_window100": float(pred[(yt == 1) & (wt == 100)].mean())})
    return pl.DataFrame(rows)


def ablation_tests(cfg: dict, model: str, sets: list[str], oofs: dict, folds: dict, level: float, alpha: float,
                   ) -> pl.DataFrame:
    """Reference vs each ablated set: corrected t-test (Holm), paired bootstrap of the MCC drop, and its reading."""
    ref = cfg["ablation"]["reference"]
    names = [ref, *sets]
    y = oofs[ref]["y"].to_numpy()
    preds = np.vstack([oofs[n]["pred"].to_numpy() for n in names])
    strata = oofs[ref][cfg["evaluation"]["bootstrap"]["stratify"]].to_numpy() \
        if cfg["evaluation"]["bootstrap"]["stratify"] else None
    counts = bootstrap_counts(y, preds, strata, int(cfg["evaluation"]["bootstrap"]["n_resamples"]), int(cfg["seed"]))
    boot = mcc_from_counts(counts[..., 0], counts[..., 1], counts[..., 2], counts[..., 3])
    pooled = {n: float(mcc_from_counts(*(int(((oofs[n]["y"] == a) & (oofs[n]["pred"] == b)).sum())
                                         for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))))) for n in names}
    rows = []
    for i, s in enumerate(sets, start=1):
        a = folds[ref].sort("outer_fold")
        b = folds[s].sort("outer_fold")
        t, p = corrected_resampled_ttest(a["test_mcc"], b["test_mcc"], int(a["n_train"][0]), int(a["n_test"][0]))
        low, high = percentile_interval(boot[:, 0] - boot[:, i], level)
        d = a["test_mcc"].to_numpy() - b["test_mcc"].to_numpy()
        rows.append({"model": model, "reference": ref, "feature_set": s, "mcc_reference": float(a["test_mcc"].mean()),
                     "mcc_ablated": float(b["test_mcc"].mean()), "drop_mean": float(d.mean()),
                     "drop_std": float(d.std(ddof=1)), "folds_with_drop": int((d > 0).sum()), "t": t, "p": p,
                     "drop_pooled": pooled[ref] - pooled[s], "ci_low": low, "ci_high": high})
    table = pl.DataFrame(rows)
    p_holm = holm(table["p"].to_numpy())
    small, large = (float(cfg["ablation"]["mcc_drop"][k]) for k in ("small", "large"))
    table = table.with_columns(pl.Series("p_holm", p_holm))
    significant = (pl.col("p_holm") < alpha) & (pl.col("ci_low") > 0) & (pl.col("drop_mean") > 0)
    size = (pl.when(pl.col("drop_mean") >= large).then(pl.lit("large"))
            .when(pl.col("drop_mean") >= small).then(pl.lit("moderate")).otherwise(pl.lit("small")))
    return table.with_columns(significant.alias("significant_drop"),
                              pl.when(significant).then(size).otherwise(pl.lit("not significant")).alias("reading"))


def recall_comparison(oofs: dict, ref: str, sets: list[str], level: float, collapse: float) -> pl.DataFrame:
    """Per-type sensitivity of the reference and of each ablated set, with the pre-registered collapse rule."""
    base = recall_by_group(oofs[ref], ref, level).filter(pl.col("level") == "label34").select(
        pl.col("group").alias("label34"), "category8", "window", "n", "sensitivity", "ci_low", "ci_high")
    out = []
    for s in sets:
        other = recall_by_group(oofs[s], s, level).filter(pl.col("level") == "label34").select(
            pl.col("group").alias("label34"), pl.col("sensitivity").alias("sensitivity_ablated"),
            pl.col("ci_low").alias("ci_low_ablated"), pl.col("ci_high").alias("ci_high_ablated"))
        out.append(base.join(other, on="label34").with_columns(
            pl.lit(s).alias("feature_set"), (pl.col("sensitivity") - pl.col("sensitivity_ablated")).alias("drop"))
            .with_columns(((pl.col("drop") >= collapse) & (pl.col("ci_high_ablated") < pl.col("ci_low")))
                          .alias("collapses")))
    return pl.concat(out).sort("feature_set", "window", "sensitivity", "label34")


def figure(recall: pl.DataFrame, ablated: str, model: str, target) -> None:
    """Central figure: per-type sensitivity with all features and without the shortcut candidates."""
    r = recall.filter(pl.col("feature_set") == ablated).sort(["window", "sensitivity"], descending=[True, False])
    fig, ax = plt.subplots(figsize=(6.6, 8.4))
    y = np.arange(r.height)[::-1]
    colour = MODEL_COLOR[model]
    for yi, a, b in zip(y, r["sensitivity"], r["sensitivity_ablated"]):
        ax.plot([a, b], [yi, yi], color=INK_MUTED, linewidth=1, zorder=1)
    ax.scatter(r["sensitivity"], y, s=32, color=colour, edgecolors="white", linewidths=1, zorder=3,
               label="all 27 features (full)")
    ax.scatter(r["sensitivity_ablated"], y, s=32, facecolors="white", edgecolors=colour, linewidths=1.5, zorder=3,
               label=f"without Number and Time_To_Live ({ablated})")
    names = [f"{n} *" if c else n for n, c in zip(r["label34"], r["collapses"])]
    ax.set_yticks(y, names, fontsize=7.5)
    split = int((r["window"] == 100).sum())
    ax.axhline(y[split - 1] - 0.5 if split else 0, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
    ax.text(0.01, y[0] + 0.6, "100-packet window", fontsize=8, color=INK_SECONDARY)
    if split < r.height:
        ax.text(0.01, y[split] + 0.6, "10-packet window", fontsize=8, color=INK_SECONDARY)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Sensitivity on the out-of-fold predictions (* = collapse by the pre-registered rule)")
    ax.set_title(f"Shortcut ablation — {MODEL_LABEL[model]}, per attack type", loc="left")
    ax.legend(loc="lower left", fontsize=8, frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    save(fig, target)


def main() -> None:
    cfg, feature_cfg = load_config(), load_config("features")
    acfg = cfg["ablation"]
    level, alpha = float(cfg["evaluation"]["confidence_level"]), float(cfg["evaluation"]["alpha"])
    model, results = best_model(cfg), path(cfg, "results_dir")
    n_outer = int(cfg["validation"]["outer"]["n_splits"])
    ref, ablated = acfg["reference"], [s for s in acfg["feature_sets"]
                                       if (results / "ablation" / s / "folds.csv").exists()]
    if acfg["feature_sets"][0] not in ablated:
        raise SystemExit(f"results/ablation/{acfg['feature_sets'][0]}/folds.csv not found; run the ablation first")
    out_dir = results / "shortcuts"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    with recorded_run(cfg, out_dir, "shortcuts") as run:
        labels = label_table(cfg)
        roots = {ref: results / "nested", **{s: results / "ablation" / s for s in ablated}}
        oofs = {n: load_oof(r, model, n_outer, labels) for n, r in roots.items()}
        folds = {n: pl.read_csv(r / "folds.csv").filter(pl.col("model") == model) for n, r in roots.items()}

        single = single_feature_models(cfg, feature_cfg)
        single_summary = (single.group_by("feature").agg(
            pl.col("mcc").mean().alias("mcc_mean"), pl.col("mcc").std().alias("mcc_std"),
            pl.col("tpr_window10").mean(), pl.col("tpr_window100").mean(), pl.col("fpr").mean())
            .join(pl.read_csv(results / "eda" / "single_feature_models.csv").select(
                "feature", pl.col("mcc_mean").alias("mcc_E"), pl.col("tpr_window10").alias("tpr_window10_E"),
                pl.col("tpr_window100").alias("tpr_window100_E"), pl.col("fpr").alias("fpr_E")), on="feature")
            .sort("mcc_mean", descending=True))
        tests = ablation_tests(cfg, model, ablated, oofs, folds, level, alpha)
        by_window = metrics_by_window(oofs, [ref, *ablated]).rename({"model": "feature_set"})
        recall = recall_comparison(oofs, ref, ablated, level, float(acfg["type_collapse"]))

        primary = tests.row(0, named=True)
        collapsed = recall.filter((pl.col("feature_set") == primary["feature_set"]) & pl.col("collapses"))
        depends = (primary["significant_drop"] and primary["reading"] in ("moderate", "large")) or collapsed.height > 0
        ttl = single_summary.filter(pl.col("feature") == TTL).row(0, named=True)
        h3 = acfg["h3"]
        repeats = (abs(ttl["mcc_mean"] - ttl["mcc_E"]) <= float(h3["mcc_tolerance"])
                   and ttl["tpr_window100"] >= float(h3["tpr_window100_min"]) and ttl["fpr"] >= float(h3["fpr_min"]))
        no_ttl = tests.filter(pl.col("feature_set") == "no_ttl")
        interp = results / "interpretation"
        perm, shap_rank = (pl.read_csv(interp / f"{n}_summary.csv").select("feature", pl.col("mean").alias(n))
                           for n in ("permutation", "shap"))
        agreement = perm.join(shap_rank, on="feature")
        verdicts = {
            "model": model,
            "Q3": {"answer": "performance depends on the shortcut candidates" if depends
                   else "performance is robust to the shortcut candidates",
                   "primary_comparison": primary, "collapsed_types": collapsed["label34"].to_list()},
            "H3": {"first_part_ttl_alone_separates_much_traffic":
                   "partially confirmed (pattern of Etapa 3 repeated on M)" if repeats
                   else "pattern of Etapa 3 not repeated on M (see the diary)",
                   "ttl_only_on_M": ttl,
                   "second_part_reflects_attacker_os": "not testable with these data (no OS metadata); hypothesis",
                   "evidence_no_ttl": no_ttl.row(0, named=True) if no_ttl.height else None},
            "interpretation_agreement": {
                "spearman_permutation_vs_shap": float(stats.spearmanr(agreement["permutation"],
                                                                      agreement["shap"]).statistic),
                "features": agreement.height}}

        single.write_csv(out_dir / "single_feature_M.csv")
        single_summary.write_csv(out_dir / "single_feature_summary.csv")
        tests.write_csv(out_dir / "ablation_tests.csv")
        by_window.write_csv(out_dir / "ablation_by_window.csv")
        recall.write_csv(out_dir / "ablation_recall_by_type.csv")
        write_json(verdicts, out_dir / "q3_h3.json")
        figure(recall, ablated[0], model, out_dir / "fig_ablation_recall_by_type.png")
        names = ("single_feature_M.csv", "single_feature_summary.csv", "ablation_tests.csv", "ablation_by_window.csv",
                 "ablation_recall_by_type.csv", "q3_h3.json", "fig_ablation_recall_by_type.png")
        run["feature_sets"] = ablated
        run["outputs"] = {rel(out_dir / n): sha256_file(out_dir / n) for n in names}
    print(tests.select("feature_set", "drop_mean", "p_holm", "ci_low", "ci_high", "reading"))
    print(json.dumps({"Q3": verdicts["Q3"]["answer"], "H3": verdicts["H3"]["first_part_ttl_alone_separates_much_traffic"]},
                     indent=1))


if __name__ == "__main__":
    main()
