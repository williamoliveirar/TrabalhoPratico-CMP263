"""Etapa 9 (D11): SHAP explanations of the best model, global and local, on the outer test folds.

Global: `interpretation.shap_rows_per_fold` rows drawn at random from each outer test fold are explained with
shap.TreeExplainer on the log-odds scale by the model of that fold (src/interpretation/fold_models.py). The values of
the one-hot columns of a categorical attribute are summed into one value for the attribute. Additivity (base value +
sum of SHAP values = log-odds of the model) is checked for every row.
Local: the errors chosen by the rule pre-registered on 2026-11-12: one typical false negative of each of the
`interpretation.local.worst_types` attack types with the lowest sensitivity in Etapa 8, and
`interpretation.local.false_positives` typical false positives on benign traffic ("typical" = score closest to the
median score of those errors; ties by position).

Outputs in results/interpretation/:
- shap_values.csv.gz: SHAP value of each attribute for the sampled rows, with position, fold, class, label, base
  value and the values of the dependence features;
- shap_importance_per_fold.csv, shap_summary.csv: mean |SHAP| per fold and over folds, ranked;
- shap_stability.json: ranking stability between folds;
- local_explanations.csv: the chosen errors, with every attribute's value and SHAP value;
- fig_shap_summary.png, fig_shap_dependence.png, fig_shap_local.png; shap_analysis.run.json.

Usage: python -m src.interpretation.shap_analysis   (make interpret)
"""
import numpy as np
import polars as pl
import shap
from scipy import stats

from src.data.labels import BENIGN
from src.evaluation.evaluate import MODEL_LABEL
from src.experiments.nested_cv import load_sample_M
from src.interpretation.common import best_model, original_feature, ranking_stability, summarise_by_feature, typical
from src.interpretation.fold_models import fold_data, fold_model, models_dir
from src.reporting.style import (ATTACK_COLOR, BENIGN_COLOR, DIVERGING, INK_MUTED, INK_SECONDARY, apply_style, plt,
                                 save)
from src.utils.config import load_config, path
from src.utils.io import rel, sha256_file, write_csv_gz, write_json
from src.utils.run_metadata import recorded_run

ADDITIVITY_TOLERANCE = 1e-6
SUMMARY_FEATURES = 15  # rows of the summary (beeswarm) figure
LOCAL_FEATURES = 8     # bars per local explanation; the rest are summed


def explain(model, X, categorical: list[str]) -> tuple[np.ndarray, float, list[str]]:
    """SHAP values per original attribute (one-hot columns summed), the base value, and the attribute names."""
    pre, est = model[:-1], model[-1]
    Xt = pre.transform(X)
    explainer = shap.TreeExplainer(est)
    sv = np.asarray(explainer.shap_values(Xt))
    if sv.ndim == 3:  # (rows, columns, classes) in some shap versions: keep the attack class
        sv = sv[..., -1]
    base = float(np.atleast_1d(explainer.expected_value)[-1])
    error = np.abs(base + sv.sum(axis=1) - est.predict(Xt, raw_score=True)).max()
    if error > ADDITIVITY_TOLERANCE:
        raise RuntimeError(f"SHAP values do not add up to the model output (max error {error:.2e})")
    names = [original_feature(n, categorical) for n in pre.get_feature_names_out()]
    attributes = list(dict.fromkeys(names))
    summed = np.column_stack([sv[:, [i for i, n in enumerate(names) if n == a]].sum(axis=1) for a in attributes])
    return summed, base, attributes


def figure_summary(values: pl.DataFrame, summary: pl.DataFrame, data_rows, target, seed: int, label: str) -> None:
    """Beeswarm-like summary: one row per attribute (top by mean |SHAP|), colour = feature value rank (low → high)."""
    top = summary["feature"].head(SUMMARY_FEATURES).to_list()
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    for i, f in enumerate(top):
        y = len(top) - 1 - i
        s = values[f"shap_{f}"].to_numpy()
        v = data_rows[f].to_numpy()
        colour = stats.rankdata(v) / len(v)
        ax.scatter(s, y + rng.uniform(-0.3, 0.3, s.size), c=colour, cmap=DIVERGING, vmin=0, vmax=1, s=2, alpha=0.5,
                   linewidths=0, rasterized=True)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8)
    ax.set_yticks(range(len(top)), top[::-1], fontsize=8)
    ax.set_xlabel("SHAP value (contribution to the log-odds of attack)")
    ax.set_title(f"SHAP summary — {label}, {values.height // values['outer_fold'].n_unique():,} test rows per "
                 "outer fold", loc="left")
    sm = plt.cm.ScalarMappable(cmap=DIVERGING)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 1])
    cbar.ax.set_yticklabels(["low", "high"], fontsize=8)
    cbar.set_label("feature value (rank)", fontsize=8, color=INK_SECONDARY)
    save(fig, target)


def figure_dependence(values: pl.DataFrame, data_rows, features: list[str], target, label: str) -> None:
    fig, axes = plt.subplots(1, len(features), figsize=(3.0 * len(features), 3.2), sharey=True)
    y = values["y"].to_numpy()
    for ax, f in zip(np.atleast_1d(axes), features):
        x, s = data_rows[f].to_numpy(), values[f"shap_{f}"].to_numpy()
        for cls, colour, label in ((0, BENIGN_COLOR, "benign"), (1, ATTACK_COLOR, "attack")):
            ax.scatter(x[y == cls], s[y == cls], s=4, alpha=0.35, color=colour, linewidths=0, label=label,
                       rasterized=True)
        if f == "IAT":
            ax.set_xscale("symlog", linthresh=1e-5)
        ax.axhline(0, color=INK_MUTED, linewidth=0.8)
        ax.set_xlabel(f)
    np.atleast_1d(axes)[0].set_ylabel("SHAP value (log-odds of attack)")
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8, markerscale=3)
    fig.suptitle(f"SHAP dependence of the shortcut candidates and IAT — {label}", x=0.01, ha="left", fontsize=10,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, target)


def figure_local(local: pl.DataFrame, target) -> None:
    cases = local.select("case", "kind", "label34", "score", "base_value").unique("case").sort("case")
    fig, axes = plt.subplots(len(cases), 1, figsize=(6.4, 2.0 * len(cases)))
    for ax, case in zip(np.atleast_1d(axes), cases.iter_rows(named=True)):
        rows = local.filter(pl.col("case") == case["case"]).sort(pl.col("shap").abs(), descending=True)
        top, rest = rows.head(LOCAL_FEATURES), rows.slice(LOCAL_FEATURES)
        names = [f"{r['feature']} = {r['value']:.4g}" for r in top.iter_rows(named=True)] + ["other features"]
        vals = [*top["shap"].to_list(), float(rest["shap"].sum())]
        y = np.arange(len(vals))[::-1]
        ax.barh(y, vals, color=[ATTACK_COLOR if v > 0 else BENIGN_COLOR for v in vals], height=0.7)
        ax.axvline(0, color=INK_MUTED, linewidth=0.8)
        ax.set_yticks(y, names, fontsize=7)
        ax.set_title(f"{case['kind']} — {case['label34']} (P(attack) {case['score']:.3f}; base log-odds "
                     f"{case['base_value']:.2f})", loc="left", fontsize=8.5)
    np.atleast_1d(axes)[-1].set_xlabel("SHAP value (log-odds of attack): orange pushes to attack, blue to benign")
    fig.tight_layout()
    save(fig, target)


def main() -> None:
    cfg, feature_cfg, search_cfg = load_config(), load_config("features"), load_config("search_spaces")
    icfg, seed = cfg["interpretation"], int(cfg["seed"])
    model_name = best_model(cfg)
    results, out_dir = path(cfg, "results_dir"), path(cfg, "results_dir") / "interpretation"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    categorical = feature_cfg["categorical"]
    rng = np.random.default_rng(seed)
    with recorded_run(cfg, out_dir, "shap_analysis") as run:
        data = fold_data(cfg, feature_cfg, load_sample_M(cfg), cfg["nested"]["feature_set"])
        models = [fold_model(cfg, feature_cfg, search_cfg, data, model_name, k, results / "nested", models_dir(cfg))
                  for k in range(len(data.folds))]
        parts, per_fold = [], []
        for k, (_, test) in enumerate(data.folds):
            rows = np.sort(rng.choice(test, size=int(icfg["shap_rows_per_fold"]), replace=False))
            sv, base, attributes = explain(models[k], data.X.iloc[rows], categorical)
            frame = pl.DataFrame({"position": rows, "outer_fold": k, "y": data.y[rows], "label34": data.labels[rows],
                                  "base_value": base})
            frame = frame.with_columns([pl.Series(f"shap_{a}", sv[:, i]) for i, a in enumerate(attributes)])
            frame = frame.with_columns([pl.Series(f"value_{f}", data.X.iloc[rows][f].to_numpy())
                                        for f in icfg["dependence_features"]])
            parts.append(frame)
            per_fold += [{"fold": k, "feature": a, "mean_abs_shap": float(np.abs(sv[:, i]).mean())}
                         for i, a in enumerate(attributes)]
            print(f"fold {k}: {rows.size} rows explained", flush=True)
        values = pl.concat(parts)
        per_fold = pl.DataFrame(per_fold)
        summary = summarise_by_feature(per_fold, "mean_abs_shap")
        stability = ranking_stability(per_fold, "mean_abs_shap", int(icfg["top_k"]))

        # local explanations: typical errors chosen by the pre-registered rule
        recall = pl.read_csv(results / "evaluation" / "recall_by_group.csv")
        worst = (recall.filter((pl.col("model") == model_name) & (pl.col("level") == "label34"))
                 .sort("sensitivity", "group")["group"].head(int(icfg["local"]["worst_types"])).to_list())
        oof = pl.concat([pl.read_csv(results / "nested" / model_name / f"fold{k}_predictions.csv.gz")
                         .with_columns(pl.lit(k).alias("outer_fold")) for k in range(len(data.folds))])
        oof = oof.with_columns(pl.Series("label34", data.labels[oof["position"].to_numpy()]))
        chosen = [typical(oof.filter((pl.col("label34") == t) & (pl.col("y") == 1) & (pl.col("pred") == 0)), 1)
                  .with_columns(pl.lit("false negative").alias("kind")) for t in worst]
        chosen.append(typical(oof.filter((pl.col("label34") == BENIGN) & (pl.col("pred") == 1)),
                              int(icfg["local"]["false_positives"])).with_columns(pl.lit("false positive").alias("kind")))
        chosen = pl.concat(chosen).with_row_index("case", offset=1)
        local = []
        for c in chosen.iter_rows(named=True):
            row = data.X.iloc[[c["position"]]]
            sv, base, attributes = explain(models[c["outer_fold"]], row, categorical)
            local += [{"case": c["case"], "kind": c["kind"], "label34": c["label34"], "position": c["position"],
                       "outer_fold": c["outer_fold"], "score": c["score"], "base_value": base, "feature": a,
                       "value": float(row[a].iloc[0]), "shap": float(sv[0, i])} for i, a in enumerate(attributes)]
        local = pl.DataFrame(local)

        write_csv_gz(values, out_dir / "shap_values.csv.gz")
        per_fold.write_csv(out_dir / "shap_importance_per_fold.csv")
        summary.write_csv(out_dir / "shap_summary.csv")
        write_json({"model": model_name, "rows_per_fold": int(icfg["shap_rows_per_fold"]), **stability},
                   out_dir / "shap_stability.json")
        local.write_csv(out_dir / "local_explanations.csv")
        data_rows = data.X.iloc[values["position"].to_numpy()]
        figure_summary(values, summary, data_rows, out_dir / "fig_shap_summary.png", seed, MODEL_LABEL[model_name])
        figure_dependence(values, data_rows, icfg["dependence_features"], out_dir / "fig_shap_dependence.png",
                          MODEL_LABEL[model_name])
        figure_local(local, out_dir / "fig_shap_local.png")
        names = ("shap_values.csv.gz", "shap_importance_per_fold.csv", "shap_summary.csv", "shap_stability.json",
                 "local_explanations.csv", "fig_shap_summary.png", "fig_shap_dependence.png", "fig_shap_local.png")
        run["model"] = model_name
        run["outputs"] = {rel(out_dir / n): sha256_file(out_dir / n) for n in names}
    print(summary.head(10))
    print(stability)


if __name__ == "__main__":
    main()
