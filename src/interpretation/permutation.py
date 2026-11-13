"""Etapa 9 (D11): permutation importance of the best model in each outer test fold.

Each fold model (src/interpretation/fold_models.py: refitted and checked against the Etapa 7 predictions) is
scored on the test fold it never saw. Each of the 27 original attributes is permuted `interpretation.permutation_repeats`
times (Protocol Type as one column, before the one-hot encoding inside the pipeline), and the importance is the drop of
the MCC. Outputs in results/interpretation/:
- permutation_per_fold.csv: mean and std over repeats per fold and feature, and the baseline MCC of the fold;
- permutation_summary.csv: mean and std over folds, ranked;
- permutation_stability.json: Spearman correlation of the importances between pairs of folds, top-k overlap;
- fig_permutation_importance.png; permutation.run.json.

Usage: python -m src.interpretation.permutation   (make interpret)
"""
import numpy as np
import polars as pl
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, matthews_corrcoef

from src.experiments.nested_cv import load_sample_M
from src.interpretation.common import best_model, ranking_stability, summarise_by_feature
from src.interpretation.fold_models import fold_data, fold_model, models_dir
from src.evaluation.evaluate import MODEL_COLOR, MODEL_LABEL
from src.reporting.style import INK, INK_MUTED, INK_SECONDARY, apply_style, plt, save
from src.utils.config import load_config, path
from src.utils.io import rel, sha256_file, write_json
from src.utils.run_metadata import recorded_run

CANDIDATES = ("Number", "Time_To_Live")  # D12 shortcut candidates, marked in the figure


def figure(summary: pl.DataFrame, target, color: str, label: str) -> None:
    s = summary.sort("rank", descending=True)
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    y = np.arange(s.height)
    ax.errorbar(s["mean"], y, xerr=s["std"].fill_null(0), fmt="none", ecolor=INK_MUTED, elinewidth=1, zorder=2)
    ax.scatter(s["mean"], y, s=26, color=color, edgecolors="white", linewidths=1, zorder=3)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8)
    labels = [f"{f} (shortcut candidate)" if f in CANDIDATES else f for f in s["feature"]]
    ax.set_yticks(y, labels, fontsize=7.5)
    for tick, f in zip(ax.get_yticklabels(), s["feature"]):
        if f in CANDIDATES:
            tick.set_color(INK)
            tick.set_fontweight("bold")
    ax.set_xlabel("Drop in test MCC when the feature is permuted (mean ± std over the 5 outer folds)")
    ax.set_title(f"Permutation importance — {label}", loc="left")
    ax.text(0, -0.08, "10 permutations per feature and fold; Protocol Type permuted before its one-hot encoding.",
            transform=ax.transAxes, fontsize=8, color=INK_SECONDARY)
    save(fig, target)


def main() -> None:
    cfg, feature_cfg, search_cfg = load_config(), load_config("features"), load_config("search_spaces")
    icfg, seed = cfg["interpretation"], int(cfg["seed"])
    model_name = best_model(cfg)
    out_dir = path(cfg, "results_dir") / "interpretation"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    scorer = make_scorer(matthews_corrcoef)
    with recorded_run(cfg, out_dir, "permutation") as run:
        data = fold_data(cfg, feature_cfg, load_sample_M(cfg), cfg["nested"]["feature_set"])
        rows = []
        for k, (_, test) in enumerate(data.folds):
            model = fold_model(cfg, feature_cfg, search_cfg, data, model_name, k,
                               path(cfg, "results_dir") / "nested", models_dir(cfg))
            model.set_params(model__n_jobs=int(cfg["n_jobs"]))  # prediction only: same trees, more threads
            Xt, yt = data.X.iloc[test], data.y[test]
            baseline = scorer(model, Xt, yt)
            r = permutation_importance(model, Xt, yt, scoring=scorer, n_repeats=int(icfg["permutation_repeats"]),
                                       random_state=seed, n_jobs=1)
            rows += [{"fold": k, "feature": f, "baseline_mcc": baseline, "importance_mean": float(r.importances_mean[i]),
                      "importance_std": float(r.importances_std[i])} for i, f in enumerate(data.features)]
            print(f"fold {k}: baseline MCC {baseline:.4f}", flush=True)
        per_fold = pl.DataFrame(rows)
        summary = summarise_by_feature(per_fold, "importance_mean")
        stability = ranking_stability(per_fold, "importance_mean", int(icfg["top_k"]))
        per_fold.write_csv(out_dir / "permutation_per_fold.csv")
        summary.write_csv(out_dir / "permutation_summary.csv")
        write_json({"model": model_name, **stability}, out_dir / "permutation_stability.json")
        figure(summary, out_dir / "fig_permutation_importance.png", MODEL_COLOR[model_name], MODEL_LABEL[model_name])
        run["model"] = model_name
        run["outputs"] = {rel(out_dir / n): sha256_file(out_dir / n) for n in
                          ("permutation_per_fold.csv", "permutation_summary.csv", "permutation_stability.json",
                           "fig_permutation_importance.png")}
    print(summary.head(10))
    print(stability)


if __name__ == "__main__":
    main()
