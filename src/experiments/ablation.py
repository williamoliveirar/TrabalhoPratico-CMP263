"""Etapa 9 (D12, Q3): the nested cross-validation of Etapa 7 repeated for the best model of Etapa 8 without the
shortcut candidates.

The best model is read from results/evaluation/hypotheses.json (rule pre-registered in Etapa 8). For each feature set
of `ablation.feature_sets` (configs/base.yaml; configs/features.yaml defines them), the same outer and inner folds,
search space, number of configurations and seed as Etapa 7 are used, so each outer fold is paired with the `full`
result in results/nested/. Outputs go to results/ablation/<feature_set>/<model>/ in the format of Etapa 7
(fold<k>.json, fold<k>_search.csv, fold<k>_predictions.csv.gz, fold<k>_timing.json), plus folds.csv and one run record
per invocation. Interrupted runs resume where they stopped.

Usage: python -m src.experiments.ablation [--feature-sets no_shortcut ...] [--folds 0 1 ...]
"""
import argparse
import json

from src.experiments.nested_cv import load_sample_M, run_nested
from src.utils.config import load_config, path


def best_model(cfg: dict) -> str:
    """Best model of Etapa 8, read from results/evaluation/hypotheses.json (pre-registered rule)."""
    source = path(cfg, "results_dir") / "evaluation" / "hypotheses.json"
    if not source.exists():
        raise SystemExit(f"{source} not found; run `make evaluate` first")
    return json.loads(source.read_text("utf-8"))["best_model"]


def main() -> None:
    """Nested CV of the best model for each requested ablated feature set, on the same outer folds as Etapa 7."""
    cfg, feature_cfg, search_cfg = load_config(), load_config("features"), load_config("search_spaces")
    sets, n_outer = cfg["ablation"]["feature_sets"], int(cfg["validation"]["outer"]["n_splits"])
    parser = argparse.ArgumentParser(description="Etapa 9: shortcut ablation (nested CV without the candidates)")
    parser.add_argument("--feature-sets", nargs="+", default=sets, choices=sets)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(n_outer)), choices=range(n_outer))
    args = parser.parse_args()
    model, M = best_model(cfg), load_sample_M(cfg)
    # a partial run (subset of folds) gets its own run record, so each resumed part keeps its provenance
    scope = [] if args.folds == list(range(n_outer)) else [f"fold{k}" for k in args.folds]
    for feature_set in args.feature_sets:
        root = path(cfg, "results_dir") / "ablation" / feature_set
        run_nested(cfg, feature_cfg, search_cfg, M, feature_set, root, [model], [model], args.folds,
                   "_".join(["ablation", *scope]))


if __name__ == "__main__":
    main()
