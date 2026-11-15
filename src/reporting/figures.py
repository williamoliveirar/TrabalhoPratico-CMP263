"""Etapa 10 — index of the figures (make report).

Every figure is drawn by the script of its own stage; nothing is redrawn or copied here. This step finds each PNG
under results/, checks that its SHA-256 matches the outputs recorded in a run.json (so it came from a recorded run),
and writes results/figures/index.csv: file, title, script, make target and SHA-256 (the commit of each run is in its
run.json). A figure with no matching run record stops the step.

Usage: python -m src.reporting.figures   (make report)
"""
import json
import re
from pathlib import Path

import polars as pl

from src.utils.config import load_config, path
from src.utils.io import ROOT, rel, sha256_file
from src.utils.run_metadata import recorded_run

TITLES = {
    "eda/fig_number_by_type.png": "Number (packets per window) by attack type, sample E",
    "eda/fig_by_type.png": "Attributes by attack type, sample E",
    "eda/fig_distributions.png": "Distributions of the attributes by class, sample E",
    "eda/fig_single_feature_mcc.png": "MCC of one-attribute trees, sample E",
    "eda/fig_spearman.png": "Spearman correlation between attributes, sample E",
    "spotcheck/fig_spotcheck_mcc.png": "Spot-checking: validation MCC per algorithm, sample E",
    "evaluation/fig_mcc_per_fold.png": "MCC per outer test fold, sample M",
    "evaluation/fig_recall_by_type.png": "Sensitivity per attack type, best model",
    "evaluation/fig_pr_roc.png": "Precision-recall and ROC curves, best model",
    "interpretation/fig_permutation_importance.png": "Permutation importance, best model",
    "interpretation/fig_shap_summary.png": "SHAP summary, best model",
    "interpretation/fig_shap_dependence.png": "SHAP dependence of Number, Time_To_Live and IAT",
    "interpretation/fig_shap_local.png": "Local SHAP explanations of typical errors",
    "shortcuts/fig_ablation_recall_by_type.png": "Shortcut ablation: sensitivity per attack type",
    "capture/fig_capture_recall.png": "Leave-capture-out: sensitivity per attack type",
}


def make_targets(makefile: Path) -> dict[str, str]:
    """Module -> make target, read from the Makefile recipes."""
    targets, current = {}, None
    for line in makefile.read_text(encoding="utf-8").splitlines():
        if m := re.match(r"^([a-z][\w-]*):", line):
            current = m.group(1)
        elif current and (m := re.search(r"-m\s+([\w.]+)", line)):  # first target that runs a module keeps it
            targets.setdefault(m.group(1), current)
    return targets


def figure_index(results: Path, makefile: Path) -> pl.DataFrame:
    """One row per PNG under results/ with its title, script, make target and SHA-256; stops at a PNG whose current
    hash no run.json records."""
    recorded = {}  # (file, sha256) -> run record
    for record in sorted(results.rglob("*.run.json")):
        d = json.loads(record.read_text(encoding="utf-8"))
        for f, h in d.get("outputs", {}).items():
            if f.endswith(".png"):
                recorded[(f, h)] = d
    targets, rows = make_targets(makefile), []
    for fig in sorted(results.rglob("*.png")):
        name, digest = rel(fig), sha256_file(fig)
        run = recorded.get((name, digest))
        if run is None:
            raise SystemExit(f"{name}: no run.json records this file with its current hash")
        key = fig.relative_to(results).as_posix()
        rows.append({"figure": name, "title": TITLES.get(key, ""), "script": run["script"],
                     "make_target": targets.get(run["script"], ""), "sha256": digest})
    return pl.DataFrame(rows)


def main() -> None:
    """Write results/figures/index.csv inside a recorded run."""
    cfg = load_config()
    results = path(cfg, "results_dir")
    out_dir = results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    with recorded_run(cfg, out_dir, "figures") as run:
        index = figure_index(results, ROOT / "Makefile")
        index.write_csv(out_dir / "index.csv")
        run["outputs"] = {rel(out_dir / "index.csv"): sha256_file(out_dir / "index.csv")}
        print(f"{index.height} figures indexed")


if __name__ == "__main__":
    main()
