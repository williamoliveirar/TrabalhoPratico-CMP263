"""D15: cross-check our ingestion (make ingest) against the Kaggle mirror, which was built from the same CSVs.

Compares rows per file, labels, categories and classes, and every value of every row (the mirror keeps the CSV
row order within each file). Differences are expected only where the mirror cleaned the data: infinite `Rate`
(capped there) and empty `Std`/`Variance` (0 there). Any other difference is an ingestion error.
Optional: `make all` does not depend on the mirror.

Output: results/inspection/ingest_vs_kaggle.json and check_ingest.run.json.

Usage: python -m src.data.check_ingest
"""
import numpy as np
import polars as pl

from src.data.check_kaggle_mirror import REL_TOL
from src.utils.config import load_config, path
from src.utils.io import write_json
from src.utils.run_metadata import recorded_run

KAGGLE_LABELS = ["label", "category", "binary_label"]
KAGGLE_EXTRA = {*KAGGLE_LABELS, "source_file", "rate_was_infinite"}
DROPPED_BY_MIRROR = "Tot size"  # duplicate of AVG


def main() -> None:
    """Compare the ingested Parquet with the Kaggle mirror file by file and cell by cell; write the JSON report."""
    cfg = load_config()
    cols = cfg["columns"]
    levels = [cols["label"], cols["category"], cols["target"]]
    ours = pl.scan_parquet(path(cfg, "interim_parquet"))
    kaggle = pl.scan_parquet(path(cfg, "kaggle_parquet"))
    out_dir = path(cfg, "results_dir") / "inspection"

    features = [c for c in ours.collect_schema().names()
                if c not in {*levels, *cols["provenance"], DROPPED_BY_MIRROR}]
    k_features = [c for c in kaggle.collect_schema().names() if c not in KAGGLE_EXTRA]
    if len(features) != len(k_features):
        raise ValueError("column sets differ; mapping by position is invalid")
    # the mirror uses its own column names but keeps the CSV column order, so columns are paired by position
    rename = dict(zip(features, k_features))

    with recorded_run(cfg, out_dir, "check_ingest") as run:
        files = (ours.group_by("source_file", *levels).agg(pl.len().alias("n_ours")).collect()
                 .join(kaggle.group_by("source_file", *KAGGLE_LABELS).agg(pl.len().alias("n_kaggle")).collect(),
                       on="source_file", how="full", coalesce=True)
                 .sort("source_file"))
        # ne_missing: a file present on only one side (null count) also counts as a mismatch
        count_mismatch = files.filter(pl.col("n_ours").ne_missing(pl.col("n_kaggle")))
        label_mismatch = files.filter(pl.col(cols["label"]).ne_missing(pl.col("label")))
        pairs = {
            "category8_vs_kaggle_category": files.select(cols["category"], "category").unique()
                                                 .sort(cols["category"], "category").to_dicts(),
            "y_vs_kaggle_binary_label": files.select(cols["target"], "binary_label").unique()
                                             .sort(cols["target"], "binary_label").to_dicts(),
        }

        per_column = {c: {"cells_differing": 0, "of_which_nonfinite_in_csv": 0} for c in features}
        n_rows_compared = 0
        for folder in sorted({s.split("/")[0] for s in files["source_file"]}):
            in_folder = pl.col("source_file").str.starts_with(folder + "/")
            a = ours.filter(in_folder).sort("source_file", "row_idx").select("source_file", *features).collect()
            b = (kaggle.filter(in_folder).select("source_file", *k_features).collect()
                 .sort("source_file", maintain_order=True))  # stable: keeps the CSV order within each file
            if not a["source_file"].equals(b["source_file"]):
                raise ValueError(f"{folder}: rows per file differ; values cannot be aligned")
            for c in features:
                x = a[c].to_numpy()
                y = b[rename[c]].cast(pl.Float64).to_numpy()
                differ = ~np.isclose(x, y, rtol=REL_TOL, atol=0.0, equal_nan=True)
                per_column[c]["cells_differing"] += int(differ.sum())
                # inf or empty (nan) in our data: cells the mirror capped or set to 0, so they are expected to differ
                per_column[c]["of_which_nonfinite_in_csv"] += int((differ & ~np.isfinite(x)).sum())
            n_rows_compared += a.height
            print(f"{folder}: {a.height} rows compared")

        unexplained = sum(v["cells_differing"] - v["of_which_nonfinite_in_csv"] for v in per_column.values())
        out = {
            "n_rows_ours": int(files["n_ours"].sum()),
            "n_rows_kaggle": int(files["n_kaggle"].sum()),
            "n_files_ours": files["n_ours"].is_not_null().sum(),
            "n_files_kaggle": files["n_kaggle"].is_not_null().sum(),
            "files_with_different_counts": count_mismatch.select("source_file", "n_ours", "n_kaggle").to_dicts(),
            "files_with_different_label34": label_mismatch.select("source_file", cols["label"], "label").to_dicts(),
            **pairs,
            "value_check": {
                "rows_compared": n_rows_compared,
                "cells_compared": n_rows_compared * len(features),
                "rel_tol": REL_TOL,
                "rename_ours_to_kaggle": rename,
                "per_column": {c: v for c, v in per_column.items() if v["cells_differing"]},
                "cells_differing_with_finite_csv_value": unexplained,
            },
        }
        write_json(out, out_dir / "ingest_vs_kaggle.json")
        run["unexplained_differences"] = unexplained
    print(f"wrote {out_dir / 'ingest_vs_kaggle.json'}; differences not explained by the mirror's cleaning: {unexplained}")


if __name__ == "__main__":
    main()
