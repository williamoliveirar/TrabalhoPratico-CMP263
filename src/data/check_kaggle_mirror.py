"""Cross-check the Kaggle cleaned Parquet (dhoogla/ciciotdataset2023) against the official per-capture CSVs.

Supports decision D15 (data source). Writes results/inspection/kaggle_vs_official.json with:
row counts per file, label and class; files whose counts differ; a value-level comparison of the first
rows of every file; and the magnitude of IAT and Time_To_Live per class.

Usage: python -m src.data.check_kaggle_mirror
"""
import json
from datetime import datetime, timezone

import numpy as np
import polars as pl

from src.utils.config import load_config, path

HEAD_ROWS = 1000  # rows compared per file in the value-level check
REL_TOL = 1e-6    # Parquet stores several columns as float32


def main() -> None:
    cfg = load_config()
    csv_dir = path(cfg, "raw_csv_dir")
    lf = pl.scan_parquet(path(cfg, "kaggle_parquet"))
    pq_features = [c for c in lf.collect_schema().names()
                   if c not in {"label", "category", "binary_label", "source_file", "rate_was_infinite"}]

    files = sorted(csv_dir.rglob("*.csv"))
    csv_header = pl.read_csv(files[0], n_rows=0).columns
    csv_features = [c for c in csv_header if c != "Tot size"]  # dropped by the mirror (duplicate of AVG)
    assert len(csv_features) == len(pq_features), "column sets differ; mapping by position is invalid"
    rename = dict(zip(csv_features, pq_features))

    per_file, value_check = [], []
    pq_counts = lf.group_by("source_file", "label").agg(pl.len().alias("n_parquet")).collect()
    for f in files:
        rel = f"{f.parent.name}/{f.name}"
        n_csv = pl.scan_csv(f, infer_schema=False).select(pl.len()).collect().item()
        per_file.append({"folder": f.parent.name, "source_file": rel, "n_csv": n_csv})

        a = (pl.read_csv(f, n_rows=HEAD_ROWS, columns=csv_features,
                         schema_overrides={c: pl.Float64 for c in csv_features})
             .rename(rename).to_numpy())
        b = (lf.filter(pl.col("source_file") == rel).head(HEAD_ROWS).select(pq_features)
             .cast(pl.Float64).collect().to_numpy())
        n = min(len(a), len(b))
        close = np.isclose(a[:n], b[:n], rtol=REL_TOL, atol=0.0, equal_nan=True)
        bad_cols = [pq_features[j] for j in np.where(~close.all(axis=0))[0]]
        value_check.append({"source_file": rel, "rows_compared": n,
                            "cells_differing": int((~close).sum()), "columns_differing": bad_cols})

    files_df = pl.DataFrame(per_file).join(pq_counts, on="source_file", how="full", coalesce=True)
    diff = files_df.filter(pl.col("n_csv") != pl.col("n_parquet"))
    by_label = (files_df.group_by("folder", "label")
                .agg(pl.len().alias("n_files"), pl.col("n_csv").sum(), pl.col("n_parquet").sum())
                .sort("n_csv"))
    by_class = lf.group_by("binary_label").agg(
        pl.len().alias("n_rows"),
        pl.col("inter_arrival_time").quantile(0.01).alias("iat_p01"),
        pl.col("inter_arrival_time").median().alias("iat_median"),
        pl.col("inter_arrival_time").quantile(0.99).alias("iat_p99"),
        pl.col("time_to_live").median().alias("ttl_median"),
    ).sort("binary_label").collect()

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "csv": {"n_files": len(files), "n_rows": int(files_df["n_csv"].sum()), "columns": csv_header},
        "parquet": {"n_files": pq_counts.height, "n_rows": int(files_df["n_parquet"].sum()),
                    "feature_columns": pq_features, "rename_csv_to_parquet": rename},
        "files_without_match": files_df.filter(pl.col("n_csv").is_null() | pl.col("n_parquet").is_null()).height,
        "files_with_different_counts": diff.select("source_file", "n_csv", "n_parquet").to_dicts(),
        "value_check": {
            "head_rows_per_file": HEAD_ROWS, "rel_tol": REL_TOL,
            "files_with_differences": [v for v in value_check if v["cells_differing"]],
            "total_cells_compared": sum(v["rows_compared"] for v in value_check) * len(pq_features),
            "total_cells_differing": sum(v["cells_differing"] for v in value_check),
        },
        "by_label": by_label.to_dicts(),
        "by_class": by_class.to_dicts(),
        "note": "by_class statistics use the full dataset, before the E/M split; descriptive only, "
                "no modelling decision is based on them (formal EDA runs on sample E, Etapa 3).",
    }
    target = path(cfg, "results_dir") / "inspection" / "kaggle_vs_official.json"
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
