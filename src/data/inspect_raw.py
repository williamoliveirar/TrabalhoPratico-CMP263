"""Etapa 1: inspection of the ingested raw data (every row, values unaltered; `make ingest` runs first).

Answers the Etapa 1 questions that the data itself can answer (PLANO §7): rows per file, label, category and
class; captures per label; column types; missing, NaN and infinite values; whether `Tot size` duplicates `AVG`;
how infinite rates and missing dispersion relate to one-packet windows; and whether every file name matches its
label. Statistics are global. Per-class feature statistics are EDA, which runs on sample E only (Etapa 3).

Outputs in results/inspection/: label_counts_by_file.csv, label_counts_total.csv, schema.json, inspect_raw.run.json.

Usage: python -m src.data.inspect_raw
"""
import polars as pl

from src.utils.config import load_config, path
from src.utils.io import rel, sha256_file, write_json
from src.utils.run_metadata import recorded_run

TOT_SIZE, AVG = "Tot size", "AVG"
NUMBER, RATE, STD, VARIANCE = "Number", "Rate", "Std", "Variance"


def column_stats(lf: pl.LazyFrame, c: str) -> dict:
    x = pl.col(c)
    finite = x.filter(x.is_finite())
    return lf.select(
        x.null_count().alias("n_null"),
        x.is_nan().sum().alias("n_nan"),
        (x == float("inf")).sum().alias("n_pos_inf"),
        (x == float("-inf")).sum().alias("n_neg_inf"),
        finite.min().alias("min_finite"),
        finite.max().alias("max_finite"),
        x.drop_nulls().n_unique().alias("n_unique"),
        (finite == finite.floor()).all().alias("integer_valued"),
    ).collect().row(0, named=True)


def label_totals(by_file: pl.DataFrame, levels: list[str]) -> pl.DataFrame:
    n_total = by_file["n_rows"].sum()
    parts = [by_file.group_by(level)
             .agg(pl.len().alias("n_files"), pl.col("n_rows").sum())
             .select(pl.lit(level).alias("level"), pl.col(level).cast(pl.String).alias("value"), "n_files", "n_rows")
             .sort(["n_rows", "value"], descending=[True, False])
             for level in levels]
    return pl.concat(parts).with_columns((pl.col("n_rows") / n_total).round(6).alias("fraction"))


def main() -> None:
    cfg = load_config()
    cols = cfg["columns"]
    levels = [cols["label"], cols["category"], cols["target"]]
    source = path(cfg, "interim_parquet")
    if not source.exists():
        raise SystemExit(f"{source} not found; run `make ingest` first")
    out_dir = path(cfg, "results_dir") / "inspection"
    lf = pl.scan_parquet(source)
    schema = lf.collect_schema()
    features = [c for c in schema.names() if c not in {*levels, *cols["provenance"]}]

    with recorded_run(cfg, out_dir, "inspect_raw") as run:
        by_file = (lf.group_by("source_file", *levels).agg(pl.len().alias("n_rows"))
                   .sort("source_file").collect())
        if by_file["source_file"].is_duplicated().any():
            raise ValueError("a capture file has more than one label")
        by_file.write_csv(out_dir / "label_counts_by_file.csv")
        label_totals(by_file, levels).write_csv(out_dir / "label_counts_total.csv")

        manifest = pl.read_csv(path(cfg, "raw_manifest"))
        file_stems = (by_file
                      .with_columns(pl.col("source_file").str.split("/").list.last()
                                    .str.replace(r"\d*\.pcap\.csv$", "").alias("file_stem"))
                      .filter(pl.col("file_stem") != pl.col(cols["label"]))
                      .select("source_file", "file_stem", cols["label"]))
        windows = (lf.group_by((pl.col(NUMBER) == 1).alias("number_is_1"),
                               pl.col(RATE).is_infinite().alias("rate_is_inf"),
                               pl.col(STD).is_null().alias("std_is_null"),
                               pl.col(VARIANCE).is_null().alias("variance_is_null"))
                   .agg(pl.len().alias("n_rows"))
                   .sort("number_is_1", "rate_is_inf", "std_is_null", "variance_is_null")
                   .collect())
        number_top = (lf.group_by(NUMBER).agg(pl.len().alias("n_rows"))
                      .sort(["n_rows", NUMBER], descending=[True, False])
                      .head(cfg["inspection"]["top_values"]).collect())
        tot_size = lf.select(
            (pl.col(TOT_SIZE) == pl.col(AVG)).sum().alias("n_equal"),
            (pl.col(TOT_SIZE) != pl.col(AVG)).sum().alias("n_different"),
            (pl.col(TOT_SIZE).is_null() | pl.col(AVG).is_null()).sum().alias("n_with_null"),
        ).collect().row(0, named=True)

        out = {
            "source": f"{rel(source)} (make ingest)",
            "n_files": by_file.height,
            "n_rows_csv": int(manifest["n_rows"].sum()),
            "n_malformed_skipped": int(manifest["n_malformed"].sum()),
            "n_rows": int(by_file["n_rows"].sum()),
            "header": {
                "n_columns": len(features),
                "same_in_every_file": manifest["n_cols"].n_unique() == 1,  # names are checked by src.data.ingest
                "columns": features,
            },
            "added_columns": [*levels, *cols["provenance"]],
            "features": [{"name": c, "dtype": str(schema[c]), **column_stats(lf, c)} for c in features],
            "checks": {
                "tot_size_vs_avg": tot_size,
                "number_rate_dispersion_crosstab": windows.to_dicts(),
                "number_top_values": number_top.to_dicts(),
                "file_name_differs_from_label": file_stems.to_dicts(),
            },
            "note": "Global statistics over every row. No per-class feature statistics: that is EDA, "
                    "on sample E only (Etapa 3).",
        }
        write_json(out, out_dir / "schema.json")
        run["outputs"] = {rel(out_dir / f): sha256_file(out_dir / f) for f in
                          ("label_counts_by_file.csv", "label_counts_total.csv", "schema.json")}
    print(f"wrote {out_dir}/label_counts_by_file.csv, label_counts_total.csv, schema.json")


if __name__ == "__main__":
    main()
