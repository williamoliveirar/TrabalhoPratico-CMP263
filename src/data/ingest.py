"""Etapa 1 / D15: convert the official per-capture CSVs into one typed Parquet without altering any value.

Each CSV is read once: its SHA-256 is computed on the same bytes that are parsed. Every feature is stored as
float64 exactly as parsed ("inf" stays inf, empty cells stay null). Rows whose field count differs from the
header are skipped and logged with their original position; nothing else is dropped. Added columns: label34,
category8, y (src/data/labels.py), source_file (`<folder>/<file>`) and row_idx (0-based data row in the CSV).
Cleaning (inf, empty cells, redundant columns) is a preprocessing decision left to Etapas 3-4.

Outputs: the interim Parquet (one row group per capture, files in alphabetical order), data/raw_manifest.csv,
results/inspection/malformed_rows.csv and results/inspection/ingest.run.json.

Usage: python -m src.data.ingest
"""
import hashlib
import io
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

from src.data.labels import CATEGORY, label_of_folder, target_of
from src.utils.config import load_config, path
from src.utils.io import rel, sha256_file
from src.utils.run_metadata import recorded_run


def capture_files(csv_dir: Path) -> list[Path]:
    """CSV files sorted by their relative POSIX path, so the order is the same on every OS (PLANO §5.2)."""
    return sorted(csv_dir.rglob("*.csv"), key=lambda f: f.relative_to(csv_dir).as_posix())


def read_capture(data: bytes, header: list[str]) -> tuple[pa.Table, np.ndarray, list[dict]]:
    """Parse one CSV. Returns the valid rows, their row_idx and the skipped (malformed) rows."""
    malformed = []

    def skip(row):  # with use_threads=False, row.number is the physical line (1 = header)
        if row.number is None:
            raise ValueError("malformed row without a line number")
        malformed.append({"row_idx": row.number - 2, "n_fields": row.actual_columns, "text": row.text})
        return "skip"

    table = pv.read_csv(
        io.BytesIO(data),
        read_options=pv.ReadOptions(use_threads=False),
        parse_options=pv.ParseOptions(invalid_row_handler=skip),
        # only empty cells become null; "inf" and "nan" are parsed as floats, not as missing values
        convert_options=pv.ConvertOptions(column_types={c: pa.float64() for c in header},
                                          null_values=[""], strings_can_be_null=False),
    )
    if table.column_names != header or any(t != pa.float64() for t in table.schema.types):
        raise ValueError("unexpected columns or types after parsing")

    n_lines = data.count(b"\n") + (not data.endswith(b"\n")) - 1  # data lines, header excluded
    if table.num_rows + len(malformed) != n_lines:
        raise ValueError(f"{table.num_rows} rows + {len(malformed)} malformed != {n_lines} data lines "
                         "(blank or multi-line records?)")
    keep = np.ones(n_lines, dtype=bool)
    keep[[m["row_idx"] for m in malformed]] = False
    return table, np.flatnonzero(keep), malformed


def main() -> None:
    cfg = load_config()
    cols = cfg["columns"]
    csv_dir = path(cfg, "raw_csv_dir")
    target = path(cfg, "interim_parquet")
    manifest_path = path(cfg, "raw_manifest")
    out_dir = path(cfg, "results_dir") / "inspection"
    files = capture_files(csv_dir)
    if not files:
        raise SystemExit(f"no CSV files under {csv_dir}")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    manifest, malformed_rows, header, writer = [], [], None, None
    with recorded_run(cfg, out_dir, "ingest") as run:
        try:
            for f in files:
                source = f.relative_to(csv_dir).as_posix()
                data = f.read_bytes()
                file_header = data.split(b"\n", 1)[0].decode("utf-8").rstrip("\r").split(",")
                header = header or file_header
                if file_header != header:
                    raise ValueError(f"{source}: header differs from the first file")
                table, row_idx, malformed = read_capture(data, header)

                label = label_of_folder(f.parent.name)
                n = table.num_rows
                table = (table
                         .append_column(cols["label"], pa.array([label] * n, pa.string()))
                         .append_column(cols["category"], pa.array([CATEGORY[label]] * n, pa.string()))
                         .append_column(cols["target"], pa.array(np.full(n, target_of(label), np.int8)))
                         .append_column("source_file", pa.array([source] * n, pa.string()))
                         .append_column("row_idx", pa.array(row_idx, pa.int64())))
                if writer is None:
                    writer = pq.ParquetWriter(tmp, table.schema, compression=cfg["ingest"]["parquet_compression"])
                writer.write_table(table, row_group_size=n)

                manifest.append({"source_file": source, "size_bytes": len(data),
                                 "sha256": hashlib.sha256(data).hexdigest(), "n_cols": len(header),
                                 "n_rows": n + len(malformed), "n_rows_valid": n, "n_malformed": len(malformed)})
                malformed_rows += [{"source_file": source, **m} for m in malformed]
                print(f"{source}: {n} rows, {len(malformed)} malformed")
            writer.close()
            tmp.replace(target)
        finally:
            if tmp.exists():
                if writer is not None:
                    writer.close()
                tmp.unlink()

        manifest_df = pl.DataFrame(manifest)
        manifest_df.write_csv(manifest_path)
        malformed_df = pl.DataFrame(malformed_rows, schema={"source_file": pl.String, "row_idx": pl.Int64,
                                                            "n_fields": pl.Int64, "text": pl.String})
        malformed_df.write_csv(out_dir / "malformed_rows.csv")
        run.update({
            "n_files": manifest_df.height,
            "n_rows_csv": int(manifest_df["n_rows"].sum()),
            "n_rows_valid": int(manifest_df["n_rows_valid"].sum()),
            "n_malformed": malformed_df.height,
            "outputs": {rel(p): sha256_file(p)
                        for p in (target, manifest_path, out_dir / "malformed_rows.csv")},
        })
    print(f"wrote {target} ({run['n_rows_valid']} rows), {manifest_path}")


if __name__ == "__main__":
    main()
