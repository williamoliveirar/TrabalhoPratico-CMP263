"""Etapa 2 (D3, D4): deterministic samples E (exploration) and M (modelling), deduplicated and disjoint.

Reads the ingested Parquet (`make ingest`), where each row group is one capture and captures are in alphabetical
order. Steps (PLANO §7, Etapa 2):
1. Targets: the attack half of E and of M is split equally among the 33 attack types (integer division). A type
   with fewer unique rows than E + M need gives all of them, `rare_fraction_E` to E and the rest to M.
2. Candidates: for each label, in alphabetical order, ceil(oversample_factor × target) rows (or all rows) are
   drawn without replacement from the label's rows ordered by (source_file, row_idx). One generator, seeded from
   the config, is consumed in this fixed order.
3. Duplicates (D4), over the features of the candidates: copies of a row with the same label34 keep the first
   one; feature vectors shared by rows with different y (conflicting) or by different attack types (ambiguous
   type) are removed with every copy. No feature vector appears twice in E ∪ M.
4. Each attack type's unique rows are shuffled and cut into E and M; benign is then cut to the attack total of
   each sample (1:1). E and M are disjoint by construction.

Outputs: data/processed/sample_E.parquet and sample_M.parquet (sorted by source_file and row_idx),
data/sample_index.csv.gz, results/sampling/sample_counts.csv, results/sampling/sampling_report.json and
results/sampling/make_sample.run.json. Re-running gives byte-identical files.

Usage: python -m src.data.make_sample
"""
import math

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.data.labels import BENIGN, CATEGORY
from src.utils.config import load_config, path
from src.utils.io import rel, sha256_file, write_csv_gz, write_json
from src.utils.run_metadata import recorded_run
from src.utils.seed import make_rng

SOURCE_FILE, ROW_IDX = "source_file", "row_idx"


def capture_index(pf: pq.ParquetFile, label: str) -> pl.DataFrame:
    """One row per row group: row_group, source_file, label and n_rows, read from the Parquet statistics."""
    md = pf.metadata
    names = pf.schema_arrow.names
    li, si = names.index(label), names.index(SOURCE_FILE)
    rows = []
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        lab, src = rg.column(li).statistics, rg.column(si).statistics
        if lab.min != lab.max or src.min != src.max:
            raise ValueError(f"row group {i} holds more than one capture; re-run make ingest")
        rows.append({"row_group": i, SOURCE_FILE: src.min, label: lab.min, "n_rows": rg.num_rows})
    index = pl.DataFrame(rows)
    if not index[SOURCE_FILE].is_sorted() or index[SOURCE_FILE].is_duplicated().any():
        raise ValueError("captures are not one row group each in alphabetical order; re-run make ingest")
    return index


def draw_candidates(pf: pq.ParquetFile, index: pl.DataFrame, n_draw: dict[str, int], label: str,
                    rng: np.random.Generator) -> pl.DataFrame:
    """Draw n_draw[l] rows of each label l without replacement, labels in alphabetical order."""
    picks = {}
    for lab in sorted(n_draw):
        groups = index.filter(pl.col(label) == lab)
        starts = np.concatenate([[0], np.cumsum(groups["n_rows"].to_numpy())])
        pos = np.sort(rng.choice(int(starts[-1]), size=n_draw[lab], replace=False))
        owner = np.searchsorted(starts, pos, side="right") - 1
        for j, rg in enumerate(groups["row_group"].to_list()):
            offsets = pos[owner == j] - starts[j]
            if offsets.size:
                picks[rg] = offsets
    parts = []
    for rg in sorted(picks):
        table = pf.read_row_group(rg)
        if not np.all(np.diff(table.column(ROW_IDX).to_numpy()) > 0):
            raise ValueError(f"row group {rg} is not in row_idx order; re-run make ingest")
        parts.append(pl.from_arrow(table.take(picks[rg])))
    return pl.concat(parts)


def deduplicate(df: pl.DataFrame, features: list[str], label: str, target: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """D4 over the candidates. Returns the unique rows and, per label, how many rows each rule removed.

    Equality follows polars: nulls equal nulls, inf equals inf and -0.0 equals 0.0.
    """
    flags = (df.sort(SOURCE_FILE, ROW_IDX)
             .with_columns(pl.col(target).n_unique().over(features).alias("_n_y"),
                           pl.col(label).n_unique().over(features).alias("_n_label"),
                           pl.struct(features).is_first_distinct().alias("_first"))
             .with_columns((pl.col("_n_y") > 1).alias("conflicting"),
                           ((pl.col("_n_y") == 1) & (pl.col("_n_label") > 1)).alias("ambiguous_type"),
                           ((pl.col("_n_label") == 1) & ~pl.col("_first")).alias("exact_copy")))
    removed = (flags.group_by(label)
               .agg(pl.len().alias("n_candidates"),
                    pl.col("exact_copy").sum().alias("n_exact_copies"),
                    pl.col("conflicting").sum().alias("n_conflicting"),
                    pl.col("ambiguous_type").sum().alias("n_ambiguous_type"))
               .sort(label))
    unique = (flags.filter(~(pl.col("conflicting") | pl.col("ambiguous_type") | pl.col("exact_copy")))
              .select(df.columns))
    return unique, removed


def split_samples(unique: pl.DataFrame, label: str, per_type: dict[str, tuple[int, int]],
                  complete: dict[str, bool], rare_fraction_E: float,
                  rng: np.random.Generator) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Cut each attack type into E and M (rare types: all rows), then benign to the attack total of each sample."""
    parts_E, parts_M, counts = [], [], []

    def cut(lab: str, n_e: int, n_m: int, rare: bool) -> None:
        rows = unique.filter(pl.col(label) == lab)
        perm = rng.permutation(rows.height)
        parts_E.append(rows[perm[:n_e]])
        parts_M.append(rows[perm[n_e:n_e + n_m]])
        counts.append({label: lab, "n_unique": rows.height, "rare": rare, "n_E": n_e, "n_M": n_m})

    for lab in sorted(per_type):
        n = unique.filter(pl.col(label) == lab).height
        e_t, m_t = per_type[lab]
        if n >= e_t + m_t:
            cut(lab, e_t, m_t, False)
        elif complete[lab]:
            n_e = math.floor(rare_fraction_E * n)
            cut(lab, n_e, n - n_e, True)
        else:
            raise ValueError(f"{lab}: {n} unique candidates < {e_t + m_t}; raise sampling.oversample_factor")
    n_e, n_m = sum(c["n_E"] for c in counts), sum(c["n_M"] for c in counts)
    n_benign = unique.filter(pl.col(label) == BENIGN).height
    if n_benign < n_e + n_m:
        raise ValueError(f"{BENIGN}: {n_benign} unique candidates < {n_e + n_m}; raise sampling.oversample_factor")
    cut(BENIGN, n_e, n_m, False)
    order = [SOURCE_FILE, ROW_IDX]
    return (pl.concat(parts_E).sort(order), pl.concat(parts_M).sort(order),
            pl.DataFrame(counts).sort(label))


def composition(df: pl.DataFrame, cols: dict) -> dict:
    by = lambda c: dict(df.group_by(c).agg(pl.len()).sort(c).iter_rows())  # noqa: E731
    return {"n_rows": df.height, "by_y": {str(k): v for k, v in by(cols["target"]).items()},
            "by_category": by(cols["category"])}


def main() -> None:
    cfg = load_config()
    cols, s = cfg["columns"], cfg["sampling"]
    label, target = cols["label"], cols["target"]
    source = path(cfg, "interim_parquet")
    if not source.exists():
        raise SystemExit(f"{source} not found; run `make ingest` first")
    out_dir = path(cfg, "results_dir") / "sampling"
    pf = pq.ParquetFile(source)
    features = [c for c in pf.schema_arrow.names if c not in {label, cols["category"], target, *cols["provenance"]}]

    index = capture_index(pf, label)
    available = dict(index.group_by(label).agg(pl.col("n_rows").sum()).iter_rows())
    attack = sorted(lab for lab in available if lab != BENIGN)
    if BENIGN not in available or len(attack) != 33:
        raise ValueError(f"expected {BENIGN} and 33 attack types, found {sorted(available)}")
    per_type = {lab: (s["E"]["attack"] // len(attack), s["M"]["attack"] // len(attack)) for lab in attack}
    wanted = {lab: e + m for lab, (e, m) in per_type.items()} | {BENIGN: s["E"]["benign"] + s["M"]["benign"]}
    n_draw = {lab: min(available[lab], math.ceil(s["oversample_factor"] * wanted[lab])) for lab in available}
    complete = {lab: n_draw[lab] == available[lab] for lab in available}

    rng = make_rng(cfg)
    with recorded_run(cfg, out_dir, "make_sample") as run:
        candidates = draw_candidates(pf, index, n_draw, label, rng)
        unique, removed = deduplicate(candidates, features, label, target)
        sample_E, sample_M, split = split_samples(unique, label, per_type, complete, s["rare_fraction_E"], rng)

        both = pl.concat([sample_E, sample_M])
        if both.select(pl.struct(SOURCE_FILE, ROW_IDX).n_unique()).item() != both.height:
            raise AssertionError("E and M share rows")
        if both.select(pl.struct(features).n_unique()).item() != both.height:
            raise AssertionError("a feature vector appears twice in E ∪ M")

        targets = {"E": path(cfg, "sample_E"), "M": path(cfg, "sample_M")}
        for name, df in (("E", sample_E), ("M", sample_M)):
            targets[name].parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(targets[name], compression=s["parquet_compression"])
        index_path = path(cfg, "sample_index")
        write_csv_gz(pl.concat([df.select(pl.lit(name).alias("sample"), SOURCE_FILE, ROW_IDX)
                                for name, df in (("E", sample_E), ("M", sample_M))]), index_path)

        counts = (pl.DataFrame({label: list(available), "n_available": list(available.values())})
                  .with_columns(pl.col(label).replace_strict(CATEGORY).alias(cols["category"]))
                  .join(removed, on=label).join(split, on=label)
                  .sort(label))
        out_dir.mkdir(parents=True, exist_ok=True)
        counts.write_csv(out_dir / "sample_counts.csv")

        files = {rel(p): sha256_file(p) for p in (*targets.values(), index_path)}
        write_json({
            "source": f"{rel(source)} (make ingest)",
            "design": {"E": s["E"], "M": s["M"], "n_attack_types": len(attack),
                       "per_type_target": {"E": per_type[attack[0]][0], "M": per_type[attack[0]][1]},
                       "oversample_factor": s["oversample_factor"], "rare_fraction_E": s["rare_fraction_E"]},
            "candidates": candidates.height,
            "duplicates": {
                "scope": "among the drawn candidates, not over the whole dataset",
                "exact_copies_removed": int(removed["n_exact_copies"].sum()),
                "conflicting_rows_removed": int(removed["n_conflicting"].sum()),
                "ambiguous_type_rows_removed": int(removed["n_ambiguous_type"].sum()),
            },
            "rare_types": split.filter(pl.col("rare"))[label].to_list(),
            "samples": {"E": composition(sample_E, cols), "M": composition(sample_M, cols)},
            "files_sha256": files,
        }, out_dir / "sampling_report.json")
        run["outputs"] = files | {rel(out_dir / f): sha256_file(out_dir / f)
                                  for f in ("sample_counts.csv", "sampling_report.json")}
    print(f"wrote {targets['E']} ({sample_E.height} rows), {targets['M']} ({sample_M.height} rows), {index_path}")


if __name__ == "__main__":
    main()
