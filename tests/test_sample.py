"""Sampling is deterministic, removes duplicates as D4 says, applies the rare-type rule and keeps E and M disjoint."""
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.labels import BENIGN
from src.data.make_sample import capture_index, deduplicate, draw_candidates, split_samples

FEATURES = ["Rate", "Std"]
LABEL, TARGET = "label34", "y"


def frame(rows: list[tuple]) -> pl.DataFrame:
    """rows: (Rate, Std, label34, y, source_file, row_idx)."""
    return pl.DataFrame(rows, schema=[("Rate", pl.Float64), ("Std", pl.Float64), (LABEL, pl.String),
                                      (TARGET, pl.Int8), ("source_file", pl.String), ("row_idx", pl.Int64)],
                        orient="row")


def test_deduplicate_rules():
    df = frame([
        (1.0, 1.0, "A", 1, "A/a.csv", 1),            # exact copy of the next row: the first by row_idx stays
        (1.0, 1.0, "A", 1, "A/a.csv", 0),
        (float("inf"), None, "A", 1, "A/a.csv", 2),  # inf and null compare equal: one copy stays
        (float("inf"), None, "A", 1, "A/a.csv", 3),
        (2.0, 2.0, "A", 1, "A/a.csv", 4),            # same features, different y: every copy goes
        (2.0, 2.0, BENIGN, 0, "B/b.csv", 0),
        (3.0, 3.0, "A", 1, "A/a.csv", 5),            # same features, two attack types: every copy goes
        (3.0, 3.0, "C", 1, "C/c.csv", 0),
        (4.0, 4.0, BENIGN, 0, "B/b.csv", 1),         # unique
    ])
    unique, removed = deduplicate(df, FEATURES, LABEL, TARGET)
    assert unique.select("source_file", "row_idx").rows() == [("A/a.csv", 0), ("A/a.csv", 2), ("B/b.csv", 1)]
    by_label = {r[LABEL]: r for r in removed.to_dicts()}
    assert by_label["A"] == {LABEL: "A", "n_candidates": 6, "n_exact_copies": 2, "n_conflicting": 1,
                             "n_ambiguous_type": 1}
    assert by_label[BENIGN]["n_conflicting"] == 1 and by_label["C"]["n_ambiguous_type"] == 1


def test_split_targets_rare_types_and_benign_balance():
    rows = ([(float(i), 0.0, "A", 1, "A/a.csv", i) for i in range(50)]
            + [(100.0 + i, 0.0, "R", 1, "R/r.csv", i) for i in range(11)]
            + [(200.0 + i, 0.0, BENIGN, 0, "B/b.csv", i) for i in range(100)])
    per_type = {"A": (5, 20), "R": (5, 20)}
    E, M, counts = split_samples(frame(rows), LABEL, per_type, {"A": False, "R": True}, 0.2,
                                 np.random.default_rng(0))
    c = {r[LABEL]: r for r in counts.to_dicts()}
    assert (c["A"]["n_E"], c["A"]["n_M"], c["A"]["rare"]) == (5, 20, False)
    assert (c["R"]["n_E"], c["R"]["n_M"], c["R"]["rare"]) == (2, 9, True)   # floor(0.2 × 11) = 2
    assert (c[BENIGN]["n_E"], c[BENIGN]["n_M"]) == (7, 29)                  # 1:1 with the attack totals
    for s in (E, M):
        assert s[TARGET].sum() * 2 == s.height
    keys = lambda s: set(s.select("source_file", "row_idx").rows())  # noqa: E731
    assert not keys(E) & keys(M)


def test_short_label_that_was_not_drawn_in_full_fails():
    rows = [(float(i), 0.0, "A", 1, "A/a.csv", i) for i in range(10)]
    with pytest.raises(ValueError, match="oversample_factor"):
        split_samples(frame(rows), LABEL, {"A": (5, 20)}, {"A": False}, 0.2, np.random.default_rng(0))


def test_draw_candidates_is_deterministic(tmp_path):
    target = tmp_path / "raw.parquet"
    captures = [("A/a1.csv", "A", 30), ("A/a2.csv", "A", 20), ("B/b.csv", BENIGN, 40)]
    schema = pa.schema([("Rate", pa.float64()), ("Std", pa.float64()), (LABEL, pa.string()), (TARGET, pa.int8()),
                        ("source_file", pa.string()), ("row_idx", pa.int64())])
    with pq.ParquetWriter(target, schema) as writer:
        for source, lab, n in captures:
            table = pa.table({"Rate": np.arange(n, dtype=float), "Std": np.zeros(n), LABEL: [lab] * n,
                              TARGET: pa.array([int(lab != BENIGN)] * n, pa.int8()), "source_file": [source] * n,
                              "row_idx": np.arange(n)}, schema=schema)
            writer.write_table(table, row_group_size=n)
    pf = pq.ParquetFile(target)
    index = capture_index(pf, LABEL)
    assert index["n_rows"].to_list() == [30, 20, 40]

    draw = lambda: draw_candidates(pf, index, {"A": 15, BENIGN: 40}, LABEL, np.random.default_rng(7))  # noqa: E731
    first, second = draw(), draw()
    assert first.equals(second)
    assert first.filter(pl.col(LABEL) == "A").height == 15
    assert first.filter(pl.col(LABEL) == BENIGN)["row_idx"].sort().to_list() == list(range(40))  # all rows
    assert first.select(pl.struct("source_file", "row_idx").n_unique()).item() == first.height
