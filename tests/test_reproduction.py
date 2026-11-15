"""Etapa 10 — comparison of a fresh-clone reproduction with the recorded results."""
import json

import polars as pl

from src.reporting.reproduction import compare_files, runs_table


def test_compare_files_flags_changed_and_missing(tmp_path):
    original, clone = tmp_path / "original", tmp_path / "clone"
    for root in (original, clone):
        (root / "results").mkdir(parents=True)
    for name, here, there in (("same.csv", b"1\n", b"1\n"), ("changed.csv", b"1\n", b"2\n"),
                              ("missing.csv", b"1\n", None)):
        (original / "results" / name).write_bytes(here)
        if there is not None:
            (clone / "results" / name).write_bytes(there)
    names = ["results/changed.csv", "results/missing.csv", "results/same.csv"]
    df = compare_files(names, original, clone)
    assert dict(zip(df["path"], df["identical"])) == {
        "results/changed.csv": False, "results/missing.csv": False, "results/same.csv": True}
    assert df.filter(pl.col("path") == "results/missing.csv")["sha256_reproduction"][0] is None


def test_runs_table_keeps_only_the_reproduction_runs_in_order(tmp_path):
    results = tmp_path / "results"
    (results / "a").mkdir(parents=True)
    (results / "b").mkdir()
    base = {"python": "3.12.7", "platform": "test", "git_dirty": False}
    records = {
        "a/x.run.json": {"script": "src.x", "started_at": "2026-11-15T02:00:00+00:00", "git_commit": "new"},
        "b/y.run.json": {"script": "src.y", "started_at": "2026-11-15T01:00:00+00:00", "git_commit": "new"},
        "b/old.run.json": {"script": "src.old", "started_at": "2026-10-01T00:00:00+00:00", "git_commit": "old"},
    }
    for name, d in records.items():
        (results / name).write_text(json.dumps({**base, **d, "elapsed_s": 1.5}), encoding="utf-8")
    df = runs_table(tmp_path, "results", "new", {"src.x": "target_x"})
    assert df["script"].to_list() == ["src.y", "src.x"]
    assert df["make_target"].to_list() == ["", "target_x"]
    assert df["record"].to_list() == ["results/b/y.run.json", "results/a/x.run.json"]
