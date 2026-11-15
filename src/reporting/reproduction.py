"""Etapa 10 — compare a reproduction from a fresh clone with the recorded results.

The reproduction is a fresh clone where `make env` and `make all` ran from the raw CSVs, after deleting
results/nested/ and results/ablation/ so that the resumable steps recompute every fold (README, "Reproducing").
This step compares, file by file, every versioned file under data/ and results/ between this copy and the clone,
except the run records (they hold timestamps), and lists the runs of the clone with their runtimes. It writes
results/reproduction/: files.csv (path, SHA-256 here and in the clone, identical), runs.csv (record, script,
make target, start, runtime, commit, dirty flag, Python, platform) and summary.json.

Usage: python -m src.reporting.reproduction <clone_dir>   (make compare-reproduction CLONE=<clone_dir>)
"""
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

from src.reporting.figures import make_targets
from src.utils.config import load_config, path
from src.utils.io import ROOT, rel, sha256_file, write_json
from src.utils.run_metadata import recorded_run

COMPARED_DIRS = ("data", "results")
SKIPPED_SUFFIXES = (".run.json", ".gitkeep")  # run records hold timestamps; placeholders are not results
OUT_SUBDIR = "reproduction"
SECONDS_DIGITS = 1  # run records store runtimes with one decimal

FILES_SCHEMA = {"path": pl.String, "sha256_original": pl.String, "sha256_reproduction": pl.String,
                "identical": pl.Boolean}
RUNS_SCHEMA = {"record": pl.String, "script": pl.String, "make_target": pl.String, "started_at": pl.String,
               "elapsed_s": pl.Float64, "git_commit": pl.String, "git_dirty": pl.Boolean, "python": pl.String,
               "platform": pl.String}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          encoding="utf-8").stdout


def versioned_files(repo: Path, out_prefix: str) -> list[str]:
    """Versioned files under data/ and results/, except run records, placeholders and this step's outputs."""
    names = git(repo, "ls-files", "-z", "--", *COMPARED_DIRS).split("\0")
    return sorted(n for n in names if n and not n.endswith(SKIPPED_SUFFIXES) and not n.startswith(out_prefix))


def compare_files(names: list[str], original: Path, clone: Path) -> pl.DataFrame:
    rows = []
    for name in names:
        here, there = original / name, clone / name
        a = sha256_file(here)
        b = sha256_file(there) if there.exists() else None
        rows.append({"path": name, "sha256_original": a, "sha256_reproduction": b, "identical": a == b})
    return pl.DataFrame(rows, schema=FILES_SCHEMA)


def runs_table(clone: Path, results_rel: str, commit: str, targets: dict[str, str]) -> pl.DataFrame:
    """Run records written by the reproduction (those at the clone's commit), in the order they ran."""
    rows = []
    for record in sorted((clone / results_rel).rglob("*.run.json")):
        d = json.loads(record.read_text(encoding="utf-8"))
        if d["git_commit"] != commit:
            continue  # kept from the original history (optional checks that make all does not run)
        rows.append({"record": record.relative_to(clone).as_posix(), "script": d["script"],
                     "make_target": targets.get(d["script"], ""), "started_at": d["started_at"],
                     "elapsed_s": float(d["elapsed_s"]), "git_commit": d["git_commit"],
                     "git_dirty": d["git_dirty"], "python": d["python"], "platform": d["platform"]})
    return pl.DataFrame(rows, schema=RUNS_SCHEMA).sort("started_at")


def summarize(files: pl.DataFrame, runs: pl.DataFrame, commit: str) -> dict:
    per_target = runs.group_by("make_target", maintain_order=True).agg(pl.col("elapsed_s").sum())
    return {
        "reproduction_commit": commit,
        "files_compared": files.height,
        "files_identical": int(files["identical"].sum()),
        "files_different": files.filter(~pl.col("identical"))["path"].to_list(),
        "runs": runs.height,
        "runs_dirty": int(runs["git_dirty"].sum()),
        "python": sorted(set(runs["python"])),
        "platform": sorted(set(runs["platform"])),
        "elapsed_s_per_target": {t: round(s, SECONDS_DIGITS) for t, s in per_target.iter_rows()},
        "elapsed_s_total": round(float(runs["elapsed_s"].sum()), SECONDS_DIGITS),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.reporting.reproduction <clone_dir>")
    clone = Path(sys.argv[1]).resolve()
    cfg = load_config()
    results = path(cfg, "results_dir")
    out_dir = results / OUT_SUBDIR
    out_prefix = rel(out_dir) + "/"
    pending = [line for line in git(ROOT, "status", "--porcelain", "--", *COMPARED_DIRS).splitlines()
               if not line.endswith(SKIPPED_SUFFIXES) and out_prefix not in line]
    if pending:
        raise SystemExit("data/ or results/ have uncommitted changes; the comparison is against the recorded "
                         "results only:\n" + "\n".join(pending))
    out_dir.mkdir(parents=True, exist_ok=True)
    with recorded_run(cfg, out_dir, "reproduction") as run:
        commit = git(clone, "rev-parse", "HEAD").strip()
        files = compare_files(versioned_files(ROOT, out_prefix), ROOT, clone)
        runs = runs_table(clone, rel(results), commit, make_targets(ROOT / "Makefile"))
        summary = summarize(files, runs, commit)
        files.write_csv(out_dir / "files.csv")
        runs.write_csv(out_dir / "runs.csv")
        write_json(summary, out_dir / "summary.json")
        run["outputs"] = {rel(out_dir / f): sha256_file(out_dir / f)
                          for f in ("files.csv", "runs.csv", "summary.json")}
        print(f"{summary['files_identical']} of {summary['files_compared']} versioned files identical; "
              f"{summary['runs']} runs at {commit[:7]} ({summary['runs_dirty']} dirty)")
        for name in summary["files_different"]:
            print(f"  differs: {name}")
        for target, seconds in summary["elapsed_s_per_target"].items():
            print(f"  {target:10s} {seconds:9.1f} s")


if __name__ == "__main__":
    main()
