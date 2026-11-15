"""Run provenance (docs/PLANO.md §5.4): every script that writes results also writes `<name>.run.json` next to them.

The record holds start time, git commit (and whether code or configs had uncommitted changes), a hash of the
configuration, seed, n_jobs, Python and library versions, elapsed time, plus whatever the script adds (e.g. the
SHA-256 of its outputs). Timestamps live only here, so the result files themselves stay byte-identical across runs.

Usage:
    with recorded_run(cfg, out_dir, "inspect_raw") as run:
        ...
        run["outputs"] = {...}
"""
import hashlib
import json
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from src.utils.config import ROOT
from src.utils.io import write_json

LIBRARIES = ("numpy", "pandas", "pyarrow", "polars", "scikit-learn", "lightgbm", "shap", "scipy",
             "matplotlib", "PyYAML", "joblib")
TRACKED = ("src", "configs", "Makefile", "requirements.txt")  # paths whose uncommitted changes make a run "dirty"


def _git(*args: str) -> str | None:
    """Stripped output of a git command run at the repository root; None if git is missing or the command fails."""
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def config_hash(cfg: dict) -> str:
    """SHA-256 of the configuration minus `paths`, which only locate the data on each machine (configs/local.yaml)."""
    content = {k: v for k, v in cfg.items() if k != "paths"}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def run_metadata(cfg: dict) -> dict:
    """Provenance fields of a run record, taken at the start of the run (elapsed_s is added by recorded_run)."""
    main = sys.modules["__main__"]
    status = _git("status", "--porcelain", "--", *TRACKED)
    return {
        "script": getattr(main.__spec__, "name", None) or Path(sys.argv[0]).name,  # module name under `python -m`
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "config_sha256": config_hash(cfg),
        "seed": cfg["seed"],
        "n_jobs": cfg["n_jobs"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {lib: metadata.version(lib) for lib in LIBRARIES},
    }


@contextmanager
def recorded_run(cfg: dict, out_dir: Path, name: str):
    """Yield the run record; write it to out_dir/<name>.run.json only if the block finishes without error."""
    record = run_metadata(cfg)
    start = time.perf_counter()
    yield record
    record["elapsed_s"] = round(time.perf_counter() - start, 1)
    write_json(record, Path(out_dir) / f"{name}.run.json")
