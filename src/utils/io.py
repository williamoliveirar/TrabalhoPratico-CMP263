"""Small I/O helpers: LF-terminated UTF-8 outputs (hashes must match across Windows and Linux) and file hashes."""
import gzip
import hashlib
import json
from pathlib import Path

import polars as pl

from src.utils.config import ROOT


def write_json(obj, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_csv_gz(df: pl.DataFrame, target: Path) -> None:
    """Gzip without a timestamp or file name in the header, so the bytes depend only on the content."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz:
        gz.write(df.write_csv().encode("utf-8"))


def sha256_file(p: Path, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while block := f.read(chunk_bytes):
            h.update(block)
    return h.hexdigest()


def rel(p: Path) -> str:
    """Path relative to the repository root, for versioned outputs (no local directory names)."""
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).name
