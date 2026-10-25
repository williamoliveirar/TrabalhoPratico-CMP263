"""Small I/O helpers: LF-terminated UTF-8 outputs (hashes must match across Windows and Linux) and file hashes."""
import hashlib
import json
from pathlib import Path

from src.utils.config import ROOT


def write_json(obj, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


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
