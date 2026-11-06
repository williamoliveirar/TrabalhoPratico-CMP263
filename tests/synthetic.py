"""Synthetic sample with the 39 columns of the data, shared by the tests of Etapas 4 and 6 (imported by name)."""
import json

import numpy as np
import polars as pl

from src.utils.config import ROOT

COLUMNS = [f["name"] for f in json.loads((ROOT / "results/inspection/schema.json").read_text("utf-8"))["features"]]
PROTOCOLS = [0.0, 1.0, 6.0, 17.0, 47.0]


def sample(n: int, seed: int) -> pl.DataFrame:
    """The 39 columns of the data and the auxiliary columns; Std null in a few rows; random labels."""
    rng = np.random.default_rng(seed)
    data = {c: rng.exponential(10.0, n) for c in COLUMNS}
    data["Protocol Type"] = rng.choice(PROTOCOLS, n)
    data["Std"] = np.where(np.arange(n) % 25 == 0, np.nan, data["Std"])
    y = rng.integers(0, 2, n)
    df = pl.DataFrame(data).with_columns(pl.col("Std").fill_nan(None))
    return df.with_columns(pl.Series("label34", np.where(y == 1, "DDoS-ICMP_Flood", "BenignTraffic")),
                           pl.Series("category8", np.where(y == 1, "DDoS", "Benign")), pl.Series("y", y),
                           pl.lit("capture.pcap.csv").alias("source_file"), pl.Series("row_idx", np.arange(n)))
