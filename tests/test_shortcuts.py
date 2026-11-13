"""Etapa 9: the pre-registered rules of the shortcut ablation and of the leave-capture-out."""
import numpy as np
import polars as pl
import pytest

from src.evaluation.stats import holm
from src.experiments.leave_capture_out import compare
from src.experiments.shortcuts import ablation_tests, recall_comparison
from src.utils.config import load_config

CFG = load_config()
LEVEL, ALPHA = 0.95, 0.05


def test_holm_keeps_untestable_comparisons_out_of_the_family():
    adjusted = holm([0.01, np.nan, 0.04])
    assert np.isnan(adjusted[1])
    np.testing.assert_allclose(adjusted[[0, 2]], [0.02, 0.04])                  # family of 2, not 3


def oof(pred: np.ndarray, y: np.ndarray, label: np.ndarray, window: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({"position": np.arange(y.size), "y": y, "pred": pred, "score": pred.astype(float),
                         "label34": label, "category8": np.where(y == 1, "Web", "Benign"), "window": window,
                         "outer_fold": np.arange(y.size) % 5})


def toy():
    y = np.repeat([0, 1, 1], [200, 200, 200])
    label = np.repeat(["BenignTraffic", "Backdoor_Malware", "DDoS-ACK_Fragmentation"], 200)
    window = np.repeat([10, 10, 100], 200)
    return y, label, window


def folds(mcc: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"outer_fold": range(5), "test_mcc": mcc, "n_train": 480, "n_test": 120})


def test_ablation_reading_follows_the_pre_registered_thresholds():
    y, label, window = toy()
    ablated = y.copy()
    ablated[200:300] = 0                                                       # half of one type is missed
    oofs = {"full": oof(y, y, label, window), "big": oof(ablated, y, label, window), "same": oof(y, y, label, window)}
    fold = {"full": folds([0.95, 0.96, 0.95, 0.97, 0.95]), "big": folds([0.70, 0.72, 0.69, 0.71, 0.70]),
            "same": folds([0.95, 0.96, 0.95, 0.97, 0.95])}
    t = ablation_tests(CFG, "m", ["big", "same"], oofs, fold, LEVEL, ALPHA)
    big, same = t.row(0, named=True), t.row(1, named=True)
    assert big["significant_drop"] and big["reading"] == "large" and big["ci_low"] > 0
    assert not same["significant_drop"] and same["reading"] == "not significant" and np.isnan(same["p_holm"])

    recall = recall_comparison(oofs, "full", ["big"], LEVEL, CFG["ablation"]["type_collapse"])
    flags = dict(zip(recall["label34"], recall["collapses"]))
    assert flags == {"Backdoor_Malware": True, "DDoS-ACK_Fragmentation": False}


def test_capture_collapse_needs_size_and_separated_intervals():
    A, B, C = "Recon-OSScan", "SqlInjection", "Uploading_Attack"                 # real labels: window looked up
    rows = []
    for design, sens in (("nested_M", [0.90, 0.90, 0.90]), ("capture_M", [0.70, 0.85, 0.79])):
        for lab, s, n in zip([A, B, C], sens, [3000, 3000, 30]):
            rows.append({"design": design, "label34": lab, "n": n, "detected": int(s * n), "sensitivity": s,
                         "ci_low": s - (0.01 if n > 100 else 0.15), "ci_high": s + (0.01 if n > 100 else 0.15)})
    table = compare(pl.DataFrame(rows), grouped={A, C}, collapse=0.10)
    flags = dict(zip(table["label34"], table["collapses"]))
    assert flags == {A: True, B: False, C: False}                             # C: big drop, overlapping intervals
    assert table.filter(pl.col("grouped"))["label34"].to_list() == [A, C]
    assert table["drop"].to_list() == pytest.approx([0.20, 0.11, 0.05])
