"""EDA helpers: window groups from the dataset paper, AUC orientation and the one-feature trees."""
from collections import Counter

import numpy as np
import polars as pl

from src.data.eda import WINDOW, auc, single_feature_models
from src.data.labels import BENIGN, CATEGORY, window_of


def test_window_groups_follow_the_dataset_paper():
    # benign and 14 attack types use 10-packet windows; the 19 DDoS, DoS and Mirai types use 100
    assert Counter(window_of(label) for label in CATEGORY) == {10: 15, 100: 19}
    assert window_of(BENIGN) == 10 and window_of("DDoS-ICMP_Flood") == 100 and window_of("XSS") == 10


def test_auc_is_the_probability_that_attack_scores_higher():
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert auc(np.array([1.0, np.inf]), np.array([1.0, 2.0])) == 0.625  # inf ranks highest; ties count 1/2


def test_single_feature_models_report_tpr_per_window_group():
    n = 40
    df = pl.DataFrame({
        "x": np.r_[np.zeros(n), np.ones(n), np.zeros(n)],  # separates the window-100 type only
        "label34": [BENIGN] * n + ["DDoS-ICMP_Flood"] * n + ["XSS"] * n,
        "y": [0] * n + [1] * (2 * n),
        WINDOW: [10] * n + [100] * n + [10] * n,
    })
    row = single_feature_models(df, ["x"], "label34", "y", depth=2, n_splits=5, seed=0).row(0, named=True)
    assert (row["tpr_window100"], row["tpr_window10"], row["fpr"]) == (1.0, 0.0, 0.0)
