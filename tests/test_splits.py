"""Etapa 5: the frozen folds are the scikit-learn folds, keep the label proportions and never mix test into training."""
import numpy as np
import polars as pl
import pytest
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from src.data.labels import BENIGN
from src.experiments.make_splits import (CAPTURE, NESTED, SPOTCHECK, UNUSED, capture_assignment, check_capture,
                                         check_kfold, check_nested, folds, inner_assignment, inner_folds,
                                         kfold_assignment, label_spread, load_assignment, outer_folds,
                                         read_assignment, spotcheck_assignment, spotcheck_folds)
from src.utils.config import load_config, path
from src.utils.io import write_csv_gz

CFG = load_config()
SEED, KEYS = 7, ["source_file", "row_idx"]
SIZES = {BENIGN: 200, "A": 53, "B": 31, "C": 17}
CAPTURES = {BENIGN: 3, "A": 7, "B": 2, "C": 1}  # A has more captures than folds, the others fewer


def synthetic() -> pl.DataFrame:
    """A sample in (source_file, row_idx) order: label34 and the capture each instance comes from."""
    rows = [(f"{lab}/{i % CAPTURES[lab]}.csv", i, lab) for lab, n in SIZES.items() for i in range(n)]
    return pl.DataFrame(rows, schema=["source_file", "row_idx", "label34"], orient="row").sort(KEYS)


def nested_frame(df: pl.DataFrame, outer: np.ndarray, inner: np.ndarray) -> pl.DataFrame:
    return df.with_columns(pl.Series("outer", outer), *(pl.Series(f"inner{k}", inner[:, k])
                                                        for k in range(inner.shape[1])))


def same_folds(a, b) -> bool:
    return len(a) == len(b) and all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
                                    for x, y in zip(a, b))


def test_spotcheck_folds_are_those_of_repeated_stratified_kfold():
    labels = synthetic()["label34"].to_numpy()
    assign = spotcheck_assignment(labels, n_splits=5, n_repeats=2, seed=SEED)
    frame = pl.DataFrame({f"repeat{r}": assign[:, r] for r in range(2)})
    expected = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
                    .split(np.zeros((len(labels), 1)), labels))
    assert same_folds(spotcheck_folds(frame), expected)


def test_outer_folds_are_disjoint_cover_the_sample_and_keep_label_proportions():
    labels = synthetic()["label34"].to_numpy()
    outer = kfold_assignment(labels, n_splits=5, seed=SEED)
    pairs = folds(outer)
    tests = np.concatenate([test for _, test in pairs])
    assert np.array_equal(np.sort(tests), np.arange(len(labels)))       # each instance tested exactly once
    for train, test in pairs:
        assert not np.intersect1d(train, test).size and train.size + test.size == len(labels)
    assert label_spread(labels, outer) <= 1                              # e.g. C (17): 3 or 4 per fold
    expected = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                    .split(np.zeros((len(labels), 1)), labels))
    assert same_folds(pairs, expected)


def test_inner_folds_stay_inside_the_outer_training_set():
    df = synthetic()
    labels = df["label34"].to_numpy()
    outer = kfold_assignment(labels, n_splits=5, seed=SEED)
    inner = inner_assignment(labels, outer, n_splits=3, seed=SEED)
    frame = nested_frame(df, outer, inner)
    for k, (train, test) in enumerate(outer_folds(frame)):
        assert (inner[test, k] == UNUSED).all()
        relative = inner_folds(frame, k)
        for tr, val in relative:                                         # positions relative to X.iloc[train]
            assert not np.isin(train[tr], test).any() and not np.isin(train[val], test).any()
        assert np.array_equal(np.sort(np.concatenate([train[val] for _, val in relative])), train)
        expected = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
                        .split(np.zeros((train.size, 1)), labels[train]))
        assert same_folds(relative, expected)
    check_nested(labels, outer, inner, n_inner=3)


def test_capture_folds_hold_whole_captures_and_leave_the_other_labels_alone():
    df = synthetic()
    labels, captures = df["label34"].to_numpy(), df["source_file"].to_numpy()
    outer = kfold_assignment(labels, n_splits=5, seed=SEED)
    capture, grouped = capture_assignment(labels, captures, outer)
    assert grouped == ["A"]
    a = labels == "A"
    assert all(np.unique(capture[captures == c]).size == 1 for c in np.unique(captures[a]))
    assert np.unique(capture[a]).size == 5 and np.array_equal(capture[~a], outer[~a])
    check_capture(labels, captures, outer, capture, grouped)


def test_checks_reject_broken_folds():
    labels = np.array(["A"] * 10 + ["B"] * 10)
    with pytest.raises(AssertionError):                                  # all of B in fold 0
        check_kfold(labels, np.array([0, 1] * 5 + [0] * 10), n_splits=2, what="skewed")
    outer = np.array([0, 1] * 10)
    inner = inner_assignment(labels, outer, n_splits=2, seed=SEED)
    inner[0, 0] = 1                                                      # a test instance of outer fold 0
    with pytest.raises(AssertionError):
        check_nested(labels, outer, inner, n_inner=2)


def test_read_assignment_checks_the_sample_row_order(tmp_path):
    df = synthetic()
    labels = df["label34"].to_numpy()
    target = tmp_path / "split.csv.gz"
    write_csv_gz(df.with_columns(pl.Series("outer", kfold_assignment(labels, 5, SEED), dtype=pl.Int8)), target)
    assert read_assignment(target, df, KEYS)["outer"].to_list() == kfold_assignment(labels, 5, SEED).tolist()
    with pytest.raises(ValueError):
        read_assignment(target, df.reverse(), KEYS)


FROZEN = {name: path(CFG, "splits_dir") / f"{name}.csv.gz" for name in (SPOTCHECK, NESTED, CAPTURE)}


@pytest.mark.skipif(not all(p.exists() for p in FROZEN.values()), reason="run make splits first")
def test_frozen_splits_meet_the_etapa5_criteria():
    """PLANO §7, Etapa 5, "Pronto quando", on the versioned files (no sample needed)."""
    v, label = CFG["validation"], CFG["columns"]["label"]
    spot = pl.read_csv(FROZEN[SPOTCHECK])
    for r in range(v["spotcheck"]["n_repeats"]):
        check_kfold(spot[label].to_numpy(), spot[f"repeat{r}"].to_numpy(), v["spotcheck"]["n_splits"], "spotcheck")
    nested = pl.read_csv(FROZEN[NESTED])
    labels, outer = nested[label].to_numpy(), nested["outer"].to_numpy()
    check_kfold(labels, outer, v["outer"]["n_splits"], "outer loop")    # disjoint, cover M, proportions kept
    inner = nested.select(f"inner{k}" for k in range(v["outer"]["n_splits"])).to_numpy()
    check_nested(labels, outer, inner, v["inner"]["n_splits"])           # no test instance in the inner loop
    capture = pl.read_csv(FROZEN[CAPTURE])
    assert capture.select(KEYS).equals(nested.select(KEYS))
    grouped = sorted(set(capture.filter(pl.col("grouped"))[label]))
    check_capture(labels, capture["source_file"].to_numpy(), outer, capture["capture_outer"].to_numpy(), grouped)
    if path(CFG, "sample_M").exists():
        sample = pl.read_parquet(path(CFG, "sample_M"), columns=KEYS)
        assert load_assignment(CFG, NESTED, sample).height == sample.height
