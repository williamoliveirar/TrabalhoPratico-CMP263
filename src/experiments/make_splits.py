"""Etapa 5 (D8): training, validation and test folds, generated once and frozen before any model is trained.

Every algorithm sees exactly the same folds (PLANO §7, Etapa 5). Folds are stored as fold assignments: one row per
instance, in the row order of the sample (source_file, row_idx), so they can be checked against the sample and read
without scikit-learn.

- data/splits/spotcheck_E.csv.gz: repeated stratified k-fold cross-validation on sample E (spot-checking, Etapa 6).
  Column `repeat<r>` is the fold whose held-out set holds the instance in repeat r. These held-out sets play the
  role of validation sets: they choose the finalists and never give the performance estimate.
- data/splits/nested_M.csv.gz: nested cross-validation on sample M (Etapas 7 and 8). Column `outer` is the test
  fold of the instance in the outer loop. Column `inner<k>` is its validation fold in the inner loop run on the
  training set of outer fold k, and -1 when the instance is in the test set of outer fold k.
- data/splits/capture_M.csv.gz: complementary leave-capture-out folds on sample M (D8 (a), proposal of
  2026-11-05). Attack types with at least n_splits captures are split by capture: whole captures go to one fold,
  balanced by GroupKFold. Every other label keeps its outer fold of nested_M, so the two designs differ only in
  the grouped types.

All folds are stratified by label34, which also keeps the class proportion (benign is a single label). Inner folds
are drawn on the outer training rows in ascending position order; `inner_folds(assign, k)` therefore returns
positions relative to X.iloc[train_idx] with train_idx ascending, which is what RandomizedSearchCV(cv=...) needs.

Outputs also include results/splits/partition_sizes.csv, results/splits/splits_report.json and
results/splits/make_splits.run.json. Re-running gives byte-identical files.

Usage: python -m src.experiments.make_splits
"""
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, StratifiedKFold

from src.data.labels import BENIGN
from src.data.make_sample import SOURCE_FILE
from src.utils.config import load_config, path
from src.utils.io import rel, sha256_file, write_csv_gz, write_json
from src.utils.run_metadata import recorded_run

SPOTCHECK, NESTED, CAPTURE = "spotcheck_E", "nested_M", "capture_M"
UNUSED = -1  # fold id of an instance that is not part of the split (inner<k> of the outer test rows)


# --- fold assignments ------------------------------------------------------------------------------------------

def _assign(n: int, held_out: list[np.ndarray]) -> np.ndarray:
    """Fold id of each of `n` instances, from the held-out positions of each fold; UNUSED where no fold holds it."""
    fold = np.full(n, UNUSED, dtype=np.int8)
    for f, rows in enumerate(held_out):
        fold[rows] = f
    return fold


def spotcheck_assignment(labels: np.ndarray, n_splits: int, n_repeats: int, seed: int) -> np.ndarray:
    """(n, n_repeats): held-out fold of each instance in each repeat of RepeatedStratifiedKFold."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    held_out = [test for _, test in cv.split(np.zeros((len(labels), 1)), labels)]
    return np.column_stack([_assign(len(labels), held_out[r * n_splits:(r + 1) * n_splits])
                            for r in range(n_repeats)])


def kfold_assignment(labels: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Held-out fold of each instance under StratifiedKFold(shuffle=True), stratified by `labels`."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return _assign(len(labels), [test for _, test in cv.split(np.zeros((len(labels), 1)), labels)])


def inner_assignment(labels: np.ndarray, outer: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """(n, n_outer): validation fold of each instance in the inner loop of outer fold k; UNUSED in its test set."""
    inner = np.full((len(labels), outer.max() + 1), UNUSED, dtype=np.int8)
    for k in range(outer.max() + 1):
        train = np.flatnonzero(outer != k)
        inner[train, k] = kfold_assignment(labels[train], n_splits, seed)
    return inner


def capture_assignment(labels: np.ndarray, captures: np.ndarray, outer: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Outer folds with whole captures for the labels that have at least as many captures as folds.

    Returns the fold of each instance and the grouped labels. The other labels keep their fold in `outer`.
    """
    n_splits = int(outer.max() + 1)
    fold, grouped = outer.copy(), []
    for lab in sorted(set(labels)):
        rows = np.flatnonzero(labels == lab)
        if len(np.unique(captures[rows])) < n_splits:
            continue
        grouped.append(lab)
        # no shuffle, no seed: GroupKFold places the largest captures first, each in the fold with fewest rows
        for f, (_, test) in enumerate(GroupKFold(n_splits=n_splits).split(rows, groups=captures[rows])):
            fold[rows[test]] = f
    return fold, grouped


# --- folds as positions (Etapas 6 to 9) ------------------------------------------------------------------------

def folds(fold: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """(training, held-out) positions for each fold id; instances marked UNUSED are in neither."""
    used = fold != UNUSED
    return [(np.flatnonzero(used & (fold != f)), np.flatnonzero(fold == f)) for f in range(int(fold.max()) + 1)]


def spotcheck_folds(assign: pl.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Training and validation positions in E, in the order of RepeatedStratifiedKFold.split."""
    repeats = sorted(c for c in assign.columns if c.startswith("repeat"))
    return [pair for r in repeats for pair in folds(assign[r].to_numpy())]


def outer_folds(assign: pl.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Training and test positions in M for each outer fold."""
    return folds(assign["outer"].to_numpy())


def inner_folds(assign: pl.DataFrame, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Training and validation positions relative to the training set of outer fold k (ascending positions)."""
    outer = assign["outer"].to_numpy()
    return folds(assign[f"inner{k}"].to_numpy()[outer != k])


def read_assignment(file: Path, sample: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Fold assignment stored in `file`, checked to be in the row order of `sample` (same keys, same order)."""
    assign = pl.read_csv(file)
    if assign.height != sample.height or not assign.select(keys).equals(sample.select(keys)):
        raise ValueError(f"{file} does not match the sample row by row; re-run make splits after make sample")
    return assign


def load_assignment(cfg: dict, name: str, sample: pl.DataFrame) -> pl.DataFrame:
    """Frozen fold assignment `name` (SPOTCHECK, NESTED or CAPTURE) for `sample`."""
    return read_assignment(path(cfg, "splits_dir") / f"{name}.csv.gz", sample, cfg["columns"]["provenance"])


# --- checks (the "Pronto quando" of Etapa 5) -------------------------------------------------------------------

def label_spread(labels: np.ndarray, fold: np.ndarray) -> int:
    """Largest difference, over labels, between the counts of one label in two folds (stratified: at most 1)."""
    used = fold != UNUSED
    n_folds = int(fold.max()) + 1
    return max(int(np.ptp(np.bincount(fold[used & (labels == lab)], minlength=n_folds)))
               for lab in np.unique(labels[used]))


def check_kfold(labels: np.ndarray, fold: np.ndarray, n_splits: int, what: str) -> None:
    """Every instance in exactly one held-out fold, all n_splits folds present, label proportions kept."""
    if not np.array_equal(np.unique(fold), np.arange(n_splits)):
        raise AssertionError(f"{what}: fold ids {np.unique(fold)} are not 0..{n_splits - 1}")
    if label_spread(labels, fold) > 1:
        raise AssertionError(f"{what}: a label differs by more than one instance between two folds")


def check_nested(labels: np.ndarray, outer: np.ndarray, inner: np.ndarray, n_inner: int) -> None:
    """No outer test instance enters the inner loop of its fold; each inner loop is a valid stratified k-fold."""
    for k in range(inner.shape[1]):
        test = outer == k
        if not (inner[test, k] == UNUSED).all():
            raise AssertionError(f"outer fold {k}: a test instance is in the inner loop")
        check_kfold(labels[~test], inner[~test, k], n_inner, f"inner loop of outer fold {k}")


def check_capture(labels: np.ndarray, captures: np.ndarray, outer: np.ndarray, capture: np.ndarray,
                  grouped: list[str]) -> None:
    """Labels not grouped keep their outer fold, each capture of a grouped label lies in a single fold, and every
    grouped label appears in every test fold."""
    is_grouped = np.isin(labels, grouped)
    if not np.array_equal(capture[~is_grouped], outer[~is_grouped]):
        raise AssertionError("capture folds: a label that is not grouped changed fold")
    folds_per_capture = (pl.DataFrame({"capture": captures[is_grouped], "fold": capture[is_grouped]})
                         .group_by("capture").agg(pl.col("fold").n_unique())["fold"])
    if (folds_per_capture != 1).any():
        raise AssertionError("capture folds: a capture of a grouped label is split between folds")
    for lab in grouped:
        rows = labels == lab
        if len(np.unique(capture[rows])) != int(outer.max()) + 1:
            raise AssertionError(f"capture folds: {lab} is missing from some test fold")


# --- report ----------------------------------------------------------------------------------------------------

def partition_sizes(labels: np.ndarray, design: str, iteration: str, pairs: dict[str, np.ndarray]) -> list[dict]:
    """One row per set of `pairs` (set name -> positions): instances, attack and benign counts, labels present and
    the smallest label count."""
    rows = []
    for name, positions in pairs.items():
        lab = labels[positions]
        counts = np.unique(lab, return_counts=True)[1]
        rows.append({"design": design, "iteration": iteration, "set": name, "n_instances": int(positions.size),
                     "n_attack": int((lab != BENIGN).sum()), "n_benign": int((lab == BENIGN).sum()),
                     "n_labels": int(counts.size), "min_per_label": int(counts.min())})
    return rows


def size_table(labels_E: np.ndarray, labels_M: np.ndarray, spot: pl.DataFrame, nested: pl.DataFrame,
               capture: pl.DataFrame) -> pl.DataFrame:
    """Sizes of every training, validation and test set of the three designs (results/splits/partition_sizes.csv)."""
    rows = []
    n_splits = int(spot["repeat0"].max()) + 1
    for i, (train, val) in enumerate(spotcheck_folds(spot)):
        rows += partition_sizes(labels_E, SPOTCHECK, f"r{i // n_splits}f{i % n_splits}",
                                {"training": train, "validation": val})
    for k, (train, test) in enumerate(outer_folds(nested)):
        rows += partition_sizes(labels_M, NESTED, f"k{k}", {"training": train, "test": test})
        for j, (tr, val) in enumerate(inner_folds(nested, k)):
            # inner positions are relative to the outer training set: map them back to positions in M
            rows += partition_sizes(labels_M, NESTED, f"k{k}j{j}", {"training": train[tr], "validation": train[val]})
    for k, (train, test) in enumerate(folds(capture["capture_outer"].to_numpy())):
        rows += partition_sizes(labels_M, CAPTURE, f"k{k}", {"training": train, "test": test})
    return pl.DataFrame(rows)


def summary(sizes: pl.DataFrame) -> dict:
    """Range of the partition sizes of each design and set, for the report."""
    out = {}
    # iterations named k<k>j<j> belong to an inner loop
    for (design, part, inner), g in sizes.with_columns(pl.col("iteration").str.contains("j").alias("inner")) \
            .group_by("design", "set", "inner", maintain_order=True):
        key = f"{design}{' inner' if inner else ''}"
        out.setdefault(key, {})[part] = {"n_iterations": g.height, "n_instances": [int(g["n_instances"].min()),
                                                                                   int(g["n_instances"].max())],
                                         "min_per_label": int(g["min_per_label"].min())}
    return out


def main() -> None:
    """Draw, check and write the frozen fold assignments of E and M, plus the partition sizes and the report."""
    cfg = load_config()
    cols, v, seed = cfg["columns"], cfg["validation"], int(cfg["seed"])
    keys, label = cols["provenance"], cols["label"]
    sources = {name: path(cfg, f"sample_{name}") for name in ("E", "M")}
    missing = [str(p) for p in sources.values() if not p.exists()]
    if missing:
        raise SystemExit(f"{missing} not found; run `make sample` first")
    out_dir, splits_dir = path(cfg, "results_dir") / "splits", path(cfg, "splits_dir")
    E, M = (pl.read_parquet(sources[name], columns=[*keys, label]) for name in ("E", "M"))
    # each source file is one capture
    lab_E, lab_M, cap_M = E[label].to_numpy(), M[label].to_numpy(), M[SOURCE_FILE].to_numpy()

    with recorded_run(cfg, out_dir, "make_splits") as run:
        spot = spotcheck_assignment(lab_E, v["spotcheck"]["n_splits"], v["spotcheck"]["n_repeats"], seed)
        outer = kfold_assignment(lab_M, v["outer"]["n_splits"], seed)
        inner = inner_assignment(lab_M, outer, v["inner"]["n_splits"], seed)
        capture, grouped = capture_assignment(lab_M, cap_M, outer)

        for r in range(spot.shape[1]):
            check_kfold(lab_E, spot[:, r], v["spotcheck"]["n_splits"], f"spot-checking repeat {r}")
        check_kfold(lab_M, outer, v["outer"]["n_splits"], "outer loop")
        check_nested(lab_M, outer, inner, v["inner"]["n_splits"])
        check_capture(lab_M, cap_M, outer, capture, grouped)

        int8 = lambda name, values: pl.Series(name, values, dtype=pl.Int8)  # noqa: E731
        tables = {
            SPOTCHECK: E.with_columns(int8(f"repeat{r}", spot[:, r]) for r in range(spot.shape[1])),
            NESTED: M.with_columns(int8("outer", outer), *(int8(f"inner{k}", inner[:, k])
                                                           for k in range(inner.shape[1]))),
            CAPTURE: M.with_columns(int8("capture_outer", capture),
                                    pl.Series("grouped", np.isin(lab_M, grouped))),
        }
        files = {}
        for name, table in tables.items():
            target = splits_dir / f"{name}.csv.gz"
            write_csv_gz(table, target)
            files[rel(target)] = sha256_file(target)

        sizes = size_table(lab_E, lab_M, tables[SPOTCHECK], tables[NESTED], tables[CAPTURE])
        out_dir.mkdir(parents=True, exist_ok=True)
        sizes.write_csv(out_dir / "partition_sizes.csv")
        captures_per_label = (M.filter(pl.col(label).is_in(grouped)).group_by(label)
                              .agg(pl.col(SOURCE_FILE).n_unique()).sort(label))
        write_json({
            "source": {rel(p): sha256_file(p) for p in sources.values()},
            "design": {
                SPOTCHECK: {"cv": "RepeatedStratifiedKFold", **v["spotcheck"], "stratified_by": label,
                            "role": "training × validation (choice of finalists)"},
                NESTED: {"outer": {"cv": "StratifiedKFold", **v["outer"], "role": "training × test"},
                         "inner": {"cv": "StratifiedKFold", **v["inner"], "role": "training × validation",
                                   "drawn_on": "training set of each outer fold"},
                         "stratified_by": label},
                CAPTURE: {"cv": "GroupKFold by source_file for the grouped labels; outer folds of nested_M "
                                "for the others",
                          "rule": "a label is grouped when it has at least as many captures as outer folds",
                          "grouped_labels": dict(captures_per_label.iter_rows()),
                          "n_grouped_instances": int(np.isin(lab_M, grouped).sum()),
                          "status": "proposal (D8 (a), diary 2026-11-05)"},
                "seed": seed,
            },
            "checks": {"max_label_spread_between_folds": {
                SPOTCHECK: max(label_spread(lab_E, spot[:, r]) for r in range(spot.shape[1])),
                f"{NESTED} outer": label_spread(lab_M, outer),
                f"{NESTED} inner": max(label_spread(lab_M[outer != k], inner[outer != k, k])
                                       for k in range(inner.shape[1]))}},
            "sizes": summary(sizes),
            "files_sha256": files,
        }, out_dir / "splits_report.json")
        run["outputs"] = files | {rel(out_dir / f): sha256_file(out_dir / f)
                                  for f in ("partition_sizes.csv", "splits_report.json")}
    print(f"wrote {', '.join(files)}; sizes in {rel(out_dir / 'partition_sizes.csv')}")


if __name__ == "__main__":
    main()
