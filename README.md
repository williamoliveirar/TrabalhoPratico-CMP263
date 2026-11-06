# Binary attack detection in IoT networks with CICIoT2023

Final project for **CMP263 – Machine Learning** (PPGC/UFRGS, 2026/2). We build and evaluate supervised models that separate benign from malicious IoT traffic in the CICIoT2023 dataset, under a leakage-free evaluation strategy: nested cross-validation with frozen folds, where the inner loop tunes hyperparameters on validation folds and the outer loop estimates performance on test folds. We ask three questions:

- **Q1 (performance):** which algorithm reaches the highest MCC, and are the differences among the best ones statistically significant?
- **Q2 (where it fails):** which attack types escape the best model (sensitivity per attack type), and what is its false positive rate on benign traffic?
- **Q3 (shortcuts):** does performance rely on features that identify the capture environment or the data processing (TTL, packet-window size) rather than the attack itself?

> **Status:** work in progress (Etapas 0–6 done; next: Etapa 7, nested cross-validation of the finalists on sample M). The full plan is in [`docs/PLANO.md`](docs/PLANO.md), the logbook with every decision and number is in [`docs/DIARIO.md`](docs/DIARIO.md), and a hands-on guide to running and validating everything is in [`docs/GUIA.md`](docs/GUIA.md). All three are in Portuguese and use the terminology of the course (glossary in `docs/PLANO.md` §1.4). Code, figures and the paper are in English.

## Machine learning workflow

The stages follow the assignment and the course (PLANO §1.3). Operations that remove instances or attributes, or apply a fixed rule, run before the split; operations that estimate parameters from data are fitted on the training set of each fold only.

| # | Stage | Where |
|---|---|---|
| 1 | Problem definition and data collection: faithful ingestion of the official CSVs | Etapas 0–1 |
| 2 | Sampling: disjoint exploration (E) and modelling (M) samples, stratified by attack type | Etapa 2 |
| 3 | Exploratory data analysis, on E only | Etapa 3 |
| 4 | Pre-processing before the split: duplicate removal, redundant and quasi-constant attributes dropped, `Std` of one-packet windows set to 0 | Etapas 2 and 4 |
| 5 | Training/validation/test split: repeated stratified k-fold on E (spot-checking); nested cross-validation on M (outer 5 folds for testing, inner 3 folds for validation) | Etapa 5 |
| 6 | Pre-processing fitted on the training folds, inside scikit-learn `Pipeline`s: one-hot encoding of the protocol code in every pipeline; log1p and standardisation for the scale-sensitive algorithms | Etapa 4 |
| 7 | Spot-checking of 8 algorithms with default hyperparameters, plus a dummy baseline; training MCC recorded to diagnose overfitting | Etapa 6 |
| 8 | Hyperparameter tuning and model selection: inner loop of the nested cross-validation | Etapa 7 |
| 9 | Performance estimation and model comparison: outer test folds, hypothesis tests | Etapas 7–8 |
| 10 | Interpretation of the best model and shortcut ablation | Etapa 9 |
| 11 | Final model: tuned and retrained on all of M; its performance estimate is the nested cross-validation one | Etapa 10 |

## Data

| Item | Value |
|---|---|
| Dataset | CICIoT2023 — Neto et al., *Sensors* 23(13):5941, 2023, [doi:10.3390/s23135941](https://doi.org/10.3390/s23135941) |
| Source of record | Official page: <https://www.unb.ca/cic/datasets/iotdataset-2023.html> |
| Version used | Per-capture CSVs (`CSV/<label>/<capture>.pcap.csv`): 309 files, 39 columns, 46,776,700 rows. The label comes from the folder name |
| Cross-check only | Kaggle cleaned mirror [`dhoogla/ciciotdataset2023`](https://www.kaggle.com/datasets/dhoogla/ciciotdataset2023) (single Parquet built from the same 309 CSVs), used to cross-check our own ingestion |

The raw data is **not** redistributed here. To reproduce:

1. Download the `CSV` folder from the official page and place it at `data/raw/CSV/` (one sub-folder per label).
   If you keep the data elsewhere, copy `configs/local.example.yaml` to `configs/local.yaml` and set the paths there.
2. Optional: download the Kaggle mirror into `data/external/` to run `make check-mirror` and `make check-ingest`.
3. `make ingest` writes `data/raw_manifest.csv` with the SHA-256 of every file. Compare it with the versioned copy to make sure you have the same data.

## Environment

- Python 3.12.7. Dependencies pinned in `requirements.txt`; top-level list in `requirements.in`.
- `make env` creates the virtual environment (default `.venv`) and installs the pinned versions.
- Windows notes:
  - GNU Make is not bundled. Install it with `winget install ezwinports.make`, or run the Python command shown under each target in the `Makefile`.
  - If the repository sits in a deep folder, `pip install` fails on the 260-character `MAX_PATH` limit. Either enable Windows long paths, or put the environment on a short path by creating a `local.mk` (not versioned) with `VENV := C:/Users/<you>/.venvs/cmp263`.
  - Notebooks are opened in VS Code through `ipykernel`; install `jupyter` if you prefer the browser UI.

## Pipeline

| Target | Stage | What it does |
|---|---|---|
| `make ingest` | 1 | Raw CSVs → one typed Parquet with values unaltered (labels, source file and row added); manifest with SHA-256; malformed rows logged |
| `make inspect` | 1 | Label counts per file and in total, schema, missing and infinite values |
| `make sample` | 2 | Deterministic exploration (E) and modelling (M) samples, deduplicated and disjoint |
| `make eda` | 3 | EDA tables and figures, on sample E only |
| — | 4 | Pre-processing pipelines (`src/models/pipelines.py`, `src/models/registry.py`); no target of its own, checked by `make test` |
| `make splits` | 5 | Frozen training/validation/test folds: spot-checking on E, outer (test) and inner (validation) folds on M, and complementary leave-capture-out folds on M (proposal) |
| `make spotcheck` | 6 | 8 algorithms (logistic regression, Gaussian NB, kNN, decision tree, RBF SVM, MLP, random forest, LightGBM) + dummy baseline with default hyperparameters on the frozen folds of E |
| `make tune` | 7 | Hyperparameter tuning and model selection with nested CV (5 outer × 3 inner folds, randomized search) for the 3 finalists on M |
| `make evaluate` | 8 | Performance estimates on the outer test folds, corrected resampled t-test, per-attack sensitivity, PR/ROC |
| `make interpret` / `make ablation` | 9 | Permutation importance, SHAP, shortcut ablation |
| `make final` / `make report` | 10 | Final model and every table and figure of the paper |
| `make all` | — | Everything above, from the raw CSVs |
| `make test` | — | Ingestion, sampling, EDA, data-leakage, split and spot-checking tests |
| `make check-mirror` / `make check-ingest` | — | Optional cross-checks against the Kaggle mirror |

## Reproducibility rules

- One seed and every path and size live in `configs/*.yaml`; no magic numbers in code.
- Folds are generated once, before any model is trained, and versioned (`data/splits/`). All algorithms use the same folds. Each file stores the fold of every instance in the row order of its sample, and the loaders refuse a sample in a different order (`tests/test_splits.py`).
- Pre-processing has two parts. Steps with no parameter estimated from data (dropping redundant columns, filling `Std` of one-packet windows with 0) run before the split, row by row. One-hot encoding and standardisation are fitted inside scikit-learn `Pipeline`s on the training rows of each fold only, and applied unchanged to its validation or test rows (`tests/test_no_leakage.py`).
- Outer test folds never inform a decision: not the pre-processing, the attributes, the hyperparameters nor the threshold.
- Every run writes a `<script>.run.json` next to its results, with timestamp, git commit, config hash, library versions, seed, runtime and output hashes.
- Results are never overwritten silently. Every number in the paper points to the result file that produced it.
- Text files use LF line endings (`.gitattributes`), so result hashes match across Windows and Linux.

## Repository layout

```
configs/      base.yaml (seed, paths, sizes), features.yaml, search_spaces.yaml
data/         raw/, external/, interim/, processed/ (not versioned); raw_manifest.csv, sample_index.csv.gz, splits/ (versioned)
src/          data/, features/, models/, experiments/, evaluation/, interpretation/, reporting/, utils/
notebooks/    exploration and visualisation only; logic lives in src/
results/      inspection/, sampling/, eda/, spotcheck/, nested/, ablation/, interpretation/, figures/, tables/
models/       final model and metadata
tests/        ingestion, sampling, EDA, data-leakage, split and spot-checking tests
docs/         PLANO.md, DIARIO.md, IA_LOG.md, enunciado.pdf, artigo/
```

## Paper figures and tables

A map from each figure and table of the paper to the script that generates it will be added in Etapa 10.

## AI usage

Generative AI tools (Claude, Claude Code) were used as assistants. Prompts, what was kept and what was rejected are logged in [`docs/IA_LOG.md`](docs/IA_LOG.md), as required by the course.
