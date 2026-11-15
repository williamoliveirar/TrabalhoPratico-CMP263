# Pipeline targets (docs/PLANO.md §5.3). Each target runs one stage; `make all` runs everything from the raw CSVs.
# Skeleton: most modules are created in the stage named next to the target.

-include local.mk  # optional, not versioned; e.g. `VENV := C:/Users/me/.venvs/cmp263` (short path, Windows MAX_PATH)
VENV ?= .venv
ifeq ($(OS),Windows_NT)
  VENV_PY := "$(VENV)/Scripts/python.exe"
else
  VENV_PY := $(VENV)/bin/python
endif
PY ?= $(VENV_PY)
BOOTSTRAP_PY ?= python

.PHONY: env lock check-mirror ingest inspect check-ingest sample eda splits spotcheck tune evaluate interpret ablation final report test all

env:            ## Etapa 0 — create the virtual env and install the pinned requirements
	$(BOOTSTRAP_PY) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

lock:           ## Etapa 0 — reinstall from requirements.in and re-pin requirements.txt (record in DIARIO)
	$(PY) -m pip install -r requirements.in
	$(PY) -m pip freeze > requirements.txt

check-mirror:   ## D15 — compare the Kaggle Parquet with the official CSVs
	$(PY) -m src.data.check_kaggle_mirror

ingest:         ## Etapa 1 (D15) — raw CSVs → data/interim Parquet (values unaltered) + data/raw_manifest.csv
	$(PY) -m src.data.ingest

inspect:        ## Etapa 1 — label counts, schema, missing/infinite values (reads the ingested Parquet)
	$(PY) -m src.data.inspect_raw

check-ingest:   ## D15 — compare the ingested Parquet with the Kaggle mirror, row by row (optional)
	$(PY) -m src.data.check_ingest

sample:         ## Etapa 2 (D3, D4) — samples E and M: deduplicated, disjoint, byte-identical across runs
	$(PY) -m src.data.make_sample

eda:            ## Etapa 3 — EDA tables and figures (sample E only)
	$(PY) -m src.data.eda

splits:         ## Etapa 5 — training/validation/test folds (E: spot-checking; M: nested CV, leave-capture-out), frozen
	$(PY) -m src.experiments.make_splits

spotcheck:      ## Etapa 6 — spot-checking on sample E
	$(PY) -m src.experiments.spotcheck

tune:           ## Etapa 7 — nested CV on sample M
	$(PY) -m src.experiments.nested_cv

evaluate:       ## Etapa 8 — comparison, statistical tests, per-attack recall
	$(PY) -m src.evaluation.evaluate

interpret:      ## Etapa 9 — permutation importance and SHAP
	$(PY) -m src.interpretation.permutation
	$(PY) -m src.interpretation.shap_analysis

ablation:       ## Etapa 9 — shortcut ablation (nested CV without the candidates), one-feature models, Q3/H3 (after interpret)
	$(PY) -m src.experiments.ablation
	$(PY) -m src.experiments.shortcuts

final:          ## Etapa 10 — final model trained on all of M; Etapa 9 leave-capture-out with its configuration
	$(PY) -m src.experiments.final_model
	$(PY) -m src.experiments.leave_capture_out

report:         ## Etapa 10 — summary tables (CSV and Markdown) and the index of the figures
	$(PY) -m src.reporting.tables
	$(PY) -m src.reporting.figures

test:
	$(PY) -m pytest -q tests

all: ingest inspect sample eda splits spotcheck tune evaluate interpret ablation final report
