"""Etapa 10: the summary tables and the figure index."""
import json

import polars as pl
import pytest

from src.reporting.figures import figure_index, make_targets
from src.reporting.tables import markdown, round_floats
from src.utils.io import ROOT, sha256_file


def test_rounding_keeps_small_p_values():
    df = round_floats(pl.DataFrame({"mcc": [0.831934], "p_holm": [6.2088e-6]}))
    assert df.row(0) == (0.8319, 6.2e-6)


def test_markdown_escapes_pipes_and_blanks_missing_cells():
    md = markdown(pl.DataFrame({"a": ["x|y", None], "b": [1, 2]}))
    assert md.splitlines() == ["| a | b |", "|---|---|", "| x\|y | 1 |", "|  | 2 |"]


def test_make_targets_follow_the_makefile():
    targets = make_targets(ROOT / "Makefile")
    assert targets["src.experiments.ablation"] == "ablation"
    assert targets["src.reporting.figures"] == "report"
    assert targets["src.data.eda"] == "eda"


def test_figure_index_needs_a_run_record_with_the_same_hash(tmp_path, monkeypatch):
    monkeypatch.setattr("src.reporting.figures.rel", lambda p: p.relative_to(tmp_path).as_posix())
    (tmp_path / "stage").mkdir()
    fig = tmp_path / "stage" / "fig_a.png"
    fig.write_bytes(b"png bytes")
    makefile = tmp_path / "Makefile"
    makefile.write_text("stage:\n\t$(PY) -m src.stage\n", encoding="utf-8")
    record = {"script": "src.stage", "git_commit": "abc", "outputs": {"stage/fig_a.png": sha256_file(fig)}}
    (tmp_path / "stage" / "stage.run.json").write_text(json.dumps(record), encoding="utf-8")
    index = figure_index(tmp_path, makefile)
    assert index.select("figure", "script", "make_target").row(0) == ("stage/fig_a.png", "src.stage", "stage")
    fig.write_bytes(b"redrawn by hand")
    with pytest.raises(SystemExit):
        figure_index(tmp_path, makefile)
