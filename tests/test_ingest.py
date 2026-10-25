"""Ingestion keeps values unaltered, skips only malformed rows and keeps the original row positions."""
import math
from collections import Counter

import pytest

from src.data.ingest import read_capture
from src.data.labels import CATEGORY, FOLDER_RENAMES, label_of_folder, target_of

HEADER = ["Rate", "Std", "Number"]


def test_values_and_row_idx():
    data = (b"Rate,Std,Number\n"
            b"10.5,0.25,3\n"
            b"inf,,1\n"           # infinite rate and empty dispersion stay as they are
            b"1,2,3,4\n"          # too many fields: skipped
            b"nan,1e-05,2\n"
            b"7.0,1")             # truncated last line, no trailing newline: skipped
    table, row_idx, malformed = read_capture(data, HEADER)
    rows = table.to_pylist()
    assert row_idx.tolist() == [0, 1, 3]
    assert [m["row_idx"] for m in malformed] == [2, 4]
    assert rows[0] == {"Rate": 10.5, "Std": 0.25, "Number": 3.0}
    assert rows[1]["Rate"] == math.inf and rows[1]["Std"] is None
    assert math.isnan(rows[2]["Rate"]) and rows[2]["Std"] == 1e-05


def test_clean_file_has_no_malformed_rows():
    table, row_idx, malformed = read_capture(b"Rate,Std,Number\n1,2,3\n4,5,6\n", HEADER)
    assert table.num_rows == 2 and row_idx.tolist() == [0, 1] and malformed == []


def test_non_numeric_value_fails_loudly():
    with pytest.raises(Exception):
        read_capture(b"Rate,Std,Number\n1,abc,3\n", HEADER)


def test_label_mapping():
    assert len(CATEGORY) == 34
    # PLANO §3.3 lists 11 DDoS attacks and omits ICMP fragmentation; the data has 12 (33 attacks in total)
    assert Counter(CATEGORY.values()) == {"DDoS": 12, "DoS": 4, "Recon": 5, "Web": 6, "BruteForce": 1,
                                          "Spoofing": 2, "Mirai": 3, "Benign": 1}
    assert all(label_of_folder(f) == label for f, label in FOLDER_RENAMES.items())
    assert target_of(label_of_folder("Benign_Final")) == 0
    assert sum(target_of(label) for label in CATEGORY) == 33
    with pytest.raises(KeyError):
        label_of_folder("Unknown_Attack")
