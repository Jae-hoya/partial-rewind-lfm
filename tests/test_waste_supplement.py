"""waste_supplement.py — 파일럿 제외 보조 추정."""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import waste_supplement as ws  # noqa: E402


def write_csv(tmp_path, rows) -> Path:
    p = tmp_path / "waste_ratio.csv"
    with io.open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return p


def row(qid, suff, re_exec=1):
    return {"query_id": qid, "round": 0, "branch": "b", "re_executed": re_exec,
            "sufficient@0.7": suff, "sufficient@0.8": suff, "sufficient@0.9": suff}


def test_load_rows_coerces_re_executed_to_int(tmp_path):
    rows = ws.load_rows(write_csv(tmp_path, [row("t000", 1)]))
    assert rows[0]["re_executed"] == 1


def test_estimate_matches_hand_calculation(tmp_path):
    rows = ws.load_rows(write_csv(tmp_path, [row("t000", 1), row("t001", 0),
                                             row("t002", 1), row("t003", 0)]))
    got = ws.estimate(rows, [0.8], 200, 42)
    assert got[0]["point"] == 0.5
    assert got[0]["n_reexec"] == 4 and got[0]["n_wasted"] == 2


def test_excluding_calibration_topics_changes_the_estimate(tmp_path):
    """파일럿 토픽이 결과를 밀었다면 이 두 값이 갈린다."""
    rows = ws.load_rows(write_csv(tmp_path, [row("t000", 1), row("t001", 1),
                                             row("t009", 0), row("t010", 0)]))
    full = ws.estimate(rows, [0.8], 200, 42)[0]
    kept = [r for r in rows if r["query_id"] not in {"t000", "t001"}]
    sub = ws.estimate(kept, [0.8], 200, 42)[0]
    assert full["point"] == 0.5
    assert sub["point"] == 0.0


def test_non_re_executed_rows_are_ignored(tmp_path):
    rows = ws.load_rows(write_csv(tmp_path, [row("t000", 1, re_exec=0), row("t001", 0)]))
    got = ws.estimate(rows, [0.8], 200, 42)[0]
    assert got["n_reexec"] == 1
