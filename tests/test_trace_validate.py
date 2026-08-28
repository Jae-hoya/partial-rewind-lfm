from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import trace_validate as tv  # noqa: E402

from lfm.trace import (  # noqa: E402
    TraceWriter,
    budget_left,
    make_action,
    make_cost,
    make_record,
    make_state,
)


def good_record() -> dict:
    return make_record(
        run_id="run-1",
        query_id="q0001",
        round_=0,
        state=make_state(
            evidence={
                "p1": {
                    "coverage": 1.0,
                    "grounding": 1.0,
                    "source_diversity": 0.0,
                    "support_strength": 0.5,
                }
            },
            budget=budget_left(0, 3),
            vertices=["p1"],
        ),
        action=make_action("RETRY_ALL", params={"top_k": 5}),
        cost=make_cost(llm_calls=1),
    )


def write(tmp_path, records) -> Path:
    p = tmp_path / "trace.jsonl"
    with TraceWriter(p) as w:
        for r in records:
            w.write(r)
    return p


def test_valid_trace_passes(tmp_path):
    p = write(tmp_path, [good_record(), good_record()])
    n, errs = tv.validate_file(p)
    assert (n, errs) == (2, [])
    assert tv.main([str(p)]) == 0


def test_extra_fields_are_allowed(tmp_path):
    """동결 스키마는 필드 추가만 허용 — 여분 키가 오류면 M2 확장이 막힌다."""
    rec = good_record()
    rec["state"]["extra_future_field"] = 1
    rec["lfm_note"] = "hello"
    _, errs = tv.validate_file(write(tmp_path, [rec]))
    assert errs == []


@pytest.mark.parametrize("key", ["run_id", "query_id", "round", "state", "action", "cost",
                                 "next_state", "reward"])
def test_missing_top_level_key_fails(tmp_path, key):
    rec = good_record()
    del rec[key]
    _, errs = tv.validate_file(write(tmp_path, [rec]))
    assert errs and key in errs[0]


@pytest.mark.parametrize("key", ["E", "R", "budget_left", "frozen", "verify_count", "graph"])
def test_missing_state_key_fails(tmp_path, key):
    rec = good_record()
    del rec["state"][key]
    _, errs = tv.validate_file(write(tmp_path, [rec]))
    assert errs


def test_non_null_reward_fails(tmp_path):
    rec = good_record()
    rec["reward"] = 1.0
    _, errs = tv.validate_file(write(tmp_path, [rec]))
    assert any("reward" in e for e in errs)


def test_bad_state_label_fails(tmp_path):
    rec = good_record()
    rec["state"]["graph"]["S"]["p1"] = "SOMETHING"
    _, errs = tv.validate_file(write(tmp_path, [rec]))
    assert any("상태 라벨" in e for e in errs)


def test_vertex_without_status_fails(tmp_path):
    rec = good_record()
    rec["state"]["graph"]["V"].append("p2")
    _, errs = tv.validate_file(write(tmp_path, [rec]))
    assert any("graph.S" in e for e in errs)


def test_missing_evidence_metric_fails(tmp_path):
    rec = good_record()
    del rec["state"]["E"]["p1"]["grounding"]
    _, errs = tv.validate_file(write(tmp_path, [rec]))
    assert any("grounding" in e for e in errs)


def test_missing_action_params_fails(tmp_path):
    rec = good_record()
    del rec["action"]["params"]
    _, errs = tv.validate_file(write(tmp_path, [rec]))
    assert any("params" in e for e in errs)


def test_malformed_json_line_is_reported(tmp_path):
    p = tmp_path / "trace.jsonl"
    with io.open(p, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(good_record(), ensure_ascii=False) + "\n")
        fh.write("{not json\n")
    n, errs = tv.validate_file(p)
    assert n == 1 and any("JSON" in e for e in errs)


def test_cli_returns_nonzero_on_bad_file(tmp_path, capsys):
    rec = good_record()
    rec["reward"] = 0.0
    p = write(tmp_path, [rec])
    assert tv.main([str(p)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_reports_missing_file(tmp_path):
    assert tv.main([str(tmp_path / "nope.jsonl")]) == 1


def test_validate_record_does_not_mutate_input():
    rec = good_record()
    before = copy.deepcopy(rec)
    tv.validate_record(rec, "x")
    assert rec == before
