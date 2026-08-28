from __future__ import annotations

import io
import json

import pytest

from lfm.trace import (
    STATE_LABELS,
    TraceWriter,
    budget_left,
    make_action,
    make_cost,
    make_record,
    make_state,
    new_run_id,
    read_trace,
)


def sample_record(round_: int = 0, max_rounds: int = 3) -> dict:
    state = make_state(
        evidence={
            "p1": {
                "coverage": 0.62,
                "grounding": 0.81,
                "source_diversity": 0.50,
                "support_strength": 0.74,
            }
        },
        budget=budget_left(round_, max_rounds),
        vertices=["p1", "p2", "p3", "p4"],
    )
    return make_record(
        run_id=new_run_id(),
        query_id="q0042",
        round_=round_,
        state=state,
        action=make_action("RETRY_ALL", params={"top_k": 10, "directive_id": 1}),
        cost=make_cost(tokens_in=100, tokens_out=50, llm_calls=2, retrievals=8, latency_ms=1234),
    )


def test_record_has_every_frozen_top_level_key():
    rec = sample_record()
    assert set(rec) == {
        "run_id",
        "query_id",
        "round",
        "state",
        "action",
        "cost",
        "reward",
        "next_state",
    }


def test_reward_is_always_null():
    assert sample_record()["reward"] is None


def test_m2_fields_exist_but_are_empty_in_m1():
    """R / verify_count / frozen 은 M1 에서 비지만 필드는 반드시 존재한다."""
    state = sample_record()["state"]
    assert state["R"] == []
    assert state["verify_count"] == {}
    assert state["frozen"] == []


def test_default_status_is_all_active():
    state = make_state(vertices=["p1", "p2"])
    assert state["graph"]["S"] == {"p1": "ACTIVE", "p2": "ACTIVE"}


def test_unknown_state_label_is_rejected():
    with pytest.raises(ValueError):
        make_state(vertices=["p1"], status={"p1": "HALF_FROZEN"})


def test_all_frozen_labels_are_accepted():
    for label in STATE_LABELS:
        make_state(vertices=["p1"], status={"p1": label})


def test_budget_left_is_deterministic_and_decreasing():
    vals = [budget_left(r, 3) for r in range(3)]
    assert vals == sorted(vals, reverse=True)
    assert vals[-1] == 0.0
    assert budget_left(0, 3) == budget_left(0, 3)
    with pytest.raises(ValueError):
        budget_left(0, 0)


def test_cost_fields_are_ints():
    cost = make_cost(tokens_in=1.9, tokens_out=2, llm_calls=1, retrievals=0, latency_ms=5)
    assert all(isinstance(v, int) for v in cost.values())


def test_action_params_records_round_perturbation():
    a = make_action("RETRY_ALL", target=["p1"], params={"top_k": 15})
    assert a["params"]["top_k"] == 15
    assert a["target"] == ["p1"]


def test_writer_emits_one_utf8_json_line_per_record(tmp_path):
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path) as w:
        w.write(sample_record(0))
        w.write(sample_record(1))
        assert w.count == 2

    raw = io.open(path, encoding="utf-8").read()
    assert raw.count("\n") == 2
    recs = list(read_trace(path))
    assert [r["round"] for r in recs] == [0, 1]
    assert all(json.loads(line)["reward"] is None for line in raw.strip().split("\n"))


def test_writer_truncates_by_default_and_appends_on_request(tmp_path):
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path) as w:
        w.write(sample_record())
    with TraceWriter(path) as w:
        w.write(sample_record())
    assert len(list(read_trace(path))) == 1
    with TraceWriter(path, append=True) as w:
        w.write(sample_record())
    assert len(list(read_trace(path))) == 2


def test_state_does_not_alias_caller_objects():
    evidence = {"p1": {"coverage": 1.0, "grounding": 1.0, "source_diversity": 1.0,
                       "support_strength": 1.0}}
    state = make_state(evidence=evidence, vertices=["p1"])
    evidence["p1"]["coverage"] = 0.0
    assert state["E"]["p1"]["coverage"] == 1.0
