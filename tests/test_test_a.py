"""Test A 집계 로직. LLM·NLI 없이 순수 함수만 검증한다."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import test_a as ta  # noqa: E402

from lfm.controller.base import Action  # noqa: E402
from lfm.evidence import EvidenceState  # noqa: E402
from lfm.pipeline import QueryResult, RoundResult  # noqa: E402
from lfm.verifier import VerifyResult  # noqa: E402

TAUS = [0.7, 0.8, 0.9]


def ev(cov: float, gr: float) -> EvidenceState:
    return EvidenceState(coverage=cov, grounding=gr, source_diversity=0.5, support_strength=0.5)


def query(qid: str, rounds_spec) -> QueryResult:
    """rounds_spec: [(action_type, {branch: (coverage, grounding)}), ...]"""
    rounds = []
    for i, (atype, branches) in enumerate(rounds_spec):
        rounds.append(
            RoundResult(
                round=i,
                evidence={b: ev(*v) for b, v in branches.items()},
                verify=VerifyResult(atype == "STOP", {}),
                action=Action(atype),
                outputs={},
                cost={"tokens_in": 0, "tokens_out": 0, "llm_calls": 0,
                      "retrievals": 0, "latency_ms": 0},
            )
        )
    return QueryResult(query_id=qid, query="q", perspectives=[], rounds=rounds, run_id="run")


# --- branch_rows ---------------------------------------------------------


def test_rows_flag_re_execution_on_retry_rounds_only():
    qr = query("q1", [("RETRY_ALL", {"a": (1.0, 1.0)}), ("STOP", {"a": (1.0, 1.0)})])
    rows = ta.branch_rows([qr], TAUS)
    assert [r["re_executed"] for r in rows] == [1, 0]


def test_sufficiency_is_labelled_at_each_tau():
    qr = query("q1", [("RETRY_ALL", {"a": (0.85, 0.85)}), ("STOP", {"a": (0.0, 0.0)})])
    row = ta.branch_rows([qr], TAUS)[0]
    assert row["sufficient@0.7"] == 1
    assert row["sufficient@0.8"] == 1
    assert row["sufficient@0.9"] == 0


def test_sufficiency_is_evaluated_before_re_execution_not_after():
    """라운드 r 의 재실행이 낭비였는지는 라운드 r 의 evidence 로 판정한다."""
    qr = query("q1", [("RETRY_ALL", {"a": (1.0, 1.0)}), ("STOP", {"a": (0.0, 0.0)})])
    rows = ta.branch_rows([qr], TAUS)
    reexec = [r for r in rows if r["re_executed"]]
    assert len(reexec) == 1
    assert reexec[0]["sufficient@0.8"] == 1, "재실행 시점(라운드 0)에는 충분했다"


# --- 낭비율 ---------------------------------------------------------------


def test_waste_ratio_counts_only_re_executed_branches():
    qr = query(
        "q1",
        [
            ("RETRY_ALL", {"a": (1.0, 1.0), "b": (0.0, 0.0)}),   # a 낭비, b 아님
            ("STOP", {"a": (1.0, 1.0), "b": (1.0, 1.0)}),        # 재실행 아님
        ],
    )
    rows = ta.branch_rows([qr], TAUS)
    per_q = ta.waste_by_query(rows, 0.8)
    assert per_q == {"q1": (1, 2)}
    assert ta.point_estimate(per_q)[0] == 0.5


def test_all_sufficient_gives_ratio_one():
    qr = query("q1", [("RETRY_ALL", {"a": (1.0, 1.0), "b": (1.0, 1.0)}), ("STOP", {"a": (0, 0)})])
    assert ta.point_estimate(ta.waste_by_query(ta.branch_rows([qr], TAUS), 0.8))[0] == 1.0


def test_none_sufficient_gives_ratio_zero():
    qr = query("q1", [("RETRY_ALL", {"a": (0.1, 0.1), "b": (0.2, 0.2)}), ("STOP", {"a": (0, 0)})])
    assert ta.point_estimate(ta.waste_by_query(ta.branch_rows([qr], TAUS), 0.8))[0] == 0.0


def test_queries_that_never_retry_are_excluded():
    passing = query("q_ok", [("STOP", {"a": (1.0, 1.0)})])
    failing = query("q_bad", [("RETRY_ALL", {"a": (0.0, 0.0)}), ("STOP", {"a": (0.0, 0.0)})])
    per_q = ta.waste_by_query(ta.branch_rows([passing, failing], TAUS), 0.8)
    assert set(per_q) == {"q_bad"}


def test_two_retry_rounds_count_branches_twice():
    qr = query(
        "q1",
        [("RETRY_ALL", {"a": (1.0, 1.0)}), ("RETRY_ALL", {"a": (1.0, 1.0)}),
         ("STOP", {"a": (0.0, 0.0)})],
    )
    assert ta.waste_by_query(ta.branch_rows([qr], TAUS), 0.8) == {"q1": (2, 2)}


def test_empty_input_is_zero_not_a_crash():
    assert ta.point_estimate({}) == (0.0, 0, 0)


# --- bootstrap ------------------------------------------------------------


def test_bootstrap_is_reproducible_from_seed():
    per_q = {"q%d" % i: (i % 3, 4) for i in range(30)}
    a = ta.bootstrap_ci(per_q, 500, 42)
    b = ta.bootstrap_ci(per_q, 500, 42)
    c = ta.bootstrap_ci(per_q, 500, 43)
    assert a == b
    assert a != c


def test_bootstrap_interval_brackets_the_point_estimate():
    per_q = {"q%d" % i: (2, 4) for i in range(40)}
    point = ta.point_estimate(per_q)[0]
    lo, hi = ta.bootstrap_ci(per_q, 1000, 42)
    assert lo <= point <= hi


def test_bootstrap_of_degenerate_data_is_a_point():
    per_q = {"q%d" % i: (4, 4) for i in range(20)}
    lo, hi = ta.bootstrap_ci(per_q, 200, 42)
    assert lo == hi == 1.0


def test_bootstrap_on_empty_set_is_nan():
    lo, hi = ta.bootstrap_ci({}, 100, 42)
    assert lo != lo and hi != hi  # NaN


def test_bootstrap_resamples_queries_not_rows():
    """쿼리 단위여야 한다. 행 단위면 branch 상관을 무시해 CI 가 좁아진다.

    한 쿼리가 전부 낭비, 다른 쿼리가 전부 비낭비인 극단 구성에서는 CI 가 넓어야 한다.
    """
    per_q = {"a": (4, 4), "b": (0, 4), "c": (4, 4), "d": (0, 4)}
    lo, hi = ta.bootstrap_ci(per_q, 2000, 42)
    assert hi - lo > 0.4


# --- 판정 밴드 -------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expect",
    [(0.75, "60%"), (0.60, "60%"), (0.50, "40~60%"), (0.40, "40~60%"),
     (0.30, "20~40%"), (0.20, "20~40%"), (0.10, "20% 미만"), (0.0, "20% 미만")],
)
def test_verdict_bands_match_masterplan(value, expect):
    assert expect in ta.verdict(value)
