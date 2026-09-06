"""M2 §3·§4·§6·§7 단위 테스트. 모델 없이 돈다.

여기서 지키는 것은 **스펙의 규칙**이다 — 우선순위, 최소 변경 단위, PRUNE 의 안전
조건, 비교선의 공정성. 수치가 아니라 규칙이 어겨지지 않는지를 본다.
"""

from __future__ import annotations

import pytest

from lfm.actions import apply_candidate, disagreement_report, mark_unresolved, plan_delta
from lfm.controller.baselines import (GlobalVerifierController, OracleController,
                                      RandomController)
from lfm.controller.lfm import LFMController, run
from lfm.graph import LFMGraph, NodeKind, State
from lfm.localize import ActionKind, EvidenceView, localize
from lfm.relation import Relation


def ev(cov=1.0, grd=1.0, div=0.5, sup=0.5, miss=(), low=()):
    return EvidenceView(cov, grd, div, sup, tuple(miss), tuple(low))


def rel(i, j, con=0.0, agr=0.0, red=0, pair=(0, 0)):
    return Relation(i=i, j=j, agreement=agr, contradiction=con,
                    redundancy=red, argmax_pair=pair, n_pairs=4)


def graph(budget=20):
    g = LFMGraph(budget=budget)
    g.add_branch("b1", ["q1", "q2"])
    g.add_branch("b2", ["q1"])
    g.add_relation_edge("b1", "b2")
    return g


# ------------------------------------------------------------- localize
def test_sufficient_uses_strict_inequality():
    assert ev(1.0, 0.81).sufficient(0.8) is True
    assert ev(1.0, 0.80).sufficient(0.8) is False       # 초과지 이상이 아니다
    assert ev(0.8, 1.0).sufficient(0.8) is False        # coverage 는 1.0 이어야


def test_priority_is_verify_expand_prune_freeze():
    e = {"b1": ev(0.5, 0.1, miss=("b1/q1",)), "b2": ev(1.0, 1.0)}
    r = [rel("b1", "b2", con=0.9)]
    got = [c.kind for c in localize(["b1", "b2"], e, r)]
    assert got[0] == ActionKind.VERIFY
    assert got == sorted(got, key=lambda k: [ActionKind.VERIFY, ActionKind.EXPAND,
                                             ActionKind.PRUNE, ActionKind.FREEZE].index(k))


def test_contradiction_blocks_freeze_and_expand():
    e = {"b1": ev(1.0, 1.0), "b2": ev(1.0, 1.0)}
    r = [rel("b1", "b2", con=0.9)]
    kinds = [c.kind for c in localize(["b1", "b2"], e, r)]
    assert ActionKind.VERIFY in kinds
    assert ActionKind.FREEZE not in kinds          # 충돌 중인 branch 는 못 얼린다


def test_freeze_when_sufficient_and_no_contradiction():
    e = {"b1": ev(1.0, 1.0), "b2": ev(1.0, 1.0)}
    r = [rel("b1", "b2", con=0.1)]
    kinds = [c.kind for c in localize(["b1", "b2"], e, r)]
    assert kinds == [ActionKind.FREEZE, ActionKind.FREEZE]


def test_verify_respects_k_cap():
    e = {"b1": ev(1.0, 1.0), "b2": ev(1.0, 1.0)}
    r = [rel("b1", "b2", con=0.9)]
    out = localize(["b1", "b2"], e, r, verify_counts={("b1", "b2"): 2}, k_verify=2)
    assert not [c for c in out if c.kind == ActionKind.VERIFY]


def test_verify_suspect_is_lower_support_strength():
    e = {"b1": ev(1.0, 1.0, sup=0.2), "b2": ev(1.0, 1.0, sup=0.9)}
    v = [c for c in localize(["b1", "b2"], e, [rel("b1", "b2", con=0.9)])
         if c.kind == ActionKind.VERIFY][0]
    assert v.suspect == ("b1",)


def test_verify_suspect_is_both_on_tie():
    e = {"b1": ev(1.0, 1.0, sup=0.5), "b2": ev(1.0, 1.0, sup=0.5)}
    v = [c for c in localize(["b1", "b2"], e, [rel("b1", "b2", con=0.9)])
         if c.kind == ActionKind.VERIFY][0]
    assert set(v.suspect) == {"b1", "b2"}


def test_prune_only_when_both_sufficient():
    both_ok = {"b1": ev(1.0, 1.0, div=0.1), "b2": ev(1.0, 1.0, div=0.9)}
    out = localize(["b1", "b2"], both_ok, [rel("b1", "b2", agr=0.9, red=1)])
    prunes = [c for c in out if c.kind == ActionKind.PRUNE]
    assert len(prunes) == 1
    assert prunes[0].branch == "b1"                # source_diversity 낮은 쪽

    not_ok = {"b1": ev(0.5, 0.1), "b2": ev(1.0, 1.0)}
    out2 = localize(["b1", "b2"], not_ok, [rel("b1", "b2", agr=0.9, red=1)])
    assert not [c for c in out2 if c.kind == ActionKind.PRUNE]   # 정보 손실 위험


def test_coverage_zero_becomes_regen_not_breadth():
    e = {"b1": ev(0.0, 0.0, miss=("b1/q1", "b1/q2"))}
    c = [x for x in localize(["b1"], e, []) if x.kind == ActionKind.EXPAND][0]
    assert c.kind_detail == "regen"


def test_expand_targets_subquestions_not_whole_branch():
    e = {"b1": ev(0.5, 0.2, miss=("b1/q1",), low=("b1/q2",))}
    c = [x for x in localize(["b1"], e, []) if x.kind == ActionKind.EXPAND][0]
    assert c.kind_detail == "breadth"
    assert set(c.subqs) == {"b1/q1", "b1/q2"}      # 최소 변경 단위 = sub-question
    assert c.calls == 2


def test_candidates_over_budget_are_dropped():
    e = {"b1": ev(0.5, 0.2, miss=("b1/q1", "b1/q2", "b1/q3"))}
    assert localize(["b1"], e, [], budget_remaining=1) == []


def test_localize_is_deterministic():
    e = {"b1": ev(0.5, 0.1, miss=("b1/q1",)), "b2": ev(1.0, 1.0)}
    r = [rel("b1", "b2", con=0.9)]
    a = [c.to_dict() for c in localize(["b1", "b2"], e, r)]
    b = [c.to_dict() for c in localize(["b1", "b2"], e, r)]
    assert a == b


# -------------------------------------------------------------- actions
def test_freeze_and_prune_cost_nothing():
    g = graph()
    e = {"b1": ev(1.0, 1.0), "b2": ev(1.0, 1.0)}
    c = localize(["b1", "b2"], e, [rel("b1", "b2", con=0.1)])[0]
    d = plan_delta(g, c)
    assert d.calls == 0 and d.dS == 1


def test_verify_creates_a_new_verification_subgraph():
    g = graph()
    e = {"b1": ev(1.0, 1.0), "b2": ev(1.0, 1.0)}
    c = [x for x in localize(["b1", "b2"], e, [rel("b1", "b2", con=0.9)])
         if x.kind == ActionKind.VERIFY][0]
    d = plan_delta(g, c)
    assert d.calls == 2
    assert any(n.kind == NodeKind.VERIFICATION for n in d.added_nodes)
    assert d.dE == 2                                  # 두 branch 에 붙는다


def test_apply_refuses_when_over_budget():
    g = graph(budget=1)
    e = {"b1": ev(1.0, 1.0), "b2": ev(1.0, 1.0)}
    c = [x for x in localize(["b1", "b2"], e, [rel("b1", "b2", con=0.9)])
         if x.kind == ActionKind.VERIFY][0]
    res = apply_candidate(g, c)
    assert res.executed is False and "예산" in res.note


def test_unresolved_conflict_marks_both_and_reports_both_sides():
    g = graph()
    mark_unresolved(g, ("b1", "b2"))
    assert g.state["b1"] == State.UNRESOLVED_CONFLICT
    rep = disagreement_report(("b1", "b2"), ["i says X"], ["j says not X"], (0, 0), 0.9)
    assert rep["claim_from_b1"] and rep["claim_from_b2"]      # 한쪽을 고르지 않는다
    assert "미해소" in rep["text"]


# ----------------------------------------------------------- controllers
def _est(view):
    return lambda active: {b: view[b] for b in active if b in view}


def test_lfm_defers_freeze_to_the_sweep():
    g = graph()
    e = {"b1": ev(1.0, 1.0), "b2": ev(1.0, 1.0)}
    ctl = LFMController()
    assert ctl.step(g, e, [rel("b1", "b2", con=0.1)]) is None   # 라운드 중엔 안 낸다
    assert len(ctl.freeze_sweep(g, e, [rel("b1", "b2", con=0.1)])) == 2


def test_run_freezes_everything_and_holds_invariants():
    g = graph()
    e = {"b1": ev(1.0, 1.0), "b2": ev(1.0, 1.0)}
    out = run(LFMController(), g, _est(e), lambda a: [rel("b1", "b2", con=0.1)])
    assert g.frozen == {"b1", "b2"}
    assert out["spent"] == 0                       # FREEZE 는 비용 0
    g.check_invariants()


def test_global_verifier_does_not_localize():
    g = graph()
    e = {"b1": ev(0.5, 0.2, miss=("b1/q1",), low=("b1/q2",)), "b2": ev(1.0, 1.0)}
    c = GlobalVerifierController().step(g, e, [])
    assert c.kind == ActionKind.EXPAND and c.branch == "b1"
    assert "국소화 없이" in c.reason


def test_random_is_seeded_and_reproducible():
    e = {"b1": ev(0.5, 0.2, miss=("b1/q1",)), "b2": ev(0.5, 0.2, miss=("b2/q1",))}
    r = [rel("b1", "b2", con=0.9)]
    a = RandomController(seed=7).step(graph(), e, r).to_dict()
    b = RandomController(seed=7).step(graph(), e, r).to_dict()
    assert a == b


def test_oracle_returns_the_injected_target_when_reachable():
    g = graph()
    e = {"b1": ev(1.0, 1.0), "b2": ev(1.0, 1.0)}
    r = [rel("b1", "b2", con=0.9)]
    o = OracleController(truth={"kind": ActionKind.VERIFY, "edge": ["b2", "b1"]})
    c = o.step(g, e, r)
    assert c is not None and c.kind == ActionKind.VERIFY      # edge 순서 무관


def test_oracle_respects_the_rules_and_gives_up_when_unreachable():
    g = graph()
    e = {"b1": ev(1.0, 1.0), "b2": ev(1.0, 1.0)}
    r = [rel("b1", "b2", con=0.9)]
    o = OracleController(truth={"kind": ActionKind.VERIFY, "edge": ["b1", "b2"]})
    g.verify_count[("b1", "b2")] = 2                 # K 소진
    assert o.step(g, e, r) is None                   # 오라클도 규칙은 지킨다


def test_all_controllers_share_one_interface():
    g = graph()
    e = {"b1": ev(0.5, 0.2, miss=("b1/q1",)), "b2": ev(1.0, 1.0)}
    r = [rel("b1", "b2", con=0.1)]
    for ctl in (LFMController(), GlobalVerifierController(), RandomController(seed=1),
                OracleController(truth={"kind": ActionKind.EXPAND, "branch": "b1"})):
        c = ctl.step(g, e, r)
        assert c is None or hasattr(c, "to_dict")    # 우회 경로 없음
