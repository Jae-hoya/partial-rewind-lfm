"""M2 §2·§5 단위 테스트. 모델 없이 돈다 — nli3/embed 를 주입한다.

여기서 지키는 것은 스펙의 **불변식**이다. 수치가 맞는지가 아니라, 어길 수 없어야
하는 것을 어길 수 없는지를 본다.
"""

from __future__ import annotations

import pytest

from lfm.graph import (BudgetExhausted, DeltaG, Edge, LFMGraph, Node, NodeKind, State)
from lfm.relation import RelationExtractor, Thresholds, _top_q_mean


# ---------------------------------------------------------------- graph
def _g(budget=10, k=2):
    g = LFMGraph(budget=budget, k_verify=k)
    g.add_branch("b1", ["q1", "q2"])
    g.add_branch("b2", ["q1"])
    g.add_relation_edge("b1", "b2")
    return g


def test_delta_size_is_sum_of_three_parts():
    d = DeltaG(added_nodes=[Node("x", NodeKind.SUBQ, "b1")],
               added_edges=[Edge("b1", "x", "contains")],
               state_changes={"b1": State.FROZEN})
    assert (d.dV, d.dE, d.dS) == (1, 1, 1)
    assert len(d) == 3
    assert d.to_dict()["size"] == 3


def test_freeze_monotonicity_is_enforced():
    g = _g()
    g.apply(DeltaG(state_changes={"b1": State.FROZEN}))
    assert g.frozen == {"b1"}
    with pytest.raises(AssertionError, match="FREEZE 단조성"):
        g.apply(DeltaG(state_changes={"b1": State.ACTIVE}))


def test_verify_cap_is_enforced():
    g = _g(k=2)
    for _ in range(2):
        g.apply(DeltaG(calls=2), verify_edge=("b1", "b2"))
    with pytest.raises(AssertionError, match="VERIFY 상한"):
        g.apply(DeltaG(calls=2), verify_edge=("b1", "b2"))


def test_verify_cap_is_per_edge_and_order_insensitive():
    g = _g(k=1)
    g.apply(DeltaG(calls=1), verify_edge=("b1", "b2"))
    with pytest.raises(AssertionError):
        g.apply(DeltaG(calls=1), verify_edge=("b2", "b1"))     # 순서만 바꿔도 같은 edge


def test_budget_counts_generation_calls_only():
    g = _g(budget=3)
    g.apply(DeltaG(calls=2))
    assert g.budget_remaining == 1
    g.apply(DeltaG(calls=0, state_changes={"b1": State.FROZEN}))   # FREEZE 는 비용 0
    assert g.budget_remaining == 1
    with pytest.raises(BudgetExhausted):
        g.apply(DeltaG(calls=2))


def test_round_bound_from_proposition_1():
    g = LFMGraph(budget=2, c_min=1)
    g.add_branch("b1")
    g.apply(DeltaG(calls=0))
    g.apply(DeltaG(calls=0))
    with pytest.raises(AssertionError, match="라운드 상한"):
        g.apply(DeltaG(calls=0))


def test_termination_reasons():
    g = _g(budget=1)
    assert g.terminated() is None
    assert g.terminated(candidates_available=False) == "no_action"
    g.apply(DeltaG(calls=1))
    assert g.terminated() == "budget"
    g2 = _g()
    g2.apply(DeltaG(state_changes={"b1": State.FROZEN, "b2": State.PRUNED}))
    assert g2.terminated() == "A=empty"


def test_invariants_pass_on_clean_run():
    g = _g()
    g.apply(DeltaG(calls=2), verify_edge=("b1", "b2"))
    g.apply(DeltaG(state_changes={"b1": State.FROZEN}))
    g.check_invariants()


# ------------------------------------------------------------- relation
def _nli_const(ent, con):
    return lambda pairs: [(ent, con)] * len(pairs)


def test_top_q_mean_ignores_the_zero_mass():
    # 무관 pair 가 대부분이라는 상황. 전체 평균은 신호를 덮는다.
    vals = [0.9, 0.8] + [0.0] * 18
    assert _top_q_mean(vals, 0.25) == pytest.approx((0.9 + 0.8 + 0 + 0 + 0) / 5)
    assert sum(vals) / len(vals) == pytest.approx(0.085)     # 전체 평균은 거의 0


def test_agreement_uses_top_quartile_not_mean():
    seq = [0.95, 0.9, 0.0, 0.0]

    def nli3(pairs):
        # 앞 두 쌍만 높게, 나머지는 0
        return [(seq[k % len(seq)], 0.0) for k in range(len(pairs))]

    rx = RelationExtractor(nli3)
    rel = rx.relate("b1", "b2", ["a", "b"], ["c", "d"])
    assert rel.n_pairs == 4
    assert rel.agreement > 0.5          # 상위 25% 라 높게 나온다
    assert rel.agreement > sum(seq) / len(seq)


def test_contradiction_is_max_and_records_argmax_pair():
    def nli3(pairs):
        out = []
        for k in range(len(pairs)):
            out.append((0.0, 0.9 if k == 2 else 0.1))
        return out

    rx = RelationExtractor(nli3)
    rel = rx.relate("b1", "b2", ["a", "b"], ["c", "d"])
    assert rel.contradiction == pytest.approx(0.9)
    assert rel.argmax_pair == (1, 0)      # 세 번째 쌍 = claims_i[1] x claims_j[0]


def test_redundancy_needs_both_agreement_and_cosine():
    rx = RelationExtractor(_nli_const(0.95, 0.0), embed=lambda xs: [[1.0, 0.0], [1.0, 0.0]])
    rel = rx.relate("b1", "b2", ["a"], ["b"], answer_i="x", answer_j="y")
    assert rel.redundancy == 1            # agreement 높고 cos=1.0

    rx2 = RelationExtractor(_nli_const(0.95, 0.0), embed=lambda xs: [[1.0, 0.0], [0.0, 1.0]])
    rel2 = rx2.relate("b1", "b2", ["a"], ["b"], answer_i="x", answer_j="y")
    assert rel2.redundancy == 0           # agreement 는 높지만 cos=0

    rx3 = RelationExtractor(_nli_const(0.1, 0.0), embed=lambda xs: [[1.0, 0.0], [1.0, 0.0]])
    rel3 = rx3.relate("b1", "b2", ["a"], ["b"], answer_i="x", answer_j="y")
    assert rel3.redundancy == 0           # cos 는 높지만 agreement 가 낮다


def test_complementarity_is_zero_when_redundant():
    rx = RelationExtractor(_nli_const(0.95, 0.0), embed=lambda xs: [[1.0], [1.0]])
    rel = rx.relate("b1", "b2", ["a"], ["b"], answer_i="x", answer_j="y",
                    relatedness=1.0, coverage_gain=1.0)
    assert rel.redundancy == 1
    assert rel.complementarity == 0.0


def test_empty_claims_give_zero_relation():
    rx = RelationExtractor(_nli_const(0.9, 0.9))
    rel = rx.relate("b1", "b2", [], ["c"])
    assert rel.n_pairs == 0
    assert rel.agreement == 0.0 and rel.contradiction == 0.0
    assert rel.argmax_pair is None


def test_negative_control_flags_failure_without_touching_thresholds():
    # 교란 쌍인데 contradiction 이 높게 나오는 상황 — 통과하면 안 된다.
    rx = RelationExtractor(_nli_const(0.0, 0.99), th=Thresholds())
    claims = {"t%d" % k: [["c1", "c2"]] for k in range(15)}
    out = rx.negative_control(claims, shift=7)
    assert out["n"] == 15
    assert out["contradiction_gt_tau_c_pct"] == pytest.approx(100.0)
    assert out["pass"] is False
    assert "tau 를 올리지 말고" in out["note"]
    assert rx.th.tau_c == 0.7             # 임계는 그대로여야 한다


def test_negative_control_passes_when_shuffled_pairs_are_quiet():
    rx = RelationExtractor(_nli_const(0.0, 0.0))
    claims = {"t%d" % k: [["c1"]] for k in range(15)}
    out = rx.negative_control(claims, shift=7)
    assert out["pass"] is True


def test_negative_control_asserts_no_self_assignment():
    rx = RelationExtractor(_nli_const(0.0, 0.0))
    claims = {"t%d" % k: [["c"]] for k in range(7)}
    with pytest.raises(AssertionError, match="같은 토픽"):
        rx.negative_control(claims, shift=7)    # 7 % 7 == 0 이면 자기 자신
