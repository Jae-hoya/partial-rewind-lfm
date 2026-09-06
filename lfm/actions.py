r"""SPEC_M2 §4 — Action 의 구조 효과와 실행.

각 action 이 그래프에 무엇을 하는지가 여기서 정해진다. **구조 효과(ΔG)와 실행(호출)을
분리한다.** 그래야 Test C 에서 "첫 action 이 오라클 표적을 맞혔는가"를 실행 없이도
채점할 수 있고, 예산 계산이 결정론이 된다.

| action | ΔG | 호출 |
|---|---|---|
| FREEZE i | ΔS: i → FROZEN | 0 |
| EXPAND (i,Q) | ΔV>0: sub-question 노드 재실행 | \|Q\| |
| VERIFY (i,j) | ΔV>0: **verification subgraph 신규 생성** | 2 |
| PRUNE i | ΔV<0: i → PRUNED | 0 |

FREEZE 와 PRUNE 은 artifact 를 **보존**한다. 지우지 않는다 — PRUNE 은 합성 단계에서
제외할 뿐이다. 되돌릴 수 없는 삭제를 제어 알고리즘이 하게 두면 안 된다.

`UNRESOLVED_CONFLICT`: 같은 edge 를 K=2 회 VERIFY 하고도 contradiction 이 안 내려가면
두 branch 를 그 상태로 두고 **disagreement report** 를 낸다. 억지로 하나를 고르지
않는다 — 문구는 "현재 evidence budget 내 미해소".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .graph import DeltaG, Edge, LFMGraph, Node, NodeKind, State
from .localize import ActionKind, Candidate

__all__ = ["ExecResult", "plan_delta", "apply_candidate", "disagreement_report"]

UNRESOLVED_TEXT = "현재 evidence budget 내 미해소"


@dataclass
class ExecResult:
    delta: DeltaG
    executed: bool
    note: str = ""


def _vnode(edge: tuple[str, str], seq: int) -> str:
    return "verify/%s|%s/%d" % (edge[0], edge[1], seq)


def plan_delta(g: LFMGraph, c: Candidate) -> DeltaG:
    """action 의 **구조 효과만** 만든다. 실행(호출)은 하지 않는다.

    Test C 의 localization accuracy 는 첫 action 의 (종류, 표적)만 보므로 이 함수만으로
    채점된다 — 생성 호출을 아끼는 설계다.
    """
    if c.kind == ActionKind.FREEZE:
        return DeltaG(state_changes={c.branch: State.FROZEN}, calls=0)

    if c.kind == ActionKind.PRUNE:
        return DeltaG(state_changes={c.branch: State.PRUNED}, calls=0)

    if c.kind == ActionKind.EXPAND:
        # sub-question 노드를 재실행한다. 노드가 새로 생기는 것이 아니라
        # 재실행된 버전이 붙는다 — 원본 artifact 는 보존한다.
        nodes, edges = [], []
        for q in c.subqs:
            nid = "%s#redo%d" % (q, g.rounds)
            nodes.append(Node(nid, NodeKind.SUBQ, parent=c.branch))
            edges.append(Edge(c.branch, nid, "contains"))
        return DeltaG(added_nodes=nodes, added_edges=edges, calls=c.calls)

    if c.kind == ActionKind.VERIFY:
        # verification subgraph 를 **새로** 만든다 (스펙 §4). 기존 노드를 고치지 않는다.
        e = c.edge
        seq = g.verify_count.get(tuple(sorted(e)), 0)
        vid = _vnode(tuple(sorted(e)), seq)
        return DeltaG(
            added_nodes=[Node(vid, NodeKind.VERIFICATION, parent=None)],
            added_edges=[Edge(e[0], vid, "contains"), Edge(e[1], vid, "contains")],
            calls=2)

    raise ValueError("모르는 action: " + repr(c.kind))


def apply_candidate(g: LFMGraph, c: Candidate,
                    execute: Callable[[Candidate], str] | None = None) -> ExecResult:
    """ΔG 를 만들고 그래프에 적용한다. `execute` 가 있으면 실제 재실행까지 한다.

    `execute` 를 주입식으로 둔 이유는 Test C-1 이 **첫 action 만** 채점하기 때문이다.
    채점만 할 때는 None 으로 두면 호출이 0 이 된다.
    """
    d = plan_delta(g, c)
    if not g.can_afford(d.calls):
        return ExecResult(d, False, "예산 부족: 남은 %d, 필요 %d" % (g.budget_remaining, d.calls))
    note = ""
    if execute is not None and d.calls:
        note = execute(c) or ""
    g.apply(d, verify_edge=c.edge if c.kind == ActionKind.VERIFY else None)
    return ExecResult(d, True, note)


def mark_unresolved(g: LFMGraph, edge: tuple[str, str]) -> DeltaG:
    """K 회 VERIFY 후에도 안 풀린 edge. 두 branch 를 UNRESOLVED_CONFLICT 로."""
    d = DeltaG(state_changes={edge[0]: State.UNRESOLVED_CONFLICT,
                              edge[1]: State.UNRESOLVED_CONFLICT}, calls=0)
    g.apply(d)
    return d


def disagreement_report(edge: tuple[str, str],
                        claims_i: Sequence[str], claims_j: Sequence[str],
                        argmax_pair: tuple[int, int] | None,
                        contradiction: float) -> dict:
    """관점별 주장과 근거를 **병기**한다. 하나를 고르지 않는다."""
    a = b = None
    if argmax_pair:
        ia, ib = argmax_pair
        a = claims_i[ia] if 0 <= ia < len(claims_i) else None
        b = claims_j[ib] if 0 <= ib < len(claims_j) else None
    return {"edge": list(edge), "status": "UNRESOLVED_CONFLICT",
            "contradiction": round(float(contradiction), 6),
            "claim_from_" + edge[0]: a, "claim_from_" + edge[1]: b,
            "text": UNRESOLVED_TEXT}
