"""SPEC §3.7 — 라운드별 (V, E, S) 스냅샷. M1 은 S 가 전부 ACTIVE 다.

M2 (SPEC_M2 §5) 가 아래쪽에 `LFMGraph` 로 들어왔다. **M1 의 `Graph` 는 건드리지
않는다** — 파이프라인·trace 가 그대로 쓰고 있고, 바꾸면 Test A 재현이 깨진다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from .trace import STATE_LABELS

__all__ = ["Graph", "State", "NodeKind", "Node", "Edge", "DeltaG", "LFMGraph",
           "BudgetExhausted"]


class Graph:
    def __init__(self, vertices: Iterable[str]) -> None:
        self.V: list[str] = list(vertices)
        self.E: list[Any] = []
        self.S: dict[str, str] = {v: "ACTIVE" for v in self.V}

    def set_state(self, vertex: str, label: str) -> None:
        if label not in STATE_LABELS:
            raise ValueError("허용되지 않은 상태 라벨: " + repr(label))
        if vertex not in self.S:
            raise KeyError("그래프에 없는 정점: " + repr(vertex))
        self.S[vertex] = label

    def active(self) -> list[str]:
        return [v for v in self.V if self.S[v] == "ACTIVE"]

    def snapshot(self) -> dict[str, Any]:
        return {"V": list(self.V), "E": list(self.E), "S": dict(self.S)}


# =====================================================================
# SPEC_M2 §5 — G_t = (V_t, E_t, S_t), ΔG, 종료
# =====================================================================
#
# LFM 의 "최소 그래프 변경" 이 무엇인지가 여기서 정해진다. Test A 의 낭비율은
# **branch** 단위였는데 LFM 의 ΔG 는 **sub-question** 단위다. 한 단계 더 작다.
#
# 스펙이 요구한 불변식 셋을 주석이 아니라 **코드로** 강제한다. 위반하면 실행 무효다.
#   Proposition 1  T <= floor(B / c_min),  VERIFY 총수 <= K * |E_R|
#   FREEZE 단조성  F_t ⊆ F_{t+1}  — 한 번 얼린 branch 는 녹지 않는다
#
# 예산 B 는 **생성 호출 수**로 센다. 토큰이 아니다 — 결정론이어야 비교가 성립한다.


class State(str, Enum):
    ACTIVE = "ACTIVE"
    FROZEN = "FROZEN"
    PRUNED = "PRUNED"
    UNRESOLVED_CONFLICT = "UNRESOLVED_CONFLICT"


class NodeKind(str, Enum):
    BRANCH = "branch"
    SUBQ = "subq"
    VERIFICATION = "verification"


TERMINAL = frozenset({State.FROZEN, State.PRUNED, State.UNRESOLVED_CONFLICT})


class BudgetExhausted(RuntimeError):
    """예산을 넘겨 호출하려 했다. 컨트롤러가 잡아 종료해야 한다."""


@dataclass(frozen=True)
class Node:
    id: str
    kind: NodeKind
    parent: str | None = None        # subq/verification 이 속한 branch


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: str                        # "contains" | "relation"


@dataclass
class DeltaG:
    """한 action 이 만든 그래프 변경. |ΔG| = |ΔV| + |ΔE| + |ΔS|."""

    added_nodes: list[Node] = field(default_factory=list)
    removed_nodes: list[str] = field(default_factory=list)
    added_edges: list[Edge] = field(default_factory=list)
    removed_edges: list[Edge] = field(default_factory=list)
    state_changes: dict[str, State] = field(default_factory=dict)
    calls: int = 0                   # 이 action 이 쓴 **생성 호출** 수

    @property
    def dV(self) -> int:
        return len(self.added_nodes) + len(self.removed_nodes)

    @property
    def dE(self) -> int:
        return len(self.added_edges) + len(self.removed_edges)

    @property
    def dS(self) -> int:
        return len(self.state_changes)

    def __len__(self) -> int:
        return self.dV + self.dE + self.dS

    def to_dict(self) -> dict[str, int]:
        return {"dV": self.dV, "dE": self.dE, "dS": self.dS,
                "size": len(self), "calls": self.calls}


class LFMGraph:
    """SPEC_M2 §5. ΔG 적용과 불변식 강제."""

    def __init__(self, budget: int, k_verify: int = 2, c_min: int = 1) -> None:
        if budget < 0:
            raise ValueError("예산은 음수일 수 없다")
        self.nodes: dict[str, Node] = {}
        self.edges: set[Edge] = set()
        self.state: dict[str, State] = {}
        self.budget = int(budget)
        self.spent = 0
        self.k_verify = int(k_verify)
        self.c_min = max(1, int(c_min))
        self.verify_count: dict[tuple[str, str], int] = {}
        self.rounds = 0
        self.history: list[DeltaG] = []
        self._frozen_seen: set[str] = set()

    # --- 구성 ---------------------------------------------------------
    def add_branch(self, bid: str, subqs: Iterable[str] = ()) -> None:
        self.nodes[bid] = Node(bid, NodeKind.BRANCH)
        self.state[bid] = State.ACTIVE
        for q in subqs:
            qid = "%s/%s" % (bid, q)
            self.nodes[qid] = Node(qid, NodeKind.SUBQ, parent=bid)
            self.edges.add(Edge(bid, qid, "contains"))

    def add_relation_edge(self, i: str, j: str) -> None:
        a, b = sorted((i, j))
        self.edges.add(Edge(a, b, "relation"))

    # --- 조회 ---------------------------------------------------------
    @property
    def active(self) -> list[str]:
        return sorted(b for b, s in self.state.items() if s == State.ACTIVE)

    @property
    def frozen(self) -> set[str]:
        return {b for b, s in self.state.items() if s == State.FROZEN}

    @property
    def relation_edges(self) -> list[tuple[str, str]]:
        return sorted((e.src, e.dst) for e in self.edges if e.kind == "relation")

    @property
    def budget_remaining(self) -> int:
        return self.budget - self.spent

    def subqs(self, bid: str) -> list[str]:
        return sorted(n.id for n in self.nodes.values()
                      if n.kind == NodeKind.SUBQ and n.parent == bid)

    def can_afford(self, calls: int) -> bool:
        return self.spent + int(calls) <= self.budget

    # --- 적용 ---------------------------------------------------------
    def apply(self, d: DeltaG, *, verify_edge: tuple[str, str] | None = None) -> None:
        if not self.can_afford(d.calls):
            raise BudgetExhausted(
                "예산 초과: 남은 %d, 요구 %d" % (self.budget_remaining, d.calls))

        for b, s in d.state_changes.items():
            if b in self._frozen_seen and s != State.FROZEN:
                raise AssertionError(
                    "FREEZE 단조성 위반: %s 가 FROZEN 에서 %s 로" % (b, s.value))

        if verify_edge is not None:
            key = tuple(sorted(verify_edge))
            n = self.verify_count.get(key, 0) + 1
            if n > self.k_verify:
                raise AssertionError(
                    "VERIFY 상한 위반: edge %s 에 %d 회 (K=%d)" % (key, n, self.k_verify))
            self.verify_count[key] = n

        for node in d.added_nodes:
            self.nodes[node.id] = node
        for nid in d.removed_nodes:
            self.nodes.pop(nid, None)
        for e in d.added_edges:
            self.edges.add(e)
        for e in d.removed_edges:
            self.edges.discard(e)
        for b, s in d.state_changes.items():
            self.state[b] = s
            if s == State.FROZEN:
                self._frozen_seen.add(b)

        self.spent += d.calls
        self.rounds += 1
        self.history.append(d)

        bound = self.budget // self.c_min
        if self.rounds > bound:
            raise AssertionError(
                "라운드 상한 위반: T=%d > floor(B/c_min)=%d" % (self.rounds, bound))

    # --- 종료 ---------------------------------------------------------
    def terminated(self, *, candidates_available: bool = True) -> str | None:
        """종료 사유. 계속해야 하면 None."""
        if not self.active:
            return "A=empty"
        if self.budget_remaining <= 0:
            return "budget"
        if not candidates_available:
            return "no_action"
        return None

    def check_invariants(self) -> None:
        still = {b for b in self._frozen_seen if b in self.state}
        assert still <= self.frozen, \
            "FREEZE 단조성 위반: 한 번 얼린 branch 가 지금 FROZEN 이 아니다"
        n_edges = max(1, len(self.relation_edges))
        total = sum(self.verify_count.values())
        assert total <= self.k_verify * n_edges, \
            "VERIFY 총수 위반: %d > K*|E_R|=%d" % (total, self.k_verify * n_edges)
        assert self.spent <= self.budget, "예산 위반: %d > %d" % (self.spent, self.budget)

    def snapshot(self) -> dict[str, Any]:
        return {"n_nodes": len(self.nodes), "n_edges": len(self.edges),
                "active": self.active,
                "state": {k: v.value for k, v in self.state.items()},
                "rounds": self.rounds, "spent": self.spent,
                "budget_remaining": self.budget_remaining,
                "verify_count": {"%s|%s" % k: v for k, v in self.verify_count.items()}}
