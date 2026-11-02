"""SPEC §3.6 — 컨트롤러 인터페이스.

CLAUDE.md: 컨트롤러는 전부 이 인터페이스를 구현한다.
**비교 공정성의 핵심이므로 baseline 전용 우회 경로를 만들지 않는다.**
M1 에는 GlobalRetry 하나뿐이고, M2 의 LFM/oracle/random 이 같은 자리에 들어온다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Action", "State", "Controller"]


@dataclass(frozen=True)
class Action:
    type: str
    target: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "target": list(self.target), "params": dict(self.params)}


@dataclass
class State:
    """컨트롤러가 보는 것 전부. 여기 없는 것에 의존하면 공정성이 깨진다."""

    query_id: str
    round: int
    max_rounds: int
    evidence: dict[str, Any]          # branch_id -> EvidenceState
    verify_passed: bool
    verify_agg: dict[str, Any] = field(default_factory=dict)
    graph: dict[str, Any] = field(default_factory=dict)
    relations: list[Any] = field(default_factory=list)   # M1 은 비어 있다
    verify_count: dict[str, int] = field(default_factory=dict)
    frozen: list[str] = field(default_factory=list)
    # R2: branch -> EXPAND 대상 sub-question 번호. 컨트롤러가 needs_subq_view 일 때만 채운다
    subq_view: dict[str, list[int]] = field(default_factory=dict)


class Controller(ABC):
    name = "base"

    @abstractmethod
    def step(self, state: State) -> Action:
        """다음 행동을 고른다. 종료는 Action("STOP")."""
