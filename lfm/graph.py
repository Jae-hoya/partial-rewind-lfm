"""SPEC §3.7 — 라운드별 (V, E, S) 스냅샷. M1 은 S 가 전부 ACTIVE 다.

M2 에서 FREEZE/PRUNE/UNRESOLVED_CONFLICT 가 들어올 자리를 인터페이스로만 남긴다.
스펙에 없는 기능을 미리 만들지 않는다 (CLAUDE.md STOP RULE 3).
"""

from __future__ import annotations

from typing import Any, Iterable

from .trace import STATE_LABELS

__all__ = ["Graph"]


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
