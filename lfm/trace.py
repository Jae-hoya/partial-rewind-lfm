"""Trace 기록 (CLAUDE.md 동결 스키마).

라운드마다 1레코드, JSONL. 필드 추가만 허용, 변경·삭제 금지.

M1 에서 R / verify_count / frozen 은 빈 값이지만 필드는 반드시 존재해야 한다 (M2 호환).
reward 는 항상 null — 종료 후 별도 스크립트가 backfill 한다 (M1 범위 아님).

budget_left: M1 에는 예산 의미론이 없다. 동결 스키마가 필드를 요구하므로 라운드 예산의
잔여 비율 1 - (round + 1) / max_rounds 로 결정론적으로 채운다. M2 에서 실제 비용 예산으로
대체된다.
"""

from __future__ import annotations

import io
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator

__all__ = [
    "STATE_LABELS",
    "new_run_id",
    "budget_left",
    "make_state",
    "make_action",
    "make_cost",
    "make_record",
    "TraceWriter",
    "read_trace",
]

# 상태 라벨: M1 은 ACTIVE 만 사용한다.
STATE_LABELS = ("ACTIVE", "FROZEN", "PRUNED", "UNRESOLVED_CONFLICT")

EVIDENCE_KEYS = ("coverage", "grounding", "source_diversity", "support_strength")


def new_run_id() -> str:
    return str(uuid.uuid4())


def budget_left(round_: int, max_rounds: int) -> float:
    """라운드 예산 잔여 비율. 결정론적 — 같은 (round, max_rounds) 면 같은 값."""
    if max_rounds <= 0:
        raise ValueError("max_rounds 는 1 이상이어야 한다")
    return round(max(0.0, 1.0 - (round_ + 1) / max_rounds), 6)


def make_state(
    *,
    evidence: dict[str, dict[str, float]] | None = None,
    relations: list[dict[str, Any]] | None = None,
    budget: float = 1.0,
    frozen: list[str] | None = None,
    verify_count: dict[str, int] | None = None,
    vertices: Iterable[str] = (),
    edges: Iterable[Any] = (),
    status: dict[str, str] | None = None,
) -> dict[str, Any]:
    """동결 스키마의 state 블록. 빈 값이어도 키는 전부 존재한다."""
    V = list(vertices)
    S = dict(status) if status is not None else {v: "ACTIVE" for v in V}
    bad = sorted(set(S.values()) - set(STATE_LABELS))
    if bad:
        raise ValueError("허용되지 않은 상태 라벨: " + repr(bad))
    return {
        "E": {k: dict(v) for k, v in (evidence or {}).items()},
        "R": list(relations or []),
        "budget_left": float(budget),
        "frozen": list(frozen or []),
        "verify_count": dict(verify_count or {}),
        "graph": {"V": V, "E": list(edges), "S": S},
    }


def make_action(
    type_: str, *, target: list[str] | None = None, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {"type": type_, "target": list(target or []), "params": dict(params or {})}


def make_cost(
    *,
    tokens_in: int = 0,
    tokens_out: int = 0,
    llm_calls: int = 0,
    retrievals: int = 0,
    latency_ms: int = 0,
) -> dict[str, int]:
    return {
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "llm_calls": int(llm_calls),
        "retrievals": int(retrievals),
        "latency_ms": int(latency_ms),
    }


def make_record(
    *,
    run_id: str,
    query_id: str,
    round_: int,
    state: dict[str, Any],
    action: dict[str, Any],
    cost: dict[str, int],
    next_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """reward 는 항상 null 로 기록한다 (동결 스키마)."""
    return {
        "run_id": run_id,
        "query_id": query_id,
        "round": int(round_),
        "state": state,
        "action": action,
        "cost": cost,
        "reward": None,
        "next_state": dict(next_state or {}),
    }


class TraceWriter:
    """JSONL append 라이터. 항상 UTF-8, 라인 단위 flush."""

    def __init__(self, path: str | os.PathLike[str], *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = io.open(self.path, "a" if append else "w", encoding="utf-8", newline="\n")
        self.count = 0

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_trace(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
