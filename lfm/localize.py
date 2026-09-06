"""SPEC_M2 §3 — Localization. (E, R) → 무엇을 고칠지.

이것이 LFM 의 "Localize" 다. 전부 재실행하는 대신, evidence 상태와 관점 간 관계를
보고 **고칠 자리만** 고른다.

우선순위는 스펙 고정이다: **VERIFY > EXPAND > PRUNE > FREEZE.**
FREEZE 는 비용 0 이라 매 라운드 마지막에 일괄로 돈다.

**최소 변경 원칙: 재실행 단위는 sub-question 이다.** branch 전체를 다시 돌리지
않는다(coverage=0 의 REGEN 만 예외). Test A 의 낭비율은 branch 단위였는데 여기 ΔG 는
한 단계 더 작다 — 그 차이가 LFM 의 주장이다.

전부 결정론이다. 같은 (E, R, 예산)이면 같은 후보 목록이 나온다. 무작위가 끼면
Test C 에서 Random 컨트롤러와 구분이 안 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .relation import Relation, Thresholds

__all__ = ["ActionKind", "Candidate", "EvidenceView", "localize"]


class ActionKind(str):
    FREEZE = "FREEZE"
    EXPAND = "EXPAND"
    VERIFY = "VERIFY"
    PRUNE = "PRUNE"


# 스펙 §3 고정 우선순위. 숫자가 작을수록 먼저.
PRIORITY = {ActionKind.VERIFY: 0, ActionKind.EXPAND: 1,
            ActionKind.PRUNE: 2, ActionKind.FREEZE: 3}


@dataclass(frozen=True)
class EvidenceView:
    """localize 가 필요로 하는 E_i 의 부분. 추정자 구현과 분리한다."""

    coverage: float
    grounding: float
    source_diversity: float
    support_strength: float
    missing_subqs: tuple[str, ...] = ()       # coverage 가 빠진 sub-question
    low_grounding_subqs: tuple[str, ...] = ()  # grounding<tau_g 문장이 속한 sub-question

    def sufficient(self, tau: float) -> bool:
        """SPEC §1 — coverage=1.0 AND grounding>tau. **엄격 부등호.**"""
        return self.coverage == 1.0 and self.grounding > tau


@dataclass
class Candidate:
    kind: str
    branch: str | None = None
    edge: tuple[str, str] | None = None
    subqs: tuple[str, ...] = ()
    suspect: tuple[str, ...] = ()      # VERIFY 에서 재검색 우선 대상
    kind_detail: str = ""              # EXPAND 의 "breadth" | "regen"
    calls: int = 0                     # 이 action 이 쓸 생성 호출 수
    reason: str = ""

    def sort_key(self) -> tuple:
        return (PRIORITY[self.kind], self.branch or "", str(self.edge or ""))

    def to_dict(self) -> dict:
        return {"type": self.kind, "target_branch": self.branch,
                "target_edge": list(self.edge) if self.edge else None,
                "target_subq": list(self.subqs), "kind": self.kind_detail,
                "suspect": list(self.suspect), "calls": self.calls,
                "reason": self.reason}


def localize(active: Sequence[str],
             evidence: Mapping[str, EvidenceView],
             relations: Iterable[Relation],
             *,
             verify_counts: Mapping[tuple[str, str], int] | None = None,
             budget_remaining: int = 10 ** 9,
             th: Thresholds | None = None,
             k_verify: int = 2) -> list[Candidate]:
    """후보 목록을 우선순위 순으로 돌려준다. 결정론."""
    th = th or Thresholds()
    verify_counts = dict(verify_counts or {})
    A = set(active)
    rels = [r for r in relations if _bare(r.i) in A and _bare(r.j) in A]
    by_branch: dict[str, list[Relation]] = {}
    for r in rels:
        by_branch.setdefault(_bare(r.i), []).append(r)
        by_branch.setdefault(_bare(r.j), []).append(r)

    out: list[Candidate] = []

    # --- VERIFY: contradiction >= tau_c, edge 당 K 회까지 --------------------
    for r in rels:
        if r.contradiction < th.tau_c:
            continue
        key = tuple(sorted((_bare(r.i), _bare(r.j))))
        if verify_counts.get(key, 0) >= k_verify:
            continue
        ei, ej = evidence.get(key[0]), evidence.get(key[1])
        suspect = _suspects(key, ei, ej)
        out.append(Candidate(
            kind=ActionKind.VERIFY, edge=key, suspect=suspect, calls=2,
            reason="contradiction %.3f >= tau_c %.2f (argmax pair %s)"
                   % (r.contradiction, th.tau_c, r.argmax_pair)))

    # --- EXPAND: non-sufficient 이고 contradiction 없음 ----------------------
    for b in sorted(A):
        e = evidence.get(b)
        if e is None or e.sufficient(th.tau_g):
            continue
        if any(r.contradiction >= th.tau_c for r in by_branch.get(b, [])):
            continue                                  # VERIFY 가 먼저다
        if e.coverage == 0.0:
            # 미인용 답안 — EXPAND 가 아니라 REGEN 분기 (스펙 §3)
            out.append(Candidate(
                kind=ActionKind.EXPAND, branch=b, subqs=tuple(e.missing_subqs),
                kind_detail="regen", calls=max(1, len(e.missing_subqs)),
                reason="coverage=0 미인용 답안 -> 인용 형식 강제 재생성"))
            continue
        targets = tuple(sorted(set(e.missing_subqs) | set(e.low_grounding_subqs)))
        if not targets:
            continue
        out.append(Candidate(
            kind=ActionKind.EXPAND, branch=b, subqs=targets,
            kind_detail="breadth", calls=len(targets),
            reason="non-sufficient (cov %.2f, grd %.2f), sub-question %d개"
                   % (e.coverage, e.grounding, len(targets))))

    # --- PRUNE: redundancy=1 이고 둘 다 sufficient ---------------------------
    for r in rels:
        if not r.redundancy:
            continue
        a, b = _bare(r.i), _bare(r.j)
        ea, eb = evidence.get(a), evidence.get(b)
        if ea is None or eb is None:
            continue
        if not (ea.sufficient(th.tau_g) and eb.sufficient(th.tau_g)):
            continue                    # 둘 다 non-sufficient 면 PRUNE 안 함 (정보 손실)
        drop = a if ea.source_diversity <= eb.source_diversity else b
        out.append(Candidate(
            kind=ActionKind.PRUNE, branch=drop, calls=0,
            reason="redundancy=1, source_diversity 낮은 쪽 (%.3f vs %.3f)"
                   % (ea.source_diversity, eb.source_diversity)))

    # --- FREEZE: sufficient 이고 어떤 j 와도 contradiction < tau_c ------------
    for b in sorted(A):
        e = evidence.get(b)
        if e is None or not e.sufficient(th.tau_g):
            continue
        if any(r.contradiction >= th.tau_c for r in by_branch.get(b, [])):
            continue
        out.append(Candidate(
            kind=ActionKind.FREEZE, branch=b, calls=0,
            reason="sufficient@%.2f 이고 모든 edge 에서 contradiction < %.2f"
                   % (th.tau_g, th.tau_c)))

    out.sort(key=Candidate.sort_key)
    return [c for c in out if c.calls <= budget_remaining]


def _bare(x: str) -> str:
    """relation 이 'topic:role' 로 들고 있을 수 있다. branch 이름만 뽑는다."""
    return x.split(":", 1)[1] if ":" in x else x


def _suspects(key: tuple[str, str], ei, ej) -> tuple[str, ...]:
    """support_strength 가 낮은 쪽이 suspect. 같으면 둘 다 (스펙 §3)."""
    if ei is None or ej is None:
        return key
    if ei.support_strength < ej.support_strength:
        return (key[0],)
    if ej.support_strength < ei.support_strength:
        return (key[1],)
    return key
