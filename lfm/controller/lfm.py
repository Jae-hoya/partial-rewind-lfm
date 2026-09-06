"""SPEC_M2 §6 — LFM 컨트롤러 v1. 규칙 기반 (학습형은 후속).

    loop until 종료:
      E ← estimator(active)        # 부분 재평가
      R ← relation(active pairs)   # 캐시 + 부분 재계산
      cands ← localize(E, R, budget)
      a ← cands[0]                 # 라운드당 action 1개
      ΔG ← apply(a)
      G ← G ⊕ ΔG ; log(trace)
    FREEZE 일괄 (비용 0)

**라운드당 action 을 1개로 묶은 것은 의도다.** Test C 가 "첫 action 이 오라클 표적을
맞히는가"로 국소화를 재기 때문이다. 여러 개를 한꺼번에 내면 그 채점이 흐려진다.

이 컨트롤러는 `controller/base.py` 의 인터페이스를 그대로 구현한다. CLAUDE.md 가
"비교 공정성의 핵심이므로 baseline 전용 우회 경로 금지"라고 못박아 둔 부분이다 —
LFM 만 아는 지름길을 만들면 Test C 의 비교가 무의미해진다.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from ..graph import LFMGraph, State
from ..localize import ActionKind, Candidate, EvidenceView, localize
from ..relation import Relation, Thresholds

__all__ = ["LFMController"]


class LFMController:
    name = "lfm"

    def __init__(self, th: Thresholds | None = None, k_verify: int = 2) -> None:
        self.th = th or Thresholds()
        self.k_verify = int(k_verify)

    def candidates(self, g: LFMGraph,
                   evidence: Mapping[str, EvidenceView],
                   relations: Sequence[Relation]) -> list[Candidate]:
        return localize(g.active, evidence, relations,
                        verify_counts=g.verify_count,
                        budget_remaining=g.budget_remaining,
                        th=self.th, k_verify=self.k_verify)

    def step(self, g: LFMGraph,
             evidence: Mapping[str, EvidenceView],
             relations: Sequence[Relation]) -> Candidate | None:
        """이번 라운드에 낼 action 하나. 없으면 None."""
        cands = self.candidates(g, evidence, relations)
        # FREEZE 는 비용 0 이라 매 라운드 마지막에 일괄로 돈다 (스펙 §3).
        actionable = [c for c in cands if c.kind != ActionKind.FREEZE]
        return actionable[0] if actionable else None

    def freeze_sweep(self, g: LFMGraph,
                     evidence: Mapping[str, EvidenceView],
                     relations: Sequence[Relation]) -> list[Candidate]:
        """비용 0 인 FREEZE 를 일괄 적용할 목록."""
        return [c for c in self.candidates(g, evidence, relations)
                if c.kind == ActionKind.FREEZE]


def run(controller, g: LFMGraph,
        estimator: Callable[[Sequence[str]], Mapping[str, EvidenceView]],
        relate: Callable[[Sequence[str]], Sequence[Relation]],
        execute: Callable[[Candidate], str] | None = None,
        max_rounds: int = 1000) -> dict:
    """§6 의 루프. 컨트롤러는 주입식이라 비교선도 같은 루프를 쓴다.

    같은 루프·같은 estimator·같은 relation·같은 예산을 쓰는 것이 Test C 의 전제다.
    """
    from ..actions import apply_candidate

    trace = []
    reason = None
    for _ in range(max_rounds):
        active = g.active
        E = estimator(active)
        R = relate(active)
        a = controller.step(g, E, R)
        reason = g.terminated(candidates_available=a is not None)
        if reason:
            break
        res = apply_candidate(g, a, execute=execute)
        if not res.executed:
            reason = "budget"
            break
        trace.append({"action": a.to_dict(), "delta_g": res.delta.to_dict(),
                      "state_after": {k: v.value for k, v in g.state.items()},
                      "budget_remaining": g.budget_remaining, "reward": None})

    # FREEZE 일괄 (비용 0)
    E = estimator(g.active)
    R = relate(g.active)
    for c in controller.freeze_sweep(g, E, R):
        if g.state.get(c.branch) == State.ACTIVE:
            res = apply_candidate(g, c, execute=None)
            trace.append({"action": c.to_dict(), "delta_g": res.delta.to_dict(),
                          "state_after": {k: v.value for k, v in g.state.items()},
                          "budget_remaining": g.budget_remaining, "reward": None})

    g.check_invariants()
    return {"controller": controller.name, "reason": reason or "loop_max",
            "rounds": g.rounds, "spent": g.spent, "trace": trace,
            "final": g.snapshot()}
