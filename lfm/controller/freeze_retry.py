"""R2 축소 컨트롤러 — FREEZE + EXPAND(재실행). 재실험설계_확정안_R2 §1, §3.

verify 가 실패하면, 이미 sufficient 한 branch 는 동결하고 나머지만 다시 돌린다.
  granularity = branch  비동결 branch 를 통째로 다시 돌린다 (BR)
  granularity = subq    비동결 branch 의 미커버 ∪ 근거 비지지 sub-question 만 다시 돌린다 (SQ)

동결 기준은 evidence.sufficient 그대로다 (g > τ ∧ c > τ, 엄격). 동결은 단조다 — 한 번 동결된
branch 는 다시 돌리지 않으므로 그 evidence 도 바뀌지 않는다. VERIFY / PRUNE 은 내지 않는다
(평가 시스템에서 제외, 확정안 §1). GlobalRetry 와 같은 State 만 본다 — baseline 전용 우회 경로 없음.
종료는 pipeline 의 range(max_rounds) 와 이 컨트롤러의 STOP 으로 보장된다.
"""

from __future__ import annotations

from ..evidence import sufficient
from .base import Action, Controller, State

__all__ = ["FreezeRetry"]


class FreezeRetry(Controller):
    name = "freeze_retry"

    def __init__(self, cfg, granularity: str = "branch") -> None:
        if granularity not in ("branch", "subq"):
            raise ValueError("granularity 는 branch | subq: " + str(granularity))
        self.cfg = cfg
        self.granularity = granularity
        self.tau = float(cfg.get("sweep.primary_tau", 0.8))
        self.needs_subq_view = granularity == "subq"

    def step(self, state: State) -> Action:
        base = {"round": state.round}
        if state.verify_passed:
            return Action("STOP", params={**base, "reason": "verify_passed", "frozen": list(state.frozen)})
        if state.round + 1 >= state.max_rounds:
            return Action("STOP", params={**base, "reason": "max_rounds", "frozen": list(state.frozen)})
        frozen = sorted(set(state.frozen) | {b for b, ev in state.evidence.items() if sufficient(ev, self.tau)})
        target = sorted(b for b in state.evidence if b not in frozen)
        if not target:
            return Action("STOP", params={**base, "reason": "all_frozen", "frozen": frozen})
        params = {**base, "next_round": state.round + 1, "frozen": frozen}
        if self.granularity == "subq":
            # 대상이 비면(이론상 sufficient 가 아닌 branch 에는 없다) branch 전체를 다시 돈다
            params["subqs"] = {b: (state.subq_view.get(b) or None) for b in target}
        return Action("RERUN", target=target, params=params)
