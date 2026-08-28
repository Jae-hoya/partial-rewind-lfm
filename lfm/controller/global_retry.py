"""SPEC §3.6 — GlobalRetry baseline.

verify 실패 시 RETRY_ALL: **모든** branch 를 재실행한다. 최대 cfg.max_rounds 라운드.

이 컨트롤러가 Test A 의 측정 대상이다. "이미 충분했던 branch 까지 전부 다시 돌린다"는
성질이 낭비율의 정의를 만든다. 선택적 재실행을 넣으면 그건 GlobalRetry 가 아니다.
"""

from __future__ import annotations

from .base import Action, Controller, State

__all__ = ["GlobalRetry"]


class GlobalRetry(Controller):
    name = "global_retry"

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def step(self, state: State) -> Action:
        if state.verify_passed:
            return Action("STOP", params={"reason": "verify_passed", "round": state.round})
        if state.round + 1 >= state.max_rounds:
            return Action("STOP", params={"reason": "max_rounds", "round": state.round})
        return Action(
            "RETRY_ALL",
            target=sorted(state.evidence),   # 전 branch. 예외 없음
            params={"round": state.round, "next_round": state.round + 1},
        )
