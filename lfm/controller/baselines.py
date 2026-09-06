"""SPEC_M2 §7 — Test C 비교선 셋. Oracle(상한) · GlobalVerifier · Random(하한).

**전부 동일 estimator·relation·예산·시드.** CLAUDE.md: "비교 공정성의 핵심이므로
baseline 전용 우회 경로 금지". 그래서 넷 다 `step()` 하나만 구현하고 같은 루프
(`controller/lfm.py:run`)를 탄다. LFM 만 아는 지름길이 있으면 Test C 가 무의미해진다.

  Oracle          C-1 의 주입 라벨을 보고 정답 action 을 낸다 — 도달 가능한 상한
  GlobalVerifier  verify 실패 시 전 branch 재실행 (= Test A 의 GlobalRetry)
  Random          활성 branch·edge 에서 균등 무작위 — 하한

Random 이 하한인 것이 중요하다. **LFM ≈ Random 이면 R_ij 가 신호를 안 준다는 뜻**이고,
그건 마스터플랜 §9-2 로 간다. 그래서 Random 은 성의 없이 짜면 안 된다 — 같은 후보
공간에서 균등하게 뽑아야 공정한 하한이다.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..graph import LFMGraph
from ..localize import ActionKind, Candidate, EvidenceView, localize
from ..relation import Relation, Thresholds

__all__ = ["OracleController", "GlobalVerifierController", "RandomController"]


class _Base:
    name = "base"

    def __init__(self, th: Thresholds | None = None, k_verify: int = 2) -> None:
        self.th = th or Thresholds()
        self.k_verify = int(k_verify)

    def _cands(self, g, evidence, relations) -> list[Candidate]:
        return localize(g.active, evidence, relations,
                        verify_counts=g.verify_count,
                        budget_remaining=g.budget_remaining,
                        th=self.th, k_verify=self.k_verify)

    def freeze_sweep(self, g, evidence, relations) -> list[Candidate]:
        return [c for c in self._cands(g, evidence, relations)
                if c.kind == ActionKind.FREEZE]


class OracleController(_Base):
    """주입 라벨을 보고 정답을 낸다. **상한이지 참가자가 아니다.**

    `truth` 는 Test C-1 이 만든 (종류, 표적). 오라클이 그걸 그대로 낸다.
    후보 목록에 정답이 없으면 (예산·K 제약) None 을 낸다 — 오라클도 규칙은 지킨다.
    """

    name = "oracle"

    def __init__(self, truth: Mapping[str, dict], **kw) -> None:
        super().__init__(**kw)
        self.truth = dict(truth)

    def step(self, g: LFMGraph, evidence: Mapping[str, EvidenceView],
             relations: Sequence[Relation]) -> Candidate | None:
        want = self.truth.get("kind")
        tb, te = self.truth.get("branch"), self.truth.get("edge")
        te = tuple(te) if te else None
        for c in self._cands(g, evidence, relations):
            if c.kind != want:
                continue
            if tb is not None and c.branch != tb:
                continue
            if te is not None and (c.edge is None or tuple(sorted(c.edge)) != tuple(sorted(te))):
                continue
            return c
        return None


class GlobalVerifierController(_Base):
    """verify 실패 시 **전 branch 재실행**. Test A 의 GlobalRetry 와 같은 규칙.

    국소화를 하지 않는다 — 그것이 이 비교선의 요점이다. Test A 에서 50질의 전부가
    매 라운드 4/4 를 재실행한 그 동작을 M2 의 그래프 위에서 그대로 재현한다.
    """

    name = "global_verifier"

    def step(self, g: LFMGraph, evidence: Mapping[str, EvidenceView],
             relations: Sequence[Relation]) -> Candidate | None:
        active = g.active
        if not active:
            return None
        # 전역 verify: 전부 sufficient 면 아무것도 안 한다
        if all(evidence[b].sufficient(self.th.tau_g) for b in active if b in evidence):
            return None
        # 실패 -> 전 branch 의 모든 sub-question 재실행. 한 라운드에 하나씩 내되
        # 표적은 branch 전체다 (국소화 없음).
        for b in active:
            e = evidence.get(b)
            if e is None:
                continue
            subqs = tuple(sorted(set(e.missing_subqs) | set(e.low_grounding_subqs)))
            if not subqs:
                continue
            c = Candidate(kind=ActionKind.EXPAND, branch=b, subqs=subqs,
                          kind_detail="breadth", calls=len(subqs),
                          reason="global retry: 국소화 없이 branch 전체 재실행")
            if c.calls <= g.budget_remaining:
                return c
        return None


class RandomController(_Base):
    """같은 후보 공간에서 **균등 무작위**. 공정한 하한이어야 한다.

    시드를 받아 결정론으로 재현한다. Test C 에서 LFM 과 같은 시드를 쓴다.
    """

    name = "random"

    def __init__(self, seed: int = 42, **kw) -> None:
        super().__init__(**kw)
        self.rng = np.random.default_rng(int(seed))

    def step(self, g: LFMGraph, evidence: Mapping[str, EvidenceView],
             relations: Sequence[Relation]) -> Candidate | None:
        pool = [c for c in self._cands(g, evidence, relations)
                if c.kind != ActionKind.FREEZE]
        if not pool:
            return None
        return pool[int(self.rng.integers(0, len(pool)))]
