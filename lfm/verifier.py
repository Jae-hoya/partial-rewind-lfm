"""SPEC §3.5 — 전역 검증. LLM judge 금지, NLI/규칙 기반만.

통과 조건 (집계 방식 명시 — SPEC 변경 #2):

    passed  <=>  mean_b grounding(b) >= cfg.verify.tau_g_global
            AND  min_b  coverage(b)  >= cfg.verify.tau_c_global

grounding 은 branch **평균**, coverage 는 branch **최솟값**이다.
이 비대칭은 의도된 것이며 report.md 에 그대로 기술한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .evidence import EvidenceState

__all__ = ["VerifyResult", "global_verify"]


@dataclass(frozen=True)
class VerifyResult:
    passed: bool
    agg: dict[str, Any] = field(default_factory=dict)


def global_verify(evidence: dict[str, EvidenceState], cfg) -> VerifyResult:
    tau_g = float(cfg["verify.tau_g_global"])
    tau_c = float(cfg["verify.tau_c_global"])

    if not evidence:
        return VerifyResult(False, {"reason": "no branches", "tau_g_global": tau_g,
                                    "tau_c_global": tau_c})

    groundings = [e.grounding for e in evidence.values()]
    coverages = [e.coverage for e in evidence.values()]
    mean_g = sum(groundings) / len(groundings)
    min_c = min(coverages)

    return VerifyResult(
        passed=(mean_g >= tau_g and min_c >= tau_c),
        agg={
            "mean_grounding": round(mean_g, 6),
            "min_coverage": round(min_c, 6),
            "tau_g_global": tau_g,
            "tau_c_global": tau_c,
            "pass_grounding": mean_g >= tau_g,
            "pass_coverage": min_c >= tau_c,
            "branches": {k: e.to_dict() for k, e in evidence.items()},
        },
    )
