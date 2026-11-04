"""SPEC §3.4 — Evidence state E_i = [coverage, grounding, source_diversity, support_strength].

CLAUDE.md 용어 규칙: 이 지표들은 "deterministic" 이 아니라
**non-generative, reproducible estimators** 다. coverage 만 완전 결정론이고 나머지 셋은
NLI 모델에 의존한다 (모델 고정 시 재현 가능).

source_diversity 는 K-cap entropy: D = min(1, H(p)/log K), K=5 고정, 단일 소스 -> 0.
(unique/citations 비율은 폐기된 구버전이다.)
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from .estimator import Estimator, NLIMaxEstimator
from .nli import NLIBase
from .types import BranchOutput, split_sentences

__all__ = ["EvidenceState", "coverage", "cited_texts", "grounding", "source_diversity", "support_strength",
           "sentences_with_premises", "evaluate"]

DIVERSITY_K = 5  # CLAUDE.md 확정: K=5 고정 (K ∈ {3,5,8} 민감도는 부록)


@dataclass(frozen=True)
class EvidenceState:
    coverage: float
    grounding: float
    source_diversity: float
    support_strength: float

    def to_dict(self) -> dict[str, float]:
        return {k: round(float(v), 6) for k, v in asdict(self).items()}


def coverage(out: BranchOutput) -> float:
    """근거 인용이 붙은 sub-question 비율. 완전 결정론."""
    subs = out.sub_answers
    if not subs:
        return 0.0
    valid = set(out.chunk_map())
    n = sum(1 for sa in subs if any(c in valid for c in sa.citations))
    return n / len(subs)


def cited_texts(sa, cmap) -> list[str]:
    """sub-answer 가 인용한 청크 본문 — **고유 청크, 첫 인용 순서** (`BranchOutput.cited_chunks()` 와 같은 규칙).

    같은 청크를 여러 번 인용해도 전제에는 한 번만 넣는다(이탈 2, 2026-11-05). 인용 횟수만큼 이어붙이면
    R2 파일럿에서 전제가 81,175자(인용 20, 고유 4)가 돼 단일 조각 채점이 불가능했다. nli_max(g₁)는
    최댓값이라 중복과 무관하다.
    """
    return [cmap[c].text for c in dict.fromkeys(c for c in sa.citations if c in cmap)]


def sentences_with_premises(out: BranchOutput) -> list[tuple[str, list[str]]]:
    """답안 문장과 그 문장이 인용한 chunk 본문. 인용이 없으면 빈 목록이다.

    grounding 의 분모를 한 곳에서 정한다 — 여기가 갈리면 표가 어긋난다.
    """
    cmap = out.chunk_map()
    items: list[tuple[str, list[str]]] = []
    for sa in out.sub_answers:
        cited = cited_texts(sa, cmap)
        for sent in split_sentences(sa.text):
            items.append((sent, cited))
    return items


def grounding(out: BranchOutput, scorer, tau_ground: float = 0.5) -> float:
    """답안 문장 중 근거가 지지하는 비율.

    인용이 없는 문장은 근거가 없는 것이므로 grounded 로 세지 않는다.

    `scorer` 로 `Estimator`(SPEC_M1prime §1) 를 주면 그 연산자로 재고, 종전처럼
    NLI 모델을 주면 **현행 g₁ 과 완전히 같은 계산**을 한다. 기존 호출부를 그대로
    두기 위한 것이며, 기본 경로의 수치는 한 자리도 바뀌지 않는다.
    """
    est = scorer if isinstance(scorer, Estimator) else NLIMaxEstimator(scorer, tau_ground)
    items = sentences_with_premises(out)
    if not items:
        return 0.0
    flags = est.grounded_many(items) if hasattr(est, "grounded_many")         else [est.grounded(s, p) for s, p in items]
    return sum(1 for f in flags if f) / len(items)


def source_diversity(out: BranchOutput, k: int = DIVERSITY_K) -> float:
    """D = min(1, H(p)/log K). 인용 분포는 **출처 문서(doc_id)** 기준. 단일 소스 -> 0.0."""
    cmap = out.chunk_map()
    docs = [cmap[c].doc_id for c in out.citations if c in cmap]
    if not docs:
        return 0.0
    counts = Counter(docs)
    if len(counts) <= 1:
        return 0.0
    total = sum(counts.values())
    h = -sum((n / total) * math.log(n / total) for n in counts.values())
    return min(1.0, h / math.log(k))


def support_strength(out: BranchOutput, nli: NLIBase) -> float:
    """claim 별 최대 entailment 확률의 평균.

    전제 후보는 이 branch 가 인용한 청크다. 인용이 하나도 없으면 근거가 없으므로 0.0.
    """
    claims = [c for c in out.claims if c and c.strip()]
    if not claims:
        return 0.0
    premises = [c.text for c in out.cited_chunks()]
    if not premises:
        return 0.0
    return sum(nli.max_entailment(premises, claim) for claim in claims) / len(claims)


def evaluate(out: BranchOutput, nli: NLIBase, cfg, est: Estimator | None = None) -> EvidenceState:
    """`est` 를 주면 grounding 만 그 연산자로 잰다 (SPEC_M1prime §1).

    support_strength 는 그대로 NLI 로 잰다 — 스펙이 바꾸라 한 것은 grounding 하나다.
    """
    tau = float(cfg.get("nli.tau_ground", 0.5))
    return EvidenceState(
        coverage=coverage(out),
        grounding=grounding(out, est if est is not None else nli, tau),
        source_diversity=source_diversity(out),
        support_strength=support_strength(out, nli),
    )


def subq_targets(out: BranchOutput, scorer, tau_ground: float = 0.5) -> list[int]:
    """sub-question 단위 EXPAND 대상 = 미커버 ∪ 근거 비지지 문장이 있는 sub-question.

    coverage·grounding 과 같은 규칙으로 센다 — 유효 인용이 없으면 미커버이고, 문장 판정은
    grounding() 과 같은 연산자다(재실험설계_확정안_R2 §8-8).
    """
    est = scorer if isinstance(scorer, Estimator) else NLIMaxEstimator(scorer, tau_ground)
    cmap = out.chunk_map()
    targets, items, owner = [], [], []
    for sa in out.sub_answers:
        cited = cited_texts(sa, cmap)
        if not cited:
            targets.append(sa.index)
            continue
        for sent in split_sentences(sa.text):
            items.append((sent, cited))
            owner.append(sa.index)
    if items:
        flags = est.grounded_many(items) if hasattr(est, "grounded_many") else [est.grounded(s, p) for s, p in items]
        targets.extend(i for i, f in zip(owner, flags) if not f)
    return sorted(set(targets))


def sufficient(ev: EvidenceState | dict[str, Any], tau: float) -> bool:
    """SPEC §4.3 — sufficient(b) := grounding(b) > tau AND coverage(b) > tau.

    **엄격 부등호다.** τ 이상이 아니라 초과. 스펙 문구 그대로 옮긴다.
    """
    g = ev.grounding if isinstance(ev, EvidenceState) else float(ev["grounding"])
    c = ev.coverage if isinstance(ev, EvidenceState) else float(ev["coverage"])
    return g > tau and c > tau
