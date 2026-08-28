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

from .nli import NLIBase
from .types import BranchOutput, split_sentences

__all__ = ["EvidenceState", "coverage", "grounding", "source_diversity", "support_strength",
           "evaluate"]

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


def grounding(out: BranchOutput, nli: NLIBase, tau_ground: float) -> float:
    """답안 문장 중 인용 chunk 가 entail 하는 비율.

    인용이 없는 문장은 근거가 없는 것이므로 grounded 로 세지 않는다.
    """
    cmap = out.chunk_map()
    sentences: list[tuple[str, list[str]]] = []
    for sa in out.sub_answers:
        for sent in split_sentences(sa.text):
            cited = [c for c in sa.citations if c in cmap]
            sentences.append((sent, cited))
    if not sentences:
        return 0.0

    pairs: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    for sent, cited in sentences:
        start = len(pairs)
        for cid in cited:
            pairs.append((cmap[cid].text, sent))
        spans.append((start, len(pairs)))

    scores = nli.entailment(pairs) if pairs else []
    grounded = 0
    for start, end in spans:
        if end > start and max(scores[start:end]) >= tau_ground:
            grounded += 1
    return grounded / len(sentences)


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


def evaluate(out: BranchOutput, nli: NLIBase, cfg) -> EvidenceState:
    tau = float(cfg.get("nli.tau_ground", 0.5))
    return EvidenceState(
        coverage=coverage(out),
        grounding=grounding(out, nli, tau),
        source_diversity=source_diversity(out),
        support_strength=support_strength(out, nli),
    )


def sufficient(ev: EvidenceState | dict[str, Any], tau: float) -> bool:
    """SPEC §4.3 — sufficient(b) := grounding(b) > tau AND coverage(b) > tau.

    **엄격 부등호다.** τ 이상이 아니라 초과. 스펙 문구 그대로 옮긴다.
    """
    g = ev.grounding if isinstance(ev, EvidenceState) else float(ev["grounding"])
    c = ev.coverage if isinstance(ev, EvidenceState) else float(ev["coverage"])
    return g > tau and c > tau
