"""파이프라인 공용 자료구조. 여기 있는 것만 모듈 간에 오간다."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "ROLES",
    "Perspective",
    "Chunk",
    "SubAnswer",
    "BranchOutput",
    "CITATION_RE",
    "split_sentences",
    "parse_citations",
]

# SPEC §3.2 — 고정 role set 4개
ROLES = (
    "overview_factual",
    "technical_mechanism",
    "impact_application",
    "controversy_limitation",
)

CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:#\-]+)\]")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    """문장 분할. 결정론적이면 충분하다 — 정확한 언어학이 목적이 아니다."""
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def parse_citations(text: str, valid: set[str] | None = None) -> list[str]:
    """[chunk_id] 표기를 뽑는다. valid 를 주면 그 안에 있는 것만 남긴다.

    등장 순서를 유지하되 중복은 제거하지 않는다 — source_diversity 는 인용 '횟수'
    분포를 쓴다.
    """
    found = CITATION_RE.findall(text or "")
    if valid is None:
        return found
    return [c for c in found if c in valid]


@dataclass(frozen=True)
class Perspective:
    role: str
    question: str
    sub_questions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "question": self.question,
                "sub_questions": list(self.sub_questions)}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: int
    title: str
    section_path: str
    text: str
    source_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubAnswer:
    """sub-question 하나에 대한 답. coverage 는 이 단위로 센다."""

    index: int
    sub_question: str
    text: str
    citations: list[str] = field(default_factory=list)
    status: str | None = None          # R2 E 프롬프트: answered | insufficient | dropped | format_fail
    produced_round: int | None = None  # 이 답을 만든 라운드. sub-question 재실행에서 물려받으면 이전 라운드


@dataclass
class BranchOutput:
    perspective: Perspective
    sub_answers: list[SubAnswer]
    claims: list[str]
    chunks: list[Chunk]                       # 이 branch 가 검색한 청크 전부
    round: int = 0
    truncated: bool = False                   # LLM 응답이 토큰 상한에 걸림
    degraded: bool = False                    # 파싱/절단으로 정상 산출이 아님
    retrievals: int = 0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    # R2 회계·진단. 기본값이면 to_dict 에 싣지 않는다 (legacy 산출 바이트 보존)
    billing: dict[str, Any] = field(default_factory=dict)   # api_calls, billed_*, by_tag
    answer_stats: dict[str, int] = field(default_factory=dict)  # n_format_fail, n_dropped_uncited, ...
    context_sha256: str = ""
    subq_executed: int = 0             # 이번 실행에서 실제로 생성한 sub-question 수

    # --- 파생 ---------------------------------------------------------
    @property
    def answer(self) -> str:
        return "\n".join(sa.text for sa in self.sub_answers if sa.text)

    @property
    def citations(self) -> list[str]:
        out: list[str] = []
        for sa in self.sub_answers:
            out.extend(sa.citations)
        return out

    def chunk_map(self) -> dict[str, Chunk]:
        return {c.chunk_id: c for c in self.chunks}

    def cited_chunks(self) -> list[Chunk]:
        cmap = self.chunk_map()
        seen: set[str] = set()
        out: list[Chunk] = []
        for cid in self.citations:
            if cid not in seen and cid in cmap:
                seen.add(cid)
                out.append(cmap[cid])
        return out

    def to_dict(self) -> dict[str, Any]:
        def sa_dict(s: SubAnswer) -> dict[str, Any]:
            d = {"index": s.index, "sub_question": s.sub_question, "text": s.text, "citations": s.citations}
            if s.status is not None:
                d["status"] = s.status
            if s.produced_round is not None:
                d["produced_round"] = s.produced_round
            return d

        out = self._legacy_dict(sa_dict)
        for k in ("billing", "answer_stats", "context_sha256", "subq_executed"):
            v = getattr(self, k)
            if v:
                out[k] = v
        return out

    def _legacy_dict(self, sa_dict) -> dict[str, Any]:
        return {
            "perspective": self.perspective.to_dict(),
            "round": self.round,
            "sub_answers": [sa_dict(s) for s in self.sub_answers],
            "claims": list(self.claims),
            "n_chunks": len(self.chunks),
            "truncated": self.truncated,
            "degraded": self.degraded,
            "cost": {
                "retrievals": self.retrievals,
                "llm_calls": self.llm_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "latency_ms": self.latency_ms,
            },
        }
