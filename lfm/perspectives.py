"""SPEC §3.2 — 고정 role set 4개로 질의를 관점 분해한다. LLM 1회.

role 은 고정이다. LLM 이 정하는 것은 role 별 관점 질문과 sub-question 목록뿐이다.
관점 수를 모델이 바꾸면 branch 수가 조건마다 달라져 비교가 깨진다.
"""

from __future__ import annotations

from typing import Any

from .jsonio import parse_structured
from .types import ROLES, Perspective

__all__ = ["ROLES", "decompose", "SYSTEM", "build_prompt"]

SYSTEM = (
    "You decompose a research question into fixed analytical perspectives for a "
    "multi-perspective RAG system. Reply with JSON only."
)

PROMPT = """Research question:
{query}

Decompose it into exactly {n} perspectives, one for each role below, in this order:
{roles}

For each perspective produce:
- "role": the role name, copied verbatim
- "question": one question (10-25 words) framing that role's angle on the research question
- "sub_questions": {lo} to {hi} concrete, factual sub-questions answerable from an
  encyclopedia. Each must be self-contained and searchable on its own.

Return a JSON array of {n} objects with exactly those keys. No prose, no code fences.{extra}"""


def build_prompt(query: str, cfg) -> str:
    lo = int(cfg.get("perspectives.min_sub_questions", 3))
    hi = int(cfg.get("perspectives.max_sub_questions", 5))
    extra = ""
    return PROMPT.format(
        query=query,
        n=len(ROLES),
        roles="\n".join("- " + r for r in ROLES),
        lo=lo,
        hi=hi,
        extra=extra,
    )


def _coerce(rows: list[dict[str, Any]], query: str, lo: int, hi: int) -> list[Perspective]:
    """role 순서와 개수를 스펙에 맞춰 강제한다. 모델 출력에 맡기지 않는다."""
    by_role: dict[str, dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role", "")).strip()
        if role in ROLES and role not in by_role:
            by_role[role] = row

    out: list[Perspective] = []
    for role in ROLES:
        row = by_role.get(role, {})
        question = str(row.get("question", "") or "").strip() or (
            "From the " + role.replace("_", " ") + " angle: " + query
        )
        raw = row.get("sub_questions") or []
        subs = [str(s).strip() for s in raw if str(s).strip()][:hi]
        while len(subs) < lo:
            # 부족분은 질의 자체로 채운다. branch 수·sub-question 수가 조건마다
            # 달라지면 coverage 분모가 흔들린다.
            subs.append(query)
        out.append(Perspective(role=role, question=question, sub_questions=tuple(subs)))
    return out


def decompose(query: str, llm, cfg, *, round_: int = 0) -> list[Perspective]:
    lo = int(cfg.get("perspectives.min_sub_questions", 3))
    hi = int(cfg.get("perspectives.max_sub_questions", 5))
    nonce = "r" + str(round_) if round_ else ""
    res = llm.call(
        build_prompt(query, cfg),
        system=SYSTEM,
        tag="decompose",
        cache_nonce=nonce,
    )
    rows = parse_structured(llm, res.text, expect=list, tag="decompose", cache_nonce=nonce)
    return _coerce(rows or [], query, lo, hi)
