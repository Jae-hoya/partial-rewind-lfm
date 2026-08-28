"""SPEC §3.3 — 관점 하나를 실행한다: 검색 -> 답안 생성(인용 필수) -> claim 분해.

라운드 교란 (SPEC §3.6, 변경 #1). sampling 파라미터(temperature/seed)에 기대지 않는다.
모델별 지원 여부가 다르기 때문이다. 교란은 세 층에서 일어난다:

  1. 검색 폭   top_k_r = retrieval.top_k * retry_k_multiplier[r]
  2. sub-question paraphrase  (LLM 1콜, tag="paraphrase")
  3. retry directive 문자열을 생성 프롬프트에 삽입

그리고 라운드 r>0 의 **모든** LLM 호출에 cache_nonce=f"r{r}" 를 준다. 이게 없으면
캐시가 라운드 1의 응답을 그대로 돌려줘 3라운드가 동일 복제가 되고, 낭비율은 측정이
아니라 항등식이 된다.
"""

from __future__ import annotations

from typing import Any

from .jsonio import parse_structured
from .llm import LLM
from .types import BranchOutput, Chunk, Perspective, SubAnswer, parse_citations

__all__ = ["run_branch", "ANSWER_SYSTEM", "CLAIM_SYSTEM", "paraphrase_sub_questions"]

ANSWER_SYSTEM = (
    "You answer from one assigned perspective using only the supplied context chunks. "
    "Every factual sentence must carry a [chunk_id] citation. Reply with JSON only."
)

ANSWER_PROMPT = """Perspective role: {role}
Perspective question: {question}

Context chunks:
{context}

Answer each sub-question below using ONLY the context chunks.

Rules:
- Cite with [chunk_id] inline, using the exact ids shown above. Every factual sentence
  needs at least one citation.
- Write DIRECT factual statements about the subject itself. Never write about the sources:
  no "the context shows", "the provided chunks indicate", "the context does not specify",
  no hedging about what is or is not documented. Such sentences describe the corpus rather
  than the world and cannot be verified against it.
- If the context does not support an answer, simply write fewer sentences — or leave the
  text empty. Do not describe the absence. Do not invent facts. Do not cite ids that are
  not listed.
- Keep each sentence short and self-contained: one fact per sentence, stated plainly.
- 2 to 5 sentences per sub-question.
- Return a JSON array. Each element: {{"index": <int>, "text": "<answer with [citations]>"}}
- Return the array only. No prose, no code fences.
{directive}
Sub-questions:
{subs}"""

CLAIM_SYSTEM = "You decompose text into atomic factual claims. Reply with JSON only."

CLAIM_PROMPT = """Decompose the text below into atomic factual claims.

Rules:
- Each claim is one self-contained sentence asserting a single fact.
- Strip citation markers like [c12] from the claim text.
- Do not add facts that are not in the text.
- At most {max_claims} claims.
- Return a JSON array of strings. No prose, no code fences.

--- TEXT ---
{answer}
--- END ---"""

PARAPHRASE_SYSTEM = "You rewrite search sub-questions. Reply with JSON only."

PARAPHRASE_PROMPT = """The previous attempt at these sub-questions retrieved insufficient
evidence. Rewrite each one to approach the same information need from a different angle:
different wording, different entities to look up, more specific facts.

Keep the same number of items and the same order. Do not change the topic.
Return a JSON array of strings only. No prose, no code fences.

{subs}"""


def format_context(chunks: list[Chunk], max_chars: int) -> str:
    parts = []
    used = 0
    for c in chunks:
        text = c.text.strip().replace("\n", " ")
        head = "[" + c.chunk_id + "] (" + c.title + " > " + c.section_path + ") "
        room = max_chars - used
        if room <= len(head) + 80:
            break
        body = text[: max(0, room - len(head))]
        parts.append(head + body)
        used += len(head) + len(body)
    return "\n".join(parts)


def paraphrase_sub_questions(
    subs: tuple[str, ...], llm: LLM, *, round_: int
) -> tuple[str, ...]:
    """SPEC §3.6 교란 2. 개수와 순서는 유지한다 — coverage 분모가 흔들리면 안 된다."""
    nonce = "r" + str(round_)
    listing = "\n".join(str(i) + ". " + s for i, s in enumerate(subs))
    res = llm.call(
        PARAPHRASE_PROMPT.format(subs=listing),
        system=PARAPHRASE_SYSTEM,
        tag="paraphrase",
        cache_nonce=nonce,
    )
    rows = parse_structured(llm, res.text, expect=list, tag="paraphrase", cache_nonce=nonce)
    if not rows:
        return subs
    new = [str(s).strip() for s in rows if str(s).strip()]
    if len(new) != len(subs):
        return subs  # 개수가 바뀌면 교란을 포기한다. 분모 보존이 우선이다.
    return tuple(new)


def run_branch(
    p: Perspective,
    retriever,
    llm: LLM,
    cfg,
    *,
    round_: int = 0,
    search_anchor: str = "",
) -> BranchOutput:
    """search_anchor: 주제의 변별 용어. BM25 질의 앞에 붙인다.

    앵커가 없으면 sub-question 의 일반 명사에 BM25 가 끌려간다 — 실측에서
    웹툰 연재 질문에 "Burgess Hill railway station" 이, 축구협회 지배구조 질문에
    "Euroscepticism" 이 올라왔다. 앵커는 라운드·조건에 무관하게 고정이므로
    retrieval 은 여전히 전 조건 공통 고정 변인이다 (SPEC §2.6).
    """
    nonce = "r" + str(round_) if round_ else ""
    top_k = cfg.top_k_for_round(round_)
    directive = ""
    if round_ > 0:
        directive = "\nIMPORTANT: " + cfg.retry_directive(round_) + "\n"

    llm.begin_scope()
    subs = p.sub_questions
    if round_ > 0:
        subs = paraphrase_sub_questions(subs, llm, round_=round_)

    # --- 검색 ---------------------------------------------------------
    chunks: list[Chunk] = []
    seen: set[str] = set()
    retrievals = 0
    anchor = (search_anchor or "").strip()
    for sq in subs:
        retrievals += 1
        query = (anchor + " " + sq).strip() if anchor else sq
        for c in retriever.search(query, top_k):
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                chunks.append(c)

    # --- 답안 생성 -----------------------------------------------------
    prompt = ANSWER_PROMPT.format(
        role=p.role,
        question=p.question,
        context=format_context(chunks, int(cfg.get("branch.max_context_chars", 12000))),
        directive=directive,
        subs="\n".join(str(i) + ". " + s for i, s in enumerate(subs)),
    )
    res = llm.call(prompt, system=ANSWER_SYSTEM, tag="answer", cache_nonce=nonce)
    truncated = LLM.is_truncated(res)

    if truncated:
        # 잘린 답안을 그대로 쓰면 coverage 0 인 빈 branch 가 낭비율 분모에 들어간다.
        # 예산을 올려 한 번만 다시 부른다. max_tokens 가 캐시 키에 있으므로 키가 갈린다.
        bumped = int(cfg.get("llm.max_tokens")) * int(cfg.get("branch.truncation_retry_mult", 2))
        res = llm.call(
            prompt, system=ANSWER_SYSTEM, tag="answer_retry", max_tokens=bumped,
            cache_nonce=nonce,
        )
        truncated = LLM.is_truncated(res)

    rows = parse_structured(llm, res.text, expect=list, tag="answer", cache_nonce=nonce)
    valid = {c.chunk_id for c in chunks}
    by_index: dict[int, str] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        by_index.setdefault(idx, str(row.get("text", "") or ""))

    sub_answers = [
        SubAnswer(
            index=i,
            sub_question=sq,
            text=by_index.get(i, ""),
            citations=parse_citations(by_index.get(i, ""), valid),
        )
        for i, sq in enumerate(subs)
    ]
    degraded = truncated or not any(sa.text for sa in sub_answers)

    # --- claim 분해 ----------------------------------------------------
    answer_text = "\n".join(sa.text for sa in sub_answers if sa.text)
    claims: list[str] = []
    if answer_text.strip():
        cres = llm.call(
            CLAIM_PROMPT.format(
                answer=answer_text[: int(cfg.get("branch.max_claim_chars", 8000))],
                max_claims=int(cfg.get("branch.max_claims", 20)),
            ),
            system=CLAIM_SYSTEM,
            tag="claims",
            cache_nonce=nonce,
        )
        parsed = parse_structured(llm, cres.text, expect=list, tag="claims", cache_nonce=nonce)
        claims = [str(c).strip() for c in (parsed or []) if str(c).strip()]

    scope = llm.end_scope()
    return BranchOutput(
        perspective=p,
        sub_answers=sub_answers,
        claims=claims,
        chunks=chunks,
        round=round_,
        truncated=truncated,
        degraded=degraded,
        retrievals=retrievals,
        llm_calls=scope["llm_calls"],
        tokens_in=scope["tokens_in"],
        tokens_out=scope["tokens_out"],
        latency_ms=scope["latency_ms"],
    )


def perturbation_params(cfg, round_: int) -> dict[str, Any]:
    """trace 의 action.params 에 그대로 실을 교란 파라미터 (Acceptance Criteria)."""
    return {
        "round": round_,
        "top_k": cfg.top_k_for_round(round_),
        "retry_k_multiplier": cfg.get("retrieval.retry_k_multiplier")[round_],
        "directive_id": round_ if round_ > 0 else None,
        "cache_nonce": "r" + str(round_) if round_ else "",
        "paraphrase": round_ > 0,
    }
