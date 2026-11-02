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

import hashlib
from typing import Any

from .jsonio import parse_structured
from .llm import LLM
from .types import BranchOutput, Chunk, Perspective, SubAnswer, parse_citations, split_sentences

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

# R2 기권 명시형 (재실험설계_확정안_R2 §2). legacy 상수는 한 글자도 바꾸지 않는다 — Test A 캐시 키 보존.
# H1·H2 사람 라벨: 기권은 옳았고, 결함은 근거가 없을 때 쓰는 비껴간 답이었다. 이 프롬프트는 그 답을
# insufficient 로 바꾸게 한다. 기권을 줄이는 장치가 아니다.
ANSWER_SYSTEM_E = (
    "You answer from one assigned perspective using only the supplied context chunks. "
    "For every sub-question you first decide whether the chunks directly answer it. "
    "Every answer sentence must carry a [chunk_id] citation. Reply with JSON only."
)

ANSWER_PROMPT_E = """Perspective role: {role}
Perspective question: {question}

Context chunks:
{context}

For each sub-question below, decide its status, then write the text.

Status rules:
- "answered": use ONLY when the chunks directly state the information the
  sub-question asks for. Answer that core and stop.
- "insufficient": use when the chunks do not directly answer the core of the
  sub-question — including when they contain only related background, neighbouring
  facts, general statements, or information about a different aspect, time, entity
  or measure than the one asked. Leave the text empty.
- Do NOT write a partial or sidestepping answer. Do not answer an easier, broader
  or neighbouring question in place of the one asked. Do not pad an answer with
  related facts. If you are unsure whether the chunks answer the core, choose
  "insufficient".
- An "insufficient" status is a correct and expected outcome. It is not a failure.

Text rules (for "answered"):
- 1 to 5 short sentences, one fact per sentence, stated plainly.
- Every sentence carries at least one [chunk_id] from the list above. A sentence
  without a valid citation will be deleted.
- Write DIRECT factual statements about the subject. Never write about the sources
  ("the context shows", "the chunks do not specify"). Do not describe the absence
  of information. Do not invent facts. Do not cite ids that are not listed.

Output:
- A JSON array. Each element:
  {{"index": <int>, "status": "answered" | "insufficient", "text": "<sentences with [citations] or empty>"}}
- One element per sub-question, same indices as below. Array only, no prose, no code fences.
- Retry instructions below never override the status rules.
{directive}
Sub-questions:
{subs}"""

PROMPT_MODES = ("legacy", "status")

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
    subq_filter: set[int] | None = None,
    carry: BranchOutput | None = None,
) -> BranchOutput:
    """search_anchor: 주제의 변별 용어. BM25 질의 앞에 붙인다.

    앵커가 없으면 sub-question 의 일반 명사에 BM25 가 끌려간다 — 실측에서
    웹툰 연재 질문에 "Burgess Hill railway station" 이, 축구협회 지배구조 질문에
    "Euroscepticism" 이 올라왔다. 앵커는 라운드·조건에 무관하게 고정이므로
    retrieval 은 여전히 전 조건 공통 고정 변인이다 (SPEC §2.6).

    R2 확장 (기본값이면 legacy 경로와 바이트 동일):
      branch.prompt_mode = status  기권 명시형 프롬프트(ANSWER_PROMPT_E)와 status 파서.
      subq_filter + carry          sub-question 단위 EXPAND. 대상 sub-question 만 paraphrase·검색·
                                   생성하고, 나머지 SubAnswer 와 그 청크는 carry 에서 그대로 물려받는다.
    """
    mode = cfg.get("branch.prompt_mode", "legacy") or "legacy"
    if mode not in PROMPT_MODES:
        raise ValueError("branch.prompt_mode 는 legacy | status: " + str(mode))
    sq_mode = subq_filter is not None
    if sq_mode and carry is None:
        raise ValueError("subq_filter 에는 이전 라운드 산출(carry)이 필요하다")
    n_all = len(p.sub_questions)
    targets = sorted(i for i in (subq_filter if sq_mode else range(n_all)) if 0 <= i < n_all)
    r2 = mode != "legacy" or sq_mode or cfg.get("run.arm", None) is not None

    nonce = "r" + str(round_) if round_ else ""
    if sq_mode:
        nonce = (nonce or "r0") + "|sq"     # 대상만 싣는 프롬프트 — BR/GR 과 다른 표본
    top_k = cfg.top_k_for_round(round_)
    directive = ""
    if round_ > 0:
        directive = "\nIMPORTANT: " + cfg.retry_directive(round_) + "\n"

    llm.begin_scope()
    subs = tuple(p.sub_questions[i] for i in targets)
    if round_ > 0:
        subs = paraphrase_sub_questions(subs, llm, round_=round_)
    sub_of = dict(zip(targets, subs))

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
    system, template = (ANSWER_SYSTEM, ANSWER_PROMPT) if mode == "legacy" else (ANSWER_SYSTEM_E, ANSWER_PROMPT_E)
    context = format_context(chunks, int(cfg.get("branch.max_context_chars", 12000)))
    prompt = template.format(
        role=p.role,
        question=p.question,
        context=context,
        directive=directive,
        subs="\n".join(str(i) + ". " + sub_of[i] for i in targets),
    )
    res = llm.call(prompt, system=system, tag="answer", cache_nonce=nonce)
    truncated = LLM.is_truncated(res)

    if truncated:
        # 잘린 답안을 그대로 쓰면 coverage 0 인 빈 branch 가 낭비율 분모에 들어간다.
        # 예산을 올려 한 번만 다시 부른다. max_tokens 가 캐시 키에 있으므로 키가 갈린다.
        bumped = int(cfg.get("llm.max_tokens")) * int(cfg.get("branch.truncation_retry_mult", 2))
        res = llm.call(
            prompt, system=system, tag="answer_retry", max_tokens=bumped,
            cache_nonce=nonce,
        )
        truncated = LLM.is_truncated(res)

    rows = parse_structured(llm, res.text, expect=list, tag="answer", cache_nonce=nonce)
    valid = {c.chunk_id for c in chunks}
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("index"))
        except (TypeError, ValueError):
            continue
        by_index.setdefault(idx, row)

    stats = {"n_format_fail": 0, "n_insufficient": 0, "n_dropped": 0,
             "n_dropped_uncited_sentences": 0, "n_fabricated_id": 0, "n_insufficient_with_text": 0}
    new_answers: dict[int, SubAnswer] = {}
    for i in targets:
        row = by_index.get(i, {})
        text = str(row.get("text", "") or "")
        status = None
        if mode == "status":
            status = str(row.get("status", "") or "").strip().lower()
            if status not in ("answered", "insufficient"):
                status, text = "format_fail", ""
                stats["n_format_fail"] += 1
            elif status == "insufficient":
                stats["n_insufficient"] += 1
                stats["n_insufficient_with_text"] += int(bool(text.strip()))
                text = ""
            else:
                sents = split_sentences(text)
                kept = [s for s in sents if parse_citations(s, valid)]
                stats["n_fabricated_id"] += sum(1 for c in parse_citations(text) if c not in valid)
                stats["n_dropped_uncited_sentences"] += len(sents) - len(kept)
                text = " ".join(kept)
                if not kept:
                    status = "dropped"
                    stats["n_dropped"] += 1
        new_answers[i] = SubAnswer(
            index=i,
            sub_question=sub_of[i],
            text=text,
            citations=parse_citations(text, valid),
            status=status,
            produced_round=round_ if r2 else None,
        )

    if sq_mode:
        sub_answers = [new_answers.get(i) or carry.sub_answers[i] for i in range(n_all)]
        have = {c.chunk_id for c in chunks}
        for c in carry.chunks:            # 물려받은 답이 인용한 청크가 chunk_map 에 남게 한다
            if c.chunk_id not in have:
                have.add(c.chunk_id)
                chunks.append(c)
    else:
        sub_answers = [new_answers[i] for i in range(n_all)]
    degraded = truncated or not any(sa.text for sa in sub_answers)

    # --- claim 분해 ----------------------------------------------------
    answer_text = "\n".join(sa.text for sa in sub_answers if sa.text)
    claims: list[str] = []
    if sq_mode and not any(new_answers[i].text for i in targets):
        claims = list(carry.claims)       # 새 텍스트가 없으면 답이 그대로다 — 다시 분해하지 않는다
    elif answer_text.strip():
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
    out = BranchOutput(
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
    if r2:
        out.billing = {k: scope[k] for k in ("api_calls", "billed_in", "billed_out",
                                              "billed_in_cached", "billed_reasoning", "by_tag")}
        out.answer_stats = stats if mode == "status" else {}
        out.context_sha256 = hashlib.sha256(context.encode("utf-8")).hexdigest()
        out.subq_executed = len(targets)
    return out


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
