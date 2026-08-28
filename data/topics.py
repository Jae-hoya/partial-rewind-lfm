"""SPEC §2.3 — 교차 도메인 토픽 후보 생성. 승인 게이트 앞에서 강제 종료한다.

토픽 조건: infobox 카테고리가 서로 다른 3개 이상 하위주제를 갖는 주제.
인덱스를 만들지 않는다 — Parquet 메타데이터 스캔(duckdb)과 용어 통계만 쓴다.

## 왜 후보를 target article 에서 뽑는가

held-out 방식은 각 토픽의 **target article 을 코퍼스에서 빼고** 나머지 문서로 답하게
한다. 그러니 후보는 반드시 코퍼스 안에 실재하는 문서를 target 으로 지목해야 한다.
토픽 문구만 만들면 `heldout.py` 가 매칭할 대상이 없어 P3 에서 막힌다.

첫 시도는 '용어'에서 후보를 뽑고 infobox 유형 수로 정렬했다. 실패했다 — 유형 수는
교차 도메인성이 아니라 **흔함**의 대리 지표라서 eight / open / point / make 같은
기능어가 상위를 채웠고, target article 도 주제와 무관했다("date" -> "Hideo Date").
그래서 순서를 뒤집는다: 먼저 내용이 두꺼운 문서를 target 으로 고르고, 그 문서의
**변별력 있는 용어**(idf 상위)로 이웃을 정의한다.

절차:
  1. target 후보: abstract·본문·참고문헌이 일정 규모 이상이고 infobox 가 있는 문서.
     disambiguation / list of / index of 는 제외한다.
  2. 변별력 용어: 그 문서의 제목+abstract 용어 중 df 가 [df_lo, df_hi] 인 것을
     idf 순으로 상위 n_terms 개.
  3. 이웃: 그 용어들 중 min_term_hits 개 이상을 담은 문서.
  4. 이웃 수 >= min_related_docs 이고 이웃의 infobox 유형이 min_infobox_types 종 이상.
  5. 주 infobox 유형당 상한을 둬 다양성을 확보한다(축구선수 50개 방지).
  6. LLM 이 target 의 제목·abstract·도메인을 보고 토픽 질의 문구를 만든다.

출력 후 **반드시 종료한다.** 사람이 검토해 `topics_pilot50.yaml` 로 저장하고
최상단에 `approved: true` 를 넣기 전까지 이후 단계는 실행되지 않는다 (SPEC §2.4).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lfm.config import load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402

TOKEN_RE = re.compile(r"[a-z][a-z0-9\-']+")

# 내용어가 아닌 것들. 교차 도메인성 판정을 오염시킨다.
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those which who whom whose what when where
    while with within without into onto from for of off on in at by to as is are was were be been
    being have has had do does did not no nor so such only own same too very can will just should
    now also its it he she they them their there here about above after again against all am any
    because before below between both during each few more most other over under until up down out
    some more many first second third one two three four five other another new old known used
    using use also may might must would could shall like well back even still way since however
    including included include based upon among across around through during between per via
    american british english national international state states united world war century year
    years time part high low large small great major minor common general public private local
    named name names called title series list group member members number total area region
    """.split()
)


def tokens(text: str, min_len: int) -> set[str]:
    return {t for t in TOKEN_RE.findall((text or "").lower()) if len(t) >= min_len and t not in STOPWORDS}


def load_docs(corpus: Path) -> list[dict[str, Any]]:
    """제목·abstract·infobox 유형만 읽는다. 본문(sections)은 여기서 필요 없다."""
    import duckdb

    con = duckdb.connect()
    rows = con.execute(
        """
        SELECT identifier, name, url, abstract,
               CASE WHEN infoboxes IS NULL OR length(infoboxes) <= 2 THEN NULL ELSE infoboxes END
        FROM read_parquet('""" + str(corpus).replace("\\", "/") + """')
        """
    ).fetchall()
    con.close()

    docs = []
    for ident, name, url, abstract, infoboxes in rows:
        types: list[str] = []
        if infoboxes:
            try:
                for ib in json.loads(infoboxes):
                    n = (ib or {}).get("name")
                    if n:
                        types.append(n)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        docs.append(
            {
                "identifier": ident,
                "name": name,
                "url": url,
                "abstract": abstract or "",
                "infobox_types": types,
            }
        )
    return docs


def build_postings(docs: list[dict[str, Any]], min_len: int) -> dict[str, set[int]]:
    postings: dict[str, set[int]] = defaultdict(set)
    for i, d in enumerate(docs):
        for t in tokens(d["name"] + " " + d["abstract"], min_len):
            postings[t].add(i)
    return postings


def primary_type(doc) -> str:
    return doc["infobox_types"][0] if doc["infobox_types"] else "(none)"


def distinctive_terms(doc, postings, n_docs, cfg) -> list[str]:
    """그 문서를 다른 문서와 갈라놓는 용어. df 가 너무 낮으면 이웃이 없고,
    너무 높으면 기능어다."""
    lo = int(cfg.get("topics.df_lo", 20))
    hi = int(cfg.get("topics.df_hi", 3000))
    n_terms = int(cfg.get("topics.n_terms", 5))
    min_len = int(cfg.get("topics.min_term_len", 4))

    scored = []
    for t in tokens(doc["name"] + " " + doc["abstract"], min_len):
        df = len(postings.get(t, ()))
        if lo <= df <= hi:
            scored.append((math.log(n_docs / df), t))
    scored.sort(reverse=True)
    return [t for _, t in scored[:n_terms]]


def find_candidates(docs, postings, meta, cfg) -> list[dict[str, Any]]:
    min_rel = int(cfg.get("topics.min_related_docs", 30))
    min_types = int(cfg.get("topics.min_infobox_types", 3))
    min_hits = int(cfg.get("topics.min_term_hits", 2))
    min_abs = int(cfg.get("topics.min_target_abstract_chars", 600))
    min_sec = int(cfg.get("topics.min_target_section_chars", 6000))
    min_ref = int(cfg.get("topics.min_target_references", 20))
    per_type = int(cfg.get("topics.max_per_infobox_type", 3))
    want = int(cfg.get("topics.n_candidates", 60))
    n_docs = len(docs)

    scored: list[dict[str, Any]] = []
    for i, d in enumerate(docs):
        sec_len, n_ref = meta.get(d["identifier"], (0, 0))
        if len(d["abstract"]) < min_abs or (sec_len or 0) < min_sec or (n_ref or 0) < min_ref:
            continue
        if not d["infobox_types"]:
            continue
        low = d["name"].lower()
        if "disambiguation" in low or low.startswith(("list of", "index of", "outline of")):
            continue

        terms = distinctive_terms(d, postings, n_docs, cfg)
        if len(terms) < min_hits + 1:
            continue
        hits: Counter[int] = Counter()
        for t in terms:
            for j in postings[t]:
                hits[j] += 1
        related = {j for j, c in hits.items() if c >= min_hits and j != i}
        if len(related) < min_rel:
            continue
        types: set[str] = set()
        for j in related:
            types.update(docs[j]["infobox_types"])
        if len(types) < min_types:
            continue
        scored.append(
            {
                "term": " ".join(terms[:3]),
                "terms": terms,
                "target_idx": i,
                "related_docs": len(related),
                "n_infobox_types": len(types),
                "infobox_types": sorted(types)[:12],
                "primary_type": primary_type(d),
            }
        )

    # 이웃이 두꺼운 순. 유형 수가 아니라 **관련 문서 수**로 정렬한다 —
    # 유형 수로 정렬하면 다시 흔한 문서가 올라온다.
    scored.sort(key=lambda c: (-c["related_docs"], -c["n_infobox_types"], c["target_idx"]))

    picked: list[dict[str, Any]] = []
    per_type_count: Counter[str] = Counter()
    seen_targets: set[int] = set()
    for c in scored:
        if c["target_idx"] in seen_targets:
            continue
        if per_type_count[c["primary_type"]] >= per_type:
            continue
        per_type_count[c["primary_type"]] += 1
        seen_targets.add(c["target_idx"])
        picked.append(c)
        if len(picked) >= want:
            break
    return picked


def load_meta(corpus: Path) -> dict[int, tuple[int, int]]:
    """identifier -> (본문 길이, 참고문헌 수). target 후보의 '두께' 판정용."""
    import duckdb

    src = str(corpus).replace("\\", "/")
    sql = (
        "SELECT identifier, length(sections), len(\"references\") "
        "FROM read_parquet('" + src + "')"
    )
    con = duckdb.connect()
    rows = con.execute(sql).fetchall()
    con.close()
    return {int(r[0]): (int(r[1] or 0), int(r[2] or 0)) for r in rows}


PHRASE_SYSTEM = (
    "You turn corpus statistics into research questions for a multi-perspective "
    "retrieval-augmented generation benchmark. Answer with JSON only."
)

PHRASE_PROMPT = """For each item below, write ONE English research question suitable for a
multi-perspective report. The question must be answerable from an encyclopedia corpus and must
invite several angles: factual overview, technical mechanism, impact/application, and
controversy/limitation.

Rules:
- 12 to 25 words. No yes/no questions.
- The question must be about the given subject, but must NOT repeat its title verbatim —
  describe it (the corpus article itself is held out at evaluation time).
- Ground the question in the summary and key terms provided.
- Return a JSON array. Each element: {{"id": <int>, "topic": "<question>"}}
- Return the array only. No prose, no code fences.

Items:
{items}
"""


def phrase_topics(llm: LLM, docs, picked, batch_size: int) -> dict[int, str]:
    out: dict[int, str] = {}
    for start in range(0, len(picked), batch_size):
        batch = picked[start : start + batch_size]
        items = []
        for k, c in enumerate(batch):
            idx = start + k
            d = docs[c["target_idx"]]
            items.append(
                json.dumps(
                    {
                        "id": idx,
                        "subject": d["name"],
                        "summary": d["abstract"][:700],
                        "key_terms": c["terms"],
                        "neighbouring_domains": c["infobox_types"][:6],
                        "related_docs": c["related_docs"],
                    },
                    ensure_ascii=False,
                )
            )
        prompt = PHRASE_PROMPT.format(items="\n".join(items))
        res = llm.call(prompt, system=PHRASE_SYSTEM, tag="topic_phrase")
        parsed = _parse_json_array(llm, res.text, prompt)
        for row in parsed:
            try:
                out[int(row["id"])] = str(row["topic"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
        print("  배치 %d-%d: %d/%d 문구 생성" % (start, start + len(batch) - 1, len(parsed), len(batch)))
    return out


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"```\s*$", "", t)
    return t.strip()


def _parse_json_array(llm: LLM, text: str, original_prompt: str) -> list[dict[str, Any]]:
    """관대하게 파싱하고, 실패하면 **프롬프트를 바꿔** 복구를 요청한다.

    같은 프롬프트로 재시도하면 캐시가 같은 불량 응답을 영원히 돌려준다.
    """
    for candidate in (_strip_fences(text), text):
        try:
            v = json.loads(candidate)
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
        m = re.search(r"\[.*\]", candidate, re.S)
        if m:
            try:
                v = json.loads(m.group(0))
                if isinstance(v, list):
                    return v
            except json.JSONDecodeError:
                pass

    repair = (
        "The following text should have been a JSON array but is malformed. "
        "Return the corrected JSON array only.\n\n--- BEGIN ---\n"
        + text[:4000]
        + "\n--- END ---"
    )
    res = llm.call(repair, system=PHRASE_SYSTEM, tag="topic_phrase_repair")
    try:
        v = json.loads(_strip_fences(res.text))
        return v if isinstance(v, list) else []
    except json.JSONDecodeError:
        return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="교차 도메인 토픽 후보 생성 (SPEC §2.3)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(str(repo_root / ".env"))
    cfg = load_config(args.config or (repo_root / "config.yaml"))

    data_dir = Path(cfg["paths.data_dir"])
    corpus = data_dir / "corpus.parquet"
    out = Path(args.out or (data_dir / "topics_pilot50_candidates.yaml"))
    min_len = int(cfg.get("topics.min_term_len", 4))

    t0 = time.time()
    print("코퍼스 로드 중 ...", flush=True)
    docs = load_docs(corpus)
    print("문서 %d건" % len(docs))

    postings = build_postings(docs, min_len)
    meta = load_meta(corpus)
    print("용어 %d개" % len(postings))

    picked = find_candidates(docs, postings, meta, cfg)
    print("교차 도메인 조건 통과 후보 %d개" % len(picked))
    if not picked:
        print("후보가 없다. topics.min_* 임계를 낮추거나 max_articles 를 올려야 한다.")
        return 1

    llm = LLM(cfg)
    print("토픽 문구 생성 중 ...", flush=True)
    phrases = phrase_topics(llm, docs, picked, int(cfg.get("topics.batch_size", 10)))

    candidates = []
    for i, c in enumerate(picked):
        d = docs[c["target_idx"]]
        candidates.append(
            {
                "id": "t%03d" % i,
                "term": c["term"],
                "terms": c["terms"],
                "primary_infobox_type": c["primary_type"],
                "topic": phrases.get(i, ""),
                "target_article": d["name"],
                "target_identifier": int(d["identifier"]),
                "target_url": d["url"],
                "related_docs": c["related_docs"],
                "n_infobox_types": c["n_infobox_types"],
                "infobox_types": c["infobox_types"],
            }
        )

    missing = [c["id"] for c in candidates if not c["topic"]]
    doc = {
        "approved": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "spec": "SPEC_M1_TestA_v1.1.md §2.3",
        "note": (
            "후보 파일이다. 사람이 검토해 topics_pilot50.yaml 로 저장하고 approved: true 를 "
            "넣기 전까지 이후 단계는 실행되지 않는다."
        ),
        "criteria": {
            "min_related_docs": cfg.get("topics.min_related_docs", 30),
            "max_related_docs": cfg.get("topics.max_related_docs", 600),
            "min_infobox_types": cfg.get("topics.min_infobox_types", 3),
        },
        "llm": {
            "model": llm.model,
            "reasoning_effort": llm.reasoning_effort,
            "calls": llm.stats.calls,
            "cache_hits": llm.stats.cache_hits,
            "tokens_in": llm.stats.tokens_in,
            "tokens_out": llm.stats.tokens_out,
            "billed_tokens_in": llm.stats.billed_tokens_in,
            "billed_tokens_out": llm.stats.billed_tokens_out,
        },
        "n_candidates": len(candidates),
        "missing_phrase": missing,
        "candidates": candidates,
    }
    with io.open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, allow_unicode=True, sort_keys=False, width=100)

    print()
    print("후보 %d개 -> %s" % (len(candidates), out))
    print("LLM 호출 %d회 (캐시 히트 %d), 토큰 in/out = %d/%d (청구분 %d/%d)"
          % (llm.stats.calls, llm.stats.cache_hits, llm.stats.tokens_in, llm.stats.tokens_out,
             llm.stats.billed_tokens_in, llm.stats.billed_tokens_out))
    print("소요 %.1fs" % (time.time() - t0))
    if missing:
        print("문구 누락: " + repr(missing))
    print()
    print("=== 승인 게이트 (SPEC §2.4) ===")
    print("여기서 종료한다. 이후 단계는 실행하지 않는다.")
    print("사람이 검토해 data/topics_pilot50.yaml 로 저장하고 approved: true 를 넣어야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
