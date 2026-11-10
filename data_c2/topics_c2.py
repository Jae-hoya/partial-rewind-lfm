"""PREREG_R3 §4.2 토픽 생성 — `data/topics.py` 를 고치지 않고 감싼다.

R2 와 같은 절차(`topics.load_docs`·`build_postings`·`load_meta`·`distinctive_terms`·`phrase_topics` 를 그대로 import).
다른 점은 §4.2 에 적은 두 가지뿐이다.
  1. **교차 분야 조건 추가**: 이웃 문서의 서로 다른 분류 ≥ topics.min_infobox_types(기존, 3) **그리고**
     서로 다른 archive ≥ topics.min_archives(2). 다양성: 주 분류당 최대 topics.max_per_infobox_type(2),
     target 주 분류의 archive 당 최대 topics.max_per_archive(8).
     `find_candidates_c2` 는 `topics.find_candidates` 의 사본에 이 두 줄만 더했다 — 두 조건을 끄면 원본과 같은 결과를 내는지
     `tests/test_c2.py` 가 대조한다.
  2. PHRASE_PROMPT 의 "an encyclopedia corpus" → "a corpus of scientific paper abstracts" 한 군데. 실행 중에만
     `topics.PHRASE_PROMPT` 를 바꾸고 두 해시를 후보 파일과 MANIFEST 에 적는다. **분해 프롬프트(lfm/perspectives.py)는 건드리지 않는다.**

  python data_c2/topics_c2.py --config config_c2.yaml      # LLM 호출 (topic_phrase, 60후보 약 6배치)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

import topics as T  # noqa: E402  (data/topics.py)
from lfm.config import load_config  # noqa: E402

PHRASE_OLD = "an encyclopedia corpus"
PHRASE_NEW = "a corpus of scientific paper abstracts"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def c2_phrase_prompt(original: str) -> str:
    assert original.count(PHRASE_OLD) == 1, "PHRASE_PROMPT 에 '%s' 가 정확히 한 번 있어야 한다" % PHRASE_OLD
    return original.replace(PHRASE_OLD, PHRASE_NEW)


def archive_of(category: str) -> str:
    return category.split(".", 1)[0]


def doc_archives(doc) -> set[str]:
    return {archive_of(t) for t in doc["infobox_types"]}


def find_candidates_c2(docs, postings, meta, cfg) -> list[dict[str, Any]]:
    """`topics.find_candidates` 사본 + (a) 이웃 archive ≥ min_archives (b) archive 당 상한. 표시한 줄만 다르다."""
    min_rel = int(cfg.get("topics.min_related_docs", 30))
    min_types = int(cfg.get("topics.min_infobox_types", 3))
    min_hits = int(cfg.get("topics.min_term_hits", 2))
    min_abs = int(cfg.get("topics.min_target_abstract_chars", 600))
    min_sec = int(cfg.get("topics.min_target_section_chars", 6000))
    min_ref = int(cfg.get("topics.min_target_references", 20))
    per_type = int(cfg.get("topics.max_per_infobox_type", 3))
    want = int(cfg.get("topics.n_candidates", 60))
    min_arch = int(cfg.get("topics.min_archives", 0))            # C2 추가
    per_arch = cfg.get("topics.max_per_archive", None)            # C2 추가 (None = 상한 없음)
    per_arch = None if per_arch is None else int(per_arch)
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

        terms = T.distinctive_terms(d, postings, n_docs, cfg)
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
        archives = {archive_of(t) for t in types}                    # C2 추가
        if len(archives) < min_arch:                                 # C2 추가
            continue
        scored.append(
            {
                "term": " ".join(terms[:3]),
                "terms": terms,
                "target_idx": i,
                "related_docs": len(related),
                "n_infobox_types": len(types),
                "infobox_types": sorted(types)[:12],
                "primary_type": T.primary_type(d),
                "n_archives": len(archives),                         # C2 추가
                "archives": sorted(archives)[:12],                   # C2 추가
            }
        )

    scored.sort(key=lambda c: (-c["related_docs"], -c["n_infobox_types"], c["target_idx"]))

    picked: list[dict[str, Any]] = []
    per_type_count: Counter[str] = Counter()
    per_arch_count: Counter[str] = Counter()                         # C2 추가
    seen_targets: set[int] = set()
    for c in scored:
        if c["target_idx"] in seen_targets:
            continue
        if per_type_count[c["primary_type"]] >= per_type:
            continue
        pa_ = archive_of(c["primary_type"])                         # C2 추가
        if per_arch is not None and per_arch_count[pa_] >= per_arch:  # C2 추가
            continue
        per_type_count[c["primary_type"]] += 1
        per_arch_count[pa_] += 1                                     # C2 추가
        seen_targets.add(c["target_idx"])
        picked.append(c)
        if len(picked) >= want:
            break
    return picked


def manifest_note(data_dir: Path, key: str, value: Any) -> None:
    p = data_dir / "MANIFEST_C2.json"
    if not p.exists():
        return
    m = json.load(io.open(p, encoding="utf-8"))
    m[key] = value
    io.open(p, "w", encoding="utf-8").write(json.dumps(m, ensure_ascii=False, indent=1))


def require_license(data_dir: Path) -> None:
    p = data_dir / "MANIFEST_C2.json"
    st = (json.load(io.open(p, encoding="utf-8")).get("license_gate") or {}).get("status") if p.exists() else "no_manifest"
    if st != "passed":
        raise SystemExit("C-L 이 통과로 기록되지 않았다 (status=%s). data_c2/license_gate_c2.py 먼저 (PREREG_R3 §4.2)" % st)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="코퍼스2 토픽 후보 (PREREG_R3 §4.2)")
    ap.add_argument("--config", default=str(ROOT / "config_c2.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-candidates", type=int, default=None, help="이탈 경로(120)에서만. 기본은 config")
    args = ap.parse_args(argv)

    from lfm.llm import LLM, BillingLedger, load_dotenv_if_present

    load_dotenv_if_present(str(ROOT / ".env"))
    cfg = load_config(args.config)
    if args.n_candidates:
        d = cfg.as_dict(); d["topics"]["n_candidates"] = args.n_candidates
        from lfm.config import Config
        cfg = Config(d, source=str(args.config) + ":n_candidates=%d" % args.n_candidates)
    data_dir = Path(cfg["paths.data_dir"])
    require_license(data_dir)
    corpus = data_dir / "corpus.parquet"
    out = Path(args.out or (data_dir / "topics_pilot50_candidates.yaml"))
    min_len = int(cfg.get("topics.min_term_len", 4))

    t0 = time.time()
    docs = T.load_docs(corpus)
    postings = T.build_postings(docs, min_len)
    meta = T.load_meta(corpus)
    print("문서 %d건, 용어 %d개" % (len(docs), len(postings)), flush=True)
    picked = find_candidates_c2(docs, postings, meta, cfg)
    print("교차 분야 조건 통과 후보 %d개" % len(picked), flush=True)
    if not picked:
        return 1

    old = T.PHRASE_PROMPT
    new = c2_phrase_prompt(old)
    hashes = {"phrase_prompt_r2_sha256": sha(old), "phrase_prompt_c2_sha256": sha(new),
              "phrase_system_sha256": sha(T.PHRASE_SYSTEM), "change": "%r -> %r (한 군데)" % (PHRASE_OLD, PHRASE_NEW)}
    ledger = BillingLedger(ROOT / "results/c2/topics_ledger.jsonl")
    llm = LLM(cfg, ledger=ledger)
    T.PHRASE_PROMPT = new
    try:
        phrases = T.phrase_topics(llm, docs, picked, int(cfg.get("topics.batch_size", 10)))
    finally:
        T.PHRASE_PROMPT = old

    candidates = []
    for i, c in enumerate(picked):
        d = docs[c["target_idx"]]
        candidates.append({
            "id": "t%03d" % i, "term": c["term"], "terms": c["terms"],
            "primary_infobox_type": c["primary_type"], "topic": phrases.get(i, ""),
            "target_article": d["name"], "target_identifier": int(d["identifier"]), "target_url": d["url"],
            "related_docs": c["related_docs"], "n_infobox_types": c["n_infobox_types"],
            "infobox_types": c["infobox_types"], "n_archives": c["n_archives"], "archives": c["archives"],
        })
    missing = [c["id"] for c in candidates if not c["topic"]]
    doc = {
        "approved": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "spec": "PREREG_R3 §4.2 (data/topics.py 절차 + archive 조건)",
        "criteria": {"min_related_docs": cfg.get("topics.min_related_docs", 30),
                     "min_infobox_types": cfg.get("topics.min_infobox_types", 3),
                     "min_archives": cfg.get("topics.min_archives"), "max_per_archive": cfg.get("topics.max_per_archive"),
                     "max_per_infobox_type": cfg.get("topics.max_per_infobox_type"),
                     "n_candidates": cfg.get("topics.n_candidates")},
        "prompt_hashes": hashes,
        "llm": {"model": llm.model, "reasoning_effort": llm.reasoning_effort, "calls": llm.stats.calls,
                "cache_hits": llm.stats.cache_hits, "api_calls": llm.stats.api_calls,
                "billed_tokens_in": llm.stats.billed_tokens_in, "billed_tokens_out": llm.stats.billed_tokens_out},
        "n_candidates": len(candidates), "missing_phrase": missing, "candidates": candidates,
    }
    with io.open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, allow_unicode=True, sort_keys=False, width=100)
    manifest_note(data_dir, "topic_phrase_prompt", hashes)
    print("후보 %d개 -> %s · LLM api %d · %.1fs" % (len(candidates), out, llm.stats.api_calls, time.time() - t0))
    if missing:
        print("문구 누락: %r" % missing)
    print("다음: data_c2/approve_c2.py --config config_c2.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
