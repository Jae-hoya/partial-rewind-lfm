"""D0 — branch 별 답안 문장 수를 뽑아 grounding 과 교차한다. **LLM 호출 0회.**

본실행의 모든 LLM 응답이 디스크 캐시에 있으므로 파이프라인을 그대로 재생하면
같은 답안이 나온다(결정성은 이미 실증됨). 여기서는 낭비율을 계산하지 않고
문장 수·인용 수만 센다.

동기: "grounding 이 높은 branch 는 답안 문장이 적어서 그런 것 아닌가" 라는 의심.
grounding = (entail 된 문장 수) / (전체 문장 수) 이므로 분모가 작으면 1.0 이 쉽다.
사실이면 grounding 은 품질이 아니라 **길이의 역수**를 재고 있는 셈이고, 이 지표를
목표로 삼는 어떤 스윕도 잘못된 방향으로 간다.

캐시가 없으면 실 API 를 호출하게 되므로 기본은 캐시 전용 모드다 (--allow-api 로 해제).
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.evidence import evaluate  # noqa: E402
from lfm.llm import LLM, LLMError, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402


class CacheOnlyTransport:
    """캐시 미스면 예외. 이 감사는 새 토큰을 쓰지 않는다."""

    def __init__(self) -> None:
        self.misses = 0

    def __call__(self, *, model, system, prompt, max_tokens, reasoning_effort=None):
        self.misses += 1
        raise LLMError("캐시 미스 — 이 스크립트는 캐시만 사용한다 (--allow-api 로 해제)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D0 문장 수 감사 (LLM 호출 0회)")
    ap.add_argument("--results-dir", default="results/test_a/20260829-041445")
    ap.add_argument("--out", default="results/diag/sentence_count.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-api", action="store_true")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(str(repo_root / ".env"))

    # 실행 당시 config 사본을 쓴다 — 현재 config 가 바뀌었어도 캐시 키가 맞아야 한다.
    rd = Path(args.results_dir)
    cfg = load_config(rd / "config.yaml")
    data_dir = Path(cfg["paths.data_dir"])

    with io.open(data_dir / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    if args.limit:
        topics = topics[: args.limit]

    transport = None if args.allow_api else CacheOnlyTransport()
    llm = LLM(cfg, transport=transport)
    nli = load_nli(cfg)
    retriever = BM25Retriever(cfg)
    max_rounds = int(cfg["max_rounds"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()

    for i, t in enumerate(topics):
        anchor = " ".join(t.get("terms") or [])
        perspectives = decompose(t["topic"], llm, cfg)
        for r in range(max_rounds):
            for p in perspectives:
                o = run_branch(p, retriever, llm, cfg, round_=r, search_anchor=anchor)
                ev = evaluate(o, nli, cfg)
                cmap = o.chunk_map()
                n_sent = 0
                n_cited_sent = 0
                for sa in o.sub_answers:
                    sents = split_sentences(sa.text)
                    n_sent += len(sents)
                    if any(c in cmap for c in sa.citations):
                        n_cited_sent += len(sents)
                rows.append(
                    {
                        "query_id": str(t["id"]),
                        "round": r,
                        "branch": p.role,
                        "n_sentences": n_sent,
                        "n_cited_sentences": n_cited_sent,
                        "n_entailed": round(ev.grounding * n_sent) if n_sent else 0,
                        "n_sub_questions": len(o.sub_answers),
                        "n_answered_sub": sum(1 for sa in o.sub_answers if sa.text.strip()),
                        "n_claims": len(o.claims),
                        "n_chunks": len(o.chunks),
                        "n_citations": len(o.citations),
                        "coverage": round(ev.coverage, 6),
                        "grounding": round(ev.grounding, 6),
                        "support_strength": round(ev.support_strength, 6),
                        "degraded": int(o.degraded),
                    }
                )
        print("[%3d/%3d] %s" % (i + 1, len(topics), t["id"]), flush=True)

    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print("행 %d -> %s (%.1f분)" % (len(rows), out, (time.time() - t0) / 60))
    print("LLM 호출 %d, 캐시 히트 %d, 청구 토큰 in/out = %d/%d"
          % (llm.stats.calls, llm.stats.cache_hits,
             llm.stats.billed_tokens_in, llm.stats.billed_tokens_out))
    if llm.stats.billed_tokens_in or llm.stats.billed_tokens_out:
        print("경고: 캐시 미스가 있었다. 감사 결과가 본실행과 다를 수 있다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
