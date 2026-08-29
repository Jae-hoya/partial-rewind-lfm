"""grounding 이 왜 안 움직이는가 — NLI entailment 확률 분포를 직접 본다.

D3 에서 4개 조건이 전부 미통과했다. 그중 C3(부분 held-out)는 **target article 의
본문을 인덱스에 남긴 조건**인데도 grounding 이 오르지 않았다. 근거가 있어도
grounding 이 안 오르면 병목은 근거 가용성이 아니라 **측정** 쪽이다.

`grounding` 은 `max_chunk p_entail(chunk, sentence) >= tau_ground` 인 문장의 비율이다.
확률이 임계 바로 아래에 몰려 있는지, 아니면 바닥에 깔려 있는지에 따라 처방이 갈린다.

  - 0.3~0.5 에 몰려 있다  -> tau_ground 캘리브레이션 문제. 임계를 재설정하면 된다
  - 0.0 근처에 깔려 있다  -> 문장 단위 NLI 로는 이 생성문을 못 잡는다. 추정자 교체 문제

LLM 호출 0회 (캐시 재생). NLI 는 GPU 로 다시 돌린다.
**낭비율을 계산하지 않는다** (SPEC M1-D D4 규칙 1).
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics as st
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_sweep import CONDITIONS, pick_topics  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="NLI entailment 확률 분포")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(str(repo_root / ".env"))
    cfg = load_config(args.config or (repo_root / "config.yaml"))

    spec = CONDITIONS[args.condition]
    data = cfg.as_dict()
    if spec["top_k"] is not None:
        data["retrieval"]["top_k"] = spec["top_k"]
    if spec["index"] == "partial":
        data["paths"]["index_dir"] = data["paths"]["index_dir_partial"]
    ccfg = Config(data, source=args.condition)

    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))

    llm = LLM(ccfg)
    nli = load_nli(ccfg)
    retriever = BM25Retriever(ccfg)
    tau = float(ccfg["nli.tau_ground"])

    scores: list[float] = []
    rows = []
    for n, t in enumerate(picked):
        anchor = " ".join(t.get("terms") or [])
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            for sa in o.sub_answers:
                cited = [c for c in sa.citations if c in cmap]
                if not cited:
                    continue
                for sent in split_sentences(sa.text):
                    best = max(nli.entailment([(cmap[c].text, sent)]) [0] for c in cited)
                    scores.append(best)
                    rows.append({"query_id": t["id"], "branch": p.role,
                                 "max_entailment": round(best, 6),
                                 "n_cited_chunks": len(cited),
                                 "sentence_chars": len(sent)})
        print("  [%2d/%2d] %s" % (n + 1, len(picked), t["id"]), flush=True)

    out = Path(args.out or ("results/diag/entail_profile_%s.csv" % args.condition))
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    s = sorted(scores)
    print()
    print("조건 %s · 인용 있는 문장 %d개" % (args.condition, len(s)))
    print("  중앙값 %.3f · 평균 %.3f · p75 %.3f · p90 %.3f · 최대 %.3f"
          % (st.median(s), st.mean(s), s[int(.75 * len(s))], s[int(.90 * len(s))], s[-1]))
    print()
    print("  구간별 분포")
    edges = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.01]
    for a, b in zip(edges, edges[1:]):
        k = sum(1 for x in s if a <= x < b)
        print("    %.2f~%.2f  %4d  %5.1f%%  %s" % (a, b, k, 100 * k / len(s), "#" * int(60 * k / len(s))))
    print()
    print("  임계별 grounded 문장 비율 (현재 tau_ground = %.2f)" % tau)
    for t2 in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        k = sum(1 for x in s if x >= t2)
        print("    tau=%.1f -> %5.1f%%%s" % (t2, 100 * k / len(s), "   <- 현재" if abs(t2 - tau) < 1e-9 else ""))
    print()
    print("LLM 호출 %d (캐시 %d), 청구 토큰 %d/%d"
          % (llm.stats.calls, llm.stats.cache_hits,
             llm.stats.billed_tokens_in, llm.stats.billed_tokens_out))
    print("-> " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
