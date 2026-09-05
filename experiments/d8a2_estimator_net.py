"""D8-A 확장 — 추정자 3종의 **순증**(정상 − 위양성)을 나란히 잰다. 생성 호출 0회.

D8-A 가 밝힌 것: 수정 후보의 위양성률이 50.5% 다. 정상 배정 57.1% 에서 빼면 순증은
6.6%p 에 불과하다. **grounded 비율만 보고 추정자를 고르면 안 된다.** 최댓값 집계는
전제 풀이 클수록 위로 치우치므로, 풀 크기가 다른 구성을 grounded 로 직접 비교하는 것은
무의미하다.

그래서 세 구성을 **같은 방식으로** 정상·교란 양쪽에서 잰다.

  E1  현행         문장 가설 × 인용 청크 통째 전제        (풀 ~3)
  E2  수정 후보    claim 가설 × 인용 청크 문장 전제       (풀 ~96)
  E3  스펙 폴백    claim 가설 × 위 문장 중 BM25 상위 5    (풀 5, k 고정·스윕 금지)

판정 지표는 grounded 가 아니라 **순증 = 정상 grounded − 교란 grounded** 다.
교란은 D8-A 와 같은 시드 고정 순환 시프트(i -> i+7 mod 15).
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from d7_probe import max_entail  # noqa: E402
from d8a_negative_control import SHIFT, spearman  # noqa: E402
from diag_sweep import CONDITIONS, pick_topics  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

TOKEN = re.compile(r"[a-z0-9]+")
BM25_K = 5


def toks(s: str) -> list[str]:
    return TOKEN.findall(s.lower())


def bm25_topk(query: str, docs: list[str], k: int) -> list[str]:
    """전제 문장 중 가설과 어휘가 가장 겹치는 상위 k개. 결정론, 모델 없음."""
    if len(docs) <= k:
        return docs
    dts = [toks(d) for d in docs]
    n = len(dts)
    avgdl = sum(len(d) for d in dts) / n if n else 1.0
    df: Counter[str] = Counter()
    for d in dts:
        df.update(set(d))
    q = toks(query)
    scored = []
    for i, d in enumerate(dts):
        tf = Counter(d)
        dl = len(d) or 1
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * dl / avgdl))
        scored.append((-s, i))
    scored.sort()
    return [docs[i] for _, i in scored[:k]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="추정자 순증 비교 (생성 호출 0회)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--outdir", default="results/diag/d8")
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

    nli = load_nli(ccfg)
    llm = LLM(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))
    retriever = BM25Retriever(ccfg)

    # 토픽별: 문장 가설 / claim 가설 / 청크 전제 / 문장 전제
    S: dict[str, dict] = {}
    for n, t in enumerate(picked):
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        sents, claims, chunks, psents = [], [], [], []
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            for sa in o.sub_answers:
                if any(c in cmap for c in sa.citations):
                    sents.extend(split_sentences(sa.text))
            claims.extend(o.claims)
            for c in o.cited_chunks():
                chunks.append(c.text)
                psents.extend(split_sentences(c.text) or [c.text])
        S[tid] = {"sents": sents, "claims": claims, "chunks": chunks, "psents": psents}
        print("  [%2d/%2d] %s" % (n + 1, len(picked), tid), flush=True)

    tids = [str(t["id"]) for t in picked]
    shuffled = {tids[i]: tids[(i + SHIFT) % len(tids)] for i in range(len(tids))}
    assert all(a != b for a, b in shuffled.items())

    def score(estimator: str, src: str, pool_topic: str) -> list[float]:
        d, pd = S[src], S[pool_topic]
        if estimator == "E1":
            return max_entail(nli, pd["chunks"], d["sents"])
        if estimator == "E2":
            return max_entail(nli, pd["psents"], d["claims"])
        out = []
        for cl in d["claims"]:
            out.extend(max_entail(nli, bm25_topk(cl, pd["psents"], BM25_K), [cl]))
        return out

    rows, summary = [], []
    for est, label in [("E1", "현행: 문장가설 × 청크전제"),
                       ("E2", "수정후보: claim × 문장전제 전체"),
                       ("E3", "폴백: claim × BM25 상위 %d문장" % BM25_K)]:
        norm, shuf, pool_sizes = [], [], []
        for tid in tids:
            n_sc = score(est, tid, tid)
            s_sc = score(est, tid, shuffled[tid])
            norm.extend(n_sc)
            shuf.extend(s_sc)
            pool = (S[tid]["chunks"] if est == "E1"
                    else S[tid]["psents"] if est == "E2" else [0] * BM25_K)
            pool_sizes.extend([len(pool)] * len(n_sc))
            for k in range(len(n_sc)):
                rows.append({"estimator": est, "topic": tid, "idx": k,
                             "p_normal": round(n_sc[k], 6),
                             "p_shuffled": round(s_sc[k], 6),
                             "pool_size": len(pool)})
        g_n = sum(1 for x in norm if x >= 0.5) / len(norm)
        g_s = sum(1 for x in shuf if x >= 0.5) / len(shuf)
        rho = spearman([float(x) for x in pool_sizes], norm)
        summary.append({"estimator": est, "label": label, "n": len(norm),
                        "pool_median": st.median(pool_sizes),
                        "grounded_normal": round(g_n, 4),
                        "grounded_shuffled": round(g_s, 4),
                        "net": round(g_n - g_s, 4),
                        "rho_poolsize_p": round(rho, 3)})
        print("  %s 완료  정상 %.1f%% / 교란 %.1f%% / 순증 %.1f%%p"
              % (est, 100 * g_n, 100 * g_s, 100 * (g_n - g_s)), flush=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with io.open(outdir / "d8a2_estimator_net.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with io.open(outdir / "d8a2_summary.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print()
    print("=== 추정자 순증 비교 (판정 지표는 순증이다) ===")
    print("| 추정자 | 전제 풀 중앙 | n | 정상 | 교란(위양성) | **순증** | rho(풀크기,p) |")
    print("|---|---|---|---|---|---|---|")
    for s in summary:
        print("| %s %s | %d | %d | %.1f%% | %.1f%% | **%.1f%%p** | %+.3f |"
              % (s["estimator"], s["label"], s["pool_median"], s["n"],
                 100 * s["grounded_normal"], 100 * s["grounded_shuffled"],
                 100 * s["net"], s["rho_poolsize_p"]))
    print()
    best = max(summary, key=lambda s: s["net"])
    print("순증 최대: %s (%.1f%%p)" % (best["estimator"], 100 * best["net"]))
    print("D8-A 통과 기준(교란 <= 5%%) 충족: %s"
          % ", ".join(s["estimator"] for s in summary if s["grounded_shuffled"] <= 0.05) or "없음")
    print()
    print("LLM 호출 %d (캐시 %d), 청구 토큰 %d/%d"
          % (llm.stats.calls, llm.stats.cache_hits,
             llm.stats.billed_tokens_in, llm.stats.billed_tokens_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
