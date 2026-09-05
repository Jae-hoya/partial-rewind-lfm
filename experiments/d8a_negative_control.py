"""D8-A — 음성 통제. 최댓값 집계가 얼마나 부풀리는지 잰다. 생성 LLM 호출 0회.

D7 은 **양성 통제만** 했다. 수정 후보(claim 가설 × 인용 청크 문장 전제 × max)는
claim 하나를 전제 문장 수십 개와 비교해 최댓값을 취한다. **최댓값 집계는 비교 대상이
많을수록 위로 치우친다.** 무관한 문장 30개 중 하나가 우연히 0.5 를 넘으면 그 claim 은
grounded 가 된다. 따라서 57.1% 가 얼마나 부풀려진 값인지 아직 모른다.

음성 통제: 각 claim 에 **다른 토픽**의 전제 풀을 배정하고 같은 계산을 돌린다.
지지될 리 없는 쌍이므로 여기서 나오는 grounded 비율이 곧 위양성률이다.

  통과 기준 (실행 전 고정): 교란 쌍의 grounded@0.5 <= 5%
  보조 보고: n_premise_sentences 와 max_p 의 Spearman rho.
             양의 상관이 뚜렷하면 최댓값 팽창이 실재한다 — 통과해도 Threats 에 적는다.

교란은 시드 고정 순환 시프트다: 토픽 i 의 claim -> 토픽 (i+7) mod 15 의 풀.
같은 토픽 배정이 0건임을 실행 전 assert 로 확인한다.
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

from d7_probe import max_entail  # noqa: E402
from diag_sweep import CONDITIONS, pick_topics  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

SHIFT = 7
PASS_THRESHOLD = 0.05


def spearman(a: list[float], b: list[float]) -> float:
    def rank(v):
        o = sorted(range(len(v)), key=lambda k: v[k])
        r = [0.0] * len(v)
        k = 0
        while k < len(o):
            j = k
            while j + 1 < len(o) and v[o[j + 1]] == v[o[k]]:
                j += 1
            for m in range(k, j + 1):
                r[o[m]] = (k + j) / 2 + 1
            k = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(len(a)))
    den = (sum((x - ma) ** 2 for x in ra) * sum((x - mb) ** 2 for x in rb)) ** 0.5
    return num / den if den else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D8-A 음성 통제 (생성 LLM 호출 0회)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--out", default="results/diag/d8/d8a_negative_control.csv")
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

    nli = load_nli(ccfg)          # torch/CUDA 를 먼저 (뒤에 올리면 세그폴트)
    llm = LLM(ccfg)

    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))
    retriever = BM25Retriever(ccfg)

    # --- 1) 토픽별 claim 과 전제 문장 풀 수집 -----------------------------
    claims_by_topic: dict[str, list[tuple[str, str]]] = {}   # tid -> [(branch, claim)]
    pool_by_topic: dict[str, list[str]] = {}
    for n, t in enumerate(picked):
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        claims_by_topic.setdefault(tid, [])
        pool: list[str] = []
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            for c in o.cited_chunks():
                pool.extend(split_sentences(c.text) or [c.text])
            for cl in o.claims:
                claims_by_topic[tid].append((p.role, cl))
        pool_by_topic[tid] = pool
        print("  [%2d/%2d] %s  claim %d · 전제문장 %d"
              % (n + 1, len(picked), tid, len(claims_by_topic[tid]), len(pool)), flush=True)

    tids = [str(t["id"]) for t in picked]
    mapping = {tids[i]: tids[(i + SHIFT) % len(tids)] for i in range(len(tids))}
    assert all(src != dst for src, dst in mapping.items()), "같은 토픽 배정이 있다"
    print()
    print("교란 배정 (i -> i+%d mod %d), 같은 토픽 0건 확인" % (SHIFT, len(tids)))

    # --- 2) 교란 풀로 채점 ------------------------------------------------
    rows = []
    for src in tids:
        dst = mapping[src]
        pool = pool_by_topic[dst]
        items = claims_by_topic[src]
        if not items or not pool:
            continue
        scores = max_entail(nli, pool, [c for _, c in items])
        for k, (branch, cl) in enumerate(items):
            rows.append({
                "claim_id": "%s:%s:%d" % (src, branch, k),
                "src_topic": src, "premise_topic": dst, "branch": branch,
                "n_premise_sentences": len(pool),
                "max_p": round(float(scores[k]), 6),
                "claim_chars": len(cl),
            })
        print("  %s <- %s : claim %d, grounded %d"
              % (src, dst, len(items), sum(1 for x in scores if x >= 0.5)), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- 3) 판정 ----------------------------------------------------------
    ps = [r["max_p"] for r in rows]
    fpr = sum(1 for x in ps if x >= 0.5) / len(ps)
    rho = spearman([float(r["n_premise_sentences"]) for r in rows], ps)

    print()
    print("=== D8-A 음성 통제 결과 ===")
    print("  교란 쌍 %d개 (claim %d × 다른 토픽 전제 풀)" % (len(rows), len(rows)))
    print("  전제 문장 수 중앙 %d" % st.median([r["n_premise_sentences"] for r in rows]))
    print("  max_p 중앙 %.4f · 평균 %.4f" % (st.median(ps), st.mean(ps)))
    print("  **위양성률 grounded@0.5 = %.1f%%**  (기준 <= %.0f%%)  -> %s"
          % (100 * fpr, 100 * PASS_THRESHOLD, "PASS" if fpr <= PASS_THRESHOLD else "FAIL"))
    print("  Spearman rho(전제 문장 수, max_p) = %+.3f" % rho)
    if rho > 0.2:
        print("    -> 최댓값 팽창이 실재한다. 통과하더라도 Threats 에 적는다.")
    print()
    print("  참고: 정상 배정에서의 claim×문장전제 grounded 는 57.1% 였다.")
    print("        위양성률을 뺀 순증이 실제 개선폭이다.")
    print()
    print("LLM 호출 %d (캐시 %d), 청구 토큰 %d/%d"
          % (llm.stats.calls, llm.stats.cache_hits,
             llm.stats.billed_tokens_in, llm.stats.billed_tokens_out))
    print("-> " + str(out))
    return 0 if fpr <= PASS_THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
