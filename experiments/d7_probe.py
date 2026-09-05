"""D7 확장 판별 — 전제 풀 3종 × 가설 단위 2종. 생성 LLM 호출 0회.

D7 스펙의 H1~H3 와 이후 제기된 두 가설을 한 번에 가른다.

전제 풀 (premise pool)
  cited      인용된 청크만 — 현행 grounding 정의
  retrieved  검색된 전 청크 — **귀속 오류** 가설 (H4). 이미 기각됨(+1.7%p)
  target     target article 전체 청크 — **파라메트릭 누출** 가설 (H5)

H5 가 왜 결정적인가: 코퍼스가 Wikipedia 이고 생성 모델이 Wikipedia 를 학습했다.
모델이 검색 청크가 아니라 **기억**으로 target article 의 사실을 쓸 수 있다. 그 사실이
target article 안에는 있지만 top-k 에 안 들어온 청크에 있다면, 검색된 청크 어느 것도
entail 하지 않는 것이 정상이다. 지어낸 것도 아니고 추정자 오류도 아닌데 0 이 나온다.

  target 풀에서 0.05 미만 덩어리가 크게 올라오면
    -> 결론이 "추정자가 틀렸다" 가 아니라 **"답안이 검색 근거에서 나오지 않는다"** 로 바뀐다
  안 올라오면
    -> 그 문장들은 target article 에도 없다. 추정자 문제이거나 실제 미지지다

가설 단위 (hypothesis unit)
  sentence   답안 문장 — 현행
  claim      저장된 atomic claim — 긴 복합문이 한 절 때문에 0 으로 떨어지는 문제 회피

**낭비율을 계산하지 않는다.** test_a / sufficient 를 import 하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics as st
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from diag_sweep import CONDITIONS, pick_topics  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

POOLS = ("cited", "retrieved", "target")
UNITS = ("sentence", "claim")
BINS = [0.0, 0.05, 0.1, 0.3, 0.5, 0.9, 1.01]


def target_chunks(cfg) -> dict[str, list[str]]:
    """토픽별 target article 의 **전 청크**. partial 인덱스가 뺀 lead 까지 포함한다."""
    import pyarrow.parquet as pq
    import tiktoken
    from chunk_index import split_by_tokens, walk_sections

    data_dir = Path(cfg["paths.data_dir"])
    with io.open(data_dir / "heldout_ids.json", encoding="utf-8") as fh:
        held = json.load(fh)
    by_topic = {k: set(int(i) for i in v["heldout_identifiers"])
                for k, v in held["by_topic"].items()}
    wanted = set().union(*by_topic.values()) if by_topic else set()

    enc = tiktoken.get_encoding("o200k_base")
    max_tokens = int(cfg.get("retrieval.chunk_max_tokens", 1200))
    per_doc: dict[int, list[str]] = {}
    pf = pq.ParquetFile(data_dir / "corpus.parquet")
    for batch in pf.iter_batches(batch_size=2000, columns=["identifier", "sections"]):
        for row in batch.to_pylist():
            did = int(row["identifier"])
            if did not in wanted or did in per_doc:
                continue
            try:
                sections = json.loads(row["sections"]) if row["sections"] else []
            except (json.JSONDecodeError, TypeError):
                sections = []
            pieces: list[str] = []
            for _, text in walk_sections(sections):
                pieces.extend(split_by_tokens(text, enc, max_tokens))
            per_doc[did] = pieces

    out: dict[str, list[str]] = {}
    for tid, ids in by_topic.items():
        acc: list[str] = []
        for did in sorted(ids):
            acc.extend(per_doc.get(did, []))
        out[tid] = acc
    return out


def max_entail(nli, premises: list[str], hypotheses: list[str]) -> list[float]:
    """가설마다 전제 전체에 대한 최댓값. 한 번에 배치로 넘긴다."""
    if not premises or not hypotheses:
        return [0.0] * len(hypotheses)
    pairs = [(p, h) for h in hypotheses for p in premises]
    scores = nli.entailment(pairs)
    n = len(premises)
    return [max(scores[i * n:(i + 1) * n]) for i in range(len(hypotheses))]


def profile(scores: list[float]) -> dict:
    s = sorted(scores)
    if not s:
        return {}
    out = {"n": len(s), "median": round(st.median(s), 4), "mean": round(st.mean(s), 4)}
    for a, b in zip(BINS, BINS[1:]):
        out["bin_%.2f_%.2f" % (a, b)] = round(sum(1 for x in s if a <= x < b) / len(s), 4)
    out["grounded_at_0.5"] = round(sum(1 for x in s if x >= 0.5) / len(s), 4)
    out["mid_band_0.1_0.9"] = round(sum(1 for x in s if 0.1 <= x < 0.9) / len(s), 4)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D7 확장 판별 (생성 LLM 호출 0회)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--outdir", default="results/diag/d7")
    ap.add_argument("--split-premises", action="store_true",
                    help="전제를 문장 단위로 쪼개 max 를 잡는다 (D7-0 이 확인한 길이 희석 회피)")
    ap.add_argument("--pools", default=",".join(POOLS),
                    help="계산할 전제 풀. 쉼표 구분")
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

    # 로드 순서가 중요하다. NLI(torch/CUDA)를 코퍼스·인덱스 뒤에 올리면 세그폴트가 난다
    # (RSS 2.5GB 상태에서 CUDA 초기화 실패). NLI 를 가장 먼저 올린다.
    nli = load_nli(ccfg)
    llm = LLM(ccfg)

    print("target article 청크 로드 중 ...", flush=True)
    tchunks = target_chunks(cfg)
    print("  토픽별 target 청크 중앙 %d개"
          % st.median([len(tchunks.get(str(t["id"]), [])) for t in picked]))

    retriever = BM25Retriever(ccfg)

    want_pools = [x.strip() for x in args.pools.split(",") if x.strip()]

    def prep(texts: list[str]) -> list[str]:
        if not args.split_premises:
            return texts
        out: list[str] = []
        for x in texts:
            out.extend(split_sentences(x) or [x])
        return out

    rows: list[dict] = []
    for n, t in enumerate(picked):
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        tpool = tchunks.get(tid, [])
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            retrieved = [c.text for c in o.chunks]

            units: dict[str, list[tuple[str, list[str]]]] = {"sentence": [], "claim": []}
            for sa in o.sub_answers:
                cited_ids = [c for c in sa.citations if c in cmap]
                if not cited_ids:
                    continue
                cited = [cmap[c].text for c in cited_ids]
                for sent in split_sentences(sa.text):
                    units["sentence"].append((sent, cited))
            branch_cited = [c.text for c in o.cited_chunks()]
            for cl in o.claims:
                units["claim"].append((cl, branch_cited))

            for unit_name, items in units.items():
                if not items:
                    continue
                hyps = [h for h, _ in items]
                pools = {
                    "cited": None,   # 항목마다 다르므로 개별 처리
                    "retrieved": retrieved,
                    "target": tpool,
                }
                cited_scores = ([max_entail(nli, prep(prem), [h])[0] for h, prem in items]
                                if "cited" in want_pools else [0.0] * len(hyps))
                retr_scores = (max_entail(nli, prep(retrieved), hyps)
                               if "retrieved" in want_pools else [0.0] * len(hyps))
                targ_scores = (max_entail(nli, prep(tpool), hyps)
                               if "target" in want_pools else [0.0] * len(hyps))
                for k, h in enumerate(hyps):
                    rows.append({
                        "query_id": tid, "branch": p.role, "unit": unit_name,
                        "text_chars": len(h),
                        "n_cited": len(items[k][1]), "n_retrieved": len(retrieved),
                        "n_target": len(tpool),
                        "p_cited": round(cited_scores[k], 6),
                        "p_retrieved": round(retr_scores[k], 6),
                        "p_target": round(targ_scores[k], 6),
                    })
        print("  [%2d/%2d] %s" % (n + 1, len(picked), tid), flush=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "_sentprem" if args.split_premises else ""
    with io.open(outdir / ("pool_unit_grid%s.csv" % tag), "w", encoding="utf-8",
                 newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = []
    print()
    print("=== 전제 풀 × 가설 단위 격자 ===")
    print("| 단위 | 전제 풀 | n | grounded@0.5 | 중간대 0.1~0.9 | <0.05 | 중앙값 |")
    print("|---|---|---|---|---|---|---|")
    for unit in UNITS:
        sub = [r for r in rows if r["unit"] == unit]
        for pool in want_pools:
            sc = [float(r["p_" + pool]) for r in sub]
            pr = profile(sc)
            if not pr:
                continue
            summary.append({"unit": unit, "pool": pool, **pr})
            print("| %s | %s | %d | %.1f%% | %.1f%% | %.1f%% | %.3f |"
                  % (unit, pool, pr["n"], 100 * pr["grounded_at_0.5"],
                     100 * pr["mid_band_0.1_0.9"], 100 * pr["bin_0.00_0.05"], pr["median"]))
    with io.open(outdir / ("pool_unit_summary%s.csv" % tag), "w", encoding="utf-8",
                 newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print()
    sent = [r for r in rows if r["unit"] == "sentence"]
    low = [r for r in sent if float(r["p_retrieved"]) < 0.05]
    lifted = [r for r in low if float(r["p_target"]) >= 0.5]
    print("=== H5 파라메트릭 누출 판별 ===")
    print("  검색 청크로 <0.05 였던 문장 %d개 중" % len(low))
    print("    target article 전체로 >=0.5 로 올라간 것: %d (%.1f%%)"
          % (len(lifted), 100 * len(lifted) / len(low) if low else 0))
    print("    여전히 <0.05 인 것: %d (%.1f%%)"
          % (sum(1 for r in low if float(r["p_target"]) < 0.05),
             100 * sum(1 for r in low if float(r["p_target"]) < 0.05) / len(low) if low else 0))
    print()
    print("LLM 호출 %d (캐시 %d), 청구 토큰 %d/%d"
          % (llm.stats.calls, llm.stats.cache_hits,
             llm.stats.billed_tokens_in, llm.stats.billed_tokens_out))
    print("NLI 쌍 %d개" % nli.n_pairs)
    print("-> " + str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
