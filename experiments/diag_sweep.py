"""SPEC M1-D v1.1 — 진단 스윕 D1~D3.

목적: 낭비율을 다시 재는 것이 **아니라**, 낭비율이 의미를 갖는 설정을 찾는 것.
branch 품질에 편차가 없으면 낭비율은 무엇을 하든 0이다.

D4 프로토콜 규칙을 코드로 강제한다:
  1. **낭비율을 계산하지 않는다.** 이 모듈은 `sufficient()` 도 `test_a` 도 import 하지
     않는다. 낭비율을 보고 조건을 고르면 본실험이 오염된다.
  2. 조건은 CONDITIONS 에 고정. 실행 중 추가하지 않는다.
  3. 판정 기준은 GATES 에 고정. 실행 후 조정하지 않는다.
  4. 프롬프트를 손대지 않는다 — 본실험 코드를 그대로 호출한다.
  5. 산출은 results/diag/{condition}/ 에 config 사본과 함께.

**라운드 0만 실행한다.** 본실험 토큰의 71.9% 가 라운드 1~2 에 쓰였고 진단에는
재실행이 필요 없다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.evidence import evaluate  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

# D2 — 조건 4개. 실행 전에 확정, 각 1회.
CONDITIONS: dict[str, dict[str, Any]] = {
    "C1_baseline": {"top_k": None, "index": "main"},
    "C2_topk15": {"top_k": 15, "index": "main"},
    "C3_partial_heldout": {"top_k": None, "index": "partial"},
    "C4_both": {"top_k": 15, "index": "partial"},
}

# D3 — 판정 기준. 실행 후 조정 금지.
GATES = {
    "intersection_rate": ("coverage=1.0 & grounding>0.8 비율", 0.10, "ge"),
    "cov1_grounding_median": ("coverage=1.0 branch 의 grounding 중앙값", 0.50, "ge"),
    "cov1_rate": ("coverage=1.0 비율", 0.30, "ge"),
    "empty_rate": ("빈 답안 비율", 0.15, "le"),
}
# 기록만 하고 판정에 쓰지 않는 것 (D3 표)
RECORD_ONLY = ("grounding_gt08_rate", "grounded_count_zero_rate", "grounded_count_median")

PILOT_IDS = ("t000", "t001", "t002", "t003", "t004")


def pick_topics(topics: list[dict], n: int, seed: int) -> list[dict]:
    """파일럿 제외 후 시드 고정 표집. 조건마다 같은 토픽이어야 비교가 성립한다."""
    pool = [t for t in topics if str(t["id"]) not in PILOT_IDS]
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False).tolist())
    return [pool[i] for i in idx]


def run_condition(name: str, spec: dict, cfg: Config, topics: list[dict], outdir: Path) -> dict:
    data = cfg.as_dict()
    if spec["top_k"] is not None:
        data["retrieval"]["top_k"] = spec["top_k"]
    if spec["index"] == "partial":
        data["paths"]["index_dir"] = data["paths"].get("index_dir_partial", "data/index_partial")
    ccfg = Config(data, source=name)

    llm = LLM(ccfg)
    nli = load_nli(ccfg)
    retriever = BM25Retriever(ccfg)

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, t in enumerate(topics):
        anchor = " ".join(t.get("terms") or [])
        perspectives = decompose(t["topic"], llm, ccfg)
        for p in perspectives:
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            ev = evaluate(o, nli, ccfg)
            cmap = o.chunk_map()
            n_sent = sum(len(split_sentences(sa.text)) for sa in o.sub_answers)
            n_entailed = round(ev.grounding * n_sent) if n_sent else 0
            cited_sub = sum(1 for sa in o.sub_answers if any(c in cmap for c in sa.citations))
            rows.append(
                {
                    "condition": name,
                    "query_id": str(t["id"]),
                    "branch": p.role,
                    "n_sentences": n_sent,
                    "n_entailed": n_entailed,
                    "n_sub_questions": len(o.sub_answers),
                    "n_cited_sub_questions": cited_sub,
                    "n_chunks": len(o.chunks),
                    "n_citations": len(o.citations),
                    "n_claims": len(o.claims),
                    "coverage": round(ev.coverage, 6),
                    "grounding_ratio": round(ev.grounding, 6),
                    "grounded_count": round(n_entailed / cited_sub, 6) if cited_sub else 0.0,
                    "source_diversity": round(ev.source_diversity, 6),
                    "support_strength": round(ev.support_strength, 6),
                    "degraded": int(o.degraded),
                }
            )
        print("  [%2d/%2d] %s" % (i + 1, len(topics), t["id"]), flush=True)

    cdir = outdir / name
    cdir.mkdir(parents=True, exist_ok=True)
    with io.open(cdir / "branches.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with io.open(cdir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(ccfg.as_dict(), fh, allow_unicode=True, sort_keys=False)

    m = metrics(rows)
    m.update(
        {
            "condition": name,
            "top_k": ccfg["retrieval.top_k"],
            "index_dir": ccfg["paths.index_dir"],
            "n_branches": len(rows),
            "elapsed_s": round(time.time() - t0, 1),
            "llm_calls": llm.stats.calls,
            "cache_hits": llm.stats.cache_hits,
            "billed_tokens_in": llm.stats.billed_tokens_in,
            "billed_tokens_out": llm.stats.billed_tokens_out,
        }
    )
    with io.open(cdir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(m, fh, ensure_ascii=False, indent=2)
    return m


def metrics(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    cov = [float(r["coverage"]) for r in rows]
    grd = [float(r["grounding_ratio"]) for r in rows]
    gc = [float(r["grounded_count"]) for r in rows]
    cov1 = [r for r in rows if float(r["coverage"]) == 1.0]
    return {
        "intersection_rate": sum(
            1 for r in rows if float(r["coverage"]) == 1.0 and float(r["grounding_ratio"]) > 0.8
        ) / n,
        "cov1_grounding_median": st.median([float(r["grounding_ratio"]) for r in cov1]) if cov1 else 0.0,
        "cov1_rate": len(cov1) / n,
        "empty_rate": sum(1 for r in rows if int(r["n_sentences"]) == 0) / n,
        "grounding_gt08_rate": sum(1 for x in grd if x > 0.8) / n,
        "grounded_count_zero_rate": sum(1 for x in gc if x == 0.0) / n,
        "grounded_count_median": st.median(gc),
        "coverage_mean": st.mean(cov),
        "grounding_mean": st.mean(grd),
        "sentences_median": st.median([int(r["n_sentences"]) for r in rows]),
    }


def judge(m: dict) -> tuple[bool, list[str]]:
    lines = []
    ok = True
    for key, (label, thr, direction) in GATES.items():
        v = m[key]
        passed = v >= thr if direction == "ge" else v <= thr
        ok = ok and passed
        sym = "≥" if direction == "ge" else "≤"
        lines.append("    %-42s %.3f  (기준 %s %.2f)  %s"
                     % (label, v, sym, thr, "PASS" if passed else "FAIL"))
    return ok, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SPEC M1-D v1.1 진단 스윕 (D1~D3)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--outdir", default="results/diag/sweep")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--only", default=None, help="조건 이름 하나만 실행 (디버그)")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(str(repo_root / ".env"))
    cfg = load_config(args.config or (repo_root / "config.yaml"))

    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config or (repo_root / "config.yaml"), outdir / "config_base.yaml")

    conds = {args.only: CONDITIONS[args.only]} if args.only else CONDITIONS
    print("토픽 %d개 (파일럿 %s 제외, seed=%s): %s"
          % (len(picked), ",".join(PILOT_IDS), cfg["seed"],
             ", ".join(str(t["id"]) for t in picked)))
    print("조건 %d개: %s" % (len(conds), ", ".join(conds)))
    print()

    results = []
    for name, spec in conds.items():
        print("=== %s (top_k=%s, index=%s) ===" % (name, spec["top_k"] or "기본", spec["index"]))
        m = run_condition(name, spec, cfg, picked, outdir)
        results.append(m)
        ok, lines = judge(m)
        print("\n".join(lines))
        print("    -> %s   (호출 %d, 캐시 %d, 청구 in/out %d/%d, %.1f분)"
              % ("전 기준 통과" if ok else "미통과", m["llm_calls"], m["cache_hits"],
                 m["billed_tokens_in"], m["billed_tokens_out"], m["elapsed_s"] / 60))
        print()

    with io.open(outdir / "summary.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print("=== D3 판정 요약 ===")
    print("| 조건 | 교집합 | cov1 grd중앙 | cov1 비율 | 빈답안 | (기록) grd>0.8 | 판정 |")
    print("|---|---|---|---|---|---|---|")
    for m in results:
        ok, _ = judge(m)
        print("| %s | %.1f%% | %.3f | %.1f%% | %.1f%% | %.1f%% | %s |"
              % (m["condition"], 100 * m["intersection_rate"], m["cov1_grounding_median"],
                 100 * m["cov1_rate"], 100 * m["empty_rate"],
                 100 * m["grounding_gt08_rate"], "PASS" if ok else "FAIL"))
    print()
    print("기준: 교집합 ≥10%% · cov1 grd중앙 ≥0.50 · cov1 비율 ≥30%% · 빈답안 ≤15%%")
    print("grounding>0.8 단독은 기록만 — 판정에 쓰지 않는다 (길이 편향 지표, D0 §2)")
    print("낭비율은 계산하지 않았다 (D4 규칙 1)")
    print("산출 -> " + str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
