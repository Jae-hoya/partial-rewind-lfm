"""M1-C 재채점 — 확정(잠정) 추정자로 Test A 600 branch-round 를 다시 채점. **호출 0회.**

**이것이 남은 경로에서 긍정 결과가 나올 수 있는 유일한 지점이다.**

Test A 는 50질의 전부가 매 라운드 4/4 branch 를 재실행했다. 선택적 재실행이 **0회**다.
낭비율 0.5% 는 낭비가 없어서가 아니라 `sufficient(b)` 가 한 번도 참이 되지 않아서다.
grounding 절은 어느 라운드에서도 0/50 통과였다.

그러므로 묻는 것은 하나다: **추정자를 바꾸면 얼릴 branch 가 생기는가?**

  생긴다  -> Partial Rewind 의 전제가 선다. 낭비율의 모집단이 비지 않는다.
  안 생긴다 -> "검증된 연산자로도 이 난이도에서는 얼릴 branch 가 없다" 가 §VIII 의 결과다.
             (원고 v0.2 에서 이것도 논문이 서는 결과다 — 부정 결과가 본문이므로.)

추정자는 **잠정** MiniCheck-Flan-T5-Large 단일문서 전제다. 사람 라벨(V′)이 오기 전이라
확정이 아니고, 그래서 산출물에 `provisional=true` 를 박는다. 라벨이 오면 같은
스크립트를 그대로 다시 돌린다(호출 0).

verify 는 **두 절 모두** 계산한다 — `mean_b grounding >= tau_g_global` AND
`min_b coverage >= tau_c_global`. 원고 v0.4 가 coverage 절을 빠뜨렸던 오류를
여기서 반복하지 않도록 코드가 둘 다 계산하고 assert 한다.

집계 규약도 Test A 와 동일하게 맞춘다 — 그래야 0.5% 와 나란히 놓을 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from diag_sweep import pick_topics  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

TAUS = (0.7, 0.8, 0.9)
TAU_G_GLOBAL = 0.80
TAU_C_GLOBAL = 0.80


def main(argv=None):
    ap = argparse.ArgumentParser(description="M1-C 재채점 (호출 0회)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-topics", type=int, default=50)
    ap.add_argument("--outdir", default="results/m1c")
    ap.add_argument("--scorer", default="legacy", choices=["legacy", "singledoc"],
                    help="legacy=쪼개기 버그 있는 원본(기존 수치 재현) · "
                         "singledoc=chunk_size 강제한 확정 연산자")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(str(root / ".env"))
    cfg = load_config(args.config or (root / "config.yaml"))
    llm = LLM(cfg)
    retriever = BM25Retriever(cfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))
    print("토픽 %d개" % len(picked))

    # SPEC_M1prime §1 — 추정자는 estimator.py 경유. 기본은 예전 그대로다.
    if getattr(args, "scorer", "legacy") == "singledoc":
        from lfm.estimator import MiniCheckScorer as Scorer   # chunk_size 강제
    else:
        from d9_minicheck import Scorer                       # 쪼개기 버그 있는 원본
    print("MiniCheck 적재 ...", flush=True)
    sc = Scorer()

    rows = []
    for n, t in enumerate(picked):
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        for p in decompose(t["topic"], llm, ccfg := cfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            n_sub = len(o.sub_answers)
            n_cited_sub = sum(1 for sa in o.sub_answers
                              if any(c in cmap for c in sa.citations))
            coverage = (n_cited_sub / n_sub) if n_sub else 0.0

            sents, grounded = 0, 0
            for sa in o.sub_answers:
                cited = [cmap[c].text for c in sa.citations if c in cmap]
                for s_ in split_sentences(sa.text):
                    sents += 1
                    if not cited:
                        continue
                    if sc.score("\n\n".join(cited), s_) > 0.5:
                        grounded += 1
            grounding = (grounded / sents) if sents else 0.0

            rows.append({"query_id": tid, "branch": p.role,
                         "coverage": round(coverage, 6),
                         "grounding": round(grounding, 6),
                         "n_sentences": sents, "n_grounded": grounded,
                         "degraded": int(sents == 0)})
        print("  [%2d/%2d] %s" % (n + 1, len(picked), tid), flush=True)

    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    with io.open(outdir / "m1c_branches.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # --- branch 단위 충분성 -------------------------------------------------
    print("")
    print("=== M1-C 재채점 (잠정 MiniCheck 단일문서, 라운드 0) ===")
    print("branch %d개 · 빈 답안 %d" % (len(rows), sum(r["degraded"] for r in rows)))
    print("  coverage  중앙 %.3f · =1.0 인 branch %d"
          % (st.median([r["coverage"] for r in rows]),
             sum(1 for r in rows if r["coverage"] == 1.0)))
    print("  grounding 중앙 %.3f · >0.8 인 branch %d"
          % (st.median([r["grounding"] for r in rows]),
             sum(1 for r in rows if r["grounding"] > 0.8)))
    print("")
    print("| tau | sufficient(cov=1.0 AND grd>tau) | (대조) Test A g1 |")
    print("|---|---|---|")
    suff = {}
    ref = {0.7: "7/400 재실행 중 낭비, 충족 branch 3/600", 0.8: "충족 3/600", 0.9: "충족 3/600"}
    for tau in TAUS:
        k = sum(1 for r in rows if r["coverage"] == 1.0 and r["grounding"] > tau)
        suff[tau] = k
        print("| %.1f | **%d / %d = %.2f%%** | %s |"
              % (tau, k, len(rows), 100.0 * k / len(rows), ref[tau]))

    # --- 질의 단위 verify (두 절 모두) ---------------------------------------
    by_q: dict[str, list[dict]] = {}
    for r in rows:
        by_q.setdefault(r["query_id"], []).append(r)
    pg = pc = both = 0
    best_g = best_c = 0.0
    for q, bs in by_q.items():
        mg = st.mean([b["grounding"] for b in bs])
        mc = min(b["coverage"] for b in bs)
        best_g, best_c = max(best_g, mg), max(best_c, mc)
        a, b_ = mg >= TAU_G_GLOBAL, mc >= TAU_C_GLOBAL
        pg += a; pc += b_; both += (a and b_)
    print("")
    print("verify (두 절 모두 계산 — mean_b grounding >= %.2f AND min_b coverage >= %.2f)"
          % (TAU_G_GLOBAL, TAU_C_GLOBAL))
    print("  grounding 절 통과 %d/%d (최고 mean %.3f)" % (pg, len(by_q), best_g))
    print("  coverage  절 통과 %d/%d (최고 min  %.3f)" % (pc, len(by_q), best_c))
    print("  **둘 다 통과 %d/%d**   (대조: Test A g1 은 0/50, grounding 절 0/50)"
          % (both, len(by_q)))

    io.open(outdir / "m1c_summary.json", "w", encoding="utf-8").write(json.dumps({
        "provisional": True,
        "estimator": "MiniCheck-Flan-T5-Large (single-document premise, >0.5)",
        "note": "사람 라벨(V') 전이라 잠정. 라벨 확정 후 같은 스크립트로 재실행(호출 0).",
        "n_branches": len(rows), "n_queries": len(by_q),
        "n_degraded": sum(r["degraded"] for r in rows),
        "coverage_eq1": sum(1 for r in rows if r["coverage"] == 1.0),
        "grounding_gt08": sum(1 for r in rows if r["grounding"] > 0.8),
        "sufficient": {str(k): v for k, v in suff.items()},
        "verify": {"tau_g_global": TAU_G_GLOBAL, "tau_c_global": TAU_C_GLOBAL,
                   "pass_grounding": pg, "pass_coverage": pc, "pass_both": both,
                   "best_mean_grounding": round(best_g, 6),
                   "best_min_coverage": round(best_c, 6)},
        "minicheck_calls": sc.calls, "chunk_split_violations": sc.violations,
    }, ensure_ascii=False, indent=1))

    print("")
    print("MiniCheck 호출 %d · 내부 청크 분할 위반 %d (0 이어야 유효)"
          % (sc.calls, sc.violations))
    print("LLM 호출 %d (캐시 %d), 청구 %d/%d"
          % (llm.stats.calls, llm.stats.cache_hits,
             llm.stats.billed_tokens_in, llm.stats.billed_tokens_out))
    print("")
    print("분기 (ORDERS): 충족 branch **30/600 이상**이면 M2-min 비교 한 번.")
    print("-> " + str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
