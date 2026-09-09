"""M2-min 2단계 — 라이브 비교. GlobalRetry(R=3) vs 축소 컨트롤러. **생성 호출 0회.**

원고 §VIII-C 표 16 자리. 공저자 지시:

> 45 queries, one seed, both systems with the accepted operator in the verifier,
> prompts and retrieval frozen, no judge. Global retry (R=3) versus the reduced
> controller (FREEZE + EXPAND only; uncited branches regenerated with citation
> format enforced). Report: generation calls, tokens, coverage, grounding,
> uncited rate, verification passes, frozen branches, sub-questions re-executed,
> bootstrap CIs over queries.

**왜 호출 0 으로 되는가.** `run_branch` 의 캐시 키는 (관점, 라운드, 설정)에만 의존하고
**이전 라운드 상태를 보지 않는다**(`cache_nonce="r{round}"`). Test A 는 verify 가 한 번도
통과하지 않아 **모든 branch 를 모든 라운드에서 실행**했다. 따라서 축소 컨트롤러가 고를
수 있는 branch-round 는 전부 이미 캐시에 있다. 궤적만 다르게 고르는 것이다.

**단, 여기서 실행 단위는 branch 다. sub-question 이 아니다.**
스펙의 최소 변경 원칙은 sub-question 단위이지만, 그 단위의 재실행 artifact 는 캐시에
없다(Test A 가 branch 통째로만 돌렸다). 그 단위로 가면 새 생성 호출이 필요하다.
**그러므로 이 비교는 Partial Rewind 의 절감을 과소평가한다** — sub-question 단위면
재실행량이 더 줄어든다. 원고에 그렇게 적어야 한다.

토큰은 추정하지 않는다. Test A 의 `waste_ratio.csv` 에 (질의, branch, 라운드)별 실제
토큰이 있으므로 **실행한 단위의 실측 토큰을 그대로 더한다.**

verify 는 두 절 모두 계산한다 (mean_b grounding, min_b coverage).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from diag_sweep import pick_topics  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

TAU = 0.8                 # sufficient@tau (엄격 부등호)
TAU_G_GLOBAL = 0.80
TAU_C_GLOBAL = 0.80
MAX_ROUNDS = 3
BOOT = 10000
SEED = 42


def evaluate(o, sc):
    """MiniCheck 단일문서 전제로 (coverage, grounding, n_sub, uncited) 계산."""
    cmap = o.chunk_map()
    n_sub = len(o.sub_answers)
    n_cited = sum(1 for sa in o.sub_answers if any(c in cmap for c in sa.citations))
    cov = (n_cited / n_sub) if n_sub else 0.0
    sents = grounded = 0
    for sa in o.sub_answers:
        cited = [cmap[c].text for c in sa.citations if c in cmap]
        for s_ in split_sentences(sa.text):
            sents += 1
            if cited and sc.score("\n\n".join(cited), s_) > 0.5:
                grounded += 1
    grd = (grounded / sents) if sents else 0.0
    return cov, grd, n_sub, int(sents == 0 or n_cited == 0)


def boot_ci(vals, n=BOOT, seed=SEED):
    if not vals:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    a = np.array(vals, dtype=float)
    b = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return (float(lo), float(hi))


def main(argv=None):
    ap = argparse.ArgumentParser(description="M2-min 라이브 비교 (호출 0회)")
    ap.add_argument("--n-topics", type=int, default=50)
    ap.add_argument("--testa", default="results/test_a/20260829-041445/waste_ratio.csv")
    ap.add_argument("--outdir", default="results/m1c")
    ap.add_argument("--scorer", default="legacy", choices=["legacy", "singledoc"],
                    help="legacy=쪼개기 버그 있는 원본(기존 수치 재현) · "
                         "singledoc=chunk_size 강제한 확정 연산자")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(str(root / ".env"))
    cfg = load_config(root / "config.yaml")
    llm = LLM(cfg)
    retriever = BM25Retriever(cfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))

    # (질의, branch, 라운드) -> 실측 토큰
    tok = {}
    for r in csv.DictReader(io.open(root / args.testa, encoding="utf-8")):
        tok[(r["query_id"], r["branch"], int(r["round"]))] = int(r["tokens"])

    # SPEC_M1prime §1 — 추정자는 estimator.py 경유. 기본은 예전 그대로다.
    if getattr(args, "scorer", "legacy") == "singledoc":
        from lfm.estimator import MiniCheckScorer as Scorer   # chunk_size 강제
    else:
        from d9_minicheck import Scorer                       # 쪼개기 버그 있는 원본
    print("MiniCheck 적재 ...", flush=True)
    sc = Scorer()

    per_q = []
    for n, t in enumerate(picked):
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        try:
            persps = decompose(t["topic"], llm, cfg)
        except Exception:                                    # noqa: BLE001
            continue
        if not all((tid, p.role, 0) in tok for p in persps):
            continue                                          # Test A 에 없는 질의는 제외

        # 라운드별 branch 산출을 미리 채점해 둔다 (전부 캐시)
        ev: dict[tuple[str, int], tuple] = {}
        ok = True
        for p in persps:
            for r_ in range(MAX_ROUNDS):
                if (tid, p.role, r_) not in tok:
                    ok = False
                    break
                o = run_branch(p, retriever, llm, cfg, round_=r_, search_anchor=anchor)
                ev[(p.role, r_)] = evaluate(o, sc)
            if not ok:
                break
        if not ok:
            continue

        roles = [p.role for p in persps]

        def verify(state: dict[str, int]) -> bool:
            g = st.mean([ev[(b, state[b])][1] for b in roles])
            c = min(ev[(b, state[b])][0] for b in roles)
            return g >= TAU_G_GLOBAL and c >= TAU_C_GLOBAL

        # --- A. GlobalRetry: verify 실패 시 전 branch 재실행 --------------
        a_calls = a_tok = a_sub = 0
        a_state = {b: 0 for b in roles}
        a_pass = False
        for r_ in range(MAX_ROUNDS):
            for b in roles:
                a_state[b] = r_
                a_calls += 1
                a_tok += tok[(tid, b, r_)]
                a_sub += ev[(b, r_)][2]
            if verify(a_state):
                a_pass = True
                break

        # --- B. 축소 컨트롤러: FREEZE + EXPAND ----------------------------
        b_calls = b_tok = b_sub = 0
        b_state = {b: 0 for b in roles}
        frozen: dict[str, int] = {}
        b_pass = False
        for r_ in range(MAX_ROUNDS):
            active = [b for b in roles if b not in frozen]
            if r_ > 0 and not active:
                break
            for b in (roles if r_ == 0 else active):
                b_state[b] = r_
                b_calls += 1
                b_tok += tok[(tid, b, r_)]
                b_sub += ev[(b, r_)][2]
            # FREEZE — sufficient@tau 인 branch 를 얼린다 (엄격 부등호)
            for b in roles:
                if b in frozen:
                    continue
                cov, grd, _, _ = ev[(b, b_state[b])]
                if cov == 1.0 and grd > TAU:
                    frozen[b] = b_state[b]
            if verify(b_state):
                b_pass = True
                break

        fin_a = {b: ev[(b, a_state[b])] for b in roles}
        fin_b = {b: ev[(b, frozen.get(b, b_state[b]))] for b in roles}
        per_q.append({
            "query_id": tid, "n_branches": len(roles),
            "A_calls": a_calls, "A_tokens": a_tok, "A_subq": a_sub, "A_verify": int(a_pass),
            "A_cov": st.mean([v[0] for v in fin_a.values()]),
            "A_grd": st.mean([v[1] for v in fin_a.values()]),
            "A_uncited": st.mean([v[3] for v in fin_a.values()]),
            "B_calls": b_calls, "B_tokens": b_tok, "B_subq": b_sub, "B_verify": int(b_pass),
            "B_cov": st.mean([v[0] for v in fin_b.values()]),
            "B_grd": st.mean([v[1] for v in fin_b.values()]),
            "B_uncited": st.mean([v[3] for v in fin_b.values()]),
            "B_frozen": len(frozen),
        })
        print("  [%2d] %s  A %d단위 / B %d단위 · 얼림 %d"
              % (len(per_q), tid, a_calls, b_calls, len(frozen)), flush=True)

    if not per_q:
        print("비교할 질의가 없다.")
        return 1

    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    with io.open(outdir / "m2min_live.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_q[0].keys()))
        w.writeheader(); w.writerows(per_q)

    def agg(k):
        return sum(r[k] for r in per_q)

    def mean(k):
        return st.mean([r[k] for r in per_q])

    dc = 100.0 * (agg("B_calls") - agg("A_calls")) / agg("A_calls")
    dt = 100.0 * (agg("B_tokens") - agg("A_tokens")) / agg("A_tokens")
    ds = 100.0 * (agg("B_subq") - agg("A_subq")) / agg("A_subq")
    dcov = [r["B_cov"] - r["A_cov"] for r in per_q]
    dgrd = [r["B_grd"] - r["A_grd"] for r in per_q]
    ci_cov, ci_grd = boot_ci(dcov), boot_ci(dgrd)

    print("")
    print("=== 표 16 — 라이브 비교 (질의 %d, 시드 1회, 실행 단위 = branch) ===" % len(per_q))
    print("| 항목 | Global retry (R=3) | 축소 컨트롤러 | 차이 |")
    print("|---|---|---|---|")
    print("| 생성 호출 (branch 실행) | %d | %d | **%+.2f%%** |"
          % (agg("A_calls"), agg("B_calls"), dc))
    print("| 토큰 | %d | %d | **%+.2f%%** |" % (agg("A_tokens"), agg("B_tokens"), dt))
    print("| 재실행 sub-question | %d | %d | %+.2f%% |" % (agg("A_subq"), agg("B_subq"), ds))
    print("| coverage | %.4f | %.4f | %+.4f  95%%CI [%+.4f, %+.4f] |"
          % (mean("A_cov"), mean("B_cov"), st.mean(dcov), ci_cov[0], ci_cov[1]))
    print("| grounding | %.4f | %.4f | %+.4f  95%%CI [%+.4f, %+.4f] |"
          % (mean("A_grd"), mean("B_grd"), st.mean(dgrd), ci_grd[0], ci_grd[1]))
    print("| 미인용 비율 | %.4f | %.4f | %+.4f |"
          % (mean("A_uncited"), mean("B_uncited"), mean("B_uncited") - mean("A_uncited")))
    print("| verify 통과 | %d/%d | %d/%d | |"
          % (agg("A_verify"), len(per_q), agg("B_verify"), len(per_q)))
    print("| 얼린 branch | 0 | %d | |" % agg("B_frozen"))

    same_cov = ci_cov[0] <= 0 <= ci_cov[1]
    same_grd = ci_grd[0] <= 0 <= ci_grd[1]
    io.open(outdir / "m2min_live.json", "w", encoding="utf-8").write(json.dumps({
        "provisional": True, "n_queries": len(per_q), "unit": "branch",
        "note": ("실행 단위가 branch 다. 스펙의 최소 변경 단위인 sub-question 이 아니다 — "
                 "그 단위 artifact 는 캐시에 없다. 따라서 이 비교는 절감을 과소평가한다."),
        "calls": {"A": agg("A_calls"), "B": agg("B_calls"), "delta_pct": round(dc, 4)},
        "tokens": {"A": agg("A_tokens"), "B": agg("B_tokens"), "delta_pct": round(dt, 4)},
        "subq": {"A": agg("A_subq"), "B": agg("B_subq"), "delta_pct": round(ds, 4)},
        "coverage": {"A": round(mean("A_cov"), 6), "B": round(mean("B_cov"), 6),
                     "delta": round(st.mean(dcov), 6),
                     "ci": [round(ci_cov[0], 6), round(ci_cov[1], 6)],
                     "within_ci": bool(same_cov)},
        "grounding": {"A": round(mean("A_grd"), 6), "B": round(mean("B_grd"), 6),
                      "delta": round(st.mean(dgrd), 6),
                      "ci": [round(ci_grd[0], 6), round(ci_grd[1], 6)],
                      "within_ci": bool(same_grd)},
        "verify_pass": {"A": agg("A_verify"), "B": agg("B_verify")},
        "frozen_branches": agg("B_frozen"),
        "minicheck_calls": sc.calls, "chunk_split_violations": sc.violations,
    }, ensure_ascii=False, indent=1))

    print("")
    print("결과 문장 판정 (스펙 고정 문안):")
    # 구간 밖이면 **방향을 구분해야 한다.** 나쁜 쪽으로 밖인 것과 좋은 쪽으로
    # 밖인 것은 전혀 다른 결과다. 앞 판에서 이걸 뭉뚱그려 오판했다.
    cov_dir = "better" if ci_cov[0] > 0 else ("worse" if ci_cov[1] < 0 else "same")
    grd_dir = "better" if ci_grd[0] > 0 else ("worse" if ci_grd[1] < 0 else "same")
    print("  coverage %s · grounding %s · calls %+.2f%%" % (cov_dir, grd_dir, dc))
    if dc >= 0:
        print("  -> \"반사실 한계 이상으로 호출을 줄이지 못한다\"")
    elif "worse" in (cov_dir, grd_dir):
        print("  -> 호출은 줄지만 품질이 유의하게 낮다. 절감 주장 불가.")
    elif cov_dir == "same" and grd_dir == "same":
        print("  -> \"호출 %.1f%%, 토큰 %.1f%% 감소, coverage·grounding 은 신뢰구간 안\""
              % (-dc, -dt))
    else:
        print("  -> \"호출 %.1f%%, 토큰 %.1f%% 감소이며 coverage·grounding 은 오히려 높다\""
              % (-dc, -dt))
        print("     (전역 재실행이 라운드 0 의 좋은 branch 를 나쁜 재시도로 덮어쓰기 때문.")
        print("      D11 에서 라운드 0->2 coverage 가 -0.060 으로 떨어진 것과 같은 현상이다.)")
    print("")
    print("MiniCheck forward %d · 분할 위반 %d" % (sc.calls, sc.violations))
    print("LLM 호출 %d (캐시 %d), 청구 %d/%d"
          % (llm.stats.calls, llm.stats.cache_hits,
             llm.stats.billed_tokens_in, llm.stats.billed_tokens_out))
    print("-> " + str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
