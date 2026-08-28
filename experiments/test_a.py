"""SPEC §4 — Test A. GlobalRetry 가 '이미 충분했던' branch 를 얼마나 낭비 재실행하는가.

낭비율 = (재실행된 branch 중 sufficient 였던 수) / (재실행된 branch 총수)

sufficient(b) := grounding(b) > tau AND coverage(b) > tau  (SPEC §4.3, 엄격 부등호)

**평가 시점이 중요하다.** 라운드 r 에서 verify 가 실패해 r+1 로 재실행할 때, 그 재실행이
낭비였는지는 **라운드 r 의 evidence** 로 판정한다. 재실행 후의 값이 아니다.

사후 라벨링은 결정론이므로 tau 마다 점추정 1회만 계산한다 (SPEC 변경 #4).
불확실성은 **쿼리 단위 bootstrap** 으로 구한다 — LLM 재호출 0회.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lfm.config import load_config  # noqa: E402
from lfm.controller import GlobalRetry  # noqa: E402
from lfm.evidence import sufficient  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import MockNLI, load_nli  # noqa: E402
from lfm.pipeline import QueryResult, run_query  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.trace import TraceWriter, new_run_id  # noqa: E402


# --- 집계 ----------------------------------------------------------------


def branch_rows(results: list[QueryResult], taus: list[float]) -> list[dict[str, Any]]:
    """(query, round, branch) 한 줄씩. re_executed 는 그 라운드 뒤에 재실행됐는지."""
    rows: list[dict[str, Any]] = []
    for qr in results:
        for r in qr.rounds:
            re_exec = int(r.action.type == "RETRY_ALL")
            for bid, ev in sorted(r.evidence.items()):
                out = r.outputs.get(bid)
                row = {
                    "query_id": qr.query_id,
                    "round": r.round,
                    "branch": bid,
                    "coverage": round(ev.coverage, 6),
                    "grounding": round(ev.grounding, 6),
                    "source_diversity": round(ev.source_diversity, 6),
                    "support_strength": round(ev.support_strength, 6),
                    "re_executed": re_exec,
                    "tokens": (out.tokens_in + out.tokens_out) if out else 0,
                    "degraded": int(bool(out and out.degraded)),
                    "failed_round_0": int(qr.failed_round_0),
                }
                for t in taus:
                    row["sufficient@%s" % t] = int(sufficient(ev, t))
                rows.append(row)
    return rows


def waste_by_query(rows: list[dict[str, Any]], tau: float) -> dict[str, tuple[int, int]]:
    """query_id -> (낭비 재실행 수, 재실행 총수). bootstrap 의 표집 단위다."""
    agg: dict[str, list[int]] = {}
    for row in rows:
        if not row["re_executed"]:
            continue
        slot = agg.setdefault(row["query_id"], [0, 0])
        slot[1] += 1
        slot[0] += int(row["sufficient@%s" % tau])
    return {q: (v[0], v[1]) for q, v in agg.items()}


def point_estimate(per_query: dict[str, tuple[int, int]]) -> tuple[float, int, int]:
    num = sum(v[0] for v in per_query.values())
    den = sum(v[1] for v in per_query.values())
    return (num / den if den else 0.0), num, den


def bootstrap_ci(
    per_query: dict[str, tuple[int, int]], n: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    """쿼리 단위 복원추출. 행 단위로 뽑으면 branch 상관을 무시해 CI 가 좁아진다."""
    qids = sorted(per_query)
    if not qids:
        return (float("nan"), float("nan"))
    nums = np.array([per_query[q][0] for q in qids], dtype=float)
    dens = np.array([per_query[q][1] for q in qids], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(qids), size=(n, len(qids)))
    num_s = nums[draws].sum(axis=1)
    den_s = dens[draws].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = np.where(den_s > 0, num_s / den_s, np.nan)
    lo, hi = np.nanpercentile(ratios, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def verdict(tau_waste: float) -> str:
    """마스터플랜 §6 go/no-go 밴드. 기준 자체는 논문에 노출하지 않는다."""
    pct = tau_waste * 100
    if pct >= 60:
        return "60% 이상 — 전제 강하게 지지"
    if pct >= 40:
        return "40~60% — 지지하되 절감 기대치 하향"
    if pct >= 20:
        return "20~40% — 더 어려운 토픽셋으로 재측정 필요"
    return "20% 미만 — 전제 오류. 연구 질문 재설계"


# --- 산출 ----------------------------------------------------------------


def write_csvs(outdir: Path, rows: list[dict[str, Any]], ci_rows: list[dict[str, Any]]) -> None:
    if rows:
        with io.open(outdir / "waste_ratio.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with io.open(outdir / "waste_ratio_ci.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["tau", "point", "ci_low", "ci_high", "n_failed_queries",
                        "n_reexec_branches", "n_wasted_branches"],
        )
        w.writeheader()
        w.writerows(ci_rows)


def write_chart(outdir: Path, ci_rows: list[dict[str, Any]], tau_global: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = [r["tau"] for r in ci_rows]
    pts = [r["point"] * 100 for r in ci_rows]
    lo = [max(0.0, r["point"] - r["ci_low"]) * 100 for r in ci_rows]
    hi = [max(0.0, r["ci_high"] - r["point"]) * 100 for r in ci_rows]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    ax.bar([str(t) for t in taus], pts, yerr=[lo, hi], capsize=6,
           color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax.set_xlabel(r"sufficiency threshold $\tau$")
    ax.set_ylabel("wasted re-execution (%)")
    ax.set_title(r"Test A: wasted branch re-execution under GlobalRetry"
                 "\n" r"($\tau_{global}=%.2f$, 95%% CI, query-level bootstrap)" % tau_global)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(pts):
        ax.text(i, min(98, v + 3), "%.1f%%" % v, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "waste_ratio.png")
    plt.close(fig)


def write_report(outdir: Path, ctx: dict[str, Any]) -> None:
    ci_rows = ctx["ci_rows"]
    tg = ctx["tau_g_global"]
    primary = ctx["primary_tau"]
    prow = next(r for r in ci_rows if abs(r["tau"] - primary) < 1e-9)

    lines: list[str] = []
    a = lines.append
    a("# Test A — GlobalRetry 낭비 재실행 측정")
    a("")
    a("생성: " + ctx["generated_at"] + " · run_id `" + ctx["run_id"] + "`")
    a("")
    a("## 판정")
    a("")
    a("**τ=%.2f 기준 낭비율 %.1f%% [95%% CI %.1f–%.1f] → %s**"
      % (primary, prow["point"] * 100, prow["ci_low"] * 100, prow["ci_high"] * 100,
         verdict(prow["point"])))
    a("")
    a("낭비율 = (재실행된 branch 중 재실행 시점에 이미 sufficient 였던 수) / (재실행된 branch 총수).")
    a("`sufficient(b) := grounding(b) > τ AND coverage(b) > τ` — 엄격 부등호 (SPEC §4.3).")
    a("")
    a("## 실행 규모")
    a("")
    a("| 항목 | 값 |")
    a("|---|---|")
    a("| 토픽(질의) 수 | %d |" % ctx["n_queries"])
    a("| 1라운드 verify 불통과 (실패 케이스) | %d |" % ctx["n_failed"])
    a("| 최종 통과 | %d |" % ctx["n_passed"])
    a("| 총 라운드 수 | %d |" % ctx["n_rounds_total"])
    a("| 재실행된 branch 총수 | %d |" % ctx["n_reexec"])
    a("| degraded branch (절단·파싱 실패) | %d |" % ctx["n_degraded"])
    a("| 모델 | `%s` (reasoning_effort=%s) |" % (ctx["model"], ctx["reasoning_effort"]))
    a("| NLI | `%s` (%s) |" % (ctx["nli_model"], ctx["nli_device"]))
    a("| 소요 | %.1f 분 |" % (ctx["elapsed_s"] / 60))
    a("")
    a("## τ 스윕 — 전역 검증 임계와 병기")
    a("")
    a("낭비율은 전역 검증 임계 τ_global 과 사후 라벨링 임계 τ 의 **상대 위치**에 직접")
    a("좌우된다. 두 값을 같은 표에 둔다 (SPEC §4.5).")
    a("")
    a("| τ | τ_global | 낭비율 | 95% CI | 낭비 branch | 재실행 branch | 실패 쿼리 |")
    a("|---|---|---|---|---|---|---|")
    for r in ci_rows:
        mark = " **(primary)**" if abs(r["tau"] - primary) < 1e-9 else ""
        a("| %.2f%s | %.2f | %.1f%% | %.1f–%.1f%% | %d | %d | %d |"
          % (r["tau"], mark, tg, r["point"] * 100, r["ci_low"] * 100, r["ci_high"] * 100,
             r["n_wasted_branches"], r["n_reexec_branches"], r["n_failed_queries"]))
    a("")
    a("CI 는 실패 쿼리 집합에서 **쿼리 단위** 복원추출 %d회 (seed=%d). 행 단위로 뽑으면"
      % (ctx["bootstrap_n"], ctx["seed"]))
    a("한 쿼리의 4개 branch 가 독립인 것처럼 취급돼 CI 가 좁아진다. LLM 재호출은 0회다.")
    a("")
    a("## 검증 규칙 (SPEC 변경 #2)")
    a("")
    a("```")
    a("passed  <=>  mean_b grounding(b) >= %.2f" % tg)
    a("        AND  min_b  coverage(b)  >= %.2f" % ctx["tau_c_global"])
    a("```")
    a("")
    a("grounding 은 branch 평균, coverage 는 branch 최솟값이다. 이 비대칭은 의도된 것이다.")
    a("")
    a("## 라운드별 토큰")
    a("")
    a("| 라운드 | 쿼리 수 | LLM 호출 | 입력 토큰 | 출력 토큰 |")
    a("|---|---|---|---|---|")
    for r, v in sorted(ctx["by_round"].items()):
        a("| %d | %d | %d | %d | %d |"
          % (r, v["queries"], v["llm_calls"], v["tokens_in"], v["tokens_out"]))
    a("")
    a("## 비용")
    a("")
    a("| 항목 | 값 |")
    a("|---|---|")
    a("| LLM 호출 | %d (캐시 히트 %d) |" % (ctx["llm_calls"], ctx["cache_hits"]))
    a("| 입력 토큰 (총/청구) | %d / %d |" % (ctx["tokens_in"], ctx["billed_in"]))
    a("| 출력 토큰 (총/청구) | %d / %d |" % (ctx["tokens_out"], ctx["billed_out"]))
    a("")
    a("출력 토큰에는 `gpt-5-mini` 의 reasoning 토큰이 포함된다 (MODEL_NOTES.md 참고).")
    a("")
    a("## 산출물")
    a("")
    a("- `waste_ratio.csv` — (query, round, branch) 단위 원자료")
    a("- `waste_ratio_ci.csv` — τ별 점추정과 CI")
    a("- `waste_ratio.png` — τ별 낭비율 막대 (CI 에러바)")
    a("- `trace.jsonl` — 전 라운드 trace (동결 스키마)")
    a("- `config.yaml` — 이 실행에 쓰인 설정 사본")
    a("- `results.json` — 쿼리별 라운드 상세")
    a("")

    with io.open(outdir / "report.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))


# --- main ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Test A (SPEC §4)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0, help="상위 N개 토픽만. 비용 시험용")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--mock", action="store_true", help="mock NLI (스모크용)")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(str(repo_root / ".env"))
    cfg = load_config(args.config or (repo_root / "config.yaml"))
    data_dir = Path(cfg["paths.data_dir"])

    with io.open(data_dir / "topics_pilot50.yaml", encoding="utf-8") as fh:
        tdoc = yaml.safe_load(fh)
    if tdoc.get("approved") is not True:
        print("topics_pilot50.yaml 에 approved: true 가 없다 (SPEC §2.4).")
        return 1
    topics = tdoc["topics"]
    if args.limit:
        topics = topics[: args.limit]

    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir or (Path(cfg["paths.results_dir"]) / "test_a" / stamp))
    outdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config or (repo_root / "config.yaml"), outdir / "config.yaml")

    llm = LLM(cfg)
    nli = MockNLI() if args.mock else load_nli(cfg)
    retriever = BM25Retriever(cfg)
    controller = GlobalRetry(cfg)
    run_id = new_run_id()

    print("토픽 %d개 · 모델 %s · NLI %s"
          % (len(topics), llm.model, "mock" if args.mock else cfg["nli.model"]))
    print("산출 -> " + str(outdir))

    t0 = time.time()
    results: list[QueryResult] = []
    with TraceWriter(outdir / "trace.jsonl") as tw:
        for i, t in enumerate(topics):
            qt = time.time()
            qr = run_query(
                query_id=str(t["id"]),
                query=t["topic"],
                retriever=retriever,
                llm=llm,
                nli=nli,
                cfg=cfg,
                controller=controller,
                run_id=run_id,
                trace_writer=tw,
            )
            results.append(qr)
            print("[%3d/%3d] %-6s rounds=%d passed=%-5s  %.1fs  %s"
                  % (i + 1, len(topics), qr.query_id, len(qr.rounds), qr.passed,
                     time.time() - qt, t["topic"][:52]), flush=True)
    elapsed = time.time() - t0

    # --- 집계 --------------------------------------------------------
    taus = [float(x) for x in cfg["sweep.tau"]]
    rows = branch_rows(results, taus)
    ci_rows = []
    for tau in taus:
        per_q = waste_by_query(rows, tau)
        point, num, den = point_estimate(per_q)
        lo, hi = bootstrap_ci(per_q, int(cfg["sweep.bootstrap_n"]), int(cfg["seed"]))
        ci_rows.append(
            {
                "tau": tau,
                "point": round(point, 6),
                "ci_low": round(lo, 6),
                "ci_high": round(hi, 6),
                "n_failed_queries": len(per_q),
                "n_reexec_branches": den,
                "n_wasted_branches": num,
            }
        )

    by_round: dict[int, dict[str, int]] = {}
    for qr in results:
        for r in qr.rounds:
            slot = by_round.setdefault(r.round, {"queries": 0, "llm_calls": 0,
                                                 "tokens_in": 0, "tokens_out": 0})
            slot["queries"] += 1
            slot["llm_calls"] += r.cost["llm_calls"]
            slot["tokens_in"] += r.cost["tokens_in"]
            slot["tokens_out"] += r.cost["tokens_out"]

    ctx = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": run_id,
        "n_queries": len(results),
        "n_failed": sum(1 for q in results if q.failed_round_0),
        "n_passed": sum(1 for q in results if q.passed),
        "n_rounds_total": sum(len(q.rounds) for q in results),
        "n_reexec": sum(r["re_executed"] for r in rows),
        "n_degraded": sum(r["degraded"] for r in rows),
        "ci_rows": ci_rows,
        "tau_g_global": float(cfg["verify.tau_g_global"]),
        "tau_c_global": float(cfg["verify.tau_c_global"]),
        "primary_tau": float(cfg["sweep.primary_tau"]),
        "bootstrap_n": int(cfg["sweep.bootstrap_n"]),
        "seed": int(cfg["seed"]),
        "model": llm.model,
        "reasoning_effort": llm.reasoning_effort,
        "nli_model": "mock" if args.mock else cfg["nli.model"],
        "nli_device": getattr(nli, "device", "mock"),
        "elapsed_s": elapsed,
        "by_round": by_round,
        "llm_calls": llm.stats.calls,
        "cache_hits": llm.stats.cache_hits,
        "tokens_in": llm.stats.tokens_in,
        "tokens_out": llm.stats.tokens_out,
        "billed_in": llm.stats.billed_tokens_in,
        "billed_out": llm.stats.billed_tokens_out,
    }

    write_csvs(outdir, rows, ci_rows)
    write_chart(outdir, ci_rows, ctx["tau_g_global"])
    write_report(outdir, ctx)
    with io.open(outdir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"context": {k: v for k, v in ctx.items() if k != "ci_rows"},
             "ci": ci_rows,
             "queries": [q.to_dict() for q in results]},
            fh, ensure_ascii=False, indent=2,
        )
    with io.open(outdir / "usage.json", "w", encoding="utf-8") as fh:
        json.dump(llm.stats.to_dict(), fh, ensure_ascii=False, indent=2)

    primary = next(r for r in ci_rows if abs(r["tau"] - ctx["primary_tau"]) < 1e-9)
    print()
    print("실패 케이스 %d/%d, 재실행 branch %d" % (ctx["n_failed"], ctx["n_queries"], ctx["n_reexec"]))
    for r in ci_rows:
        print("  tau=%.2f  낭비율 %.1f%%  [%.1f–%.1f]  (%d/%d)"
              % (r["tau"], r["point"] * 100, r["ci_low"] * 100, r["ci_high"] * 100,
                 r["n_wasted_branches"], r["n_reexec_branches"]))
    print()
    print("판정 (tau=%.2f): %s" % (ctx["primary_tau"], verdict(primary["point"])))
    print("LLM 호출 %d (캐시 %d), 청구 토큰 in/out = %d/%d"
          % (llm.stats.calls, llm.stats.cache_hits,
             llm.stats.billed_tokens_in, llm.stats.billed_tokens_out))
    print("소요 %.1f분 -> %s" % (elapsed / 60, outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
