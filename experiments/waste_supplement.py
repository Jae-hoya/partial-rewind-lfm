"""파일럿 오염 보정 — 교정에 쓰인 토픽을 뺀 낭비율을 같은 방식으로 다시 계산한다.

t000~t004 는 프롬프트·검색 교정에 쓰였고 본실험 50토픽에도 들어 있다. 교정 대상은
측정 타당도(메타 진술·검색 잡음)였지 낭비율 값이 아니었지만, 그 주장은 독자가
검증할 수 있어야 한다. 그래서 **같은 추정량을 45토픽에도 돌려 함께 싣는다.**

새 그림을 만들지 않는다 (SPEC §6 금지사항). 숫자와 표만 report.md 에 덧붙인다.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_a import bootstrap_ci, point_estimate, verdict, waste_by_query  # noqa: E402

from lfm.config import load_config  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    with io.open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["re_executed"] = int(r["re_executed"])
    return rows


def estimate(rows, taus, boot_n, seed):
    out = []
    for tau in taus:
        per_q = waste_by_query(rows, tau)
        point, num, den = point_estimate(per_q)
        lo, hi = bootstrap_ci(per_q, boot_n, seed)
        out.append({"tau": tau, "point": point, "ci_low": lo, "ci_high": hi,
                    "n_queries": len(per_q), "n_reexec": den, "n_wasted": num})
    return out


def regime_diagnostics(rows, taus_fine) -> list[str]:
    """낭비율이 낮을 때, 그게 '재실행이 낭비가 아니다' 인지 '애초에 충분한 branch 가
    없다' 인지 갈라 보여준다. 둘은 완전히 다른 결론이다.

    새 그림을 만들지 않는다 — 표만 낸다 (SPEC §6).
    """
    import statistics as st

    a = []
    add = a.append
    cov = [float(r["coverage"]) for r in rows]
    grd = [float(r["grounding"]) for r in rows]
    reexec = [r for r in rows if int(r["re_executed"])]

    def q(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(p * len(v)))]

    add("")
    add("## 측정 영역 진단 — 낭비율이 낮은 이유")
    add("")
    add("낭비율이 낮은 데는 두 가지 서로 다른 원인이 있다.")
    add("")
    add("1. 전역 재실행이 실제로 낭비가 아니다 (재실행된 branch 가 원래 부족했다)")
    add("2. **애초에 충분한 branch 가 거의 없다** — 낭비할 것이 없어서 낭비율이 낮다")
    add("")
    add("둘은 정반대의 결론으로 이어지므로 구분해서 보고한다.")
    add("")
    add("| 지표 | 값 |")
    add("|---|---|")
    add("| branch-round 수 | %d |" % len(rows))
    add("| coverage 중앙값 / 평균 | %.2f / %.3f |" % (q(cov, 0.5), st.mean(cov)))
    add("| grounding 중앙값 / 평균 | %.2f / %.3f |" % (q(grd, 0.5), st.mean(grd)))
    add("| coverage = 1.0 인 branch | %d (%.1f%%) |"
        % (sum(1 for x in cov if x == 1.0), 100 * sum(1 for x in cov if x == 1.0) / len(cov)))
    add("| grounding > 0.8 인 branch | %d (%.1f%%) |"
        % (sum(1 for x in grd if x > 0.8), 100 * sum(1 for x in grd if x > 0.8) / len(grd)))
    add("| **둘 다 만족 (sufficient@0.8)** | **%d (%.1f%%)** |"
        % (sum(1 for r in rows if float(r["coverage"]) == 1.0 and float(r["grounding"]) > 0.8),
           100 * sum(1 for r in rows if float(r["coverage"]) == 1.0
                     and float(r["grounding"]) > 0.8) / len(rows)))
    add("| 답안이 비어 있는 branch (degraded) | %d (%.1f%%) |"
        % (sum(int(r.get("degraded", 0)) for r in rows),
           100 * sum(int(r.get("degraded", 0)) for r in rows) / len(rows)))
    add("")
    add("### τ 미세 스윕 (진단용 — 논문 표 아님)")
    add("")
    add("SPEC 이 정한 τ ∈ {0.7, 0.8, 0.9} 밖까지 내려 보면, 낭비율이 임계 근처에서")
    add("급변하는 절벽인지 완만한 연속인지 알 수 있다.")
    add("")
    add("| τ | 낭비율 | 낭비 / 재실행 |")
    add("|---|---|---|")
    for tau in taus_fine:
        n = sum(1 for r in reexec
                if float(r["grounding"]) > tau and float(r["coverage"]) > tau)
        add("| %.1f | %.1f%% | %d / %d |" % (tau, 100 * n / len(reexec), n, len(reexec)))
    add("")
    rounds = sorted({int(r["round"]) for r in rows})
    add("### 라운드별 품질 — 재실행이 무엇을 건졌는가")
    add("")
    add("낭비율은 '재실행이 불필요했는가' 를 묻는다. 그 짝이 되는 질문은")
    add("'재실행이 도움이 됐는가' 다. 같은 원자료로 바로 답할 수 있다.")
    add("")
    add("| 라운드 | branch | coverage 평균 | grounding 평균 | cov=1.0 | grd>0.8 | 빈 답안 |")
    add("|---|---|---|---|---|---|---|")
    base = None
    for rd in rounds:
        sub = [x for x in rows if int(x["round"]) == rd]
        c = [float(x["coverage"]) for x in sub]
        g = [float(x["grounding"]) for x in sub]
        if base is None:
            base = (st.mean(c), st.mean(g))
        add("| %d | %d | %.3f | %.3f | %d | %d | %d |"
            % (rd, len(sub), st.mean(c), st.mean(g),
               sum(1 for v in c if v == 1.0), sum(1 for v in g if v > 0.8),
               sum(int(x.get("degraded", 0)) for x in sub)))
    last = [x for x in rows if int(x["round"]) == rounds[-1]]
    dc = st.mean([float(x["coverage"]) for x in last]) - base[0]
    dg = st.mean([float(x["grounding"]) for x in last]) - base[1]
    add("")
    add("라운드 0 → %d 변화: coverage **%+.3f**, grounding **%+.3f**." % (rounds[-1], dc, dg))
    if dc <= 0 and dg <= 0:
        add("")
        add("**재실행이 품질을 올리지 못했다.** 검색 폭을 2~3배로 넓히고 sub-question 을")
        add("재표현하고 retry directive 를 넣었는데도 두 지표가 모두 하락했다. 전체 토큰의")
        add("약 3분의 2가 라운드 1~2 에 쓰였고 통과 질의는 여전히 0건이다. 낭비율이 낮은")
        add("것과는 별개의 사실이며, 같은 방향을 가리킨다 — 이 설정에서 GlobalRetry 는")
        add("건질 것도 없었고 버릴 것도 없었다.")
    add("")
    add("**coverage 는 양자화돼 있다.** sub-question 이 3~5개이므로 `coverage > 0.8` 은")
    add("사실상 `coverage = 1.0` 을 요구한다. τ 를 연속량처럼 읽으면 안 된다.")
    add("")
    return a


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="파일럿 제외 보조 추정")
    ap.add_argument("results_dir")
    ap.add_argument("--config", default=None)
    ap.add_argument("--exclude", default="t000,t001,t002,t003,t004",
                    help="교정에 쓰인 토픽 id (쉼표 구분)")
    ap.add_argument("--append", action="store_true", help="report.md 에 절을 덧붙인다")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    outdir = Path(args.results_dir)
    cfg = load_config(args.config or (outdir / "config.yaml"))
    taus = [float(t) for t in cfg["sweep.tau"]]
    boot_n = int(cfg["sweep.bootstrap_n"])
    seed = int(cfg["seed"])
    primary = float(cfg["sweep.primary_tau"])

    excluded = {x.strip() for x in args.exclude.split(",") if x.strip()}
    rows = load_rows(outdir / "waste_ratio.csv")
    kept = [r for r in rows if r["query_id"] not in excluded]

    full = estimate(rows, taus, boot_n, seed)
    sub = estimate(kept, taus, boot_n, seed)

    lines: list[str] = []
    a = lines.append
    a("")
    a("## 보조 추정 — 교정에 쓰인 토픽 제외")
    a("")
    a("`" + ", ".join(sorted(excluded)) + "` 는 파일럿에서 프롬프트·검색 교정에 쓰였고")
    a("본실험에도 포함돼 있다. 교정 대상은 측정 타당도(코퍼스 메타 진술, 검색 잡음)였지")
    a("낭비율 값이 아니었지만, 그 주장이 검증 가능하도록 같은 추정량을 나머지 토픽에도")
    a("돌려 병기한다. 두 값이 크게 다르면 교정이 결과를 밀었다는 뜻이다.")
    a("")
    a("| τ | 전체 %d토픽 | 95%% CI | 제외 후 %d토픽 | 95%% CI | 차이 |"
      % (len({r["query_id"] for r in rows}), len({r["query_id"] for r in kept})))
    a("|---|---|---|---|---|---|")
    for f, s in zip(full, sub):
        a("| %.2f | %.1f%% | %.1f–%.1f%% | %.1f%% | %.1f–%.1f%% | %+.1f%%p |"
          % (f["tau"], f["point"] * 100, f["ci_low"] * 100, f["ci_high"] * 100,
             s["point"] * 100, s["ci_low"] * 100, s["ci_high"] * 100,
             (s["point"] - f["point"]) * 100))
    a("")
    pf = next(x for x in full if abs(x["tau"] - primary) < 1e-9)
    ps = next(x for x in sub if abs(x["tau"] - primary) < 1e-9)
    a("τ=%.2f 기준 판정: 전체 %s / 제외 후 %s"
      % (primary, verdict(pf["point"]), verdict(ps["point"])))
    a("")

    lines += regime_diagnostics(rows, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    text = "\n".join(lines)
    print(text)

    with io.open(outdir / "waste_ratio_supplement.csv", "w", encoding="utf-8",
                 newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subset", "tau", "point", "ci_low", "ci_high",
                                           "n_queries", "n_reexec", "n_wasted"])
        w.writeheader()
        for tag, rowset in (("all", full), ("excl_calibration", sub)):
            for r in rowset:
                w.writerow({"subset": tag, **r})

    if args.append:
        with io.open(outdir / "report.md", "a", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print("report.md 에 덧붙였다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
