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
