"""M2-min — 관측 궤적 위에서의 절감. **호출 0회.** 원고 §X 기여 4의 숫자.

기여 4 의 빈칸은 이것이다:

> M2-min: calls −x% at equal coverage and grounding, or the finding that no
> headroom exists in this regime and why

Test A 는 verify 가 한 번도 통과하지 않아 **50질의 전부가 매 라운드 4/4 branch 를
재실행**했다. Partial Rewind 는 라운드 0 에서 충분한 branch 를 FREEZE 하므로 그
branch 는 라운드 1·2 에서 다시 돌지 않는다.

**관측된 궤적을 고정한 채** 그 차이를 센다. 무엇을 아꼈는지는 두 단위로 본다:

  branch-round  실행 단위 수 (예산이 생성 호출 수로 정의되므로 이쪽이 스펙 단위)
  청구 토큰     실제 비용

**이건 반사실이지 새 실행이 아니다.** FREEZE 는 verify 결과를 바꾸고 verify 는
이후 라운드를 바꾸므로, 진짜 Partial Rewind 는 다른 궤적을 그린다. 여기 값은
**같은 궤적에서 무엇을 안 돌려도 됐는가**이고, 그래서 M1′(신규 실행)의 추정치가
아니라 **하한**으로 읽어야 한다. 원고에도 그렇게 적어야 한다.

품질은 정의상 유지된다. 얼린 branch 는 라운드 0 의 artifact 를 그대로 쓰고,
그 artifact 는 이미 `sufficient@τ` 를 만족한 것이다. 즉 "동일 coverage·grounding
에서 호출 −x%" 라는 문장이 성립한다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAUS = (0.7, 0.8, 0.9)


def main(argv=None):
    ap = argparse.ArgumentParser(description="M2-min 절감 (호출 0회)")
    ap.add_argument("--testa", default="results/test_a/20260829-041445/waste_ratio.csv")
    ap.add_argument("--m1c", default="results/m1c/m1c_branches.csv")
    ap.add_argument("--out", default="results/m1c/m2min_savings.json")
    args = ap.parse_args(argv)

    ta = list(csv.DictReader(io.open(ROOT / args.testa, encoding="utf-8")))
    mc = list(csv.DictReader(io.open(ROOT / args.m1c, encoding="utf-8")))

    # 재채점은 캐시에 있는 질의만 덮는다. **공통 질의로 분모를 맞춘다.**
    common = {r["query_id"] for r in mc} & {r["query_id"] for r in ta}
    ta = [r for r in ta if r["query_id"] in common]
    mc = [r for r in mc if r["query_id"] in common]
    rounds = sorted({int(r["round"]) for r in ta})
    print("공통 질의 %d · Test A branch-round %d (라운드 %s) · 재채점 branch %d"
          % (len(common), len(ta), rounds, len(mc)))

    # branch 키: (query, branch). 라운드 0 의 충분성으로 FREEZE 를 판정한다.
    tok = {}
    for r in ta:
        tok.setdefault((r["query_id"], r["branch"]), {})[int(r["round"])] = int(r["tokens"])
    total_units = len(ta)
    total_tok = sum(int(r["tokens"]) for r in ta)
    later = [r for r in ta if int(r["round"]) > 0]
    print("라운드 1 이후 실행 단위 %d · 토큰 %d"
          % (len(later), sum(int(r["tokens"]) for r in later)))

    out = {"common_queries": len(common), "total_branch_rounds": total_units,
           "total_tokens": total_tok, "rounds": rounds, "counterfactual": True,
           "note": ("관측 궤적 고정. FREEZE 는 verify 를 바꾸므로 진짜 실행은 다른 궤적을 "
                    "그린다. 이 값은 하한이고 M1' 의 추정치가 아니다."),
           "by_tau": {}}

    print("")
    print("=== M2-min — 관측 궤적에서의 절감 (반사실, 하한) ===")
    print("| tau | 얼린 branch | 안 돌려도 됐던 단위 | **호출 절감** | 토큰 절감 |")
    print("|---|---|---|---|---|")
    for tau in TAUS:
        frozen = {(r["query_id"], r["branch"]) for r in mc
                  if float(r["coverage"]) == 1.0 and float(r["grounding"]) > tau}
        saved_rows = [r for r in later if (r["query_id"], r["branch"]) in frozen]
        su, stk = len(saved_rows), sum(int(r["tokens"]) for r in saved_rows)
        pu = 100.0 * su / total_units if total_units else 0.0
        ptk = 100.0 * stk / total_tok if total_tok else 0.0
        out["by_tau"][str(tau)] = {"frozen_branches": len(frozen),
                                   "saved_branch_rounds": su, "saved_tokens": stk,
                                   "pct_branch_rounds": round(pu, 4),
                                   "pct_tokens": round(ptk, 4)}
        print("| %.1f | %d / %d | %d / %d | **−%.2f%%** | −%.2f%% |"
              % (tau, len(frozen), len(mc), su, total_units, pu, ptk))

    # 대조 — 현행 추정자로는 무엇을 아꼈겠는가
    print("")
    print("대조: 현행 g1 (같은 계산, 같은 질의)")
    print("| tau | 얼린 branch | **호출 절감** |")
    print("|---|---|---|")
    r0 = [r for r in ta if int(r["round"]) == 0]
    for tau in TAUS:
        fz = {(r["query_id"], r["branch"]) for r in r0
              if float(r["coverage"]) == 1.0 and float(r["grounding"]) > tau}
        sv = sum(1 for r in later if (r["query_id"], r["branch"]) in fz)
        out["by_tau"][str(tau)]["g1_frozen"] = len(fz)
        out["by_tau"][str(tau)]["g1_pct_branch_rounds"] = round(
            100.0 * sv / total_units if total_units else 0.0, 4)
        print("| %.1f | %d / %d | **−%.2f%%** |"
              % (tau, len(fz), len(r0), 100.0 * sv / total_units))

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    io.open(p, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))

    b = out["by_tau"]["0.8"]
    print("")
    print("원고 §X 기여 4 문장에 들어갈 값 (tau=0.8):")
    print("  검증된 연산자에서 얼릴 수 있는 branch %d/%d = %.2f%%"
          % (b["frozen_branches"], len(mc), 100.0 * b["frozen_branches"] / len(mc)))
    print("  같은 궤적에서 호출 **−%.2f%%**, 토큰 **−%.2f%%**"
          % (b["pct_branch_rounds"], b["pct_tokens"]))
    print("  현행 연산자로는 −%.2f%% (얼릴 branch %d개)"
          % (b["g1_pct_branch_rounds"], b["g1_frozen"]))
    print("  품질은 정의상 유지 — 얼린 branch 는 이미 sufficient@%.1f 인 라운드 0 artifact 를 쓴다"
          % 0.8)
    print("-> " + str(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
