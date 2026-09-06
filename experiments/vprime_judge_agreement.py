"""판정자(nano) 대 사람 일치율. 원고의 [[fill]] 자리. **호출 0.**

D8-B 스펙은 사람 게이트를 25건으로 잡고 **일치율 ≥ 90%** 를 기준으로 뒀다. 그 시트는
안 채워졌지만, V′ 60건 중 일부가 **D8-B 판정 표본과 같은 문장**이다. 문장 텍스트로
맞추면 사람 라벨과 판정자 라벨을 직접 비교할 수 있다 — 새 라벨링 없이.

**이 표본은 D8-B 게이트가 지정한 25건이 아니다.** 우연히 겹친 것이고 층 비율도
다르다. 그러므로 스펙의 게이트를 이것으로 "통과/탈락" 판정하지 않는다.
관측된 일치율을 보고할 뿐이고, 판정은 공저자가 한다.

왜 중요한가: D8-B 의 FNR 72.5% 와 G층 precision 0.913 이 **판정자 라벨 기준**이다.
판정자가 사람과 얼마나 맞는지가 그 두 숫자의 신뢰도를 정한다.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = {"지지": "supported", "불지지": "unsupported", "판단불가": "unclear"}
GATE = 90.0            # D8-B 스펙의 사람 게이트 기준 (참고용으로만 표시)


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * max(0.0, (c - h) / d), 100 * min(1.0, (c + h) / d))


def main() -> int:
    v = list(csv.DictReader(io.open(ROOT / "results/vprime/labels_stage1.csv",
                                    encoding="utf-8-sig")))
    j = list(csv.DictReader(io.open(ROOT / "results/diag/d8/d8b_judge_sample.csv",
                                    encoding="utf-8")))
    jm = {x["sentence"].strip(): (x["judge_label"], x["stratum"]) for x in j}

    rows = []
    for x in v:
        s = x["문장"].strip()
        if s not in jm:
            continue
        jl, stn = jm[s]
        hl = MAP.get(x["human_label"].strip(), "")
        if not hl:
            continue
        rows.append({"item_id": x["item_id"], "stratum": stn,
                     "judge": jl, "human": hl, "agree": int(jl == hl)})

    if not rows:
        print("겹치는 문장이 없다.")
        return 1

    n = len(rows)
    ag = sum(r["agree"] for r in rows)
    lo, hi = wilson(ag, n)

    out = ROOT / "results/vprime"
    with io.open(out / "judge_agreement.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("=" * 74)
    print("판정자(gpt-5.4-nano) 대 사람 — V' 60건과 D8-B 표본이 겹치는 문장")
    print("=" * 74)
    print("  겹친 문장 %d건 · 일치 %d · **%.1f%% [%.1f, %.1f]**"
          % (n, ag, 100.0 * ag / n, lo, hi))
    print("  (참고: D8-B 스펙의 사람 게이트 기준은 >= %.0f%% 였다. 다만 그 게이트가"
          % GATE)
    print("   지정한 25건 표본이 아니라 우연히 겹친 %d건이므로 통과/탈락 판정을" % n)
    print("   여기서 내리지 않는다.)")

    print("")
    print("  불일치 유형:")
    dis = Counter((r["judge"], r["human"]) for r in rows if not r["agree"])
    for (jl, hl), c in dis.most_common():
        print("    판정자 %-12s 사람 %-12s %d건" % (jl, hl, c))
    print("  -> %s"
          % ("한쪽으로 치우치지 않는다. 체계적 관대/엄격이 아니라 잡음이다."
             if abs(dis.get(("unsupported", "supported"), 0)
                    - dis.get(("supported", "unsupported"), 0)) <= 2
             else "한쪽으로 치우친다. 체계적 편향을 의심해야 한다."))

    print("")
    print("  층별:")
    for stn in ("G", "N", "M"):
        sub = [r for r in rows if r["stratum"] == stn]
        if sub:
            k = sum(r["agree"] for r in sub)
            print("    %s층 n=%2d  일치 %2d = %5.1f%%" % (stn, len(sub), k, 100.0 * k / len(sub)))

    io.open(out / "judge_agreement.json", "w", encoding="utf-8").write(json.dumps({
        "n_overlap": n, "n_agree": ag, "agreement_pct": round(100.0 * ag / n, 4),
        "ci": [round(lo, 4), round(hi, 4)], "spec_gate_pct": GATE,
        "is_spec_sample": False,
        "note": ("D8-B 게이트가 지정한 25건 표본이 아니라 V' 60건과 우연히 겹친 문장이다. "
                 "관측 일치율만 보고하고 통과/탈락 판정은 하지 않는다."),
        "disagreements": {"%s|%s" % k: c for k, c in dis.items()},
    }, ensure_ascii=False, indent=1))

    print("")
    print("**함의.** D8-B 의 FNR 72.5% 와 G층 precision 0.913 은 판정자 라벨 기준이다.")
    print("판정자가 사람과 %.1f%% 만 맞는다면 그 두 숫자는 사람 기준값으로 대체하거나"
          % (100.0 * ag / n))
    print("이 일치율과 함께 보고해야 한다. **V' 의 사람 기준 수치는 이 문제가 없다** —")
    print("N층 회수 0.0% 대 66.7% 는 사람 라벨로 직접 잰 것이다.")
    print("-> %s" % (out / "judge_agreement.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
