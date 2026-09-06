"""추정자 비교표 — 공저자가 "논문의 중심 표"로 지목한 것. 계산 없음, 조립만.

공저자 판단: 감사 논문에 비교 **시스템**은 필요 없다. 심사자가 요구하는 것은 비교
**추정자**다. 표 7 을 아래 네 열로 확장하면 "baseline 이 없다"는 지적이 사라진다.

  1. 양성 통제 4x 유지율      전제 길이를 4배로 늘려도 지지 판정을 지키는가 (D10/D13)
  2. 음성 통제 위양성          다른 토픽 전제를 지지라고 하는가 (D10-N, 게이트 <=5%)
  3. 라벨 대비 회수·유지       D8-B N층 지지 29건 회수 / G층 42건 유지 (D12)
  4. 순증                      정상 - 교란. **판정은 이것으로만 한다**

이 스크립트는 각 실험의 CSV 를 읽어 표를 만들 뿐이다. 새로 계산하지 않는다.
숫자는 전부 **원행에서 다시 세어** 반올림끼리 빼는 일이 없게 한다.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rd(p):
    q = ROOT / p
    return list(csv.DictReader(io.open(q, encoding="utf-8"))) if q.exists() else []


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def pctci(k, n):
    lo, hi = wilson(k, n)
    return "%.1f%% [%.1f, %.1f]" % (100 * k / n, 100 * lo, 100 * hi) if n else "-"


SHORT = {
    "cross-encoder/nli-deberta-v3-base": "DeBERTa-v3-base (현행 g1)",
    "cross-encoder/nli-deberta-v3-large": "DeBERTa-v3-large",
    "roberta-large-mnli": "RoBERTa-large-MNLI",
    "facebook/bart-large-mnli": "BART-large-MNLI",
}

# --- 원행에서 다시 센다 -----------------------------------------------------
neg = rd("results/diag/d10/d10n_negative_by_checkpoint.csv")
pad = rd("results/diag/d10/d10_padding_by_checkpoint.csv")
rec = rd("results/diag/d12/d12_recovery_rows.csv")

cks = [c for c in SHORT if any(r["checkpoint"] == c for r in neg)]

print("# 추정자 비교표")
print("")
print("전 수치는 원행에서 다시 셌다. 반올림된 값끼리 빼지 않는다.")
print("괄호는 Wilson 95% 구간. 게이트는 D8-A 에서 실행 전 고정한 **교란 ≤ 5%**.")
print("")
print("| 추정자 | 양성 4x 유지 | 정상 | **교란(위양성)** | **순증** | 게이트 |")
print("|---|---|---|---|---|---|")
rowsum = {}
for c in cks:
    sub = [r for r in neg if r["checkpoint"] == c]
    kn = sum(1 for r in sub if float(r["p_normal"]) >= 0.5)
    ks = sum(1 for r in sub if float(r["p_shuffled"]) >= 0.5)
    n = len(sub)
    p4 = [r for r in pad if r["checkpoint"] == c and int(r["multiplier"]) == 4]
    k4 = sum(1 for r in p4 if float(r["p_entail"]) >= 0.5)
    net = 100.0 * (kn - ks) / n
    gate = ks / n <= 0.05
    rowsum[c] = {"net": net, "gate": gate, "n": n}
    print("| %s | %s | %s | %s | **%.2f%%p** | %s |"
          % (SHORT[c], pctci(k4, len(p4)) if p4 else "-", pctci(kn, n), pctci(ks, n),
             net, "PASS" if gate else "**FAIL**"))

print("")
print("| 추정자 | N층 지지 29건 **회수** | N층 미지지 10건 **오회수** | G층 42건 **유지** |")
print("|---|---|---|---|")
for c in cks:
    sub = [r for r in rec if r["checkpoint"] == c]
    out = []
    for g in ("N층 supported (회수 대상)", "N층 unsupported (오회수 대조)",
              "G층 supported (유지 대조)"):
        it = [r for r in sub if r["stratum"] + "|" + r["judge_label"] ==
              ("N|supported" if "회수 대상" in g else
               "N|unsupported" if "오회수" in g else "G|supported")]
        k = sum(1 for r in it if int(r["grounded"]))
        out.append(pctci(k, len(it)))
    print("| %s | %s | %s | %s |" % (SHORT[c], out[0], out[1], out[2]))

print("")
print("## 읽는 법")
print("")
print("- **판정은 순증으로만 한다.** grounded 만 보고 고르면 D7 의 실수를 반복한다")
print("  (D7 은 57.1% 를 보고 채택했다가 위양성 50.5% 로 철회됐다).")
print("- **회수와 오회수는 함께 본다.** 회수만 오르고 오회수가 안 오르면 진짜 개선이고,")
print("  둘이 같이 오르면 관대한 것이다. 다만 오회수 칸은 n=10 이라 구간이 넓다 —")
print("  이 칸만으로 판정하면 안 된다.")
print("- **기준 라벨이 gpt-5.4-nano 다.** 사람 게이트 전이므로 아래 두 표의 우변은")
print("  전부 **잠정치**다. 사람 라벨이 들어오면 호출 0회로 다시 돌린다.")
print("")
ok = [c for c in cks if rowsum[c]["gate"]]
if ok:
    b = max(ok, key=lambda c: rowsum[c]["net"])
    print("게이트 통과 중 순증 최대: **%s (%.2f%%p)**" % (SHORT[b], rowsum[b]["net"]))
print("")
print("## 아직 비어 있는 열")
print("")
print("- **MiniCheck (D9)** — 문서 단위 검증기. 계열 밖 후보.")
print("- **LLM judge 를 추정자로** — nano 에 같은 교란을 걸어 위양성을 재는 열.")
print("  \"그냥 LLM 쓰면 되지 않느냐\"는 심사자 질문에 대한 답이 된다. 판정 300호출.")
print("- **사람 라벨 대비 정밀·재현** — 현재 기준 라벨이 nano 라서 비어 있다.")
print("- **통과 시 낭비율 W** — D9-C 재채점. 사전 등록 게이트에 걸려 아직 못 연다.")
