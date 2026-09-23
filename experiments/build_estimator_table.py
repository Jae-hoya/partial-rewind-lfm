"""추정자 비교표 — 공저자가 "논문의 중심 표"로 지목한 것. 계산 없음, 조립만.

공저자 판단: 감사 논문에 비교 **시스템**은 필요 없다. 검토자가 요구하는 것은 비교
**추정자**다. 표 7 을 확장해 아래 열로 만들면 "baseline 이 없다"는 지적이 사라진다.

  1. 양성 통제 1x 동의 / 4x 유지   관문 P 의 두 기준 (D10/D13/D9-P)
  2. 음성 통제 위양성               다른 토픽 전제를 지지라고 하는가 (D10-N/D9-A, 게이트 <=5%)
  3. 라벨 대비 회수 / 오회수 / 유지  D8-B 층별 (D12/D9-V)
  4. 순증                           정상 - 교란. **판정은 이것으로만 한다**

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


def ci(k, n):
    if not n:
        return "-"
    lo, hi = wilson(k, n)
    return "%.1f%% [%.1f, %.1f]" % (100 * k / n, 100 * lo, 100 * hi)


NLI = {
    "cross-encoder/nli-deberta-v3-base": "DeBERTa-v3-base (현행 g1)",
    "cross-encoder/nli-deberta-v3-large": "DeBERTa-v3-large",
    "roberta-large-mnli": "RoBERTa-large-MNLI",
    "facebook/bart-large-mnli": "BART-large-MNLI",
}
MC = "MiniCheck-Flan-T5-Large"

neg = rd("results/diag/d10/d10n_negative_by_checkpoint.csv")
pad = rd("results/diag/d10/d10_padding_by_checkpoint.csv")
rec = rd("results/diag/d12/d12_recovery_rows.csv")
d9a = rd("results/diag/d9/d9a_negative.csv")
d9p = rd("results/diag/d9/d9p_padding.csv")
d9v = rd("results/diag/d9/d9v_recovery.csv")

order = ["cross-encoder/nli-deberta-v3-base", MC, "facebook/bart-large-mnli",
         "roberta-large-mnli", "cross-encoder/nli-deberta-v3-large"]

out = []
w = out.append
w("# 추정자 비교표 — 논문 중심 표 후보")
w("")
w("전 수치는 **원행에서 다시 셌다.** 반올림된 값끼리 빼지 않는다.")
w("괄호는 Wilson 95% 구간. 음성 통제 게이트는 D8-A 에서 실행 전 고정한 **교란 ≤ 5%**.")
w("관문 P 는 D9 스펙에서 실행 전 고정: **1x 동의 ≥ 32/35, 4x 유지 ≥ 80%**.")
w("")
w("## 1. 관문 P (양성 통제) — **통과하는 추정자가 없다. 현행도 못 한다.**")
w("")
w("| 추정자 | 1x 동의 (≥32/35) | 4x 유지 (≥80%) | 관문 P |")
w("|---|---|---|---|")
pgate = {}
for c in order:
    if c == MC:
        s1 = [r for r in d9p if int(r["multiplier"]) == 1]
        s4 = [r for r in d9p if int(r["multiplier"]) == 4]
        col = "p_support"
    else:
        s1 = [r for r in pad if r["checkpoint"] == c and int(r["multiplier"]) == 1]
        s4 = [r for r in pad if r["checkpoint"] == c and int(r["multiplier"]) == 4]
        col = "p_entail"
    if not s1:
        continue
    k1 = sum(1 for r in s1 if float(r[col]) >= 0.5)
    k4 = sum(1 for r in s4 if float(r[col]) >= 0.5)
    ok = (k1 >= 32) and (100.0 * k4 / len(s4) >= 80.0)
    pgate[c] = ok
    w("| %s | %d/%d %s | %s %s | %s |"
      % (NLI.get(c, c), k1, len(s1), "✓" if k1 >= 32 else "✗",
         ci(k4, len(s4)), "✓" if 100.0 * k4 / len(s4) >= 80 else "✗",
         "PASS" if ok else "**FAIL**"))
w("")
w("**두 기준이 서로를 배제한다.** 1x 동의는 g1 과 같이 판단할 것을 요구하고(양성 쌍의")
w("정의가 `g1 의 p > 0.90` 이다), g1 처럼 판단하면 4x 에서 무너진다. **감사받는 대상을**")
w("**감사의 기준으로 썼다.** 관문을 사후에 고치지 않고 결과로 기록한다.")
w("")
w("## 2. 음성 통제 — **판정은 순증으로만 한다**")
w("")
w("| 추정자 | 정상 | **교란(위양성)** | **순증** | 게이트(≤5%) |")
w("|---|---|---|---|---|")
best = None
for c in order:
    sub = d9a if c == MC else [r for r in neg if r["checkpoint"] == c]
    if not sub:
        continue
    n = len(sub)
    kn = sum(1 for r in sub if float(r["p_normal"]) >= 0.5)
    ks = sum(1 for r in sub if float(r["p_shuffled"]) >= 0.5)
    net = 100.0 * (kn - ks) / n
    ok = ks / n <= 0.05
    if ok and (best is None or net > best[1]):
        best = (NLI.get(c, c), net)
    w("| %s | %s | %s | **%.2f%%p** | %s |"
      % (NLI.get(c, c), ci(kn, n), ci(ks, n), net, "PASS" if ok else "**FAIL**"))
w("")
w("## 3. 라벨 대비 — 회수는 오회수와 **함께** 본다")
w("")
w("| 추정자 | N층 지지 29 **회수** | N층 미지지 10 **오회수** | 차이 | G층 42 **유지** |")
w("|---|---|---|---|---|")
V = {"N층 supported (회수)": ("N", "supported"),
     "N층 unsupported (오회수)": ("N", "unsupported"),
     "G층 supported (유지)": ("G", "supported")}
for c in order:
    cells, rates = [], []
    for gname, (stn, lab) in V.items():
        if c == MC:
            it = [r for r in d9v if r["group"] == gname]
        else:
            it = [r for r in rec if r["checkpoint"] == c
                  and r["stratum"] == stn and r["judge_label"] == lab]
        k = sum(1 for r in it if int(r["grounded"]))
        cells.append(ci(k, len(it)))
        rates.append(100.0 * k / len(it) if it else float("nan"))
    if not cells[0] or cells[0] == "-":
        continue
    w("| %s | %s | %s | **%+.1f%%p** | %s |"
      % (NLI.get(c, c), cells[0], cells[1], rates[0] - rates[1], cells[2]))
w("")
w("## 읽는 법")
w("")
w("- **grounded 만 보고 고르면 안 된다.** D7 은 57.1% 를 보고 채택했다가 위양성 50.5%")
w("  로 철회됐다. 판정은 순증이다.")
w("- **회수만 오르고 오회수가 안 오르면 진짜 개선이다.** 둘이 같이 오르면 관대한 것이다.")
w("  다만 오회수 칸은 **n=10** 이라 구간이 넓다 — 이 칸 하나로 판정하지 않는다.")
w("- **기준 라벨이 `gpt-5.4-nano` 다.** 사람 게이트 전이므로 3절은 전부 **잠정치**다.")
w("  사람 라벨이 들어오면 호출 0회로 다시 돌린다.")
w("")
if best:
    w("게이트 통과 중 순증 최대: **%s (%.2f%%p)**" % best)
w("")
w("**그런데 순증 1위는 관문 P 에서 탈락했다.** 관문이 최선의 후보를 떨어뜨렸다는 사실")
w("자체가 이 논문의 결과다.")
w("")
w("## 아직 비어 있는 열")
w("")
w("- **LLM judge 를 추정자로** — nano 에 같은 교란을 걸어 위양성을 재는 행.")
w("  \"그냥 LLM 쓰면 되지 않느냐\"는 검토자 질문에 대한 답. 판정 약 300호출.")
w("- **사람 라벨 대비 정밀·재현** — 현재 기준 라벨이 nano 라서 비어 있다.")
w("- **통과 시 낭비율 W** — D9-C 재채점. 사전 등록 게이트에 걸려 아직 못 연다.")
w("- **전제 구성 축** — D14 의 `trunc`/`union` 은 추정자 교체가 아니라 전제 구성")
w("  변경이라 다른 축이다. 음성 통제 후 별도 표로 낼지 여기 합칠지 미정.")

txt = "\n".join(out)
print(txt)
io.open(ROOT / "results/ESTIMATOR_TABLE.md", "w", encoding="utf-8").write(txt + "\n")
