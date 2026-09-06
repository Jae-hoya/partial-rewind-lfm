"""D11 — 원고에 빠져 있던 통계. 자료는 그대로, 계산만 추가한다. **호출 0회.**

공저자가 지적한 구멍 중 계산으로 메울 수 있는 것들이다.

  A. 라운드 0→2 악화에 유의성이 없다. -0.060, -0.042 는 지금 "주장이지 결과가 아니다".
     질의별로 짝지은 값이 있으므로 **paired bootstrap** 으로 구간을 낸다.
  B. 원고의 비율에 구간이 없다. 표 3·7·8 전부에 **Wilson 구간**을 붙인다.
  C. 빈 답안 35% 가 왜 비었는지 원고에 한 줄도 없다. 원인별로 분류한다.
  D. 임계 tau_g 를 촘촘히 쓸어 이진성이 임계 무관임을 곡선으로 보인다.

사전 등록 성격: 게이트 없음. 기존 수치의 불확실성을 드러내는 것이지 새 판정이 아니다.
Wilson 을 쓰는 이유는 비율이 0 이나 1 에 붙어 있어 정규근사가 무너지기 때문이다
(위양성 0.0%, verify 통과 0/50 처럼).
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOOT_N = 10000
SEED = 42


def rd(p):
    return list(csv.DictReader(io.open(ROOT / p, encoding="utf-8")))


def f(r, k):
    return float(r[k])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 구간. k/n 이 0 이나 1 이어도 무너지지 않는다."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def wline(lbl: str, k: int, n: int) -> None:
    lo, hi = wilson(k, n)
    print("  %-46s %3d/%-4d = %6.1f%%  [%5.1f, %5.1f]"
          % (lbl, k, n, 100 * k / n if n else 0.0, 100 * lo, 100 * hi))


print("=" * 78)
print("A. 라운드 0 -> 2 악화 — paired bootstrap (질의 단위, n=50, B=%d)" % BOOT_N)
print("=" * 78)
rows = rd("results/test_a/20260829-041445/waste_ratio.csv")
by_q: dict[str, dict[int, list[dict]]] = {}
for r in rows:
    by_q.setdefault(r["query_id"], {}).setdefault(int(r["round"]), []).append(r)

qids = sorted(by_q)
print("질의 %d개 · 질의당 라운드 %s" % (len(qids), sorted(by_q[qids[0]])))
rng = np.random.default_rng(SEED)

for metric in ("coverage", "grounding"):
    # 질의별 branch 평균의 라운드 차이 — 짝지은 값
    diffs = []
    for q in qids:
        a = st.mean([f(r, metric) for r in by_q[q][0]])
        b = st.mean([f(r, metric) for r in by_q[q][2]])
        diffs.append(b - a)
    point = st.mean(diffs)
    arr = np.array(diffs)
    boot = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(BOOT_N)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    worse = sum(1 for d in diffs if d < 0)
    better = sum(1 for d in diffs if d > 0)
    same = len(diffs) - worse - better
    print("")
    print("  %s  라운드0 -> 라운드2" % metric)
    print("    평균 변화 %+.4f   95%% CI [%+.4f, %+.4f]   %s"
          % (point, lo, hi, "0 을 포함하지 않는다 (유의)" if hi < 0 or lo > 0 else "0 을 포함한다"))
    print("    나빠진 질의 %d · 좋아진 질의 %d · 그대로 %d" % (worse, better, same))
    # 부호검정 (그대로인 것 제외)
    n_eff = worse + better
    if n_eff:
        p_sign = 2 * sum(math.comb(n_eff, k) for k in range(min(worse, better) + 1)) / 2 ** n_eff
        print("    부호검정 p = %.4g  (n=%d)" % (min(p_sign, 1.0), n_eff))

print("")
print("=" * 78)
print("B. 원고 비율에 Wilson 95% 구간")
print("=" * 78)
print(" [Test A]")
wline("verify 통과 질의", 0, 50)
wline("낭비 재실행 tau=0.8", 2, 400)
wline("낭비 재실행 tau=0.7", 7, 400)
wline("cov=1.0 인 branch", 128, 600)
wline("grd>0.8 인 branch", 30, 600)
wline("충족 branch (cov=1 & grd>0.8)", 3, 600)
wline("빈 답안", 215, 600)

print(" [D1~D3 스윕, 조건당 60 branch]")
for cond, lbl in [("C1_baseline", "C1 기준선"), ("C2_topk15", "C2 k=15"),
                  ("C3_partial_heldout", "C3 부분 held-out"), ("C4_both", "C4 둘 다")]:
    m = json.load(io.open(ROOT / ("results/diag/sweep/%s/metrics.json" % cond), encoding="utf-8"))
    wline(lbl + " cov=1.0", round(m["cov1_rate"] * 60), 60)
    wline(lbl + " 빈 답안", round(m["empty_rate"] * 60), 60)

print(" [D8-A 음성 통제]")
g = rd("results/diag/d7/pool_unit_grid.csv")
S = [r for r in g if r["unit"] == "sentence"]
net = rd("results/diag/d8/d8a2_estimator_net.csv")
for est, n_ in [("E1", 294), ("E2", 546), ("E3", 546)]:
    sub = [r for r in net if r["estimator"] == est]
    wline(est + " 정상 grounded", sum(1 for r in sub if f(r, "p_normal") >= 0.5), len(sub))
    wline(est + " 교란 위양성", sum(1 for r in sub if f(r, "p_shuffled") >= 0.5), len(sub))

print(" [D8-B 판정 표본]")
jb = rd("results/diag/d8/d8b_judge_sample.csv")
for stn, lbl in [("G", "G층 precision"), ("N", "N층 FNR"), ("M", "M층 지지율")]:
    sub = [r for r in jb if r["stratum"] == stn]
    wline(lbl, sum(1 for r in sub if r["judge_label"] == "supported"), len(sub))
fid = rd("results/diag/d8/d8b_fidelity.csv")
wline("claim 충실도 complete", sum(1 for r in fid if r["label"] == "complete"), len(fid))
wline("claim 충실도 addition", sum(1 for r in fid if r["label"] == "addition"), len(fid))

print("")
print("=" * 78)
print("C. 빈 답안 215건은 왜 비었나")
print("=" * 78)
sc = rd("results/diag/sentence_count.csv")
empty = [r for r in sc if int(r["n_sentences"]) == 0]
print("빈 답안 %d건 (전체 %d 중 %.1f%%)" % (len(empty), len(sc), 100 * len(empty) / len(sc)))
cause = Counter()
for r in empty:
    n_chunk = int(r["n_chunks"])
    n_sub = int(r["n_sub_questions"])
    n_ans = int(r["n_answered_sub"])
    if n_chunk == 0:
        cause["검색 결과 0건"] += 1
    elif n_sub == 0:
        cause["sub-question 분해 실패"] += 1
    elif n_ans == 0:
        cause["청크는 있는데 인용 붙은 sub-answer 0"] += 1
    else:
        cause["sub-answer 는 있는데 문장 0 (빈 문자열)"] += 1
for k, v in cause.most_common():
    print("  %-42s %3d  (%.1f%%)" % (k, v, 100 * v / len(empty)))
print("")
print("  빈 답안의 검색 청크 수: 중앙 %.0f · 0건인 것 %d"
      % (st.median([int(r["n_chunks"]) for r in empty]),
         sum(1 for r in empty if int(r["n_chunks"]) == 0)))
print("  빈 답안의 sub-question 수: 중앙 %.0f"
      % st.median([int(r["n_sub_questions"]) for r in empty]))
ne = [r for r in sc if int(r["n_sentences"]) > 0]
print("  대조 — 비어있지 않은 %d건의 검색 청크 수 중앙 %.0f"
      % (len(ne), st.median([int(r["n_chunks"]) for r in ne])))
print("")
print("  라운드별 빈 답안:")
for rd_ in (0, 1, 2):
    sub = [r for r in sc if int(r["round"]) == rd_]
    wline("    라운드%d" % rd_, sum(1 for r in sub if int(r["n_sentences"]) == 0), len(sub))

print("")
print("=" * 78)
print("D. 임계 tau_g 곡선 — 이진성이 임계와 무관함을 보인다 (C3, n=294 / 546)")
print("=" * 78)
sp = rd("results/diag/d7/pool_unit_grid_sentprem.csv")
cfgs = [("문장 가설 × 청크 전제 (현행)", [f(r, "p_cited") for r in S]),
        ("claim 가설 × 청크 전제",
         [f(r, "p_cited") for r in g if r["unit"] == "claim"]),
        ("문장 가설 × 문장 전제",
         [f(r, "p_cited") for r in sp if r["unit"] == "sentence"]),
        ("claim 가설 × 문장 전제",
         [f(r, "p_cited") for r in sp if r["unit"] == "claim"])]
taus = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
print("  tau      " + "".join("%7.2f" % t for t in taus))
for lbl, v in cfgs:
    n = len(v)
    print("  %-28s" % lbl[:28] + "".join(
        "%6.1f%%" % (100 * sum(1 for x in v if x >= t) / n) for t in taus))
print("")
for lbl, v in cfgs:
    n = len(v)
    a = 100 * sum(1 for x in v if x >= 0.05) / n
    b = 100 * sum(1 for x in v if x >= 0.95) / n
    print("  %-28s tau 0.05 -> 0.95 전 구간 변동폭 %.1f%%p" % (lbl[:28], a - b))
print("")
print("  임계를 0.05 에서 0.95 로 19배 움직여도 값이 이만큼밖에 안 변한다.")
print("  임계 캘리브레이션으로는 아무것도 못 고친다는 뜻이다.")
