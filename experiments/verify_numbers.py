"""보고한 모든 헤드라인 수치를 원자료에서 재계산해 대조한다.

논문에 들어갈 숫자가 여러 문서·여러 대화를 거치며 옮겨 적혔다. 한 번은 기계로
다시 세어야 한다. 이미 한 건(E2 정상 57.1 vs 62.3)이 옮겨 적는 과정이 아니라
정의 차이에서 났고, 그건 이 대조로는 안 잡혔을 것이다 — 그래서 정의도 함께 적는다.

읽기 전용. 계산만 한다.
"""

from __future__ import annotations

import csv
import io
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OK, BAD = "OK  ", "MISMATCH"
fails: list[str] = []


def rd(p: str) -> list[dict]:
    return list(csv.DictReader(io.open(ROOT / p, encoding="utf-8")))


def chk(label: str, got, want, tol: float = 0.051) -> None:
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        ok = abs(float(got) - float(want)) <= tol
    else:
        ok = got == want
    mark = OK if ok else BAD
    if not ok:
        fails.append(label)
    g = ("%.3f" % got) if isinstance(got, float) else str(got)
    w = ("%.3f" % want) if isinstance(want, float) else str(want)
    print("  [%s] %-52s 보고 %-10s 실측 %s" % (mark, label, w, g))


def pct(xs, pred) -> float:
    return 100.0 * sum(1 for x in xs if pred(x)) / len(xs) if xs else 0.0


print("=" * 78)
print("1. Test A 본실험  results/test_a/20260829-041445/")
print("=" * 78)
D = "results/test_a/20260829-041445/"
rows = rd(D + "waste_ratio.csv")
ci = rd(D + "waste_ratio_ci.csv")
f = lambda r, k: float(r[k])

chk("branch-round 행 수", len(rows), 600)
for r in ci:
    tau = float(r["tau"])
    want = {0.7: 1.8, 0.8: 0.5, 0.9: 0.5}[round(tau, 1)]
    chk("낭비율 tau=%.1f (%%)" % tau, 100 * float(r["point"]), want)
chk("낭비/재실행 tau=0.8", "%s/%s" % (ci[1]["n_wasted_branches"], ci[1]["n_reexec_branches"]),
    "2/400")
res = json.load(io.open(ROOT / D / "results.json", encoding="utf-8"))
chk("verify 통과 질의", res["context"]["n_passed"], 0)
chk("실패 케이스(라운드0 불통과)", res["context"]["n_failed"], 50)
chk("coverage 중앙값", st.median([f(r, "coverage") for r in rows]), 0.40)
chk("grounding 중앙값", st.median([f(r, "grounding") for r in rows]), 0.00)
chk("coverage=1.0 branch", sum(1 for r in rows if f(r, "coverage") == 1.0), 128)
chk("grounding>0.8 branch", sum(1 for r in rows if f(r, "grounding") > 0.8), 30)
chk("둘 다 만족", sum(1 for r in rows if f(r, "coverage") == 1.0 and f(r, "grounding") > 0.8), 3)
chk("빈 답안(degraded)", sum(int(r["degraded"]) for r in rows), 215)

print("\n  라운드별 (정의: 라운드 r 의 200 branch)")
r0 = [r for r in rows if int(r["round"]) == 0]
r2 = [r for r in rows if int(r["round"]) == 2]
for lbl, sub, wc, wg in [("라운드0", r0, 0.461, 0.200), ("라운드2", r2, 0.401, 0.158)]:
    chk("%s coverage 평균" % lbl, st.mean([f(r, "coverage") for r in sub]), wc, 0.002)
    chk("%s grounding 평균" % lbl, st.mean([f(r, "grounding") for r in sub]), wg, 0.002)
chk("라운드0→2 coverage 변화",
    st.mean([f(r, "coverage") for r in r2]) - st.mean([f(r, "coverage") for r in r0]), -0.060, 0.002)
chk("라운드0→2 grounding 변화",
    st.mean([f(r, "grounding") for r in r2]) - st.mean([f(r, "grounding") for r in r0]), -0.042, 0.002)
tk = {rd_: sum(int(r["tokens"]) for r in rows if int(r["round"]) == rd_) for rd_ in (0, 1, 2)}
chk("라운드1~2 토큰 비중 (%)", 100 * (tk[1] + tk[2]) / sum(tk.values()), 71.9, 0.2)

print("\n" + "=" * 78)
print("2. D0 문장 수 감사  results/diag/sentence_count.csv")
print("=" * 78)
sc = rd("results/diag/sentence_count.csv")
hi = [r for r in sc if f(r, "grounding") > 0.8]
chk("grounding>0.8 행 수", len(hi), 30)
chk("그 문장 수 중앙값", st.median([int(r["n_sentences"]) for r in hi]), 2.0, 0.01)
one = [r for r in sc if f(r, "grounding") == 1.0 and int(r["n_sentences"]) == 1]
chk("grounding=1.0 이면서 문장 1개", len(one), 14)
c1 = [r for r in sc if f(r, "coverage") == 1.0 and int(r["n_sentences"]) > 0]
chk("coverage=1.0 의 문장수 중앙값", st.median([int(r["n_sentences"]) for r in c1]), 10.0, 0.01)
chk("coverage=1.0 중 1~2문장", sum(1 for r in c1 if int(r["n_sentences"]) <= 2), 0)

print("\n" + "=" * 78)
print("3. D1~D3 진단 스윕  results/diag/sweep/")
print("=" * 78)
want = {"C1_baseline": (0.0, 0.000, 28.3, 35.0), "C2_topk15": (0.0, 0.136, 23.3, 40.0),
        "C3_partial_heldout": (0.0, 0.000, 33.3, 13.3), "C4_both": (1.7, 0.136, 33.3, 20.0)}
for cond, (wi, wm, wc, we) in want.items():
    m = json.load(io.open(ROOT / ("results/diag/sweep/%s/metrics.json" % cond), encoding="utf-8"))
    chk("%s 교집합(%%)" % cond, 100 * m["intersection_rate"], wi, 0.1)
    chk("%s cov1 grd중앙" % cond, m["cov1_grounding_median"], wm, 0.001)
    chk("%s cov1 비율(%%)" % cond, 100 * m["cov1_rate"], wc, 0.1)
    chk("%s 빈답안(%%)" % cond, 100 * m["empty_rate"], we, 0.1)

print("\n" + "=" * 78)
print("4. D7-0 패딩 통제  results/diag/d7/padding_control.csv")
print("=" * 78)
pc = rd("results/diag/d7/padding_control.csv")
chk("양성 쌍 수", len({r["sentence_id"] for r in pc}), 35)
for m_, wt, wtr, wp, wk in [(1, 142, 0, 0.992, 100), (2, 274, 6, 0.152, 34),
                            (3, 443, 15, 0.003, 11), (4, 626, 28, 0.001, 3)]:
    sub = [r for r in pc if int(r["multiplier"]) == m_]
    chk("%dx 전제 토큰 중앙" % m_, st.median([int(r["premise_tokens_after_tokenizer"]) for r in sub]), wt, 1)
    chk("%dx 토크나이저 절단" % m_, sum(int(r["truncated_by_tokenizer"]) for r in sub), wtr)
    chk("%dx p_entail 중앙" % m_, st.median([f(r, "p_entail") for r in sub]), wp, 0.002)
    chk("%dx >=0.5 유지(%%)" % m_, pct([f(r, "p_entail") for r in sub], lambda x: x >= 0.5), wk, 1.5)
p1 = {r["sentence_id"]: f(r, "p_entail") for r in pc if int(r["multiplier"]) == 1}
p4 = {r["sentence_id"]: f(r, "p_entail") for r in pc if int(r["multiplier"]) == 4}
chk("1x→4x 하락 중앙", st.median([p1[k] - p4[k] for k in p1]), 0.983, 0.002)
chk("0.5 아래로 무너진 쌍", sum(1 for k in p1 if p1[k] >= 0.5 > p4[k]), 34)

print("\n" + "=" * 78)
print("5. D7-1/2 격자  (정의: 전제 풀 = branch 의 인용 청크)")
print("=" * 78)
g = rd("results/diag/d7/pool_unit_grid.csv")
sp = rd("results/diag/d7/pool_unit_grid_sentprem.csv")
for lbl, src, unit, col, w in [
        ("문장가설×청크전제 (현행)", g, "sentence", "p_cited", 15.6),
        ("문장가설×retrieved", g, "sentence", "p_retrieved", 17.3),
        ("문장가설×target", g, "sentence", "p_target", 17.3),
        ("claim가설×청크전제", g, "claim", "p_cited", 31.7),
        ("claim가설×retrieved", g, "claim", "p_retrieved", 31.9),
        ("claim가설×target", g, "claim", "p_target", 17.9),
        ("문장가설×문장전제", sp, "sentence", "p_cited", 38.4),
        ("claim가설×문장전제", sp, "claim", "p_cited", 57.1)]:
    v = [f(r, col) for r in src if r["unit"] == unit]
    chk("%s grounded(%%)" % lbl, pct(v, lambda x: x >= 0.5), w, 0.15)

print("\n" + "=" * 78)
print("6. D8-A 음성 통제  (정의: 전제 풀 = 토픽 4 branch 인용 청크 합)")
print("=" * 78)
# 요약 CSV(d8a2_summary) 는 4자리로 반올림해 저장한다. 그 값을 다시 반올림하면
# 15.6463% 가 0.1565 -> 15.7% 로 튄다 (실제로 원고에 15.7% 가 그렇게 들어갔다).
# 따라서 **원행에서 다시 센다.**
net_rows = rd("results/diag/d8/d8a2_estimator_net.csv")
for est, wn, ws, wnet in [("E1", 15.6, 0.0, 15.6), ("E2", 62.3, 50.5, 11.7), ("E3", 53.5, 15.2, 38.3)]:
    sub = [x for x in net_rows if x["estimator"] == est]
    gn = pct(sub, lambda x: f(x, "p_normal") >= 0.5)
    gs = pct(sub, lambda x: f(x, "p_shuffled") >= 0.5)
    chk("%s 정상(%%) [원행 %d개]" % (est, len(sub)), gn, wn, 0.05)
    chk("%s 교란(%%)" % est, gs, ws, 0.05)
    chk("%s 순증(%%p)" % est, gn - gs, wnet, 0.05)
neg = rd("results/diag/d8/d8a_negative_control.csv")
chk("D8-A 교란 쌍 수", len(neg), 546)
chk("D8-A 위양성(%)", pct([f(r, "max_p") for r in neg], lambda x: x >= 0.5), 50.5, 0.15)
chk("전제 문장 수 중앙", st.median([int(r["n_premise_sentences"]) for r in neg]), 96, 1)
chk("같은 토픽 배정 0건", sum(1 for r in neg if r["src_topic"] == r["premise_topic"]), 0)

print("\n" + "=" * 78)
print("6b. 논문이 보고하는 %p 차이 — 반올림끼리 빼지 않았는지 (원행 재계산)")
print("=" * 78)
# 세 번 당했다: 15.7(이중 반올림) · 11.8(반올림끼리 뺌) · 5.2(반올림끼리 뺌).
# 눈으로 잡을 게 아니라 구조로 잡는다. 모든 %p 는 **원행에서 센 비율끼리** 뺀다.
G = [r for r in g if r["unit"] == "sentence"]
GC = [r for r in g if r["unit"] == "claim"]
SPC = [r for r in sp if r["unit"] == "claim"]
E2 = [r for r in net_rows if r["estimator"] == "E2"]
p_ = lambda rows, k: pct(rows, lambda r: f(r, k) >= 0.5)
for lbl, a, b, want in [
        ("전제 풀 확대 cited->retrieved", p_(G, "p_retrieved"), p_(G, "p_cited"), 1.7),
        ("가설 단위 문장->claim", p_(GC, "p_cited"), p_(G, "p_cited"), 16.0),
        ("전제 풀 branch->토픽 (E2)", p_(E2, "p_normal"), p_(SPC, "p_cited"), 5.1),
        ("E2 순증 (정상-교란)", p_(E2, "p_normal"), p_(E2, "p_shuffled"), 11.7)]:
    chk(lbl + " (%p)", a - b, want, 0.05)

print("\n" + "=" * 78)
print("7. D8-B 판정 표본  results/diag/d8/d8b_judge_sample.csv")
print("=" * 78)
jb = rd("results/diag/d8/d8b_judge_sample.csv")
for st_, wn, wsup, wr in [("G", 46, 42, 91.3), ("N", 40, 29, 72.5), ("M", 20, 18, 90.0)]:
    sub = [r for r in jb if r["stratum"] == st_]
    chk("%s층 n" % st_, len(sub), wn)
    chk("%s층 supported" % st_, sum(1 for r in sub if r["judge_label"] == "supported"), wsup)
    chk("%s층 지지 비율(%%)" % st_, pct(sub, lambda r: r["judge_label"] == "supported"), wr, 0.15)
chk("층 경계 위반 (G 는 max_p>=0.5)",
    sum(1 for r in jb if r["stratum"] == "G" and f(r, "max_p") < 0.5), 0)
chk("층 경계 위반 (N 은 max_p<0.05)",
    sum(1 for r in jb if r["stratum"] == "N" and f(r, "max_p") >= 0.05), 0)

print("\n" + "=" * 78)
if fails:
    print("불일치 %d건:" % len(fails))
    for x in fails:
        print("  - " + x)
    sys.exit(1)
print("전 항목 일치. 보고된 수치가 원자료에서 재현된다.")
