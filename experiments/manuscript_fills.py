# -*- coding: utf-8 -*-
"""원고 v0.4 의 [[ ]] 수치 자리를 원자료에서 채운다. 읽기 전용."""
import csv, io, json, statistics as st
from pathlib import Path
R = Path(".")
def rd(p): return list(csv.DictReader(io.open(R/p, encoding="utf-8")))
def f(r,k): return float(r[k])

def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda k: v[k]); r=[0.0]*len(v); k=0
        while k < len(o):
            j = k
            while j+1 < len(o) and v[o[j+1]] == v[o[k]]: j += 1
            for m in range(k, j+1): r[o[m]] = (k+j)/2 + 1
            k = j+1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((ra[i]-ma)*(rb[i]-mb) for i in range(len(a)))
    den = (sum((x-ma)**2 for x in ra)*sum((x-mb)**2 for x in rb))**0.5
    return num/den if den else 0.0

print("### 1. Table 3 — C1~C4 평균 (branches.csv 에서 직접)")
print("| 조건 | mean cov | mean grd | mean grd (비어있지 않은 답안) | n non-empty |")
print("|---|---|---|---|---|")
for c in ("C1_baseline","C2_topk15","C3_partial_heldout","C4_both"):
    b = rd("results/diag/sweep/%s/branches.csv" % c)
    ne = [r for r in b if int(r.get("degraded", 0)) == 0]
    print("| %s | %.3f | %.3f | %.3f | %d/60 |" % (
        c, st.mean([f(r,"coverage") for r in b]), st.mean([f(r,"grounding_ratio") for r in b]),
        st.mean([f(r,"grounding_ratio") for r in ne]) if ne else float("nan"), len(ne)))

print()
print("### 2. Table 2 — 문장 수 구간 (sentence_count.csv, n=600)")
sc = rd("results/diag/sentence_count.csv")
ne = [r for r in sc if int(r["n_sentences"]) > 0]
bands = [("0<grd<=0.5", lambda g: 0 < g <= 0.5), ("0.5<grd<=0.8", lambda g: 0.5 < g <= 0.8),
         ("grd>0.8", lambda g: g > 0.8)]
print("| 구간 | n | 문장 수 중앙 | 범위 |")
print("|---|---|---|---|")
for lbl, p in bands:
    s = [int(r["n_sentences"]) for r in sc if p(f(r,"grounding"))]
    print("| %s | %d | %.1f | %d–%d |" % (lbl, len(s), st.median(s), min(s), max(s)))
c1 = [int(r["n_sentences"]) for r in sc if f(r,"coverage")==1.0]
print("| cov=1.0 (grounding 무관) | %d | %.1f | **%d–%d** |" % (len(c1), st.median(c1), min(c1), max(c1)))
hi = [int(r["n_sentences"]) for r in sc if f(r,"grounding") > 0.8]
print()
print("grd>0.8 30건 중 1문장 %d · 2문장 %d · 3문장 %d · 4문장 %d"
      % tuple(sum(1 for x in hi if x == k) for k in (1,2,3,4)))
print("rho(문장수, grounding)  전체 600 = %+.3f   |   비어있지 않은 %d건 = %+.3f"
      % (spearman([int(r["n_sentences"]) for r in sc], [f(r,"grounding") for r in sc]),
         len(ne), spearman([int(r["n_sentences"]) for r in ne], [f(r,"grounding") for r in ne])))
suf = [r for r in sc if f(r,"coverage")==1.0 and f(r,"grounding")>0.8]
print("충족 branch %d건: 문장 %s · 인용 %s · claim %s"
      % (len(suf), [int(r["n_sentences"]) for r in suf],
         [int(r["n_citations"]) for r in suf], [int(r["n_claims"]) for r in suf]))

print()
print("### 3. §VI-D 추정자 해부 (pool_unit_grid.csv, unit=sentence, n=294)")
g = rd("results/diag/d7/pool_unit_grid.csv")
S = [r for r in g if r["unit"]=="sentence"]
pc = [f(r,"p_cited") for r in S]
n = len(pc)
print("n=%d  <0.05 %.1f%%  >=0.90 %.1f%%  중간대 0.05~0.90 %.1f%%  중간대 0.1~0.9 %.1f%%"
      % (n, 100*sum(1 for x in pc if x<0.05)/n, 100*sum(1 for x in pc if x>=0.90)/n,
         100*sum(1 for x in pc if 0.05<=x<0.90)/n, 100*sum(1 for x in pc if 0.1<=x<=0.9)/n))
for tau in (0.5, 0.3, 0.2, 0.1):
    print("  tau_g=%.1f -> grounded %.1f%%" % (tau, 100*sum(1 for x in pc if x>=tau)/n))
low = [r for r in S if f(r,"p_cited") < 0.05]
print("귀속: 인용청크 <0.05 인 문장 %d개 중, 전체 retrieved 로도 <0.05 인 것 %d (%.1f%%)"
      % (len(low), sum(1 for r in low if f(r,"p_retrieved")<0.05),
         100*sum(1 for r in low if f(r,"p_retrieved")<0.05)/len(low)))
rem = [r for r in low if f(r,"p_retrieved") < 0.05]
print("누출: 그 %d개 중 target article 전체로 >=0.5 %d (%.1f%%) · 여전히 <0.05 %d"
      % (len(rem), sum(1 for r in rem if f(r,"p_target")>=0.5),
         100*sum(1 for r in rem if f(r,"p_target")>=0.5)/len(rem),
         sum(1 for r in rem if f(r,"p_target")<0.05)))
print("전제 풀 중앙: cited %.0f · retrieved %.0f · target %.0f"
      % tuple(st.median([int(r[k]) for r in S]) for k in ("n_cited","n_retrieved","n_target")))
sp = rd("results/diag/d7/pool_unit_grid_sentprem.csv")
print("\n중간대(0.1~0.9): ", end="")
for lbl, src, u in [("문장×청크", g,"sentence"), ("문장×문장전제", sp,"sentence"),
                    ("claim×청크", g,"claim"), ("claim×문장전제", sp,"claim")]:
    v = [f(r,"p_cited") for r in src if r["unit"]==u]
    print("%s %.1f%%  " % (lbl, 100*sum(1 for x in v if 0.1<=x<=0.9)/len(v)), end="")
print()
print("문장전제 풀 중앙 (claim 행): %.0f 문장"
      % st.median([int(r["n_cited"]) for r in sp if r["unit"]=="claim"]))
for lbl, src, u in [("문장×청크", g,"sentence"), ("문장×문장전제", sp,"sentence"),
                    ("claim×청크", g,"claim"), ("claim×문장전제", sp,"claim")]:
    v = [f(r,"p_cited") for r in src if r["unit"]==u]
    print("  %-14s grounded %.1f%%  중앙 %.3f" % (lbl, 100*sum(1 for x in v if x>=0.5)/len(v), st.median(v)))

print()
print("### 4. rho 불일치 해소 — +0.293 vs -0.045")
neg = rd("results/diag/d8/d8a_negative_control.csv")
r_shuf = spearman([float(r["n_premise_sentences"]) for r in neg], [f(r,"max_p") for r in neg])
print("D8-A 음성통제 스크립트: rho(전제문장수, max_p) on **교란 배정** = %+.3f  (n=%d)"
      % (r_shuf, len(neg)))
net = rd("results/diag/d8/d8a2_estimator_net.csv")
for est in ("E1","E2","E3"):
    sub = [r for r in net if r["estimator"]==est]
    ps = [float(r["pool_size"]) for r in sub]
    print("d8a2 %s: rho(pool_size, p_normal) = %+.3f | rho(pool_size, p_shuffled) = %+.3f | n=%d, 풀 중앙 %.0f"
          % (est, spearman(ps, [float(r["p_normal"]) for r in sub]),
             spearman(ps, [float(r["p_shuffled"]) for r in sub]), len(sub), st.median(ps)))
