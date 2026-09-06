"""D12·D8-B 를 **사람 라벨** 기준으로 재집계. 호출 0, 모델 적재 0 — 순수 조인.

왜: 판정자(gpt-5.4-nano) 대 사람 일치율이 65.6% 였다(`judge_agreement.json`).
스펙의 사람 게이트 기준은 90% 다. 그래서 D8-B 의 FNR 72.5% 와 G층 precision
0.913, 그리고 그 라벨 위에 세운 D12 회수표는 **인용 근거로 쓸 수 없다.**
같은 계산을 사람 라벨로 다시 한다. 공저자 지시 2026-09-06 — 결과는 부록으로.

**여기서 모델을 다시 돌리지 않는다.** `d12_recovery_rows.csv` 에 항목별 점수가
이미 있고, V′ 정답키에 `p_g1`·`p_mc`·`p_bart` 가 있다. 라벨만 갈아 끼운다.
두 출처의 전제 범위가 같은지는 §0 에서 **점수 대조로 검증**한다 — 오늘 전제 범위
불일치를 두 번 냈고, `whole_max` 는 범위가 달라도 같은 값이 나와서 기준선 재현으로는
안 잡힌다(`PROGRESS.md` 오류 4).

내는 표 넷:
  A. D12 회수표 재집계 — V′ ∩ D12 모집단. 판정자 라벨 → 사람 라벨.
  B. D8-B 표8 재집계 — V′ ∩ D8-B 표본. 층별 사람 지지율.
  C. V′ 자체 회수표 — 분모가 훨씬 크다. A 의 상위 대체재.
  D. V′ 4칸(P1/P2/N1/N2) 사람 지지율 + **오회수 칸**.

A 와 C 는 모집단이 다르다. C 가 크고, A 는 D12 원표와 행이 맞는다. 둘 다 낸다.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/vprime"
POS, NEG, UNK = "지지", "불지지", "판단불가"

# D8-B 층 정의 (`d8b_judge.py:50`). V′ 는 이 층을 안 쓰므로 여기서 다시 매긴다.
STRATA = [("G", 0.5, 1.01), ("M", 0.05, 0.5), ("N", 0.0, 0.05)]
TAU = 0.5


def rd(p, enc="utf-8"):
    return list(csv.DictReader(io.open(ROOT / p, encoding=enc)))


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * max(0.0, (c - h) / d), 100 * min(1.0, (c + h) / d))


def cell(k, n):
    if not n:
        return "—"
    lo, hi = wilson(k, n)
    return "%d/%d = %.1f%% [%.1f, %.1f]" % (k, n, 100.0 * k / n, lo, hi)


def stratum_of(p):
    for name, lo, hi in STRATA:
        if lo <= p < hi:
            return name
    return "G"


def main() -> int:
    lab = rd("results/vprime/labels_stage1.csv", "utf-8-sig")
    key = {r["item_id"]: r for r in rd("results/vprime/_key_stage1.csv", "utf-8-sig")}
    jb = rd("results/diag/d8/d8b_judge_sample.csv")
    d12 = rd("results/diag/d12/d12_recovery_rows.csv")

    jmap = {r["sentence"].strip(): r for r in jb}
    dmap: dict[str, dict[str, dict]] = {}
    for r in d12:
        dmap.setdefault(r["claim_id"], {})[r["checkpoint"]] = r
    cks = sorted({r["checkpoint"] for r in d12})

    # V′ 항목 하나로 묶는다 — 사람 라벨 + 정답키 + (있으면) D8-B/D12 대응
    items = []
    for x in lab:
        h = x["human_label"].strip()
        k = key.get(x["item_id"])
        if h not in (POS, NEG) or not k:
            continue                                    # 판단불가·빈칸은 분모에서 뺀다
        j = jmap.get(x["문장"].strip())
        items.append({"id": x["item_id"], "human": h, "key": k, "judge": j,
                      "d12": dmap.get(j["claim_id"]) if j else None,
                      "p_g1": float(k["p_g1"]), "cell": k["cell"]})

    L: list[str] = []
    P = L.append
    P("# D12·D8-B 사람 라벨 재집계")
    P("")
    P("호출 0 · 모델 적재 0. `d12_human_relabel.py`. 판정자 일치율 65.6% 라 "
      "판정자 기준 표를 사람 기준으로 갈아 끼운다.")
    P("")

    # ---------------- §0 전제 범위 정합성 -----------------------------------
    ov = [i for i in items if i["d12"]]
    bad = []
    for i in ov:
        for col, ck in [("p_bart", "facebook/bart-large-mnli"),
                        ("p_g1", "cross-encoder/nli-deberta-v3-base")]:
            if ck in i["d12"]:
                a = float(i["d12"][ck]["new_max_p"])
                b = float(i["key"][col])
                if abs(a - b) > 1e-4:
                    bad.append((i["id"], ck, a, b))
    P("## 0. 전제 범위 정합성 — 두 출처를 섞어도 되는가")
    P("")
    P("V′ 정답키와 D12 행은 **따로 계산된 점수**다. 전제 풀이 다르면 섞으면 안 된다. "
      "겹치는 %d건에서 같은 모델의 점수를 대조했다." % len(ov))
    P("")
    P("**불일치 %d / %d.** %s" % (len(bad), 2 * len(ov),
                                 "두 출처의 전제 범위가 같다. 섞어 쓴다."
                                 if not bad else "**범위가 어긋난다. 아래 표는 못 쓴다.**"))
    for b in bad[:5]:
        P("  - %s %s: %.4f vs %.4f" % b)
    P("")
    if bad:
        io.open(OUT / "D12_HUMAN.md", "w", encoding="utf-8").write("\n".join(L))
        print("전제 범위 불일치 %d건 — 중단한다." % len(bad))
        return 1

    # ---------------- §A D12 재집계 ----------------------------------------
    grp = {
        "N층 사람=지지 (회수 대상)":
            [i for i in ov if i["judge"]["stratum"] == "N" and i["human"] == POS],
        "N층 사람=불지지 (오회수 대조)":
            [i for i in ov if i["judge"]["stratum"] == "N" and i["human"] == NEG],
        "G층 사람=지지 (유지 대조)":
            [i for i in ov if i["judge"]["stratum"] == "G" and i["human"] == POS],
        "M층 사람=지지 (참고)":
            [i for i in ov if i["judge"]["stratum"] == "M" and i["human"] == POS],
    }
    P("## A. D12 회수표 — 사람 라벨 (V′ ∩ D12 모집단, %d건)" % len(ov))
    P("")
    P("D12 원표는 `judge_label` 로 층 안의 지지/불지지를 갈랐다. 그 라벨만 사람 것으로 "
      "바꾼다. 층 자체(G/N/M)는 현행 추정자 점수로 정의되므로 그대로다.")
    P("")
    P("| 체크포인트 | " + " | ".join("%s n=%d" % (g, len(v)) for g, v in grp.items()) + " |")
    P("|---" * (len(grp) + 1) + "|")
    rows_a = []
    for ck in cks + ["MiniCheck (정답키 lab_mc)"]:
        line, rec = ["| " + ck.split("/")[-1]], {"checkpoint": ck}
        for g, v in grp.items():
            if not v:
                line.append("—")
                continue
            if ck.startswith("MiniCheck"):
                hit = sum(1 for i in v if i["key"]["lab_mc"] == "1")
            else:
                hit = sum(1 for i in v if float(i["d12"][ck]["new_max_p"]) >= TAU)
            line.append(cell(hit, len(v)))
            rec[g] = "%d/%d" % (hit, len(v))
        P(" | ".join(line) + " |")
        rows_a.append(rec)
    P("")
    P("**이 표로는 아무것도 결정하지 못한다.** 회수 대상 칸이 **n=%d** 이라 Wilson 구간이 "
      "표 전체를 덮는다. D12 원표(판정자 기준)의 같은 칸은 n=29 였다 — 사람 라벨 60건이 "
      "D8-B 표본을 겨냥해 뽑힌 것이 아니라 우연히 겹쳤기 때문이다."
      % len(grp["N층 사람=지지 (회수 대상)"]))
    P("")

    # ---------------- §B D8-B 표8 재집계 ------------------------------------
    jov = [i for i in items if i["judge"]]
    P("## B. D8-B 표8 — 사람 라벨 (V′ ∩ D8-B 표본, %d건)" % len(jov))
    P("")
    P("| 층 | 정의 | n | 사람 '지지' | 판정자 기준 원값 |")
    P("|---|---|---|---|---|")
    rows_b = []
    orig = {"G": "precision 0.913", "N": "FNR 0.725 (= 지지 72.5%)", "M": "기록만"}
    for st, dfn in [("G", "max_p ≥ 0.5"), ("N", "max_p < 0.05"), ("M", "0.05 ≤ max_p < 0.5")]:
        sub = [i for i in jov if i["judge"]["stratum"] == st]
        k = sum(1 for i in sub if i["human"] == POS)
        P("| %s | `%s` | %d | %s | %s |" % (st, dfn, len(sub), cell(k, len(sub)), orig[st]))
        rows_b.append({"stratum": st, "n": len(sub), "human_supported": k})
    P("")
    P("**G층은 사람 기준으로도 높다** — 판정자 0.913 대 사람 위 표. "
      "N층은 n 이 작아 참고값이다. 이 %d건은 D8-B 게이트가 지정한 25건 표본이 아니라 "
      "V′ 60건과 우연히 겹친 것이고 층 비율도 다르다." % len(jov))
    P("")

    # ---------------- §C V′ 자체 회수표 -------------------------------------
    miss = [i for i in items if i["key"]["lab_g1"] == "0"]
    tgt = [i for i in miss if i["human"] == POS]
    fls = [i for i in miss if i["human"] == NEG]
    keep = [i for i in items if i["key"]["lab_g1"] == "1" and i["human"] == POS]
    P("## C. V′ 자체 회수표 — 같은 질문, 분모가 크다 (권장)")
    P("")
    P("A 의 회수 칸이 n=%d 인 이유는 D8-B 표본과의 겹침에 기댔기 때문이다. 그런데 "
      "**같은 질문을 V′ 60건 전체에서 물을 수 있다** — \"현행이 미지지라 한 것 중 사람이 "
      "지지라 한 문장을, 후보는 되찾는가\". 분모가 %d 가 된다."
      % (len(grp["N층 사람=지지 (회수 대상)"]), len(tgt)))
    P("")
    P("| 후보 | 회수 (n=%d) | **오회수** (n=%d) | 유지 (n=%d) |" % (len(tgt), len(fls), len(keep)))
    P("|---|---|---|---|")
    rows_c = []
    for est, col in [("현행 g₁", "lab_g1"), ("MiniCheck", "lab_mc"), ("bart", "lab_bart")]:
        v = [sum(1 for i in s if i["key"][col] == "1") for s in (tgt, fls, keep)]
        P("| %s | %s | %s | %s |" % (est, cell(v[0], len(tgt)),
                                     cell(v[1], len(fls)), cell(v[2], len(keep))))
        rows_c.append({"estimator": est, "recovery_k": v[0], "recovery_n": len(tgt),
                       "false_recovery_k": v[1], "false_recovery_n": len(fls),
                       "retention_k": v[2], "retention_n": len(keep)})
    P("")
    P("**오회수 칸을 회수 칸 옆에 둔다.** 회수만 보면 관대한 모델이 무조건 이긴다. "
      "스펙 §0 이 \"LFM 에서 위험한 오류는 잘못된 FREEZE\" 라고 못박았으므로 "
      "정밀도 쪽이 판정축이다.")
    P("")

    # ---------------- §D 4칸 + like-for-like --------------------------------
    P("## D. V′ 4칸 사람 지지율")
    P("")
    P("| 칸 | 정의 | n | 사람 '지지' |")
    P("|---|---|---|---|")
    rows_d = []
    dfns = {"P1": "MiniCheck 지지 · g₁ 지지", "P2": "MiniCheck 지지 · g₁ 미지지",
            "N1": "MiniCheck 미지지 · g₁ 미지지", "N2": "MiniCheck 미지지 · g₁ 지지"}
    for c in ("P1", "P2", "N1", "N2"):
        sub = [i for i in items if i["cell"] == c]
        k = sum(1 for i in sub if i["human"] == POS)
        P("| %s | %s | %d | %s |" % (c, dfns[c], len(sub), cell(k, len(sub))))
        rows_d.append({"cell": c, "n": len(sub), "human_supported": k})
    P("")
    n1 = [i for i in items if i["cell"] == "N1"]
    n1k = sum(1 for i in n1 if i["human"] == POS)
    P("**N1 이 핵심이다.** 두 추정자가 **함께** 미지지라 한 칸인데 사람은 %d/%d 를 "
      "지지라 한다. MiniCheck 도 아직 많이 놓친다 — 논문에 그대로 넣는다." % (n1k, len(n1)))
    P("")
    P("### D-2. V′ 회수 분모는 D8-B 의 N층과 같은 것인가 — **아니다**")
    P("")
    P("V′ 의 \"현행 미지지\" 는 `p_g1 < 0.5` 다. D8-B 의 N층은 `p_g1 < 0.05` 이고 그 "
      "사이 구간은 M층이다. 회수 대상 %d건을 D8-B 규칙으로 다시 가르면:" % len(tgt))
    P("")
    P("| D8-B 규칙 | n | 비율 |")
    P("|---|---|---|")
    cnt = {"N": 0, "M": 0, "G": 0}
    for i in tgt:
        cnt[stratum_of(i["p_g1"])] += 1
    for st in ("N", "M", "G"):
        P("| %s | %d | %.1f%% |" % (st, cnt[st], 100.0 * cnt[st] / len(tgt)))
    P("")
    P("**%s** C 의 회수 수치를 D12 의 N층 행과 나란히 놓을 때는 이 차이를 적어야 한다."
      % ("대부분이 N층이라 사실상 같은 것을 재고 있다."
         if cnt["N"] >= 0.8 * len(tgt) else
         "M층이 %.0f%% 섞여 있어 D12 의 N층 행과 같은 것을 재는 것이 아니다."
         % (100.0 * cnt["M"] / len(tgt))))
    P("")

    io.open(OUT / "D12_HUMAN.md", "w", encoding="utf-8").write("\n".join(L))
    for name, d_ in [("d12_human_A.csv", rows_a), ("d8b_human_B.csv", rows_b),
                     ("vprime_recovery_C.csv", rows_c), ("vprime_cells_D.csv", rows_d)]:
        keys = sorted({k for r in d_ for k in r})
        with io.open(OUT / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(d_)
    io.open(OUT / "d12_human.json", "w", encoding="utf-8").write(json.dumps(
        {"n_overlap_d12": len(ov), "n_overlap_d8b": len(jov),
         "recovery_n": len(tgt), "false_recovery_n": len(fls), "retention_n": len(keep),
         "recovery_stratified_by_d8b_rule": cnt,
         "A": rows_a, "B": rows_b, "C": rows_c, "D": rows_d}, ensure_ascii=False, indent=1))

    print("\n".join(L))
    print("-> %s" % (OUT / "D12_HUMAN.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
