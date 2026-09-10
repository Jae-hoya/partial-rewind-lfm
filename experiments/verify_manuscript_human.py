"""원고의 **사람 라벨 기준 수치**를 원행에서 다시 세어 대조한다. 호출 0.

왜 따로 만드는가: `verify_manuscript.py` 는 v0.5 시점의 판정자 기준 수치를 본다.
v0.6~v0.7 에서 그 자리에 **사람 라벨 기준 값**이 들어갔는데, 그 값들은 내가 채팅으로
타이핑해 보낸 것을 공저자가 옮겨 적은 것이다. **15.6% 가 15.7% 가 된 경로가 정확히
그것이다.** 그러니 옮겨 적힌 값을 원자료에서 다시 세어 대조한다.

**요약 파일을 읽지 않는다.** `labels_stage1.csv`(사람 라벨) + `_key_stage1.csv`(추정자
라벨) + `d8b_judge_sample.csv`(판정자 라벨)에서 매번 다시 센다. `d12_human.json` 은
내가 만든 산출물이라 그것과 맞춰 봐야 아무것도 검증되지 않는다.

%p 차이는 전부 원행 비율끼리 뺀다. 반올림끼리 빼지 않는다.
"""

from __future__ import annotations

import csv
import io
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "specs_from_chat/manuscript_PartialRewind_v1.0.md"
POS, NEG = "지지", "불지지"

FAIL = 0
MISS = 0


def rd(p, enc="utf-8"):
    return list(csv.DictReader(io.open(ROOT / p, encoding=enc)))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * max(0.0, (c - h) / d), 100 * min(1.0, (c + h) / d))


def chk(label, got, want, tol=0.05):
    """실측 got 이 원고에 적힌 want 와 tol 안에서 같은가."""
    global FAIL
    ok = abs(got - want) <= tol
    FAIL += 0 if ok else 1
    print("  [%s] %-46s 원고 %-10s 자료 %.4f"
          % ("OK  " if ok else "불일치", label, ("%g" % want), got))


def has(label, pattern, text):
    """원고에 이 문자열이 실제로 있는가. 없으면 '못 찾음' 으로 보고한다 — 조용히 넘기지 않는다."""
    global MISS
    ok = re.search(pattern, text) is not None
    MISS += 0 if ok else 1
    print("  [%s] %s" % ("OK  " if ok else "못찾음", label))
    return ok


def main() -> int:
    if not MS.exists():
        print("원고를 찾지 못했다: %s" % MS)
        return 1
    T = re.sub(r"\s+", " ", io.open(MS, encoding="utf-8").read())

    lab = rd("results/vprime/labels_stage1.csv", "utf-8-sig")
    key = {r["item_id"]: r for r in rd("results/vprime/_key_stage1.csv", "utf-8-sig")}
    jb = {r["sentence"].strip(): r for r in rd("results/diag/d8/d8b_judge_sample.csv")}
    # **MiniCheck 라벨은 정답키가 아니라 재채점 결과에서 온다.** 정답키의 lab_mc 는
    # 쪼개기 버그가 있던 채점(294중 189건 분할)이고, v1.0 이 싣는 값은 단일 문서
    # 재채점이다. g1·bart 는 전제 처리가 안 바뀌어 정답키 그대로 쓴다.
    mcnew = {r["item_id"]: float(r["p_mc_new"]) > 0.5
             for r in rd("results/vprime/rescore_singledoc_rows.csv")}

    it = []
    for x in lab:
        h = x["human_label"].strip()
        k = key.get(x["item_id"])
        if h in (POS, NEG) and k and x["item_id"] in mcnew:
            k = dict(k); k["lab_mc"] = "1" if mcnew[x["item_id"]] else "0"
            it.append({"h": h, "k": k, "j": jb.get(x["문장"].strip())})

    print("=" * 78)
    print("§VI-E  V′ 관문 — 사람 라벨 60건 (원행 재집계)")
    print("=" * 78)
    print("  분모: 라벨 %d건 중 지지/불지지 %d건 (판단불가·빈칸 제외)" % (len(lab), len(it)))

    sup = [i for i in it if i["h"] == POS]
    for est, col, pr, rc, nr in [("g1", "lab_g1", 0.800, 0.327, 0.000),
                                 ("MiniCheck", "lab_mc", 0.864, 0.776, 0.758),
                                 ("bart", "lab_bart", 0.795, 0.633, 0.545)]:
        pn = [i for i in it if i["k"][col] == "1"]
        pk = sum(1 for i in pn if i["h"] == POS)
        rk = sum(1 for i in sup if i["k"][col] == "1")
        tgt = [i for i in sup if i["k"]["lab_g1"] == "0"]
        nk = sum(1 for i in tgt if i["k"][col] == "1")
        print("  -- %s  (precision %d/%d · recall %d/%d · N층 회수 %d/%d)"
              % (est, pk, len(pn), rk, len(sup), nk, len(tgt)))
        chk("%s precision" % est, pk / len(pn), pr, 0.0006)
        chk("%s recall" % est, rk / len(sup), rc, 0.0006)
        chk("%s N층 회수" % est, nk / len(tgt), nr, 0.0006)

    # v1.0 에서 미결 사유가 **recall 미달 -> precision 하한 미달**로 바뀌었다.
    mk = sum(1 for i in sup if i["k"]["lab_mc"] == "1")
    chk("MiniCheck recall 이 바 0.70 을 넘는다", mk / len(sup) - 0.70, 0.0755, 0.0006)
    pn2 = [i for i in it if i["k"]["lab_mc"] == "1"]
    pk2 = sum(1 for i in pn2 if i["h"] == POS)
    lo, hi = wilson(pk2, len(pn2))
    chk("precision Wilson 하한(%)", lo, 73.29, 0.05)
    chk("precision 하한이 바-0.10=75 에 미달하는 폭(%p)", 75.0 - lo, 1.71, 0.05)

    print("")
    print("=" * 78)
    print("§VI-F  판정자 검증 (사람과 겹치는 문장)")
    print("=" * 78)
    ov = [i for i in it if i["j"]]
    MAP = {POS: "supported", NEG: "unsupported"}
    ag = sum(1 for i in ov if i["j"]["judge_label"] == MAP[i["h"]])
    print("  겹친 문장 %d건 · 일치 %d" % (len(ov), ag))
    chk("판정자-사람 일치율(%)", 100.0 * ag / len(ov), 65.6, 0.05)
    lo, hi = wilson(ag, len(ov))
    chk("일치율 Wilson 하한(%)", lo, 48.3, 0.05)
    chk("일치율 Wilson 상한(%)", hi, 79.6, 0.05)

    print("")
    print("=" * 78)
    print("부록  D8-B 층별 사람 지지율 · V′ 4칸 · 5추정자 회수표")
    print("=" * 78)
    for st, want_k, want_n in [("G", 16, 20), ("N", 9, 10)]:
        sub = [i for i in ov if i["j"]["stratum"] == st]
        k = sum(1 for i in sub if i["h"] == POS)
        chk("D8-B %s층 사람 지지 (분자)" % st, k, want_k, 0)
        chk("D8-B %s층 사람 지지 (분모)" % st, len(sub), want_n, 0)

    for c, want in [("P1", 80.0), ("P2", 88.0), ("N1", 73.3), ("N2", 80.0)]:
        sub = [i for i in it if i["k"]["cell"] == c]
        k = sum(1 for i in sub if i["h"] == POS)
        chk("V′ %s칸 사람 지지율(%%)" % c, 100.0 * k / len(sub), want, 0.05)

    miss = [i for i in it if i["k"]["lab_g1"] == "0"]
    tgt = [i for i in miss if i["h"] == POS]
    fls = [i for i in miss if i["h"] == NEG]
    keep = [i for i in it if i["k"]["lab_g1"] == "1" and i["h"] == POS]
    chk("회수 분모", len(tgt), 33, 0)
    chk("오회수 분모", len(fls), 7, 0)
    chk("유지 분모", len(keep), 16, 0)
    mrec = sum(1 for i in tgt if i["k"]["lab_mc"] == "1")
    mfls = sum(1 for i in fls if i["k"]["lab_mc"] == "1")
    mkep = sum(1 for i in keep if i["k"]["lab_mc"] == "1")
    chk("MiniCheck 오회수(%)", 100.0 * mfls / len(fls), 42.9, 0.05)
    chk("MiniCheck 유지 (분자)", mkep, 13, 0)
    chk("MiniCheck 순증(%p, 원행끼리)",
        100.0 * mrec / len(tgt) - 100.0 * mfls / len(fls), 32.9, 0.1)
    chk("현행 g1 유지 (분자)", sum(1 for i in keep if i["k"]["lab_g1"] == "1"), 16, 0)

    # like-for-like: 회수 대상 33건을 D8-B 규칙(N: p_g1<0.05)으로 다시 가른다
    n_true = sum(1 for i in tgt if float(i["k"]["p_g1"]) < 0.05)
    chk("회수 대상 중 D8-B N층", n_true, 30, 0)

    print("")
    print("=" * 78)
    print("원고 문장 — 숫자가 아니라 말로 적힌 주장")
    print("=" * 78)
    g1p = sum(1 for i in it if i["k"]["lab_g1"] == "1" and i["h"] == POS) \
        / sum(1 for i in it if i["k"]["lab_g1"] == "1")
    has("초록·결론이 'four times in five' 로 적었다 (실측 %.3f)" % g1p,
        r"right four times in five", T)
    has("'three times in four' 가 남아 있지 않다 (판정자 72.5% 근거였다)",
        r"^(?!.*three times in four).*$", T)
    has("결론이 '33 sentences' 를 사람 기준으로 인용한다",
        r"recovers none of the 33 sentences", T)
    has("기여 1이 '(0\\.80)' 과 'recall 0\\.33' 을 적었다",
        r"precise \(0\.80\).*recall 0\.33", T)
    has("§VI-E 가 미결(inconclusive)을 공개한다",
        r"inconclusive verdict", T)
    has("§VI-E 가 미결 사유를 precision 하한으로 적었다", r"0\.733|0\.73", T)
    has("§VIII-A 가 M1′ 실측 8.5% 를 적었다", r"8\.5%.*5\.5|W_\{0\.8\}.*8\.5", T)
    has("§V 가 청크 분할 버그를 적었다", r"chunk|aggregation freedom", T)
    has("'zero aggregation freedom' 주장이 철회됐다",
        r"^(?!.*only configuration in this paper with zero aggregation freedom).*$", T)
    has("§VI-F 판정자 절이 65.6% 와 게이트 미달을 적었다", r"65\.6", T)
    has("유지 칸(13/16)이 원고에 있다", r"13/16|13 of 16|13 \(81\.2", T)

    print("")
    print("=" * 78)
    if FAIL or MISS:
        print("**불일치 %d · 못 찾음 %d.** 원고의 사람 기준 수치를 고쳐야 한다." % (FAIL, MISS))
        return 1
    print("사람 기준 수치가 원행 재집계와 전부 일치한다. 문장 주장도 전부 찾았다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
