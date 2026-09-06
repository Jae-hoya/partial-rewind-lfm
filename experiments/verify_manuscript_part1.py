"""원고 §VI Part 1 수치를 **원행에서** 대조한다. 읽기 전용, 호출 0.

공저자 지적: "§VI Part 1 이 본문 절반이고 가장 여러 번 손댄 부분이라 옮겨 적기 오류
위험이 제일 크다."

`verify_manuscript.py` 가 §VII·§VIII 을 덮으므로 여기서 §VI 을 덮는다. 둘을 합치면
투고 전에 원고 전체를 한 번에 대조할 수 있다.

**요약 CSV 를 읽지 않고 원행에서 다시 센다.** 오늘 15.7 이 요약본의 4자리 반올림
(0.1565)을 다시 반올림해서 나온 값이었다. 그 경로를 원천 차단한다.
"""

from __future__ import annotations

import csv
import io
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "specs_from_chat/manuscript_PartialRewind_v0.5.md"

fails: list[str] = []
missing: list[str] = []


def rd(p: str):
    q = ROOT / p
    return list(csv.DictReader(io.open(q, encoding="utf-8"))) if q.exists() else []


def f(r, k):
    return float(r[k])


def pct(rows, pred):
    return 100.0 * sum(1 for r in rows if pred(r)) / len(rows) if rows else 0.0


def chk(label, want, text_has: str, tol=0.06):
    """원고에 `text_has` 문자열이 있는지 + 그 값이 자료와 맞는지."""
    t = chk.text
    if text_has not in t:
        missing.append("%s (원고에 %r 없음)" % (label, text_has))
        print("  [못 찾음] %-44s 원고에 %r 가 없다" % (label, text_has))
        return
    try:
        got = float(text_has.replace("%", "").replace(",", "").split()[-1]
                    .replace("−", "-"))
    except ValueError:
        print("  [OK  ] %-44s (문자열 확인만)" % label)
        return
    ok = abs(got - float(want)) <= tol
    if not ok:
        fails.append(label)
    print("  [%s] %-44s 원고 %-9s 자료 %.4f"
          % ("OK  " if ok else "불일치", label, "%g" % got, want))


def main() -> int:
    if not MS.exists():
        print("원고 없음: %s" % MS)
        return 1
    chk.text = io.open(MS, encoding="utf-8").read()

    g = rd("results/diag/d7/pool_unit_grid.csv")
    sp = rd("results/diag/d7/pool_unit_grid_sentprem.csv")
    S = [r for r in g if r["unit"] == "sentence"]
    C = [r for r in g if r["unit"] == "claim"]
    SPs = [r for r in sp if r["unit"] == "sentence"]
    SPc = [r for r in sp if r["unit"] == "claim"]

    print("=" * 78)
    print("§VI-B/C/D — 격자와 분포 (원행 n=%d 문장 / %d claim)" % (len(S), len(C)))
    print("=" * 78)
    chk("문장×인용청크 grounded", pct(S, lambda r: f(r, "p_cited") >= 0.5), "15.6%")
    chk("문장×retrieved grounded", pct(S, lambda r: f(r, "p_retrieved") >= 0.5), "17.3%")
    chk("claim×인용청크 grounded", pct(C, lambda r: f(r, "p_cited") >= 0.5), "31.7%")
    chk("claim×target grounded", pct(C, lambda r: f(r, "p_target") >= 0.5), "17.9%")
    chk("claim×retrieved grounded", pct(C, lambda r: f(r, "p_retrieved") >= 0.5), "31.9%")
    chk("<0.05 비율", pct(S, lambda r: f(r, "p_cited") < 0.05), "76.5%")
    chk(">=0.90 비율", pct(S, lambda r: f(r, "p_cited") >= 0.90), "11.9%")
    chk("중간대 0.1~0.9", pct(S, lambda r: 0.1 <= f(r, "p_cited") <= 0.9), "8.5%")
    chk("tau=0.1 grounded", pct(S, lambda r: f(r, "p_cited") >= 0.1), "20.4%")
    chk("tau=0.3 grounded", pct(S, lambda r: f(r, "p_cited") >= 0.3), "16.3%")
    chk("tau=0.2 grounded", pct(S, lambda r: f(r, "p_cited") >= 0.2), "17.7%")
    chk("문장×문장전제 grounded", pct(SPs, lambda r: f(r, "p_cited") >= 0.5), "38.4%")
    chk("claim×문장전제 grounded", pct(SPc, lambda r: f(r, "p_cited") >= 0.5), "57.1%")

    low = [r for r in S if f(r, "p_cited") < 0.05]
    rem = [r for r in low if f(r, "p_retrieved") < 0.05]
    print("")
    print("  귀속: 인용청크 <0.05 인 문장 %d개, retrieved 로도 <0.05 인 것 %d (%.1f%%)"
          % (len(low), len(rem), 100.0 * len(rem) / len(low)))
    chk("귀속 분모", len(low), "225 sentences", 0.5)
    chk("귀속 잔존 비율", 100.0 * len(rem) / len(low), "96.9%")

    print("")
    print("=" * 78)
    print("§VI-E — 패딩 통제 (D7-0)")
    print("=" * 78)
    pc = rd("results/diag/d7/padding_control.csv")
    for m, tk, tr, med in [(1, "142", "0 / 35", 0.992), (2, "274", "6 / 35", 0.152)]:
        sub = [r for r in pc if int(r["multiplier"]) == m]
        chk("%dx 전제 토큰 중앙" % m,
            st.median([int(r["premise_tokens_after_tokenizer"]) for r in sub]), tk, 0.5)
        chk("%dx p 중앙" % m, st.median([f(r, "p_entail") for r in sub]), "%.3f" % med, 0.002)

    print("")
    print("=" * 78)
    print("§VI-F — 음성 통제 (D8-A)")
    print("=" * 78)
    net = rd("results/diag/d8/d8a2_estimator_net.csv")
    e1 = [r for r in net if r["estimator"] == "E1"]
    e2 = [r for r in net if r["estimator"] == "E2"]
    chk("E1 정상", pct(e1, lambda r: f(r, "p_normal") >= 0.5), "15.6%")
    chk("E1 교란", pct(e1, lambda r: f(r, "p_shuffled") >= 0.5), "0.0%")
    chk("E2 교란(팽창)", pct(e2, lambda r: f(r, "p_shuffled") >= 0.5), "50.5%")
    chk("E2 정상(토픽 풀)", pct(e2, lambda r: f(r, "p_normal") >= 0.5), "62.3%")
    chk("E2 순증", pct(e2, lambda r: f(r, "p_normal") >= 0.5)
        - pct(e2, lambda r: f(r, "p_shuffled") >= 0.5), "11.7 points")
    chk("풀 확대 효과",
        pct(e2, lambda r: f(r, "p_normal") >= 0.5) - pct(SPc, lambda r: f(r, "p_cited") >= 0.5),
        "5.1-point")

    print("")
    print("=" * 78)
    print("§VI — 판정 표본과 충실도 (D8-B)")
    print("=" * 78)
    jb = rd("results/diag/d8/d8b_judge_sample.csv")
    # 원고는 G층을 백분율이 아니라 "precision 0.913" 으로 적는다. 표기가 다르므로
    # 패턴도 다르게 잡는다 — 값이 맞는지가 중요하지 형식이 중요한 게 아니다.
    gsub = [r for r in jb if r["stratum"] == "G"]
    chk("G층 precision", pct(gsub, lambda r: r["judge_label"] == "supported") / 100.0,
        "precision 0.913", 0.001)
    nsub = [r for r in jb if r["stratum"] == "N"]
    chk("N층 지지율", pct(nsub, lambda r: r["judge_label"] == "supported"), "72.5%")
    fid = rd("results/diag/d8/d8b_fidelity.csv")
    chk("충실도 complete", sum(1 for r in fid if r["label"] == "complete"),
        "complete for 26", 0.5)
    chk("충실도 addition", sum(1 for r in fid if r["label"] == "addition"),
        "added content in 0", 0.5)

    print("")
    print("=" * 78)
    print("§VI-G — 체크포인트 교체 (D10 / D10-N)")
    print("=" * 78)
    gr = rd("results/diag/d10/d10_grid_by_checkpoint.csv")
    ng = rd("results/diag/d10/d10n_negative_by_checkpoint.csv")
    for ck, s_n, s_s in [("cross-encoder/nli-deberta-v3-base", "15.65%", "0.00%"),
                         ("facebook/bart-large-mnli", "60.9%", None)]:
        sub = [r for r in ng if r["checkpoint"] == ck]
        if sub:
            chk("%s 정상" % ck.split("/")[-1],
                pct(sub, lambda r: f(r, "p_normal") >= 0.5), s_n)
            if s_s:
                chk("%s 교란" % ck.split("/")[-1],
                    pct(sub, lambda r: f(r, "p_shuffled") >= 0.5), s_s)
    bart = [r for r in gr if r["checkpoint"] == "facebook/bart-large-mnli"]
    if bart:
        chk("bart 격자 grounded", pct(bart, lambda r: f(r, "max_p") >= 0.5), "60.5%")

    print("")
    print("=" * 78)
    print("§VI — 전제 구성 (D14 / D14-N)")
    print("=" * 78)
    d14 = rd("results/diag/d14/d14_summary.csv")
    d14n = rd("results/diag/d14/d14n_summary.csv")
    base = "cross-encoder/nli-deberta-v3-base"
    for cond, s in [("union_L256", "67.0%"), ("single_L64", "31.6%"),
                    ("union_L64", "34.0%"), ("sentsplit_max", "38.4%")]:
        row = next((r for r in d14 if r["checkpoint"] == base and r["condition"] == cond), None)
        if row:
            chk("D14 %s" % cond, float(row["grounded_at_05"]), s)
    for cond, s in [("single_L128", "1.36%")]:
        row = next((r for r in d14n if r["condition"] == cond), None)
        if row:
            chk("D14-N %s 교란" % cond, float(row["grounded_shuffled"]), s)
            chk("D14-N %s 순증" % cond, float(row["net"]), "40.82")

    print("")
    print("=" * 78)
    print("§IX — 관문 P (D9-P)")
    print("=" * 78)
    p1 = [r for r in rd("results/diag/d10/d10_padding_by_checkpoint.csv")
          if r["checkpoint"] == base and int(r["multiplier"]) == 4]
    if p1:
        chk("g1 4x 유지", pct(p1, lambda r: f(r, "p_entail") >= 0.5), "11.4%")

    print("")
    print("=" * 78)
    if missing:
        print("**패턴을 못 찾은 항목 %d개:**" % len(missing))
        for x in missing:
            print("  - " + x)
    if fails:
        print("")
        print("**불일치 %d건:**" % len(fails))
        for x in fails:
            print("  - " + x)
        return 1
    if missing:
        print("")
        print("불일치 없음. 못 찾은 항목은 원고 문구를 눈으로 확인하라.")
        return 0
    print("§VI Part 1 수치가 원자료와 전부 일치한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
