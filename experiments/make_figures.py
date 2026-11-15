"""Fig.1 · Fig.4 생성. 300dpi, 흑백 인쇄에서 구분되게. 호출 0.

공저자 지시:

  Fig.1  M1-C 로 그린다. 두 열(g1 / 확정 추정자) x tau 3점, 세로축은 sufficient
         branch 비율, **막대 위에 실제 개수**(1, 19 같은 것). 논문 헤드라인이
         "1에서 19" 라 그 두 숫자가 그림에서 바로 읽혀야 한다. 캡션에 잠정 표기.
  Fig.4  한 질의를 고르되 **가장 극적인 것 말고 중앙값에 가까운 것**으로.
         얼린 branch 수가 전체 중앙값에 가까운 질의. **어떤 질의를 왜 골랐는지
         캡션에 적으면 체리피킹 시비가 안 붙는다.**

Fig.2 아키텍처는 공저자가 만든다. Fig.3 Pareto 는 Test B 가 없어 목록에서 빠졌다.

**흑백 대응**: 색만으로 구분하지 않는다. 해칭(빗금)과 테두리로도 갈라 회색조 인쇄에서
읽히게 한다. 폰트는 샤딩 논문 규격(본문 9pt, 축 8pt), 300dpi.

숫자는 전부 원자료에서 다시 센다. 그림에 들어간 값은 `figs_data.json` 으로 같이
떨궈서 `verify_manuscript` 가 대조할 수 있게 한다.
"""

from __future__ import annotations

import csv
import io
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import Patch     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figs"
TAUS = (0.7, 0.8, 0.9)
DPI = 300

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})


def rd(p: str):
    q = ROOT / p
    return list(csv.DictReader(io.open(q, encoding="utf-8"))) if q.exists() else []


def fig1() -> dict:
    """M1-C — 충분 branch 비율. 두 추정자 x tau 3점."""
    mc = rd("results/m1c/m1c_branches.csv")
    ta = [r for r in rd("results/test_a/20260829-041445/waste_ratio.csv")
          if int(r["round"]) == 0]
    common = {r["query_id"] for r in mc} & {r["query_id"] for r in ta}
    mc = [r for r in mc if r["query_id"] in common]
    ta = [r for r in ta if r["query_id"] in common]
    n = len(mc)
    assert n and len(ta) == n, "분모가 안 맞는다: mc=%d ta=%d" % (n, len(ta))

    def cnt(rows, tau):
        return sum(1 for r in rows
                   if float(r["coverage"]) == 1.0 and float(r["grounding"]) > tau)

    g1 = [cnt(ta, t) for t in TAUS]
    acc = [cnt(mc, t) for t in TAUS]

    x = list(range(len(TAUS)))
    w = 0.36
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    b1 = ax.bar([i - w / 2 for i in x], [100.0 * v / n for v in g1], w,
                label="Standard operator ($g_1$)",
                facecolor="white", edgecolor="black", hatch="////", linewidth=0.9)
    b2 = ax.bar([i + w / 2 for i in x], [100.0 * v / n for v in acc], w,
                label="Accepted operator", facecolor="0.45", edgecolor="black",
                linewidth=0.9)
    for bars, vals in ((b1, g1), (b2, acc)):
        for r, v in zip(bars, vals):
            ax.annotate(str(v), (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=8)
    ax.set_xticks(x, [r"$\tau$ = %.1f" % t for t in TAUS])
    ax.set_ylabel("Sufficient branches (%%) of %d" % n)
    ax.set_ylim(0, max(100.0 * v / n for v in acc) * 1.35)
    ax.legend(frameon=False, loc="upper right")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / ("fig1_sufficient.%s" % ext), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return {"n_branches": n, "n_queries": len(common),
            "g1": dict(zip([str(t) for t in TAUS], g1)),
            "accepted": dict(zip([str(t) for t in TAUS], acc))}


def fig4() -> dict:
    """궤적 예시. **2단으로 만든다.**

    공저자 규칙은 "얼린 branch 수가 중앙값에 가까운 질의" 였는데, 실측하니
    **중앙값이 0** 이다 — 45질의 중 23개가 하나도 못 얼린다. 그 규칙을 그대로 쓰면
    아무 차이도 안 보이는 그림이 된다.

    규칙을 몰래 바꾸지 않는다. 대신 **(a) 분포 전체**를 먼저 보이고,
    **(b) 0 이 아닌 질의들의 중앙값**에 해당하는 궤적을 예시로 붙인다.
    (a) 가 있으면 (b) 가 대표성을 주장하지 않아도 되므로 체리피킹 시비가 원천 차단된다.
    최종 형태는 공저자가 정한다.
    """
    live = rd("results/m1c_singledoc/m2min_live.csv")  # corrected single-document operator (results/m1c is the chunk-split version)
    assert live, "m2min_live.csv 가 없다"
    fr = [int(r["B_frozen"]) for r in live]
    med_all = st.median(fr)
    nz = [r for r in live if int(r["B_frozen"]) > 0]
    med_nz = st.median([int(r["B_frozen"]) for r in nz]) if nz else 0
    pick = sorted(nz, key=lambda r: (abs(int(r["B_frozen"]) - med_nz), r["query_id"]))[0]
    qid, nb, nfz = pick["query_id"], int(pick["n_branches"]), int(pick["B_frozen"])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.9, 2.3),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) 분포 — 몇 개 질의가 몇 개를 얼렸는가
    from collections import Counter
    c = Counter(fr)
    ks = sorted(c)
    axa.bar(ks, [c[k] for k in ks], 0.6, facecolor="0.45",
            edgecolor="black", linewidth=0.9)
    for k in ks:
        axa.annotate(str(c[k]), (k, c[k]), textcoords="offset points",
                     xytext=(0, 2), ha="center", fontsize=8)
    axa.set_xticks(ks)
    axa.set_xlabel("Branches frozen in a query")
    axa.set_ylabel("Queries (of %d)" % len(live))
    axa.set_ylim(0, max(c.values()) * 1.2)
    axa.set_title("(a) Distribution", fontsize=9, loc="left")

    # (b) 대표 궤적
    rounds = [0, 1, 2]
    a_exec = [nb] * 3
    b_exec = [nb, nb - nfz, nb - nfz]
    axb.step(rounds, a_exec, where="mid", color="black", linewidth=1.2,
             linestyle="--", label="Global retry")
    axb.step(rounds, b_exec, where="mid", color="black", linewidth=1.8,
             label="Freeze + branch re-run")
    axb.fill_between(rounds, b_exec, a_exec, step="mid", facecolor="0.85",
                     edgecolor="none")
    axb.set_xticks(rounds, ["R0", "R1", "R2"])
    axb.set_yticks(range(0, nb + 1))
    axb.set_ylabel("Branches executed")
    axb.set_ylim(-0.2, nb + 0.7)
    axb.legend(frameon=False, loc="lower left")
    axb.annotate("%d frozen" % nfz, (1, b_exec[1]), textcoords="offset points",
                 xytext=(4, 6), fontsize=8)
    axb.set_title("(b) Query %s" % qid, fontsize=9, loc="left")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / ("fig4_trajectory.%s" % ext), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return {"query_id": qid, "n_branches": nb, "frozen": nfz,
            "median_frozen_all": med_all, "median_frozen_nonzero": med_nz,
            "queries_freezing_none": sum(1 for x in fr if x == 0),
            "queries_freezing_any": len(nz), "n_queries": len(live),
            "distribution": {str(k): c[k] for k in ks},
            "selection_rule": ("전 질의 중앙값이 0 이라 그 규칙으로는 빈 그림이 된다. "
                               "(a) 에 분포 전체를 싣고 (b) 는 0 이 아닌 질의의 "
                               "중앙값에 해당하는 것을 골랐다. 동률이면 query_id 사전순."),
            "A_calls": int(pick["A_calls"]), "B_calls": int(pick["B_calls"])}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"provisional": True,
            "note": "추정자는 사람 라벨 전이라 잠정. 라벨 확정 후 같은 스크립트로 재생성.",
            "dpi": DPI}
    print("Fig.1 — 충분 branch 비율 ...")
    data["fig1"] = fig1()
    d = data["fig1"]
    print("  branch %d (질의 %d)" % (d["n_branches"], d["n_queries"]))
    for t in TAUS:
        print("    tau=%.1f  g1 %2d  확정 %2d" % (t, d["g1"][str(t)], d["accepted"][str(t)]))

    print("Fig.4 — 궤적 예시 ...")
    data["fig4"] = fig4()
    e = data["fig4"]
    print("  분포: 하나도 못 얼린 질의 %d / %d · 하나라도 얼린 질의 %d"
          % (e["queries_freezing_none"], e["n_queries"], e["queries_freezing_any"]))
    print("  전 질의 중앙값 %g · 0 이 아닌 것들의 중앙값 %g"
          % (e["median_frozen_all"], e["median_frozen_nonzero"]))
    print("  (b) 예시 질의 %s — branch %d, 얼림 %d, 호출 %d -> %d"
          % (e["query_id"], e["n_branches"], e["frozen"], e["A_calls"], e["B_calls"]))

    io.open(OUT / "figs_data.json", "w", encoding="utf-8").write(
        json.dumps(data, ensure_ascii=False, indent=1))

    print("")
    print("캡션 초안 (공저자가 확정):")
    print("  Fig.1  Sufficient branches at round 0 under the standard and accepted")
    print("         operators, %d branches from %d queries. Counts above bars."
          % (d["n_branches"], d["n_queries"]))
    print("         Operator provisional pending human labels.")
    print("  Fig.4  (a) Number of branches frozen per query across all %d queries."
          % e["n_queries"])
    print("         (b) Branch executions per round for query %s. The median over all"
          % e["query_id"])
    print("         queries is %g — %d of %d queries freeze nothing — so panel (b) shows"
          % (e["median_frozen_all"], e["queries_freezing_none"], e["n_queries"]))
    print("         the median among queries that freeze at least one branch (%g)."
          % e["median_frozen_nonzero"])
    print("         Panel (a) carries the representativeness; (b) is illustrative only.")
    print("")
    print("-> %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
