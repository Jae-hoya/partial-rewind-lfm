"""PREREG_R3 §4.6·§4.8 코퍼스2 분석 — `experiments/r2_analyze.py` 를 **고치지 않고** 그 `analyze()` 를 그대로 부른다. LLM 0.

PREREG_R3 §4.8 은 "r2_analyze.py 에 --root 인자만" 이라고 적었다. r2_analyze 는 이미 run_dir 을 위치 인자로 받으므로
계산 함수는 그대로 쓰고, 코퍼스에 묶인 다섯 군데만 이 래퍼에서 바꾼다 (모듈 속성 교체, 실행 뒤 원복).
  1. --rescore 의 연산자 설정과 청크 본문: config.yaml·data/index → config_c2.yaml·data_c2/index
  2. G4(= C-G4) 외삽 기본값: 35토픽, 15M/5.6M, 추정 9.9M/3.7M → 25토픽, 12.6M/3.4M, 추정 8.39M/2.22M (×25)
  3. G1 ②(= C-G1 ②): "분해 API 0" → 토픽마다 decompose(+repair) API ≤ 2, 첫 arm·seed 밖에서는 0
  4. PILOT_DEV(t000~t004, R2 의 30토픽 민감도) → 코퍼스2 에는 해당 없음: 빈 집합, sensitivity_30 블록은 지운다
  5. 코퍼스2 판정을 더한다: C-G2 퇴화, FREEZE 사건, assay(A0) 규칙, C-G6(a) 다리 조건, C-H1~C-H5 판정 단어,
     코퍼스1 정합(두 코퍼스 독립 부트스트랩의 차이 95% CI)
부트스트랩: topic × seed 2단 10,000회 seed 42 (r2_analyze 기본). 코퍼스1 정합의 코퍼스1 쪽 부트스트랩만 seed 43 —
같은 seed 를 두 코퍼스에 쓰면 두 난수열이 같은 비트에서 나와 상관된다(독립 부트스트랩이 아니게 된다).

  python experiments/c2_analyze.py results/c2/pilot3 --pilot --rescore
  python experiments/c2_analyze.py results/c2/heldout25 --rescore --corpus1 results/r2/heldout35 \
      --negctl results/c2/pilot3/NEGCTL_C2.json
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import r2_analyze as RA  # noqa: E402  (바꾸지 않는다)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_C2 = ROOT / "config_c2.yaml"
DELTA = 0.05
DEGENERATE = 0.05
C1_BOOT_SEED = 43
G4_C2 = dict(n_topics_main=25, n_seeds_main=3, limit_in=12_600_000, limit_out=3_400_000,
             est_in=8_390_000, est_out=2_220_000)


def g1_c2(orig, runs, usage, ledger):
    out = orig(runs, usage, ledger)
    qids = {q for recs in runs.values() for q in recs}
    dec_ledger = sum(1 for r in ledger if str(r.get("tag", "")).startswith("decompose"))
    keys = sorted(usage, key=lambda k: (k[1], RA.ORDER.index(RA.parse_arm(k[0])[0]) if RA.parse_arm(k[0])[0] in RA.ORDER else 99))
    first = keys[0] if keys else None
    by = {"%s·s%d" % k: int(v.get("decompose_api_calls", 0)) for k, v in usage.items()}
    later = {k: v for k, v in by.items() if first is None or k != "%s·s%d" % first}
    out["2_decompose_api_calls"] = {
        "rule": "C-G1 ②: 토픽마다 decompose + decompose_repair API ≤ 2, 그 토픽의 두 번째 arm 부터 0",
        "ledger_decompose_rows": dec_ledger, "n_topics": len(qids), "limit": 2 * len(qids),
        "first_arm_seed": ("%s·s%d" % first) if first else None, "usage_by_arm_seed": by,
        "pass": bool(qids) and dec_ledger <= 2 * len(qids) and all(v == 0 for v in later.values())
                and sum(by.values()) == dec_ledger}
    return out


def freeze_events(runs: dict) -> dict[str, int]:
    out = defaultdict(int)
    for (arm, _s), recs in runs.items():
        for rec in recs.values():
            for rr in rec["rounds"]:
                fr = (rr.get("action") or {}).get("params", {}).get("frozen") or []
                out[arm] += int(bool(fr))
    return dict(out)


def round0_rates(runs: dict, arm: str = "E-GR") -> dict[str, Any]:
    """C-G2: E-GR 라운드 0 answered(비기권) sub-answer 비율. seed 별과 합."""
    per, tot_a, tot_n = {}, 0, 0
    for (a, s), recs in sorted(runs.items()):
        if a != arm:
            continue
        na = n = 0
        for rec in recs.values():
            for br in rec["rounds"][0]["branches"]:
                for sa in br.get("sub_answers") or []:
                    n += 1
                    na += int(sa.get("status") == "answered")
        per["s%d" % s] = (na / n) if n else None
        tot_a += na; tot_n += n
    rate = (tot_a / tot_n) if tot_n else None
    return {"answered_rate_by_seed": per, "answered_rate_pooled": rate, "threshold": DEGENERATE,
            "degenerate": (rate is not None and rate < DEGENERATE)}


def _mat(per_query: dict, arm: str, metric: str, topics=None):
    keys = sorted(k for k in per_query if k.split("·")[0] == arm)
    seeds = [k.split("·s")[1] for k in keys]
    ts = sorted(topics if topics is not None else set.intersection(*[set(per_query[k]) for k in keys])) if keys else []
    return np.array([[per_query[k][t][metric] for k in keys] for t in ts], dtype=float), ts, seeds


def assay(per_query: dict, metric: str, n_boot: int, delta: float = DELTA) -> dict[str, Any]:
    """A0(전부 기권, coverage = grounding = 0) 가 같은 비열등 검정을 통과하지 못하려면 E-GR 값의 90% 하한 > δ."""
    A, ts, seeds = _mat(per_query, "E-GR", metric)
    if not ts:
        return {"available": False, "sensitive": False}
    boots = RA.boot_meandiff(A, np.zeros_like(A), n_boot)
    lo = float(np.percentile(boots, 5))
    return {"available": True, "metric": metric, "egr_mean": float(A.mean()), "egr_ci90_lo": lo, "delta": delta,
            "sensitive": bool(lo > delta), "n_topics": len(ts), "seeds": seeds}


def verdict_savings(s: dict, freeze_br: int | None = None) -> dict[str, Any]:
    if not s.get("available"):
        return {"verdict": "판정 불가", "reason": "짝 없음"}
    p = s["point"]
    if freeze_br == 0:
        return {"verdict": "재현 실패", "reason": "sufficient branch 없음 (FREEZE 0회, BR ≡ GR)", "point": p}
    if p is not None and p > 0 and s["claim"]:
        v = "재현"
    elif p is not None and p > 0:
        v = "방향만 일치"
    else:
        v = "재현 실패"
    return {"verdict": v, "point": p, "ci95": s["ci95"], "per_seed": s["per_seed"], "claim": s["claim"]}


def verdict_noninf(legs: dict[str, dict], assays: dict[str, dict], grounding_ok: bool, blocked: str | None) -> dict[str, Any]:
    if blocked:
        return {"verdict": "판정 불가", "reason": blocked}
    counted, detail = [], {}
    for m, s in legs.items():
        why = None
        if not s.get("available"):
            why = "짝 없음"
        elif not assays[m]["sensitive"]:
            why = "판정 불가(민감도 부족: E-GR 90%% 하한 %.3f ≤ δ)" % assays[m]["egr_ci90_lo"] if assays[m].get("available") else "assay 불가"
        elif m == "grounding" and not grounding_ok:
            why = "C-G6(a) 미통과 또는 미측정 — grounding 다리 제외 (operator not validated on this corpus)"
        detail[m] = {"counted": why is None, "reason": why, "diff": s.get("diff"), "ci90": s.get("ci90"),
                     "noninferior": s.get("noninferior")}
        if why is None:
            counted.append(m)
    if not counted:
        return {"verdict": "판정 불가", "reason": "센 다리 없음", "legs": detail}
    ok = all(legs[m]["noninferior"] for m in counted)
    return {"verdict": "재현" if ok else "재현 실패", "counted_legs": counted, "legs": detail}


def consistency(c2_pq: dict, c1_pq: dict, a: str, b: str, metric: str, kind: str, n_boot: int) -> dict[str, Any]:
    """(코퍼스2 − 코퍼스1) 95% CI 가 0 을 포함하고 점추정 방향이 같음 = 정합."""
    out = {}
    pts, boots = [], []
    for pq, seed in ((c2_pq, RA.BOOT_SEED), (c1_pq, C1_BOOT_SEED)):
        A, ts, _ = _mat(pq, a, metric)
        B, ts_b, _ = _mat(pq, b, metric, topics=ts)
        if not ts or A.shape != B.shape:
            return {"available": False}
        if kind == "savings":
            pts.append(float(1 - A.sum() / B.sum()))
            boots.append(RA.boot_savings(A, B, n_boot, seed))
        else:
            pts.append(float((A - B).mean()))
            boots.append(RA.boot_meandiff(A, B, n_boot, seed))
    d = boots[0] - boots[1]
    lo, hi = (float(x) for x in np.nanpercentile(d, [2.5, 97.5]))
    out.update({"available": True, "c2_point": pts[0], "c1_point": pts[1], "diff_point": pts[0] - pts[1],
                "diff_ci95": [lo, hi], "same_direction": bool(np.sign(pts[0]) == np.sign(pts[1])),
                "consistent": bool(lo <= 0 <= hi and np.sign(pts[0]) == np.sign(pts[1])),
                "boot_seeds": [RA.BOOT_SEED, C1_BOOT_SEED]})
    return out


def c2_verdicts(res: dict, raw: dict, *, negctl: dict | None, corpus1: dict | None, n_boot: int, pilot: bool) -> dict[str, Any]:
    runs = RA.with_r1(raw)
    fz = freeze_events(raw)
    g2 = round0_rates(raw)
    g3 = res.get("gates", {}).get("G3") or RA.g3_check(raw)
    sv = res["all_topics"]["savings"]
    ni = res["all_topics"]["noninferiority"]
    pq = res["per_query"]
    assays = {m: assay(pq, m, n_boot) for m in ("coverage", "grounding")}
    g6a = bool(negctl and negctl.get("offtopic", {}).get("pass"))
    rescored = res.get("sentence_support") == "minicheck_rescored"
    out: dict[str, Any] = {
        "C-G2": {**g2, "E_round0_legacy_coverage_by_seed": {k: v.get("mean_coverage") for k, v in res["arm_seed"].items() if k.startswith("E-R1")}},
        "freeze_events_by_arm": fz,
        "C-G3": {"pass": g3.get("pass"), **({} if "gates" in res else {"by_seed": g3.get("by_seed")})},
        "C-G6a": {"pass": g6a, "source": (negctl or {}).get("path"), "note": None if negctl else "--negctl 없음: grounding 다리 제외"},
        "assay": assays,
        "grounding_rescored": rescored,
    }
    if pilot:
        out["switch_rule_4_2_1"] = {
            "a_degenerate_E-GR_round0": g2["degenerate"], "freeze_L-BR": fz.get("L-BR", 0),
            "switch": bool(g2["degenerate"]),
            "note": "FREEZE 0회는 단독으로 전환을 일으키지 않는다 (결정). C-L 실패(b)는 실행 전 관문"}
        return out
    blocked = "퇴화 (E-GR 라운드 0 answered %.3f < 0.05, 세 seed 합)" % g2["answered_rate_pooled"] if g2["degenerate"] else None
    g3_block = None if g3.get("pass") else "C-G3 미통과 (SQ 기전)"
    out["C-H1"] = verdict_savings(sv["L-BR_vs_L-GR"]["accounting_tokens"], fz.get("L-BR", 0))
    out["C-H2"] = (verdict_savings(sv["E-SQ_vs_E-GR"]["accounting_tokens"]) if not g3_block
                   else {"verdict": "판정 불가", "reason": g3_block})
    out["C-H3"] = verdict_noninf(ni["E-SQ_vs_E-GR"], assays, g6a and rescored, blocked or g3_block)
    out["C-H4"] = verdict_noninf(ni["E-R1_vs_E-GR"], assays, g6a and rescored, blocked)
    out["C-H5_secondary"] = {**verdict_savings(sv["E-BR_vs_E-SQ"]["accounting_tokens"]),
                             "definition": "1 − Σ(E-BR)/Σ(E-SQ) (r2_analyze SAVINGS 정의 그대로; 보고만)"}
    if corpus1 is not None:
        c1 = corpus1["per_query"]
        out["corpus1_consistency"] = {
            "C-H1": consistency(pq, c1, "L-BR", "L-GR", "accounting_tokens", "savings", n_boot),
            "C-H2": consistency(pq, c1, "E-SQ", "E-GR", "accounting_tokens", "savings", n_boot),
            "C-H3_coverage": consistency(pq, c1, "E-SQ", "E-GR", "coverage", "diff", n_boot),
            "C-H3_grounding": consistency(pq, c1, "E-SQ", "E-GR", "grounding", "diff", n_boot),
            "C-H4_coverage": consistency(pq, c1, "E-R1", "E-GR", "coverage", "diff", n_boot),
            "C-H4_grounding": consistency(pq, c1, "E-R1", "E-GR", "grounding", "diff", n_boot),
            "source": corpus1.get("run_dir")}
    if not rescored:
        out["warning"] = "--rescore 없이 돌렸다: 등록 분석은 offline 재채점(루프 안 판정 100% 일치 assert)을 요구한다"
    return out


def analyze_c2(run_dir: Path, *, rescore: bool, pilot: bool, n_boot: int, corpus1: Path | None, negctl: Path | None,
               config_c2: Path = CONFIG_C2) -> dict[str, Any]:
    sup = RA.Support()
    if rescore:
        from lfm.config import Config, load_config
        from lfm.estimator import load_estimator
        base = load_config(config_c2)
        d = base.as_dict(); d.setdefault("estimator", {})["kind"] = "minicheck_doc"
        sup = RA.Support(load_estimator(Config(d, source="c2_analyze")), RA.load_chunk_text(ROOT / base["paths.index_dir"]))
    saved = (RA.PILOT_DEV, RA.g4_check, RA.g1_checks)
    RA.PILOT_DEV = frozenset()
    RA.g4_check = functools.partial(saved[1], **G4_C2)
    RA.g1_checks = functools.partial(g1_c2, saved[2])
    try:
        res = RA.analyze(run_dir, sup=sup, pilot=pilot, n_boot=n_boot)
    finally:
        RA.PILOT_DEV, RA.g4_check, RA.g1_checks = saved
    res.pop("sensitivity_30", None)
    res["notes"].append("코퍼스2: R2 의 30토픽 민감도(t000~t004 제외)는 해당 없음 — 블록을 지웠다")
    if pilot:
        res["gates"]["G4"]["multiplier_note"] = "C-G4: 파일럿 원장 × 25 (= 25토픽 × 3 seed ÷ 3 토픽-seed)"
        res["gates"]["G4"]["gpu_limit_hours"] = 8
    if res.get("rescore_consistency") and not res["rescore_consistency"]["pass"]:
        res["notes"].append("offline 재채점이 루프 안 판정과 다르다 — 등록 assert 실패")
    c1 = json.load(io.open(Path(corpus1) / "ANALYSIS.json", encoding="utf-8")) if corpus1 else None
    ng = None
    if negctl:
        ng = json.load(io.open(negctl, encoding="utf-8")); ng["path"] = str(negctl)
    res["c2"] = c2_verdicts(res, RA.load_run(Path(run_dir)), negctl=ng, corpus1=c1, n_boot=n_boot, pilot=pilot)
    return res


def c2_markdown(res: dict) -> str:
    c = res["c2"]
    L = ["# 코퍼스2 판정 — `%s`" % res["run_dir"], "", "자동 생성(`experiments/c2_analyze.py`). 수치는 ANALYSIS.json 의 `c2` 블록.", ""]
    L.append("- C-G2 E-GR 라운드 0 answered: %s (퇴화 %s)" % (RA._f(c["C-G2"]["answered_rate_pooled"]), c["C-G2"]["degenerate"]))
    L.append("- FREEZE 사건(라운드 수): %s" % c["freeze_events_by_arm"])
    L.append("- C-G3: %s · C-G6(a): %s · grounding 재채점: %s" % (c["C-G3"]["pass"], c["C-G6a"]["pass"], c["grounding_rescored"]))
    for m, a in c["assay"].items():
        if a.get("available"):
            L.append("- assay %s: E-GR 평균 %s, 90%% 하한 %s > δ %s → %s" % (m, RA._f(a["egr_mean"]), RA._f(a["egr_ci90_lo"]), a["delta"], a["sensitive"]))
    if "switch_rule_4_2_1" in c:
        L.append("- §4.2.1 전환: %s" % c["switch_rule_4_2_1"])
    for h in ("C-H1", "C-H2", "C-H3", "C-H4", "C-H5_secondary"):
        if h in c:
            L.append("- **%s**: %s %s" % (h, c[h]["verdict"], {k: v for k, v in c[h].items() if k != "verdict"}))
    if "corpus1_consistency" in c:
        L += ["", "## 코퍼스1 정합", "", "| 항목 | c2 | c1 | 차이 95% CI | 정합 |", "|---|---|---|---|---|"]
        for k, v in c["corpus1_consistency"].items():
            if isinstance(v, dict) and v.get("available"):
                L.append("| %s | %s | %s | [%s, %s] | %s |" % (k, RA._f(v["c2_point"]), RA._f(v["c1_point"]),
                         RA._f(v["diff_ci95"][0]), RA._f(v["diff_ci95"][1]), v["consistent"]))
    if c.get("warning"):
        L += ["", "**경고**: " + c["warning"]]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="코퍼스2 분석 (LLM 0)")
    ap.add_argument("run_dir")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--n-boot", type=int, default=RA.N_BOOT)
    ap.add_argument("--corpus1", default=None, help="코퍼스1 run_dir (ANALYSIS.json 의 per_query 를 읽는다)")
    ap.add_argument("--negctl", default=None, help="c2_negative_control.py 출력 JSON")
    ap.add_argument("--config", default=str(CONFIG_C2))
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    res = analyze_c2(run_dir, rescore=args.rescore, pilot=args.pilot, n_boot=args.n_boot,
                     corpus1=(ROOT / args.corpus1) if args.corpus1 and not Path(args.corpus1).is_absolute() else args.corpus1,
                     negctl=args.negctl, config_c2=Path(args.config))
    j, m = RA.write_outputs(res, run_dir)
    p = run_dir / "C2_VERDICTS.md"
    io.open(p, "w", encoding="utf-8").write(c2_markdown(res))
    print("-> %s\n-> %s\n-> %s" % (j, m, p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
