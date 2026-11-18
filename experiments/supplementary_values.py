"""원고·부록이 "검증 에이전트가 계산했고 배포 스크립트는 없다"고 적은 값을 배포 파일만으로 다시 계산한다. **API 0, GPU 0.**

  python experiments/supplementary_values.py    # → results/extra/SUPPLEMENTARY_VALUES.json

(a) τ = 0.7 branch 단위 낭비율, 원고 §III-C 의 정의 그대로:
    S_τ(E_i) = 1[cov_i = 1.0 ∧ grd_i > τ], 낭비율 = (재실행된 branch-round 중 S_τ = 1 인 수) / (재실행된 branch-round 수).
    Table I·XV 는 cov 다리를 cov_i > τ 로 계산했다(test_a.py `sufficient@τ` 열 = lfm.evidence.sufficient).
    그래서 test_a.waste_by_query 는 그대로 쓸 수 없다(미리 계산된 `sufficient@τ` 열을 읽는다) — 같은 모집단
    (re_executed = 1)에서 coverage 다리만 바꿔 센다. test_a / waste_supplement 는 lfm.llm·모델 로더를 끌어오므로
    import 하지 않고 waste_ratio.csv 를 직접 읽는다. cov > τ 판은 waste_ratio_ci.csv 와 대조해 모집단이 같음을 보인다.
    g_1 = Test A (results/test_a/20260829-041445), 채택 연산자 = MiniCheck M1′ (results/m1prime/seed42).

(b) V′ C50 50문항, 2차(R3) 채점 열 p_mc_new > 0.5 (연산자 양성 38) 기준 precision/recall.
    r3_reliability.load_static() 의 라벨·점수를 그대로 쓴다(저자 = labels_stage1_filled + d18 stage2, A-v1 = r2 bundle vc_a).
    이진화는 r3_reliability.b: 지지 = 1, 불지지 ∪ 판단불가 = 0. 1차 기준(정답키 lab_mc, 양성 34)도 참고로 함께 낸다.

기대값은 계산 뒤 비교에만 쓴다. 하나라도 다르면 종료 코드 1.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from r3_reliability import CATS, b, load_static  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/extra/SUPPLEMENTARY_VALUES.json"
TAU = 0.7
WASTE = {"g1": "results/test_a/20260829-041445",
         "selected_operator (MiniCheck M1')": "results/m1prime/seed42"}
# 보고된 값 — 비교에만 쓴다
REPORTED = {"waste_tau0.7_cov1.0/g1": (6, 400),
            "waste_tau0.7_cov1.0/selected_operator (MiniCheck M1')": (53, 400),
            "r2_precision/author": (32, 38),
            "r2_recall/author": (32, 40),
            "r2_precision/A-v1": (9, 38)}
B_INPUTS = ["results/vprime/labels_stage1_filled.csv", "results/d18/labels_all_filled.csv",
            "results/vprime/_key_stage1.csv", "results/vprime/_key_stage2.csv",
            "results/vprime/rescore_singledoc_rows.csv", "results/vprime/c50_sealed/_key_vc50.csv",
            "results/r2/bundle/labels/labels_r2_bundle_filled.csv"]
B_LOADER_ONLY = ["results/vprime/VPRIME_DEDUP_CHECK.json", "results/vprime/stage2_strata.json",
                 "results/vprime/premise_lengths.csv"]


def rd(p: Path) -> list[dict]:
    with io.open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def frac(k: int, n: int) -> dict:
    return {"value": k / n if n else None, "numerator": k, "denominator": n}


# --------------------------------------------------------------------------- (a)
def waste(d: str) -> dict:
    rows = [r for r in rd(ROOT / d / "waste_ratio.csv") if int(r["re_executed"])]
    ci = {float(r["tau"]): r for r in rd(ROOT / d / "waste_ratio_ci.csv")}
    out = {"n_reexec": len(rows), "by_tau": {}}
    for t in sorted(ci):
        eq = sum(1 for r in rows if float(r["coverage"]) == 1.0 and float(r["grounding"]) > t)
        gt = sum(1 for r in rows if float(r["coverage"]) > t and float(r["grounding"]) > t)
        # 같은 모집단·같은 grounding 다리임을 배포 표 값과 대조
        assert len(rows) == int(ci[t]["n_reexec_branches"]), (d, t)
        assert gt == int(ci[t]["n_wasted_branches"]), (d, t, gt)
        out["by_tau"][t] = {"cov_eq_1": eq, "cov_gt_tau": gt, "released_ci_csv_n_wasted": int(ci[t]["n_wasted_branches"])}
    for t in (0.8, 0.9):
        assert out["by_tau"][t]["cov_eq_1"] == out["by_tau"][t]["cov_gt_tau"], (d, t)
    return out


# --------------------------------------------------------------------------- (b)
def second_round() -> dict:
    S = load_static()
    c50 = S["c50"]  # C50 시트 id → V′ 원 문항 id
    assert len(c50) == 50 and set(S["av1"]) == set(c50), "A-v1 문항이 C50 과 다름"
    bad = {i: x for i, x in S["av1"].items() if x not in CATS}
    bad.update({s: S["lab"][s] for s in c50.values() if S["lab"][s] not in CATS})
    assert not bad, "범주 밖 라벨: %r" % bad
    ids = sorted(c50)
    author = {i: b(S["lab"][c50[i]]) for i in ids}
    av1 = {i: b(S["av1"][i]) for i in ids}
    bases = {"second_round (p_mc_new > 0.5)": {i: S["pnew"][c50[i]] > 0.5 for i in ids},
             "first_round (_key lab_mc == 1)": {i: S["key"][c50[i]]["lab_mc"] == "1" for i in ids}}
    res = {"n_items": len(ids), "author_support": sum(author.values()), "A-v1_support": sum(av1.values()),
           "author_labels": dict(Counter(S["lab"][c50[i]] for i in ids)), "A-v1_labels": dict(Counter(S["av1"][i] for i in ids))}
    for name, pred in bases.items():
        P = [i for i in ids if pred[i]]
        res[name] = {"operator_positives": len(P),
                     "precision_author": frac(sum(author[i] for i in P), len(P)),
                     "recall_author": frac(sum(author[i] for i in P), sum(author.values())),
                     "precision_A-v1": frac(sum(av1[i] for i in P), len(P)),
                     "recall_A-v1": frac(sum(av1[i] for i in P), sum(av1.values()))}
    # 배포된 R3 채점 결과와도 대조(입력이 아니라 확인용)
    rel = json.load(io.open(ROOT / "results/r3/R3_RELIABILITY.json", encoding="utf-8"))
    m = rel["operator"]["V1c"]["MiniCheck (registered_dup, p>0.5)"]
    r2 = res["second_round (p_mc_new > 0.5)"]
    assert (m["precision"]["k"], m["precision"]["n"]) == (r2["precision_author"]["numerator"], r2["precision_author"]["denominator"])
    assert (m["recall"]["k"], m["recall"]["n"]) == (r2["recall_author"]["numerator"], r2["recall_author"]["denominator"])
    return res


def main() -> int:
    a = {name: waste(d) for name, d in WASTE.items()}
    bb = second_round()
    r2 = bb["second_round (p_mc_new > 0.5)"]
    got = {"r2_precision/author": r2["precision_author"], "r2_recall/author": r2["recall_author"],
           "r2_precision/A-v1": r2["precision_A-v1"]}
    for name in WASTE:
        x = a[name]["by_tau"][TAU]
        got["waste_tau0.7_cov1.0/%s" % name] = frac(x["cov_eq_1"], a[name]["n_reexec"])

    ok = True
    values = {}
    print("값 | 분자/분모 | 비율 | 보고값 | 일치")
    for k, (rk, rn) in REPORTED.items():
        g = got[k]
        match = (g["numerator"], g["denominator"]) == (rk, rn)
        ok &= match
        print("%s | %d/%d | %.4f | %d/%d | %s" % (k, g["numerator"], g["denominator"], g["value"], rk, rn,
                                                "MATCH" if match else "MISMATCH"))
        values[k] = {**g, "reported": "%d/%d" % (rk, rn), "match": match}
    print()
    for name in WASTE:
        for t, x in a[name]["by_tau"].items():
            print("  [참고] %s τ=%.1f: cov=1.0 %d/400, cov>τ %d/400 (waste_ratio_ci.csv %d)"
                  % (name, t, x["cov_eq_1"], x["cov_gt_tau"], x["released_ci_csv_n_wasted"]))
    r1 = bb["first_round (_key lab_mc == 1)"]
    print("  [참고] 1차 기준(lab_mc, 양성 %d): 저자 precision %d/%d = %.3f, A-v1 precision %d/%d = %.3f"
          % (r1["operator_positives"], r1["precision_author"]["numerator"], r1["precision_author"]["denominator"],
             r1["precision_author"]["value"], r1["precision_A-v1"]["numerator"], r1["precision_A-v1"]["denominator"],
             r1["precision_A-v1"]["value"]))

    for name, d in WASTE.items():
        values["waste_tau0.7_cov1.0/%s" % name]["inputs"] = [d + "/waste_ratio.csv"]
        values["waste_tau0.7_cov1.0/%s" % name]["cross_check"] = [d + "/waste_ratio_ci.csv"]
    for k in ("r2_precision/author", "r2_recall/author", "r2_precision/A-v1"):
        values[k]["inputs"] = B_INPUTS
        values[k]["read_by_loader_not_used"] = B_LOADER_ONLY
        values[k]["cross_check"] = ["results/r3/R3_RELIABILITY.json"] if "author" in k else []
    out = {"script": "experiments/supplementary_values.py",
           "definitions": {
               "waste_tau0.7_cov1.0": "re_executed = 1 인 branch-round 중 coverage == 1.0 이고 grounding > 0.7 인 비율 "
                                      "(manuscript §III-C S_τ). grounding 열은 각 실행이 기록한 값 그대로.",
               "r2_precision/recall": "V′ C50 50문항; 연산자 양성 = rescore_singledoc_rows.csv p_mc_new > 0.5; "
                                      "사람 양성 = 라벨 '지지' (불지지·판단불가 = 0); precision = 양성 중 사람 '지지' 비율, "
                                      "recall = 사람 '지지' 중 연산자 양성 비율. 비가중."},
           "values": values,
           "context": {"waste_by_tau": {n: {str(t): x for t, x in v["by_tau"].items()} | {"n_reexec": v["n_reexec"]}
                                        for n, v in a.items()},
                       "precision_recall": bb},
           "all_match": ok}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("\n→ %s" % OUT.relative_to(ROOT).as_posix())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
