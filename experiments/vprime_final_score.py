"""V′ 최종 판정 — 1단계 60 + 2단계 40 = 100건. 호출 0. SPEC_M1Vprime §4 "보류 → 2단계 +40 후 100건 점추정으로 최종 판정
(그때는 하한 조건 없이 점추정만)". 정의는 vprime_score.py 와 같다.

연산자는 현행 채택 연산자(MiniCheck 단일 문서, p_mc_new > 0.5)와 g1(p >= 0.5). 칸(층)은 등록대로 원 MiniCheck × g1.
리뷰어 R1-3 의 요구대로 층 가중(Hájek, w = N_h / n_h) 모집단 추정치를 함께 낸다. 판정은 등록된 점추정 규칙으로만 한다.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vprime_score import wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "results/vprime"
BARS = {"precision": 0.85, "recall": 0.70, "recovery": 0.50}   # 음성 통제(≤5%)는 이미 0.00%


def main() -> int:
    lab = {r["item_id"]: r["human_label"].strip()
           for r in csv.DictReader(io.open(V / "labels_stage1_filled.csv", encoding="utf-8-sig"))}
    for r in csv.DictReader(io.open(ROOT / "results/d18/labels_all_filled.csv", encoding="utf-8-sig")):
        if r["set"] == "stage2":
            assert r["item_id"] not in lab
            lab[r["item_id"]] = r["human_label"].strip()
    key = {}
    for st in ("stage1", "stage2"):
        for r in csv.DictReader(io.open(V / ("_key_%s.csv" % st), encoding="utf-8")):
            key[r["item_id"]] = {**r, "stage": st}
    pnew = {r["item_id"]: float(r["p_mc_new"])
            for r in csv.DictReader(io.open(V / "rescore_singledoc_rows.csv", encoding="utf-8"))}
    strata = json.load(io.open(V / "stage2_strata.json", encoding="utf-8"))["strata"]
    assert set(lab) == set(key) and len(lab) == 100

    n_h = {c: strata[c]["n_stage1"] + strata[c]["n_stage2"] for c in strata}
    w = {c: strata[c]["N_h"] / n_h[c] for c in strata}
    usable = [(iid, v) for iid, v in lab.items() if v in ("지지", "불지지")]

    def score(pred_of, weighted):
        acc = {"precision": [0.0, 0.0], "recall": [0.0, 0.0], "recovery": [0.0, 0.0]}
        for iid, v in usable:
            k = key[iid]
            ww = w[k["cell"]] if weighted else 1.0
            pred, pos = pred_of(iid), v == "지지"
            if pred:
                acc["precision"][1] += ww; acc["precision"][0] += ww * pos
            if pos:
                acc["recall"][1] += ww; acc["recall"][0] += ww * pred
                if k["lab_g1"] == "0":
                    acc["recovery"][1] += ww; acc["recovery"][0] += ww * pred
        return acc

    ops = {"MiniCheck 단일문서 (채택)": lambda i: pnew[i] > 0.5,
           "g1 (현행 baseline)": lambda i: key[i]["lab_g1"] == "1"}
    out = {"n": len(lab), "n_unclear": sum(1 for v in lab.values() if v == "판단불가"),
           "human_support": sum(1 for v in lab.values() if v == "지지"),
           "strata": {c: {"N_h": strata[c]["N_h"], "n_h": n_h[c], "weight": round(w[c], 3)} for c in strata},
           "operators": {}}
    for name, f in ops.items():
        raw, wt = score(f, False), score(f, True)
        o = {}
        for m in BARS:
            k, n = raw[m]
            lo, hi = wilson(int(k), int(n))
            o[m] = {"k": int(k), "n": int(n), "rate": k / n if n else None, "wilson": [lo, hi],
                    "weighted": wt[m][0] / wt[m][1] if wt[m][1] else None,
                    "bar": BARS[m], "pass_point": (k / n > BARS[m]) if m == "recovery" else (k / n >= BARS[m])}
        o["verdict_registered"] = "통과" if all(o[m]["pass_point"] for m in BARS) else "탈락"
        out["operators"][name] = o
    out["by_cell"] = {c: {"n": sum(1 for i in lab if key[i]["cell"] == c),
                          "human_support": sum(1 for i in lab if key[i]["cell"] == c and lab[i] == "지지")}
                      for c in strata}
    out["stage2_only_support_rate"] = sum(1 for i, v in lab.items() if key[i]["stage"] == "stage2" and v == "지지") / 40
    io.open(V / "VPRIME_FINAL.json", "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
