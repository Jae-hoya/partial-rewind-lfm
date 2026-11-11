"""PREREG_R3 §4.6 C-G7 (C2-V) 채점. 보고만(관문 아님). LLM 0, 모델 0.

입력: 저자 CSV(`results/c2/c2v/labels_c2v_filled.csv`, 받은 그대로 먼저 커밋) + 봉인 키.
양성 = 사람 '지지'. '판단불가'는 precision·recall 분모에서 빼고 개수를 따로 적는다.
  precision = P(사람 지지 | MiniCheck 지지)    — 지지 층 20개 안의 비율(층 안 단순 비율, 가중 불필요)
  recall    = P(MiniCheck 지지 | 사람 지지)    — 층 표집이 20/20 이므로 Hájek 가중 w = N_cell / n_cell 로 모집단 값을 추정
  둘 다 Wilson 95%(recall 은 Kish 유효 n 으로 근사), 가중 안 한 표본 recall 도 함께 적는다.
원고 문구: "single-author check, n = 40; the author knew the hypotheses but not the operator's verdicts".

  python experiments/c2v_score.py --attempt 2   # 주 결과 (DEVIATIONS 28)
  python experiments/c2v_score.py --attempt 1   # 1차 시도, 보고만
"""

from __future__ import annotations

import csv
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_negative_control import wilson  # noqa: E402

ATTEMPTS = {1: ("labels_c2v_filled.csv", "c2v_sheet_v1", "C2V_RESULT_attempt1.json"),
            2: ("labels_c2v_attempt2.csv", "c2v_sheet_v2", "C2V_RESULT.json")}   # DEVIATIONS 28: 2차 = 주 결과


def wilson_p(p: float, n_eff: float, z: float = 1.959963984540054) -> tuple[float, float]:
    if n_eff <= 0:
        return (0.0, 1.0)
    den = 1 + z * z / n_eff
    c = (p + z * z / (2 * n_eff)) / den
    h = z * math.sqrt(p * (1 - p) / n_eff + z * z / (4 * n_eff * n_eff)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", type=int, choices=(1, 2), default=2)
    a = ap.parse_args(argv)
    lab_name, VERSION, out_name = ATTEMPTS[a.attempt]
    lab_p = ROOT / "results/c2/c2v" / lab_name
    key_p = ROOT / "results/c2/c2v_sealed/_key_c2v.csv"
    labs = {r["item_id"]: r for r in csv.DictReader(io.open(lab_p, encoding="utf-8-sig"))}
    key = {r["item_id"]: r for r in csv.DictReader(io.open(key_p, encoding="utf-8"))}
    assert set(labs) == set(key) and len(key) == 40, "문항 집합 불일치"
    vers = {r.get("sheet_version", "") for r in labs.values()}
    assert vers == {VERSION}, "sheet_version 불일치: %s" % vers
    empty = [i for i, r in labs.items() if not r["human_label"]]
    assert not empty, "빈 라벨 %d" % len(empty)
    rows = []
    for iid, k in key.items():
        h = labs[iid]["human_label"]
        rows.append({"mc": int(k["mc_supported"]), "h": h, "w": int(k["N_cell"]) / int(k["n_cell"])})
    undec = [r for r in rows if r["h"] == "판단불가"]
    use = [r for r in rows if r["h"] != "판단불가"]
    sup = [r for r in use if r["mc"] == 1]
    tp = sum(r["h"] == "지지" for r in sup)
    prec = {"k": tp, "n": len(sup), "rate": tp / len(sup) if sup else None, "wilson95": list(wilson(tp, len(sup)))}
    hpos = [r for r in use if r["h"] == "지지"]
    raw_k = sum(r["mc"] for r in hpos)
    W = sum(r["w"] for r in hpos)
    rec_w = sum(r["w"] * r["mc"] for r in hpos) / W if W else None
    n_eff = (W * W / sum(r["w"] ** 2 for r in hpos)) if hpos else 0.0
    rec = {"k_raw": raw_k, "n_raw": len(hpos), "rate_raw": raw_k / len(hpos) if hpos else None,
           "rate_hajek": rec_w, "n_eff_kish": round(n_eff, 2),
           "wilson95_hajek_approx": list(wilson_p(rec_w, n_eff)) if hpos else [0.0, 1.0]}
    res = {"spec": "PREREG_R3 §4.6 C-G7 (report only)", "n": len(rows), "n_undecidable": len(undec),
           "undecidable_by_mc": {"sup": sum(r["mc"] for r in undec), "uns": sum(1 - r["mc"] for r in undec)},
           "precision": prec, "recall": rec,
           "translator_used": {t: sum(labs[i]["translator_used"] == t for i in labs) for t in sorted({r["translator_used"] for r in labs.values()})},
           "scrolled_to_bottom": sum(int(labs[i]["scrolled_to_bottom"] or 0) for i in labs),
           "manuscript_note": "single-author check, n = 40; the author knew the hypotheses but not the operator's verdicts"}
    res["attempt"] = a.attempt
    res["role"] = "primary" if a.attempt == 2 else "attempt 1 (fast pass, reported; DEVIATIONS 28)"
    out = ROOT / "results/c2/c2v" / out_name
    io.open(out, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
