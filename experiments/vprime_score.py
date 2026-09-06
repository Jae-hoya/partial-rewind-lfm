"""M1-V′ 채점. SPEC_M1V′ §2. **호출 0회.**

사람 라벨이 기준이다. 각 후보 추정자에 대해 사람 '지지'를 양성으로 놓고 잰다.

  precision   후보 '지지' 중 사람 '지지' 비율        (P1+P2 에서, bart 는 자기 양성 칸에서)
  recall      사람 '지지' 중 후보 '지지' 비율        (60건 전체)
  N층 회수    (g1 미지지 ∧ 사람 지지) 중 후보 '지지'  (P2+N1 의 사람-지지에서)

판단불가는 **분모에서 제외**하고 건수만 보고한다. 각 비율에 Wilson 95% 구간.

관문 숫자를 여기서 새로 만들지 않는다. 스펙 §4 의 판정은 이 표를 보고 공저자가 한다.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POS, NEG, UNK = "지지", "불지지", "판단불가"
ALIAS = {"지지": POS, "supported": POS, "s": POS,
         "불지지": NEG, "미지지": NEG, "unsupported": NEG, "u": NEG,
         "판단불가": UNK, "unclear": UNK, "?": UNK}


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def cell(k, n):
    if not n:
        return "-"
    lo, hi = wilson(k, n)
    return "%d/%d = %.1f%% [%.1f, %.1f]" % (k, n, 100 * k / n, 100 * lo, 100 * hi)


def main(argv=None):
    ap = argparse.ArgumentParser(description="V' 채점 (호출 0회)")
    ap.add_argument("--dir", default="results/vprime")
    ap.add_argument("--stage", default="stage1")
    args = ap.parse_args(argv)

    d = ROOT / args.dir
    lab = {r["item_id"]: ALIAS.get(r["human_label"].strip().lower(),
                                   ALIAS.get(r["human_label"].strip(), ""))
           for r in csv.DictReader(io.open(d / ("labels_%s.csv" % args.stage),
                                           encoding="utf-8-sig"))}
    key = {r["item_id"]: r for r in csv.DictReader(io.open(d / ("_key_%s.csv" % args.stage),
                                                           encoding="utf-8"))}
    filled = {k: v for k, v in lab.items() if v in (POS, NEG, UNK)}
    blank = len(lab) - len(filled)
    unk = sum(1 for v in filled.values() if v == UNK)
    usable = {k: v for k, v in filled.items() if v != UNK}

    print("라벨 %d건 중 채워진 것 %d, 빈칸 %d, 판단불가 %d (분모에서 제외)"
          % (len(lab), len(filled), blank, unk))
    if blank:
        print("**빈칸이 있다. 전부 채운 뒤 다시 돌려라.**")
    if not usable:
        print("채점할 것이 없다.")
        return 1

    print("")
    print("| 후보 | precision (양성 칸) | recall (전체) | N층 회수 |")
    print("|---|---|---|---|")
    rows = []
    for est, col in [("g1 (현행)", "lab_g1"), ("MiniCheck", "lab_mc"), ("bart", "lab_bart")]:
        pk = pn = rk = rn = nk = nn = 0
        for iid, human in usable.items():
            k = key.get(iid)
            if not k:
                continue
            pred = k[col] == "1"
            if pred:                        # precision 분모
                pn += 1
                pk += human == POS
            if human == POS:                # recall 분모
                rn += 1
                rk += pred
                if k["lab_g1"] == "0":      # g1 이 놓친 것 중
                    nn += 1
                    nk += pred
        print("| %s | %s | %s | %s |"
              % (est, cell(pk, pn), cell(rk, rn), cell(nk, nn)))
        rows.append({"estimator": est, "precision_k": pk, "precision_n": pn,
                     "recall_k": rk, "recall_n": rn,
                     "n_recovery_k": nk, "n_recovery_n": nn,
                     "n_unclear_excluded": unk})

    with io.open(d / ("scores_%s.csv" % args.stage), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("")
    print("칸별 사람 라벨 분포 (모집단 확인용):")
    for c in ("P1", "P2", "N1", "N2"):
        ids = [i for i in usable if key.get(i, {}).get("cell") == c]
        p = sum(1 for i in ids if usable[i] == POS)
        print("  %s  n=%2d  사람 지지 %2d (%.0f%%)"
              % (c, len(ids), p, 100 * p / len(ids) if ids else 0))
    print("")
    print("음성 통제는 이미 있다 — MiniCheck 0.00%, bart 1.02%, 현행 0.00% (전부 <=5%).")
    print("판정은 이 표로 공저자가 한다. 여기서 관문 숫자를 새로 만들지 않는다.")
    print("-> %s" % (d / ("scores_%s.csv" % args.stage)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
