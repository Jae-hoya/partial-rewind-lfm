"""H1 사람 라벨 채점 — D16 블라인드 시트 86건을 정답 키와 D17 자동 증명에 대조한다. 호출 0.

라벨(labels_d16_filled.csv)은 채점 전에 커밋됐다(1a515cb). 여기서 처음으로 키를 연다.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from d16_abstention_cause import cluster_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("답있음", "일부", "답없음")


def main() -> int:
    d = ROOT / "results/d16"
    lab = {r["item_id"]: r["human_label"] for r in csv.DictReader(io.open(d / "labels_d16_filled.csv", encoding="utf-8-sig"))}
    key = list(csv.DictReader(io.open(d / "_key_d16.csv", encoding="utf-8")))
    assert set(lab) == {k["item_id"] for k in key}, "라벨과 키의 문항 집합이 다르다"
    assert all(v in LABELS for v in lab.values()), "빈 칸 또는 알 수 없는 라벨"

    d17 = {(r["query_id"], r["branch"], r["sq_index"]): r
           for r in csv.DictReader(io.open(ROOT / "results/d17/d17_items.csv", encoding="utf-8"))}

    rows = []
    for k in key:
        a = d17[(k["query_id"], k["branch"], k["sq_index"])]
        rows.append({**k, "human": lab[k["item_id"]], "model_empty": int(k["model_empty"]),
                     "proved_via_c4": int(a["proved_via_c4"]),
                     "proved_own": int(a["proved_own"])})

    out = {}
    for name, grp in (("model_empty", [r for r in rows if r["model_empty"]]),
                      ("model_answered", [r for r in rows if not r["model_empty"]])):
        c = Counter(r["human"] for r in grp)
        out[name] = {
            "n": len(grp), "counts": {l: c.get(l, 0) for l in LABELS},
            "answerable_strict": cluster_ci([(r["query_id"], int(r["human"] == "답있음")) for r in grp]),
            "answerable_loose": cluster_ci([(r["query_id"], int(r["human"] != "답없음")) for r in grp]),
            "by_branch": {b: dict(Counter(r["human"] for r in grp if r["branch"] == b))
                          for b in sorted({r["branch"] for r in grp})},
        }

    # D17 자동 증명과의 대조. 증명됐는데 사람이 답없음이면 둘 중 하나가 틀렸다.
    def cross(col, grp):
        t = Counter((r[col], r["human"]) for r in grp)
        return {"%s=%d" % (col, p): {l: t.get((p, l), 0) for l in LABELS} for p in (0, 1)}

    out["vs_d17_empty_proved_via_c4"] = cross("proved_via_c4", [r for r in rows if r["model_empty"]])
    out["vs_d17_answered_proved_own"] = cross("proved_own", [r for r in rows if not r["model_empty"]])

    with io.open(d / "h1_scored.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with io.open(d / "h1_summary.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
