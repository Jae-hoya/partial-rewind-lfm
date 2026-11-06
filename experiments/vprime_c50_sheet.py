"""PREREG_R2 §9 C — 2nd annotator V′ 50건 시트. LLM 0, 모델 0.

V′ 100건(1단계 60 + 2단계 40)은 저자가 라벨했다. 저자가 아닌 라벨러가 그중 50건을 **원래 시트와 같은 표시**
(문장 + 그 문장이 인용한 근거, 같은 판단 기준)로 라벨해 저자–라벨러 일치(κ)를 낸다. 층(cell = 원 MiniCheck × g1)
비례로 뽑고, 문항 id 는 새로 매겨 원래 id 와의 대응은 봉인 키에만 둔다.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_combined_sheet import HTML  # noqa: E402

V = ROOT / "results/vprime"
OUT, SEALED = V / "c50", V / "c50_sealed"
RNG_SEED = 4210
N = 50
LABELS = ["지지", "불지지", "판단불가"]


def items(path: Path) -> list[dict]:
    h = io.open(path, encoding="utf-8").read()
    return json.loads(re.search(r"const ITEMS = (\[.*?\]);\n", h, re.S).group(1))


def main() -> int:
    pool = {it["id"]: it for f in ("labeler.html", "labeler_stage2.html") for it in items(V / f)}
    key = {}
    for st in ("stage1", "stage2"):
        for r in csv.DictReader(io.open(V / ("_key_%s.csv" % st), encoding="utf-8")):
            key[r["item_id"]] = {**r, "stage": st}
    assert set(pool) == set(key) and len(pool) == 100
    rng = np.random.default_rng(RNG_SEED)
    by = {}
    for iid in sorted(pool):
        by.setdefault(key[iid]["cell"], []).append(iid)
    # 층 비례 배분(최대 나머지법), 층마다 무작위 추출
    quota = {c: N * len(v) / len(pool) for c, v in by.items()}
    alloc = {c: int(q) for c, q in quota.items()}
    for c in sorted(quota, key=lambda c: -(quota[c] - alloc[c]))[: N - sum(alloc.values())]:
        alloc[c] += 1
    picked = [iid for c in sorted(by) for iid in rng.choice(by[c], alloc[c], replace=False).tolist()]
    order = rng.permutation(len(picked)).tolist()
    rows, sheet = [], []
    for j, i in enumerate(order):
        src = pool[picked[i]]
        new = "vc%03d" % j
        sheet.append({"id": new, "rules": src["rules"], "claim": src["claim"], "ev": src["ev"]})
        rows.append({"item_id": new, "source_item": picked[i], "cell": key[picked[i]]["cell"],
                     "stage": key[picked[i]]["stage"], "N_cell_in_100": len(by[key[picked[i]]["cell"]]),
                     "n_cell": alloc[key[picked[i]]["cell"]]})
    sets = [{"id": "vc_a", "name": "세트 1 (문장 지지)", "labels": LABELS,
             "h_claim": "문장 (모델이 쓴 답의 한 문장)", "h_ev": "근거 (그 문장이 인용한 글)", "items": sheet}]
    h = HTML
    for a, b in (("재투고 라벨 — 통합", "라벨 — V"), ("r2_combined_v1", "r2_vc50_v1"),
                 ("labels_all_filled.csv", "labels_vc50_filled.csv")):
        assert a in h, a
        h = h.replace(a, b)
    OUT.mkdir(parents=True, exist_ok=True)
    SEALED.mkdir(parents=True, exist_ok=True)
    io.open(OUT / "labeler_vc50.html", "w", encoding="utf-8").write(h.replace("__SETS__", json.dumps(sets, ensure_ascii=False)))
    with io.open(SEALED / "_key_vc50.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("V′ C50: %d items, alloc %s -> %s" % (len(sheet), alloc, OUT / "labeler_vc50.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
