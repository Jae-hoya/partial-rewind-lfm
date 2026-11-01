"""D16 블라인드 라벨 시트 — 정답 청크가 컨텍스트에 있는 sub-question 이 실제로 답할 수 있었나. 호출 0.

D16 의 P2(정답 청크가 실렸는데도 빈 답 40%)는 "정답 문서의 청크가 실렸다" 는 사실만 본다.
그 청크가 그 질문에 답을 담고 있었는지는 사람이 봐야 한다. 그래서:

  - C3 에서 정답 청크가 실렸는데 **비운** sub-question 전부
  - 같은 조건에서 **답한** sub-question 을 같은 수만큼 무작위(seed 42)
  - 둘을 섞고, 모델이 답했는지·어느 청크가 정답 문서인지는 **파일에 넣지 않는다**
  - 라벨러는 sub-question 과 모델이 받은 그 질문의 검색 청크(실린 그대로)만 본다

정답 키는 `results/d16/_key_d16.csv` 에 따로 둔다. 화면 양식은 V′ 라벨 도구를 그대로 쓴다.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np

from make_label_tool import HTML as VPRIME_HTML

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/d16/d16_subq.jsonl"
OUT = ROOT / "results/d16/labeler_d16.html"
KEY = ROOT / "results/d16/_key_d16.csv"

RULES = (
    "아래 컨텍스트만 보고 판단한다. 바깥 지식은 쓰지 않는다 / "
    "답있음 = 질문이 묻는 것에 대한 사실 진술이 컨텍스트에 있다. 일부만 묻는 질문이면 그 일부로 충분 / "
    "일부 = 질문과 관련된 사실은 있으나 묻는 것의 핵심에는 못 미친다 / "
    "답없음 = 질문에 답하는 사실이 없다. 주제만 같고 다른 내용이면 답없음"
)


def build_html(n: int) -> str:
    h = VPRIME_HTML
    for a, b in (
        ("불지지", "답없음"), ("지지", "답있음"), ("판단불가", "일부"),
        ("V′ 라벨 — 60건", "D16 라벨 — %d건" % n), ("60건 완료", "%d건 완료" % n),
        ("판정할 문장", "질문 (sub-question)"),
        ("근거 — 이 문장이 인용한 청크 전문", "컨텍스트 — 모델이 받은 이 질문의 검색 청크 (실린 그대로)"),
        ("vprime_stage1_v1", "d16_answerable_v1"), ("labels_stage1_filled.csv", "labels_d16_filled.csv"),
    ):
        assert a in h, a
        h = h.replace(a, b)
    return h


def main() -> int:
    rows = [json.loads(x) for x in io.open(SRC, encoding="utf-8")]
    c3 = [r for r in rows if r["condition"] == "C3_partial_heldout" and r["gold_in_ctx"]]
    empty = [r for r in c3 if r["empty"]]
    answered = [r for r in c3 if not r["empty"]]
    rng = np.random.default_rng(42)
    ctrl = [answered[i] for i in sorted(rng.choice(len(answered), size=len(empty), replace=False))]
    picked = empty + ctrl
    order = rng.permutation(len(picked))

    items, keys = [], []
    for j, k in enumerate(order):
        r = picked[int(k)]
        iid = "q%03d" % j
        ev = "\n\n".join(r["ctx_texts"][cid] for cid in r["own_in_ctx"])
        items.append({"id": iid, "rules": RULES, "claim": r["sub_question"], "ev": ev})
        keys.append({"item_id": iid, "query_id": r["query_id"], "branch": r["branch"],
                     "sq_index": r["sq_index"], "model_empty": r["empty"],
                     "n_gold_in_ctx": len(r["gold_in_ctx"]), "gold_chunks": " ".join(r["gold_in_ctx"])})

    io.open(OUT, "w", encoding="utf-8").write(
        build_html(len(items)).replace("__ITEMS__", json.dumps(items, ensure_ascii=False)))
    with io.open(KEY, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(keys[0].keys()))
        w.writeheader()
        w.writerows(keys)
    ev = sorted(len(x["ev"]) for x in items)
    print("문항 %d개 (빈 답 %d + 답한 대조 %d) -> %s" % (len(items), len(empty), len(ctrl), OUT))
    print("컨텍스트 길이 중앙 %d자 · 최대 %d자" % (ev[len(ev) // 2], ev[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
