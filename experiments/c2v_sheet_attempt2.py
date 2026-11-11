"""C2-V 2차 시도 시트 (DEVIATIONS 28). 생성 LLM 0, 모델 0, GPU 0.

1차 시트(`results/c2/c2v/labeler_c2v.html`, 8be49c6)의 40문항을 **문항·id·내용 그대로** 가져와
  - 표시 순서만 `default_rng(4506)` 로 다시 섞고(1차 순서 기억을 줄이려고),
  - 저장 키 `c2v_v2`(1차 답이 미리 채워지지 않게), 판 `c2v_sheet_v2`, 내보내기 `labels_c2v_attempt2.csv`,
  - **버튼 관문**: 번역 기록 + 근거 상자를 끝까지 스크롤(`scrolled_to_bottom = 1`, 상자에 다 들어가면 즉시 1)
    두 가지를 다 채워야 라벨 버튼이 켜진다(1차는 번역 기록만). 체류 하한은 두지 않는다.
봉인 키는 1차와 같다(`results/c2/c2v_sealed/_key_c2v.csv`, 문항 id 동일).

  python experiments/c2v_sheet_attempt2.py   # -> results/c2/c2v/labeler_c2v_attempt2.html
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c2v_sheet as C1  # noqa: E402

SEED2, VERSION2 = 4506, "c2v_sheet_v2"


def main() -> int:
    src = io.open(ROOT / "results/c2/c2v/labeler_c2v.html", encoding="utf-8").read()
    sets = json.loads(re.search(r"const SETS = (\[.*?\]);\n", src, re.S).group(1))
    items = sets[0]["items"]
    assert len(items) == 40 and len({x["id"] for x in items}) == 40
    order = np.random.default_rng(SEED2).permutation(40).tolist()
    items2 = [items[k] for k in order]
    h = C1.build_html(items2)
    reps = [
        ('"c2v_v1"', '"c2v_v2"'),
        ("labels_c2v_filled.csv", "labels_c2v_attempt2.csv"),
        ('"%s"' % C1.VERSION, '"%s"' % VERSION2),
        ("b.disabled = !m.translator_used)", "b.disabled = !(m.translator_used && m.scrolled_to_bottom))"),
        ("if (!m.translator_used) return;", "if (!(m.translator_used && m.scrolled_to_bottom)) return;"),
        ("  if (ev) m.scroll_events++;\n  save();\n}", "  if (ev) m.scroll_events++;\n  save();\n  gate();\n}"),
        ("(먼저 골라야 라벨 버튼이 켜집니다)", "(번역 사용을 고르고 근거를 끝까지 스크롤해야 라벨 버튼이 켜집니다)"),
    ]
    for a, b in reps:
        n = h.count(a)
        assert n >= 1, ("패치 위치 없음", a)
        h = h.replace(a, b)
    assert "c2v_v1" not in h and C1.VERSION not in h
    out = ROOT / "results/c2/c2v/labeler_c2v_attempt2.html"
    io.open(out, "w", encoding="utf-8", newline="\n").write(h)
    meta = {"spec": "PREREG_R3 §4.6 C-G7, attempt 2 (DEVIATIONS 28)", "source_sheet": "results/c2/c2v/labeler_c2v.html",
            "order_seed": SEED2, "sheet_version": VERSION2, "n_items": 40,
            "gate": "translator_used AND scrolled_to_bottom", "html_sha256": hashlib.sha256(h.encode("utf-8")).hexdigest()}
    io.open(ROOT / "results/c2/c2v/_meta_c2v_attempt2.json", "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
