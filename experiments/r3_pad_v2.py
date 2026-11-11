"""R3 v2 간소 시트 CSV(7칸) → 등록 채점기 머리줄(COLS)로 맞춘 사본 (DEVIATIONS 33). 값은 바꾸지 않는다.

  python experiments/r3_pad_v2.py RAW_DIR OUT_DIR   # RAW_DIR/labels_{A,B}.csv (받은 그대로, 커밋됨) -> OUT_DIR/labels_{A,B}.csv
없는 칸은 빈칸. 원본에 COLS 밖의 칸이 있거나 판이 r3_sheet_v2 가 아니면 멈춘다.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_sheet import COLS  # noqa: E402
from r3_sheet_min import OUT_COLS, VERSION  # noqa: E402


def pad(src: Path, dst: Path) -> int:
    rows = list(csv.DictReader(io.open(src, encoding="utf-8-sig")))
    assert rows and list(rows[0]) == OUT_COLS, (src, list(rows[0]) if rows else None)
    assert {r["sheet_version"] for r in rows} == {VERSION}, src
    dst.parent.mkdir(parents=True, exist_ok=True)
    with io.open(dst, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})
    return len(rows)


def main(argv=None) -> int:
    a = argv if argv is not None else sys.argv[1:]
    raw, out = Path(a[0]), Path(a[1])
    for x in ("A", "B"):
        p = raw / ("labels_%s.csv" % x)
        if p.exists():
            print(x, pad(p, out / p.name), "rows ->", out / p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
