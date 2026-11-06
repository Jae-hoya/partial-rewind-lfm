"""라벨러에게 줄 **한 파일** — H4 → H5 → H3 → V′C50 을 순서대로 묶고, 앞 단계를 다 채워야 다음 단계가 열린다.

시트 간 누출(DEVIATIONS 15: H3·H5 를 먼저 보면 H4 문항 일부의 arm 을 추론할 수 있다)을 막으려고 순서를 화면에서
강제한다. 문항·판단 기준·표시는 각 시트 그대로이고 탭 이름만 단계 번호로 바꾼다. LLM 0.
CSV 는 `set,item_id,human_label` 한 개로 나온다(item_id 접두사 h4·h5·h3·vc 로 세트가 갈린다).
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_combined_sheet import HTML  # noqa: E402

SHEETS = [ROOT / "results/r2/heldout35/sheets/labeler_h4.html",
          ROOT / "results/r2/heldout35/sheets/labeler_h5.html",
          ROOT / "results/r2/heldout35/sheets/labeler_h3.html",
          ROOT / "results/vprime/c50/labeler_vc50.html"]
OUT = ROOT / "results/r2/bundle/labeler_r2_bundle.html"

# 앞 단계가 다 차야 다음 단계를 연다. 위치 복원·다음·안 채운 곳으로도 잠긴 단계로는 가지 않는다
LOCK_JS = {
    'let si = 0, i = 0;':
        'let si = 0, i = 0;\nconst unlocked = k => SETS.slice(0, k).every(s => filled(s) === s.items.length);',
    'try { const p = JSON.parse(localStorage.getItem(KEY + "_pos") || "null"); if (p) { si = p[0]; i = p[1]; } } catch (e) {}':
        'try { const p = JSON.parse(localStorage.getItem(KEY + "_pos") || "null"); if (p) { si = p[0]; i = p[1]; } } catch (e) {}\n'
        'if (!(si >= 0 && si < SETS.length) || !unlocked(si)) { si = 0; i = 0; }',
    '    b.onclick = () => { si = k; i = 0;':
        '    if (!unlocked(k)) { b.className += " locked"; b.innerHTML = "🔒 " + b.innerHTML;\n'
        '      b.onclick = () => alert("앞 단계를 끝까지 채워야 열립니다."); t.appendChild(b); return; }\n'
        '    b.onclick = () => { si = k; i = 0;',
    'document.getElementById("next").onclick = () => { if (i < SETS[si].items.length - 1) { i++; render(); } };':
        'document.getElementById("next").onclick = () => {\n'
        '  if (i < SETS[si].items.length - 1) { i++; render(); }\n'
        '  else if (si < SETS.length - 1 && unlocked(si + 1)) { si++; i = 0; render(); }\n'
        '  else if (si < SETS.length - 1) alert("이 단계에 안 채운 문항이 있습니다. \\"안 채운 곳으로\\"를 누르세요."); };',
    '    const k = (si + d) % SETS.length, j':
        '    const k = (si + d) % SETS.length; if (!unlocked(k)) continue; const j',
}


def sets_of(path: Path) -> list[dict]:
    h = io.open(path, encoding="utf-8").read()
    return json.loads(re.search(r"const SETS = (\[.*?\]);\n", h, re.S).group(1))


def main() -> int:
    sets = []
    for n, p in enumerate(SHEETS, 1):
        ss = sets_of(p)
        assert len(ss) == 1, p
        s = dict(ss[0])
        s["name"] = "%d단계" % n
        sets.append(s)
    ids = [x["id"] for s in sets for x in s["items"]]
    assert len(ids) == len(set(ids)), "문항 id 가 겹친다"
    h = HTML
    for a, b in (("재투고 라벨 — 통합", "라벨 작업"), ("r2_combined_v1", "r2_bundle_v1"),
                 ("labels_all_filled.csv", "labels_r2_bundle_filled.csv")):
        assert a in h, a
        h = h.replace(a, b)
    for a, b in LOCK_JS.items():
        assert h.count(a) == 1, a
        h = h.replace(a, b)
    h = h.replace("</style>", ".tab.locked{opacity:.45;cursor:not-allowed}</style>", 1)
    html = h.replace("__SETS__", json.dumps(sets, ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(html)
    print("bundle: %s -> %s (sha256 %s)" % ([(s["id"], len(s["items"])) for s in sets], OUT,
                                            hashlib.sha256(html.encode("utf-8")).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
