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
    'document.onkeydown = e => {':
        'document.onkeydown = e => {\n  if (document.getElementById("guide").style.display !== "none") return;',
}


# 라벨러가 파일 하나만 받도록 안내문을 첫 화면에 넣는다. "시작"을 누르기 전에는 키 입력으로 라벨이 찍히지 않는다
GUIDE = """
<div id="guide" style="position:fixed;inset:0;z-index:50;background:var(--bg,#fff);overflow:auto">
<div style="max-width:760px;margin:0 auto;padding:28px 20px;line-height:1.7;font-size:15px">
<h2 style="margin-top:0">라벨 작업 안내</h2>
<p>영어로 된 질문과 답(또는 문장과 근거 글)을 읽고, 화면의 버튼 하나를 고르는 작업입니다.
<b>4단계, 186문항</b>이고 4~5시간쯤 걸립니다. 나눠서 하셔도 됩니다.
진행 상황은 <b>같은 컴퓨터의 같은 브라우저</b>에서만 기억됩니다(시크릿 창은 기억하지 않습니다).</p>
<h3>순서</h3>
<p>1단계 → 2단계 → 3단계 → 4단계. 앞 단계를 다 채워야 다음 단계가 열립니다(잠긴 단계에는 🔒).
단계를 다 채우면 "다음 →"을 눌러 넘어가세요.</p>
<h3>판단 방법</h3>
<ul>
<li>각 문항 위에 판단 기준이 적혀 있습니다. 그 기준만 따라 주세요. 단계마다 기준이 다릅니다.
<ul>
<li><b>1·3단계</b>: 모델의 답이 질문이 묻는 것에 답하는가 → 답함 / 일부 / 답안함</li>
<li><b>2단계</b>: 질문이 묻는 것에 대한 사실이 컨텍스트(근거 글)에 있는가 → 답있음 / 일부 / 답없음</li>
<li><b>4단계</b>: 문장의 모든 사실이 근거 글에 있거나 직접 따라 나오는가 → 지지 / 불지지 / 판단불가
(부분만 맞아도 불지지. 판단불가는 문장 자체가 모호할 때만. 근거가 모자라면 불지지)</li>
</ul></li>
<li>1~3단계는 답이 맞는지가 아니라, 질문에 대한 답인지(근거가 있는지)만 봅니다.</li>
<li>인터넷 검색이나 배경지식 없이, 화면에 보이는 글만 보고 판단해 주세요.</li>
<li>어려운 영어 문장은 번역기로 한국어로 옮겨 읽으셔도 됩니다. 다만 판단을 번역기에 맡기지는 말아 주세요.
번역기를 썼다면 끝나고 <b>어떤 번역기를, 대략 몇 문항에</b> 썼는지 알려 주세요.</li>
<li>1~3단계에서 애매하면 "일부"를 고르셔도 됩니다. 오래 고민하지 않으셔도 됩니다.</li>
<li>다른 사람과 상의하지 말고 혼자 판단해 주세요.</li>
<li>키보드 1·2·3으로 고르고, ← →로 앞뒤 문항을 오갈 수 있습니다.</li>
</ul>
<h3>끝나면</h3>
<p>위쪽에 "전체 n / 186"이 나오고, 다 채우면 "전부 완료"가 뜹니다. 그때 <b>"CSV 내려받기"</b>를 눌러
내려받은 파일(labels_r2_bundle_filled.csv) 하나를 보내 주세요.</p>
<p style="text-align:center;margin-top:28px"><button id="startbtn" style="font-size:17px;padding:10px 28px"
onclick="document.getElementById('guide').style.display='none'">시작하기</button></p>
</div></div>
<button id="guidebtn" style="position:fixed;right:12px;bottom:70px;z-index:40;opacity:.8"
onclick="document.getElementById('guide').style.display='block'">안내 보기</button>
"""


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
    assert h.count("</style></head><body>") == 1
    h = h.replace("</style></head><body>", "</style></head><body>" + GUIDE, 1)
    html = h.replace("__SETS__", json.dumps(sets, ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(html)
    print("bundle: %s -> %s (sha256 %s)" % ([(s["id"], len(s["items"])) for s in sets], OUT,
                                            hashlib.sha256(html.encode("utf-8")).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
