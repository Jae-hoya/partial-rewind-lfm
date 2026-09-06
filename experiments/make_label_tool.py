"""라벨 60건을 사람이 실제로 채울 수 있는 도구를 만든다. 호출 0.

CSV 를 그대로 주면 안 된다. 근거 칸이 중앙 3,826자, 최대 29,578자다 —
엑셀 셀에서 읽을 수 있는 분량이 아니다. 그래서 한 화면에 한 문항씩 보이는
단일 HTML 파일을 만든다. 인터넷 없이 파일을 더블클릭하면 열린다.

들어가는 것:
  - 문항 60개를 JSON 으로 파일 안에 넣는다 (외부 의존 0)
  - 판정 규칙은 접어 두고 필요할 때 편다
  - 판정할 문장을 크게, 근거는 스크롤 되는 상자에
  - 문장이 인용한 청크 표시 [d...] 를 근거에서 강조
  - 버튼 세 개 + 키보드 1/2/3, 화살표로 이동
  - 답은 브라우저에 자동 저장(localStorage) — 창을 닫아도 남는다
  - 다 채우면 CSV 를 내려받아 저장소에 넣는다

**정답을 파일에 넣지 않는다.** 추정자 점수도 층도 없다. 그게 이 라벨의 전부다 —
사람이 기준이 되어야 관문의 순환이 깨진다.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/vprime/labels_stage1.csv"
OUT = ROOT / "results/vprime/labeler.html"

HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V′ 라벨 — 60건</title>
<style>
  :root{--bg:#fbfbfa;--fg:#1a1a19;--mut:#6b6b68;--line:#e0e0dc;--acc:#2b5cab;
        --ok:#1c7c4a;--no:#a33;--unk:#7a6a2a;}
  @media (prefers-color-scheme:dark){:root{--bg:#17181a;--fg:#e8e8e6;--mut:#9a9a97;
        --line:#31333a;--acc:#7aa2e8;--ok:#5cc38a;--no:#e08b8b;--unk:#d8c07a;}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.65 -apple-system,"Segoe UI",Roboto,"Malgun Gothic",sans-serif}
  header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
         padding:10px 16px;z-index:5}
  .bar{height:4px;background:var(--line);border-radius:2px;overflow:hidden;margin-top:8px}
  .bar i{display:block;height:100%;background:var(--acc);transition:width .2s}
  .wrap{max-width:860px;margin:0 auto;padding:0 16px 120px}
  .row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  .mut{color:var(--mut);font-size:13px}
  details{margin:10px 0;border:1px solid var(--line);border-radius:8px;padding:8px 12px}
  summary{cursor:pointer;font-weight:600;font-size:13px}
  details p{margin:8px 0 2px;font-size:13.5px}
  h2{font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--mut);
     margin:22px 0 6px}
  .claim{font-size:19px;line-height:1.6;padding:16px 18px;border-radius:10px;
         border:2px solid var(--acc);background:color-mix(in srgb,var(--acc) 7%,transparent)}
  .ev{white-space:pre-wrap;font-size:14px;max-height:46vh;overflow:auto;padding:14px 16px;
      border:1px solid var(--line);border-radius:10px;background:color-mix(in srgb,var(--fg) 3%,transparent)}
  mark{background:color-mix(in srgb,var(--acc) 28%,transparent);color:inherit;
       border-radius:3px;padding:0 2px}
  footer{position:fixed;left:0;right:0;bottom:0;background:var(--bg);
         border-top:1px solid var(--line);padding:10px 16px}
  .btns{max-width:860px;margin:0 auto;display:flex;gap:10px;flex-wrap:wrap}
  button{font:inherit;padding:11px 16px;border-radius:9px;border:1.5px solid var(--line);
         background:transparent;color:var(--fg);cursor:pointer}
  button:hover{border-color:var(--acc)}
  .b1{border-color:var(--ok);color:var(--ok)} .b2{border-color:var(--no);color:var(--no)}
  .b3{border-color:var(--unk);color:var(--unk)}
  .b1.on{background:var(--ok);color:#fff} .b2.on{background:var(--no);color:#fff}
  .b3.on{background:var(--unk);color:#fff}
  .sp{flex:1} kbd{border:1px solid var(--line);border-radius:4px;padding:0 5px;font-size:12px}
  .done{color:var(--ok);font-weight:600}
</style></head><body>
<header><div class="row">
  <strong id="pos"></strong><span class="mut" id="idl"></span><span class="sp"></span>
  <span class="mut" id="cnt"></span>
  <button id="dl">CSV 내려받기</button>
</div><div class="bar"><i id="pb"></i></div></header>

<div class="wrap">
  <details id="rules"><summary>판정 규칙 (누르면 펼침)</summary><div id="rulebody"></div></details>
  <h2>판정할 문장</h2>
  <div class="claim" id="claim"></div>
  <h2>근거 — 이 문장이 인용한 청크 전문</h2>
  <div class="ev" id="ev"></div>
</div>

<footer><div class="btns">
  <button class="b1" data-v="지지">지지 <kbd>1</kbd></button>
  <button class="b2" data-v="불지지">불지지 <kbd>2</kbd></button>
  <button class="b3" data-v="판단불가">판단불가 <kbd>3</kbd></button>
  <span class="sp"></span>
  <button id="prev">← 이전</button><button id="next">다음 →</button>
  <button id="jump">안 채운 곳으로</button>
</div></footer>

<script>
const ITEMS = __ITEMS__;
const KEY = "vprime_stage1_v1";
let ans = {}; try { ans = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let i = 0;

const esc = s => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(ans)); } catch (e) {} };

function render() {
  const it = ITEMS[i];
  document.getElementById("pos").textContent = (i + 1) + " / " + ITEMS.length;
  document.getElementById("idl").textContent = it.id;
  document.getElementById("rulebody").innerHTML =
    it.rules.split(" / ").map(x => "<p>· " + esc(x) + "</p>").join("");
  document.getElementById("claim").textContent = it.claim;
  // 문장이 단 인용 표시를 근거에서 강조한다
  const marks = [...new Set((it.claim.match(/\\[d[0-9a-z]+\\]/g) || []))];
  let ev = esc(it.ev);
  for (const m of marks) ev = ev.split(esc(m)).join("<mark>" + esc(m) + "</mark>");
  document.getElementById("ev").innerHTML = ev;
  document.getElementById("ev").scrollTop = 0;
  const n = Object.keys(ans).length;
  document.getElementById("cnt").innerHTML = n === ITEMS.length
    ? '<span class="done">60건 완료 — 내려받으세요</span>' : ("채움 " + n + " / " + ITEMS.length);
  document.getElementById("pb").style.width = (100 * n / ITEMS.length) + "%";
  document.querySelectorAll("[data-v]").forEach(b =>
    b.classList.toggle("on", ans[it.id] === b.dataset.v));
}
function pick(v) {
  ans[ITEMS[i].id] = v; save();
  if (i < ITEMS.length - 1) i++;           // 고르면 자동으로 다음
  render();
}
document.querySelectorAll("[data-v]").forEach(b =>
  b.onclick = () => pick(b.dataset.v));
document.getElementById("prev").onclick = () => { if (i > 0) { i--; render(); } };
document.getElementById("next").onclick = () => { if (i < ITEMS.length - 1) { i++; render(); } };
document.getElementById("jump").onclick = () => {
  const k = ITEMS.findIndex(x => !ans[x.id]);
  if (k < 0) { alert("전부 채웠습니다. CSV 를 내려받으세요."); return; }
  i = k; render();
};
document.onkeydown = e => {
  if (e.key === "1") pick("지지");
  else if (e.key === "2") pick("불지지");
  else if (e.key === "3") pick("판단불가");
  else if (e.key === "ArrowLeft") document.getElementById("prev").click();
  else if (e.key === "ArrowRight") document.getElementById("next").click();
};
document.getElementById("dl").onclick = () => {
  const q = s => '"' + String(s).replace(/"/g, '""') + '"';
  const rows = [["item_id", "human_label"].join(",")]
    .concat(ITEMS.map(x => [q(x.id), q(ans[x.id] || "")].join(",")));
  const blob = new Blob(["\\ufeff" + rows.join("\\r\\n")], {type: "text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels_stage1_filled.csv";
  a.click();
};
render();
</script></body></html>
"""


def main() -> int:
    rows = list(csv.DictReader(io.open(SRC, encoding="utf-8-sig")))
    items = [{"id": r["item_id"], "rules": r["판정규칙"],
              "claim": r["문장"], "ev": r["근거"]} for r in rows]
    html = HTML.replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
    io.open(OUT, "w", encoding="utf-8").write(html)
    ev = [len(x["ev"]) for x in items]
    print("문항 %d개 -> %s (%.0f KB)" % (len(items), OUT, OUT.stat().st_size / 1024))
    print("근거 길이 중앙 %d자 · 최대 %d자 — CSV 로는 못 읽을 분량이라 이 도구를 만들었다."
          % (sorted(ev)[len(ev) // 2], max(ev)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
