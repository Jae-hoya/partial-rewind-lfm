"""라벨 시트 4종을 HTML 하나로 묶는다. 호출 0.

문항은 각 시트 HTML 에 들어 있는 ITEMS 를 그대로 꺼내 쓴다 — 다시 뽑지 않으므로 표본·순서·블라인드가
개별 시트와 바이트 단위로 같다. 세트마다 판정 규칙과 버튼이 다르므로 섞지 않고 탭으로 나눈다.
CSV 는 한 파일(set, item_id, human_label)로 내려받는다. 채점 때 set 열로 다시 나눈다.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/d18/labeler_all.html"

SETS = [
    ("stage2", "V′ 2단계", "results/vprime/labeler_stage2.html", ["지지", "불지지", "판단불가"],
     "판정할 문장", "근거 — 이 문장이 인용한 청크 전문"),
    ("h2dev", "H2-dev", "results/d18/labeler_h2dev.html", ["답함", "답안함", "일부"],
     "질문 (sub-question)", "모델의 답 (인용 표시는 지웠다)"),
    ("h2hoB", "H2-held-out-B", "results/d18/labeler_h2hoB.html", ["답있음", "답없음", "일부"],
     "질문 (sub-question)", "컨텍스트 — 모델이 받은 이 질문의 검색 청크 (실린 그대로)"),
    ("h2hoA", "H2-held-out-A", "results/d18/labeler_h2hoA.html", ["답함", "답안함", "일부"],
     "질문 (sub-question)", "모델의 답 (인용 표시는 지웠다)"),
]

HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>재투고 라벨 — 통합</title>
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
  .tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
  .tab{font:inherit;font-size:13px;padding:6px 10px;border-radius:8px;border:1.5px solid var(--line);
       background:transparent;color:var(--fg);cursor:pointer}
  .tab.on{border-color:var(--acc);background:color-mix(in srgb,var(--acc) 12%,transparent);font-weight:600}
  .tab .c{color:var(--mut);font-weight:400;margin-left:4px}
  .tab.full .c{color:var(--ok);font-weight:600}
  .bar{height:4px;background:var(--line);border-radius:2px;overflow:hidden;margin-top:8px}
  .bar i{display:block;height:100%;background:var(--acc);transition:width .2s}
  .wrap{max-width:860px;margin:0 auto;padding:0 16px 120px}
  .row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  .mut{color:var(--mut);font-size:13px}
  details{margin:10px 0;border:1px solid var(--line);border-radius:8px;padding:8px 12px}
  summary{cursor:pointer;font-weight:600;font-size:13px}
  details p{margin:8px 0 2px;font-size:13.5px}
  h2{font-size:13px;letter-spacing:.04em;color:var(--mut);margin:22px 0 6px}
  .claim{font-size:19px;line-height:1.6;padding:16px 18px;border-radius:10px;
         border:2px solid var(--acc);background:color-mix(in srgb,var(--acc) 7%,transparent)}
  .ev{white-space:pre-wrap;font-size:14px;max-height:46vh;overflow:auto;padding:14px 16px;
      border:1px solid var(--line);border-radius:10px;background:color-mix(in srgb,var(--fg) 3%,transparent)}
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
<header>
  <div class="tabs" id="tabs"></div>
  <div class="row">
    <strong id="pos"></strong><span class="mut" id="idl"></span><span class="sp"></span>
    <span class="mut" id="cnt"></span>
    <button id="dl">CSV 내려받기 (전체)</button>
  </div><div class="bar"><i id="pb"></i></div>
</header>

<div class="wrap">
  <details id="rules" open><summary>판정 규칙 — 세트마다 다르다 (누르면 접힘)</summary><div id="rulebody"></div></details>
  <h2 id="h_claim"></h2>
  <div class="claim" id="claim"></div>
  <h2 id="h_ev"></h2>
  <div class="ev" id="ev"></div>
</div>

<footer><div class="btns">
  <button class="b1" data-k="0"><span></span> <kbd>1</kbd></button>
  <button class="b2" data-k="1"><span></span> <kbd>2</kbd></button>
  <button class="b3" data-k="2"><span></span> <kbd>3</kbd></button>
  <span class="sp"></span>
  <button id="prev">← 이전</button><button id="next">다음 →</button>
  <button id="jump">안 채운 곳으로</button>
</div></footer>

<script>
const SETS = __SETS__;
const KEY = "r2_combined_v1";
let ans = {}; try { ans = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
for (const s of SETS) ans[s.id] = ans[s.id] || {};
let si = 0, i = 0;
try { const p = JSON.parse(localStorage.getItem(KEY + "_pos") || "null"); if (p) { si = p[0]; i = p[1]; } } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(ans));
                           localStorage.setItem(KEY + "_pos", JSON.stringify([si, i])); } catch (e) {} };
const esc = s => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const filled = s => Object.keys(ans[s.id]).filter(k => s.items.some(x => x.id === k)).length;

function renderTabs() {
  const t = document.getElementById("tabs"); t.innerHTML = "";
  SETS.forEach((s, k) => {
    const b = document.createElement("button");
    const n = filled(s);
    b.className = "tab" + (k === si ? " on" : "") + (n === s.items.length ? " full" : "");
    b.innerHTML = esc(s.name) + '<span class="c">' + n + "/" + s.items.length + "</span>";
    b.onclick = () => { si = k; i = 0; const j = s.items.findIndex(x => !ans[s.id][x.id]); if (j >= 0) i = j; render(); };
    t.appendChild(b);
  });
}
function render() {
  const s = SETS[si], it = s.items[i];
  renderTabs();
  document.getElementById("pos").textContent = s.name + "  " + (i + 1) + " / " + s.items.length;
  document.getElementById("idl").textContent = it.id;
  document.getElementById("rulebody").innerHTML =
    it.rules.split(" / ").map(x => "<p>· " + esc(x) + "</p>").join("");
  document.getElementById("h_claim").textContent = s.h_claim;
  document.getElementById("h_ev").textContent = s.h_ev;
  document.getElementById("claim").textContent = it.claim;
  document.getElementById("ev").textContent = it.ev;
  document.getElementById("ev").scrollTop = 0;
  document.querySelectorAll("[data-k]").forEach(b => {
    const v = s.labels[+b.dataset.k];
    b.querySelector("span").textContent = v;
    b.classList.toggle("on", ans[s.id][it.id] === v);
  });
  const tot = SETS.reduce((a, x) => a + x.items.length, 0), got = SETS.reduce((a, x) => a + filled(x), 0);
  document.getElementById("cnt").innerHTML = got === tot
    ? '<span class="done">전부 완료 — CSV 를 내려받으세요</span>' : ("전체 " + got + " / " + tot);
  document.getElementById("pb").style.width = (100 * got / tot) + "%";
  save();
}
function pick(k) {
  const s = SETS[si];
  ans[s.id][s.items[i].id] = s.labels[k]; save();
  if (i < s.items.length - 1) i++;
  render();
}
document.querySelectorAll("[data-k]").forEach(b => b.onclick = () => pick(+b.dataset.k));
document.getElementById("prev").onclick = () => { if (i > 0) { i--; render(); } };
document.getElementById("next").onclick = () => { if (i < SETS[si].items.length - 1) { i++; render(); } };
document.getElementById("jump").onclick = () => {
  for (let d = 0; d < SETS.length; d++) {
    const k = (si + d) % SETS.length, j = SETS[k].items.findIndex(x => !ans[SETS[k].id][x.id]);
    if (j >= 0) { si = k; i = j; render(); return; }
  }
  alert("전부 채웠습니다. CSV 를 내려받으세요.");
};
document.onkeydown = e => {
  if (e.key === "1") pick(0); else if (e.key === "2") pick(1); else if (e.key === "3") pick(2);
  else if (e.key === "ArrowLeft") document.getElementById("prev").click();
  else if (e.key === "ArrowRight") document.getElementById("next").click();
};
document.getElementById("dl").onclick = () => {
  const q = s => '"' + String(s).replace(/"/g, '""') + '"';
  const rows = [["set", "item_id", "human_label"].join(",")];
  for (const s of SETS) for (const x of s.items) rows.push([q(s.id), q(x.id), q(ans[s.id][x.id] || "")].join(","));
  const blob = new Blob(["\\ufeff" + rows.join("\\r\\n")], {type: "text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "labels_all_filled.csv"; a.click();
};
render();
</script></body></html>
"""


def items_of(path: Path) -> list[dict]:
    h = io.open(path, encoding="utf-8").read()
    return json.loads(re.search(r"const ITEMS = (\[.*?\]);\n", h, re.S).group(1))


def main() -> int:
    sets = []
    for sid, name, path, labels, h_claim, h_ev in SETS:
        sets.append({"id": sid, "name": name, "labels": labels, "h_claim": h_claim, "h_ev": h_ev,
                     "items": items_of(ROOT / path)})
    ids = [x["id"] for s in sets for x in s["items"]]
    assert len(ids) == len(set(ids)), "세트 간 문항 id 가 겹친다"
    io.open(OUT, "w", encoding="utf-8").write(HTML.replace("__SETS__", json.dumps(sets, ensure_ascii=False)))
    print("통합 시트 -> %s  (%s)" % (OUT, ", ".join("%s %d" % (s["name"], len(s["items"])) for s in sets)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
