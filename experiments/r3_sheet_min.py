"""R3 A·B 간소 시트 (DEVIATIONS 33). 생성 LLM 0, 모델 0.

d810136 의 `results/r3/sheets/labeler_A.html`·`labeler_B.html` 에서 문항(같은 id·내용·순서·단계)을 그대로 꺼내
**문항 + 선택지 + CSV 내려받기만** 남긴 시트를 만든다.
  - 뺀 것: 안내·쉬는 화면, 시간 측정, 최소 읽기 시간, 스크롤 관문, 번역 기록, 단계 잠금, 연습 피드백(참고 답).
  - 남긴 것: 문항 표시(질문·답·청크·하이라이트, 문장·근거), 판단 기준 표/문구, 선택 버튼, 진행 저장(브라우저), CSV.
  - CSV 는 stage·item_id·label·C·K·sheet_version·annotator 7칸뿐이다(시간 칸 없음). 받은 원본은 그대로 커밋하고,
    `r3_pad_v2.py` 가 등록 머리줄(COLS)의 나머지 칸을 빈칸으로 채운 사본을 만들어
    `r3_reliability.py`·`r3_quality_score.py` 를 고치지 않고 `--sheet-version r3_sheet_v2` 로 돌린다.
  - 저장 키 `r3_A_v2`/`r3_B_v2`(이전 판 진행이 섞이지 않게), 판 `r3_sheet_v2`, 파일 이름 `labels_A.csv`/`labels_B.csv`.

  python experiments/r3_sheet_min.py    # -> results/r3/sheets_v2/labeler_A.html, labeler_B.html
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
import r3_sheet_template as T  # noqa: E402
from r3_sheet import COLS  # noqa: E402  (채점 쪽 머리줄: r3_pad_v2.py 가 맞춘다)

OUT_COLS = ["stage", "item_id", "label", "C", "K", "sheet_version", "annotator"]   # 시트가 내보내는 칸(시간 칸 없음)

VERSION = "r3_sheet_v2"
SRC = ROOT / "results/r3/sheets"
OUT = ROOT / "results/r3/sheets_v2"
DROP_ITEM_KEYS = ("ref", "tmin")          # 참고 답(피드백 없음)·최소 시간은 시트에 싣지 않는다

BODY = r"""
<header>
  <div class="tabs" id="tabs"></div>
  <div class="row">
    <strong id="pos"></strong><span class="sp"></span>
    <span class="mut" id="cnt"></span>
    <button id="dl">CSV 내려받기</button>
  </div><div class="bar"><i id="pb"></i></div>
</header>
<div class="wrap" id="view"></div>
<footer>
  <div class="btns" style="flex-direction:column;gap:6px">
    <div id="lab"></div>
    <div class="row"><span class="sp"></span>
      <button id="prev">← 이전</button><button id="next">다음 →</button><button id="jump">안 채운 곳으로</button></div>
  </div>
</footer>
"""

JS = r"""
const D = __DATA__;
const COLS = __COLS__;
const KEY = D.store_key, VERSION = D.version;
const CS = [["C2a", "C2a 지지(인용)"], ["C2b", "C2b 지지(다른 청크)"], ["C1", "C1 미지지"], ["C0", "C0 오류"]];
const KS = [["K2", "K2 완전"], ["K1", "K1 부분"], ["K0", "K0 비응답"]];
const S = D.stages;
let st = {it: {}, pos: [0, 0]};
try { const s = JSON.parse(localStorage.getItem(KEY) || "null"); if (s && s.it) st = s; } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} };
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const rec = id => st.it[id] || (st.it[id] = {label: "", C: "", K: ""});
const done = (s, x) => { const r = st.it[x.id]; return !!r && (s.kind === "v" ? !!r.label : (!!r.C && !!r.K)); };
const nDone = s => s.items.filter(x => done(s, x)).length;
let si = 0, i = 0, pend = {C: "", K: ""}, cur = null;
if (Array.isArray(st.pos)) { si = st.pos[0] | 0; i = st.pos[1] | 0; }
if (!(si >= 0 && si < S.length) || !(i >= 0 && i < S[si].items.length)) { si = 0; i = 0; }
function hl(text, spans) {
  let out = "", p = 0;
  for (const [a, b] of (spans || [])) { out += esc(text.slice(p, a)) + "<mark>" + esc(text.slice(a, b)) + "</mark>"; p = b; }
  return out + esc(text.slice(p));
}
function viewQ(x) {
  let h = '<div class="meta">관점: <b>' + esc(x.persp) + "</b> · " + esc(x.id) + "</div>";
  h += '<details id="rules"' + (i === 0 ? " open" : "") + '><summary>판단 기준 (C 정답성 · K 완전성) — 누르면 펼침/접힘</summary>' + D.rubric_html + "</details>";
  h += '<h2>질문</h2><div class="q">' + esc(x.q) + "</div>";
  h += '<h2>모델의 답</h2><div class="ans">' + esc(x.ans) + "</div>";
  h += '<h2>인용된 청크</h2><div class="ev">';
  h += x.cited.length ? x.cited.map(c => '<div class="ch"><div class="chh">[' + c.n + "] (" + esc(c.h) + ")</div>" + hl(c.t, c.hl) + "</div>").join("")
                      : '<div class="mut">(답이 인용한 청크가 컨텍스트에 없다)</div>';
  h += "</div>";
  h += '<details><summary>나머지 컨텍스트 (인용되지 않은 청크 ' + x.other.length + "개) — 누르면 펼침</summary>" +
       x.other.map(c => '<div class="ch"><div class="chh">(' + esc(c.h) + ")</div>" + hl(c.t, c.hl) + "</div>").join("") + "</details>";
  return h;
}
function viewV(x) {
  let h = '<div class="meta">' + esc(x.id) + "</div>";
  h += '<details id="rules" open><summary>판단 기준 (누르면 접힘)</summary>' + x.rules.split(" / ").map(y => "<p>· " + esc(y) + "</p>").join("") + "</details>";
  h += "<h2>" + esc(D.h_claim) + '</h2><div class="claim">' + esc(x.claim) + "</div>";
  h += "<h2>" + esc(D.h_ev) + '</h2><div class="ev">' + esc(x.ev) + "</div>";
  return h;
}
function renderTabs() {
  const t = document.getElementById("tabs"); t.innerHTML = "";
  S.forEach((s, k) => {
    const b = document.createElement("button"), n = nDone(s);
    b.className = "tab" + (k === si ? " on" : "") + (n === s.items.length ? " full" : "");
    b.innerHTML = esc(s.name) + '<span class="c">' + n + "/" + s.items.length + "</span>";
    b.onclick = () => { si = k; i = Math.max(0, s.items.findIndex(x => !done(s, x))); render(); };
    t.appendChild(b);
  });
}
function labButtons(s, x) {
  const L = document.getElementById("lab"), r = rec(x.id);
  if (s.kind === "v") {
    L.innerHTML = '<div class="grp">' + D.v_labels.map((v, k) =>
      '<button class="opt' + (r.label === v ? " on" : "") + '" data-v="' + esc(v) + '">' + esc(v) + " <kbd>" + (k + 1) + "</kbd></button>").join("") + "</div>";
    L.querySelectorAll("[data-v]").forEach(b => b.onclick = () => pickV(b.dataset.v));
  } else {
    L.innerHTML = '<div class="grp"><span class="lab">C</span>' + CS.map(([v, t]) => '<button class="opt" data-c="' + v + '">' + esc(t) + "</button>").join("") + "</div>" +
      '<div class="grp" style="margin-top:6px"><span class="lab">K</span>' + KS.map(([v, t]) => '<button class="opt" data-k="' + v + '">' + esc(t) + "</button>").join("") +
      '<span class="sp"></span><button id="saveq">저장하고 다음 →</button></div>';
    L.querySelectorAll("[data-c]").forEach(b => b.onclick = () => { pend.C = b.dataset.c; markQ(); });
    L.querySelectorAll("[data-k]").forEach(b => b.onclick = () => { pend.K = b.dataset.k; markQ(); });
    document.getElementById("saveq").onclick = saveQ;
    markQ();
  }
}
function markQ() {
  document.querySelectorAll("[data-c]").forEach(b => b.classList.toggle("on", b.dataset.c === pend.C));
  document.querySelectorAll("[data-k]").forEach(b => b.classList.toggle("on", b.dataset.k === pend.K));
  const sv = document.getElementById("saveq"); if (sv) sv.disabled = !pend.C || !pend.K;
}
function render() {
  const s = S[si], x = s.items[i];
  if (cur !== x.id) { const r = st.it[x.id] || {}; pend = {C: r.C || "", K: r.K || ""}; cur = x.id; }
  renderTabs();
  document.getElementById("pos").textContent = s.name + "  " + (i + 1) + " / " + s.items.length;
  document.getElementById("view").innerHTML = s.kind === "q" ? viewQ(x) : viewV(x);
  labButtons(s, x);
  const f = document.querySelector("footer"); document.getElementById("view").style.paddingBottom = (f.offsetHeight + 32) + "px";
  const tot = S.reduce((a, y) => a + y.items.length, 0), got = S.reduce((a, y) => a + nDone(y), 0);
  document.getElementById("cnt").innerHTML = got === tot ? '<span class="done">전부 완료 — CSV 를 내려받으세요</span>' : ("전체 " + got + " / " + tot);
  document.getElementById("pb").style.width = (100 * got / tot) + "%";
  st.pos = [si, i]; save();
  window.scrollTo(0, 0);
}
function next() {
  if (i < S[si].items.length - 1) i++; else if (si < S.length - 1) { si++; i = 0; }
  render();
}
function pickV(v) { rec(S[si].items[i].id).label = v; save(); next(); }
function saveQ() {
  if (!pend.C || !pend.K) return;
  const r = rec(S[si].items[i].id); r.C = pend.C; r.K = pend.K; save(); next();
}
document.getElementById("prev").onclick = () => { if (i > 0) i--; else if (si > 0) { si--; i = S[si].items.length - 1; } render(); };
document.getElementById("next").onclick = next;
document.getElementById("jump").onclick = () => {
  for (let k = 0; k < S.length; k++) { const j = S[k].items.findIndex(x => !done(S[k], x)); if (j >= 0) { si = k; i = j; render(); return; } }
  alert("전부 채웠습니다. \"CSV 내려받기\"를 누르세요.");
};
document.onkeydown = e => {
  if (S[si].kind === "v" && ["1", "2", "3"].includes(e.key)) { pickV(D.v_labels[+e.key - 1]); return; }
  if (e.key === "ArrowLeft") document.getElementById("prev").click();
  else if (e.key === "ArrowRight") document.getElementById("next").click();
};
function buildCsv() {
  const q = v => '"' + String(v == null ? "" : v).replace(/"/g, '""') + '"';
  const rows = [COLS.join(",")];
  S.forEach((s, k) => s.items.forEach(x => {
    const r = st.it[x.id] || {};
    const v = {stage: k + 1, item_id: x.id, label: s.kind === "v" ? (r.label || "") : "", C: s.kind === "q" ? (r.C || "") : "",
               K: s.kind === "q" ? (r.K || "") : "", sheet_version: VERSION, annotator: D.annotator};
    rows.push(COLS.map(c => q(v[c])).join(","));
  }));
  return rows.join("\r\n");
}
document.getElementById("dl").onclick = () => {
  const blob = new Blob(["\ufeff" + buildCsv()], {type: "text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = D.csv_name; a.click();
};
render();
"""


def load_data(p: Path) -> dict:
    h = io.open(p, encoding="utf-8").read()
    m = re.search(r"const D = (\{.*?\});\nconst TEST", h, re.S)
    assert m, p
    return json.loads(m.group(1).replace("<\\/", "</"))


def build(annot: str) -> tuple[str, dict]:
    d = load_data(SRC / ("labeler_%s.html" % annot))
    assert d["annotator"] == annot and d["csv_name"] == "labels_%s.csv" % annot
    for s in d["stages"]:
        s["items"] = [{k: v for k, v in x.items() if k not in DROP_ITEM_KEYS} for x in s["items"]]
    d.update(version=VERSION, store_key="r3_%s_v2" % annot)
    data_json = json.dumps(d, ensure_ascii=False).replace("</", "<\\/")
    js = JS.replace("__DATA__", data_json).replace("__COLS__", json.dumps(OUT_COLS))
    page = ('<!doctype html>\n<html lang="ko"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1">\n<title>라벨 작업</title>\n'
            "<style>" + T.BASE_CSS + T.EXTRA_CSS + "</style></head><body>\n" + BODY +
            "<script>\n(function () {\n" + js + "})();\n</script></body></html>\n")
    return page, d


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"spec": "PREREG_R3 A/B sheets, minimal v2 (DEVIATIONS 33)", "source": "results/r3/sheets/labeler_{A,B}.html (d810136)",
            "sheet_version": VERSION, "dropped": "guide/break screens, timing, min reading time, scroll gate, translator record, stage locks, practice feedback"}
    for a in ("A", "B"):
        page, d = build(a)
        for bad in ('"ref"', '"tmin"', "performance.now", "toISOString", "setInterval"):   # 참고 답·시간 장치가 없어야 한다
            assert bad not in page, (a, bad)
        io.open(OUT / ("labeler_%s.html" % a), "w", encoding="utf-8", newline="\n").write(page)
        meta[a] = {"n_items": sum(len(s["items"]) for s in d["stages"]), "stages": [len(s["items"]) for s in d["stages"]],
                   "order": [x["id"] for s in d["stages"] for x in s["items"]][:5],
                   "html_sha256": hashlib.sha256(page.encode("utf-8")).hexdigest()}
    io.open(OUT / "_meta_v2.json", "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
