"""PREREG_R3 §2.4 공용 시트 템플릿 — 판정자당 HTML 한 파일, 4단계 잠금. LLM 0.

`make_combined_sheet.HTML` 의 CSS(색·글꼴·문장/근거 상자·하단 버튼 줄)를 그대로 가져오고, 첫 화면 안내는
`r2_bundle_sheet.GUIDE` 형식(작업 개요 → 순서와 잠금 → 단계별 판단 기준 → 번역기 규칙 → 상의 금지 → 끝나면 CSV 하나)을 따른다.
JS 는 R3 장치 때문에 새로 썼다(두 축 라벨 C·K, 스크롤 관문, 최소 체류, 번역 기록, 첫 답 잠금·피드백, 문항별 시간 기록,
2→3단계 쉬는 화면과 "3단계 시작" 뒤 1·2단계 읽기 전용, 2초 넘는 tick = idle, "나머지 컨텍스트" 펼친 횟수).
화면에 보이는 글(제목·제목줄·파일 이름)에는 라운드 표시를 넣지 않는다(`r3_sheet.leak_check`).

자리표시자: __CSS__, __DATA__, __GUIDE_COUNTS__, __TITLE__.
`?test=1` 은 시험용으로만 체류 하한을 3초로 줄이고, localStorage 키에 `_test` 를 붙이며, CSV 의 sheet_version 에 `-test` 를 남긴다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_combined_sheet import HTML as _COMBINED  # noqa: E402

BASE_CSS = re.search(r"<style>(.*?)</style>", _COMBINED, re.S).group(1)
# 문항 표시는 원 시트와 같고, 근거 상자 높이만 §2.4(55vh 고정, 내부 스크롤)로 바꾼다
assert "max-height:46vh" in BASE_CSS
BASE_CSS = BASE_CSS.replace("max-height:46vh", "height:55vh")

EXTRA_CSS = r"""
  .tab.locked{opacity:.45;cursor:not-allowed}
  .meta{font-size:13px;color:var(--mut);margin-top:14px}
  .ans{white-space:pre-wrap;font-size:16px;line-height:1.65;padding:12px 16px;border-radius:10px;
       border:1px solid var(--acc)}
  .q{font-size:17px;padding:12px 16px;border-radius:10px;border:2px solid var(--acc);
     background:color-mix(in srgb,var(--acc) 7%,transparent)}
  .ch{margin:0 0 14px} .chh{font-weight:600;font-size:13px;color:var(--acc)}
  mark{background:#fff2a8;color:inherit} @media (prefers-color-scheme:dark){mark{background:#6b5d12}}
  .gate{font-size:13px;color:var(--mut)} .gate b{color:var(--fg)}
  .tr label{margin-right:10px;font-size:13.5px;white-space:nowrap}
  .grp{display:flex;gap:6px;flex-wrap:wrap;align-items:center} .grp .lab{font-size:13px;color:var(--mut);min-width:44px}
  button:disabled{opacity:.35;cursor:not-allowed}
  .opt.on{background:var(--acc);color:#fff;border-color:var(--acc)}
  .fb{margin:8px auto 0;max-width:860px;padding:8px 12px;border-radius:8px;border:1.5px solid var(--ok);display:none}
  .testbanner{background:#a33;color:#fff;text-align:center;font-weight:700;padding:4px}
  footer{max-height:48vh;overflow:auto}
  .rub td{border-top:1px solid var(--line);padding:4px 6px;vertical-align:top;font-size:13px}
"""

GUIDE = r"""
<div id="guide" style="position:fixed;inset:0;z-index:50;background:var(--bg,#fff);overflow:auto">
<div style="max-width:780px;margin:0 auto;padding:28px 20px;line-height:1.7;font-size:15px">
<h2 style="margin-top:0">라벨 작업 안내</h2>
<p>영어로 된 질문·답·근거 글을 읽고 버튼을 고르는 작업입니다. <b>4단계, __N_ALL__문항</b>이고
5~6시간쯤 걸립니다(문항마다 최소 읽기 시간이 있습니다). 나눠서 하셔도 됩니다.
진행 상황은 <b>같은 컴퓨터의 같은 브라우저</b>에서만 기억됩니다(시크릿 창은 기억하지 않습니다).
쉬는 것을 권합니다: 2단계 중간에 한 번, 2단계와 3단계 사이에 한 번(각 30분 이상, 강제하지 않음).</p>
<h3>순서와 잠금</h3>
<p>1단계(__N1__) → 2단계(__N2__) → 3단계(__N3__) → 4단계(__N4__). 앞 단계를 다 채워야 다음 단계가 열립니다(잠긴 단계에는 🔒).</p>
<ul>
<li><b>1단계 (연습, 답의 품질)</b>: C와 K를 고르고 "저장하고 다음 →"을 누르면 첫 답이 잠기고 참고 답이 보입니다. "확인"을 누르면 다음 문항으로 갑니다.</li>
<li><b>2단계 (본 판정, 답의 품질)</b>: 피드백이 없습니다. 3단계를 시작하기 전까지는 돌아와 고칠 수 있습니다(처음 답과 고친 횟수가 기록됩니다).</li>
<li><b>쉬는 화면</b>: 2단계를 다 채우면 쉬는 화면이 나옵니다. 준비되면 <b>"3단계 시작"</b>을 누르세요. <b>누른 뒤에는 1·2단계 답을 바꿀 수 없습니다</b>(볼 수는 있습니다).</li>
<li><b>3단계 (연습, 문장 지지)</b>: 라벨을 고르면 첫 답이 잠기고 참고 라벨만 보입니다(설명은 없습니다). "확인" 뒤 다음 문항.</li>
<li><b>4단계 (본 판정, 문장 지지)</b>: 피드백이 없습니다. 고칠 수 있습니다.</li>
</ul>
<h3>낱말 뜻</h3>
<ul>
<li><b>청크</b> = 근거 글의 한 조각(문서의 한 단락쯤). 화면에는 "(문서 제목 > 절 이름)"과 함께 보입니다.</li>
<li><b>컨텍스트</b> = 모델이 답을 쓸 때 받은 청크 전부. 위의 "인용된 청크"와 아래에 접힌 "나머지 컨텍스트"를 합친 것입니다.</li>
<li><b>인용</b> = 답 속의 [1], [2] 표시. 답이 그 번호의 청크를 근거로 삼았다는 뜻입니다.</li>
</ul>
<h3>판단 방법</h3>
<ul>
<li><b>1·2단계</b> — 질문, 모델의 답, 그 답이 받은 컨텍스트를 봅니다. 답마다 두 가지를 따로 고릅니다.
<ul>
<li><b>C 정답성(근거 대비)</b>: C2a 지지(인용) = 모든 사실 주장이 답이 인용한 청크([1], [2] …)에서 확인됨(요약·바꿔 쓰기 허용) /
C2b 지지(다른 청크) = 모두 컨텍스트에서 확인되지만 적어도 하나는 인용하지 않은 청크에서만 확인됨 /
C1 미지지 = 적어도 한 주장이 컨텍스트 어디에도 없음(모순은 아님) /
C0 오류 = 적어도 한 주장이 컨텍스트와 <b>모순</b>(숫자·날짜·이름·주체·인과), 또는 다른 대상에 대한 근거를 그 대상인 양 씀.</li>
<li>"사실 주장" = 참·거짓을 따질 수 있는 문장·절. 연결어, 질문 되풀이, "근거에 따르면" 같은 틀 문장은 세지 않습니다.</li>
<li><b>K 완전성(질문 대비)</b>: K2 완전 = 질문의 모든 부분에 직접 답 / K1 부분 = 일부만, 또는 간접·일반적 / K0 비응답 = 다른 주제, 회피, 배경만.</li>
<li>예시(가상, 질문 "언제, 누가 다리를 설계했나?"):
<ul>
<li>답 "1932년 J. Smith가 설계했다 [1]", [1]에 둘 다 있음 → <b>C2a · K2</b>.</li>
<li>[1]에 연도만 있고 설계자 이름은 인용하지 않은 다른 청크에만 있음 → <b>C2b</b>.</li>
<li><b>C1과 C0의 차이</b>: 설계자 이름이 컨텍스트 <b>어디에도 없으면</b> C1 미지지. 컨텍스트에 "A. Brown이 설계했다"처럼 <b>다른 값이 적혀 있으면</b> C0 오류. [1]에 "1923년"이라고 적혀 있는데 답이 1932년이라고 하면 C0.</li>
<li>연도만 답하고 설계자는 말하지 않음 → <b>K1</b>.</li>
<li><b>K0 예</b>: 답 "그 다리는 강 위에 놓인 철교다 [1]" — 맞는 말이어도 질문(언제·누가)에 답하지 않고 배경만 말함 → K0. C는 따로 판단합니다(여기서는 [1]에 있으면 C2a).</li>
<li><b>"컨텍스트에 X가 없다"고 말하는 답</b>: 예 "주어진 자료에는 설계자가 나오지 않는다." 이런 문장은 컨텍스트에 대한 말이라 사실 주장으로 세지 않습니다.
답이 이것뿐이면 <b>C2a · K0</b>를 고릅니다(실제로 컨텍스트에 설계자가 있어도 K0). 다른 사실 주장이 함께 있으면 C는 그 주장들로, K는 질문에 답한 정도로 판단합니다.</li>
</ul></li>
<li>기준은 <b>보여 준 컨텍스트뿐</b>입니다. 바깥 지식으로 맞아 보여도 컨텍스트에 없으면 <b>"C1 미지지"</b>입니다.</li>
<li>노란 하이라이트는 찾기 도움일 뿐입니다. 하이라이트에 없으면 "나머지 컨텍스트"를 펼쳐 확인한 뒤, 어디에도 없을 때 "C1 미지지"를 고르세요.
인용된 청크는 위에 펼쳐 있고, 나머지 컨텍스트는 아래에 접혀 있습니다(C2b인지 보려면 펼쳐야 합니다).</li>
</ul></li>
<li><b>3·4단계</b> — 문장 하나와 그 문장이 인용한 근거 글을 봅니다. 문항 위의 판단 기준(지지 / 불지지 / 판단불가)만 따릅니다.</li>
<li>근거는 <b>끝까지</b> 읽어 주세요. 근거가 길다는 이유로 불지지를 주지 마세요. 브라우저 찾기(Ctrl+F)를 써도 됩니다.</li>
<li>근거 상자를 끝까지 스크롤하고 최소 읽기 시간이 지나야 버튼이 켜집니다(남은 시간이 버튼 위에 보입니다). 다른 탭을 보는 동안이나 컴퓨터가 절전·잠금 상태인 동안은 시간이 가지 않습니다.</li>
<li>번역기를 써도 됩니다. 다만 <b>판단은 번역기에 맡기지 마세요</b>. 문항마다 번역기를 어디에 썼는지(안 씀 / 문장만 / 근거 일부 / 근거 전체) 골라야 버튼이 켜집니다.</li>
<li>인터넷 검색이나 배경지식 없이, 화면에 보이는 글만 보고 판단해 주세요. 다른 사람과 상의하지 말고 혼자 판단해 주세요.</li>
<li>3·4단계는 키보드 1·2·3으로도 고를 수 있습니다(버튼이 켜졌을 때만). ← →로 앞뒤 문항을 오갑니다.</li>
</ul>
<h3>시작 전에 (한 번만)</h3>
<p>쓸 번역 도구 이름(안 쓰면 "없음"): <input id="g_tool" style="font:inherit;width:260px"><br>
영어 실력 자기 평가 한 줄(예: "논문 초록은 사전 없이 읽음"): <input id="g_prof" style="font:inherit;width:420px"></p>
<h3>끝나면</h3>
<p>위쪽에 "전체 n / __N_ALL__"이 나오고, 다 채우면 "전부 완료"가 뜹니다. 그때 <b>"CSV 내려받기 (전체)"</b>를 눌러
내려받은 파일(<b>__CSVNAME__</b>) 하나를 보내 주세요.</p>
<p style="text-align:center;margin-top:28px"><button id="startbtn" style="font-size:17px;padding:10px 28px">시작하기</button></p>
</div></div>
<div id="brk" style="position:fixed;inset:0;z-index:45;background:var(--bg,#fff);overflow:auto;display:none">
<div style="max-width:680px;margin:0 auto;padding:48px 20px;line-height:1.75;font-size:16px;text-align:center">
<h2>2단계를 모두 채웠습니다</h2>
<p>여기서 쉬어 가기를 권합니다(30분 이상, 강제하지 않음). 창을 닫아도 진행은 저장됩니다.</p>
<p>다음은 3단계(문장 지지 연습)와 4단계입니다. <b>"3단계 시작"을 누르면 1·2단계 답은 더 바꿀 수 없습니다</b>(보기만 할 수 있습니다).
고칠 답이 있으면 "2단계로 돌아가기"를 눌러 고친 뒤, 3단계 탭(또는 2단계 마지막 문항의 "다음 →")으로 이 화면에 다시 오세요.</p>
<p style="margin-top:28px"><button id="brkback" style="font-size:15px;padding:8px 18px">2단계로 돌아가기</button>
<button id="s3btn" style="font-size:17px;padding:10px 28px;margin-left:12px">3단계 시작</button></p>
</div></div>
"""

BODY = r"""
<header>
  <div class="tabs" id="tabs"></div>
  <div class="row">
    <strong id="pos"></strong><span class="sp"></span>
    <span class="mut" id="cnt"></span>
    <button id="guidebtn">안내 보기</button>
    <button id="dl">CSV 내려받기 (전체)</button>
  </div><div class="bar"><i id="pb"></i></div>
</header>
<div class="wrap" id="view"></div>
<footer>
  <div class="btns" style="flex-direction:column;gap:6px">
    <div class="tr" id="trrow"><span class="mut">번역기 사용(필수):</span></div>
    <div class="gate" id="gate"></div>
    <div id="lab"></div>
    <div class="row"><span class="sp"></span>
      <button id="prev">← 이전</button><button id="next">다음 →</button><button id="jump">안 채운 곳으로</button></div>
  </div>
  <div class="fb" id="fb"></div>
</footer>
"""

JS = r"""
const D = __DATA__;
const TEST = new URLSearchParams(location.search).get("test") === "1";
const VERSION = D.version + (TEST ? "-test" : "");
const KEY = D.store_key + (TEST ? "_test" : "");
const TEST_TMIN = 3;
const TR = ["안 씀", "문장만", "근거 일부", "근거 전체"];
const CS = [["C2a", "C2a 지지(인용)"], ["C2b", "C2b 지지(다른 청크)"], ["C1", "C1 미지지"], ["C0", "C0 오류"]];
const KS = [["K2", "K2 완전"], ["K1", "K1 부분"], ["K0", "K0 비응답"]];
const S = D.stages;
const S3 = S.findIndex(s => s.kind === "v" && s.mode === "practice");   // 3단계 위치(보정 참조 시트에는 없음 → -1)
const IDLE_GAP_MS = 2000;                                               // 한 tick 이 2초를 넘으면(절전·잠금·멈춤) active 가 아니라 idle
let st = {start: null, it: {}, pos: [0, 0], s3_started_at: ""};
try { const s = JSON.parse(localStorage.getItem(KEY) || "null"); if (s && s.it) st = s; } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} };
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const iso = () => new Date().toISOString();
const rec = id => st.it[id] || (st.it[id] = {label: "", C: "", K: "", first_label: "", first_C: "", first_K: "",
  n_changes: 0, active_ms: 0, total_ms: 0, hidden_ms: 0, idle_ms: 0, t_bottom_ms: "", scrolled: 0, max_frac: 0,
  scroll_events: 0, visits: 0, trans: "", shown_at: "", answered_at: "", locked: 0, confirmed: 0, fold_opens: 0});
// 3단계를 시작한 뒤(쉬는 화면의 "3단계 시작" 또는 3단계 문항이 한 번이라도 보였으면) 1·2단계는 보기만 된다(§2.4)
const s2Locked = () => S3 >= 0 && (!!st.s3_started_at || S[S3].items.some(x => st.it[x.id] && st.it[x.id].shown_at));
const qLocked = s => s.kind === "q" && s2Locked();
const breakOpen = () => document.getElementById("brk").style.display !== "none";
const feedbackStage = s => s.mode === "calib" || s.mode === "practice";
function itemDone(s, x) {
  const r = st.it[x.id]; if (!r) return false;
  const a = s.kind === "v" ? !!r.label : (!!r.C && !!r.K);
  return a && (!feedbackStage(s) || !!r.confirmed);
}
const nDone = s => s.items.filter(x => itemDone(s, x)).length;
const unlocked = k => S.slice(0, k).every(s => nDone(s) === s.items.length);
let si = 0, i = 0, curId = null, pend = {C: "", K: ""};
if (Array.isArray(st.pos)) { si = st.pos[0] | 0; i = st.pos[1] | 0; }
if (!(si >= 0 && si < S.length) || !unlocked(si) || !(i >= 0 && i < S[si].items.length)) { si = 0; i = 0; }
if (S3 >= 0 && si >= S3 && !s2Locked()) { si = S3 - 1; i = S[si].items.length - 1; }   // 3단계는 "3단계 시작"으로만 연다
const guideOpen = () => document.getElementById("guide").style.display !== "none";
const tminEff = x => TEST ? Math.min(TEST_TMIN, x.tmin) : x.tmin;

// ---------- 시간: 직전 tick 의 가시 상태로 dt 를 나눈다(숨겨졌다 돌아온 시간이 active 로 들어가지 않게)
let last = performance.now(), vis = document.visibilityState === "visible";
function tick() {
  const now = performance.now(), dt = now - last; last = now;
  const wasVis = vis; vis = document.visibilityState === "visible";
  if (!curId || guideOpen() || breakOpen() || qLocked(S[si])) return;   // 잠긴 1·2단계 문항은 다시 봐도 시간을 더하지 않는다
  const r = rec(curId);
  r.total_ms += dt;
  if (!wasVis) r.hidden_ms += dt;
  else if (dt > IDLE_GAP_MS) r.idle_ms = (r.idle_ms || 0) + dt;      // 절전·화면 잠금 뒤 깨어난 한 번의 긴 tick
  else r.active_ms += dt;
  updateGate();
}
document.addEventListener("visibilitychange", () => { tick(); save(); });
let nt = 0;
setInterval(() => { tick(); if (++nt % 8 === 0) save(); }, 250);

// ---------- 표시
function hl(text, spans) {
  let out = "", p = 0;
  for (const [a, b] of (spans || [])) { out += esc(text.slice(p, a)) + "<mark>" + esc(text.slice(a, b)) + "</mark>"; p = b; }
  return out + esc(text.slice(p));
}
function renderTabs() {
  const t = document.getElementById("tabs"); t.innerHTML = "";
  S.forEach((s, k) => {
    const b = document.createElement("button"), n = nDone(s);
    b.className = "tab" + (k === si ? " on" : "") + (n === s.items.length ? " full" : "");
    b.innerHTML = (unlocked(k) ? "" : "🔒 ") + esc(s.name) + '<span class="c">' + n + "/" + s.items.length + "</span>";
    if (!unlocked(k)) { b.className += " locked"; b.onclick = () => alert("앞 단계를 끝까지 채워야 열립니다."); }
    else b.onclick = () => goStage(k, Math.max(0, s.items.findIndex(x => !itemDone(s, x))));
    t.appendChild(b);
  });
}
function goStage(k, idx) {
  if (S3 >= 0 && k >= S3 && !s2Locked()) { showBreak(); return; }
  si = k; i = idx; render();
}
function showBreak() { tick(); document.getElementById("brk").style.display = "block"; }
function viewQ(s, x) {
  let h = '<div class="meta">관점: <b>' + esc(x.persp) + "</b> · " + esc(x.id) + "</div>";
  h += '<details id="rules"' + (i === 0 ? " open" : "") + '><summary>판단 기준 (C 정답성 · K 완전성) — 누르면 펼침/접힘</summary>' + D.rubric_html + "</details>";
  h += "<h2>질문</h2><div class=\"q\">" + esc(x.q) + "</div>";
  h += "<h2>모델의 답</h2><div class=\"ans\">" + esc(x.ans) + "</div>";
  h += "<h2>인용된 청크 — 끝까지 스크롤해 주세요</h2><div class=\"ev\" id=\"gatebox\">";
  h += x.cited.length ? x.cited.map(c => '<div class="ch"><div class="chh">[' + c.n + "] (" + esc(c.h) + ")</div>" + hl(c.t, c.hl) + "</div>").join("")
                      : '<div class="mut">(답이 인용한 청크가 컨텍스트에 없다)</div>';
  h += "</div>";
  h += '<details id="otherctx"><summary>나머지 컨텍스트 (인용되지 않은 청크 ' + x.other.length + "개) — 누르면 펼침</summary>" +
       x.other.map(c => '<div class="ch"><div class="chh">(' + esc(c.h) + ")</div>" + hl(c.t, c.hl) + "</div>").join("") + "</details>";
  return h;
}
function viewV(s, x) {
  let h = '<div class="meta">' + esc(x.id) + "</div>";
  h += '<details id="rules" open><summary>판단 기준 (누르면 접힘)</summary>' + x.rules.split(" / ").map(y => "<p>· " + esc(y) + "</p>").join("") + "</details>";
  h += "<h2>" + esc(D.h_claim) + '</h2><div class="claim">' + esc(x.claim) + "</div>";
  h += "<h2>" + esc(D.h_ev) + ' — 끝까지 스크롤해 주세요</h2><div class="ev" id="gatebox">' + esc(x.ev) + "</div>";
  return h;
}
function labButtons(s, x, r) {
  const L = document.getElementById("lab");
  if (s.kind === "v") {
    L.innerHTML = '<div class="grp">' + D.v_labels.map((v, k) =>
      '<button class="opt b' + (k + 1) + (r.label === v ? " on" : "") + '" data-v="' + esc(v) + '">' + esc(v) + " <kbd>" + (k + 1) + "</kbd></button>").join("") + "</div>";
    L.querySelectorAll("[data-v]").forEach(b => b.onclick = () => pickV(b.dataset.v));
  } else {
    if (!pend.C && r.C) pend.C = r.C; if (!pend.K && r.K) pend.K = r.K;
    L.innerHTML = '<div class="grp"><span class="lab">C</span>' + CS.map(([v, t]) =>
      '<button class="opt" data-c="' + v + '">' + esc(t) + "</button>").join("") + "</div>" +
      '<div class="grp" style="margin-top:6px"><span class="lab">K</span>' + KS.map(([v, t]) =>
      '<button class="opt" data-k="' + v + '">' + esc(t) + "</button>").join("") +
      '<span class="sp"></span><button id="saveq">저장하고 다음 →</button></div>';
    L.querySelectorAll("[data-c]").forEach(b => b.onclick = () => { pend.C = b.dataset.c; markQ(); updateGate(); });
    L.querySelectorAll("[data-k]").forEach(b => b.onclick = () => { pend.K = b.dataset.k; markQ(); updateGate(); });
    document.getElementById("saveq").onclick = saveQ;
    markQ();
  }
}
function markQ() {
  document.querySelectorAll("[data-c]").forEach(b => b.classList.toggle("on", b.dataset.c === pend.C));
  document.querySelectorAll("[data-k]").forEach(b => b.classList.toggle("on", b.dataset.k === pend.K));
}
function trRow(r) {
  const T = document.getElementById("trrow");
  T.innerHTML = '<span class="mut">번역기 사용(필수):</span> ' + TR.map(v =>
    '<label><input type="radio" name="tr" value="' + esc(v) + '"' + (r.trans === v ? " checked" : "") + "> " + esc(v) + "</label>").join("");
  T.querySelectorAll("input").forEach(e => e.onchange = () => { rec(curId).trans = e.value; save(); updateGate(); });
}
function gateOpen(x, r) { return !!r.scrolled && r.active_ms >= tminEff(x) * 1000 && !!r.trans; }
function updateGate() {
  if (!curId) return;
  const s = S[si], x = s.items[i], r = rec(x.id);
  const left = Math.max(0, Math.ceil(tminEff(x) - r.active_ms / 1000));
  const ro = qLocked(s);
  const ok = gateOpen(x, r) && !ro, lock = (feedbackStage(s) && r.locked) || ro;
  const msg = [];
  if (!r.scrolled) msg.push("근거를 끝까지 스크롤");
  if (left > 0) msg.push("최소 읽기 시간 <b>" + left + "초</b> 남음");
  if (!r.trans) msg.push("번역기 사용 선택");
  document.getElementById("gate").innerHTML = (ro ? "3단계를 시작했으므로 1·2단계 답은 더 바꿀 수 없습니다(보기만 됩니다)." :
    lock ? "첫 답이 잠겼습니다. 아래 참고 답을 보고 \"확인\"을 누르세요." :
    (ok ? "버튼이 켜졌습니다." : "버튼을 켜려면: " + msg.join(" · "))) + (TEST ? " [TEST MODE: 체류 " + TEST_TMIN + "초]" : "");
  document.querySelectorAll("#lab button").forEach(b => { b.disabled = !ok || !!lock; });
  const sv = document.getElementById("saveq");
  if (sv) sv.disabled = !ok || !!lock || !pend.C || !pend.K;
  document.querySelectorAll("#trrow input").forEach(e => { e.disabled = !!lock; });
}
function onScroll(e) {
  if (qLocked(S[si])) return;
  const g = e.target, r = rec(curId);
  r.scroll_events++;
  const f = g.scrollHeight > 0 ? Math.min(1, (g.scrollTop + g.clientHeight) / g.scrollHeight) : 1;
  if (f > r.max_frac) r.max_frac = f;
  if (!r.scrolled && g.scrollTop + g.clientHeight >= g.scrollHeight - 4) { r.scrolled = 1; r.t_bottom_ms = Math.round(r.active_ms); }
  updateGate();
}
function checkFit() {
  const g = document.getElementById("gatebox"); if (!g || !curId) return;
  const r = rec(curId);
  if (g.scrollHeight <= g.clientHeight + 4) { if (!r.scrolled) { r.scrolled = 1; r.t_bottom_ms = Math.round(r.active_ms); } r.max_frac = 1; }
  updateGate();
}
function showFeedback(s, x, r) {
  const F = document.getElementById("fb");
  if (!(feedbackStage(s) && r.locked)) { F.style.display = "none"; F.innerHTML = ""; return; }
  const ref = s.kind === "v" ? ("참고 라벨: <b>" + esc(x.ref.label) + "</b>")
    : ("참고 답: <b>" + esc(CS.find(c => c[0] === x.ref.C)[1]) + "</b> · <b>" + esc(KS.find(k => k[0] === x.ref.K)[1]) + "</b>");
  const mine = s.kind === "v" ? esc(r.first_label) : esc(r.first_C + " · " + r.first_K);
  F.innerHTML = "내 첫 답: " + mine + " — " + ref + ' <button id="okbtn">확인</button>';
  F.style.display = "block";
  document.getElementById("okbtn").onclick = () => { r.confirmed = 1; save(); advance(); };
}
function render() {
  const s = S[si], x = s.items[i];
  if (curId !== x.id && !guideOpen()) {
    pend = {C: "", K: ""};
    const r = rec(x.id); if (!qLocked(s)) r.visits++; if (!r.shown_at) r.shown_at = iso();   // 잠긴 1·2단계 문항은 다시 봐도 기록을 바꾸지 않는다
    curId = x.id;
  }
  const r = rec(x.id);
  renderTabs();
  document.getElementById("pos").textContent = s.name + "  " + (i + 1) + " / " + s.items.length;
  document.getElementById("view").innerHTML = (s.kind === "q" ? viewQ : viewV)(s, x);
  const g = document.getElementById("gatebox");
  g.scrollTop = 0; g.addEventListener("scroll", onScroll);
  const oc = document.getElementById("otherctx");                 // "나머지 컨텍스트" 펼친 횟수(§2.4 ctx_fold_opens)
  if (oc) oc.addEventListener("toggle", () => { if (oc.open && !qLocked(S[si])) { const rr = rec(x.id); rr.fold_opens = (rr.fold_opens || 0) + 1; save(); } });
  trRow(r); labButtons(s, x, r); showFeedback(s, x, r);
  padForFooter();
  const tot = S.reduce((a, y) => a + y.items.length, 0), got = S.reduce((a, y) => a + nDone(y), 0);
  let note = "";
  if (s.mode === "main" && s.kind === "q" && nDone(s) >= Math.floor(s.items.length / 2) && nDone(s) < s.items.length) note = " · 2단계 절반 — 쉬어 가기를 권합니다";
  if (qLocked(s)) note = " · 3단계 시작 뒤라 이 단계는 보기만 됩니다";
  document.getElementById("cnt").innerHTML = (got === tot ? '<span class="done">전부 완료 — CSV 를 내려받으세요</span>' : ("전체 " + got + " / " + tot)) + esc(note);
  document.getElementById("pb").style.width = (100 * got / tot) + "%";
  st.pos = [si, i]; save();
  updateGate();
  requestAnimationFrame(checkFit); setTimeout(checkFit, 120);
}
// 고정된 아래 버튼 줄이 페이지 끝("나머지 컨텍스트" 접힘)을 가리지 않게 본문 아래 여백을 버튼 줄 높이에 맞춘다
function padForFooter() { const f = document.querySelector("footer"); document.getElementById("view").style.paddingBottom = (f.offsetHeight + 32) + "px"; }
window.addEventListener("resize", padForFooter);
function commitCommon(r) { r.answered_at = iso(); }
function pickV(v) {
  const s = S[si], x = s.items[i], r = rec(x.id);
  if (!gateOpen(x, r) || (feedbackStage(s) && r.locked)) return;
  if (!r.first_label) r.first_label = v; else if (r.label !== v) r.n_changes++;
  r.label = v; commitCommon(r);
  if (feedbackStage(s)) { r.locked = 1; save(); render(); return; }
  save(); advance();
}
function saveQ() {
  const s = S[si], x = s.items[i], r = rec(x.id);
  if (qLocked(s) || !gateOpen(x, r) || !pend.C || !pend.K || (feedbackStage(s) && r.locked)) return;
  if (!r.first_C) { r.first_C = pend.C; r.first_K = pend.K; }
  else if (r.C !== pend.C || r.K !== pend.K) r.n_changes++;
  r.C = pend.C; r.K = pend.K; commitCommon(r);
  if (feedbackStage(s)) { r.locked = 1; save(); render(); return; }
  save(); advance();
}
function advance() {
  const s = S[si];
  if (i < s.items.length - 1) { i++; render(); return; }
  if (si < S.length - 1 && unlocked(si + 1)) { goStage(si + 1, 0); return; }   // 2단계 끝 → 쉬는 화면
  render();
  if (si < S.length - 1) alert("이 단계에 안 채운 문항이 있습니다. \"안 채운 곳으로\"를 누르세요.");
}
document.getElementById("prev").onclick = () => { if (i > 0) { i--; render(); } else if (si > 0) { si--; i = S[si].items.length - 1; render(); } };
document.getElementById("next").onclick = () => {
  if (i < S[si].items.length - 1) { i++; render(); }
  else if (si < S.length - 1 && unlocked(si + 1)) goStage(si + 1, 0);
  else if (si < S.length - 1) alert("이 단계에 안 채운 문항이 있습니다. \"안 채운 곳으로\"를 누르세요.");
};
document.getElementById("jump").onclick = () => {
  for (let k = 0; k < S.length; k++) {
    if (!unlocked(k)) break;
    const j = S[k].items.findIndex(x => !itemDone(S[k], x));
    if (j >= 0) { goStage(k, j); return; }
  }
  alert("전부 채웠습니다. \"CSV 내려받기 (전체)\"를 누르세요.");
};
document.getElementById("s3btn").onclick = () => {
  if (S3 < 0 || !unlocked(S3)) return;
  if (!st.s3_started_at) st.s3_started_at = iso();
  save(); document.getElementById("brk").style.display = "none"; last = performance.now();
  si = S3; i = 0; render();
};
document.getElementById("brkback").onclick = () => { document.getElementById("brk").style.display = "none"; last = performance.now(); render(); };
document.onkeydown = e => {
  if (guideOpen() || breakOpen()) return;
  const inp = e.target.tagName === "INPUT", radio = inp && e.target.type === "radio";
  if (inp && !radio) return;                       // 첫 화면 글 칸
  const s = S[si];
  if (s.kind === "v" && ["1", "2", "3"].includes(e.key)) { e.preventDefault(); pickV(D.v_labels[+e.key - 1]); return; }
  if (radio) return;                               // 번역 라디오 위의 ← → 는 라디오 선택 이동(기본 동작)
  else if (e.key === "ArrowLeft") document.getElementById("prev").click();
  else if (e.key === "ArrowRight") document.getElementById("next").click();
};

// ---------- CSV
const COLS = __COLS__;
function buildCsv() {
  const q = v => '"' + String(v == null ? "" : v).replace(/"/g, '""') + '"';
  const rows = [COLS.join(",")];
  const start = st.start || {};
  S.forEach((s, k) => s.items.forEach(x => {
    const r = st.it[x.id] || rec(x.id);
    const v = {stage: k + 1, item_id: x.id, label: s.kind === "v" ? r.label : "", C: s.kind === "q" ? r.C : "", K: s.kind === "q" ? r.K : "",
      first_label: s.kind === "v" ? r.first_label : "", first_C: s.kind === "q" ? r.first_C : "", first_K: s.kind === "q" ? r.first_K : "",
      n_changes: r.n_changes, active_ms: Math.round(r.active_ms), total_ms: Math.round(r.total_ms), hidden_ms: Math.round(r.hidden_ms),
      t_bottom_ms: r.t_bottom_ms, scrolled_to_bottom: r.scrolled ? 1 : 0, max_scroll_frac: (+r.max_frac).toFixed(3),
      scroll_events: r.scroll_events, revisits: Math.max(0, r.visits - 1), t_min_s: x.tmin, translator_used: r.trans,
      translator_tool: start.tool || "", proficiency: start.prof || "", shown_at_iso: r.shown_at, answered_at_iso: r.answered_at,
      sheet_version: VERSION, annotator: D.annotator, ref_label: "", ref_C: "", ref_K: "", correct_first: "",
      idle_ms: Math.round(r.idle_ms || 0), ctx_fold_opens: s.kind === "q" ? (r.fold_opens || 0) : "", s3_started_at_iso: st.s3_started_at || ""};
    if (feedbackStage(s)) {
      if (s.kind === "v") { v.ref_label = x.ref.label; v.correct_first = r.first_label ? (r.first_label === x.ref.label ? 1 : 0) : ""; }
      else { v.ref_C = x.ref.C; v.ref_K = x.ref.K; v.correct_first = r.first_C ? ((r.first_C === x.ref.C && r.first_K === x.ref.K) ? 1 : 0) : ""; }
    }
    rows.push(COLS.map(c => q(v[c])).join(","));
  }));
  return rows.join("\r\n");
}
if (TEST) {                                        // 시험 판(?test=1)에서만 콘솔 훅을 연다(실제 시트에는 없음)
  window.buildCsv = buildCsv;
  window.__r3 = {state: () => st, pos: () => [si, i], tminEff, unlocked, s2Locked, TEST, VERSION, KEY};
}
document.getElementById("dl").onclick = () => {
  const blob = new Blob(["\ufeff" + buildCsv()], {type: "text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = D.csv_name; a.click();
};

// ---------- 첫 화면
const G = document.getElementById("guide");
if (st.start) { document.getElementById("g_tool").value = st.start.tool; document.getElementById("g_prof").value = st.start.prof; }
const startOk = () => document.getElementById("g_tool").value.trim() && document.getElementById("g_prof").value.trim();
const sb = document.getElementById("startbtn");
const syncStart = () => { sb.disabled = !startOk(); };
["g_tool", "g_prof"].forEach(id => document.getElementById(id).oninput = syncStart); syncStart();
sb.onclick = () => {
  if (!startOk()) return;
  const t = document.getElementById("g_tool").value.trim(), p = document.getElementById("g_prof").value.trim();
  st.start = Object.assign({started_at: iso()}, st.start || {}, {tool: t, prof: p});
  save(); G.style.display = "none"; last = performance.now(); render();
};
document.getElementById("guidebtn").onclick = () => { tick(); G.style.display = "block"; };
if (TEST) document.body.insertAdjacentHTML("afterbegin", '<div class="testbanner">TEST MODE (?test=1) — 체류 하한 ' + TEST_TMIN + "초, 기록에 -test 표시. 실제 판정에 쓰지 마세요</div>");
render();
"""

PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>__CSS__</style></head><body>
__GUIDE__
__BODY__
<script>
(function () {   // 함수 범위로 감싼다: 페이지 상태·buildCsv 가 전역(콘솔)에 나오지 않게(시험 판의 훅만 예외, §2.4 8)
__JS__
})();
</script></body></html>
"""

RUBRIC_HTML = (
    '<table class="rub">'
    "<tr><td><b>C2a</b> 지지(인용)<br><i>supported by cited chunks</i></td><td>모든 사실 주장이 답이 인용한 청크에서 확인된다. 요약·바꿔 쓰기 허용</td></tr>"
    "<tr><td><b>C2b</b> 지지(다른 청크)<br><i>supported by context, not (only) cited</i></td><td>모든 주장이 컨텍스트에서 확인되지만 적어도 하나는 인용하지 않은 청크에서만 확인된다</td></tr>"
    "<tr><td><b>C1</b> 미지지<br><i>unsupported</i></td><td>적어도 한 주장이 컨텍스트 어디에서도 확인되지 않는다(모순은 아님)</td></tr>"
    "<tr><td><b>C0</b> 오류<br><i>contradicted / wrong entity</i></td><td>적어도 한 주장이 컨텍스트와 모순된다(숫자·날짜·이름·주체·인과), 또는 다른 대상에 대한 근거를 그 대상인 양 썼다</td></tr>"
    "<tr><td><b>K2</b> 완전<br><i>complete</i></td><td>질문의 모든 부분에 직접 답한다</td></tr>"
    "<tr><td><b>K1</b> 부분<br><i>partial</i></td><td>일부만, 또는 간접·일반적으로 답한다</td></tr>"
    "<tr><td><b>K0</b> 비응답<br><i>non-answer</i></td><td>다른 주제, 회피, 배경만</td></tr>"
    "</table><p class=\"mut\">청크 = 근거 글 조각, 컨텍스트 = 모델이 받은 청크 전부, 인용 = 답의 [1][2]가 가리키는 청크. "
    "기준은 보여 준 컨텍스트뿐이다. 바깥 지식으로 맞아 보여도 컨텍스트에 없으면 'C1 미지지'. "
    "컨텍스트에 다른 값이 적혀 있으면(모순) 'C0 오류'. "
    "'사실 주장' = 참·거짓을 따질 수 있는 문장·절(연결어·질문 되풀이·'근거에 따르면' 같은 틀 문장, '자료에 X가 없다' 같은 컨텍스트에 대한 말은 세지 않는다; "
    "답이 그것뿐이면 C2a · K0). "
    "하이라이트는 찾기 도움일 뿐이다. 하이라이트에 없으면 '나머지 컨텍스트'를 펼쳐 확인한 뒤, 어디에도 없을 때 'C1 미지지'를 고른다.</p>"
)


def render_page(*, title: str, data_json: str, cols_json: str, counts: dict, csv_name: str) -> str:
    guide = GUIDE
    for k, v in {"__N_ALL__": counts["all"], "__N1__": counts[1], "__N2__": counts[2], "__N3__": counts[3],
                 "__N4__": counts[4], "__CSVNAME__": csv_name}.items():
        assert k in guide, k
        guide = guide.replace(k, str(v))
    data_json = data_json.replace("</", "<\\/")       # 문항 글 안의 "</script" 가 블록을 끊지 않게(JSON 값은 같다)
    js = JS.replace("__DATA__", data_json).replace("__COLS__", cols_json)
    page = PAGE
    for k, v in (("__TITLE__", title), ("__CSS__", BASE_CSS + EXTRA_CSS), ("__GUIDE__", guide), ("__BODY__", BODY)):
        page = page.replace(k, v)
    return page.replace("__JS__", js)
