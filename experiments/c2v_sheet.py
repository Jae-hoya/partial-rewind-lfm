"""PREREG_R3 §4.6 C-G7 (C2-V) — 코퍼스2 연산자 점검용 저자 40문장 시트. 생성 LLM 0, MiniCheck(GPU) 만.

모집단: 본실험 s1(`results/c2/heldout25`, 모든 arm·라운드)의 answered 문장 단위 — `c2_negative_control.collect` 와
**같은 규칙**(그 라운드에 새로 만든 answered sub-answer, (토픽, 문장, 고유 인용 청크 순서열), 중복 제거·정렬).
각 단위를 원 전제(인용 청크를 SEP 로 이은 단일 문서)로 MiniCheck 채점 → 지지/비지지 두 층.
`default_rng(4504)` 로 층마다 20개(색인 정렬), 그다음 같은 rng 로 40개 순서 셔플.

시트: V′ 시트와 같은 표시(`make_combined_sheet.HTML`, 같은 RULES 3값), MiniCheck 판정·arm·층 미표시.
§2.4 장치 가운데 **스크롤 기록과 번역 기록만** 더한다(체류 하한·단계 잠금 없음). 번역 선택 전에는 라벨 버튼이 꺼진다.
봉인 키(`results/c2/c2v_sealed/_key_c2v.csv`)에 층·MiniCheck 확률·모집단 층 크기를 쓰고 내용을 출력하지 않는다.

  python experiments/c2v_sheet.py            # -> results/c2/c2v/labeler_c2v.html
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import c2_negative_control as NC  # noqa: E402
import r2_analyze as RA  # noqa: E402
from make_combined_sheet import HTML  # noqa: E402
from vprime_make_sheet import RULES  # noqa: E402

SEED, N_PER = 4504, 20
LABELS = ["지지", "불지지", "판단불가"]
VERSION = "c2v_sheet_v1"
TRANS = ["안 씀", "문장만", "근거 일부", "근거 전체"]

# --- 시트 패치: 스크롤·번역 기록만 (문항 표시는 바이트 그대로) ---------------------------------
PATCH_HTML = ('<div id="tr" style="margin:6px 0;font-size:14px">번역기 사용: '
              + "".join('<label style="margin-right:10px"><input type="radio" name="tr" value="%s"> %s</label>' % (t, t)
                        for t in TRANS)
              + ' <span style="color:#a33">(먼저 골라야 라벨 버튼이 켜집니다)</span></div>')
PATCH_JS = r"""
const META = (ans.__meta = ans.__meta || {});
const ensure = id => (META[id] = META[id] || {max_scroll_frac: 0, scrolled_to_bottom: 0, scroll_events: 0,
                                             translator_used: "", first_label: "", n_changes: 0, shown_at_iso: "", answered_at_iso: ""});
const EV = document.getElementById("ev");
const curId = () => SETS[si].items[i].id;
function scrollCheck(ev) {
  const m = ensure(curId());
  const denom = Math.max(1, EV.scrollHeight - EV.clientHeight);
  const frac = EV.scrollHeight <= EV.clientHeight + 4 ? 1 : Math.min(1, EV.scrollTop / denom);
  m.max_scroll_frac = Math.max(m.max_scroll_frac, Math.round(frac * 1000) / 1000);
  if (EV.scrollTop + EV.clientHeight >= EV.scrollHeight - 4) m.scrolled_to_bottom = 1;
  if (ev) m.scroll_events++;
  save();
}
EV.addEventListener("scroll", scrollCheck);
function gate() {
  const m = ensure(curId());
  document.querySelectorAll('input[name="tr"]').forEach(r => r.checked = (r.value === m.translator_used));
  document.querySelectorAll("[data-k]").forEach(b => b.disabled = !m.translator_used);
}
document.querySelectorAll('input[name="tr"]').forEach(r => r.onchange = () => { ensure(curId()).translator_used = r.value; save(); gate(); });
const _render = render;
render = function () {
  _render();
  const m = ensure(curId());
  if (!m.shown_at_iso) m.shown_at_iso = new Date().toISOString();
  setTimeout(() => scrollCheck(null), 0);
  gate();
};
const _pick = pick;
pick = function (k) {
  const m = ensure(curId());
  if (!m.translator_used) return;
  const v = SETS[si].labels[k], prev = ans[SETS[si].id][curId()];
  if (!m.first_label) m.first_label = v; else if (prev && prev !== v) m.n_changes++;
  m.answered_at_iso = new Date().toISOString();
  _pick(k);
};
document.querySelectorAll("[data-k]").forEach(b => b.onclick = () => pick(+b.dataset.k));
document.onkeydown = e => {
  if (e.key === "1") pick(0); else if (e.key === "2") pick(1); else if (e.key === "3") pick(2);
  else if (e.key === "ArrowLeft") document.getElementById("prev").click();
  else if (e.key === "ArrowRight") document.getElementById("next").click();
};
document.getElementById("dl").onclick = () => {
  const q = s => '"' + String(s).replace(/"/g, '""') + '"';
  const cols = ["set", "item_id", "human_label", "first_label", "n_changes", "translator_used", "scrolled_to_bottom",
                "max_scroll_frac", "scroll_events", "shown_at_iso", "answered_at_iso", "sheet_version"];
  const rows = [cols.join(",")];
  for (const s of SETS) for (const x of s.items) {
    const m = ensure(x.id);
    rows.push([s.id, x.id, ans[s.id][x.id] || "", m.first_label, m.n_changes, m.translator_used, m.scrolled_to_bottom,
               m.max_scroll_frac, m.scroll_events, m.shown_at_iso, m.answered_at_iso, "__VERSION__"].map(q).join(","));
  }
  const blob = new Blob(["\ufeff" + rows.join("\r\n")], {type: "text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "labels_c2v_filled.csv"; a.click();
};
render();
"""


def build_html(sheet: list[dict]) -> str:
    sets = [{"id": "c2v", "name": "문장 지지 (40)", "labels": LABELS,
             "h_claim": "문장 (모델이 쓴 답의 한 문장)", "h_ev": "근거 (그 문장이 인용한 글)", "items": sheet}]
    h = HTML
    for a, b in (("재투고 라벨 — 통합", "라벨 — 문장 지지"), ("r2_combined_v1", "c2v_v1"),
                 ("labels_all_filled.csv", "labels_c2v_filled.csv")):
        assert a in h, a
        h = h.replace(a, b)
    # 'const SETS' 가 객체 키 '__meta' 를 세트처럼 세지 않도록 filled() 는 items 로 거른다(원 코드 그대로 안전).
    anchor = '<footer'
    assert h.count(anchor) == 1
    h = h.replace(anchor, PATCH_HTML + "\n" + anchor)
    tail = "render();\n</script></body></html>"
    assert h.rstrip().endswith(tail.strip()), "HTML 끝 형식이 바뀌었다"
    h = h[: h.rstrip().rfind("render();")] + PATCH_JS.replace("__VERSION__", VERSION) + "</script></body></html>\n"
    return h.replace("__SETS__", json.dumps(sets, ensure_ascii=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C2-V 저자 40문장 시트 (LLM 0)")
    ap.add_argument("--run-dir", default="results/c2/heldout25")
    ap.add_argument("--seed-run", type=int, default=1)
    ap.add_argument("--config", default=str(ROOT / "config_c2.yaml"))
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    from lfm.config import Config, load_config
    from lfm.estimator import load_estimator
    base = load_config(args.config)
    d = base.as_dict(); d.setdefault("estimator", {})["kind"] = "minicheck_doc"
    est = load_estimator(Config(d, source="c2v_sheet"))
    chunk_text = RA.load_chunk_text(ROOT / base["paths.index_dir"])
    units, _pool = NC.collect(ROOT / args.run_dir, args.seed_run, chunk_text)
    flags = list(est.grounded_many([(s, [chunk_text[c] for c in cites]) for _q, s, cites in units]))
    pos = [k for k, f in enumerate(flags) if f]
    neg = [k for k, f in enumerate(flags) if not f]
    assert len(pos) >= N_PER and len(neg) >= N_PER, (len(pos), len(neg))
    rng = np.random.default_rng(SEED)
    picked = [("sup", k) for k in sorted(int(x) for x in rng.choice(pos, N_PER, replace=False))] + \
             [("uns", k) for k in sorted(int(x) for x in rng.choice(neg, N_PER, replace=False))]
    order = rng.permutation(len(picked)).tolist()
    sheet, key = [], []
    for j, o in enumerate(order):
        cell, k = picked[o]
        qid, sent, cites = units[k]
        iid = "c2v%03d" % j
        sheet.append({"id": iid, "rules": RULES, "claim": sent, "ev": "\n\n".join(chunk_text[c] for c in cites)})
        key.append({"item_id": iid, "cell": cell, "mc_supported": int(cell == "sup"), "topic": qid,
                    "n_cites": len(cites), "N_cell": len(pos) if cell == "sup" else len(neg), "n_cell": N_PER})
    out, sealed = ROOT / "results/c2/c2v", ROOT / "results/c2/c2v_sealed"
    out.mkdir(parents=True, exist_ok=True); sealed.mkdir(parents=True, exist_ok=True)
    html = build_html(sheet)
    io.open(out / "labeler_c2v.html", "w", encoding="utf-8").write(html)
    with io.open(sealed / "_key_c2v.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0])); w.writeheader(); w.writerows(key)
    meta = {"spec": "PREREG_R3 §4.6 C-G7", "run_dir": args.run_dir, "seed_run": args.seed_run, "rng_seed": SEED,
            "n_units": len(units), "N_sup": len(pos), "N_uns": len(neg), "n_items": len(sheet), "sheet_version": VERSION,
            "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest()}
    io.open(out / "_meta_c2v.json", "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps({k: meta[k] for k in ("n_units", "n_items", "html_sha256")}, ensure_ascii=False))
    print("-> %s  (봉인 키는 열지 않는다)" % (out / "labeler_c2v.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
