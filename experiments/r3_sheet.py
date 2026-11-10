"""PREREG_R3 C1 — 판정자 A·B 의 HTML 시트(4단계 잠금), 봉인 키, 안내. **LLM 호출 0, GPU 0.**

  # (1) C1 전: 보정 3문항을 뽑아 저자 참조 라벨 시트를 만든다 (§3.6, seed 4405)
  python experiments/r3_sheet.py calib-draw
  #     → results/r3/calib/labeler_r3_calib_ref.html. 저자가 C·K 를 달아 내려받은 CSV 를
  #       results/r3/calib/labels_r3_calib_ref.csv 로 두고 **커밋**한다.
  # (2) C1: 두 시트 + 봉인 키 (참조 라벨 CSV 가 없으면 멈춘다)
  python experiments/r3_sheet.py build
  # (시험용) 참조 라벨 자리표시자로 results/r3/_test_build/ 에 만든다. 배포 금지 표시가 붙는다
  python experiments/r3_sheet.py build --placeholder-ref

산출 (build)
  results/r3/sheets/labeler_A.html, labeler_B.html, 안내.md   (판정자에게 보이는 이름에는 라운드 표시가 없다)
  results/r3/sealed/_key_r3_v.csv, _key_q.csv, _meta_q.json, _note_overlap.json
단계: 1 품질 보정(3) → 2 품질 본(45슬롯 69 + 점검 4) → [쉬는 화면, "3단계 시작"] → 3 V′ 연습(10) → 4 V′ 시험(C50).
§3 표집은 두 방향 층(E-GR 답함 여부 × 토픽, §3.3).

봉인: 화면·HTML 에는 arm, seed, 질의 id, 청크 id(§3 문항), cell, 점수, 저자 라벨(시험·본 단계)이 없다. 보정·연습 단계의
참조 라벨만 피드백용으로 들어간다(§2.4 공시 1). 이 스크립트는 문항별 arm·cell·라벨을 출력하지 않고 집계만 출력한다.
점검 문항 내용은 출력하지 않고 봉인 키에만 쓴다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from r2_sheets import (ContextBuilder, _included_lengths, load_run, slot_rows, strip_cites,  # noqa: E402
                       with_r1)
from r2_analyze import R1  # noqa: E402
from r3_sheet_template import RUBRIC_HTML, render_page  # noqa: E402

from lfm.branch import format_context  # noqa: E402
from lfm.types import _SENT_RE, CITATION_RE  # noqa: E402

V = ROOT / "results/vprime"
R3 = ROOT / "results/r3"
CALIB_DIR = R3 / "calib"
CALIB_REF = CALIB_DIR / "labels_r3_calib_ref.csv"
VERSION = "r3_sheet_v1"
SALT = "r3-quality-v1"
CAP = 120.0
SEED = {"practice": 4301, "v_A": 4302, "v_B": 4303, "draw": 4401, "q_A": 4402, "q_B": 4403,
        "catch": 4404, "calib": 4405}
N_G1 = 25            # §3.3: E-GR 답한 슬롯 수(층 g1 × 토픽, √N_h 배분)
K_SMALL = 4          # §3.3: g0(E-GR 기권) 토픽 층 중 N_h ≤ 4 는 한 층("g0|small")으로 묶는다.
                     # 규칙 = 기대 본 문항 < 70 인 가장 작은 정수 한계(실제 추출도 ≤ 70). draw_quality 가 규칙을 다시 계산해 assert
MAX_Q_ITEMS = 70     # §3.3: 본 문항 상한
ARMS_Q = ("E-SQ", "E-GR", R1)
V_LABELS = ["지지", "불지지", "판단불가"]
H_CLAIM, H_EV = "문장 (모델이 쓴 답의 한 문장)", "근거 (그 문장이 인용한 글)"
PERSP = {"overview_factual": "개요·사실 (overview / factual)", "technical_mechanism": "기술·작동 방식 (technical / mechanism)",
         "impact_application": "영향·응용 (impact / application)", "controversy_limitation": "논쟁·한계 (controversy / limitation)"}
COLS = ["stage", "item_id", "label", "C", "K", "first_label", "first_C", "first_K", "n_changes", "active_ms", "total_ms",
        "hidden_ms", "t_bottom_ms", "scrolled_to_bottom", "max_scroll_frac", "scroll_events", "revisits", "t_min_s",
        "translator_used", "translator_tool", "proficiency", "shown_at_iso", "answered_at_iso", "sheet_version",
        "annotator", "ref_label", "ref_C", "ref_K", "correct_first", "idle_ms", "ctx_fold_opens", "s3_started_at_iso"]
# 하이라이트 내용어 불용어(고정 목록, §3.5)
STOP = frozenset("""a an the and or but if then than that this these those there their they them his her its our your
my we you he she it is are was were be been being am do does did done has have had having of in on at to from by for
with without within into onto over under about above below after before during between among through across against
as so such not no nor only also very more most less least much many some any each every all both either neither one
two can could may might must shall should will would which who whom whose what when where why how while because
since until upon via per etc i me us him himself herself itself themselves ourselves yourself yourselves than too
own same other another out up down off again further once here just now then still yet ever never""".split())


# --------------------------------------------------------------------------- 공용
def guard_no_llm():
    """시트 생성은 LLM 클라이언트를 만들지 않는다. 만들려 하면 바로 멈춘다(api_calls 0 의 구조적 보장)."""
    import lfm.llm as L

    def _blocked(self, *a, **k):
        raise RuntimeError("PREREG_R3 §0: 시트·채점 스크립트에서 LLM 생성 금지")
    L.LLM.__init__ = _blocked
    return L


def rd(p: Path) -> list[dict]:
    return list(csv.DictReader(io.open(p, encoding="utf-8-sig")))


def sha(s: str, n: int = 6) -> str:
    return hashlib.sha256((SALT + "|" + s).encode("utf-8")).hexdigest()[:n]


def arm_hash(arm: str) -> str:
    return hashlib.sha256((SALT + "|arm|" + arm).encode("utf-8")).hexdigest()[:10]


def tmin(tokens: float) -> float:
    return round(min(CAP, 10.0 + float(tokens) / 12.0), 1)


def items_of(path: Path, var: str = "ITEMS") -> list:
    h = io.open(path, encoding="utf-8").read()
    return json.loads(re.search(r"const %s = (\[.*?\]);\n" % var, h, re.S).group(1))


def write_csv(path: Path, rows: list[dict], cols: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})


# --------------------------------------------------------------------------- §2 V′
def len_bin(t: int) -> str:
    return "short" if t < 613 else ("long" if t >= 2073 else "mid")


def build_v() -> dict:
    key = {}
    for st in ("stage1", "stage2"):
        for r in rd(V / ("_key_%s.csv" % st)):
            key[r["item_id"]] = {**r, "stage": st}
    c50 = rd(V / "c50_sealed/_key_vc50.csv")
    c50src = [r["source_item"] for r in c50]
    assert len(key) == 100, len(key)
    assert len(c50src) == 50 and set(c50src) <= set(key)
    rest = sorted(k for k in key if k not in set(c50src))
    assert len(rest) == 50
    lab = {r["item_id"]: r["human_label"].strip() for r in rd(V / "labels_stage1_filled.csv")}
    for r in rd(ROOT / "results/d18/labels_all_filled.csv"):
        if r["set"] == "stage2":
            lab[r["item_id"]] = r["human_label"].strip()
    assert all(lab.get(k) for k in key), "저자 라벨 100/100"
    assert all(lab.get(s) for s in c50src), "C50 저자 라벨 50/50"
    pool = {it["id"]: it for f in ("labeler.html", "labeler_stage2.html") for it in items_of(V / f)}
    assert set(pool) == set(key)
    vc_sets = items_of(V / "c50/labeler_vc50.html", "SETS")
    assert len(vc_sets) == 1
    vc_items = vc_sets[0]["items"]
    assert vc_sets[0]["labels"] == V_LABELS and vc_sets[0]["h_claim"] == H_CLAIM and vc_sets[0]["h_ev"] == H_EV
    vc_src = {r["item_id"]: r["source_item"] for r in c50}
    for it in vc_items:                    # §2.2: C50 텍스트 = 원 1·2단계 시트 텍스트(바이트)
        src = pool[vc_src[it["id"]]]
        assert (it["rules"], it["claim"], it["ev"]) == (src["rules"], src["claim"], src["ev"]), it["id"]
    pl = {r["item_id"]: int(r["doc_tokens"]) for r in rd(V / "premise_lengths.csv")}
    srt = sorted(pl.values())
    assert len(srt) == 294 and (srt[98], srt[196]) == (613, 2073), (srt[98], srt[196])
    strata = json.load(io.open(V / "stage2_strata.json", encoding="utf-8"))["strata"]
    assert {c: strata[c]["N_h"] for c in strata} == {"P1": 37, "P2": 171, "N1": 77, "N2": 9}
    assert Counter(key[s]["cell"] for s in c50src) == Counter({"P1": 13, "P2": 21, "N1": 12, "N2": 4})
    assert Counter(len_bin(pl[s]) for s in c50src) == Counter({"short": 20, "mid": 16, "long": 14})
    # 연습 10: vprime_c50_sheet 과 같은 비례 최대 나머지 → default_rng(4301), cell sorted(), 후보 item_id 정렬
    by = defaultdict(list)
    for k in rest:
        by[key[k]["cell"]].append(k)
    assert {c: len(v) for c, v in by.items()} == {"P1": 12, "P2": 21, "N1": 13, "N2": 4}
    quota = {c: 10 * len(v) / 50 for c, v in by.items()}
    alloc = {c: int(q) for c, q in quota.items()}
    fr = sorted(quota, key=lambda c: -(quota[c] - alloc[c]))
    assert len({round(quota[c] - alloc[c], 9) for c in fr}) == len(fr), "소수부 동점"
    for c in fr[: 10 - sum(alloc.values())]:
        alloc[c] += 1
    assert alloc == {"P1": 2, "P2": 4, "N1": 3, "N2": 1}, alloc
    rng = np.random.default_rng(SEED["practice"])
    picked = [i for c in sorted(by) for i in rng.choice(sorted(by[c]), alloc[c], replace=False).tolist()]
    assert Counter(len_bin(pl[i]) for i in picked) == Counter({"short": 3, "mid": 6, "long": 1})
    assert sum(lab[i] == "지지" for i in picked) == 7, "§1.3 연습 저자 '지지' 7/10"
    practice = []
    for s in picked:
        src = pool[s]
        practice.append({"id": "vp" + sha("practice|" + s, 4), "rules": src["rules"], "claim": src["claim"],
                         "ev": src["ev"], "tmin": tmin(pl[s]), "ref": {"label": lab[s]}})
    test = [{"id": it["id"], "rules": it["rules"], "claim": it["claim"], "ev": it["ev"],
             "tmin": tmin(pl[vc_src[it["id"]]])} for it in vc_items]
    keyrows = []
    for it, s in zip(practice, picked):
        keyrows.append({"sheet_item_id": it["id"], "stage": "practice", "source_item": s, "cell": key[s]["cell"],
                        "doc_tokens": pl[s], "len_bin": len_bin(pl[s]), "t_min_s": it["tmin"]})
    for it in test:
        s = vc_src[it["id"]]
        keyrows.append({"sheet_item_id": it["id"], "stage": "test", "source_item": s, "cell": key[s]["cell"],
                        "doc_tokens": pl[s], "len_bin": len_bin(pl[s]), "t_min_s": it["tmin"]})
    assert len({r["sheet_item_id"] for r in keyrows}) == 60
    agg = {"practice_alloc": alloc, "practice_bins": dict(Counter(len_bin(pl[i]) for i in picked)),
           "practice_author_support": sum(lab[i] == "지지" for i in picked),
           "c50_bins": dict(Counter(len_bin(pl[s]) for s in c50src)),
           "tmin_h": {"practice": round(sum(x["tmin"] for x in practice) / 3600, 3),
                      "test": round(sum(x["tmin"] for x in test) / 3600, 3)}}
    return {"practice": practice, "test": test, "key": keyrows, "agg": agg}


# --------------------------------------------------------------------------- §3 컨텍스트·표시
class FullContext:
    """ContextBuilder 와 같은 검색·format_context 로 그 branch·라운드 컨텍스트 전체를 다시 만든다.
    `context_sha256` 이 원 실행과 다르면 멈춘다. 반환: [(Chunk, 실린 글자 수)] (모델이 본 순서)."""

    def __init__(self, cb: ContextBuilder):
        self.cb, self.memo, self.qmemo = cb, {}, {}

    def _search(self, q: str, k: int):
        if (q, k) not in self.qmemo:
            self.qmemo[(q, k)] = self.cb.retriever.search(q, k)
        return self.qmemo[(q, k)]

    def get(self, rec: dict, branch: str, produced_round: int):
        mk = (rec.get("run_tag"), str(rec["query_id"]), branch, produced_round, id(rec))
        if mk in self.memo:
            return self.memo[mk]
        rr = next(r for r in rec["rounds"] if r["round"] == produced_round)
        br = next(b for b in rr["branches"] if b["branch"] == branch)
        sas = br["sub_answers"]
        if any(sa.get("produced_round") is not None for sa in sas):
            targets = [sa for sa in sas if sa.get("produced_round") == produced_round]
        else:
            targets = list(sas)
        anchor = (self.cb.anchors.get(str(rec["query_id"])) or "").strip()
        k = self.cb.top_k(produced_round)
        chunks, seen = [], set()
        for sa in targets:
            q = (anchor + " " + sa["sub_question"]).strip() if anchor else sa["sub_question"]
            for c in self._search(q, k):
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    chunks.append(c)
        ctx = format_context(chunks, self.cb.max_chars)
        got = hashlib.sha256(ctx.encode("utf-8")).hexdigest()
        want = br.get("context_sha256") or ""
        assert want and got == want, "context_sha256 불일치 — 시트를 만들지 않는다"
        inc = _included_lengths(chunks, self.cb.max_chars)
        out = ([(c, inc[c.chunk_id]) for c in chunks if c.chunk_id in inc], got)
        self.memo[mk] = out
        return out


def sent_spans(text: str) -> list[tuple[int, int]]:
    out, pos = [], 0
    for piece in _SENT_RE.split(text):
        p = piece.strip()
        if not p:
            continue
        a = text.find(p, pos)
        assert a >= 0
        out.append((a, a + len(p)))
        pos = a + len(p)
    return out


def cwords(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 3 and w not in STOP}


def highlights(answer_plain: str, chunk_texts: list[str]) -> list[list[list[int]]]:
    """답 문장마다 컨텍스트 전체 문장 중 내용어 자카드 최대 2문장(동점은 앞 문장). 반환: 청크별 [[a,b],...]."""
    sents = [(ci, a, b, cwords(t[a:b])) for ci, t in enumerate(chunk_texts) for a, b in sent_spans(t)]
    chosen = set()
    for a0, b0 in sent_spans(answer_plain):
        aw = cwords(answer_plain[a0:b0])
        if not aw:
            continue
        sc = []
        for j, (_ci, _a, _b, w) in enumerate(sents):
            u = aw | w
            jac = len(aw & w) / len(u) if u else 0.0
            if jac > 0:
                sc.append((-jac, j))
        for _negj, j in sorted(sc)[:2]:
            chosen.add(j)
    out = [[] for _ in chunk_texts]
    for j in sorted(chosen):
        ci, a, b, _w = sents[j]
        out[ci].append([a, b])
    return out


CHUNK_ID_RE = re.compile(r"^d\d+s\d+p\d+$")


def render_answer(text: str, ctx_ids: list[str]) -> tuple[str, list[str], str]:
    """인용 표시를 [1]…[k](첫 등장 순)로 다시 번호. 컨텍스트 밖 청크 id 표시는 지운다. 반환 (표시 답, 인용 청크 순서, 표시 없는 답)."""
    inctx, order = set(ctx_ids), []

    def sub(m):
        cid = m.group(1)
        if cid in inctx:
            if cid not in order:
                order.append(cid)
            return "[%d]" % (order.index(cid) + 1)
        return "" if CHUNK_ID_RE.match(cid) else m.group(0)
    shown = re.sub(r"\s+", " ", CITATION_RE.sub(sub, text or "")).replace(" .", ".").strip()
    plain = re.sub(r"\s+", " ", re.sub(r"\[\d+\]", "", shown)).replace(" .", ".").strip()
    return shown, order, plain


def view_item(row: dict, ctx: list, tok, *, answer_text: str | None = None, cite_map: list[str] | None = None) -> dict:
    """한 문항의 화면 데이터. answer_text 는 점검 문항(바꾼 답), cite_map 은 근거 바꿈형(답의 [n] → 기증 컨텍스트 인용 청크).
    §3.5: 주제어(topic term)는 1·2단계 모든 문항에서 보이지 않는다(점검 문항이 주제로 드러나지 않게)."""
    text = row["text"] if answer_text is None else answer_text
    ids = [c.chunk_id for c, _n in ctx]
    shown, order, plain = render_answer(text, ids)
    if cite_map is not None:
        order = list(cite_map)
    disp = {c.chunk_id: c.text.strip().replace("\n", " ")[:n] for c, n in ctx}
    head = {c.chunk_id: c.title + " > " + c.section_path for c, _n in ctx}
    texts = [disp[i] for i in ids]
    hl = dict(zip(ids, highlights(plain, texts)))
    cited = [{"n": j + 1, "h": head[c], "t": disp[c], "hl": hl[c]} for j, c in enumerate(order)]
    other = [{"h": head[c], "t": disp[c], "hl": hl[c]} for c in ids if c not in order]
    ntok = sum(len(tok.encode(disp[c], truncation=False)) for c in order) if tok else 0
    return {"persp": PERSP.get(row["branch"], row["branch"]), "q": row["question"], "ans": shown,
            "cited": cited, "other": other, "tmin": tmin(ntok), "_tokens": ntok, "_plain": plain, "_order": order}


def load_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("lytang/MiniCheck-Flan-T5-Large", local_files_only=True)


def make_builder(cfg_path: Path) -> ContextBuilder:
    import yaml
    from lfm.config import Config
    from lfm.retriever import BM25Retriever
    cfg = Config(yaml.safe_load(io.open(cfg_path, encoding="utf-8")), source=str(cfg_path))
    with io.open(ROOT / "data/topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    anchors = {str(t["id"]): " ".join(t.get("terms") or []) for t in topics}
    return ContextBuilder(BM25Retriever(cfg), anchors, cfg.top_k_for_round, int(cfg.get("branch.max_context_chars", 12000)))


def topic_terms() -> dict[str, str]:
    import yaml
    with io.open(ROOT / "data/topics_pilot50.yaml", encoding="utf-8") as fh:
        return {str(t["id"]): t["term"] for t in yaml.safe_load(fh)["topics"]}


# --------------------------------------------------------------------------- §3 표집
def sqrt_alloc(Nh: dict[str, int], n: int) -> dict[str, int]:
    sq = {k: math.sqrt(v) for k, v in Nh.items()}
    S = sum(sq.values())
    raw = {k: n * sq[k] / S for k in Nh}
    out = {k: max(1, min(Nh[k], int(math.floor(raw[k])))) for k in Nh}
    while sum(out.values()) < n:
        k = sorted([k for k in Nh if out[k] < Nh[k]], key=lambda k: (-(raw[k] - out[k]), k))[0]
        out[k] += 1
    while sum(out.values()) > n:
        k = sorted([k for k in Nh if out[k] > 1], key=lambda k: ((raw[k] - out[k]), k))[0]
        out[k] -= 1
    return out


def draw_quality() -> dict:
    runs = with_r1(load_run(ROOT / "results/r2/heldout35"))
    per = {(a, s): {(r["query_id"], r["branch"], r["sq_index"]): r for r in slot_rows(runs, a, s, final=True)}
           for a in ARMS_Q for s in (1, 2, 3)}
    allslots = set()
    for s in (1, 2, 3):
        for a in ARMS_Q:
            allslots |= {(s,) + k for k in per[(a, s)]}
    assert len(allslots) == 1752, len(allslots)
    UE = []
    for sl in sorted(allslots):
        rows = {a: per[(a, sl[0])].get(sl[1:]) for a in ARMS_Q}
        if any(r and strip_cites(r["text"]) for r in rows.values()):
            UE.append((sl, rows))
    distinct = lambda rows: {strip_cites(r["text"]) for r in rows.values() if r and strip_cites(r["text"])}
    n_items = sum(len(distinct(rows)) for _, rows in UE)
    Nh = Counter(sl[1] for sl, _ in UE)
    ans = {a: sum(1 for _, rows in UE if rows[a] and strip_cites(rows[a]["text"])) for a in ARMS_Q}
    assert (len(UE), n_items, len(Nh)) == (357, 496, 28), (len(UE), n_items, len(Nh))
    assert Counter(sl[0] for sl, _ in UE) == Counter({1: 123, 2: 116, 3: 118})
    assert ans == {"E-SQ": 277, "E-GR": 126, R1: 269}, ans
    # §3.3 두 방향 층: g = 1[E-GR 이 그 슬롯에 답함]. g1 은 토픽마다 한 층, g0 은 토픽마다 한 층이되 N_h ≤ K_SMALL 인 토픽은
    # "g0|small" 한 층으로 묶는다(모든 층 n_h ≥ 1 을 지키면서 본 문항 ≤ 70 을 맞추려고). 라벨이 아니라 답 개수만 쓴다.
    egr = lambda rows: bool(rows["E-GR"] and strip_cites(rows["E-GR"]["text"]))
    g0N = Counter(sl[1] for sl, rows in UE if not egr(rows))

    def cells_for(k: int):
        """묶음 한계 k 의 층 → (층별 슬롯, 배분, 기대 문항)."""
        b = defaultdict(list)
        for sl, rows in UE:
            b[("g1|" + sl[1]) if egr(rows) else ("g0|" + (sl[1] if g0N[sl[1]] > k else "small"))].append((sl, rows))
        nc = {c: len(v) for c, v in b.items()}
        al = {**sqrt_alloc({c: n for c, n in nc.items() if c.startswith("g1|")}, N_G1), **{c: 1 for c in nc if c.startswith("g0|")}}
        ex = sum(al[c] * float(np.mean([len(distinct(rows)) for _, rows in b[c]])) for c in al)
        return b, al, ex

    # §3.3 규칙: K_SMALL = 기대 문항 < MAX_Q_ITEMS 인 가장 작은 정수 한계 (k = 0, 1, … 순서로 다시 계산)
    exp_by_k = {k: round(cells_for(k)[2], 2) for k in range(0, K_SMALL + 1)}
    assert min(k for k, e in exp_by_k.items() if e < MAX_Q_ITEMS) == K_SMALL, exp_by_k
    by, alloc, exp_items = cells_for(K_SMALL)
    Ncell = {c: len(v) for c, v in by.items()}
    g1 = {c: n for c, n in Ncell.items() if c.startswith("g1|")}
    g0 = {c: n for c, n in Ncell.items() if c.startswith("g0|")}
    assert (len(g1), sum(g1.values()), len(g0), sum(g0.values()), Ncell["g0|small"]) == (24, 126, 20, 231, 21), \
        (len(g1), sum(g1.values()), len(g0), sum(g0.values()), Ncell.get("g0|small"))
    assert alloc == {**sqrt_alloc(g1, N_G1), **{c: 1 for c in g0}}   # g0: 층마다 1 (20)
    assert sum(alloc.values()) == 45 and Counter(alloc.values()) == Counter({1: 43, 2: 1}) and alloc["g1|t036"] == 2, alloc
    assert round(exp_items, 2) == 69.31, exp_items
    rng = np.random.default_rng(SEED["draw"])
    pick = []
    for c in sorted(by):                                  # 층 순서 = 층 id 사전순, 층 안 후보 = (seed, query_id, branch, sq_index)
        cand = sorted(by[c], key=lambda x: x[0])
        idx = rng.permutation(len(cand))[: alloc[c]]
        pick += [(*cand[i], Ncell[c], alloc[c], c) for i in sorted(idx.tolist())]
    n_pick_items = sum(len(distinct(rows)) for _, rows, _N, _n, _c in pick)
    ans_pick = {a: sum(1 for _, rows, _N, _n, _c in pick if rows[a] and strip_cites(rows[a]["text"])) for a in ARMS_Q}
    assert n_pick_items <= MAX_Q_ITEMS, n_pick_items
    assert n_pick_items == 69, n_pick_items
    assert ans_pick == {"E-SQ": 29, "E-GR": 25, R1: 30}, ans_pick
    n_qs = len({(sl[0], sl[1]) for sl in allslots})
    return {"runs": runs, "pick": pick, "agg": {"U_E_slots": len(UE), "U_E_items": n_items, "topics": len(Nh),
            "U_E_by_seed": dict(Counter(sl[0] for sl, _ in UE)), "U_E_answered": ans, "all_slots": len(allslots),
            "n_query_seeds_all": n_qs, "design": "two-way: (E-GR answered g1/g0) x topic, g0 topics with N_h<=%d pooled" % K_SMALL,
            "alloc": dict(sorted(alloc.items())), "N_cell": dict(sorted(Ncell.items())), "N_topic": dict(sorted(Nh.items())),
            "expected_items": round(exp_items, 2), "expected_items_by_pool_limit": exp_by_k, "K_SMALL": K_SMALL, "n_slots": sum(alloc.values()), "n_items": n_pick_items,
            "answered_in_draw": ans_pick}}


def pilot_pool(fc: FullContext) -> tuple[dict, list[dict]]:
    """pilot5(dev) E-SQ/E-GR s1 최종 답한 sub-answer, 같은 글자 묶음, 정렬 (query_id, branch, sq_index, arm)."""
    runs = load_run(ROOT / "results/r2/pilot5")
    rows = []
    for a in ("E-SQ", "E-GR"):
        rows += [r for r in slot_rows(runs, a, 1, final=True) if r["status"] == "answered" and strip_cites(r["text"])]
    rows.sort(key=lambda r: (r["query_id"], r["branch"], r["sq_index"], r["arm"]))
    seen, pool = set(), []
    for r in rows:
        t = strip_cites(r["text"])
        if t in seen:
            continue
        seen.add(t)
        pool.append(r)
    return runs, pool


def draw_calib(pool: list[dict]) -> list[dict]:
    """§3.6: default_rng(4405).permutation(len(pool)) 순서로 걸으며 차례로 받는다. 단 세 번째 후보는 앞의 두 개와 합쳐
    dev 토픽(query_id)이 2개 이상이 될 때만 받는다(아니면 건너뛴다). 결과를 pool 순서로 정렬."""
    rng = np.random.default_rng(SEED["calib"])
    pick = []
    for i in rng.permutation(len(pool)).tolist():
        if len(pick) == 2 and len({pool[j]["query_id"] for j in pick} | {pool[i]["query_id"]}) < 2:
            continue
        pick.append(i)
        if len(pick) == 3:
            break
    out = [pool[i] for i in sorted(pick)]
    assert len({r["query_id"] for r in out}) >= 2
    return out


def calib_id(r: dict) -> str:
    return "qc" + sha("calib|%s|%s|%s|%s" % (r["arm"], r["query_id"], r["branch"], r["sq_index"]), 5)


# --------------------------------------------------------------------------- §3 점검
YEAR_RE = re.compile(r"(?<![\d.,])(1[0-9]{3}|20[0-9]{2})(?![\d.,])")
NUM_RE = re.compile(r"(?<![\d.,])(\d+(?:\.\d+)?)(?![\d.,])")


HARD_RUNGS = (("long", "back", "outside"), ("back", "outside"), ("outside",))   # §3.6 사다리: 0 → 1 → 2


def make_catches(runs, pool, calib, fc: FullContext, tok, n_main: int) -> tuple[list[dict], dict]:
    """§3.6: 모순형 2(어려운) + 근거 바꿈형 2(쉬운), seed 4404. 내용은 출력하지 않는다(후보 수만).
    후보 = pool 에서 보정 3문항의 dev 토픽(query_id)을 모두 뺀 답(기증 컨텍스트도 같은 후보에서만)."""
    rng = np.random.default_rng(SEED["catch"])
    calib_q = {r["query_id"] for r in calib}
    cands = [r for r in pool if r["query_id"] not in calib_q]
    counts = Counter({"candidates_after_calib_topic_exclusion": len(cands)})
    allopts = []                     # (row_index, [option with flags])
    for ri, r in enumerate(cands):
        ctx, _sha = fc.get(runs[(r["arm"], 1)][r["query_id"]], r["branch"], r["produced_round"])
        ids = [c.chunk_id for c, _ in ctx]
        disp = {c.chunk_id: c.text.strip().replace("\n", " ")[:n] for c, n in ctx}
        whole = " ".join(disp.values())
        shown, order, plain = render_answer(r["text"], ids)
        if not order:
            continue
        opts = []
        cite_spans = [m.span() for m in CITATION_RE.finditer(r["text"])]
        in_cite = lambda m: any(a <= m.start(1) < b for a, b in cite_spans)
        found = [m for m in YEAR_RE.finditer(r["text"])] + [m for m in NUM_RE.finditer(r["text"]) if not YEAR_RE.fullmatch(m.group(1))]
        for m in [m for m in found if not in_cite(m)]:
            val = m.group(1)
            counts["numbers_in_answers"] += 1
            for cid in order:                                   # 인용 청크 안의 같은 수
                t = disp[cid]
                for a, b in sent_spans(t):
                    if not re.search(r"(?<![\d.,])" + re.escape(val) + r"(?![\d.,])", t[a:b]):
                        continue
                    counts["in_cited_chunk_sentence"] += 1
                    isyear = bool(YEAR_RE.fullmatch(val))
                    for new in ([str(int(val) + 7), str(int(val) - 7)] if isyear else
                                [(("%g" % (float(val) * 2 + 3)) if "." in val else str(int(val) * 2 + 3))]):
                        if re.search(r"(?<![\d.,])" + re.escape(new) + r"(?![\d.,])", whole):
                            continue
                        counts["replacement_absent_from_context"] += 1
                        newtext = r["text"][: m.start(1)] + new + r["text"][m.end(1):]
                        _s, _o, nplain = render_answer(newtext, ids)
                        hl = dict(zip(ids, highlights(nplain, [disp[i] for i in ids])))
                        fl = {"long": len(t) >= 1100, "back": a >= len(t) / 2, "outside": not any(x == [a, b] for x in hl[cid])}
                        opts.append({"cid": cid, "val": val, "new": new, "newtext": newtext,
                                     "kind": "year" if isyear else "number", "flags": fl})
        if opts:
            allopts.append((ri, opts))
    hard, rung = [], None
    for k, req in enumerate(HARD_RUNGS):                   # 사다리: 가장 엄격한 단부터, 후보 답 ≥ 2 인 첫 단
        hard = [(ri, [o for o in opts if all(o["flags"][f] for f in req)]) for ri, opts in allopts]
        hard = [(ri, o) for ri, o in hard if o]
        counts["hard_candidate_answers_rung%d" % k] = len(hard)
        if len(hard) >= 2:
            rung = k
            break
    if rung is None:
        raise SystemExit("점검 모순형 후보 < 2 (사다리 끝까지) — 멈춘다. 후보 수: %s" % dict(counts))
    counts["hard_rung_applied"] = rung
    counts["hard_candidate_answers"] = len(hard)
    hpick = sorted(rng.choice(len(hard), 2, replace=False).tolist())
    catches = []
    used = set()
    for j in hpick:
        ri, opts = hard[j]
        o = opts[int(rng.integers(len(opts)))]
        r = cands[ri]
        used.add(ri)
        ctx, csha = fc.get(runs[(r["arm"], 1)][r["query_id"]], r["branch"], r["produced_round"])
        v = view_item(r, ctx, tok, answer_text=o["newtext"])
        catches.append({"view": v, "type": "contradiction", "difficulty": "hard", "expected": "C0", "rung": rung,
                        "src": r, "ctx_sha": csha, "detail": {**{k: o[k] for k in ("val", "new", "kind")}, "rung": rung}})
    # 근거 바꿈형: 남은 답 2개, 기증 답은 다른 토픽. 원 인용 청크가 기증 컨텍스트에 있으면 rng 순서의 다음 기증 답(§3.6)
    rest = [i for i in range(len(cands)) if i not in used]
    epick = sorted(rng.choice(len(rest), 2, replace=False).tolist())
    for j in epick:
        r = cands[rest[j]]
        own = set(re.findall(r"\[(d\d+s\d+p\d+)\]", r["text"]))
        donors = [d for d in cands if d["query_id"] != r["query_id"]]
        assert donors, "근거 바꿈형: 다른 토픽 기증 답 없음"
        d = None
        for q in rng.permutation(len(donors)).tolist():
            dd = donors[q]
            dctx, dsha = fc.get(runs[(dd["arm"], 1)][dd["query_id"]], dd["branch"], dd["produced_round"])
            dids = [c.chunk_id for c, _ in dctx]
            if not (own & set(dids)):
                d = dd
                break
            counts["evidence_swap_donor_skipped"] += 1
        assert d is not None, "근거 바꿈형: 쓸 수 있는 기증 답 없음"
        _s, dorder, _p = render_answer(d["text"], dids)
        # 답의 [n] 을 기증 컨텍스트의 인용 청크(없으면 앞 청크)에 차례로 대응
        _s2, rorder, _p2 = render_answer(r["text"], sorted(own))
        k = max(1, len(rorder))
        tgt = (dorder or dids)
        cmap = [tgt[q % len(tgt)] for q in range(k)]
        cmap = list(dict.fromkeys(cmap))
        txt = r["text"]
        for q, cid in enumerate(rorder):
            txt = txt.replace("[" + cid + "]", "[" + cmap[min(q, len(cmap) - 1)] + "]")
        v = view_item(r, dctx, tok, answer_text=txt, cite_map=cmap)
        catches.append({"view": v, "type": "evidence_swap", "difficulty": "easy", "expected": "C<=1",
                        "src": r, "ctx_sha": dsha, "detail": {"donor_query": d["query_id"], "donor_arm": d["arm"],
                                                              "donor_branch": d["branch"], "donor_sq": d["sq_index"]}})
    order = rng.permutation(4).tolist()                    # 위치 배정: 2단계(본 n_main + 4) 앞 절반 2, 뒤 절반 2
    n2 = n_main + 4
    half = n2 // 2
    pos = sorted(rng.choice(half, 2, replace=False).tolist()) + sorted((half + rng.choice(n2 - half, 2, replace=False)).tolist())
    catches = [catches[k] for k in order]
    for c, p in zip(catches, pos):
        c["pos"] = int(p)
        c["id"] = "q" + sha("catch|%s|%s|%s|%s|%s" % (c["type"], c["src"]["arm"], c["src"]["query_id"],
                                                      c["src"]["branch"], c["src"]["sq_index"]))
    return catches, dict(counts)


# --------------------------------------------------------------------------- 순서
def no_adjacent(items: list[dict]) -> list[dict]:
    """같은 질문(query_id, branch, sq_index — seed 무시) 문항이 연달아 오면 뒤 문항을 그다음 '다른 질문' 문항과 맞바꾼다(결정론, §3.5)."""
    xs = list(items)
    for _ in range(len(xs) * len(xs)):
        bad = next((j for j in range(len(xs) - 1) if xs[j]["_slot"] == xs[j + 1]["_slot"]), None)
        if bad is None:
            return xs
        k = next((k for k in range(bad + 2, len(xs)) if xs[k]["_slot"] != xs[bad]["_slot"]), None)
        if k is None:
            k = next(k for k in range(bad) if xs[k]["_slot"] != xs[bad]["_slot"]
                     and (k == 0 or xs[k - 1]["_slot"] != xs[bad + 1]["_slot"]))
        xs[bad + 1], xs[k] = xs[k], xs[bad + 1]
    raise RuntimeError("인접 회피 실패")


def public(x: dict) -> dict:
    return {k: v for k, v in x.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- 누출 검사
FORBID_ANY = [r"E-SQ", r"E-GR", r"E-R1", r"E-BR", r"L-GR", r"L-BR", r"GlobalRetry", r"FreezeRetry", r"global_retry",
              r"freeze_retry", r"heldout", r"pilot5", r"r2-blind", r"r3-quality", r"arm_hash", r"source_item",
              r"p_mc", r"lab_g1", r"context_sha", r"query_id", r"sq_index", r"produced_round", r"\bt0[0-5]\d\b"]
FORBID_TEMPLATE = [r"\barm\b", r"\barms\b", r"\bseed\b", r"\bcell\b", r"\bstratum\b", r"\bN_h\b", r"\bn_h\b",
                   r"\bP1\b", r"\bP2\b", r"\bN1\b", r"\bN2\b", r"sealed", r"\bscore\b"]
Q_KEYS = {"id", "persp", "q", "ans", "cited", "other", "tmin", "ref"}
V_KEYS = {"id", "rules", "claim", "ev", "tmin", "ref"}


def leak_check(html: str, data: dict) -> dict:
    """HTML 에 arm·seed·점수·키 정보가 없는지. V′ 문장(3·4단계)은 원 시트 바이트 그대로라 인용 청크 id 표기가 남는다(§2.2)
    — 청크 id 검사는 §3 문항에만 하고, V′ 문항은 바이트 동일성으로 대신한다."""
    for pat in FORBID_ANY:
        m = re.search(pat, html)
        assert not m, "누출: %s → %r" % (pat, html[max(0, m.start() - 40): m.end() + 40])
    skel = html
    for s in data["stages"]:
        for x in s["items"]:
            assert set(x) <= (Q_KEYS if s["kind"] == "q" else V_KEYS), set(x)
            assert ("ref" in x) == (s["mode"] in ("calib", "practice")), (s["mode"], x["id"])
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")   # render_page 와 같은 이스케이프
    skel = html.replace(blob, "")
    assert skel != html, "데이터 블록을 찾지 못함"
    for pat in FORBID_TEMPLATE + [r"\bR3\b", r"labels_r3_", r"labeler_r3_"]:     # 라운드 표시(§2.4)도 화면에 없다
        m = re.search(pat, skel)
        assert not m, "템플릿 누출: %s → %r" % (pat, skel[max(0, m.start() - 40): m.end() + 40])
    n_cid = 0
    for s in data["stages"]:
        if s["kind"] != "q":
            continue
        txt = json.dumps(s["items"], ensure_ascii=False)
        assert not re.search(r"d\d+s\d+p\d+", txt), "§3 문항에 청크 id"
    for s in data["stages"]:
        if s["kind"] == "v":
            n_cid += sum(len(re.findall(r"\[d\d+s\d+p\d+\]", x["claim"])) for x in s["items"])
    return {"ok": True, "v_claim_chunk_id_markers_inherited": n_cid}


# --------------------------------------------------------------------------- 쓰기
def stage_defs(q_calib, q_main, v_prac, v_test):
    return [{"id": 1, "kind": "q", "mode": "calib", "name": "1단계", "items": q_calib},
            {"id": 2, "kind": "q", "mode": "main", "name": "2단계", "items": q_main},
            {"id": 3, "kind": "v", "mode": "practice", "name": "3단계", "items": v_prac},
            {"id": 4, "kind": "v", "mode": "test", "name": "4단계", "items": v_test}]


def page_for(annot: str, stages: list[dict], version: str) -> tuple[str, dict]:
    # 시험판은 다른 저장 키(같은 브라우저의 file:// 는 저장소를 함께 쓰므로 시험 진행이 실제 시트에 섞이지 않게)
    data = {"annotator": annot, "version": version,
            "store_key": "r3_%s_v1" % annot + ("_placeholder" if "PLACEHOLDER" in version else ""),
            "csv_name": "labels_%s.csv" % annot, "v_labels": V_LABELS, "h_claim": H_CLAIM, "h_ev": H_EV,
            "rubric_html": RUBRIC_HTML, "stages": stages}
    counts = {k + 1: len(s["items"]) for k, s in enumerate(stages)}
    counts["all"] = sum(len(s["items"]) for s in stages)
    title = "라벨 작업" + (" — TEST BUILD, 배포 금지" if "PLACEHOLDER" in version else "")
    html = render_page(title=title, data_json=json.dumps(data, ensure_ascii=False), cols_json=json.dumps(COLS),
                       counts=counts, csv_name=data["csv_name"])
    return html, data


def cmd_calib_draw(args) -> int:
    guard_no_llm()
    fc = FullContext(make_builder(ROOT / "results/r2/pilot5/E-GR/s1/config.yaml"))
    tok = load_tok()
    runs, pool = pilot_pool(fc)
    calib = draw_calib(pool)
    items = []
    for r in calib:
        ctx, _ = fc.get(runs[(r["arm"], 1)][r["query_id"]], r["branch"], r["produced_round"])
        items.append({"id": calib_id(r), **public(view_item(r, ctx, tok))})
    stages = [{"id": 1, "kind": "q", "mode": "main", "name": "참조 라벨", "items": items}]
    data = {"annotator": "author_ref", "version": VERSION + "+calib_ref", "store_key": "r3_calibref_v1",
            "csv_name": "labels_r3_calib_ref.csv", "v_labels": V_LABELS, "h_claim": H_CLAIM, "h_ev": H_EV,
            "rubric_html": RUBRIC_HTML, "stages": stages}
    html = render_page(title="R3 보정 참조 라벨 (저자)", data_json=json.dumps(data, ensure_ascii=False),
                       cols_json=json.dumps(COLS), counts={"all": 3, 1: 3, 2: 0, 3: 0, 4: 0}, csv_name=data["csv_name"])
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    io.open(CALIB_DIR / "labeler_r3_calib_ref.html", "w", encoding="utf-8", newline="\n").write(html)
    print("calib-draw: pool %d, 3문항 (dev 토픽 %s) -> %s (참조 라벨 CSV 를 %s 로 두고 커밋)" % (
        len(pool), sorted(Counter(r["query_id"] for r in calib).items()), CALIB_DIR / "labeler_r3_calib_ref.html", CALIB_REF))
    return 0


def cmd_build(args) -> int:
    L = guard_no_llm()
    placeholder = bool(args.placeholder_ref)
    out_root = (R3 / "_test_build") if placeholder else R3
    sheets, sealed = out_root / "sheets", out_root / "sealed"
    version = VERSION + ("+PLACEHOLDER_REF" if placeholder else "")
    vv = build_v()
    print("§2 V′: 연습 %d (alloc %s, 길이 %s, 저자 지지 %d/10), 시험 %d (길이 %s)" % (
        len(vv["practice"]), vv["agg"]["practice_alloc"], vv["agg"]["practice_bins"],
        vv["agg"]["practice_author_support"], len(vv["test"]), vv["agg"]["c50_bins"]))
    dq = draw_quality()
    print("§3 U_E %d 슬롯 / %d 문항 / %d 토픽; 두 방향 층 %d개, 표집 %d슬롯 → %d 문항(기대 %.1f), 답한 슬롯 %s" % (
        dq["agg"]["U_E_slots"], dq["agg"]["U_E_items"], dq["agg"]["topics"], len(dq["agg"]["alloc"]), dq["agg"]["n_slots"],
        dq["agg"]["n_items"], dq["agg"]["expected_items"],
        {("arm%d" % k): v for k, v in enumerate(dq["agg"]["answered_in_draw"].values())}))
    fc = FullContext(make_builder(ROOT / "results/r2/heldout35/E-GR/s1/config.yaml"))
    tok = load_tok()
    runs = dq["runs"]
    # ---- 본 문항 (69)
    h4 = {x["ev"] for s in items_of(ROOT / "results/r2/heldout35/sheets/labeler_h4.html", "SETS") for x in s["items"]}
    main, keyq, overlap, n_ctx_diff = [], [], [], 0
    prio = {R1: 0, "E-SQ": 1, "E-GR": 2}
    for sl, rows, N_h, n_h, cell in dq["pick"]:
        s, qid, br, sq = sl
        egr = int(bool(rows["E-GR"] and strip_cites(rows["E-GR"]["text"])))
        groups = defaultdict(list)
        for a in ARMS_Q:
            r = rows[a]
            if r and strip_cites(r["text"]):
                groups[strip_cites(r["text"])].append(r)
            else:
                keyq.append({"item_id": "", "stage": "main", "arm": a, "arm_hash": arm_hash(a), "seed": s, "query_id": qid,
                             "branch": br, "sq_index": sq, "produced_round": (r or {}).get("produced_round", ""),
                             "answered": 0, "status": (r or {}).get("status", "missing"), "stratum": cell, "topic": qid,
                             "egr_answered": egr, "N_h": N_h, "n_h": n_h, "catch": 0})
        for txt, rs in sorted(groups.items()):
            # 컨텍스트 출처: 가장 이른 생성 라운드, 동률이면 E-R1 > E-SQ > E-GR (고정)
            src = sorted(rs, key=lambda r: (r["produced_round"], prio[r["arm"]]))[0]
            ctx, csha = fc.get(runs[(src["arm"], s)][qid], br, src["produced_round"])
            shas = set()
            for r in rs:
                shas.add(fc.get(runs[(r["arm"], s)][qid], br, r["produced_round"])[1])
            n_ctx_diff += int(len(shas) > 1)
            iid = "q" + sha("main|%s|%s|%s|%s|%s" % (s, qid, br, sq, txt))
            v = view_item(src, ctx, tok)
            ov = txt in h4
            if ov:
                overlap.append(iid)
            main.append({"id": iid, **v, "_slot": (qid, br, sq)})    # 인접 회피 열쇠: 질문(seed 무시), §3.5
            shared = ";".join(arm_hash(r["arm"]) for r in rs)
            for r in rs:
                keyq.append({"item_id": iid, "stage": "main", "arm": r["arm"], "arm_hash": arm_hash(r["arm"]), "seed": s,
                             "query_id": qid, "branch": br, "sq_index": sq, "produced_round": r["produced_round"],
                             "answered": 1, "status": r["status"], "asked_ne_question": int(r["asked"] != r["question"]),
                             "stratum": cell, "topic": qid, "egr_answered": egr, "N_h": N_h, "n_h": n_h, "shared_with": shared, "h4_overlap": int(ov),
                             "catch": 0, "context_source_arm": src["arm"], "context_sha256": csha,
                             "contexts_differ_in_group": int(len(shas) > 1), "cited_tokens": v["_tokens"], "t_min_s": v["tmin"]})
    n_main = dq["agg"]["n_items"]
    assert len(main) == n_main == 69 and len({x["id"] for x in main}) == n_main
    # ---- 보정 3 + 점검 4 (pilot5, dev 토픽)
    fcp = FullContext(make_builder(ROOT / "results/r2/pilot5/E-GR/s1/config.yaml"))
    prun, pool = pilot_pool(fcp)
    calib = draw_calib(pool)
    calib_items = []
    for r in calib:
        ctx, csha = fcp.get(prun[(r["arm"], 1)][r["query_id"]], r["branch"], r["produced_round"])
        calib_items.append({"id": calib_id(r), **view_item(r, ctx, tok), "_src": r, "_sha": csha})
    if placeholder:
        refs = {x["id"]: {"C": "C2a", "K": "K2"} for x in calib_items}
    else:
        if not CALIB_REF.exists():
            raise SystemExit("보정 참조 라벨이 없다: %s — `calib-draw` 시트로 저자가 달고 커밋한 뒤 다시 실행한다 (§3.6, C1 전)" % CALIB_REF)
        ref_rows = rd(CALIB_REF)
        vers = {r.get("sheet_version", "") for r in ref_rows}
        assert vers == {VERSION + "+calib_ref"}, "보정 참조 CSV 판 %s — ?test=1 시험 판 CSV 는 받지 않는다(§3.6)" % vers
        got = {r["item_id"]: {"C": r["C"].strip(), "K": r["K"].strip()} for r in ref_rows if r.get("C", "").strip()}
        assert set(got) == {x["id"] for x in calib_items}, "참조 라벨 id 가 seed 4405 추출과 다르다"
        assert all(v["C"] in ("C2a", "C2b", "C1", "C0") and v["K"] in ("K2", "K1", "K0") for v in got.values())
        refs = got
    for x in calib_items:
        x["ref"] = refs[x["id"]]
        r = x["_src"]
        keyq.append({"item_id": x["id"], "stage": "calib", "arm": r["arm"], "arm_hash": arm_hash(r["arm"]), "seed": 1,
                     "query_id": r["query_id"], "branch": r["branch"], "sq_index": r["sq_index"],
                     "produced_round": r["produced_round"], "answered": 1, "status": r["status"], "catch": 0,
                     "ref_C": x["ref"]["C"], "ref_K": x["ref"]["K"], "context_sha256": x["_sha"],
                     "cited_tokens": x["_tokens"], "t_min_s": x["tmin"]})
    catches, ccount = make_catches(prun, pool, calib, fcp, tok, n_main)
    calib_topics = sorted(Counter(r["query_id"] for r in calib).items())
    catch_topics = sorted(Counter(c["src"]["query_id"] for c in catches).items())
    assert not ({q for q, _ in calib_topics} & {q for q, _ in catch_topics}), "점검 토픽 ∩ 보정 토픽"
    assert len(calib_topics) >= 2
    for c in catches:
        r = c["src"]
        keyq.append({"item_id": c["id"], "stage": "main", "arm": "catch:" + r["arm"], "seed": 1, "query_id": r["query_id"],
                     "branch": r["branch"], "sq_index": r["sq_index"], "produced_round": r["produced_round"], "answered": 1,
                     "catch": 1, "catch_type": c["type"], "catch_difficulty": c["difficulty"], "expected_C": c["expected"],
                     "catch_pos": c["pos"], "catch_detail": json.dumps(c["detail"], ensure_ascii=False),
                     "context_sha256": c["ctx_sha"], "cited_tokens": c["view"]["_tokens"], "t_min_s": c["view"]["tmin"]})
    # ---- 판정자별 순서
    pages, datas = {}, {}
    for annot in ("A", "B"):
        rq = np.random.default_rng(SEED["q_" + annot])
        c_ord = [calib_items[i] for i in rq.permutation(3).tolist()]
        m_ord = no_adjacent([main[i] for i in rq.permutation(len(main)).tolist()])
        for c in sorted(catches, key=lambda c: c["pos"]):
            m_ord.insert(c["pos"], {"id": c["id"], **c["view"]})
        rv = np.random.default_rng(SEED["v_" + annot])
        p_ord = [vv["practice"][i] for i in rv.permutation(10).tolist()]
        t_ord = [vv["test"][i] for i in rv.permutation(50).tolist()]
        stages = stage_defs([public(x) for x in c_ord], [public(x) for x in m_ord], p_ord, t_ord)
        html, data = page_for(annot, stages, version)
        chk = leak_check(html, data)
        pages[annot], datas[annot] = html, data
    # A·B 는 순서만 다르다
    for k in range(4):
        a, b = datas["A"]["stages"][k]["items"], datas["B"]["stages"][k]["items"]
        assert sorted(json.dumps(x, sort_keys=True, ensure_ascii=False) for x in a) == \
               sorted(json.dumps(x, sort_keys=True, ensure_ascii=False) for x in b)
        assert [x["id"] for x in a] != [x["id"] for x in b] or len(a) < 2
    strip = lambda d: {k: v for k, v in d.items() if k not in ("annotator", "store_key", "csv_name", "stages")}
    assert strip(datas["A"]) == strip(datas["B"])
    n_counts = [len(s["items"]) for s in datas["A"]["stages"]]
    assert n_counts == [3, n_main + 4, 10, 50] == [3, 73, 10, 50], n_counts
    # ---- 쓰기
    sheets.mkdir(parents=True, exist_ok=True)
    sealed.mkdir(parents=True, exist_ok=True)
    hsha = {}
    for annot, html in pages.items():
        p = sheets / ("labeler_%s.html" % annot)
        io.open(p, "w", encoding="utf-8", newline="\n").write(html)
        hsha[annot] = hashlib.sha256(html.encode("utf-8")).hexdigest()
    write_csv(sealed / "_key_r3_v.csv", vv["key"], ["sheet_item_id", "stage", "source_item", "cell", "doc_tokens", "len_bin", "t_min_s"])
    qcols = ["item_id", "stage", "arm", "arm_hash", "seed", "query_id", "branch", "sq_index", "produced_round", "answered",
             "status", "asked_ne_question", "stratum", "topic", "egr_answered", "N_h", "n_h", "shared_with", "h4_overlap", "catch", "catch_type",
             "catch_difficulty", "expected_C", "catch_pos", "catch_detail", "ref_C", "ref_K", "context_source_arm",
             "context_sha256", "contexts_differ_in_group", "cited_tokens", "t_min_s"]
    write_csv(sealed / "_key_q.csv", keyq, qcols)
    stage_tmin = {k + 1: round(sum(x["tmin"] for x in s["items"]) / 3600, 3) for k, s in enumerate(datas["A"]["stages"])}
    stage_tmin_mean = {k + 1: round(sum(x["tmin"] for x in s["items"]) / len(s["items"]), 1) for k, s in enumerate(datas["A"]["stages"])}
    meta = {"spec": "PREREG_R3 §2.2, §2.4, §3.3, §3.5, §3.6", "version": version, "salt": SALT, "seeds": SEED, "cap_s": CAP,
            "n_items_by_stage": dict(zip((1, 2, 3, 4), n_counts)), "v": vv["agg"], "q": {k: v for k, v in dq["agg"].items()},
            "q_items_main": len(main), "h4_overlap": len(overlap), "groups_with_differing_contexts": n_ctx_diff,
            "catch_candidate_counts": ccount, "catch_positions": sorted(c["pos"] for c in catches),
            "calib_dev_topics": calib_topics, "catch_dev_topics": catch_topics, "hard_rung_applied": ccount["hard_rung_applied"],
            "n_query_seeds_all": dq["agg"]["n_query_seeds_all"],
            "calib_pool": len(pool), "tmin_sum_h_by_stage": stage_tmin, "tmin_mean_s_by_stage": stage_tmin_mean,
            "html_sha256": hsha, "placeholder_ref": placeholder, "time_estimate": time_estimate(n_counts),
            "llm_instances": 0, "api_calls": 0, "leak_check": chk}
    io.open(sealed / "_meta_q.json", "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=1))
    io.open(sealed / "_note_overlap.json", "w", encoding="utf-8").write(json.dumps(
        {"h4_overlap_items": sorted(overlap), "rule": "§3.3: H4 36문항과 글자가 같은 문항, 포함·제외로 보고"}, ensure_ascii=False, indent=1))
    io.open(sheets / "안내.md", "w", encoding="utf-8", newline="\n").write(guide_md(n_counts, stage_tmin, stage_tmin_mean))
    print("단계 문항 수 %s, 강제 체류 하한 합(시간) %s, 평균(초) %s" % (n_counts, stage_tmin, stage_tmin_mean))
    print("H4 겹침 %d, 그룹 안 컨텍스트 다름 %d, 점검 후보 수 %s" % (len(overlap), n_ctx_diff, ccount))
    print("보정 dev 토픽 %s, 점검 dev 토픽 %s (겹침 없음), 어려운 점검 사다리 단 %d" % (calib_topics, catch_topics, ccount["hard_rung_applied"]))
    print("-> %s (A %s…, B %s…), 봉인 %s; LLM 0" % (sheets, hsha["A"][:12], hsha["B"][:12], sealed))
    return 0


PACE_MIN = {"q": 2.6, "v": 2.0}   # 문항당 예상 분(추정, 실측 아님): 품질 2.6, 문장 2.0
GUIDE_READ_H = 0.25


def time_estimate(n) -> dict:
    """안내.md·§5.3 의 예상 시간(계산값). n = 단계별 문항 수 [1, 2, 3, 4]."""
    est = {1: n[0] * PACE_MIN["q"] / 60, 2: n[1] * PACE_MIN["q"] / 60, 3: n[2] * PACE_MIN["v"] / 60, 4: n[3] * PACE_MIN["v"] / 60}
    return {"by_stage_h": {k: round(v, 3) for k, v in est.items()}, "guide_h": GUIDE_READ_H,
            "total_h": round(sum(est.values()) + GUIDE_READ_H, 2)}


def guide_md(n, tmin_h, tmin_s) -> str:
    te = time_estimate(n)
    est = te["by_stage_h"]
    rows = "\n".join("| %d단계 | %d | %.2f시간 (평균 %.0f초) | 약 %.2f시간 |" % (k, n[k - 1], tmin_h[k], tmin_s[k], est[k]) for k in (1, 2, 3, 4))
    return """# 라벨 작업 안내

받는 파일: **labeler_A.html** 또는 **labeler_B.html** 중 본인 것 하나. 다른 사람의 파일은 열지 마세요.

1. 파일을 크롬·엣지 같은 브라우저로 엽니다(더블클릭). 인터넷 연결은 필요 없습니다.
2. 첫 화면 안내를 끝까지 읽고, 번역 도구 이름과 영어 실력 한 줄을 적은 뒤 "시작하기"를 누릅니다.
3. 1단계 → 2단계 → 3단계 → 4단계 순서로 진행합니다. 앞 단계를 다 채워야 다음 단계가 열립니다.
4. 2단계를 다 채우면 쉬는 화면이 나옵니다. 준비되면 "3단계 시작"을 누릅니다. **누른 뒤에는 1·2단계 답을 바꿀 수 없습니다.**
5. 근거 상자를 끝까지 스크롤하고, 최소 읽기 시간이 지나고, 번역기 사용 여부를 골라야 버튼이 켜집니다.
6. 진행은 같은 컴퓨터·같은 브라우저에만 저장됩니다. 시크릿 창은 쓰지 마세요. 브라우저 기록·사이트 데이터를 지우면 진행이 사라집니다.
7. 쉬는 것을 권합니다: 2단계 중간, 2단계와 3단계 사이(각 30분 이상).
8. 다른 사람과 상의하지 마세요. 인터넷 검색·배경지식 없이 화면의 글만 보고 판단해 주세요.
9. 다 끝나면 "CSV 내려받기 (전체)"를 눌러 받은 파일(labels_A.csv 또는 labels_B.csv) **하나만** 교신저자에게 보냅니다.

## 예상 시간

| 단계 | 문항 | 최소 읽기 시간 합 | 예상 소요 |
|---|---|---|---|
""" + rows + """

예상 소요는 품질 문항(1·2단계) 약 %.1f분, 문장 문항(3·4단계) 약 %.1f분으로 잡은 추정입니다(실측 아님).
네 단계 합 %.2f시간에 안내 읽기 약 %d분을 더해 약 %.1f시간입니다.
최소 읽기 시간은 하한일 뿐이고, 필요한 만큼 읽어 주세요.
""" % (PACE_MIN["q"], PACE_MIN["v"], sum(est.values()), round(GUIDE_READ_H * 60), te["total_h"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PREREG_R3 시트 (LLM 0)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("calib-draw")
    b = sp.add_parser("build")
    b.add_argument("--placeholder-ref", action="store_true", help="시험용: 보정 참조 라벨 자리표시자, results/r3/_test_build/ 에 쓴다")
    args = ap.parse_args(argv)
    return {"calib-draw": cmd_calib_draw, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
