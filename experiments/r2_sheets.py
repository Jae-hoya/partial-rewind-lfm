"""R2 사람 라벨 시트 — P-H, H3, H4, H5 (PREREG_R2 §9). **LLM 호출 0.**

  python experiments/r2_sheets.py ph --run-dir results/r2/pilot5
  python experiments/r2_sheets.py h3 --run-dir results/r2/heldout35
  python experiments/r2_sheets.py h4 --run-dir results/r2/heldout35 --slots 200
  python experiments/r2_sheets.py h5 --run-dir results/r2/heldout35
  python experiments/r2_sheets.py g5 --key results/r2/pilot5/sheets_sealed/_key_ph.csv --labels labels_ph_filled.csv

산출(세트 family 마다 하나): <out>/labeler_<set>.html (탭으로 나눈 통합 시트, make_combined_sheet 의 템플릿),
<out>/_key_<set>.csv (봉인 키: arm, seed, 층, N_h, n_h, status, produced_round …), <out>/_meta_<set>.json.

블라인드 규칙 (PREREG §9)
  - 화면에는 arm 이름, status, 점수, 층이 없다. 탭 이름도 중립이다("세트 1 (A)").
  - 문항 id 는 섞은 뒤 순번이다. 키에는 arm 원명과 함께 salt 해시(arm_hash)를 둔다.
  - 빈 답은 "(no answer)" 로 통일한다. 인용 표시는 지운다.
  - RNG seed 는 세트마다 고정이다(SET_SEEDS). 값은 키 메타에 남긴다.

B 라벨의 컨텍스트는 산출에 없어 다시 만든다. 검색은 결정론이다: 질의 = 토픽 terms 앵커 + 그 라운드의
sub-question 문구, top_k = cfg.top_k_for_round(round), lfm.branch.format_context. 다시 만든 branch 컨텍스트의
SHA-256 이 산출의 context_sha256 과 다르면 멈춘다.
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_combined_sheet import HTML as COMBINED_HTML  # noqa: E402
from r2_analyze import R1, final_round, load_run, parse_arm, with_r1  # noqa: E402

from lfm.branch import format_context  # noqa: E402
from lfm.types import CITATION_RE, Chunk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ("overview_factual", "technical_mechanism", "impact_application", "controversy_limitation")
MIN_BODY = 200                       # d15_abstention_diag.MIN_BODY — 사전등록 노출 기준(본문 200자 이상)
NO_ANSWER = "(no answer)"
SET_SEEDS = {"ph": 4201, "h3": 4203, "h4": 4204, "h5": 4205}   # [PREREG 에 적을 값] 세트별 고정 RNG
ARM_SALT = "r2-blind-v1"


def sealed_dir(out_dir) -> Path:
    """봉인 키는 시트와 다른 폴더(<out>_sealed)에 둔다. 라벨러는 시트 폴더만 받는다."""
    d = Path(out_dir)
    d = d.with_name(d.name + "_sealed")
    d.mkdir(parents=True, exist_ok=True)
    return d
G5_DEFAULT, G5_FLOOR = 200, 120

# make_h2_sheets.RULES_A / RULES_B 와 같은 문구 (그 모듈은 import 하면 d15 를 끌어온다)
RULES_A = (
    "모델의 답이 질문이 묻는 것에 답하는지만 본다. 답이 사실인지는 따지지 않는다(그건 다른 검사다) / "
    "답함 = 질문이 묻는 대상(수치·시기·원인·비판 등)에 대한 진술이 있다. 여러 개를 묻는 질문이면 하나라도 직접 답하면 답함 / "
    "일부 = 관련은 있으나 묻는 핵심에서 비껴간다(주변 사실, 일반론) / "
    "답안함 = 같은 주제의 다른 이야기만 한다. 질문이 묻는 것에 대한 진술이 없다 / "
    "(no answer) 로 보이면 답안함"
)
RULES_B = (
    "아래 컨텍스트만 보고 판단한다. 바깥 지식은 쓰지 않는다 / "
    "답있음 = 질문이 묻는 것에 대한 사실 진술이 컨텍스트에 있다. 일부만 묻는 질문이면 그 일부로 충분 / "
    "일부 = 질문과 관련된 사실은 있으나 묻는 것의 핵심에는 못 미친다 / "
    "답없음 = 질문에 답하는 사실이 없다. 주제만 같고 다른 내용이면 답없음"
)
LABELS_A = ["답함", "답안함", "일부"]
LABELS_B = ["답있음", "답없음", "일부"]
H_A = ("질문 (sub-question)", "모델의 답 (인용 표시는 지웠다)")
H_B = ("질문 (sub-question)", "컨텍스트 — 모델이 받은 이 질문의 검색 청크 (실린 그대로)")


# --------------------------------------------------------------------------- 공용
def strip_cites(s: str) -> str:
    return re.sub(r"\s+", " ", CITATION_RE.sub("", s or "")).replace(" .", ".").strip()


def shown_answer(text: str) -> str:
    t = strip_cites(text)
    return t if t else NO_ANSWER


def arm_hash(arm: str) -> str:
    return hashlib.sha256((ARM_SALT + "|" + arm).encode("utf-8")).hexdigest()[:10]


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """비례 배분(최대 나머지). 층 크기를 넘지 않는다."""
    tot = sum(sizes.values())
    if tot == 0 or n <= 0:
        return {k: 0 for k in sizes}
    n = min(n, tot)
    raw = {k: n * v / tot for k, v in sizes.items()}
    out = {k: min(sizes[k], int(math.floor(x))) for k, x in raw.items()}
    rest = n - sum(out.values())
    for k in sorted(sizes, key=lambda k: (-(raw[k] - math.floor(raw[k])), k)):
        if rest <= 0:
            break
        if out[k] < sizes[k]:
            out[k] += 1
            rest -= 1
    while rest > 0:                                  # 작은 층이 꽉 차서 남은 몫
        k = next((k for k in sorted(sizes) if out[k] < sizes[k]), None)
        if k is None:
            break
        out[k] += 1
        rest -= 1
    return out


def stratified_sample(pool: list[dict], strat: Callable[[dict], str], n: int, rng) -> list[dict]:
    """층 비례 추출. 반환 행에 stratum, N_h, n_h 를 붙인다."""
    by = defaultdict(list)
    for r in pool:
        by[strat(r)].append(r)
    alloc = allocate({k: len(v) for k, v in by.items()}, n)
    out = []
    for k in sorted(by):
        rows = by[k]
        idx = rng.permutation(len(rows))[: alloc[k]]
        for i in sorted(idx.tolist()):
            out.append({**rows[i], "stratum": k, "N_h": len(rows), "n_h": alloc[k]})
    return out


def slot_rows(recs_by_armseed: dict, arm: str, seed: int, *, final: bool) -> list[dict]:
    """(arm, seed) 의 sub-answer 행. final 이면 최종 라운드(물려받은 답 포함), 아니면 라운드 0.
    question 은 원래 문구(라운드 0)다."""
    rows = []
    for qid, rec in sorted(recs_by_armseed.get((arm, seed), {}).items()):
        r0 = {b["branch"]: {sa["index"]: sa for sa in b["sub_answers"]} for b in rec["rounds"][0]["branches"]}
        rr = final_round(rec) if final else rec["rounds"][0]
        for br in rr["branches"]:
            for sa in br["sub_answers"]:
                pr = sa.get("produced_round")
                rows.append({"arm": arm, "seed": seed, "query_id": qid, "branch": br["branch"],
                             "sq_index": sa["index"], "question": r0[br["branch"]][sa["index"]]["sub_question"],
                             "asked": sa["sub_question"], "text": sa.get("text") or "",
                             "citations": sa.get("citations") or [], "status": sa.get("status"),
                             "produced_round": rr["round"] if pr is None else pr, "round": rr["round"],
                             "r0_status": r0[br["branch"]][sa["index"]].get("status")})
    return rows


def is_answered(r: dict) -> bool:
    if r["status"] is not None:
        return r["status"] == "answered"
    return bool(r["citations"]) and bool(r["text"].strip())     # legacy: 유효 인용이 붙은 답


# --------------------------------------------------------------------------- B 컨텍스트 재구성
class ContextMismatch(RuntimeError):
    pass


class ContextBuilder:
    """retriever.search(query, top_k) 로 그 라운드의 branch 컨텍스트를 다시 만든다. LLM 0."""

    def __init__(self, retriever, anchors: dict[str, str], top_k_for_round: Callable[[int], int],
                 max_chars: int = 12000, verify: bool = True):
        self.retriever, self.anchors = retriever, anchors
        self.top_k, self.max_chars, self.verify = top_k_for_round, int(max_chars), verify

    def for_subq(self, rec: dict, branch: str, idx: int, produced_round: int) -> str:
        rr = next(r for r in rec["rounds"] if r["round"] == produced_round)
        br = next(b for b in rr["branches"] if b["branch"] == branch)
        sas = br["sub_answers"]
        if any(sa.get("produced_round") is not None for sa in sas):
            targets = [sa for sa in sas if sa.get("produced_round") == produced_round]
        else:
            targets = list(sas)
        anchor = (self.anchors.get(str(rec["query_id"])) or "").strip()
        k = self.top_k(produced_round)
        chunks: list[Chunk] = []
        seen: set[str] = set()
        own: dict[int, list[str]] = {}
        for sa in targets:                          # run_branch 와 같은 순서: 대상 sub-question 순
            q = (anchor + " " + sa["sub_question"]).strip() if anchor else sa["sub_question"]
            res = self.retriever.search(q, k)
            own[sa["index"]] = [c.chunk_id for c in res]
            for c in res:
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    chunks.append(c)
        ctx = format_context(chunks, self.max_chars)
        want = br.get("context_sha256") or ""
        if self.verify and want and hashlib.sha256(ctx.encode("utf-8")).hexdigest() != want:
            raise ContextMismatch("컨텍스트 재구성 불일치: %s %s r%d" % (rec["query_id"], branch, produced_round))
        included = _included_lengths(chunks, self.max_chars)
        cmap = {c.chunk_id: c for c in chunks}
        parts = []
        for cid in own.get(idx, []):
            n = included.get(cid, 0)
            if n >= MIN_BODY:
                c = cmap[cid]
                parts.append("(" + c.title + " > " + c.section_path + ") " + c.text.strip().replace("\n", " ")[:n])
        return "\n\n".join(parts) if parts else "(컨텍스트 없음)"


def _included_lengths(chunks: list[Chunk], max_chars: int) -> dict[str, int]:
    """format_context 가 각 청크 본문을 몇 자 실었는지 (d16 recording_format_context 와 같은 계산)."""
    used, out = 0, {}
    for c in chunks:
        text = c.text.strip().replace("\n", " ")
        head = "[" + c.chunk_id + "] (" + c.title + " > " + c.section_path + ") "
        room = max_chars - used
        if room <= len(head) + 80:
            break
        body = text[: max(0, room - len(head))]
        out[c.chunk_id] = len(body)
        used += len(head) + len(body)
    return out


def default_builder(run_dir: Path, arm: str, seed: int) -> ContextBuilder:
    """저장소 BM25 인덱스와 그 arm·seed 의 config.yaml 로 만든다 (GPU·LLM 없음)."""
    import yaml
    from lfm.config import Config, load_config
    from lfm.retriever import BM25Retriever
    p = Path(run_dir) / arm / ("s%d" % seed) / "config.yaml"
    cfg = Config(yaml.safe_load(io.open(p, encoding="utf-8")), source=str(p)) if p.exists() else load_config(ROOT / "config.yaml")
    with io.open(ROOT / "data/topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    anchors = {str(t["id"]): " ".join(t.get("terms") or []) for t in topics}
    return ContextBuilder(BM25Retriever(cfg), anchors, cfg.top_k_for_round,
                          int(cfg.get("branch.max_context_chars", 12000)))


# --------------------------------------------------------------------------- 시트 쓰기
def write_family(out_dir: Path, family: str, title: str, tabs: list[dict], keys: list[dict], meta: dict) -> dict:
    """tabs: [{id, name, kind(A|B), items:[{id, claim, ev}]}]. 템플릿은 make_combined_sheet.HTML."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = []
    for t in tabs:
        labels, (hc, he), rules = (LABELS_A, H_A, RULES_A) if t["kind"] == "A" else (LABELS_B, H_B, RULES_B)
        sets.append({"id": t["id"], "name": t["name"], "labels": labels, "h_claim": hc, "h_ev": he,
                     "items": [{"id": x["id"], "rules": rules, "claim": x["claim"], "ev": x["ev"]} for x in t["items"]]})
    ids = [x["id"] for s in sets for x in s["items"]]
    assert len(ids) == len(set(ids)), "문항 id 가 겹친다"
    h = COMBINED_HTML
    for a, b in (("재투고 라벨 — 통합", title), ("r2_combined_v1", "r2_%s_v1" % family),
                 ("labels_all_filled.csv", "labels_%s_filled.csv" % family)):
        assert a in h, a
        h = h.replace(a, b)
    html = h.replace("__SETS__", json.dumps(sets, ensure_ascii=False))
    hp = out_dir / ("labeler_%s.html" % family)
    io.open(hp, "w", encoding="utf-8").write(html)
    kp = sealed_dir(out_dir) / ("_key_%s.csv" % family)
    cols = ["item_id", "tab", "arm", "arm_hash", "seed", "query_id", "branch", "sq_index", "produced_round",
            "status", "r0_status", "stratum", "N_h", "n_h", "source", "shared_with"]
    with io.open(kp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for k in keys:
            w.writerow({c: ("" if k.get(c) is None else k.get(c)) for c in cols})
    meta = {**meta, "family": family, "rng_seed": SET_SEEDS.get(family), "arm_salt": ARM_SALT,
            "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
            "n_items": {t["id"]: len(t["items"]) for t in tabs}}
    io.open(sealed_dir(out_dir) / ("_meta_%s.json" % family), "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=1))
    print("%s: %s -> %s" % (family, ", ".join("%s %d" % (t["name"], len(t["items"])) for t in tabs), hp))
    return {"html": hp, "key": kp, "meta": meta}


def _number(items_keys: list[tuple[dict, dict]], prefix: str, rng) -> tuple[list[dict], list[dict]]:
    order = rng.permutation(len(items_keys)).tolist()
    items, keys = [], []
    for j, i in enumerate(order):
        it, k = items_keys[i]
        iid = "%s%03d" % (prefix, j)
        items.append({**it, "id": iid})
        keys.append({**k, "item_id": iid})
    return items, keys


def _key(r: dict, tab: str, **extra) -> dict:
    return {"tab": tab, "arm": r["arm"], "arm_hash": arm_hash(r["arm"]), "seed": r["seed"], "query_id": r["query_id"],
            "branch": r["branch"], "sq_index": r["sq_index"], "produced_round": r["produced_round"],
            "status": r["status"] if r["status"] is not None else ("legacy_answered" if is_answered(r) else "legacy_empty"),
            "r0_status": r.get("r0_status"), "stratum": r.get("stratum"), "N_h": r.get("N_h"), "n_h": r.get("n_h"), **extra}


# --------------------------------------------------------------------------- 세트
def build_ph(run_dir: Path, out_dir: Path, builder: ContextBuilder | None = None, *, seed: int = 1,
             e_arm="E-GR", l_arm="L-GR", n_eb=30, n_ea=30, n_la=20) -> dict:
    """P-H: pilot5 s1 라운드 0. E insufficient 30 (B) + E answered 30 (A) + L answered 20 (A). 관점 층화 비례."""
    runs = load_run(run_dir)
    rng = np.random.default_rng(SET_SEEDS["ph"])
    e0, l0 = slot_rows(runs, e_arm, seed, final=False), slot_rows(runs, l_arm, seed, final=False)
    eb = stratified_sample([r for r in e0 if r["status"] == "insufficient"], lambda r: r["branch"], n_eb, rng)
    ea = stratified_sample([r for r in e0 if r["status"] == "answered"], lambda r: r["branch"], n_ea, rng)
    la = stratified_sample([r for r in l0 if is_answered(r)], lambda r: r["branch"], n_la, rng)
    for rows, tag in ((eb, "E:insufficient"), (ea, "E:answered"), (la, "L:answered")):
        for r in rows:
            r["stratum"] = tag + ":" + r["stratum"]
    a_pairs = [({"claim": r["question"], "ev": shown_answer(r["text"])}, _key(r, "A")) for r in ea + la]
    # G5 짝을 만들기 위해: 뽑힌 슬롯마다 다른 arm 의 라운드 0 답이 비었으면 UA=0 행을 키에만 남긴다 (문항 아님)
    e_by = {(r["query_id"], r["branch"], r["sq_index"]): r for r in e0}
    l_by = {(r["query_id"], r["branch"], r["sq_index"]): r for r in l0}
    empty_keys, seen = [], set()
    for rows, other in ((eb + ea, l_by), (la, e_by)):
        for r in rows:
            sl = (r["query_id"], r["branch"], r["sq_index"])
            o = other.get(sl)
            if o is None or (o["arm"], sl) in seen or (o["status"] is None and is_answered(o)) or o["status"] == "answered":
                continue
            seen.add((o["arm"], sl))
            empty_keys.append(_key({**o, "stratum": "pair_empty", "N_h": "na", "n_h": "na"}, "", source="empty:UA=0"))
    a_items, a_keys = _number(a_pairs, "pa", rng)
    b_items, b_keys = [], []
    if eb:
        builder = builder or default_builder(run_dir, e_arm, seed)
        b_pairs = [({"claim": r["question"],
                     "ev": builder.for_subq(runs[(e_arm, seed)][r["query_id"]], r["branch"], r["sq_index"], r["produced_round"])},
                    _key(r, "B")) for r in eb]
        b_items, b_keys = _number(b_pairs, "pb", rng)
    tabs = [{"id": "ph_a", "name": "세트 1 (A)", "kind": "A", "items": a_items},
            {"id": "ph_b", "name": "세트 2 (B)", "kind": "B", "items": b_items}]
    meta = {"source": str(run_dir), "seed": seed, "spec": "PREREG_R2 §9 P-H",
            "pools": {"E_insufficient": sum(1 for r in e0 if r["status"] == "insufficient"),
                      "E_answered": sum(1 for r in e0 if r["status"] == "answered"),
                      "L_answered": sum(1 for r in l0 if is_answered(r))}}
    return write_family(out_dir, "ph", "R2 라벨 — P-H", tabs, a_keys + b_keys + empty_keys, meta)


def _items_from_html(path: Path) -> list[dict]:
    h = io.open(path, encoding="utf-8").read()
    return json.loads(re.search(r"const ITEMS = (\[.*?\]);\n", h, re.S).group(1))


def build_h3(run_dir: Path, out_dir: Path, *, seed: int = 1, e_arm="E-GR", n_ea=35, n_anchor=25,
             h2_dir: Path | None = None) -> dict:
    """H3: held-out s1 라운드 0. E answered 약 35 (A) + H2-A 앵커 25 (A). H2-B 100건은 B 를 다시 라벨하지 않고
    E 의 라운드 0 status 를 키에 붙여 재사용한다(_key_h3_reuseB.csv)."""
    h2_dir = Path(h2_dir) if h2_dir else ROOT / "results/d18"
    runs = load_run(run_dir)
    rng = np.random.default_rng(SET_SEEDS["h3"])
    e0 = slot_rows(runs, e_arm, seed, final=False)
    ea = stratified_sample([r for r in e0 if r["status"] == "answered"], lambda r: r["branch"], n_ea, rng)
    for r in ea:
        r["stratum"] = "E:answered:" + r["stratum"]
    pairs = [({"claim": r["question"], "ev": shown_answer(r["text"])}, _key(r, "A")) for r in ea]
    keyA, htmlA = h2_dir / "_key_h2hoA.csv", h2_dir / "labeler_h2hoA.html"
    n_anchor_got = 0
    if keyA.exists() and htmlA.exists():
        items = {x["id"]: x for x in _items_from_html(htmlA)}
        pool = [dict(k) for k in csv.DictReader(io.open(keyA, encoding="utf-8"))]
        got = stratified_sample(pool, lambda k: k["branch"], n_anchor, rng)
        for k in got:
            it = items[k["item_id"]]
            pairs.append(({"claim": it["claim"], "ev": it["ev"] or NO_ANSWER},
                          {"tab": "A", "arm": "H2-anchor", "arm_hash": arm_hash("H2-anchor"), "seed": "",
                           "query_id": k["query_id"], "branch": k["branch"], "sq_index": k["sq_index"],
                           "stratum": "anchor:" + k["branch"], "N_h": k["N_h"], "n_h": k["n_h"],
                           "source": "h2hoA:" + k["item_id"]}))
        n_anchor_got = len(got)
    else:
        print("H2-A 앵커 파일이 없다 (%s) — 앵커 없이 만든다" % keyA)
    items, keys = _number(pairs, "h3", rng)
    reuse = []
    keyB = h2_dir / "_key_h2hoB.csv"
    if keyB.exists():
        by = {(r["query_id"], r["branch"], int(r["sq_index"])): r for r in e0}
        for k in csv.DictReader(io.open(keyB, encoding="utf-8")):
            r = by.get((k["query_id"], k["branch"], int(k["sq_index"])))
            reuse.append({"h2_item": k["item_id"], "query_id": k["query_id"], "branch": k["branch"],
                          "sq_index": k["sq_index"], "h2_stratum": k.get("stratum"),
                          "e_status_r0": r["status"] if r else "", "found": int(r is not None)})
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with io.open(sealed_dir(out_dir) / "_key_h3_reuseB.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["h2_item", "query_id", "branch", "sq_index", "h2_stratum", "e_status_r0", "found"])
            w.writeheader(); w.writerows(reuse)
    tabs = [{"id": "h3_a", "name": "세트 1 (A)", "kind": "A", "items": items}]
    meta = {"source": str(run_dir), "seed": seed, "spec": "PREREG_R2 §9 H3",
            "n_anchor": n_anchor_got, "n_reuseB": len(reuse)}
    return write_family(out_dir, "h3", "R2 라벨 — H3", tabs, keys, meta)


def build_h4(run_dir: Path, out_dir: Path, *, n_slots: int = G5_DEFAULT, seeds=(1, 2, 3),
             arms=("E-SQ", "E-GR", R1)) -> dict:
    """H4: held-out 최종 출력. 슬롯 = (seed, 질의, 관점, 번호), seed × 관점 층화 비례 추출.
    arm 마다 최종 출력(슬롯별 마지막 생성본, E-R1 = E-GR 라운드 0). 질문은 원래 문구.
    빈 답은 라벨하지 않는다(키에 UA=0 으로 남는다). 한 슬롯에서 글자가 같은 답은 한 문항으로 묶는다."""
    runs = with_r1(load_run(run_dir))
    rng = np.random.default_rng(SET_SEEDS["h4"])
    per = {(a, s): {(r["query_id"], r["branch"], r["sq_index"]): r for r in slot_rows(runs, a, s, final=True)}
           for a in arms for s in seeds if (a, s) in runs}
    slots = []
    for s in seeds:
        have = [set(per[(a, s)]) for a in arms if (a, s) in per]
        if len(have) != len(arms):
            continue
        for sl in sorted(set.intersection(*have)):
            slots.append({"seed": s, "slot": sl})
    picked = stratified_sample(slots, lambda x: "s%d:%s" % (x["seed"], x["slot"][1]), n_slots, rng)
    pairs, keys_empty = [], []
    for p in picked:
        rows = [per[(a, p["seed"])][p["slot"]] for a in arms]
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            r = {**r, "stratum": p["stratum"], "N_h": p["N_h"], "n_h": p["n_h"]}
            if not strip_cites(r["text"]):
                keys_empty.append(_key(r, "", source="empty:UA=0"))
            else:
                groups[strip_cites(r["text"])].append(r)
        for txt, rs in groups.items():
            shared = ";".join(arm_hash(r["arm"]) for r in rs)
            item = {"claim": rs[0]["question"], "ev": txt}
            # 키는 arm 마다 한 줄 — 같은 문항을 공유한다
            pairs.append((item, [_key(r, "A", shared_with=shared) for r in rs]))
    order = rng.permutation(len(pairs)).tolist()
    items, keys = [], []
    for j, i in enumerate(order):
        it, ks = pairs[i]
        iid = "h4%03d" % j
        items.append({**it, "id": iid})
        keys += [{**k, "item_id": iid} for k in ks]
    tabs = [{"id": "h4_a", "name": "세트 1 (A)", "kind": "A", "items": items}]
    meta = {"source": str(run_dir), "spec": "PREREG_R2 §9 H4", "n_slots": len(picked), "n_slots_requested": n_slots,
            "n_slot_universe": len(slots), "n_empty_answers": len(keys_empty), "arms": list(arms), "seeds": list(seeds)}
    return write_family(out_dir, "h4", "R2 라벨 — H4", tabs, keys + keys_empty, meta)


def build_h5(run_dir: Path, out_dir: Path, builder: ContextBuilder | None = None, *, seeds=(1, 2, 3),
             arm="E-SQ", n_insufficient=30, n_new_answered=10) -> dict:
    """H5: held-out E-SQ 최종 insufficient (seed × 관점 층화) + 라운드 ≥ 1 에서 새로 answered 가 된 층. B 라벨.
    컨텍스트는 그 답을 만든 라운드의 검색으로 다시 만든다. 질문은 원래 문구."""
    runs = load_run(run_dir)
    rng = np.random.default_rng(SET_SEEDS["h5"])
    fin = [r for s in seeds for r in slot_rows(runs, arm, s, final=True)]
    ins = stratified_sample([r for r in fin if r["status"] == "insufficient"],
                            lambda r: "ins:s%d:%s" % (r["seed"], r["branch"]), n_insufficient, rng)
    new = stratified_sample([r for r in fin if r["status"] == "answered" and r["produced_round"] >= 1
                             and r["r0_status"] in ("insufficient", "format_fail", "dropped")],
                            lambda r: "new_answered:s%d:%s" % (r["seed"], r["branch"]), n_new_answered, rng)
    pairs = []
    for r in ins + new:
        b = builder or default_builder(run_dir, arm, r["seed"])
        builder = b
        ev = b.for_subq(runs[(arm, r["seed"])][r["query_id"]], r["branch"], r["sq_index"], r["produced_round"])
        pairs.append(({"claim": r["question"], "ev": ev}, _key(r, "B")))
    items, keys = _number(pairs, "h5", rng)
    tabs = [{"id": "h5_b", "name": "세트 1 (B)", "kind": "B", "items": items}]
    meta = {"source": str(run_dir), "spec": "PREREG_R2 §9 H5", "seeds": list(seeds),
            "alloc": {"insufficient": len(ins), "new_answered": len(new)}}
    return write_family(out_dir, "h5", "R2 라벨 — H5", tabs, keys, meta)


# --------------------------------------------------------------------------- G5
def g5_slots(sd: float | None) -> int:
    """PREREG §6 G5: n = ⌈(1.645 + 0.842)² · SD² / 0.10² · 1.2⌉, 하한 120, SD 가 없으면 기본 200."""
    if sd is None or not np.isfinite(sd):
        return G5_DEFAULT
    return max(G5_FLOOR, int(math.ceil((1.645 + 0.842) ** 2 * sd ** 2 / 0.10 ** 2 * 1.2)))


def g5_from_labels(key_csv: Path, labels_csv: Path, support: dict[tuple, bool] | None = None,
                   e_arm="E-GR", l_arm="L-GR") -> dict[str, Any]:
    """P-H 라벨로 UA 짝 차이(E − L) SD 를 추정한다. UA = 사람 '답함' ∧ (support 가 있으면) 모든 문장 MiniCheck 지지.
    E 기권은 UA 0. 짝은 같은 (seed, 질의, 관점, 번호) 에서 두 arm 값이 모두 정해진 슬롯만."""
    keys = list(csv.DictReader(io.open(key_csv, encoding="utf-8")))
    lab = {}
    for r in csv.DictReader(io.open(labels_csv, encoding="utf-8-sig")):
        lab[(r.get("set", ""), r["item_id"])] = r["human_label"]
        lab[("", r["item_id"])] = r["human_label"]
    ua = defaultdict(dict)
    for k in keys:
        slot = (k["seed"], k["query_id"], k["branch"], k["sq_index"])
        if k["status"] in ("insufficient", "format_fail", "dropped", "legacy_empty"):
            ua[slot][k["arm"]] = 0
            continue
        h = lab.get(("", k["item_id"]), "")
        if not h:
            continue
        ok = h == "답함"
        if support is not None:
            ok = ok and bool(support.get((k["arm"],) + slot, False))
        ua[slot][k["arm"]] = int(ok)
    d = [v[e_arm] - v[l_arm] for v in ua.values() if e_arm in v and l_arm in v]
    sd = float(np.std(d, ddof=1)) if len(d) >= 2 else None
    note = "짝 %d개로 SD 를 추정했다" % len(d)
    if len(d) < 30:
        note += " — 짝이 30개 미만이라 SD 추정이 약하다. 보고에 적고 기본값 %d 과 함께 본다" % G5_DEFAULT
    print("G5: " + note)
    return {"n_pairs": len(d), "sd": sd, "slots": g5_slots(sd), "support_used": support is not None, "note": note}


# --------------------------------------------------------------------------- G2 (P-H)
NO_CONTEXT = "(컨텍스트 없음)"


def ph_blinding_note(key_csv: Path, html: Path) -> dict[str, Any]:
    """P-H 시트의 구조적 누출과 빈 컨텍스트 문항 — 라벨 **전에** 봉인 폴더에 적는다(독립 검증 지적, 2026-11-05).

    - 탭 B 는 E insufficient 슬롯뿐이고 같은 슬롯의 E 는 answered 일 수 없다. 그래서 탭 A 문항의 질문이 탭 B 에도 있으면
      그 A 문항은 L 이라고 추론할 수 있다 → `cross_tab_inferable`.
    - 탭 B 에서 컨텍스트가 비어 보이는 문항은 자명하게 '답없음'이다 → `b_no_context`."""
    keys = list(csv.DictReader(io.open(key_csv, encoding="utf-8")))
    slot = lambda k: (k["query_id"], k["branch"], k["sq_index"])
    b_slots = {slot(k) for k in keys if k["tab"] == "B"}
    inferable = sorted(k["item_id"] for k in keys if k["tab"] == "A" and slot(k) in b_slots)
    h = io.open(html, encoding="utf-8").read()
    sets = json.loads(re.search(r"const SETS = (\[.*?\]);\n", h, re.S).group(1))      # write_family 형식(탭 묶음)
    no_ctx = sorted(it["id"] for s in sets if s["id"] == "ph_b" for it in s["items"] if NO_CONTEXT in (it.get("ev") or ""))
    return {"cross_tab_inferable": inferable, "b_no_context": no_ctx,
            "mitigation": "탭 A 를 끝까지 라벨한 뒤 탭 B 를 연다. G2(a)(b) 를 이 문항 포함·제외로 함께 보고한다"}


def g2_from_labels(key_csv: Path, labels_csv: Path, analysis_json: Path | None = None,
                   note: dict | None = None, e_arm="E-GR", l_arm="L-GR") -> dict[str, Any]:
    """PREREG §6 G2 — 등록 문구 그대로 계산한다. 라벨이 들어오기 **전에** 작성·커밋한다.

    (a) E insufficient(탭 B) 중 사람 '답있음' ≤ 4건(≤ 15%). '일부'는 '답있음'이 아니다(라벨 규칙 문구)
    (b) E answered 의 '답함' 비율 − L answered 의 '답함' 비율 ≥ 0.10 (비가중; P-H 는 관점 층화 비례 추출)
    (c) sub-question 단위 insufficient 비율 45~85% — P-H 대상인 라운드 0(= E-R1)에서
    (d) E arm 평균 legacy coverage ≥ 0.15 — 문구가 해석을 정하지 않아 모든 읽기를 내고, 판정은 등록 전 고정한 읽기로 한다
    """
    keys = {k["item_id"]: k for k in csv.DictReader(io.open(key_csv, encoding="utf-8")) if k["item_id"]}
    lab = {r["item_id"]: r["human_label"].strip() for r in csv.DictReader(io.open(labels_csv, encoding="utf-8-sig"))}
    note = note or {"cross_tab_inferable": [], "b_no_context": []}
    missing = sorted(i for i, k in keys.items() if k["tab"] in ("A", "B") and not lab.get(i))

    def part_a(excl=()):
        b = [i for i, k in keys.items() if k["tab"] == "B" and i not in excl]
        n_yes = sum(1 for i in b if lab.get(i) == "답있음")
        return {"n": len(b), "답있음": n_yes, "일부": sum(1 for i in b if lab.get(i) == "일부"),
                "rate": n_yes / len(b) if b else None, "pass": bool(b) and n_yes <= 4 and n_yes / len(b) <= 0.15}

    def part_b(excl=()):
        rate = {}
        for arm in (e_arm, l_arm):
            a = [i for i, k in keys.items() if k["tab"] == "A" and k["arm"] == arm and i not in excl]
            rate[arm] = {"n": len(a), "답함": sum(1 for i in a if lab.get(i) == "답함"),
                         "rate": sum(1 for i in a if lab.get(i) == "답함") / len(a) if a else None}
        d = (rate[e_arm]["rate"] - rate[l_arm]["rate"]) if None not in (rate[e_arm]["rate"], rate[l_arm]["rate"]) else None
        return {**rate, "diff": d, "pass": d is not None and d >= 0.10}

    out = {"missing_labels": missing,
           "a": part_a(), "a_excl_no_context": part_a(set(note.get("b_no_context", []))),
           "b": part_b(), "b_excl_inferable": part_b(set(note.get("cross_tab_inferable", [])))}
    if analysis_json and Path(analysis_json).exists():
        an = json.load(io.open(analysis_json, encoding="utf-8"))["arm_seed"]
        r1 = an.get("E-R1·s1", {})
        ins = (r1.get("generated_rates") or {}).get("insufficient")
        out["c"] = {"round0_insufficient_rate": ins, "range": [0.45, 0.85],
                    "pass": ins is not None and 0.45 <= ins <= 0.85}
        cov = {k: v.get("mean_coverage") for k, v in an.items() if k.startswith("E-")}
        finals = [cov[k] for k in ("E-GR·s1", "E-SQ·s1", "E-BR·s1") if cov.get(k) is not None]
        readings = {"round0 (E-R1)": cov.get("E-R1·s1"),
                    "final, pooled E-GR/E-SQ/E-BR": float(np.mean(finals)) if finals else None,
                    "final, min over E-GR/E-SQ/E-BR": min(finals) if finals else None}
        out["d"] = {"coverage_by_arm": cov, "readings": readings, "bar": 0.15,
                    "pass_by_reading": {k: (v is not None and v >= 0.15) for k, v in readings.items()}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R2 라벨 시트 (LLM 호출 0)")
    ap.add_argument("set", choices=["ph", "h3", "h4", "h5", "g5", "g2", "ph_note"])
    ap.add_argument("--run-dir")
    ap.add_argument("--out-dir", default=None, help="기본: <run-dir>/sheets")
    ap.add_argument("--slots", type=int, default=G5_DEFAULT)
    ap.add_argument("--key"); ap.add_argument("--labels")
    args = ap.parse_args(argv)
    if args.set == "g5":
        print(json.dumps(g5_from_labels(Path(args.key), Path(args.labels)), ensure_ascii=False, indent=1))
        return 0
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        run_dir = ROOT / run_dir
    sealed = sealed_dir(Path(args.out_dir) if args.out_dir else run_dir / "sheets")
    if args.set == "ph_note":
        n = ph_blinding_note(sealed / "_key_ph.csv", (Path(args.out_dir) if args.out_dir else run_dir / "sheets") / "labeler_ph.html")
        io.open(sealed / "_note_ph_blinding.json", "w", encoding="utf-8").write(json.dumps(n, ensure_ascii=False, indent=1))
        print("-> %s (누출 %d · 빈 컨텍스트 %d)" % (sealed / "_note_ph_blinding.json", len(n["cross_tab_inferable"]), len(n["b_no_context"])))
        return 0
    if args.set == "g2":
        np_ = sealed / "_note_ph_blinding.json"
        res = g2_from_labels(Path(args.key) if args.key else sealed / "_key_ph.csv", Path(args.labels),
                             run_dir / "ANALYSIS.json", json.load(io.open(np_, encoding="utf-8")) if np_.exists() else None)
        io.open(run_dir / "G2_RESULT.json", "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    out = Path(args.out_dir) if args.out_dir else run_dir / "sheets"
    {"ph": lambda: build_ph(run_dir, out), "h3": lambda: build_h3(run_dir, out),
     "h4": lambda: build_h4(run_dir, out, n_slots=args.slots), "h5": lambda: build_h5(run_dir, out)}[args.set]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
