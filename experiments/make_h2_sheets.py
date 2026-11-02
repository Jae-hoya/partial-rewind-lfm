"""H2 라벨 시트 3종 — SPEC_R2 연산자 A(질문-답 관련성)·B(질문-컨텍스트 답 가능성) 검증용. 호출 0.
등록: results/d18/PREREG_D18.md (커밋 8b1dbad).

  H2-dev   (A)  H1 에서 모델이 답한 43건. 질문 + 답. H1 의 재판정이라는 점을 공시한다
  H2-ho-A  (A)  held-out 35토픽, Test A 라운드 0, 유효 인용이 붙은 답. 관점별 25, 토픽·관점당 최대 3
  H2-ho-B  (B)  같은 토픽·라운드, 자기 검색 청크가 200자 이상 실린 sub-question.
                빈 답 50 + 답한 답 50, 각 관점 균등. H2-ho-A 와 겹치지 않는다

화면에는 모델이 답했는지, 점수, 층이 없다. 키는 `_key_h2*.csv` 로 따로 둔다.
층별 모집단 N_h 와 표본 n_h 를 키에 함께 남겨 Hájek 가중에 쓴다.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from d15_abstention_diag import MIN_BODY  # noqa: E402
from diag_sweep import pick_topics  # noqa: E402
from make_label_tool import HTML as VPRIME_HTML  # noqa: E402

from lfm.types import CITATION_RE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/d18"
BRANCHES = ("overview_factual", "technical_mechanism", "impact_application", "controversy_limitation")

RULES_A = (
    "모델의 답이 질문이 묻는 것에 답하는지만 본다. 답이 사실인지는 따지지 않는다(그건 다른 검사다) / "
    "답함 = 질문이 묻는 대상(수치·시기·원인·비판 등)에 대한 진술이 있다. 여러 개를 묻는 질문이면 하나라도 직접 답하면 답함 / "
    "일부 = 관련은 있으나 묻는 핵심에서 비껴간다(주변 사실, 일반론) / "
    "답안함 = 같은 주제의 다른 이야기만 한다. 질문이 묻는 것에 대한 진술이 없다"
)
RULES_B = (
    "아래 컨텍스트만 보고 판단한다. 바깥 지식은 쓰지 않는다 / "
    "답있음 = 질문이 묻는 것에 대한 사실 진술이 컨텍스트에 있다. 일부만 묻는 질문이면 그 일부로 충분 / "
    "일부 = 질문과 관련된 사실은 있으나 묻는 것의 핵심에는 못 미친다 / "
    "답없음 = 질문에 답하는 사실이 없다. 주제만 같고 다른 내용이면 답없음"
)


def strip_cites(s: str) -> str:
    return re.sub(r"\s+", " ", CITATION_RE.sub("", s or "")).replace(" .", ".").strip()


def html_for(kind: str, n: int, title: str, store_key: str, filename: str) -> str:
    h = VPRIME_HTML
    if kind == "A":
        pairs = [("불지지", "답안함"), ("지지", "답함"), ("판단불가", "일부"),
                 ("판정할 문장", "질문 (sub-question)"),
                 ("근거 — 이 문장이 인용한 청크 전문", "모델의 답 (인용 표시는 지웠다)")]
    else:
        pairs = [("불지지", "답없음"), ("지지", "답있음"), ("판단불가", "일부"),
                 ("판정할 문장", "질문 (sub-question)"),
                 ("근거 — 이 문장이 인용한 청크 전문", "컨텍스트 — 모델이 받은 이 질문의 검색 청크 (실린 그대로)")]
    pairs += [("V′ 라벨 — 60건", "%s — %d건" % (title, n)), ("60건 완료", "%d건 완료" % n),
              ("vprime_stage1_v1", store_key), ("labels_stage1_filled.csv", filename)]
    for a, b in pairs:
        assert a in h, a
        h = h.replace(a, b)
    return h


def write_sheet(name, kind, title, items, keys):
    html = html_for(kind, len(items), title, name + "_v1", "labels_%s_filled.csv" % name)
    io.open(OUT / ("labeler_%s.html" % name), "w", encoding="utf-8").write(
        html.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)))
    with io.open(OUT / ("_key_%s.csv" % name), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(keys[0].keys()))
        w.writeheader()
        w.writerows(keys)
    print("%-8s %3d건 -> labeler_%s.html" % (name, len(items), name))


def stratified(pool, per_stratum, rng, cap_key=None, cap=None):
    """층별로 섞은 뒤 앞에서부터 고른다. cap_key 가 같은 것은 cap 개까지만."""
    order = rng.permutation(len(pool)).tolist()
    taken, used = [], defaultdict(int)
    for i in order:
        r = pool[i]
        if len(taken) >= per_stratum:
            break
        if cap_key is not None and used[cap_key(r)] >= cap:
            continue
        used[cap_key(r) if cap_key else None] += 1
        taken.append(r)
    return taken


def main() -> int:
    with io.open(ROOT / "data/topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    dev = {str(t["id"]) for t in pick_topics(topics, 15, 42)}
    held = sorted(str(t["id"]) for t in topics if str(t["id"]) not in dev)
    assert len(dev) == 15 and len(held) == 35
    rng = np.random.default_rng(42)

    # --- H2-dev ------------------------------------------------------------
    d16 = {}
    for line in io.open(ROOT / "results/d16/d16_subq.jsonl", encoding="utf-8"):
        x = json.loads(line)
        if x["condition"] == "C3_partial_heldout":
            d16[(x["query_id"], x["branch"], int(x["sq_index"]))] = x
    h1 = [k for k in csv.DictReader(io.open(ROOT / "results/d16/_key_d16.csv", encoding="utf-8"))
          if k["model_empty"] == "0"]
    picked = [d16[(k["query_id"], k["branch"], int(k["sq_index"]))] for k in h1]
    order = rng.permutation(len(picked)).tolist()
    items, keys = [], []
    for j, i in enumerate(order):
        x = picked[i]
        iid = "d%03d" % j
        items.append({"id": iid, "rules": RULES_A, "claim": x["sub_question"], "ev": strip_cites(x["answer"])})
        keys.append({"item_id": iid, "query_id": x["query_id"], "branch": x["branch"], "sq_index": x["sq_index"],
                     "h1_item": next(k["item_id"] for k in h1 if (k["query_id"], k["branch"], int(k["sq_index"]))
                                     == (x["query_id"], x["branch"], int(x["sq_index"]))),
                     "stratum": "all", "N_h": len(picked), "n_h": len(picked)})
    write_sheet("h2dev", "A", "H2-dev 답-질문 관련성", items, keys)

    # --- held-out 모집단 ------------------------------------------------------
    rows = [json.loads(l) for l in io.open(OUT / "subq_text.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["round"] == 0 and r["query_id"] in held]
    chunk_txt = {json.loads(l)["chunk_id"]: json.loads(l)["text"] for l in io.open(OUT / "chunks.jsonl", encoding="utf-8")}
    lmax = defaultdict(int)
    for l in io.open(OUT / "subq_text.jsonl", encoding="utf-8"):
        for cid, n in json.loads(l)["own_included"].items():
            lmax[cid] = max(lmax[cid], n)

    def shown(cid, n):
        # chunks.jsonl 은 가장 길게 실린 판본이다. 이 질문에 실린 길이만큼만 잘라 보여 준다.
        t = chunk_txt[cid]
        return t[: len(t) - (lmax[cid] - n)]

    # --- H2-ho-A -------------------------------------------------------------
    items, keys, picked_ids = [], [], set()
    for b in BRANCHES:
        pool = [r for r in rows if r["branch"] == b and r["cited"]]
        got = stratified(pool, 25, rng, cap_key=lambda r: r["query_id"], cap=3)
        for r in got:
            picked_ids.add((r["query_id"], r["branch"], r["sq_index"]))
            keys.append({"query_id": r["query_id"], "branch": b, "sq_index": r["sq_index"],
                         "stratum": b, "N_h": len(pool), "n_h": len(got), "answer": r["answer"]})
    order = rng.permutation(len(keys)).tolist()
    keys = [keys[i] for i in order]
    for j, k in enumerate(keys):
        k["item_id"] = "a%03d" % j
        r = next(x for x in rows if (x["query_id"], x["branch"], x["sq_index"]) == (k["query_id"], k["branch"], k["sq_index"]))
        items.append({"id": k["item_id"], "rules": RULES_A, "claim": r["sub_question"], "ev": strip_cites(k.pop("answer"))})
    keys = [{"item_id": k["item_id"], **{x: k[x] for x in ("query_id", "branch", "sq_index", "stratum", "N_h", "n_h")}} for k in keys]
    write_sheet("h2hoA", "A", "H2-held-out-A 답-질문 관련성", items, keys)

    # --- H2-ho-B -------------------------------------------------------------
    items, keys = [], []
    alloc = {"empty": [13, 13, 12, 12], "answered": [13, 13, 12, 12]}
    for status in ("empty", "answered"):
        for b, want in zip(BRANCHES, alloc[status]):
            pool = [r for r in rows if r["branch"] == b
                    and (r["empty"] if status == "empty" else r["cited"])
                    and (r["query_id"], r["branch"], r["sq_index"]) not in picked_ids
                    and any(n >= MIN_BODY for n in r["own_included"].values())]
            got = stratified(pool, want, rng, cap_key=lambda r: r["query_id"], cap=3)
            for r in got:
                ctx = [cid for cid in r["own"] if r["own_included"].get(cid, 0) >= MIN_BODY]
                keys.append({"query_id": r["query_id"], "branch": b, "sq_index": r["sq_index"],
                             "stratum": status + ":" + b, "model_empty": r["empty"],
                             "N_h": len(pool), "n_h": len(got),
                             "_q": r["sub_question"],
                             "_ev": "\n\n".join(shown(cid, r["own_included"][cid]) for cid in ctx)})
    order = rng.permutation(len(keys)).tolist()
    keys = [keys[i] for i in order]
    for j, k in enumerate(keys):
        k["item_id"] = "b%03d" % j
        items.append({"id": k["item_id"], "rules": RULES_B, "claim": k.pop("_q"), "ev": k.pop("_ev")})
    keys = [{"item_id": k["item_id"], **{x: k[x] for x in ("query_id", "branch", "sq_index", "stratum",
                                                            "model_empty", "N_h", "n_h")}} for k in keys]
    write_sheet("h2hoB", "B", "H2-held-out-B 답 가능성", items, keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
