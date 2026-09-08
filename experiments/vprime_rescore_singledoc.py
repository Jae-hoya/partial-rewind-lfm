"""V′ 재채점 — **진짜 단일 문서 전제**로. API 호출 0. 공저자 지시 순서 2.

이전 채점은 `chunk_size` 를 안 넘겨 `minicheck.py:148` 의 하드코딩 500 이 먹었고,
294문장 중 189건(64.3%)이 쪼개져 **청크 최댓값**으로 채점됐다. 여기서는
`chunk_size=7200` 을 직접 넘겨 한 조각을 강제한다(가장 긴 전제 7,129 토큰).

**전제 범위를 두 곳에서 다르게 쓴다. 섞으면 안 된다** — 오늘 두 번 낸 오류다.

  관문 3축(precision·recall·N층 회수)  문장이 **인용한 청크**만 이어붙인 전제
  음성 통제                            **토픽 전체**의 인용 청크 풀 (D9-A 와 동일)

음성 통제는 D9-A 와 같은 방식이다 — 토픽 i 의 문장에 토픽 (i+7) mod 15 의 풀을
붙이고, 위양성 비율이 5% 이하인지 본다. 새 설정이 설계 구간(2,048) 밖인
7,200 토큰에서 도는 것이므로, **타당성은 이 통제가 판정한다.**

여기서 관문 판정을 내리지 않는다. 수치만 낸다 — 판정은 공저자 몫이다.
"""

from __future__ import annotations

import csv
import io
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml                                          # noqa: E402

from diag_sweep import CONDITIONS, pick_topics       # noqa: E402

from lfm.branch import run_branch                    # noqa: E402
from lfm.config import Config, load_config           # noqa: E402
from lfm.estimator import MiniCheckScorer            # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present      # noqa: E402
from lfm.perspectives import decompose               # noqa: E402
from lfm.retriever import BM25Retriever              # noqa: E402
from lfm.types import split_sentences                # noqa: E402

OUT = ROOT / "results/vprime"
POS, NEG = "지지", "불지지"
TAU = 0.5                # 엄격 초과
SHIFT = 7                # D9-A 와 동일한 토픽 교란


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * max(0.0, (c - h) / d), 100 * min(1.0, (c + h) / d))


def cell(k, n):
    if not n:
        return "—"
    lo, hi = wilson(k, n)
    return "%d/%d = %.4f [%.3f, %.3f]" % (k, n, k / n, lo / 100, hi / 100)


def main() -> int:
    load_dotenv_if_present(str(ROOT / ".env"))
    cfg = load_config(ROOT / "config.yaml")
    spec = CONDITIONS["C3_partial_heldout"]
    data = cfg.as_dict()
    if spec["top_k"] is not None:
        data["retrieval"]["top_k"] = spec["top_k"]
    if spec["index"] == "partial":
        data["paths"]["index_dir"] = data["paths"]["index_dir_partial"]
    ccfg = Config(data, source="C3_partial_heldout")

    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, 15, int(cfg["seed"]))

    print("모집단 재생 (캐시 재생, 생성 호출 0) ...", flush=True)
    pop, sents_by, pool_by = [], {}, {}
    for n, t in enumerate(picked):
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        s_, p_ = [], []
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            p_ += [c.text for c in o.cited_chunks()]
            for sa in o.sub_answers:
                cited = [cmap[c].text for c in sa.citations if c in cmap]
                if not cited:
                    continue
                for sent in split_sentences(sa.text):
                    pop.append({"item_id": "v%03d" % len(pop), "topic": tid,
                                "sentence": sent, "premises": cited})
                    s_.append(sent)
        sents_by[tid], pool_by[tid] = s_, p_
        print("  [%2d/%2d] %s  누적 %d" % (n + 1, len(picked), tid, len(pop)), flush=True)

    sc = MiniCheckScorer()
    print("")
    print("MiniCheck: chunk_size=%d · max_model_len=%d" % (sc.chunk_size, sc.max_model_len))

    # ---------------- 1. 문장 단위 채점 (인용 청크 전제) --------------------
    print("")
    print("문장 %d건 채점 (인용 청크 전제) ..." % len(pop), flush=True)
    t0 = time.time()
    splits = 0
    for i, r in enumerate(pop):
        doc = "\n\n".join(r["premises"])
        r["p_mc_new"] = sc.score(doc, r["sentence"])
        r["n_chunks"] = sc.last_n_chunks
        if (sc.last_n_chunks or 1) > 1:
            splits += 1
        if (i + 1) % 50 == 0:
            print("  %d/%d  (%.0f분 경과)" % (i + 1, len(pop), (time.time() - t0) / 60),
                  flush=True)
    print("  **쪼개진 건 %d / %d**  (%.0f분)" % (splits, len(pop), (time.time() - t0) / 60))
    if splits:
        print("  **0 이 아니다. chunk_size 를 더 올려야 한다. 여기서 멈춘다.**")
        return 1

    # ---------------- 2. 관문 3축 (사람 라벨 60건) --------------------------
    lab = {r["item_id"]: r["human_label"].strip() for r in
           csv.DictReader(io.open(OUT / "labels_stage1.csv", encoding="utf-8-sig"))}
    key = {r["item_id"]: r for r in
           csv.DictReader(io.open(OUT / "_key_stage1.csv", encoding="utf-8-sig"))}
    P = {r["item_id"]: r for r in pop}

    it = [{"id": i, "h": h, "k": key[i], "p": P[i]["p_mc_new"]}
          for i, h in lab.items() if h in (POS, NEG) and i in key and i in P]
    sup = [x for x in it if x["h"] == POS]
    tgt = [x for x in sup if x["k"]["lab_g1"] == "0"]
    fls = [x for x in it if x["k"]["lab_g1"] == "0" and x["h"] == NEG]
    keep = [x for x in it if x["k"]["lab_g1"] == "1" and x["h"] == POS]

    def g(x):
        return x["p"] > TAU

    pn = [x for x in it if g(x)]
    res = {
        "precision": (sum(1 for x in pn if x["h"] == POS), len(pn)),
        "recall": (sum(1 for x in sup if g(x)), len(sup)),
        "n_recovery": (sum(1 for x in tgt if g(x)), len(tgt)),
        "false_recovery": (sum(1 for x in fls if g(x)), len(fls)),
        "retention": (sum(1 for x in keep if g(x)), len(keep)),
    }
    print("")
    print("=" * 74)
    print("관문 3축 — 사람 라벨 %d건 · 전제 = 문장이 인용한 청크" % len(it))
    print("=" * 74)
    OLD = {"precision": "34/40 = 0.8500", "recall": "34/49 = 0.6939",
           "n_recovery": "22/33 = 0.6667", "false_recovery": "3/7 = 0.4286",
           "retention": "12/16 = 0.7500"}
    for k_, (a, b) in res.items():
        print("  %-16s %-34s   (이전 %s)" % (k_, cell(a, b), OLD[k_]))

    # ---------------- 3. 음성 통제 (토픽 풀 전제) ---------------------------
    tids = [str(t["id"]) for t in picked]
    mapping = {tids[i]: tids[(i + SHIFT) % len(tids)] for i in range(len(tids))}
    assert all(a != b for a, b in mapping.items())
    print("")
    print("음성 통제 (토픽 전체 풀 전제, i -> (i+%d) mod %d) ..." % (SHIFT, len(tids)),
          flush=True)
    t0 = time.time()
    norm, shuf, nrows, nsplit = [], [], [], 0
    for tid in tids:
        own = "\n\n".join(pool_by[tid])
        other = "\n\n".join(pool_by[mapping[tid]])
        if not sents_by[tid] or not own or not other:
            continue
        for s_ in sents_by[tid]:
            a = sc.score(own, s_);   ca = sc.last_n_chunks
            b = sc.score(other, s_); cb = sc.last_n_chunks
            nsplit += int((ca or 1) > 1) + int((cb or 1) > 1)
            norm.append(a); shuf.append(b)
            nrows.append({"src_topic": tid, "premise_topic": mapping[tid],
                          "n_premise_chunks": len(pool_by[tid]),
                          "p_normal": round(a, 6), "p_shuffled": round(b, 6)})
        print("  %s  누적 %d  (%.0f분)" % (tid, len(norm), (time.time() - t0) / 60), flush=True)
    gn = 100.0 * sum(1 for x in norm if x > TAU) / len(norm)
    gs = 100.0 * sum(1 for x in shuf if x > TAU) / len(shuf)
    print("")
    print("=" * 74)
    print("음성 통제 — 전제 = 토픽 전체 풀 (관문 3축과 범위가 다르다)")
    print("=" * 74)
    print("  n=%d · 쪼개진 건 %d · 정상 %.2f%% · **교란 %.2f%%** · 순증 %.2f%%p"
          % (len(norm), nsplit, gn, gs, gn - gs))
    print("  기준 교란 <= 5%%  ->  %s" % ("통과" if gs <= 5.0 else "**탈락**"))
    print("  이전(쪼갬 포함): 정상 64.29% · 교란 0.00% · 순증 64.29%p")

    # ---------------- 산출 -------------------------------------------------
    with io.open(OUT / "rescore_singledoc_rows.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "topic", "n_chunks", "p_mc_new"],
                           extrasaction="ignore")
        w.writeheader(); w.writerows(pop)
    with io.open(OUT / "rescore_singledoc_negative.csv", "w", encoding="utf-8",
                 newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(nrows[0].keys()))
        w.writeheader(); w.writerows(nrows)
    io.open(OUT / "rescore_singledoc.json", "w", encoding="utf-8").write(json.dumps({
        "scorer": {"chunk_size": sc.chunk_size, "max_model_len": sc.max_model_len,
                   "calls": sc.calls},
        "n_sentences": len(pop), "n_split_sentence_scope": splits,
        "n_split_negative_control": nsplit,
        "gates": {k_: {"k": a, "n": b, "value": (a / b if b else None),
                       "ci": wilson(a, b)} for k_, (a, b) in res.items()},
        "negative_control": {"n": len(norm), "normal_pct": round(gn, 4),
                             "shuffled_pct": round(gs, 4), "net_pp": round(gn - gs, 4),
                             "premise_scope": "topic pool"},
        "previous_with_chunking": {"precision": [34, 40], "recall": [34, 49],
                                   "n_recovery": [22, 33],
                                   "normal_pct": 64.29, "shuffled_pct": 0.00},
    }, ensure_ascii=False, indent=1))
    print("")
    print("-> %s" % (OUT / "rescore_singledoc.json"))
    print("판정은 공저자 몫이다. 여기서는 수치만 낸다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
