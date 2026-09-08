"""MiniCheck 가 단일 문서 전제를 내부에서 쪼갠 적이 있는가. **API 호출 0.**

왜 급한가: 원고 §VI-I 가 이렇게 적었다 —

  "run with the cited chunks as a single document, with an assertion on every call
   that no internal chunking occurred — **and none did, on any of the calls.**
   It is the only configuration in this paper with zero aggregation freedom."

그런데 그 검사가 **구현된 적이 없다.** `experiments/d9_minicheck.py` 의 `Scorer` 는
`self.violations = 0` 을 만들어 두고 마지막에 출력만 하고, `score()` 안에서 한 번도
증가시키지 않는다. 즉 D9·V′ 가 찍은 "내부 청크 분할 위반: 0 건" 은 **절대 실패할 수
없는 검사**였다.

M1′ 배선 중 실제로 작동하는 가드를 달자마자 Test A 첫 토픽에서 3,791자 전제가
2조각으로 쪼개졌다. MiniCheck 는 청크별 점수의 **최댓값**을 문서 점수로 쓴다.
쪼개지면 논문이 "없앴다" 고 주장한 집계 자유도가 그대로 돌아온다.

여기서는 고치지 않는다. **얼마나 자주 일어났는지만 센다.** 대상은 원고의 수치가
실제로 나온 두 모집단이다:

  (1) V′ 60건        — precision 0.850 / recall 0.694 / N층 회수 0.667 이 나온 표본
  (2) 전체 294문장    — 정상 64.29% · 교란 0.00% · 순증 64.29pp 가 나온 모집단

쪼개진 건이 있으면 그 건들의 점수가 최댓값 집계로 얻어진 것이므로, 해당 수치의
"집계 자유도 0" 서술을 고쳐야 한다.
"""

from __future__ import annotations

import csv
import io
import json
import sys
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
from lfm.nli import load_nli                         # noqa: E402
from lfm.perspectives import decompose               # noqa: E402
from lfm.retriever import BM25Retriever              # noqa: E402
from lfm.types import split_sentences                # noqa: E402

OUT = ROOT / "results/vprime"


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

    load_nli(ccfg)                                   # torch/CUDA 순서를 make_sheet 와 맞춘다
    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, 15, int(cfg["seed"]))

    print("모집단 재생 (캐시 재생, 생성 호출 0) ...", flush=True)
    pop: list[dict] = []
    for n, t in enumerate(picked):
        anchor = " ".join(t.get("terms") or [])
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            for sa in o.sub_answers:
                cited = [cmap[c].text for c in sa.citations if c in cmap]
                if not cited:
                    continue
                for sent in split_sentences(sa.text):
                    pop.append({"item_id": "v%03d" % len(pop), "sentence": sent,
                                "premises": cited})
        print("  [%2d/%2d] %s  누적 %d" % (n + 1, len(picked), t["id"], len(pop)), flush=True)

    vids = {r["item_id"] for r in
            csv.DictReader(io.open(OUT / "labels_stage1.csv", encoding="utf-8-sig"))}
    print("")
    print("모집단 %d문장 · 그중 V′ 표본 %d건" % (len(pop), len(vids)))

    sc = MiniCheckScorer()
    rows = []
    print("")
    print("채점 (MiniCheck, 단일 문서 전제) ...", flush=True)
    for i, r in enumerate(pop):
        doc = "\n\n".join(r["premises"])
        p = sc.score(doc, r["sentence"])
        rows.append({"item_id": r["item_id"], "in_vprime": int(r["item_id"] in vids),
                     "n_premises": len(r["premises"]), "doc_chars": len(doc),
                     "n_chunks": sc.last_n_chunks, "p": round(p, 6),
                     "truncated_before_scoring": 0})
        if (i + 1) % 60 == 0:
            print("  %d/%d" % (i + 1, len(pop)), flush=True)

    def rep(name, sub):
        if not sub:
            print("  %-18s (없음)" % name)
            return {}
        split = [x for x in sub if (x["n_chunks"] or 1) > 1]
        gr = [x for x in sub if x["p"] > 0.5]
        gr_split = [x for x in split if x["p"] > 0.5]
        print("  %-18s n=%3d · **쪼개짐 %d건 (%.1f%%)** · 최대 %d조각"
              % (name, len(sub), len(split), 100.0 * len(split) / len(sub),
                 max((x["n_chunks"] or 1) for x in sub)))
        print("  %-18s 그중 지지 판정 %d건 — 이 건들의 점수는 청크 최댓값이다"
              % ("", len(gr_split)))
        print("  %-18s 전체 지지 %d건 중 쪼개진 것이 %d건 (%.1f%%)"
              % ("", len(gr), len(gr_split),
                 100.0 * len(gr_split) / len(gr) if gr else 0.0))
        return {"n": len(sub), "n_split": len(split),
                "pct_split": round(100.0 * len(split) / len(sub), 4),
                "max_chunks": max((x["n_chunks"] or 1) for x in sub),
                "n_grounded": len(gr), "n_grounded_split": len(gr_split)}

    print("")
    print("=" * 74)
    print("내부 청크 분할 — 원고는 '한 번도 없었다' 고 적었다")
    print("=" * 74)
    summary = {"all_294": rep("전체 모집단", rows),
               "vprime_60": rep("V′ 표본", [x for x in rows if x["in_vprime"]])}

    out = ROOT / "results/vprime"
    with io.open(out / "minicheck_chunk_audit.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    io.open(out / "minicheck_chunk_audit.json", "w", encoding="utf-8").write(
        json.dumps({"summary": summary,
                    "scorer": {"model": MiniCheckScorer.MODEL,
                               "max_model_len": MiniCheckScorer.MAX_LEN,
                               "truncate_headroom": MiniCheckScorer.HEADROOM,
                               "truncated_calls": sc.truncated}},
                   ensure_ascii=False, indent=1))
    print("")
    print("scorer 자체 자르기(2048-64 토큰 초과): %d건" % sc.truncated)
    print("-> %s" % (out / "minicheck_chunk_audit.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
