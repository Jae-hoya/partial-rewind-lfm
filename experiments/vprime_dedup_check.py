"""이탈 2 근거 — 인용 중복을 없앤 단일 문서 전제로 V′ 100건 관문을 다시 잰다. **API 0** (cache_only).

R2 파일럿(2026-11-05)에서 sub-answer 하나가 같은 청크를 15번 인용해(인용 20, 고유 4) 전제가 81,175자가
됐다. 등록 구성(인용 횟수만큼 본문을 이어붙임)은 한 조각으로 채점할 수 없다(어떤 장치에서도). 그래서 전제를
**고유 청크, 첫 인용 순서**로 바꾼다 — `BranchOutput.cited_chunks()` 와 같은 규칙이다.

이 스크립트는 V′ 모집단(294문장)을 `vprime_rescore_singledoc.py` 와 똑같이 캐시에서 재생해,
중복이 있는 문장만 새 전제로 다시 채점하고, `vprime_final_score.py` 와 같은 정의로 P / R / 회수 /
Hájek 가중 P 를 낸다. 판정 규칙은 등록 그대로이며 바꾸지 않는다. 음성 통제는 토픽 풀
(`cited_chunks()`, 이미 고유)을 써서 영향이 없다.
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
from vprime_score import wilson                      # noqa: E402

from lfm.branch import run_branch                    # noqa: E402
from lfm.config import Config, load_config           # noqa: E402
from lfm.estimator import MiniCheckScorer            # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present      # noqa: E402
from lfm.perspectives import decompose               # noqa: E402
from lfm.retriever import BM25Retriever              # noqa: E402
from lfm.types import split_sentences                # noqa: E402

V = ROOT / "results/vprime"
BARS = {"precision": 0.85, "recall": 0.70, "recovery": 0.50}


def main() -> int:
    load_dotenv_if_present(str(ROOT / ".env"))
    cfg = load_config(ROOT / "config.yaml")
    spec = CONDITIONS["C3_partial_heldout"]
    data = cfg.as_dict()
    if spec["top_k"] is not None:
        data["retrieval"]["top_k"] = spec["top_k"]
    if spec["index"] == "partial":
        data["paths"]["index_dir"] = data["paths"]["index_dir_partial"]
    data.setdefault("llm", {})["cache_policy"] = "cache_only"      # 미스는 API 대신 예외
    ccfg = Config(data, source="C3_partial_heldout")
    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, 15, int(cfg["seed"]))

    pop = []
    for t in picked:
        anchor = " ".join(t.get("terms") or [])
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            for sa in o.sub_answers:
                cited = [cmap[c].text for c in sa.citations if c in cmap]
                if not cited:
                    continue
                uniq = [cmap[c].text for c in dict.fromkeys(c for c in sa.citations if c in cmap)]
                for sent in split_sentences(sa.text):
                    pop.append({"item_id": "v%03d" % len(pop), "topic": str(t["id"]), "sentence": sent,
                                "dup": cited, "uniq": uniq})
    assert llm.stats.to_dict()["api_calls"] == 0

    stored = {r["item_id"]: r for r in csv.DictReader(io.open(V / "rescore_singledoc_rows.csv", encoding="utf-8"))}
    assert len(pop) == len(stored), (len(pop), len(stored))

    lab = {r["item_id"]: r["human_label"].strip()
           for r in csv.DictReader(io.open(V / "labels_stage1_filled.csv", encoding="utf-8-sig"))}
    for r in csv.DictReader(io.open(ROOT / "results/d18/labels_all_filled.csv", encoding="utf-8-sig")):
        if r["set"] == "stage2":
            lab[r["item_id"]] = r["human_label"].strip()
    key = {}
    for st in ("stage1", "stage2"):
        for r in csv.DictReader(io.open(V / ("_key_%s.csv" % st), encoding="utf-8")):
            key[r["item_id"]] = r
    assert len(lab) == 100 and set(lab) == set(key)

    sc = MiniCheckScorer()
    p_old = {r["item_id"]: float(r["p_mc_new"]) for r in stored.values()}
    p_new, changed = dict(p_old), []
    # 재생 정합성: 중복 없는 문장 몇 건을 다시 채점해 저장값과 같은지 본다
    sanity = [r for r in pop if r["dup"] == r["uniq"]][:5]
    sanity_diff = max(abs(sc.score("\n\n".join(r["dup"]), r["sentence"]) - p_old[r["item_id"]]) for r in sanity)
    for r in pop:
        if r["dup"] != r["uniq"]:
            p_new[r["item_id"]] = sc.score("\n\n".join(r["uniq"]), r["sentence"])
            assert sc.last_n_chunks == 1
            changed.append({"item_id": r["item_id"], "labeled": r["item_id"] in lab,
                            "human": lab.get(r["item_id"]), "chars_dup": len("\n\n".join(r["dup"])),
                            "chars_uniq": len("\n\n".join(r["uniq"])),
                            "p_old": p_old[r["item_id"]], "p_new": p_new[r["item_id"]],
                            "flip": (p_old[r["item_id"]] > 0.5) != (p_new[r["item_id"]] > 0.5)})

    strata = json.load(io.open(V / "stage2_strata.json", encoding="utf-8"))["strata"]
    n_h = {c: strata[c]["n_stage1"] + strata[c]["n_stage2"] for c in strata}
    w = {c: strata[c]["N_h"] / n_h[c] for c in strata}
    usable = [(i, v) for i, v in lab.items() if v in ("지지", "불지지")]

    def metrics(pv):
        out = {}
        for weighted in (False, True):
            acc = {m: [0.0, 0.0] for m in BARS}
            for iid, v in usable:
                k = key[iid]
                ww = w[k["cell"]] if weighted else 1.0
                pred, pos = pv[iid] > 0.5, v == "지지"
                if pred:
                    acc["precision"][1] += ww; acc["precision"][0] += ww * pos
                if pos:
                    acc["recall"][1] += ww; acc["recall"][0] += ww * pred
                    if k["lab_g1"] == "0":
                        acc["recovery"][1] += ww; acc["recovery"][0] += ww * pred
            for m in BARS:
                kk, nn = acc[m]
                if weighted:
                    out[m + "_weighted"] = kk / nn if nn else None
                else:
                    out[m] = {"k": int(kk), "n": int(nn), "rate": kk / nn if nn else None,
                              "wilson": wilson(int(kk), int(nn)),
                              "pass_point": (kk / nn > BARS[m]) if m == "recovery" else (kk / nn >= BARS[m])}
        out["verdict_registered"] = "통과" if all(out[m]["pass_point"] for m in BARS) else "탈락"
        return out

    res = {"n_population": len(pop), "n_with_duplicate_citations": len(changed),
           "n_labeled_with_duplicates": sum(c["labeled"] for c in changed),
           "n_flips_all": sum(c["flip"] for c in changed),
           "n_flips_labeled": sum(c["flip"] and c["labeled"] for c in changed),
           "replay_sanity_max_absdiff": sanity_diff,
           "registered_dup": metrics(p_old), "dedup": metrics(p_new), "changed": changed}
    io.open(V / "VPRIME_DEDUP_CHECK.json", "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "changed"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
