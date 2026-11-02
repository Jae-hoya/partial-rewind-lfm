"""V′ 2단계 라벨 시트 (+40). SPEC_M1Vprime §1 "2단계 +40건 ... 같은 비율(P1 10 / P2 17 / N1 10 / N2 3)". 생성 호출 0회.

1단계(vprime_make_sheet.py)와 같은 모집단·같은 칸 정의(원 MiniCheck d9 Scorer × g1)로 뽑는다.
칸 정의를 단일 문서 MiniCheck 로 바꾸면 사전등록한 층화 설계가 바뀌므로 그대로 둔다.
재현 검사: 1단계 60건의 칸과 점수를 다시 계산해 `_key_stage1.csv` 와 맞춰 본다. 어긋나면 시트를 만들지 않는다.
리뷰어 R1-3 이 "prescribed second labelling stage that was not completed" 로 지적한 바로 그 단계다.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from d15_abstention_diag import guard_transport  # noqa: E402
from diag_sweep import CONDITIONS, pick_topics  # noqa: E402
from make_label_tool import HTML as VPRIME_HTML  # noqa: E402
from vprime_make_sheet import RULES  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

STAGE2 = [("P1", True, True, 10), ("P2", True, False, 17), ("N1", False, False, 10), ("N2", False, True, 3)]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out = root / "results/vprime"
    cfg = load_config(root / "config.yaml")
    spec = CONDITIONS["C3_partial_heldout"]
    data = cfg.as_dict()
    if spec["top_k"] is not None:
        data["retrieval"]["top_k"] = spec["top_k"]
    data["paths"]["index_dir"] = data["paths"]["index_dir_partial"]
    ccfg = Config(data, source="C3_partial_heldout")

    nli = load_nli(ccfg)
    llm = LLM(ccfg, transport=guard_transport)
    retriever = BM25Retriever(ccfg)
    with io.open(root / "data/topics_pilot50.yaml", encoding="utf-8") as fh:
        picked = pick_topics(yaml.safe_load(fh)["topics"], 15, int(cfg["seed"]))

    pop = []
    for t in picked:
        anchor = " ".join(t.get("terms") or [])
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            for sa in o.sub_answers:
                cited_ids = [c for c in sa.citations if c in cmap]
                if not cited_ids:
                    continue
                cited = [cmap[c].text for c in cited_ids]
                for sent in split_sentences(sa.text):
                    pop.append({"item_id": "v%03d" % len(pop), "topic": str(t["id"]),
                                "branch": p.role, "sentence": sent, "premises": cited})
    print("모집단 %d 문장 · LLM api_calls %d" % (len(pop), llm.stats.api_calls), flush=True)

    for r in pop:
        r["p_g1"] = max(nli.entailment([(c, r["sentence"]) for c in r["premises"]]))
        r["lab_g1"] = r["p_g1"] >= 0.5
    del nli
    from d9_minicheck import Scorer           # 1단계와 같은 (쪼개기 버그가 있는) 원 연산자
    sc = Scorer()
    for r in pop:
        r["p_mc"] = sc.score("\n\n".join(r["premises"]), r["sentence"])
        r["lab_mc"] = r["p_mc"] > 0.5

    # 1단계 재현 검사
    key1 = {k["item_id"]: k for k in csv.DictReader(io.open(out / "_key_stage1.csv", encoding="utf-8"))}
    byid = {r["item_id"]: r for r in pop}
    cellof = {(mc, g1): name for name, mc, g1, _ in STAGE2}
    bad = []
    for iid, k in key1.items():
        r = byid.get(iid)
        if r is None or cellof[(r["lab_mc"], r["lab_g1"])] != k["cell"] \
                or abs(r["p_g1"] - float(k["p_g1"])) > 1e-3 or abs(r["p_mc"] - float(k["p_mc"])) > 1e-3:
            bad.append(iid)
    print("1단계 재현: %d/60 일치" % (60 - len(bad)), flush=True)
    if bad:
        print("불일치:", bad)
        return 2

    rng = np.random.default_rng(int(cfg["seed"]) + 1)   # 1단계(seed 42)와 다른 흐름
    sample, strata = [], {}
    for name, mc, g1, want in STAGE2:
        pool = [r for r in pop if r["lab_mc"] == mc and r["lab_g1"] == g1 and r["item_id"] not in key1]
        n_all = sum(1 for r in pop if r["lab_mc"] == mc and r["lab_g1"] == g1)
        take = min(want, len(pool))
        idx = sorted(rng.choice(len(pool), size=take, replace=False).tolist()) if pool else []
        strata[name] = {"N_h": n_all, "n_stage1": sum(1 for k in key1.values() if k["cell"] == name),
                        "remaining": len(pool), "n_stage2": take}
        for i in idx:
            sample.append({**pool[i], "cell": name})
    sample = [sample[i] for i in rng.permutation(len(sample)).tolist()]

    items = [{"id": r["item_id"], "rules": RULES, "claim": r["sentence"], "ev": "\n\n".join(r["premises"])}
             for r in sample]
    html = VPRIME_HTML
    for a, b in (("V′ 라벨 — 60건", "V′ 2단계 — %d건" % len(items)), ("60건 완료", "%d건 완료" % len(items)),
                 ("vprime_stage1_v1", "vprime_stage2_v1"), ("labels_stage1_filled.csv", "labels_stage2_filled.csv")):
        assert a in html, a
        html = html.replace(a, b)
    io.open(out / "labeler_stage2.html", "w", encoding="utf-8").write(
        html.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)))
    with io.open(out / "_key_stage2.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "cell", "topic", "branch", "lab_g1", "lab_mc", "p_g1", "p_mc"])
        w.writeheader()
        for r in sample:
            w.writerow({"item_id": r["item_id"], "cell": r["cell"], "topic": r["topic"], "branch": r["branch"],
                        "lab_g1": int(r["lab_g1"]), "lab_mc": int(r["lab_mc"]),
                        "p_g1": round(r["p_g1"], 6), "p_mc": round(r["p_mc"], 6)})
    with io.open(out / "stage2_strata.json", "w", encoding="utf-8") as fh:
        json.dump({"population": len(pop), "strata": strata}, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"n": len(items), "strata": strata}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
