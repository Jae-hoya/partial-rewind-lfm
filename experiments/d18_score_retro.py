"""D18 4단계 — Test A 2520 sub-question 전량을 고정된 연산자 A·B 로 채점하고 새 지표로 소급 재집계한다. 호출 0, GPU.

연산자와 θ 는 operator_dev.json(커밋 b36fd48)에서 읽는다. 여기서 바꾸지 않는다.
grounding 은 재정의하지 않는다 — Test A(g1)와 M1′(MiniCheck) 의 waste_ratio.csv 값을 그대로 쓴다.
mean_g 의 집계 대상도 바꾸지 않는다(등록 1항). 이것은 cached outputs 의 retrospective re-scoring 이다.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from d15_abstention_diag import MIN_BODY  # noqa: E402
from diag_sweep import pick_topics  # noqa: E402
from lfm.relevance import RerankerScorer, strip_ids  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/d18"
MODELS = {"bge-reranker-base": ("BAAI/bge-reranker-base", 512), "bge-reranker-v2-m3": ("BAAI/bge-reranker-v2-m3", 1024)}
GROUNDING = {"g1 (Test A)": "results/test_a/20260829-041445/waste_ratio.csv",
             "MiniCheck (M1′)": "results/m1prime/seed42/waste_ratio.csv"}
TAUS = (0.7, 0.8, 0.9)
K_MIN = 2


def main() -> int:
    dev = json.load(io.open(D / "operator_dev.json", encoding="utf-8"))
    selA, selB = dev["selected_A"], dev["selected_B"]
    rows = [json.loads(l) for l in io.open(D / "subq_text.jsonl", encoding="utf-8")]
    chunk_txt = {}
    for l in io.open(D / "chunks.jsonl", encoding="utf-8"):
        x = json.loads(l)
        chunk_txt[x["chunk_id"]] = x["text"]
    lmax = defaultdict(int)
    for r in rows:
        for cid, n in r["own_included"].items():
            lmax[cid] = max(lmax[cid], n)

    def shown(cid, n):
        t = chunk_txt[cid]
        return t[: len(t) - (lmax[cid] - n)]

    # --- 채점 (선택된 연산자만; 두 연산자가 같은 모델이면 한 번만 올린다) -----------------
    scorers = {}
    for sel in (selA, selB):
        if sel["operator"] not in scorers:
            scorers[sel["operator"]] = RerankerScorer(*MODELS[sel["operator"]])
    sa = scorers[selA["operator"]].score([(r["sub_question"], strip_ids(r["answer"])) if r["cited"] else (r["sub_question"], "")
                                          for r in rows])
    flat = [(i, cid) for i, r in enumerate(rows) for cid in r["own"] if r["own_included"].get(cid, 0) >= MIN_BODY]
    raw = scorers[selB["operator"]].score([(rows[i]["sub_question"], shown(cid, rows[i]["own_included"][cid])) for i, cid in flat])
    sb = [0.0] * len(rows)
    for (i, _), v in zip(flat, raw):
        sb[i] = max(sb[i], v)
    for r, a, b in zip(rows, sa, sb):
        r["s_A"] = a if r["cited"] else None
        r["s_B"] = b
        r["relevant"] = int(bool(r["cited"]) and a >= selA["theta"])
        r["answerable"] = int(b >= selB["theta"])
    with io.open(D / "scores.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "round", "branch", "sq_index", "empty", "cited", "s_A", "s_B", "relevant", "answerable"])
        for r in rows:
            w.writerow([r["query_id"], r["round"], r["branch"], r["sq_index"], r["empty"], r["cited"],
                        "" if r["s_A"] is None else round(r["s_A"], 6), round(r["s_B"], 6), r["relevant"], r["answerable"]])

    # --- branch 단위 지표 ---------------------------------------------------------------
    br = defaultdict(list)
    for r in rows:
        br[(r["query_id"], r["round"], r["branch"])].append(r)
    cov = {}
    for k, subs in br.items():
        n = len(subs)
        U = [s for s in subs if s["answerable"]]
        cov[k] = {"L": sum(s["cited"] for s in subs) / n, "A": sum(s["relevant"] for s in subs) / n,
                  "U": (sum(s["relevant"] for s in U) / len(U)) if len(U) >= K_MIN else 0.0,
                  "U_k1": (sum(s["relevant"] for s in U) / len(U)) if len(U) >= 1 else 0.0,
                  "n_ans": len(U), "n": n}

    with io.open(ROOT / "data/topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    devset = {str(t["id"]) for t in pick_topics(topics, 15, 42)}
    scopes = {"held_out_35": lambda q: q not in devset, "dev_15": lambda q: q in devset, "all_50": lambda q: True}

    out = {"operators": {"A": selA, "B": selB}, "k_min": K_MIN, "scopes": {}}
    for gname, gpath in GROUNDING.items():
        wr = list(csv.DictReader(io.open(ROOT / gpath, encoding="utf-8")))
        for x in wr:
            c = cov[(x["query_id"], int(x["round"]), x["branch"])]
            assert abs(c["L"] - float(x["coverage"])) < 1e-6, "legacy coverage 재현 실패"
        for sname, inscope in scopes.items():
            sel = [x for x in wr if inscope(x["query_id"])]
            res = {}
            for key in ("L", "A", "U"):
                qr = defaultdict(list)
                for x in sel:
                    qr[(x["query_id"], x["round"])].append((float(x["grounding"]), cov[(x["query_id"], int(x["round"]), x["branch"])][key]))
                pass_g = sum(np.mean([g for g, _ in v]) >= 0.8 for v in qr.values())
                pass_c = sum(min(c for _, c in v) >= 0.8 for v in qr.values())
                both = sum(np.mean([g for g, _ in v]) >= 0.8 and min(c for _, c in v) >= 0.8 for v in qr.values())
                covs = [cov[(x["query_id"], int(x["round"]), x["branch"])][key] for x in sel]
                W = {}
                for t in TAUS:
                    re_rows = [x for x in sel if x["re_executed"] == "1"]
                    num = sum(float(x["grounding"]) > t and cov[(x["query_id"], int(x["round"]), x["branch"])][key] > t for x in re_rows)
                    W[str(t)] = {"num": int(num), "den": len(re_rows), "W": num / len(re_rows) if re_rows else None}
                res[key] = {"query_rounds": len(qr), "pass_grounding": int(pass_g), "pass_coverage": int(pass_c),
                            "pass_both": int(both), "coverage_mean": float(np.mean(covs)),
                            "coverage_median": float(np.median(covs)), "coverage_eq1": int(sum(c == 1.0 for c in covs)),
                            "W": W}
            out["scopes"].setdefault(sname, {})[gname] = res

    def acc(rs):
        cited = [r for r in rs if r["cited"]]
        emp = [r for r in rs if r["empty"]]
        ans = [r for r in rs if r["answerable"]]
        una = [r for r in rs if not r["answerable"]]
        return {"n": len(rs),
                "off_question": sum(1 - r["relevant"] for r in cited) / len(cited) if cited else None,
                "over_abstain": sum(r["empty"] for r in ans) / len(ans) if ans else None,
                "abstention_precision": sum(1 - r["answerable"] for r in emp) / len(emp) if emp else None,
                "answerable_rate": len(ans) / len(rs) if rs else None,
                "answered_unanswerable": sum(r["cited"] for r in una) / len(una) if una else None}

    out["accessory"] = {}
    for sname, inscope in scopes.items():
        rs = [r for r in rows if inscope(r["query_id"])]
        out["accessory"][sname] = {"all": acc(rs), **{b: acc([r for r in rs if r["branch"] == b])
                                                      for b in sorted({r["branch"] for r in rs})}}
    out["degenerate_branch_rounds"] = int(sum(1 for c in cov.values() if c["n_ans"] < K_MIN))
    io.open(D / "D18_retro.json", "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"held_out_35": out["scopes"]["held_out_35"], "accessory_held_out": out["accessory"]["held_out_35"]["all"],
                      "degenerate": out["degenerate_branch_rounds"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
