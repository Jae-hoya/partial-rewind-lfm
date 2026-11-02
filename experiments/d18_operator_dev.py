"""D18 3단계 — 연산자 A·B 후보를 dev 라벨로 채점하고 θ 를 고정한다. 호출 0, GPU. 규칙: PREREG_D18_AMEND1.md

dev 만 읽는다: A = H2-dev 43건, B = H1 86건 (둘 다 D16 15토픽, C3 조건).
held-out 라벨(H2-ho-A, H2-ho-B)은 이 스크립트가 열지 않는다. 산출물 operator_dev.json 을 커밋한 뒤에야 연다.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lfm.estimator import MiniCheckScorer  # noqa: E402
from lfm.relevance import NLITemplateScorer, RerankerScorer, strip_ids  # noqa: E402
from lfm.types import CITATION_RE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CANDS = [("bge-reranker-v2-m3", lambda: RerankerScorer("BAAI/bge-reranker-v2-m3", 1024)),
         ("bge-reranker-base", lambda: RerankerScorer("BAAI/bge-reranker-base", 512)),
         ("nli-deberta-v3-large", lambda: NLITemplateScorer("cross-encoder/nli-deberta-v3-large", 512))]


def snapshot(model: str) -> str:
    d = Path.home() / ".cache/huggingface/hub" / ("models--" + model.replace("/", "--")) / "snapshots"
    return ",".join(sorted(p.name for p in d.iterdir()))


def youden(y, s):
    """J = TPR - FPR 최대. 인접 정렬 점수의 중점들 가운데서, 동점이면 그 중앙값."""
    y, s = np.asarray(y), np.asarray(s)
    u = np.unique(s)
    cuts = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]])
    P, N = y.sum(), len(y) - y.sum()
    js = np.array([((s >= c) & (y == 1)).sum() / P - ((s >= c) & (y == 0)).sum() / N for c in cuts])
    best = cuts[js == js.max()]
    return float(np.median(best)), float(js.max())


def confusion(y, s, th):
    p = np.asarray(s) >= th
    y = np.asarray(y).astype(bool)
    return {"tp": int((p & y).sum()), "fp": int((p & ~y).sum()), "fn": int((~p & y).sum()), "tn": int((~p & ~y).sum())}


def precision_rules(y, s, kind):
    """민감도: SPEC_R2 원 규칙. A = precision(양성 예측) >= 0.85 인 가장 낮은 θ, B = precision(음성 예측) >= 0.90 인 가장 높은 θ."""
    y, s = np.asarray(y), np.asarray(s)
    cuts = np.unique(s)
    ok = []
    for c in cuts:
        p = s >= c
        if kind == "A" and p.sum() and (y[p] == 1).mean() >= 0.85:
            ok.append(c)
        if kind == "B" and (~p).sum() and (y[~p] == 0).mean() >= 0.90:
            ok.append(c)
    if not ok:
        return None
    return float(min(ok) if kind == "A" else max(ok))


def main() -> int:
    d16 = {}
    for line in io.open(ROOT / "results/d16/d16_subq.jsonl", encoding="utf-8"):
        x = json.loads(line)
        if x["condition"] == "C3_partial_heldout":
            d16[(x["query_id"], x["branch"], int(x["sq_index"]))] = x
    labs = {r["item_id"]: r["human_label"] for r in csv.DictReader(io.open(ROOT / "results/d18/labels_all_filled.csv", encoding="utf-8-sig")) if r["set"] == "h2dev"}
    A = []
    for k in csv.DictReader(io.open(ROOT / "results/d18/_key_h2dev.csv", encoding="utf-8")):
        x = d16[(k["query_id"], k["branch"], int(k["sq_index"]))]
        cited = [x["ctx_texts"][c] for c in dict.fromkeys(CITATION_RE.findall(x["answer"])) if c in x["ctx_texts"]]
        A.append({"topic": k["query_id"], "q": x["sub_question"], "a": strip_ids(x["answer"]),
                  "cited_docs": cited, "label": labs[k["item_id"]]})
    h1 = {r["item_id"]: r["human"] for r in csv.DictReader(io.open(ROOT / "results/d16/h1_scored.csv", encoding="utf-8"))}
    B = []
    for k in csv.DictReader(io.open(ROOT / "results/d16/_key_d16.csv", encoding="utf-8")):
        x = d16[(k["query_id"], k["branch"], int(k["sq_index"]))]
        B.append({"topic": k["query_id"], "q": x["sub_question"],
                  "chunks": [x["ctx_texts"][c] for c in x["own_in_ctx"]], "label": h1[k["item_id"]]})
    yA = [int(r["label"] == "답함") for r in A]
    yA_loose = [int(r["label"] != "답안함") for r in A]
    yB = [int(r["label"] != "답없음") for r in B]
    print("dev A %d (양성 %d) · B %d (양성 %d)" % (len(A), sum(yA), len(B), sum(yB)), flush=True)

    out = {"dev_sizes": {"A": len(A), "A_pos": sum(yA), "B": len(B), "B_pos": sum(yB)}, "candidates": {}}
    for name, make in CANDS:
        sc = make()
        sA = sc.score([(r["q"], r["a"]) for r in A])
        flat = [(i, c) for i, r in enumerate(B) for c in r["chunks"]]
        raw = sc.score([(B[i]["q"], c) for i, c in flat])
        sB = [max([v for (i, _), v in zip(flat, raw) if i == j], default=0.0) for j in range(len(B))]
        res = {"revision": snapshot(sc.name)}
        for kind, y, s, rows in (("A", yA, sA, A), ("B", yB, sB, B)):
            th, j = youden(y, s)
            topics = sorted({r["topic"] for r in rows})
            loto = [youden([y[i] for i, r in enumerate(rows) if r["topic"] != t],
                           [s[i] for i, r in enumerate(rows) if r["topic"] != t])[0] for t in topics]
            res[kind] = {"auroc": float(roc_auc_score(y, s)), "theta": th, "youden_j": j,
                         "confusion": confusion(y, s, th), "theta_precision_rule": precision_rules(y, s, kind),
                         "loto_theta": {"median": float(np.median(loto)), "min": float(min(loto)), "max": float(max(loto))},
                         "scores": [round(v, 6) for v in s]}
        res["A"]["auroc_loose"] = float(roc_auc_score(yA_loose, sA))
        out["candidates"][name] = res
        print("%-22s A AUROC %.3f θ %.4f · B AUROC %.3f θ %.4f"
              % (name, res["A"]["auroc"], res["A"]["theta"], res["B"]["auroc"], res["B"]["theta"]), flush=True)
        del sc
        import torch
        torch.cuda.empty_cache()

    mc = MiniCheckScorer()
    sM = [mc.score("\n\n".join(r["cited_docs"]), r["a"]) if r["cited_docs"] else 0.0 for r in A]
    out["negative_control_minicheck_A"] = {"auroc": float(roc_auc_score(yA, sM)),
                                           "n_with_cited_text": sum(1 for r in A if r["cited_docs"]),
                                           "scores": [round(v, 6) for v in sM]}
    print("MiniCheck(음성 대조) A AUROC %.3f" % out["negative_control_minicheck_A"]["auroc"], flush=True)

    for kind in ("A", "B"):
        best = max(out["candidates"], key=lambda n: out["candidates"][n][kind]["auroc"])
        out["selected_" + kind] = {"operator": best, "theta": out["candidates"][best][kind]["theta"],
                                   "auroc": out["candidates"][best][kind]["auroc"]}
    io.open(ROOT / "results/d18/operator_dev.json", "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k.startswith("selected")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
