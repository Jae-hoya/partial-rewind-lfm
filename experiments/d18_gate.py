"""D18 5단계 — held-out 관문. 고정된 연산자(operator_dev.json)의 판정을 held-out 사람 라벨과 한 번 대조한다. 호출 0.

관문 (PREREG_D18_AMEND1): 일치율 ≥ 0.80, Cohen κ ≥ 0.60, κ 토픽 군집 95% CI 하한 ≥ 0.40,
음성 예측(A: 질문에 답 안 함, B: 답 없음)의 Hájek 가중 precision ≥ 0.80 이고 CI 하한 ≥ 0.70.
실패하면 그 연산자의 지표는 민감도로 내린다. 2순위 후보로 다시 열지 않는다.
"""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/d18"
N_BOOT = 2000


def kappa(y, p):
    y, p = np.asarray(y), np.asarray(p)
    po = (y == p).mean()
    pe = y.mean() * p.mean() + (1 - y.mean()) * (1 - p.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def stats(rows):
    y = np.array([r["y"] for r in rows]); p = np.array([r["p"] for r in rows])
    w = np.array([r["w"] for r in rows])
    neg = p == 0
    wp = float((w[neg] * (y[neg] == 0)).sum() / w[neg].sum()) if neg.any() else float("nan")
    pos = p == 1
    return {"agree": float((y == p).mean()), "kappa": kappa(y, p), "neg_pred_precision_w": wp,
            "pos_pred_precision_w": float((w[pos] * (y[pos] == 1)).sum() / w[pos].sum()) if pos.any() else float("nan"),
            "n_pred_neg": int(neg.sum()), "human_pos_rate": float(y.mean()), "pred_pos_rate": float(p.mean())}


def gate(rows):
    st = stats(rows)
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["topic"]].append(r)
    qs = sorted(by_q)
    rng = np.random.default_rng(42)
    ks, ws = [], []
    for _ in range(N_BOOT):
        s = [r for q in rng.choice(qs, size=len(qs), replace=True) for r in by_q[q]]
        b = stats(s)
        ks.append(b["kappa"]); ws.append(b["neg_pred_precision_w"])
    st["kappa_ci"] = [float(np.nanpercentile(ks, 2.5)), float(np.nanpercentile(ks, 97.5))]
    st["neg_pred_precision_w_ci"] = [float(np.nanpercentile(ws, 2.5)), float(np.nanpercentile(ws, 97.5))]
    checks = {"agree>=0.80": st["agree"] >= 0.80, "kappa>=0.60": st["kappa"] >= 0.60,
              "kappa_ci_lo>=0.40": st["kappa_ci"][0] >= 0.40,
              "negprec_w>=0.80": st["neg_pred_precision_w"] >= 0.80,
              "negprec_w_ci_lo>=0.70": st["neg_pred_precision_w_ci"][0] >= 0.70}
    st["checks"] = checks
    st["pass"] = all(checks.values())
    return st


def main() -> int:
    dev = json.load(io.open(D / "operator_dev.json", encoding="utf-8"))
    sc = {(r["query_id"], int(r["round"]), r["branch"], int(r["sq_index"])): r
          for r in csv.DictReader(io.open(D / "scores.csv", encoding="utf-8"))}
    labs = defaultdict(dict)
    for r in csv.DictReader(io.open(D / "labels_all_filled.csv", encoding="utf-8-sig")):
        labs[r["set"]][r["item_id"]] = r["human_label"]

    out = {"operators": {"A": dev["selected_A"], "B": dev["selected_B"]}}
    for name, kind in (("h2hoA", "A"), ("h2hoB", "B")):
        th = dev["selected_" + kind]["theta"]
        rows = []
        for k in csv.DictReader(io.open(D / ("_key_%s.csv" % name), encoding="utf-8")):
            s = sc[(k["query_id"], 0, k["branch"], int(k["sq_index"]))]
            score = float(s["s_A"] if kind == "A" else s["s_B"])
            h = labs[name][k["item_id"]]
            rows.append({"topic": k["query_id"], "w": float(k["N_h"]) / float(k["n_h"]), "p": int(score >= th),
                         "y": int(h == "답함") if kind == "A" else int(h != "답없음"),
                         "y_loose": int(h != "답안함") if kind == "A" else None,
                         "stratum": k["stratum"]})
        res = {"primary": gate(rows)}
        if kind == "A":
            res["sensitivity_partial_as_answer"] = gate([{**r, "y": r["y_loose"]} for r in rows])
        else:
            for st in ("empty", "answered"):
                sub = [r for r in rows if r["stratum"].startswith(st)]
                res["by_model_status_" + st] = stats(sub)
        out[kind] = res
    io.open(D / "gate.json", "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
