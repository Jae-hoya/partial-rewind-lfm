"""H2 사람 라벨만으로 본 결과 (연산자 채점 전). 호출 0.

연산자 A·B 의 dev 선택과 held-out 관문은 SPEC_R2 다음 단계다. 여기서는 사람 라벨이 직접 말하는 것만 집계한다:
  held-out-A  모델이 답한 sub-answer 중 질문에 답한 비율 (= 1 - off_question)
  held-out-B  답 가능성 — 모델이 비운 것 / 답한 것 각각
  dev         H1(컨텍스트 답 가능성)과 H2-dev(답-질문 관련성)의 교차표
층 가중은 Hájek(w = N_h / n_h), CI 는 토픽 군집 부트스트랩(비가중)이다.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from d16_abstention_cause import cluster_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/d18"


def load(name):
    lab = {r["item_id"]: r["human_label"] for r in csv.DictReader(io.open(D / "labels_all_filled.csv", encoding="utf-8-sig"))
           if r["set"] == name}
    key = list(csv.DictReader(io.open(D / ("_key_%s.csv" % name), encoding="utf-8")))
    assert {k["item_id"] for k in key} == set(lab)
    return [{**k, "human": lab[k["item_id"]]} for k in key]


def hajek(rows, hit):
    num = sum(float(r["N_h"]) / float(r["n_h"]) * hit(r) for r in rows)
    den = sum(float(r["N_h"]) / float(r["n_h"]) for r in rows)
    return num / den if den else None


def summ(rows, hit):
    return {**cluster_ci([(r["query_id"], int(hit(r))) for r in rows]), "weighted": hajek(rows, hit)}


def main() -> int:
    out = {}
    a = load("h2hoA")
    out["held_out_A"] = {
        "n": len(a), "counts": dict(Counter(r["human"] for r in a)),
        "answers_question_strict": summ(a, lambda r: r["human"] == "답함"),
        "answers_question_loose": summ(a, lambda r: r["human"] != "답안함"),
        "off_question_strict": summ(a, lambda r: r["human"] == "답안함"),
        "by_branch": {b: dict(Counter(r["human"] for r in a if r["branch"] == b))
                      for b in sorted({r["branch"] for r in a})},
    }
    b = load("h2hoB")
    out["held_out_B"] = {}
    for st, name in (("1", "model_empty"), ("0", "model_answered")):
        g = [r for r in b if r["model_empty"] == st]
        out["held_out_B"][name] = {
            "n": len(g), "counts": dict(Counter(r["human"] for r in g)),
            "answerable_strict": summ(g, lambda r: r["human"] == "답있음"),
            "answerable_loose": summ(g, lambda r: r["human"] != "답없음"),
            "by_branch": {x: dict(Counter(r["human"] for r in g if r["branch"] == x))
                          for x in sorted({r["branch"] for r in g})},
        }
    dev = load("h2dev")
    h1 = {r["item_id"]: r["human"] for r in csv.DictReader(io.open(ROOT / "results/d16/h1_scored.csv", encoding="utf-8"))}
    cross = Counter((h1[r["h1_item"]], r["human"]) for r in dev)
    out["dev"] = {"n": len(dev), "counts": dict(Counter(r["human"] for r in dev)),
                  "cross_H1context_x_H2answer": {"%s|%s" % k: v for k, v in sorted(cross.items())}}
    io.open(D / "H2_human_summary.json", "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
