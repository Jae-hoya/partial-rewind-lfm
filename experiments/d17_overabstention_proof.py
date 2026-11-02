"""D17 — 과잉 기권의 존재 증명. LLM 호출 0, GPU 는 MiniCheck 채점만. 사전등록: results/d17/PREREG_D17.md

C3 에서 정답 청크가 실렸는데 비운 sub-question 이, 같은 근거로 답할 수 있었는지 사람 없이 보인다.
같은 질문이 C4_both 에서 답해졌고, 그 답 문장이 C3 모델이 본 바로 그 정답 청크 본문에
entail 되면(MiniCheck 단일 문서 > 0.5) "답할 수 있었다" 가 증명된다. 안 되면 미결이다.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from d16_abstention_cause import SWEEP, cluster_ci, replay  # noqa: E402
from diag_sweep import pick_topics  # noqa: E402
from lfm.estimator import MiniCheckScorer  # noqa: E402
from lfm.types import CITATION_RE, split_sentences  # noqa: E402

TAU = 0.5
C4 = "C4_both"


def strip_cites(s: str) -> str:
    return re.sub(r"\s+", " ", CITATION_RE.sub("", s)).strip()


def body(ctx_text: str) -> str:
    # d16 이 저장한 형식: "(title > section) " + 실린 본문. 프롬프트에도 이 머리말이 붙어 있었다.
    return ctx_text


def candidates(answer: str, gold_ids: set[str]) -> list[tuple[str, str]]:
    """(문장, 인용한 정답 청크 id) — 문장이 직접 단 표기만 본다."""
    out = []
    for sent in split_sentences(answer):
        for cid in CITATION_RE.findall(sent):
            if cid in gold_ids:
                out.append((strip_cites(sent), cid))
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with io.open(root / "data" / "topics_pilot50.yaml", encoding="utf-8") as fh:
        picked = pick_topics(yaml.safe_load(fh)["topics"], 15, 42)
    with io.open(root / "data" / "heldout_ids.json", encoding="utf-8") as fh:
        held = json.load(fh)["by_topic"]
    gold = {q: set(int(x) for x in v["heldout_identifiers"]) for q, v in held.items()}

    c3 = {}
    for line in io.open(root / "results/d16/d16_subq.jsonl", encoding="utf-8"):
        x = json.loads(line)
        if x["condition"] == "C3_partial_heldout" and x["gold_in_ctx"]:
            c3[(x["query_id"], x["branch"], x["sq_index"])] = x

    c4_rows, c4_deg, c4_stats = replay(C4, root, picked, gold)
    with io.open(root / SWEEP / C4 / "branches.csv", encoding="utf-8") as fh:
        ref = {(r["query_id"], r["branch"]): int(r["degraded"]) for r in csv.DictReader(fh)}
    mismatch = [list(k) for k, v in c4_deg.items() if ref.get(k) != v]
    c4 = {(x["query_id"], x["branch"], x["sq_index"]): x for x in c4_rows}

    print("MiniCheck 적재 ...", flush=True)
    scorer = MiniCheckScorer()

    def best(pairs: list[tuple[str, str]], ctx: dict[str, str]):
        scored = [(scorer.score(body(ctx[cid]), s), s, cid) for s, cid in pairs if s]
        return max(scored, default=None)

    items, unpaired = [], 0
    for k, x in sorted(c3.items()):
        y = c4.get(k)
        if y is None or y["sub_question"] != x["sub_question"]:
            unpaired += 1
            continue
        gids = set(x["gold_in_ctx"])
        via_c4 = best(candidates(y["answer"], gids), x["ctx_texts"])
        own = best(candidates(x["answer"], gids), x["ctx_texts"]) if not x["empty"] else None
        items.append({
            "query_id": k[0], "branch": k[1], "sq_index": k[2], "c3_empty": x["empty"],
            "sub_question": x["sub_question"], "c4_empty": y["empty"],
            "c4_cites_c3_gold": int(bool(candidates(y["answer"], gids))),
            "p_via_c4": None if via_c4 is None else round(float(via_c4[0]), 4),
            "proved_via_c4": int(via_c4 is not None and via_c4[0] > TAU),
            "proof_sentence": "" if via_c4 is None else via_c4[1],
            "p_own": None if own is None else round(float(own[0]), 4),
            "proved_own": int(own is not None and own[0] > TAU),
        })
        print("  %s %s %d" % k, flush=True)

    tgt = [i for i in items if i["c3_empty"]]
    ctl = [i for i in items if not i["c3_empty"]]
    q1 = cluster_ci([(i["query_id"], i["proved_via_c4"]) for i in tgt])
    q2 = cluster_ci([(i["query_id"], i["proved_via_c4"]) for i in ctl])
    q3 = cluster_ci([(i["query_id"], i["proved_own"]) for i in ctl])
    ratio = (q1["rate"] / q2["rate"]) if q1["rate"] is not None and q2["rate"] else None
    if q1["rate"] is None:
        verdict = "판정 불가"
    elif q1["rate"] >= 1 / 3:
        verdict = "과잉 기권 상당 (Q1 >= 1/3)"
    elif ratio is not None and ratio >= 0.7:
        verdict = "검출력 한계 — 과잉 기권 쪽 (Q1/Q2 >= 0.7)"
    else:
        verdict = "과잉 기권 증거 약함 (답할 수 없는 질문 쪽, 증명 아님)"

    summary = {
        "llm_stats": c4_stats, "c4_degraded_mismatch": mismatch, "n_c4_branches": len(c4_deg),
        "n_items": len(items), "n_unpaired": unpaired, "n_target": len(tgt), "n_control": len(ctl),
        "target_c4_answered": sum(1 - i["c4_empty"] for i in tgt),
        "target_c4_cites_gold": sum(i["c4_cites_c3_gold"] for i in tgt),
        "Q1_target_proved_via_c4": q1, "Q2_control_proved_via_c4": q2,
        "Q3_control_proved_own": q3, "Q1_over_Q2": ratio, "verdict": verdict,
    }
    outdir = root / "results/d17"
    with io.open(outdir / "d17_items.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(items[0].keys()))
        w.writeheader()
        w.writerows(items)
    with io.open(outdir / "d17_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 2 if mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
