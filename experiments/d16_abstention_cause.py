"""D16 — 기권의 원인: 근거 부족인가, 근거가 있는데도 기권인가. **호출 0회.** 사전등록: results/d16/PREREG_D16.md

진단 스윕의 C1(현행 인덱스)과 C3(target article 본문을 남긴 인덱스)을 캐시만으로 다시 돌려
같은 sub-question 끼리 짝짓는다. 정답 문서를 넣었을 때 빈 답이 채워지면 그 기권은 근거 부족
때문이고, 정답 청크가 컨텍스트에 있는데도 비면 프롬프트가 허용한 과잉 기권이다.

각 조건은 스윕이 저장한 config 사본으로 재구성한다 — 캐시 키에 프롬프트가 들어가므로
현행 config 와 조금만 달라도 미스가 난다. 미스는 d15 의 가드가 예외로 막는다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import lfm.branch as branch_mod  # noqa: E402
from d15_abstention_diag import MIN_BODY, N_BOOT, Recorder, guard_transport  # noqa: E402
from diag_sweep import pick_topics  # noqa: E402
from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config  # noqa: E402
from lfm.llm import LLM  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402

SWEEP = "results/diag/sweep"
CONDS = ("C1_baseline", "C3_partial_heldout")
DOC_RE = re.compile(r"^d(\d+)s")


def doc_id(chunk_id: str) -> int:
    m = DOC_RE.match(chunk_id)
    return int(m.group(1)) if m else -1


def replay(name: str, root: Path, topics: list[dict], gold: dict[str, set[int]]):
    with io.open(root / SWEEP / name / "config.yaml", encoding="utf-8") as fh:
        cfg = Config(yaml.safe_load(fh), source=name)
    llm = LLM(cfg, transport=guard_transport)
    rec = Recorder(BM25Retriever(cfg))
    texts: dict[str, str] = {}

    original = branch_mod.format_context

    def recording_format_context(chunks, mc):
        text = original(chunks, mc)
        parts, used = [], 0
        rec.n_chunks = len(chunks)
        for c in chunks:
            body_full = c.text.strip().replace("\n", " ")
            head = "[" + c.chunk_id + "] (" + c.title + " > " + c.section_path + ") "
            room = mc - used
            rec.full_len[c.chunk_id] = len(body_full)
            if room <= len(head) + 80:
                break
            body = body_full[: max(0, room - len(head))]
            parts.append(head + body)
            used += len(head) + len(body)
            rec.included[c.chunk_id] = len(body)
            texts[c.chunk_id] = "(" + c.title + " > " + c.section_path + ") " + body
        assert "\n".join(parts) == text, "컨텍스트 재구성이 원본과 다르다"
        return text

    branch_mod.format_context = recording_format_context
    rows, degraded = [], {}
    try:
        for t in topics:
            qid = str(t["id"])
            anchor = " ".join(t.get("terms") or [])
            g = gold[qid]
            for p in decompose(t["topic"], llm, cfg):
                rec.reset()
                out = run_branch(p, rec, llm, cfg, round_=0, search_anchor=anchor)
                degraded[(qid, p.role)] = int(out.degraded)
                assert len(rec.searches) == len(out.sub_answers)
                for sa, own in zip(out.sub_answers, rec.searches):
                    in_ctx = [cid for cid in own if rec.included.get(cid, 0) >= MIN_BODY]
                    gold_ctx = [cid for cid in in_ctx if doc_id(cid) in g]
                    rows.append({
                        "condition": name, "query_id": qid, "branch": p.role,
                        "sq_index": sa.index, "sub_question": sa.sub_question,
                        "empty": int(not sa.text.strip()), "answer": sa.text,
                        "citations": list(sa.citations),
                        "cites_gold": int(any(doc_id(c) in g for c in sa.citations)),
                        "own_in_ctx": in_ctx, "gold_in_ctx": gold_ctx,
                        "ctx_texts": {cid: texts[cid] for cid in in_ctx},
                    })
            print("  %s %s" % (name, qid), flush=True)
    finally:
        branch_mod.format_context = original
    return rows, degraded, llm.stats.to_dict()


def cluster_ci(items: list[tuple[str, int]], seed: int = 42):
    """(토픽, 0/1) 목록의 비율과 토픽 군집 부트스트랩 95% CI."""
    if not items:
        return {"n": 0, "rate": None, "ci": None}
    by_q: dict[str, list[int]] = {}
    for q, v in items:
        by_q.setdefault(q, []).append(v)
    qs = sorted(by_q)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT):
        vals = [v for q in rng.choice(qs, size=len(qs), replace=True) for v in by_q[q]]
        boots.append(np.mean(vals))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(items), "k": int(sum(v for _, v in items)),
            "rate": float(np.mean([v for _, v in items])), "ci": [float(lo), float(hi)]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="D16 기권 원인 (호출 0회)")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--outdir", default="results/d16")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    with io.open(root / "data" / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, 42)
    with io.open(root / "data" / "heldout_ids.json", encoding="utf-8") as fh:
        held = json.load(fh)["by_topic"]
    gold = {q: set(int(x) for x in v["heldout_identifiers"]) for q, v in held.items()}

    rows, checks, stats = [], {}, {}
    for name in CONDS:
        r, deg, st = replay(name, root, picked, gold)
        rows += r
        stats[name] = st
        with io.open(root / SWEEP / name / "branches.csv", encoding="utf-8") as fh:
            ref = {(x["query_id"], x["branch"]): int(x["degraded"]) for x in csv.DictReader(fh)}
        checks[name] = {"n": len(deg), "mismatch": [list(k) for k, v in deg.items()
                                                     if ref.get(k) != v]}

    # C1 은 Test A 라운드 0 과 같은 실행이다 — D15 재생값과도 대조한다.
    with io.open(root / "results/d15/d15_subq.csv", encoding="utf-8") as fh:
        d15 = {(x["query_id"], x["branch"], int(x["sq_index"])): int(x["empty"])
               for x in csv.DictReader(fh) if x["round"] == "0"}
    c1 = {(x["query_id"], x["branch"], x["sq_index"]): x for x in rows if x["condition"] == CONDS[0]}
    c3 = {(x["query_id"], x["branch"], x["sq_index"]): x for x in rows if x["condition"] == CONDS[1]}
    checks["c1_vs_d15_mismatch"] = sum(1 for k, x in c1.items() if d15.get(k) != x["empty"])

    pairs = [(k, c1[k], c3[k]) for k in c1 if k in c3 and c1[k]["sub_question"] == c3[k]["sub_question"]]
    n_unpaired = len(c1) + len(c3) - 2 * len(pairs)

    p1 = [(k[0], int((not b["empty"]) and b["cites_gold"])) for k, a, b in pairs if a["empty"]]
    p1_any = [(k[0], int(not b["empty"])) for k, a, b in pairs if a["empty"]]
    p2 = [(x["query_id"], x["empty"]) for x in c3.values() if x["gold_in_ctx"]]
    reverse = sum(1 for _, a, b in pairs if (not a["empty"]) and b["empty"])

    def by(key, cond_rows):
        out = {}
        for x in cond_rows:
            out.setdefault(str(x[key]), []).append(x["empty"])
        return {k: {"n": len(v), "empty": float(np.mean(v))} for k, v in sorted(out.items())}

    summary = {
        "topics": [str(t["id"]) for t in picked], "llm_stats": stats, "checks": checks,
        "n_pairs": len(pairs), "n_unpaired": n_unpaired,
        "P1_rescued_with_gold_citation": cluster_ci(p1),
        "P1b_rescued_any": cluster_ci(p1_any),
        "P2_residual_abstention_gold_in_ctx": cluster_ci(p2),
        "reverse_answered_c1_empty_c3": reverse,
        "empty_rate": {n: cluster_ci([(x["query_id"], x["empty"]) for x in rows if x["condition"] == n])
                       for n in CONDS},
        "c3_gold_in_ctx_rate": cluster_ci([(x["query_id"], int(bool(x["gold_in_ctx"])))
                                           for x in c3.values()]),
        "by_branch": {n: by("branch", [x for x in rows if x["condition"] == n]) for n in CONDS},
        "by_position": {n: by("sq_index", [x for x in rows if x["condition"] == n]) for n in CONDS},
    }
    p1v, p2v = summary["P1_rescued_with_gold_citation"]["rate"], summary["P2_residual_abstention_gold_in_ctx"]["rate"]
    summary["verdict"] = {
        "P1": None if p1v is None else ("근거 부족이 주원인" if p1v >= 0.5 else "근거 부족만으로 설명 안 됨"),
        "P2": None if p2v is None else ("과잉 기권 있음" if p2v >= 0.2 else "과잉 기권 작음"),
    }

    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    with io.open(outdir / "d16_subq.jsonl", "w", encoding="utf-8") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    with io.open(outdir / "d16_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_branch", "by_position")},
                     ensure_ascii=False, indent=1))
    bad = any(c["mismatch"] for c in checks.values() if isinstance(c, dict)) or checks["c1_vs_d15_mismatch"]
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
