"""D15 — Test A 빈 답안(215/600)의 원인 진단. **호출 0회.** 사전등록: results/d15/PREREG_D15.md

묻는 것: 빈 sub-answer 는 (A) 그 sub-question 의 근거 청크가 12000자 컨텍스트 상한에 잘려
프롬프트에 못 들어갔기 때문인가, (B) 근거가 들어갔는데도 모델이 기권했기 때문인가.
ANSWER 프롬프트는 "or leave the text empty" 로 빈 답을 허용한다 (branch.py).

Test A 를 캐시만으로 재생한다. 캐시 미스는 transport 가 예외로 막는다 — llm.py 는
미스 때 경고 없이 실제 API 를 부르므로, 이 가드가 없으면 "호출 0" 을 주장할 수 없다.
재생한 degraded 가 waste_ratio.csv 와 600/600 일치해야 진단이 유효하다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import lfm.branch as branch_mod  # noqa: E402
from lfm.branch import run_branch  # noqa: E402
from lfm.config import load_config  # noqa: E402
from lfm.llm import LLM  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402

TEST_A = "results/test_a/20260829-041445"
MIN_BODY = 200          # 사전등록 primary 노출 기준: 본문 200자 이상
N_BOOT = 2000


class CacheMiss(Exception):
    """재시도 불가 예외. llm._retryable 이 False 를 돌려 곧바로 LLMError 가 된다."""


def guard_transport(**kw):
    raise CacheMiss("캐시 미스 — 실제 API 호출을 막았다 (tag 미상, prompt %d자)"
                    % len(kw.get("prompt", "")))


class Recorder:
    """run_branch 한 번 동안 검색 결과와 컨텍스트 적재 내역을 모은다."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.searches: list[list[str]] = []
        self.included: dict[str, int] = {}     # chunk_id -> 실린 본문 길이
        self.full_len: dict[str, int] = {}     # chunk_id -> 원 본문 길이
        self.n_chunks = 0

    def reset(self):
        self.searches, self.included, self.full_len, self.n_chunks = [], {}, {}, 0

    def search(self, query, top_k):
        out = self.retriever.search(query, top_k)
        self.searches.append([c.chunk_id for c in out])
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="D15 빈 답안 원인 진단 (호출 0회)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0, help="상위 N개 토픽만 (스모크)")
    ap.add_argument("--outdir", default="results/d15")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config or (root / "config.yaml"))
    llm = LLM(cfg, transport=guard_transport)
    rec = Recorder(BM25Retriever(cfg))
    max_chars = int(cfg.get("branch.max_context_chars", 12000))

    original = branch_mod.format_context

    def recording_format_context(chunks, mc):
        # format_context 와 같은 규칙으로 청크별 실린 길이를 센다. 결과 문자열이 원본과
        # 바이트 동일한지 assert 해서 재구성이 어긋나지 않았음을 보장한다.
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
        assert "\n".join(parts) == text, "컨텍스트 재구성이 원본과 다르다"
        return text

    branch_mod.format_context = recording_format_context

    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    if args.limit:
        topics = topics[: args.limit]

    ref = {}
    with io.open(root / TEST_A / "waste_ratio.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ref[(r["query_id"], int(r["round"]), r["branch"])] = int(r["degraded"])

    rows, branches, mismatch = [], [], []
    max_rounds = int(cfg["max_rounds"])
    for ti, t in enumerate(topics):
        qid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        perspectives = decompose(t["topic"], llm, cfg)
        for r in range(max_rounds):
            for p in perspectives:
                rec.reset()
                out = run_branch(p, rec, llm, cfg, round_=r, search_anchor=anchor)
                key = (qid, r, p.role)
                if key not in ref or ref[key] != int(out.degraded):
                    mismatch.append({"key": list(key), "ref": ref.get(key), "got": int(out.degraded)})
                n_inc = len(rec.included)
                n_trunc = sum(1 for cid, n in rec.included.items() if n < rec.full_len[cid])
                assert len(rec.searches) == len(out.sub_answers), "검색 수 != sub-question 수"
                n_exposed_all = 0
                for sa, own in zip(out.sub_answers, rec.searches):
                    inc = [rec.included.get(cid, 0) for cid in own]
                    full = [rec.included.get(cid, -1) == rec.full_len.get(cid, -2) for cid in own]
                    exp200 = any(n >= MIN_BODY for n in inc)
                    n_exposed_all += exp200
                    rows.append({
                        "query_id": qid, "round": r, "branch": p.role, "sq_index": sa.index,
                        "n_subq": len(out.sub_answers), "n_own": len(own),
                        "own_in_ctx_any": int(any(n > 0 for n in inc)),
                        "own_in_ctx_200": int(exp200),
                        "own_in_ctx_full": int(any(full)),
                        "own_first_pos": next((i for i, cid in enumerate(own)
                                               if rec.included.get(cid, 0) > 0), -1),
                        "empty": int(not sa.text.strip()),
                        "n_citations": len(sa.citations),
                        "branch_degraded": int(out.degraded),
                    })
                branches.append({
                    "query_id": qid, "round": r, "branch": p.role,
                    "degraded": int(out.degraded), "truncated": int(out.truncated),
                    "n_retrieved": rec.n_chunks, "n_in_ctx": n_inc, "n_cut": n_trunc,
                    "n_subq": len(out.sub_answers), "n_subq_exposed": n_exposed_all,
                })
        print("[%2d/%2d] %s" % (ti + 1, len(topics), qid), flush=True)

    stats = llm.stats.to_dict()
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    for name, data in (("d15_subq.csv", rows), ("d15_branch.csv", branches)):
        with io.open(outdir / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    summary = {
        "n_topics": len(topics), "n_branch_rounds": len(branches), "n_subq": len(rows),
        "llm_stats": stats, "degraded_mismatch": mismatch,
        "n_degraded": sum(b["degraded"] for b in branches),
        "primary": rr_report(rows, "own_in_ctx_200"),
        "sensitivity_any": rr_report(rows, "own_in_ctx_any"),
        "sensitivity_full": rr_report(rows, "own_in_ctx_full"),
        "by_position": by_position(rows, "own_in_ctx_200"),
        "degraded_all_exposed": sum(1 for b in branches
                                    if b["degraded"] and b["n_subq_exposed"] == b["n_subq"]),
        "degraded_none_exposed": sum(1 for b in branches
                                     if b["degraded"] and b["n_subq_exposed"] == 0),
        "ctx": {k: pct([b[k] for b in branches]) for k in ("n_retrieved", "n_in_ctx", "n_cut")},
    }
    with io.open(outdir / "d15_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "degraded_mismatch"},
                     ensure_ascii=False, indent=1))
    print("degraded 불일치 %d건" % len(mismatch))
    return 0 if not mismatch else 2


def pct(xs):
    a = np.asarray(xs, dtype=float)
    return {"median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def rates(rows, col):
    e1 = [r["empty"] for r in rows if not r[col]]
    e0 = [r["empty"] for r in rows if r[col]]
    return e1, e0


def rr_report(rows, col):
    e1, e0 = rates(rows, col)
    out = {"n_unexposed": len(e1), "n_exposed": len(e0),
           "E1": float(np.mean(e1)) if e1 else None, "E0": float(np.mean(e0)) if e0 else None}
    if len(e1) < 30 or not e0 or np.mean(e0) == 0:
        out["verdict"] = "판정 불가"
        return out
    out["RR"] = out["E1"] / out["E0"]
    by_q: dict[str, list[dict]] = {}
    for r in rows:
        by_q.setdefault(r["query_id"], []).append(r)
    qs = sorted(by_q)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(N_BOOT):
        sample = [r for q in rng.choice(qs, size=len(qs), replace=True) for r in by_q[q]]
        b1, b0 = rates(sample, col)
        if b1 and b0 and np.mean(b0) > 0:
            boots.append(np.mean(b1) / np.mean(b0))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out["RR_ci"] = [float(lo), float(hi)]
    out["verdict"] = "A형" if (out["RR"] >= 2 and lo > 1) else "B형"
    return out


def by_position(rows, col):
    res = {}
    for pos in sorted({r["sq_index"] for r in rows}):
        sub = [r for r in rows if r["sq_index"] == pos]
        e1, e0 = rates(sub, col)
        res[str(pos)] = {"n_unexposed": len(e1), "E1": float(np.mean(e1)) if e1 else None,
                         "n_exposed": len(e0), "E0": float(np.mean(e0)) if e0 else None}
    return res


if __name__ == "__main__":
    sys.exit(main())
