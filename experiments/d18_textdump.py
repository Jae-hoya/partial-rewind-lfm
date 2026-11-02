"""D18 1단계 — Test A 를 캐시로 재생해 sub-question 단위 텍스트를 복원한다. **호출 0회.** 등록: results/d18/PREREG_D18.md

trace 에는 질문·답·청크 본문이 없다. 그래서 d15 와 같은 방식으로 다시 돌려서 남긴다.
R_i(그 sub-question 자기 검색 결과)는 Recorder 가 중복 제거 **전** 검색 결과를 질문별로 잡으므로
`seen` 중복 제거 때문에 생기는 귀속 문제가 없다.

산출:
  subq_text.jsonl  2520행 — 질문(라운드 1·2 는 paraphrase), 답, 인용, 자기 검색 id, 실린 길이
  chunks.jsonl     컨텍스트에 실린 청크의 본문(실린 그대로, 머리말 포함). 청크 id 로 중복 제거
가드: 캐시 미스 0, degraded 600/600, legacy coverage·grounding 이 waste_ratio.csv 와 일치.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

import lfm.branch as branch_mod  # noqa: E402
from d15_abstention_diag import TEST_A, Recorder, guard_transport  # noqa: E402
from lfm.branch import run_branch  # noqa: E402
from lfm.config import load_config  # noqa: E402
from lfm.evidence import coverage  # noqa: E402
from lfm.llm import LLM  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.yaml")
    llm = LLM(cfg, transport=guard_transport)
    rec = Recorder(BM25Retriever(cfg))
    seen_text: dict[str, str] = {}

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
            shown = "(" + c.title + " > " + c.section_path + ") " + body
            prev = seen_text.setdefault(c.chunk_id, shown)
            if len(shown) > len(prev):          # 같은 청크가 더 길게 실린 적이 있으면 그것을 남긴다
                seen_text[c.chunk_id] = shown
        assert "\n".join(parts) == text, "컨텍스트 재구성이 원본과 다르다"
        return text

    branch_mod.format_context = recording_format_context

    with io.open(root / "data" / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    ref = {}
    with io.open(root / TEST_A / "waste_ratio.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ref[(r["query_id"], int(r["round"]), r["branch"])] = r

    outdir = root / "results/d18"
    rows, bad = [], []
    for ti, t in enumerate(topics):
        qid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        for r in range(int(cfg["max_rounds"])):
            for p in decompose(t["topic"], llm, cfg):
                rec.reset()
                out = run_branch(p, rec, llm, cfg, round_=r, search_anchor=anchor)
                want = ref[(qid, r, p.role)]
                if int(want["degraded"]) != int(out.degraded) or abs(float(want["coverage"]) - coverage(out)) > 1e-6:
                    bad.append([qid, r, p.role])
                for sa, own in zip(out.sub_answers, rec.searches):
                    rows.append({
                        "query_id": qid, "round": r, "branch": p.role, "sq_index": sa.index,
                        "n_subq": len(out.sub_answers), "sub_question": sa.sub_question,
                        "answer": sa.text, "citations": list(sa.citations),
                        "empty": int(not sa.text.strip()), "cited": int(bool(sa.citations)),
                        "own": own, "own_included": {cid: rec.included[cid] for cid in own if cid in rec.included},
                        "branch_degraded": int(out.degraded),
                    })
        print("[%2d/%2d] %s" % (ti + 1, len(topics), qid), flush=True)

    stats = llm.stats.to_dict()
    with io.open(outdir / "subq_text.jsonl", "w", encoding="utf-8") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    with io.open(outdir / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for cid, txt in sorted(seen_text.items()):
            fh.write(json.dumps({"chunk_id": cid, "text": txt}, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "chunks": len(seen_text), "api_calls": stats["api_calls"],
                      "calls": stats["calls"], "cache_hits": stats["cache_hits"], "mismatch": bad},
                     ensure_ascii=False))
    return 2 if bad or stats["api_calls"] else 0


if __name__ == "__main__":
    sys.exit(main())
