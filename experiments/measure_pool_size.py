# -*- coding: utf-8 -*-
"""branch 범위 전제 문장 풀 크기 실측. NLI 미적재, 생성 호출 0 (캐시 재생)."""
import io, statistics as st, sys
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "experiments")
import yaml
from diag_sweep import CONDITIONS, pick_topics
from lfm.branch import run_branch
from lfm.config import Config, load_config
from lfm.llm import LLM, load_dotenv_if_present
from lfm.retriever import BM25Retriever
from lfm.types import split_sentences

load_dotenv_if_present(".env")
cfg = load_config("config.yaml")
spec = CONDITIONS["C3_partial_heldout"]
d = cfg.as_dict()
if spec["top_k"] is not None: d["retrieval"]["top_k"] = spec["top_k"]
if spec["index"] == "partial": d["paths"]["index_dir"] = d["paths"]["index_dir_partial"]
ccfg = Config(d, source="C3_partial_heldout")
llm = LLM(ccfg); retriever = BM25Retriever(ccfg)
topics = yaml.safe_load(io.open(Path(cfg["paths.data_dir"])/"topics_pilot50.yaml", encoding="utf-8"))["topics"]
picked = pick_topics(topics, 15, int(cfg["seed"]))
from lfm.perspectives import decompose
per_branch, per_topic, per_sub = [], [], []
for t in picked:
    tid = str(t["id"]); anchor = " ".join(t.get("terms") or [])
    tot = 0
    for p in decompose(t["topic"], llm, ccfg):
        o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
        ps = []
        for c in o.cited_chunks(): ps.extend(split_sentences(c.text) or [c.text])
        if ps: per_branch.append(len(ps)); tot += len(ps)
        cmap = o.chunk_map()
        for sa in o.sub_answers:
            cited = [cmap[c].text for c in sa.citations if c in cmap]
            if not cited: continue
            n = sum(len(split_sentences(x) or [x]) for x in cited)
            per_sub.append(n)
    if tot: per_topic.append(tot)
    print("  %s" % tid, flush=True)
def s(lbl, v):
    print("%-38s n=%3d  중앙 %5.1f  평균 %6.1f  범위 %d-%d"
          % (lbl, len(v), st.median(v), st.mean(v), min(v), max(v)))
print()
print("전제 문장 풀 크기 (C3, 15토픽) — 실측")
s("branch 범위 (claim 가설, D7 표6)", per_branch)
s("sub-answer 범위 (문장 가설, D7 표6)", per_sub)
s("토픽 범위 (4 branch 합, D8-A 표7)", per_topic)
print("\nLLM 호출 %d (캐시 %d), 청구 %d/%d"
      % (llm.stats.calls, llm.stats.cache_hits, llm.stats.billed_tokens_in, llm.stats.billed_tokens_out))
