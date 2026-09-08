"""전제 길이 분포 — (나) 안이 성립하는지 먼저 확인한다. **호출 0, 모델 적재 0.**

공저자 지시 순서 1: "max_model_len 을 올려 진짜 단일 문서로 만들고, 전제 길이 분포를
먼저 보고하세요 — 최대 몇 토큰인지, 새 상한에서도 쪼개지거나 잘리는 건이 0인지.
105건 잘림도 여기서 사라져야 합니다. 안 사라지면 (나)가 성립 안 하니 거기서 멈춥니다."

**왜 쪼개졌는지 원인을 먼저 적는다.** `minicheck/minicheck.py:148` 이

    self.model.chunk_size = 500 if self.model.model_name == 'flan-t5-large' else 400

로 **하드코딩**한다. `max_model_len=2048` 은 쪼개기에 아무 영향이 없다 — 그 값은
개별 청크를 자를 때만 쓰인다(`inference.py:128`). 그리고 `inference()` 주석이
"Standard aggregation (**max**) over chunks of doc" 라고 적어 두었다. 즉 500토큰을
넘는 전제는 전부 쪼개져 최댓값 집계로 채점됐다.

고치는 방법은 `max_model_len` 이 아니라 `score(..., chunk_size=N)` 이다.
`inference.py:444` 가 `chunk_size < max_model_len` 을 강제하므로 둘 다 올려야 한다.

여기서는 **아무것도 고치지 않는다.** N 을 얼마로 잡아야 하는지, 그 N 에서 잘림이
0 이 되는지를 데이터로 답한다.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml                                          # noqa: E402

from diag_sweep import CONDITIONS, pick_topics       # noqa: E402

from lfm.branch import run_branch                    # noqa: E402
from lfm.config import Config, load_config           # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present      # noqa: E402
from lfm.perspectives import decompose               # noqa: E402
from lfm.retriever import BM25Retriever              # noqa: E402
from lfm.types import split_sentences                # noqa: E402

OUT = ROOT / "results/vprime"
CUR_CHUNK = 500          # minicheck.py:148 하드코딩 값 (flan-t5-large)
CUR_MAXLEN = 2048        # d9_minicheck.Scorer / MiniCheckScorer 가 넘긴 값
HEADROOM = 300           # inference.py:300 이 claim 자리로 빼 두는 양


def pct(v, q):
    s = sorted(v)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[i]


def main() -> int:
    load_dotenv_if_present(str(ROOT / ".env"))
    cfg = load_config(ROOT / "config.yaml")
    spec = CONDITIONS["C3_partial_heldout"]
    data = cfg.as_dict()
    if spec["top_k"] is not None:
        data["retrieval"]["top_k"] = spec["top_k"]
    if spec["index"] == "partial":
        data["paths"]["index_dir"] = data["paths"]["index_dir_partial"]
    ccfg = Config(data, source="C3_partial_heldout")

    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, 15, int(cfg["seed"]))

    print("모집단 재생 (캐시 재생, 생성 호출 0) ...", flush=True)
    pop = []
    for n, t in enumerate(picked):
        anchor = " ".join(t.get("terms") or [])
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            for sa in o.sub_answers:
                cited = [cmap[c].text for c in sa.citations if c in cmap]
                if not cited:
                    continue
                for sent in split_sentences(sa.text):
                    pop.append({"item_id": "v%03d" % len(pop),
                                "doc": "\n\n".join(cited), "claim": sent,
                                "n_premises": len(cited)})
        print("  [%2d/%2d] %s  누적 %d" % (n + 1, len(picked), t["id"], len(pop)), flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("lytang/MiniCheck-Flan-T5-Large")

    vids = {r["item_id"] for r in
            csv.DictReader(io.open(OUT / "labels_stage1.csv", encoding="utf-8-sig"))}

    rows = []
    for r in pop:
        nd = len(tok.encode(r["doc"], truncation=False))
        nc = len(tok.encode(r["claim"], truncation=False))
        rows.append({"item_id": r["item_id"], "in_vprime": int(r["item_id"] in vids),
                     "n_premises": r["n_premises"], "doc_tokens": nd, "claim_tokens": nc,
                     "doc_plus_claim": nd + nc})

    d = [r["doc_tokens"] for r in rows]
    c = [r["claim_tokens"] for r in rows]
    tot = [r["doc_plus_claim"] for r in rows]

    print("")
    print("=" * 74)
    print("전제 길이 분포 (MiniCheck-Flan-T5-Large 토크나이저, n=%d)" % len(rows))
    print("=" * 74)
    for name, v in [("전제 doc", d), ("문장 claim", c), ("doc+claim", tot)]:
        print("  %-10s 중앙 %5d · p90 %5d · p99 %5d · **최대 %5d**"
              % (name, pct(v, .5), pct(v, .9), pct(v, .99), max(v)))
    print("")
    print("  현행 쪼개기 기준 chunk_size=%d 초과: **%d건 (%.1f%%)**"
          % (CUR_CHUNK, sum(1 for x in d if x > CUR_CHUNK),
             100.0 * sum(1 for x in d if x > CUR_CHUNK) / len(d)))
    print("  현행 자르기 기준 %d토큰 초과: **%d건 (%.1f%%)**"
          % (CUR_MAXLEN - 64, sum(1 for x in d if x > CUR_MAXLEN - 64),
             100.0 * sum(1 for x in d if x > CUR_MAXLEN - 64) / len(d)))

    # 필요한 상한: 모든 전제가 한 조각에 들어가고 아무것도 안 잘리는 최소값
    need_chunk = max(d)
    need_maxlen = max(tot) + HEADROOM
    print("")
    print("=" * 74)
    print("(나) 를 성립시키려면")
    print("=" * 74)
    print("  chunk_size   >= **%d** (가장 긴 전제)" % need_chunk)
    print("  max_model_len >= **%d** (doc+claim 최대 %d + inference.py:300 여유 %d)"
          % (need_maxlen, max(tot), HEADROOM))
    print("  또한 inference.py:444 가 chunk_size < max_model_len 을 강제한다.")

    CAND = 4096
    over_c = [x for x in d if x > CAND]
    over_m = [r for r in rows if r["doc_plus_claim"] > CAND]
    print("")
    print("  후보 상한 %d 로 잡으면:" % CAND)
    print("    쪼개지는 건 %d · 잘리는 건 %d  -> %s"
          % (len(over_c), len(over_m),
             "**둘 다 0. (나) 성립.**" if not over_c and not over_m
             else "**0 이 아니다. (나) 재검토.**"))

    sub = [r for r in rows if r["in_vprime"]]
    print("")
    print("  V′ 60건만 보면 최대 전제 %d 토큰 · 최대 doc+claim %d 토큰"
          % (max(r["doc_tokens"] for r in sub), max(r["doc_plus_claim"] for r in sub)))

    with io.open(OUT / "premise_lengths.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    io.open(OUT / "premise_lengths.json", "w", encoding="utf-8").write(json.dumps({
        "n": len(rows),
        "doc_tokens": {"median": pct(d, .5), "p90": pct(d, .9), "p99": pct(d, .99), "max": max(d)},
        "doc_plus_claim_max": max(tot),
        "current": {"chunk_size": CUR_CHUNK, "max_model_len": CUR_MAXLEN,
                    "n_over_chunk": sum(1 for x in d if x > CUR_CHUNK),
                    "n_over_truncate": sum(1 for x in d if x > CUR_MAXLEN - 64)},
        "required": {"chunk_size_min": need_chunk, "max_model_len_min": need_maxlen},
        "candidate_4096": {"n_split": len(over_c), "n_truncated": len(over_m)},
        "cause": ("minicheck/minicheck.py:148 이 flan-t5-large 의 chunk_size 를 500 으로 "
                  "하드코딩한다. max_model_len 은 쪼개기에 영향이 없다. "
                  "inference() 는 청크 최댓값으로 집계한다."),
    }, ensure_ascii=False, indent=1))
    print("")
    print("-> %s" % (OUT / "premise_lengths.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
