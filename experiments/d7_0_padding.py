"""D7-0 — 양성 통제. 전제 길이만 늘려 p_entail 이 무너지는지 본다.

기존 통제 쌍(entail 0.997 / contradiction 1.000)은 **짧은 전제**로 쟀다. 따라서
"NLI 자체는 정상"이라는 결론은 **그 길이 조건에서의 정상**일 뿐이고, H2(전제 길이)를
반증하지 못한다.

설계: 이미 `p_entail > 0.90` 으로 판정된 쌍을 가져다 **내용은 그대로 두고 전제 길이만**
1x / 2x / 3x / 4x 로 늘린다. 붙이는 텍스트는 같은 코퍼스의 무관한 청크(시드 고정,
가설과 어휘 겹침 없음)다. 가설 문장은 건드리지 않는다.

  배수가 오를수록 무너지면  -> H2 는 **확인된 실패 양식**이다. 다른 해석의 여지가 없다
  안 무너지면              -> H2 는 사실상 배제된다

`truncated_by_tokenizer` 를 반드시 기록한다. 512 토큰에서 잘리면 그 잘림 자체가
교란 요인이고, 기록이 없으면 결과를 해석할 수 없다.

생성 LLM 호출 0회. NLI forward pass 만 쓴다.
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics as st
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_sweep import CONDITIONS, pick_topics  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

MULTIPLIERS = (1, 2, 3, 4)


def content_words(text: str) -> set[str]:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split()
            if len(w) >= 4}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D7-0 패딩 통제 (생성 LLM 호출 0회)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--out", default="results/diag/d7/padding_control.csv")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(str(repo_root / ".env"))
    cfg = load_config(args.config or (repo_root / "config.yaml"))

    spec = CONDITIONS[args.condition]
    data = cfg.as_dict()
    if spec["top_k"] is not None:
        data["retrieval"]["top_k"] = spec["top_k"]
    if spec["index"] == "partial":
        data["paths"]["index_dir"] = data["paths"]["index_dir_partial"]
    ccfg = Config(data, source=args.condition)

    nli = load_nli(ccfg)          # torch/CUDA 를 먼저 올린다 (뒤에 올리면 세그폴트)
    llm = LLM(ccfg)
    tokenizer = nli.tokenizer
    max_len = nli.max_length

    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))
    retriever = BM25Retriever(ccfg)

    # --- 1) 양성 쌍 수집 + 패딩 후보 풀 ---------------------------------
    positives: list[dict] = []
    filler: list[str] = []
    for n, t in enumerate(picked):
        anchor = " ".join(t.get("terms") or [])
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            filler.extend(c.text for c in o.chunks)
            for sa in o.sub_answers:
                cited = [cmap[c].text for c in sa.citations if c in cmap]
                if not cited:
                    continue
                for sent in split_sentences(sa.text):
                    scores = nli.entailment([(c, sent) for c in cited])
                    best = max(range(len(scores)), key=lambda i: scores[i])
                    if scores[best] > args.threshold:
                        positives.append({"query_id": str(t["id"]), "branch": p.role,
                                          "premise": cited[best], "hypothesis": sent,
                                          "p0": scores[best]})
        print("  [%2d/%2d] %s  (양성 %d)" % (n + 1, len(picked), t["id"], len(positives)),
              flush=True)

    print()
    print("양성 쌍 %d개 (p_entail > %.2f), 패딩 후보 청크 %d개"
          % (len(positives), args.threshold, len(filler)))
    if not positives:
        print("양성 쌍이 없다. threshold 를 낮춰라.")
        return 1

    # --- 2) 배수별 패딩 후 재측정 ----------------------------------------
    rng = np.random.default_rng(int(cfg["seed"]))
    rows = []
    for idx, pos in enumerate(positives):
        hyp_words = content_words(pos["hypothesis"])
        # 가설과 어휘가 겹치지 않는 청크만 패딩에 쓴다
        pool = [f for f in filler
                if f != pos["premise"] and not (content_words(f) & hyp_words)]
        if len(pool) < max(MULTIPLIERS) - 1:
            pool = [f for f in filler if f != pos["premise"]]
        order = rng.permutation(len(pool))[: max(MULTIPLIERS) - 1]
        pads = [pool[i] for i in order]

        base_tokens = len(tokenizer.encode(pos["premise"], add_special_tokens=False))
        for m in MULTIPLIERS:
            premise = " ".join([pos["premise"]] + pads[: m - 1])
            n_tok = len(tokenizer.encode(premise, add_special_tokens=False))
            enc = tokenizer(premise, pos["hypothesis"], truncation=True,
                            max_length=max_len)
            truncated = (len(tokenizer.encode(premise, add_special_tokens=False))
                         + len(tokenizer.encode(pos["hypothesis"], add_special_tokens=False))
                         + 3) > max_len
            score = nli.entailment([(premise, pos["hypothesis"])])[0]
            rows.append({
                "sentence_id": idx, "query_id": pos["query_id"], "branch": pos["branch"],
                "multiplier": m,
                "premise_tokens_before": base_tokens,
                "premise_tokens_after_tokenizer": n_tok,
                "encoded_length": len(enc["input_ids"]),
                "truncated_by_tokenizer": int(truncated),
                "p_entail": round(float(score), 6),
                "p_entail_at_1x": round(float(pos["p0"]), 6),
            })
        if (idx + 1) % 10 == 0:
            print("  패딩 %d/%d" % (idx + 1, len(positives)), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- 3) 보고 ---------------------------------------------------------
    print()
    print("=== 배수별 p_entail (양성 쌍 %d개) ===" % len(positives))
    print("| 배수 | 전제 토큰 중앙 | 잘림 | p_entail 중앙 | 평균 | >=0.5 유지 | >=0.9 유지 |")
    print("|---|---|---|---|---|---|---|")
    for m in MULTIPLIERS:
        sub = [r for r in rows if r["multiplier"] == m]
        ps = [r["p_entail"] for r in sub]
        print("| %dx | %d | %d/%d | %.3f | %.3f | %.0f%% | %.0f%% |"
              % (m, st.median([r["premise_tokens_after_tokenizer"] for r in sub]),
                 sum(r["truncated_by_tokenizer"] for r in sub), len(sub),
                 st.median(ps), st.mean(ps),
                 100 * sum(1 for x in ps if x >= 0.5) / len(ps),
                 100 * sum(1 for x in ps if x >= 0.9) / len(ps)))
    print()
    p1 = {r["sentence_id"]: r["p_entail"] for r in rows if r["multiplier"] == 1}
    p4 = {r["sentence_id"]: r["p_entail"] for r in rows if r["multiplier"] == 4}
    drops = [p1[k] - p4[k] for k in p1]
    print("1x -> 4x 하락폭: 중앙 %.3f · 평균 %.3f · 0.5 이상 무너진 쌍 %d/%d"
          % (st.median(drops), st.mean(drops),
             sum(1 for k in p1 if p1[k] >= 0.5 > p4[k]), len(p1)))
    print()
    print("판정: %s" % ("H2 확인 — 내용을 그대로 두고 길이만 늘렸는데 무너진다"
                       if st.mean(drops) > 0.2 else
                       "H2 배제 — 길이를 4배로 늘려도 유지된다"))
    print()
    print("LLM 호출 %d (캐시 %d), 청구 토큰 %d/%d"
          % (llm.stats.calls, llm.stats.cache_hits,
             llm.stats.billed_tokens_in, llm.stats.billed_tokens_out))
    print("-> " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
