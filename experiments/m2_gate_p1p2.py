"""SPEC_M2 v1.1 게이트 P1·P2 (양성). **판정 호출 nano 100회.** ORDERS §B-1 이 허용한 유일한 생성.

N1(음성)은 이미 탈락했다 — 같은 branch·같은 sub-question claim 쌍의 26.4% 가
contradiction 으로 잡혔다(기준 5%). ORDERS §C 의 분기가 이미 정해져 있다:

> 관계 v1.1 통제 탈락 → Test C-1 미실행, `− relation` 구성이 Full 이 된다.
> SPEC_M2 §12 탈락 문장 사용. **계속 진행.**

그래도 P1·P2 를 돌리는 이유는 ORDERS §B-1 이 세 통제를 함께 산출하라고 하기 때문이고,
**원고의 ablation 절이 "관계 모듈이 왜 꺼졌는지"를 한 문장으로 말하려면 이 둘이 필요**
하기 때문이다. 위양성만 많은 것인가, 검출도 못 하는 것인가는 다른 이야기다.

  P1 양성  claim 50개의 **부정문**과의 쌍 → contradiction 검출 ≥ 80%
  P2 양성  claim 50개의 **paraphrase** 와의 쌍 → 대칭 entailment ≥ 80%

부정문·paraphrase 는 `gpt-5.4-nano` 로 만든다. 프롬프트는 `prompts/m2_negate.txt`,
`prompts/m2_paraphrase.txt` 에 고정하고 실행 후 수정하지 않는다.

**임계는 고정이다.** 결과를 보고 바꾸지 않는다. v1.2 는 없다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from diag_sweep import CONDITIONS, pick_topics  # noqa: E402
from m2_relation_run import MAX_CLAIMS, RELATION_CK  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.relation_v11 import ThresholdsV11  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402

N_SAMPLE = 50
GATE_P1 = 80.0        # 부정문 검출률
GATE_P2 = 80.0        # paraphrase 대칭 entailment
JUDGE = "gpt-5.4-nano"


def main(argv=None):
    ap = argparse.ArgumentParser(description="v1.1 게이트 P1·P2 (nano 100회)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--out", default="results/m2/relation/gate_p1p2.json")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(str(root / ".env"))
    cfg = load_config(root / "config.yaml")
    spec = CONDITIONS[args.condition]
    data = cfg.as_dict()
    if spec["top_k"] is not None:
        data["retrieval"]["top_k"] = spec["top_k"]
    if spec["index"] == "partial":
        data["paths"]["index_dir"] = data["paths"]["index_dir_partial"]
    ccfg = Config(data, source=args.condition)

    d2 = ccfg.as_dict()
    d2["nli"]["model"] = RELATION_CK
    nli = load_nli(Config(d2, source="m2:p1p2"))
    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))
    th = ThresholdsV11()

    print("claim 수집 (캐시 재생, 생성 호출 0) ...")
    pool = []
    for t in picked:
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            for c in [x for x in o.claims if x.strip()][:MAX_CLAIMS]:
                pool.append({"topic": tid, "branch": p.role, "claim": c})
    print("claim 모집단 %d" % len(pool))

    rng = np.random.default_rng(int(cfg["seed"]))
    idx = sorted(rng.choice(len(pool), size=min(N_SAMPLE, len(pool)), replace=False).tolist())
    sample = [pool[i] for i in idx]
    print("표본 %d (시드 고정)" % len(sample))

    jd = cfg.as_dict()
    jd["llm"]["model"] = JUDGE
    jd["llm"]["max_tokens"] = 256
    gen = LLM(Config(jd, source="m2:p1p2"))
    neg_t = io.open(root / "prompts/m2_negate.txt", encoding="utf-8").read()
    par_t = io.open(root / "prompts/m2_paraphrase.txt", encoding="utf-8").read()

    print("")
    print("부정문·paraphrase 생성 (%s, %d x 2 호출) ..." % (JUDGE, len(sample)), flush=True)
    for i, r in enumerate(sample):
        r["negation"] = gen.call(neg_t.format(claim=r["claim"]),
                                 tag="m2_negate", max_tokens=256).text.strip()
        r["paraphrase"] = gen.call(par_t.format(claim=r["claim"]),
                                   tag="m2_paraphrase", max_tokens=256).text.strip()
        if (i + 1) % 10 == 0:
            print("  %d/%d" % (i + 1, len(sample)), flush=True)

    print("")
    print("NLI 채점 ...", flush=True)
    rows = []
    for r in sample:
        c, n_, p_ = r["claim"], r["negation"], r["paraphrase"]
        ok_n = bool(n_) and n_ != c
        ok_p = bool(p_) and p_ != c
        sn = nli.score([(c, n_), (n_, c)]) if ok_n else None
        sp = nli.score([(c, p_), (p_, c)]) if ok_p else None
        rows.append({
            "topic": r["topic"], "branch": r["branch"],
            "neg_usable": int(ok_n), "par_usable": int(ok_p),
            "p1_contradiction": round(max(sn[0].contradiction, sn[1].contradiction), 6)
            if sn else None,
            "p2_entailment": round((sp[0].entailment + sp[1].entailment) / 2.0, 6)
            if sp else None,
            "claim": c, "negation": n_, "paraphrase": p_})

    p1 = [r["p1_contradiction"] for r in rows if r["p1_contradiction"] is not None]
    p2 = [r["p2_entailment"] for r in rows if r["p2_entailment"] is not None]
    r1 = 100.0 * sum(1 for x in p1 if x > th.tau_c) / len(p1) if p1 else 0.0
    r2 = 100.0 * sum(1 for x in p2 if x >= 0.5) / len(p2) if p2 else 0.0
    ok1, ok2 = r1 >= GATE_P1, r2 >= GATE_P2

    outp = root / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with io.open(outp.with_suffix(".csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    io.open(outp, "w", encoding="utf-8").write(json.dumps({
        "checkpoint": RELATION_CK, "generator": JUDGE, "n": len(sample),
        "tau_c": th.tau_c,
        "P1": {"n": len(p1), "median": round(st.median(p1), 6) if p1 else None,
               "detect_pct": round(r1, 4), "gate": GATE_P1, "pass": bool(ok1)},
        "P2": {"n": len(p2), "median": round(st.median(p2), 6) if p2 else None,
               "entail_pct": round(r2, 4), "gate": GATE_P2, "pass": bool(ok2)},
        "calls": gen.stats.calls,
        "billed_in": gen.stats.billed_tokens_in, "billed_out": gen.stats.billed_tokens_out,
    }, ensure_ascii=False, indent=1))

    print("")
    print("=== 게이트 P1 (부정문 검출) ===")
    print("  n=%d  contradiction 중앙 %.4f  **> tau_c 인 비율 %.1f%%**  (기준 >=%.0f%%)  -> %s"
          % (len(p1), st.median(p1) if p1 else 0.0, r1, GATE_P1, "PASS" if ok1 else "**FAIL**"))
    print("=== 게이트 P2 (paraphrase 대칭 entailment) ===")
    print("  n=%d  entailment 중앙 %.4f  **>=0.5 인 비율 %.1f%%**  (기준 >=%.0f%%)  -> %s"
          % (len(p2), st.median(p2) if p2 else 0.0, r2, GATE_P2, "PASS" if ok2 else "**FAIL**"))
    print("")
    print("N1 은 이미 탈락(26.4%%). ORDERS §C: 관계 모듈 비활성으로 계속 진행.")
    print("생성 호출 %d (캐시 %d), 청구 %d/%d"
          % (gen.stats.calls, gen.stats.cache_hits,
             gen.stats.billed_tokens_in, gen.stats.billed_tokens_out))
    print("-> " + str(outp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
