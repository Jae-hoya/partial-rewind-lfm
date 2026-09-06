"""SPEC_M2 v1.1 게이트 N1 (음성). **호출 0회.**

v1.1 은 교란 토픽을 음성 통제로 못 쓴다 — 관련 pair 로 제한하면 다른 토픽 쌍은
관련 pair 가 거의 0 이 되어 통제가 공허해진다. 그래서 통제를 바꿨다.

**N1 음성**: **같은 branch 안, 같은 sub-question 에 귀속된 claim 쌍.**
관련이 있고(같은 sub-question), 충돌은 없어야 한다(같은 답안에서 나온 사실들이다).
즉 v1.1 이 "관련 있는데 충돌 없음" 을 제대로 음성으로 판정하는지를 본다.

  통과 기준 (실행 전 고정): contradiction > tau_c 인 쌍이 **≤ 5%**

**N1 은 max 팽창도 같이 잡는다.** 관련 pair 안에서도 max 가 넘치면 여기서 떨어지고,
그때 집계를 바꾸는 v1.2 는 만들지 않는다 — 마스터플랜 §9-2 로 간다.

같은 sub-question 안의 claim 쌍이므로 정의상 aboutness 가 일치한다. v1.0 이
실패한 원인(무관한 주제를 충돌로 읽음)이 여기서는 원천적으로 없다.
그런데도 충돌이 5% 를 넘으면, 문제는 aboutness 가 아니라 **NLI 자체**다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics as st
import sys
from itertools import combinations
from pathlib import Path

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
from lfm.relation_v11 import ThresholdsV11, attribute_claims  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402

GATE = 5.0


def main(argv=None):
    ap = argparse.ArgumentParser(description="v1.1 게이트 N1 (호출 0회)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--out", default="results/m2/relation/gate_n1.json")
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
    nli = load_nli(Config(d2, source="m2:n1"))
    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))
    th = ThresholdsV11()

    print("claim 수집 + sub-question 귀속 (캐시 재생, 호출 0) ...")
    groups: list[dict] = []       # 같은 sub-question 에 귀속된 claim 묶음
    attr_stat = {"attributed": 0, "failed": 0}
    for n, t in enumerate(picked):
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            claims = [c for c in o.claims if c.strip()][:MAX_CLAIMS]
            subs = [sa.text for sa in o.sub_answers]
            if not claims or not subs:
                continue
            attr = attribute_claims(claims, subs)
            attr_stat["attributed"] += sum(1 for x in attr if x >= 0)
            attr_stat["failed"] += sum(1 for x in attr if x < 0)
            by_q: dict[int, list[int]] = {}
            for k, q in enumerate(attr):
                if q >= 0:
                    by_q.setdefault(q, []).append(k)
            for q, idxs in by_q.items():
                if len(idxs) >= 2:
                    groups.append({"topic": tid, "branch": p.role, "subq": q,
                                   "claims": [claims[k] for k in idxs]})
        print("  [%2d/%2d] %s  누적 묶음 %d" % (n + 1, len(picked), tid, len(groups)),
              flush=True)

    print("")
    print("귀속: 성공 %d · 실패 %d (%.1f%%)"
          % (attr_stat["attributed"], attr_stat["failed"],
             100.0 * attr_stat["failed"] / max(1, sum(attr_stat.values()))))
    print("같은 sub-question 에 claim 2개 이상인 묶음 %d개" % len(groups))

    print("")
    print("N1 채점 ...", flush=True)
    rows, cons = [], []
    for g in groups:
        cl = g["claims"]
        for a, b in combinations(range(len(cl)), 2):
            sc = nli.score([(cl[a], cl[b]), (cl[b], cl[a])])
            con = max(sc[0].contradiction, sc[1].contradiction)
            ent = (sc[0].entailment + sc[1].entailment) / 2.0
            cons.append(con)
            rows.append({"topic": g["topic"], "branch": g["branch"], "subq": g["subq"],
                         "a": a, "b": b, "contradiction": round(con, 6),
                         "entailment": round(ent, 6)})

    if not cons:
        print("쌍이 없다. 귀속이 전부 흩어졌거나 claim 이 부족하다.")
        return 1

    hit = sum(1 for x in cons if x > th.tau_c)
    rate = 100.0 * hit / len(cons)
    ok = rate <= GATE

    outp = root / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with io.open(outp.with_suffix(".csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    io.open(outp, "w", encoding="utf-8").write(json.dumps({
        "gate": "N1", "checkpoint": RELATION_CK, "tau_c": th.tau_c, "threshold_pct": GATE,
        "n_pairs": len(cons), "n_groups": len(groups),
        "contradiction_median": round(st.median(cons), 6),
        "contradiction_mean": round(st.mean(cons), 6),
        "gt_tau_c_pct": round(rate, 4), "pass": bool(ok),
        "attribution": attr_stat,
        "note": ("통과" if ok else
                 "5% 초과. 관련 pair 안에서도 충돌이 넘친다 — 집계를 바꾸는 v1.2 는 "
                 "만들지 않는다(마스터플랜 §9-2).")}, ensure_ascii=False, indent=1))

    print("")
    print("=== 게이트 N1 (음성) — 같은 branch·같은 sub-question claim 쌍 ===")
    print("  쌍 %d개 (묶음 %d)" % (len(cons), len(groups)))
    print("  contradiction 중앙 %.6f · 평균 %.6f" % (st.median(cons), st.mean(cons)))
    print("  **> tau_c(%.2f) 인 쌍 %d개 = %.2f%%**  (기준 <=%.0f%%)"
          % (th.tau_c, hit, rate, GATE))
    print("")
    print("판정: %s" % ("PASS" if ok else "**FAIL**"))
    print("")
    print("대조 — v1.0 전 pair: 정상 16.3%% / 교란 28.3%% (pair 단위)")
    print("LLM 호출 %d (캐시 %d), 청구 %d/%d"
          % (llm.stats.calls, llm.stats.cache_hits,
             llm.stats.billed_tokens_in, llm.stats.billed_tokens_out))
    print("-> " + str(outp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
