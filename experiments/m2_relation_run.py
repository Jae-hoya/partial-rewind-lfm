"""M2 첫 산출물 — 실자료에서 R_ij 를 뽑고 교란 음성 통제를 건다. **생성 호출 0회.**

SPEC_M2 §10 STOP RULE: "relation.py + 교란 통제 산출 후 중단·보고".

이건 추정자 확정(V′)과 **무관하게** 지금 돌릴 수 있다. relation 은 3-way 출력이
필요해서 어차피 `bart-large-mnli` 로 고정돼 있고(MiniCheck 는 3-way 가 없다),
어떤 게이트 결과에도 영향받지 않는다. claim 은 캐시에 이미 있다.

묻는 것은 하나다: **관점 사이의 contradiction 신호가 실재하는가, 아니면 아무 쌍에나
붙는 잡음인가.** 후자면 LFM 의 국소화 전제가 무너진다.

판정은 SPEC_M2 §2 의 사전 등록 기준이다 —
교란 쌍에서 `contradiction > τ_c` 비율과 `agreement > τ_a` 비율이 **둘 다 ≤5%**.
5% 를 넘으면 **τ 를 올리지 않는다.** 그 사실을 보고하고 Test C-1 해석에 반영한다.
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

from d8a_negative_control import SHIFT  # noqa: E402
from diag_sweep import CONDITIONS, pick_topics  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.relation import RelationExtractor, Thresholds  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402

RELATION_CK = "facebook/bart-large-mnli"     # SPEC_M2 §2 — 3-way 가 필요하다
MAX_CLAIMS = 12                              # branch 당 상한 (복잡도 6 pair x 144 x 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="M2 relation + 교란 통제 (호출 0회)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--outdir", default="results/m2/relation")
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
    nli = load_nli(Config(d2, source="m2:relation"))
    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))

    print("claim 수집 (캐시 재생, 생성 호출 0) ...")
    claims_by: dict[str, dict[str, list[str]]] = {}
    answers_by: dict[str, dict[str, str]] = {}
    subq_by: dict[str, dict[str, list[str]]] = {}
    for n, t in enumerate(picked):
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        claims_by[tid], answers_by[tid], subq_by[tid] = {}, {}, {}
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            claims_by[tid][p.role] = [c for c in o.claims if c.strip()][:MAX_CLAIMS]
            answers_by[tid][p.role] = "\n".join(sa.text for sa in o.sub_answers)
            subq_by[tid][p.role] = [sa.question for sa in o.sub_answers
                                    if getattr(sa, "question", None)]
        print("  [%2d/%2d] %s  branch %d · claim %s"
              % (n + 1, len(picked), tid, len(claims_by[tid]),
                 [len(v) for v in claims_by[tid].values()]), flush=True)

    def nli3(pairs):
        sc = nli.score(list(pairs))
        return [(s.entailment, s.contradiction) for s in sc]

    rx = RelationExtractor(nli3, th=Thresholds())
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- 정상 배정 -------------------------------------------------
    print("")
    print("정상 배정에서 R_ij 계산 ...", flush=True)
    rows, pair_ct = [], 0
    for tid in sorted(claims_by):
        roles = sorted(claims_by[tid])
        recs = []
        for a, b in combinations(roles, 2):
            ci, cj = claims_by[tid][a], claims_by[tid][b]
            if not ci or not cj:
                continue
            rel = rx.relate(tid + ":" + a, tid + ":" + b, ci, cj)
            # relatedness: sub-question 문자열 겹침 (무호출). coverage_gain 은
            # 인용 기반이라 M2 파이프라인에서 채운다 — 여기서는 0 으로 둔다.
            recs.append(rel.to_dict())
            rows.append({"topic": tid, "i": a, "j": b, **rel.to_dict()})
            pair_ct += rel.n_pairs
        with io.open(outdir / ("%s_r0.json" % tid), "w", encoding="utf-8") as fh:
            json.dump({"topic": tid, "round": 0, "checkpoint": RELATION_CK,
                       "relations": recs}, fh, ensure_ascii=False, indent=1)
        print("  %s  edge %d" % (tid, len(recs)), flush=True)

    th = rx.th
    con = [r["contradiction"] for r in rows]
    agr = [r["agreement"] for r in rows]
    n = len(rows)
    print("")
    print("정상 배정 edge %d개 (NLI 쌍 %d)" % (n, 2 * pair_ct))
    print("  contradiction  중앙 %.3f  >tau_c(%.2f) %d개 = %.1f%%"
          % (st.median(con), th.tau_c, sum(1 for x in con if x > th.tau_c),
             100.0 * sum(1 for x in con if x > th.tau_c) / n))
    print("  agreement      중앙 %.3f  >tau_a(%.2f) %d개 = %.1f%%"
          % (st.median(agr), th.tau_a, sum(1 for x in agr if x > th.tau_a),
             100.0 * sum(1 for x in agr if x > th.tau_a) / n))

    # ---------- 교란 음성 통제 --------------------------------------------
    print("")
    print("교란 음성 통제 (i -> i+%d mod %d) ..." % (SHIFT, len(claims_by)), flush=True)
    per_topic = {tid: [claims_by[tid][r] for r in sorted(claims_by[tid])]
                 for tid in sorted(claims_by)}
    nc = rx.negative_control(per_topic, shift=SHIFT)

    with io.open(outdir / "relations_normal.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with io.open(outdir / "negative_control.json", "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": RELATION_CK, "thresholds": th.__dict__,
                   "normal": {"n_edges": n,
                              "contradiction_gt_tau_c_pct":
                                  round(100.0 * sum(1 for x in con if x > th.tau_c) / n, 4),
                              "agreement_gt_tau_a_pct":
                                  round(100.0 * sum(1 for x in agr if x > th.tau_a) / n, 4)},
                   "shuffled": nc}, fh, ensure_ascii=False, indent=1)

    print("")
    print("=== SPEC_M2 §2 교란 통제 판정 (사전 등록, 사후 변경 금지) ===")
    print("| 배정 | contradiction > %.2f | agreement > %.2f |" % (th.tau_c, th.tau_a))
    print("|---|---|---|")
    print("| 정상 | %.1f%% | %.1f%% |"
          % (100.0 * sum(1 for x in con if x > th.tau_c) / n,
             100.0 * sum(1 for x in agr if x > th.tau_a) / n))
    print("| **교란** | **%.1f%%** | **%.1f%%** | (기준 둘 다 <=5%%)"
          % (nc["contradiction_gt_tau_c_pct"], nc["agreement_gt_tau_a_pct"]))
    print("")
    print("판정: %s" % ("PASS" if nc["pass"] else "**FAIL**"))
    print(nc["note"])
    print("")
    print("LLM 호출 %d (캐시 %d), 청구 %d/%d"
          % (llm.stats.calls, llm.stats.cache_hits,
             llm.stats.billed_tokens_in, llm.stats.billed_tokens_out))
    print("-> " + str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
