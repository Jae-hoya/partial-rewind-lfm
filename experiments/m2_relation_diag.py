"""M2 교란 통제 실패의 **원인 진단**. 임계는 건드리지 않는다. 호출 0회.

SPEC_M2 §2: "둘 다 <=5% 가 아니면 tau 를 올리는 게 아니라 그 사실을 보고하고
Test C-1 결과 해석에 반영한다."

그래서 여기서 하는 것은 **왜 실패했는지**를 재는 것뿐이다. 고치지 않는다.
고칠지 말지, 어떻게 고칠지는 스펙 개정 사항이고 공저자가 정한다.

묻는 것 셋:

  A. 정상과 교란의 **분포 자체**가 같은가? 최댓값만 같고 분포는 다를 수도 있다.
  B. **pair 단위**에는 신호가 있는데 max 가 죽인 것인가?
     max 는 pair 수가 많을수록 위로 치우친다 — D8-A 가 밝힌 그 성질이다.
  C. 신호가 **pair 수와 상관**돼 있는가? 그렇다면 팽창이 원인이다.

C 가 크고 B 가 살아 있으면 "관점 간 충돌이 없다"가 아니라 "max 집계로는 못 잡는다"
가 된다. 두 진술은 논문에서 전혀 다른 문장이다.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics as st
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from d8a_negative_control import SHIFT, spearman  # noqa: E402
from diag_sweep import CONDITIONS, pick_topics  # noqa: E402
from m2_relation_run import MAX_CLAIMS, RELATION_CK  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.relation import RelationExtractor, Thresholds  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402


def q(v, p):
    return st.quantiles(v, n=100)[p - 1] if len(v) > 2 else (max(v) if v else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description="M2 교란 통제 실패 진단 (호출 0회)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--out", default="results/m2/relation/diagnosis.json")
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
    nli = load_nli(Config(d2, source="m2:diag"))
    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))

    print("claim 수집 (캐시 재생) ...")
    claims: dict[str, dict[str, list[str]]] = {}
    for t in picked:
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        claims[tid] = {}
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            c = [x for x in o.claims if x.strip()][:MAX_CLAIMS]
            if c:
                claims[tid][p.role] = c

    def nli3(pairs):
        return [(s.entailment, s.contradiction) for s in nli.score(list(pairs))]

    rx = RelationExtractor(nli3, th=Thresholds())
    th = rx.th
    tids = sorted(claims)
    mapping = {tids[k]: tids[(k + SHIFT) % len(tids)] for k in range(len(tids))}

    # 정상 / 교란 각각에서 pair 단위 점수를 모은다
    def collect(pairs_of):
        pair_con, edge_con, npairs = [], [], []
        for src, roles in pairs_of:
            for ci, cj in roles:
                ps = rx.pair_scores(ci, cj)
                if not ps:
                    continue
                pair_con.extend(p.con for p in ps)
                edge_con.append(max(p.con for p in ps))
                npairs.append(len(ps))
        return pair_con, edge_con, npairs

    normal_src = []
    for tid in tids:
        roles = sorted(claims[tid])
        normal_src.append((tid, [(claims[tid][a], claims[tid][b])
                                 for a, b in combinations(roles, 2)]))
    shuf_src = []
    for tid in tids:
        dst = mapping[tid]
        shuf_src.append((tid, [(ci, cj) for ci in claims[tid].values()
                               for cj in claims[dst].values()]))

    print("정상 pair 채점 ...", flush=True)
    pc_n, ec_n, np_n = collect(normal_src)
    print("교란 pair 채점 ...", flush=True)
    pc_s, ec_s, np_s = collect(shuf_src)

    out = {
        "checkpoint": RELATION_CK, "tau_c": th.tau_c,
        "A_edge_level_max": {
            "normal": {"n": len(ec_n), "median": round(st.median(ec_n), 4),
                       "gt_tau_c_pct": round(100.0 * sum(1 for x in ec_n if x > th.tau_c)
                                             / len(ec_n), 2)},
            "shuffled": {"n": len(ec_s), "median": round(st.median(ec_s), 4),
                         "gt_tau_c_pct": round(100.0 * sum(1 for x in ec_s if x > th.tau_c)
                                               / len(ec_s), 2)}},
        "B_pair_level": {
            "normal": {"n": len(pc_n), "median": round(st.median(pc_n), 6),
                       "mean": round(st.mean(pc_n), 6),
                       "p95": round(q(pc_n, 95), 6),
                       "gt_tau_c_pct": round(100.0 * sum(1 for x in pc_n if x > th.tau_c)
                                             / len(pc_n), 3)},
            "shuffled": {"n": len(pc_s), "median": round(st.median(pc_s), 6),
                         "mean": round(st.mean(pc_s), 6),
                         "p95": round(q(pc_s, 95), 6),
                         "gt_tau_c_pct": round(100.0 * sum(1 for x in pc_s if x > th.tau_c)
                                               / len(pc_s), 3)}},
        "C_inflation": {
            "rho_npairs_vs_edgemax_normal": round(
                spearman([float(x) for x in np_n], ec_n), 4),
            "rho_npairs_vs_edgemax_shuffled": round(
                spearman([float(x) for x in np_s], ec_s), 4),
            "npairs_median_normal": st.median(np_n),
            "npairs_median_shuffled": st.median(np_s)},
    }

    p = root / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    io.open(p, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))

    a, b = out["A_edge_level_max"]["normal"], out["A_edge_level_max"]["shuffled"]
    c, d = out["B_pair_level"]["normal"], out["B_pair_level"]["shuffled"]
    print("")
    print("=== A. edge 단위 (max 집계) — 스펙이 쓰는 값 ===")
    print("  정상 n=%d 중앙 %.4f  >tau_c %.2f%%" % (a["n"], a["median"], a["gt_tau_c_pct"]))
    print("  교란 n=%d 중앙 %.4f  >tau_c %.2f%%" % (b["n"], b["median"], b["gt_tau_c_pct"]))
    print("")
    print("=== B. pair 단위 — max 가 죽였는지 본다 ===")
    print("  정상 n=%d 중앙 %.6f 평균 %.6f p95 %.4f  >tau_c %.3f%%"
          % (c["n"], c["median"], c["mean"], c["p95"], c["gt_tau_c_pct"]))
    print("  교란 n=%d 중앙 %.6f 평균 %.6f p95 %.4f  >tau_c %.3f%%"
          % (d["n"], d["median"], d["mean"], d["p95"], d["gt_tau_c_pct"]))
    print("  **pair 단위 차이 %+.3f%%p** (정상 - 교란)"
          % (c["gt_tau_c_pct"] - d["gt_tau_c_pct"]))
    print("")
    print("=== C. 팽창 — pair 수와 edge 최댓값의 상관 ===")
    print("  정상 rho %+.4f (pair 중앙 %d) · 교란 rho %+.4f (pair 중앙 %d)"
          % (out["C_inflation"]["rho_npairs_vs_edgemax_normal"],
             out["C_inflation"]["npairs_median_normal"],
             out["C_inflation"]["rho_npairs_vs_edgemax_shuffled"],
             out["C_inflation"]["npairs_median_shuffled"]))
    print("")
    print("해석은 하지 않는다. 임계도 바꾸지 않는다. 이 표를 공저자에게 보낸다.")
    print("-> " + str(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
