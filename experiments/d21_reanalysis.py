"""D21 — 기존 파일 재분석 두 건. **API/LLM 호출 0, 캐시 재생 없음.**

(A) R1-5a: N1 모순 양성률을 pair / claim / bundle 단위로 다시 센다 (topic 군집 bootstrap).
(B) R3: 빈(degraded) branch 처리 (a)/(b)/(c)/(c') 에 따른 coverage·grounding·verify·sufficient·W0.8
    (query 군집 bootstrap).

사전 등록: results/d21/PREREG_D21.md (시작 시 SHA-256 기록).
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "d21"
PREREG = OUT / "PREREG_D21.md"
B = 2000
SEED = 42
TAUS_A = [0.70, 0.95]
CKS = ["facebook/bart-large-mnli", "roberta-large-mnli",
       "cross-encoder/nli-deberta-v3-large", "cross-encoder/nli-deberta-v3-base"]
RUNS = {
    "g1": ROOT / "results/test_a/20260829-041445",
    "minicheck": ROOT / "results/m1prime/seed42",
}
TAU_V = 0.8  # verify (>=) 와 sufficient (>) 모두 0.8


# --- bootstrap -------------------------------------------------------------


def boot_draws(n_clusters: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.integers(0, n_clusters, size=(B, n_clusters))


def ratio_ci(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    d = boot_draws(len(num))
    ns, ds = num[d].sum(axis=1), den[d].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(ds > 0, ns / ds, np.nan)
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return float(lo), float(hi)


def count_ci(cnt: np.ndarray) -> tuple[float, float]:
    d = boot_draws(len(cnt))
    s = cnt[d].sum(axis=1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(lo), float(hi)


def rnd(x, k=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


# --- A ---------------------------------------------------------------------


def part_a() -> dict:
    sc = pd.read_csv(ROOT / "results/d20/d20_scores.csv", low_memory=False)
    sc = sc[sc["set"] == "N1"].copy()
    sc["subq"] = sc["subq"].astype(int)
    sc["b"] = sc["b"].astype(int)
    pairs = pd.read_csv(ROOT / "results/d20/n1_pairs.csv")
    key = ["topic", "branch", "subq", "a", "b"]
    wide = (sc.pivot_table(index=key + ["checkpoint"], columns="direction",
                           values="contradiction").reset_index())
    wide["max"] = wide[["ab", "ba"]].max(axis=1)

    # 선행 재현
    orig = pd.read_csv(ROOT / "results/m2/relation/gate_n1.csv")
    bart = wide[wide.checkpoint == CKS[0]][key + ["max"]]
    j = orig.merge(bart, on=key, how="outer", indicator=True)
    repro = {
        "n_orig": int(len(orig)), "n_new": int(len(bart)),
        "matched": int((j["_merge"] == "both").sum()),
        "max_abs_diff": float((j["contradiction"] - j["max"]).abs().max()),
        "orig_gt070": int((orig["contradiction"] > 0.70).sum()),
        "new_gt070": int((bart["max"] > 0.70).sum()),
    }
    repro["pass"] = bool(repro["n_orig"] == repro["n_new"] == repro["matched"] == 823
                         and repro["max_abs_diff"] <= 1e-3
                         and repro["orig_gt070"] == repro["new_gt070"] == 217)
    if not repro["pass"]:
        print("A 재현 실패:", repro)
    assert len(pairs) == 823 and pairs.merge(bart, on=key).shape[0] == 823

    topics = sorted(pairs.topic.unique())
    tix = {t: i for i, t in enumerate(topics)}
    bundles = pairs.groupby(["topic", "branch", "subq"]).size().reset_index(name="n_pairs")
    claims_tbl = pd.concat([
        pairs[["topic", "branch", "subq", "a"]].rename(columns={"a": "idx"}),
        pairs[["topic", "branch", "subq", "b"]].rename(columns={"b": "idx"}),
    ]).drop_duplicates()
    bsize = claims_tbl.groupby(["topic", "branch", "subq"]).size()
    # 크기 k 묶음은 k(k-1)/2 쌍 — 일관성 확인
    chk = bundles.set_index(["topic", "branch", "subq"])["n_pairs"]
    assert (chk == bsize * (bsize - 1) // 2).all()

    res = {"reproduction_bart": repro, "n_topics": len(topics),
           "n_pairs": int(len(pairs)), "n_claims": int(len(claims_tbl)),
           "n_bundles": int(len(bundles)),
           "bundle_size_dist": {int(k): int(v) for k, v in bsize.value_counts().sort_index().items()},
           "per_checkpoint": {}}

    for ck in CKS:
        w = wide[wide.checkpoint == ck][key + ["max"]]
        p = pairs[key].merge(w, on=key)
        assert len(p) == 823
        out_ck = {}
        for tau in TAUS_A:
            p["pos"] = (p["max"] > tau).astype(int)
            # claim
            cl = pd.concat([
                p[["topic", "branch", "subq", "a", "pos"]].rename(columns={"a": "idx"}),
                p[["topic", "branch", "subq", "b", "pos"]].rename(columns={"b": "idx"}),
            ]).groupby(["topic", "branch", "subq", "idx"])["pos"].max().reset_index()
            bu = p.groupby(["topic", "branch", "subq"])["pos"].max().reset_index()
            units = {}
            for name, df in (("pair", p), ("claim", cl), ("bundle", bu)):
                g = df.groupby("topic")["pos"].agg(["sum", "count"]).reindex(topics).fillna(0)
                num, den = g["sum"].to_numpy(float), g["count"].to_numpy(float)
                lo, hi = ratio_ci(num, den)
                units[name] = {"k": int(num.sum()), "n": int(den.sum()),
                               "rate_pct": rnd(100 * num.sum() / den.sum(), 3),
                               "ci95_pct": [rnd(100 * lo, 3), rnd(100 * hi, 3)]}
            # 묶음 크기별 bundle 양성률 (기계적 상승 확인용)
            bu2 = bu.merge(bsize.rename("size").reset_index(), on=["topic", "branch", "subq"])
            units["bundle_by_size"] = {int(s): {"k": int(g.pos.sum()), "n": int(len(g))}
                                       for s, g in bu2.groupby("size")}
            out_ck["%.2f" % tau] = units
        res["per_checkpoint"][ck] = out_ck
    # 사후 추가 (사전 등록 외): 단독 claim 묶음까지 분모에 넣은 점추정. 토픽별 단독 수가 없어 CI 없음.
    attr = json.load(io.open(ROOT / "results/m2/relation/gate_n1.json", encoding="utf-8"))["attribution"]
    single = attr["attributed"] - res["n_claims"]
    post = {"source": "results/m2/relation/gate_n1.json attribution", "attributed": attr["attributed"],
            "failed": attr["failed"], "singleton_claims_eq_bundles": single, "per_tau_bart": {}}
    for tau, u in res["per_checkpoint"][CKS[0]].items():
        post["per_tau_bart"][tau] = {
            "claim_all_attributed": [u["claim"]["k"], attr["attributed"],
                                     rnd(100 * u["claim"]["k"] / attr["attributed"], 3)],
            "bundle_incl_singletons": [u["bundle"]["k"], res["n_bundles"] + single,
                                       rnd(100 * u["bundle"]["k"] / (res["n_bundles"] + single), 3)]}
    res["post_hoc_denominators"] = post
    return res


# --- B ---------------------------------------------------------------------


def treat(d: pd.DataFrame, name: str) -> pd.DataFrame:
    """처리별로 cov/grd 의 정의 여부 열(cov_def, grd_def)과 모집단 포함(inpop)을 만든다."""
    d = d.copy()
    d["cov_v"], d["grd_v"] = d["coverage"].astype(float), d["grounding"].astype(float)
    d["cov_def"], d["grd_def"], d["inpop"] = True, True, True
    deg = d["degraded"] == 1
    z = d["coverage"] == 0
    if name == "a":
        pass
    elif name == "b215":
        d.loc[deg, ["cov_def", "grd_def", "inpop"]] = False
    elif name == "b219":
        d.loc[z, ["cov_def", "grd_def", "inpop"]] = False
    elif name == "c":
        d.loc[deg, "cov_def"] = False
    elif name == "c_prime":
        d.loc[deg, ["cov_def", "grd_def"]] = False
    else:
        raise ValueError(name)
    # sufficient: 정의된 값으로만, 미정의면 False (엄격 >)
    d["suff"] = (d.grd_def & d.cov_def & (d.grd_v > TAU_V) & (d.cov_v > TAU_V)).astype(int)
    return d


def metrics(d: pd.DataFrame, qids: list[str], rounds=None) -> dict:
    x = d if rounds is None else d[d["round"].isin(rounds)]
    out = {}
    # 평균 (정의된 행만)
    for col, dc, nm in (("cov_v", "cov_def", "coverage_mean"), ("grd_v", "grd_def", "grounding_mean")):
        y = x[x[dc]]
        g = y.groupby("query_id")[col].agg(["sum", "count"]).reindex(qids).fillna(0)
        num, den = g["sum"].to_numpy(float), g["count"].to_numpy(float)
        lo, hi = ratio_ci(num, den)
        out[nm] = {"value": rnd(num.sum() / den.sum()), "n_rows": int(den.sum()),
                   "ci95": [rnd(lo), rnd(hi)]}
    # verify (query-round)
    vr = []
    for (q, r), g in x.groupby(["query_id", "round"]):
        gd = g[g.grd_def]["grd_v"]
        cd = g[g.cov_def]["cov_v"]
        undefined = gd.empty or cd.empty
        pg = (not gd.empty) and gd.mean() >= TAU_V
        pc = (not cd.empty) and cd.min() >= TAU_V
        vr.append({"query_id": q, "round": r, "pass": int(pg and pc), "pass_g": int(pg),
                   "pass_c": int(pc), "undef": int(undefined),
                   "n_cov_def": int(len(cd)), "n_grd_def": int(len(gd)),
                   "mean_g": None if gd.empty else float(gd.mean()),
                   "min_c": None if cd.empty else float(cd.min())})
    vr = pd.DataFrame(vr)
    for c in ("pass", "pass_g", "pass_c", "undef"):
        cnt = vr.groupby("query_id")[c].sum().reindex(qids).fillna(0).to_numpy(float)
        lo, hi = count_ci(cnt)
        out["verify_" + c] = {"k": int(cnt.sum()), "n_rounds": int(len(vr)), "ci95": [lo, hi]}
    out["verify_best_mean_g"] = rnd(vr["mean_g"].dropna().max())
    out["verify_pass_rounds"] = vr[vr["pass"] == 1][
        ["query_id", "round", "n_cov_def", "n_grd_def", "mean_g", "min_c"]].to_dict(orient="records")
    # sufficient 개수 (모집단 행 중)
    pop = x[x.inpop]
    cnt = pop.groupby("query_id")["suff"].sum().reindex(qids).fillna(0).to_numpy(float)
    lo, hi = count_ci(cnt)
    out["sufficient@0.8"] = {"k": int(cnt.sum()), "n_rows": int(len(pop)), "ci95": [lo, hi]}
    # W0.8 — waste_by_query 와 같은 정의 (모집단 안의 re_executed 행)
    re = pop[pop.re_executed == 1]
    g = re.groupby("query_id")["suff"].agg(["sum", "count"]).reindex(qids).fillna(0)
    num, den = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    lo, hi = ratio_ci(num, den)
    out["W0.8"] = {"k": int(num.sum()), "n": int(den.sum()),
                   "value_pct": rnd(100 * num.sum() / den.sum(), 3) if den.sum() else None,
                   "ci95_pct": [rnd(100 * lo, 3), rnd(100 * hi, 3)]}
    return out


def part_b() -> dict:
    res = {}
    frames = {}
    for run, path in RUNS.items():
        d = pd.read_csv(path / "waste_ratio.csv")
        ctx = json.load(io.open(path / "results.json", encoding="utf-8"))["context"]
        frames[run] = d
        qids = sorted(d.query_id.unique())
        assert len(d) == 600 and len(qids) == 50
        deg, z = d.degraded == 1, d.coverage == 0
        sets = {
            "D215": int(deg.sum()), "Z219": int(z.sum()),
            "D215_subset_Z219": bool((~deg | z).all()),
            "extra_nondegraded_cov0": d[z & ~deg][["query_id", "round", "branch", "grounding"]]
            .to_dict(orient="records"),
            "degraded_by_round": {int(k): int(v) for k, v in d[deg].groupby("round").size().items()},
            "degraded_reexecuted": int((deg & (d.re_executed == 1)).sum()),
            "degraded_grounding_max": float(d[deg].grounding.max()),
        }
        r = {"context": {"estimator": ctx.get("estimator", "(미기재)"),
                         "nli_model": ctx.get("nli_model")},
             "sets": sets, "treatments": {}}
        for t in ("a", "b215", "b219", "c", "c_prime"):
            dt = treat(d, t)
            r["treatments"][t] = {"all_rounds": metrics(dt, qids),
                                  "round0": metrics(dt, qids, rounds=[0])}
        res[run] = r
    # 두 실행의 coverage/degraded/re_executed 동일 여부
    k = ["query_id", "round", "branch"]
    m = frames["g1"].merge(frames["minicheck"], on=k)
    res["runs_identical_except_grounding"] = bool(
        len(m) == 600 and (m.coverage_x == m.coverage_y).all()
        and (m.degraded_x == m.degraded_y).all() and (m.re_executed_x == m.re_executed_y).all())
    # 고정점
    g1a = res["g1"]["treatments"]["a"]["all_rounds"]
    mca = res["minicheck"]["treatments"]["a"]["all_rounds"]
    g1r0 = res["g1"]["treatments"]["a"]["round0"]
    res["anchors"] = {
        "g1_W08_2of400": g1a["W0.8"]["k"] == 2 and g1a["W0.8"]["n"] == 400,
        "g1_suff_3of600": g1a["sufficient@0.8"]["k"] == 3,
        "g1_verify_0of150": g1a["verify_pass"]["k"] == 0 and g1a["verify_pass"]["n_rounds"] == 150,
        "mc_W08_34of400": mca["W0.8"]["k"] == 34 and mca["W0.8"]["n"] == 400,
        "round0_degraded_67of200": res["g1"]["sets"]["degraded_by_round"].get(0) == 67,
        # 원고 :557 의 0.474 는 :262 의 15-토픽 진단 baseline 값과 같다 — 50-토픽 round 0 값과 대조만 한다
        "round0_cov_mean_0.474": round(g1r0["coverage_mean"]["value"], 3) == 0.474,
        "round0_cov_mean_observed": g1r0["coverage_mean"]["value"],
    }
    return res


def main() -> int:
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    print("PREREG sha256", sha)
    a = part_a()
    b = part_b()
    summ = {"prereg": "results/d21/PREREG_D21.md", "prereg_sha256": sha,
            "api_calls": 0, "bootstrap": {"B": B, "seed": SEED, "method": "cluster percentile"},
            "A_R1_5a": a, "B_R3": b}
    OUT.mkdir(parents=True, exist_ok=True)
    io.open(OUT / "d21_summary.json", "w", encoding="utf-8").write(
        json.dumps(summ, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({"A_repro": a["reproduction_bart"], "anchors": b["anchors"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
