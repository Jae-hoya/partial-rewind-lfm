"""D20 — 모순 통제 쌍(N1/P1/P2)의 다중 체크포인트 오프라인 재채점. **API 호출 0회.**

사전 등록: `results/d20/PREREG_D20.md` (시작 시 SHA-256 을 계산해 요약에 기록한다. 파일이 없으면 멈춘다).

  N1  `m2_gate_n1.py` 의 묶음 루프를 **cache_only** 로 재생해 쌍 텍스트를 복원한다.
      `.env` 를 읽지 않는다. CacheMiss 는 잡지 않는다 — 미스면 실행이 죽는다.
  P1/P2  `results/m2/relation/gate_p1p2.csv` 의 텍스트를 그대로 쓴다.

재현 검증(bart 로 gate_n1.csv / gate_p1p2.csv 대조)을 먼저 하고, 실패하면 결과에 기록한다.

  python experiments/d20_contradiction_rescore.py
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.pop("OPENAI_API_KEY", None)      # 키가 프로세스에 없으면 어떤 경로로도 API 를 못 부른다

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import statistics as st  # noqa: E402
import sys  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from diag_sweep import CONDITIONS, pick_topics  # noqa: E402
from m2_relation_run import MAX_CLAIMS  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM  # noqa: E402
from lfm.nli import NLI  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.relation_v11 import attribute_claims  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402

CKPTS_3WAY = [
    "facebook/bart-large-mnli",
    "roberta-large-mnli",
    "cross-encoder/nli-deberta-v3-large",
    "cross-encoder/nli-deberta-v3-base",
]
MINICHECK = "lytang/MiniCheck-Flan-T5-Large"
FIXED = (0.70, 0.95)
SWEEP = [round(0.50 + 0.01 * k, 2) for k in range(50)]   # 0.50 .. 0.99
AGGS = ("max", "min", "mean")
GATE_P1 = 80.0
GATE_N1 = 5.0


def agg(c1: float, c2: float, how: str) -> float:
    if how == "max":
        return max(c1, c2)
    if how == "min":
        return min(c1, c2)
    return (c1 + c2) / 2.0


def pct(xs, t):
    return 100.0 * sum(1 for x in xs if x > t) / len(xs) if xs else None


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def replay_n1(root: Path, condition: str, n_topics: int) -> tuple[list[dict], dict]:
    cfg = load_config(root / "config.yaml")
    spec = CONDITIONS[condition]
    data = cfg.as_dict()
    if spec["top_k"] is not None:
        data["retrieval"]["top_k"] = spec["top_k"]
    if spec["index"] == "partial":
        data["paths"]["index_dir"] = data["paths"]["index_dir_partial"]
    data["llm"]["cache_policy"] = "cache_only"          # 미스면 CacheMiss — API 를 부르지 않는다
    ccfg = Config(data, source="d20:" + condition)
    llm = LLM(ccfg)
    assert llm.cache_policy == "cache_only"
    retriever = BM25Retriever(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, n_topics, int(cfg["seed"]))

    pairs = []
    n_groups = 0
    for t in picked:
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            claims = [c for c in o.claims if c.strip()][:MAX_CLAIMS]
            subs = [sa.text for sa in o.sub_answers]
            if not claims or not subs:
                continue
            attr = attribute_claims(claims, subs)
            by_q: dict[int, list[int]] = {}
            for k, q in enumerate(attr):
                if q >= 0:
                    by_q.setdefault(q, []).append(k)
            for q, idxs in by_q.items():
                if len(idxs) >= 2:
                    n_groups += 1
                    cl = [claims[k] for k in idxs]
                    for a, b in combinations(range(len(cl)), 2):
                        pairs.append({"set": "N1", "topic": tid, "branch": p.role, "subq": q,
                                      "a": a, "b": b, "text_a": cl[a], "text_b": cl[b]})
        print("  replay %s  누적 쌍 %d" % (tid, len(pairs)), flush=True)
    stats = {"api_calls": llm.stats.api_calls,
             "calls_total": llm.stats.calls, "cache_hits": llm.stats.cache_hits,
             "billed_in": llm.stats.billed_tokens_in, "billed_out": llm.stats.billed_tokens_out,
             "n_groups": n_groups}
    return pairs, stats


def load_p(root: Path) -> list[dict]:
    out = []
    with io.open(root / "results/m2/relation/gate_p1p2.csv", encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if r["neg_usable"] == "1":
                out.append({"set": "P1", "topic": r["topic"], "branch": r["branch"], "subq": "",
                            "a": i, "b": "neg", "text_a": r["claim"], "text_b": r["negation"],
                            "orig": float(r["p1_contradiction"])})
            if r["par_usable"] == "1":
                out.append({"set": "P2", "topic": r["topic"], "branch": r["branch"], "subq": "",
                            "a": i, "b": "par", "text_a": r["claim"], "text_b": r["paraphrase"]})
    return out


class _Cfg:
    def __init__(self, d):
        self.d = d

    def get(self, k, default=None):
        return self.d.get(k, default)


def score_3way(ck: str, pairs: list[dict]) -> list[tuple]:
    nli = NLI(_Cfg({"nli.model": ck, "nli.batch_size": 16, "nli.max_length": 512, "nli.device": "auto"}))
    flat = []
    for r in pairs:
        flat += [(r["text_a"], r["text_b"]), (r["text_b"], r["text_a"])]
    sc = nli.score(flat)
    out = [(sc[2 * i], sc[2 * i + 1]) for i in range(len(pairs))]
    dev = nli.device
    del nli
    import torch
    torch.cuda.empty_cache()
    return out, dev


def score_minicheck(pairs: list[dict]) -> list[tuple]:
    from minicheck.minicheck import MiniCheck
    m = MiniCheck(model_name="flan-t5-large", max_model_len=2048, batch_size=8)
    docs, claims = [], []
    for r in pairs:
        docs += [r["text_a"], r["text_b"]]
        claims += [r["text_b"], r["text_a"]]
    _, probs, _, _ = m.score(docs=docs, claims=claims)
    return [(float(probs[2 * i]), float(probs[2 * i + 1])) for i in range(len(pairs))]


def main(argv=None):
    ap = argparse.ArgumentParser(description="D20 모순 통제 다중 체크포인트 재채점 (API 0)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--no-minicheck", action="store_true")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    outd = root / "results/d20"
    prereg = outd / "PREREG_D20.md"
    if not prereg.exists():
        raise SystemExit("PREREG_D20.md 가 없다. 사전 등록 없이 채점하지 않는다.")
    prereg_sha = sha256(prereg)
    print("PREREG sha256 =", prereg_sha)

    print("N1 재생 (cache_only) ...", flush=True)
    n1, replay_stats = replay_n1(root, args.condition, args.n_topics)
    pp = load_p(root)
    allp = n1 + pp
    with io.open(outd / "n1_pairs.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["topic", "branch", "subq", "a", "b", "text_a", "text_b"])
        w.writeheader()
        for r in n1:
            w.writerow({k: r[k] for k in w.fieldnames})
    print("N1 %d · P1 %d · P2 %d" % (len(n1), sum(r["set"] == "P1" for r in pp),
                                    sum(r["set"] == "P2" for r in pp)))

    import torch
    import transformers
    env = {"torch": torch.__version__, "transformers": transformers.__version__,
           "cuda": torch.cuda.is_available(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    scores: dict[str, list] = {}
    for ck in CKPTS_3WAY:
        print("채점 %s ..." % ck, flush=True)
        s, dev = score_3way(ck, allp)
        scores[ck] = s
        env["device_" + ck] = dev
    mc = None
    if not args.no_minicheck:
        print("채점 %s (보조) ..." % MINICHECK, flush=True)
        mc = score_minicheck(allp)

    # --- 원자료 CSV
    with io.open(outd / "d20_scores.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "topic", "branch", "subq", "a", "b", "checkpoint", "direction",
                    "contradiction", "entailment", "neutral", "support"])
        for i, r in enumerate(allp):
            for ck in CKPTS_3WAY:
                for d, s in zip(("ab", "ba"), scores[ck][i]):
                    w.writerow([r["set"], r["topic"], r["branch"], r["subq"], r["a"], r["b"], ck, d,
                                round(s.contradiction, 6), round(s.entailment, 6),
                                round(s.neutral, 6), ""])
            if mc is not None:
                for d, v in zip(("ab", "ba"), mc[i]):
                    w.writerow([r["set"], r["topic"], r["branch"], r["subq"], r["a"], r["b"],
                                MINICHECK, d, "", "", "", round(v, 6)])

    idx = {s: [i for i, r in enumerate(allp) if r["set"] == s] for s in ("N1", "P1", "P2")}

    # --- 재현 검증 (bart)
    orig = {}
    with io.open(root / "results/m2/relation/gate_n1.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            orig[(r["topic"], r["branch"], int(r["subq"]), int(r["a"]), int(r["b"]))] = float(r["contradiction"])
    bart = scores["facebook/bart-large-mnli"]
    newk = {}
    for i in idx["N1"]:
        r = allp[i]
        newk[(r["topic"], r["branch"], int(r["subq"]), int(r["a"]), int(r["b"]))] = \
            max(bart[i][0].contradiction, bart[i][1].contradiction)
    common = set(orig) & set(newk)
    diffs = [abs(orig[k] - newk[k]) for k in common]
    p1_d = [abs(allp[i]["orig"] - max(bart[i][0].contradiction, bart[i][1].contradiction))
            for i in idx["P1"]]
    repro = {
        "n1_orig_pairs": len(orig), "n1_new_pairs": len(newk), "n1_keys_matched": len(common),
        "n1_keys_only_orig": len(set(orig) - set(newk)), "n1_keys_only_new": len(set(newk) - set(orig)),
        "n1_max_abs_diff": max(diffs) if diffs else None,
        "n1_orig_gt070_pct": pct(list(orig.values()), 0.70),
        "n1_new_gt070_pct": pct(list(newk.values()), 0.70),
        "p1_n": len(p1_d), "p1_max_abs_diff": max(p1_d) if p1_d else None,
    }
    repro["pass"] = bool(len(orig) == len(newk) == len(common) and diffs and max(diffs) <= 1e-3
                         and p1_d and max(p1_d) <= 1e-3)
    print("재현:", json.dumps(repro, ensure_ascii=False))

    # --- 구성별 양성률
    per_ck = {}
    usable = []
    for ck in CKPTS_3WAY:
        per_ck[ck] = {}
        for how in AGGS:
            v = {s: [agg(scores[ck][i][0].contradiction, scores[ck][i][1].contradiction, how)
                     for i in idx[s]] for s in idx}
            fixed = {("%.2f" % t): {s: round(pct(v[s], t), 4) for s in v} for t in FIXED}
            sweep = {("%.2f" % t): {s: round(pct(v[s], t), 4) for s in v} for t in SWEEP}
            for t, d in sweep.items():
                if d["P1"] >= GATE_P1 and d["N1"] <= GATE_N1:
                    usable.append({"checkpoint": ck, "agg": how, "tau": t, **d})
            cand = [(d["N1"], t, d) for t, d in sweep.items() if d["P1"] >= GATE_P1]
            best = min(cand) if cand else None
            per_ck[ck][how] = {
                "fixed": fixed,
                "sweep": sweep,
                "median": {s: round(st.median(v[s]), 6) for s in v},
                "best_under_P1ge80": ({"tau": best[1], **best[2]} if best else None),
            }

    # --- MiniCheck 기술 통계
    mc_desc = None
    if mc is not None:
        mc_desc = {}
        for s in idx:
            mx = [max(mc[i]) for i in idx[s]]
            mn = [min(mc[i]) for i in idx[s]]
            mc_desc[s] = {"n": len(mx),
                          "median_max_support": round(st.median(mx), 6),
                          "median_min_support": round(st.median(mn), 6),
                          "pct_max_support_le_0.5": round(100.0 * sum(1 for x in mx if x <= 0.5) / len(mx), 4)}

    # --- 체크포인트 간 N1 플래그 일치 (0.95, max)
    flags = {ck: {i for i in idx["N1"]
                  if max(scores[ck][i][0].contradiction, scores[ck][i][1].contradiction) > 0.95}
             for ck in CKPTS_3WAY}
    jac = {}
    for x, y in combinations(CKPTS_3WAY, 2):
        u = flags[x] | flags[y]
        jac["%s | %s" % (x, y)] = round(len(flags[x] & flags[y]) / len(u), 4) if u else None
    allc = set.intersection(*flags.values())
    anyc = set.union(*flags.values())
    with io.open(outd / "n1_consensus_flagged.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["topic", "branch", "subq", "a", "b"] + ["con_max:" + ck for ck in CKPTS_3WAY]
                   + ["text_a", "text_b"])
        for i in sorted(allc):
            r = allp[i]
            w.writerow([r["topic"], r["branch"], r["subq"], r["a"], r["b"]]
                       + [round(max(scores[ck][i][0].contradiction, scores[ck][i][1].contradiction), 6)
                          for ck in CKPTS_3WAY] + [r["text_a"], r["text_b"]])
    overlap = {"tau": 0.95, "agg": "max",
               "n_flagged": {ck: len(flags[ck]) for ck in CKPTS_3WAY},
               "n_flagged_by_all": len(allc), "n_flagged_by_any": len(anyc),
               "jaccard": jac}

    summary = {
        "prereg": str(prereg.relative_to(root)).replace("\\", "/"), "prereg_sha256": prereg_sha,
        "condition": args.condition, "n_topics": args.n_topics,
        "n": {s: len(idx[s]) for s in idx}, "replay": replay_stats, "env": env,
        "reproduction_bart": repro,
        "gates": {"P1_recall_ge": GATE_P1, "N1_rate_le": GATE_N1},
        "usable_configs": usable,
        "generalizes_no_usable_config": len(usable) == 0,
        "per_checkpoint": per_ck,
        "minicheck_descriptive": mc_desc,
        "n1_flag_overlap": overlap,
    }
    io.open(outd / "d20_summary.json", "w", encoding="utf-8").write(
        json.dumps(summary, ensure_ascii=False, indent=1))

    print("")
    print("=== D20 고정 임계 (주 집계 max) — N1 / P1 / P2 양성률 % ===")
    for ck in CKPTS_3WAY:
        f = per_ck[ck]["max"]["fixed"]
        print("  %-38s 0.70: %6.2f / %6.2f / %6.2f   0.95: %6.2f / %6.2f / %6.2f"
              % (ck, f["0.70"]["N1"], f["0.70"]["P1"], f["0.70"]["P2"],
                 f["0.95"]["N1"], f["0.95"]["P1"], f["0.95"]["P2"]))
    print("쓸 만한 구성 수:", len(usable))
    print("N1 플래그 일치:", json.dumps(overlap, ensure_ascii=False))
    if mc_desc:
        print("MiniCheck:", json.dumps(mc_desc, ensure_ascii=False))
    print("-> " + str(outd / "d20_summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
