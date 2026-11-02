"""R2 재실험 드라이버 — arm × seed 를 토픽 단위로 돌린다. 재실험설계_확정안_R2 §1, §5, §6.

  python experiments/r2_run.py --arms L-BR L-GR E-SQ E-BR E-GR --seeds 1 --topics pilot5
  python experiments/r2_run.py --replay-testa --topics pilot5     # P0: Test A 를 cache_only 로 재생

arm 은 같은 seed 안에서 정해진 순서(L-BR → L-GR → E-SQ → E-BR → E-GR)로 돈다. 같은 family 의 BR 이
다시 돌리는 branch 는 GR 과 nonce 가 같아 바이트가 같다(공통 난수, 확정안 §1). 축소 컨트롤러의 호출이
실제 청구가 되도록 먼저 돈다. E-R1 은 실행하지 않는다 — E-GR 의 라운드 0 출력이 곧 E-R1 이다.

산출: results/r2/<run_tag>/<arm>/s<seed>/topics/<qid>.json  (토픽 단위 체크포인트 — 있으면 건너뛴다)
      .../trace.jsonl, usage.json, config.yaml
분해는 모든 arm·seed 에서 Test A 캐시(.cache/llm, 읽기 전용)를 그대로 재생한다.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_sweep import pick_topics  # noqa: E402

from lfm.config import Config, load_config  # noqa: E402
from lfm.controller import FreezeRetry, GlobalRetry  # noqa: E402
from lfm.estimator import MemoEstimator, load_estimator  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.pipeline import run_query  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.trace import TraceWriter, new_run_id  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data/split_r2.json"

# arm -> (family, prompt_mode, controller, granularity)
ARMS = {
    "L-GR": ("L", "legacy", "global_retry", "branch"),
    "L-BR": ("L", "legacy", "freeze_retry", "branch"),
    "E-GR": ("E", "status", "global_retry", "branch"),
    "E-SQ": ("E", "status", "freeze_retry", "subq"),
    "E-BR": ("E", "status", "freeze_retry", "branch"),
}
ORDER = ["L-BR", "L-GR", "E-SQ", "E-BR", "E-GR"]


def make_split(topics: list[dict]) -> dict:
    """dev15 = D16 과 같은 15토픽, pilot5 ⊂ dev15 (seed 42), heldout35 = 나머지. 한 번 만들면 고정."""
    if SPLIT.exists():
        return json.load(io.open(SPLIT, encoding="utf-8"))
    dev15 = sorted(str(t["id"]) for t in pick_topics(topics, 15, 42))
    rng = np.random.default_rng(42)
    pilot5 = sorted(rng.choice(dev15, size=5, replace=False).tolist())
    held = sorted(str(t["id"]) for t in topics if str(t["id"]) not in dev15)
    split = {"dev15": dev15, "pilot5": pilot5, "heldout35": held,
             "note": "held-out 에는 개발 파일럿 t000~t004 가 포함된다 (pick_topics 가 dev 표집에서 뺀다). 30토픽 민감도를 함께 낸다."}
    io.open(SPLIT, "w", encoding="utf-8").write(json.dumps(split, ensure_ascii=False, indent=1))
    return split


def arm_cfg(base: Config, arm: str | None, seed: int | None, args) -> Config:
    d = base.as_dict()
    d.setdefault("run", {}); d.setdefault("controller", {}); d.setdefault("budget", {})
    if arm is None:                        # Test A 재생 (P0)
        d["llm"]["cache_policy"] = "cache_only"
        return Config(d, source="r2:testa-replay")
    family, mode, kind, gran = ARMS[arm]
    d["run"].update({"seed": seed, "family": family, "arm": arm})
    d["branch"]["prompt_mode"] = mode
    d["controller"].update({"kind": kind, "granularity": gran})
    d["paths"]["cache_dir"] = args.cache_dir
    d["paths"]["cache_dir_ro"] = args.cache_dir_ro
    d["llm"]["cache_policy"] = "cache_only" if args.cache_only else "live"
    if args.model:
        d["llm"]["model"] = args.model
    d["budget"].update({"max_billed_in": args.max_billed_in, "max_billed_out": args.max_billed_out})
    return Config(d, source="r2:%s:s%s" % (arm, seed))


def make_controller(cfg):
    if cfg.get("controller.kind", "global_retry") == "freeze_retry":
        return FreezeRetry(cfg, granularity=cfg.get("controller.granularity", "branch"))
    return GlobalRetry(cfg)


def branch_record(qr, rr, bid) -> dict:
    o, ev = rr.outputs[bid], rr.evidence[bid]
    executed = bid in rr.executed
    return {
        "round": rr.round, "branch": bid, "executed": executed,
        "evidence": ev.to_dict(),
        "tokens_in": o.tokens_in if executed else 0, "tokens_out": o.tokens_out if executed else 0,
        "llm_calls": o.llm_calls if executed else 0,
        "billing": o.billing if executed else {}, "answer_stats": o.answer_stats if executed else {},
        "subq_executed": o.subq_executed if executed else 0, "context_sha256": o.context_sha256,
        "degraded": o.degraded, "truncated": o.truncated,
        "sub_answers": [{"index": s.index, "sub_question": s.sub_question, "text": s.text,
                         "citations": s.citations, "status": s.status, "produced_round": s.produced_round}
                        for s in o.sub_answers],
        "n_chunks": len(o.chunks),
    }


def run_one(cfg, topics, outdir: Path, retriever, nli, est, run_tag: str) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "topics").mkdir(exist_ok=True)
    yaml.safe_dump(cfg.as_dict(), io.open(outdir / "config.yaml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    llm = LLM(cfg)
    controller = make_controller(cfg)
    run_id = new_run_id()
    with TraceWriter(outdir / "trace.jsonl", append=True) as tw:
        for i, t in enumerate(topics):
            qid = str(t["id"])
            done = outdir / "topics" / (qid + ".json")
            if done.exists():
                continue
            t0 = time.time()
            qr = run_query(query_id=qid, query=t["topic"], retriever=retriever, llm=llm, nli=nli, cfg=cfg,
                           controller=controller, run_id=run_id, trace_writer=tw, est=est,
                           search_anchor=" ".join(t.get("terms") or []))
            rec = {"query_id": qid, "run_tag": run_tag, "passed": qr.passed, "n_rounds": len(qr.rounds),
                   "rounds": [{"round": rr.round, "action": rr.action.to_dict(), "verify": {"passed": rr.verify.passed, "agg": rr.verify.agg},
                               "executed": rr.executed, "cost": rr.cost,
                               "branches": [branch_record(qr, rr, b) for b in sorted(rr.outputs)]}
                              for rr in qr.rounds]}
            tmp = done.with_suffix(".tmp")
            io.open(tmp, "w", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False))
            tmp.replace(done)
            print("  [%2d/%2d] %s rounds=%d passed=%s %.0fs billed in/out %d/%d"
                  % (i + 1, len(topics), qid, len(qr.rounds), qr.passed, time.time() - t0,
                     llm.stats.billed_tokens_in, llm.stats.billed_tokens_out), flush=True)
    usage = llm.stats.to_dict()
    prev = outdir / "usage.json"
    hist = json.load(io.open(prev, encoding="utf-8")) if prev.exists() else []
    hist.append({"finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "usage": usage})
    io.open(prev, "w", encoding="utf-8").write(json.dumps(hist, ensure_ascii=False, indent=1))
    return usage


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R2 재실험 (확정안 R2)")
    ap.add_argument("--arms", nargs="*", default=ORDER)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1])
    ap.add_argument("--topics", default="pilot5", help="pilot5 | dev15 | heldout35 | 쉼표로 묶은 토픽 id")
    ap.add_argument("--run-tag", default=None, help="results/r2/<run_tag>. 기본은 --topics 값")
    ap.add_argument("--cache-dir", default=".cache/llm_r2")
    ap.add_argument("--cache-dir-ro", default=".cache/llm")
    ap.add_argument("--cache-only", action="store_true", help="캐시 미스를 예외로 (API 0 재생)")
    ap.add_argument("--replay-testa", action="store_true", help="P0: seed 없이 Test A 를 cache_only 로 재생")
    ap.add_argument("--model", default=None, help="N arm: gpt-5.4-nano")
    ap.add_argument("--max-billed-in", type=int, default=15_000_000)
    ap.add_argument("--max-billed-out", type=int, default=5_600_000)
    args = ap.parse_args(argv)

    load_dotenv_if_present(str(ROOT / ".env"))
    base = load_config(ROOT / "config.yaml")
    with io.open(ROOT / "data/topics_pilot50.yaml", encoding="utf-8") as fh:
        all_topics = yaml.safe_load(fh)["topics"]
    split = make_split(all_topics)
    ids = split[args.topics] if args.topics in split else args.topics.split(",")
    topics = [t for t in all_topics if str(t["id"]) in set(ids)]
    run_tag = args.run_tag or args.topics.replace(",", "_")
    print("토픽 %d개 (%s) · split sha %s" % (len(topics), args.topics,
          hashlib.sha256(SPLIT.read_bytes()).hexdigest()[:12]), flush=True)

    retriever = BM25Retriever(base)
    nli = load_nli(base)
    if args.replay_testa:
        cfg = arm_cfg(base, None, None, args)
        est = load_estimator(cfg, nli=nli)                 # Test A 원 연산자 (nli_max)
        usage = run_one(cfg, topics, ROOT / "results/r2" / run_tag / "TESTA-replay", retriever, nli, est, run_tag)
        print(json.dumps({k: usage[k] for k in ("calls", "cache_hits", "api_calls")}, ensure_ascii=False))
        return 0 if usage["api_calls"] == 0 else 2

    d = base.as_dict(); d.setdefault("estimator", {})["kind"] = "minicheck_doc"
    est = MemoEstimator(load_estimator(Config(d, source="r2:est"), nli=nli))   # 루프 안 결정 연산자 = MiniCheck
    arms = [a for a in ORDER if a in set(args.arms)]
    unknown = set(args.arms) - set(ARMS)
    if unknown:
        raise SystemExit("모르는 arm: %s (E-R1 은 E-GR 라운드 0 에서 분석 때 만든다)" % sorted(unknown))
    for seed in args.seeds:
        for arm in arms:
            cfg = arm_cfg(base, arm, seed, args)
            name = arm + ("@" + args.model if args.model else "")
            print("=== %s s%d ===" % (name, seed), flush=True)
            usage = run_one(cfg, topics, ROOT / "results/r2" / run_tag / name / ("s%d" % seed), retriever, nli, est, run_tag)
            print("    usage calls %d · api %d · billed in/out %d/%d"
                  % (usage["calls"], usage["api_calls"], usage["billed_tokens_in"], usage["billed_tokens_out"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
