"""PREREG_R3 §4.3 회귀 테스트 (C1-C2 조건) — wiki 경로가 코퍼스2 작업 뒤에도 바이트 동일한가. **API 0, GPU 사용.**

c2_run.py 는 r2_run.py 를 고치지 않고 그 함수(`arm_cfg`, `run_one`)를 그대로 쓴다. 그래서 이 테스트는
(1) `git diff HEAD -- experiments/r2_run.py experiments/r2_analyze.py lfm/ config.yaml` 이 비었는지,
(2) heldout35 의 한 토픽·한 arm·한 seed 를 r2_run 의 `arm_cfg`·`run_one` 으로 **cache_only** 재생해(r2_run.main 의 기본값:
    cache_dir .cache/llm_r2, cache_dir_ro .cache/llm, MiniCheck 결정 연산자) 토픽 JSON 이 R2 산출과 바이트 같은지,
(3) 재생의 `api_calls == 0` 인지를 본다. 출력은 임시 폴더에만 쓴다(results/r2 는 읽기만).

run_tag 는 토픽 JSON 안에 들어가므로 원 run_tag("heldout35")를 그대로 넘긴다. trace.jsonl(새 run_id·시각)은 비교하지 않는다.

**바이트 동일의 범위.** branch 의 `billing` 블록(api_calls·billed_*·by_tag.*.api_calls)은 정의상 "이 프로세스가 API 로
청구한 몫"이라 원 실행(API)과 cache_only 재생(캐시)에서 반드시 다르다(재생은 전부 0). 그래서 판정은
(i) 두 JSON 에서 `billing` 을 뺀 정규화 바이트가 같다, (ii) 재생의 billing 이 모두 0 이다, (iii) api_calls == 0 이다.
원 파일의 raw 바이트 비교 결과도 함께 낸다(billing 때문에 false 가 정상).

  python experiments/c2_wiki_regress.py --arm E-SQ --seed 1 --topic t053 --out <scratch>/regress
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
REGISTERED = ("experiments/r2_run.py", "experiments/r2_analyze.py", "lfm/", "config.yaml")


def strip_billing(rec: dict) -> dict:
    import copy
    r = copy.deepcopy(rec)
    for rr in r["rounds"]:
        for br in rr["branches"]:
            br.pop("billing", None)
    return r


def canon(rec: dict) -> bytes:
    return json.dumps(rec, ensure_ascii=False, sort_keys=True).encode("utf-8")


def registered_diff() -> str:
    return subprocess.run(["git", "diff", "HEAD", "--stat", "--", *REGISTERED], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", default="heldout35")
    ap.add_argument("--arm", default="E-SQ")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    out = Path(args.out)
    assert not out.exists() or not any(out.iterdir()), "빈 폴더여야 한다: %s" % out

    import yaml
    import r2_run
    from lfm.config import Config, load_config
    from lfm.estimator import MemoEstimator, load_estimator
    from lfm.llm import BillingLedger
    from lfm.nli import load_nli
    from lfm.retriever import BM25Retriever

    diff = registered_diff()
    base = load_config(ROOT / "config.yaml")
    all_topics = yaml.safe_load(open(ROOT / "data/topics_pilot50.yaml", encoding="utf-8"))["topics"]
    split = r2_run.make_split(all_topics)                       # 이미 있으면 읽기만
    assert args.topic in split[args.run_tag] if args.run_tag in split else True
    topics = [t for t in all_topics if str(t["id"]) == args.topic]
    assert len(topics) == 1
    ns = argparse.Namespace(cache_dir=".cache/llm_r2", cache_dir_ro=".cache/llm", cache_only=True, model=None,
                            max_billed_in=15_000_000, max_billed_out=5_600_000)   # r2_run.main 기본값 + --cache-only
    cfg = r2_run.arm_cfg(base, args.arm, args.seed, ns)
    retriever = BM25Retriever(base)
    nli = load_nli(base)
    d = base.as_dict(); d.setdefault("estimator", {})["kind"] = "minicheck_doc"
    est = MemoEstimator(load_estimator(Config(d, source="r2:est"), nli=nli))
    outdir = out / args.run_tag / args.arm / ("s%d" % args.seed)
    usage = r2_run.run_one(cfg, topics, outdir, retriever, nli, est, args.run_tag,
                           ledger=BillingLedger(out / args.run_tag / "billing_ledger.jsonl"))
    ref = ROOT / "results/r2" / args.run_tag / args.arm / ("s%d" % args.seed) / "topics" / (args.topic + ".json")
    new = outdir / "topics" / (args.topic + ".json")
    same = filecmp.cmp(ref, new, shallow=False)
    a = json.load(open(ref, encoding="utf-8")); b = json.load(open(new, encoding="utf-8"))
    bills = [br.get("billing") or {} for rr in b["rounds"] for br in rr["branches"]]
    replay_zero = all(int(x.get(k, 0)) == 0 for x in bills for k in ("api_calls", "billed_in", "billed_out", "billed_reasoning"))
    same_nb = canon(strip_billing(a)) == canon(strip_billing(b))
    res = {"registered_code_diff_vs_HEAD": diff or "(empty)", "topic": args.topic, "arm": args.arm, "seed": args.seed,
           "ref": str(ref), "new": str(new), "raw_bytes_identical": same, "identical_excluding_billing": same_nb,
           "replay_billing_all_zero": replay_zero,
           "ref_bytes": ref.stat().st_size, "new_bytes": new.stat().st_size,
           "api_calls": usage["api_calls"], "calls": usage["calls"], "cache_hits": usage["cache_hits"]}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    ok = same_nb and replay_zero and usage["api_calls"] == 0 and not diff
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
