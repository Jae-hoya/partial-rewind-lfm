"""PREREG_R3 §4.3·§4.7 코퍼스2 드라이버 — `experiments/r2_run.py` 를 **고치지 않고** 그 함수를 그대로 쓴다.

PREREG_R3 §4.3 은 "r2_run.py 에 --corpus {wiki,c2}" 라고 적었다. 같은 의미를 r2_run.py 바이트 변경 없이 내려고 이 래퍼를 둔다:
`arm_cfg`·`run_one`·`make_controller`·`ARMS`·`ORDER` 는 r2_run 의 것을 import 하고, r2_run.main 의 하드코딩 다섯 가지만
코퍼스2 값으로 바꾼 main 을 둔다.

| r2_run.main (wiki) | 여기 (c2) |
|---|---|
| config.yaml | config_c2.yaml (`c2_config_diff.check(require_manifest=True)` 통과 필수) |
| data/topics_pilot50.yaml, data/split_r2.json (`make_split`) | data_c2/topics_pilot50.yaml (approved: true), data_c2/split_c2.json (`make_split_c2`, seed 4501) |
| results/r2/<run_tag> | results/c2/<run_tag> |
| --cache-dir .cache/llm_r2, --cache-dir-ro .cache/llm | .cache/llm_c2, 읽기 전용 캐시 없음(None) — 분해가 토픽마다 한 번 라이브 |
| 예산 기본 15M / 5.6M | 기본 없음: --max-billed-in/out 필수 (pilot3 0.6M/0.16M, heldout25 12.6M/3.4M) |

시작 시 assert: E/legacy 답 프롬프트와 분해 프롬프트 SHA-256(R2 와 바이트 동일), C-L 통과, heldout_ids.json·인덱스 존재.

  python experiments/c2_run.py --topics pilot3 --seeds 1 --max-billed-in 600000 --max-billed-out 160000
  python experiments/c2_run.py --topics heldout25 --seeds 1 2 3 --max-billed-in 12600000 --max-billed-out 3400000
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import r2_run  # noqa: E402  (experiments/r2_run.py — 바꾸지 않는다)
from lfm.config import Config, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG_C2 = ROOT / "config_c2.yaml"
DATA_C2 = ROOT / "data_c2"
TOPICS_C2 = DATA_C2 / "topics_pilot50.yaml"
SPLIT_C2 = DATA_C2 / "split_c2.json"
REDUCED_C2 = DATA_C2 / "split_c2_reduced.json"
MANIFEST = DATA_C2 / "MANIFEST_C2.json"
RESULTS_C2 = ROOT / "results" / "c2"
CACHE_C2 = ".cache/llm_c2"
SPLIT_SEED, REDUCE_SEED, N_PILOT = 4501, 4502, 3

# PREREG_R2 §0 / PREREG_R3 §0 — R2 와 바이트 동일해야 한다
PROMPT_SHA256 = {
    ("lfm.branch", "ANSWER_SYSTEM"): "9eed32d587b8bda711bcb20df7031a58df1ba1abf2ff57ae48892b26bf674cd8",
    ("lfm.branch", "ANSWER_PROMPT"): "7aacf26b0d542f85127b53d7c43cd0327a913a9da831d1cfa4b4c22dd2e061a4",
    ("lfm.branch", "ANSWER_SYSTEM_E"): "ecb1a9a08fc3d911efef92d3b47aa624c598a70255edf601d4c9314a4b72854d",
    ("lfm.branch", "ANSWER_PROMPT_E"): "14665f93038924ba090f5405d15d78d18ae6c3d06295987388e4408a6f79117b",
    ("lfm.perspectives", "SYSTEM"): "32162e1b6efd1cffb744960d9dff9120703b604ad81eacfd1bc40cd746c164cf",
    ("lfm.perspectives", "PROMPT"): "bb0f0521d0d304376659b3bbf4d1e0d1087e47f87ef0895dc35983d4f4e148b6",
}


def assert_prompts() -> dict[str, str]:
    import importlib
    got = {}
    for (mod, name), want in PROMPT_SHA256.items():
        h = hashlib.sha256(getattr(importlib.import_module(mod), name).encode("utf-8")).hexdigest()
        assert h == want, "%s.%s SHA-256 %s != 등록값 %s" % (mod, name, h, want)
        got["%s.%s" % (mod, name)] = h
    return got


def load_topics_c2(path: Path = TOPICS_C2) -> list[dict]:
    doc = yaml.safe_load(io.open(path, encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("approved") is not True:
        raise SystemExit("%s 에 approved: true 가 없다 (data_c2/approve_c2.py)" % path)
    return doc["topics"]


def make_split_c2(topics: list[dict], path: Path = SPLIT_C2) -> dict:
    """pilot3 = default_rng(4501).choice(n, 3) — 승인 순서(= topics.py 정렬 순서) 기준 위치. heldout25 = 나머지. 한 번 만들면 고정."""
    if path.exists():
        return json.load(io.open(path, encoding="utf-8"))
    ids = [str(t["id"]) for t in topics]
    n = len(ids)
    pos = sorted(int(i) for i in np.random.default_rng(SPLIT_SEED).choice(n, size=N_PILOT, replace=False))
    pilot = [ids[i] for i in pos]
    held = [i for i in ids if i not in set(pilot)]
    split = {"pilot3": pilot, "heldout25": held, "n_approved": n, "seed": SPLIT_SEED,
             "rule": "default_rng(%d).choice(%d, 3, replace=False) over approved order; heldout25 = rest" % (SPLIT_SEED, n),
             "note": "" if n == 28 else "승인 %d개 (28 아님) — 이탈로 기록. heldout 키 이름은 그대로" % n}
    io.open(path, "w", encoding="utf-8").write(json.dumps(split, ensure_ascii=False, indent=1))
    return split


def make_reduced(split: dict, n: int, path: Path = REDUCED_C2) -> list[str]:
    """C-G4 축소: heldout25 에서 default_rng(4502) 로 n개 (최소 20). 한 번 만들면 고정."""
    if path.exists():
        d = json.load(io.open(path, encoding="utf-8"))
        assert d["n"] == n, "이미 n=%d 로 축소했다" % d["n"]
        return d["ids"]
    if n < 20:
        raise SystemExit("C-G4: n=%d < 20 — 실행하지 않고 보고한다 (§4.6)" % n)
    held = split["heldout25"]
    ids = sorted(str(held[int(i)]) for i in np.random.default_rng(REDUCE_SEED).choice(len(held), size=n, replace=False))
    io.open(path, "w", encoding="utf-8").write(json.dumps({"n": n, "seed": REDUCE_SEED, "ids": ids}, indent=1))
    return ids


def preflight(base: Config) -> None:
    import c2_config_diff
    errs = c2_config_diff.check(require_manifest=True)
    if errs:
        raise SystemExit("config_c2 대조 실패:\n  " + "\n  ".join(errs))
    m = json.load(io.open(MANIFEST, encoding="utf-8"))
    if (m.get("license_gate") or {}).get("status") != "passed":
        raise SystemExit("C-G0(3): C-L 이 passed 로 기록되지 않았다 — 실행하지 않는다")
    if base.get("paths.cache_dir_ro", None) is not None:
        raise SystemExit("config_c2 의 paths.cache_dir_ro 는 null 이어야 한다")
    for p in (DATA_C2 / "heldout_ids.json", Path(base["paths.index_dir"]) / "chunks.jsonl"):
        if not (ROOT / p).exists() and not Path(p).exists():
            raise SystemExit("없다: %s (heldout_c2.py → data/chunk_index.py --config config_c2.yaml 순서)" % p)


def c2_args(ns) -> argparse.Namespace:
    """r2_run.arm_cfg 가 읽는 필드(cache_dir, cache_dir_ro, cache_only, model, max_billed_in/out)를 c2 값으로."""
    return argparse.Namespace(cache_dir=ns.cache_dir, cache_dir_ro=None, cache_only=ns.cache_only, model=None,
                              max_billed_in=ns.max_billed_in, max_billed_out=ns.max_billed_out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="코퍼스2 라이브 재현 (PREREG_R3 §4)")
    ap.add_argument("--arms", nargs="*", default=r2_run.ORDER)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1])
    ap.add_argument("--topics", default="pilot3", help="pilot3 | heldout25 | reduced | 쉼표로 묶은 토픽 id")
    ap.add_argument("--reduce-to", type=int, default=None, help="C-G4 축소 n (--topics reduced 와 함께)")
    ap.add_argument("--run-tag", default=None, help="results/c2/<run_tag>. 기본은 --topics 값 (reduced 면 heldout25)")
    ap.add_argument("--cache-dir", default=CACHE_C2)
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--max-billed-in", type=int, required=True)
    ap.add_argument("--max-billed-out", type=int, required=True)
    args = ap.parse_args(argv)
    os.chdir(ROOT)                                   # config 의 상대 경로(data_c2/index 등) 기준

    from lfm.estimator import MemoEstimator, load_estimator
    from lfm.llm import BillingLedger, load_dotenv_if_present
    from lfm.nli import load_nli
    from lfm.retriever import BM25Retriever

    hashes = assert_prompts()
    load_dotenv_if_present(str(ROOT / ".env"))
    base = load_config(CONFIG_C2)
    preflight(base)
    all_topics = load_topics_c2()
    split = make_split_c2(all_topics)
    if args.topics == "reduced":
        ids = make_reduced(split, int(args.reduce_to or 0))
        run_tag = args.run_tag or "heldout25"
    else:
        ids = split[args.topics] if args.topics in ("pilot3", "heldout25") else args.topics.split(",")
        run_tag = args.run_tag or args.topics.replace(",", "_")
    topics = [t for t in all_topics if str(t["id"]) in set(ids)]
    unknown = set(args.arms) - set(r2_run.ARMS)
    if unknown:
        raise SystemExit("모르는 arm: %s (E-R1 은 E-GR 라운드 0 에서 분석 때 만든다)" % sorted(unknown))
    arms = [a for a in r2_run.ORDER if a in set(args.arms)]
    print("C2 토픽 %d개 (%s) · split sha %s · prompts %s" % (
        len(topics), args.topics, hashlib.sha256(SPLIT_C2.read_bytes()).hexdigest()[:12],
        ",".join(h[:8] for h in hashes.values())), flush=True)

    retriever = BM25Retriever(base)
    nli = load_nli(base)
    d = base.as_dict(); d.setdefault("estimator", {})["kind"] = "minicheck_doc"
    est = MemoEstimator(load_estimator(Config(d, source="c2:est"), nli=nli))
    ledger = BillingLedger(RESULTS_C2 / run_tag / "billing_ledger.jsonl")
    print("원장 시작 청구 in/out %d/%d (상한 %d/%d)" % (ledger.billed_in, ledger.billed_out,
          args.max_billed_in, args.max_billed_out), flush=True)
    a2 = c2_args(args)
    for seed in args.seeds:
        for arm in arms:
            cfg = r2_run.arm_cfg(base, arm, seed, a2)
            assert cfg.get("paths.cache_dir_ro", None) is None and cfg.get("paths.cache_dir") == args.cache_dir
            print("=== %s s%d ===" % (arm, seed), flush=True)
            usage = r2_run.run_one(cfg, topics, RESULTS_C2 / run_tag / arm / ("s%d" % seed), retriever, nli, est,
                                   run_tag, ledger=ledger)
            print("    usage calls %d · api %d · billed in/out %d/%d"
                  % (usage["calls"], usage["api_calls"], usage["billed_tokens_in"], usage["billed_tokens_out"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
