"""PREREG_R3 §4.6 C-G6 — 연산자 음성 통제 (label-free). 생성 LLM 0, MiniCheck(GPU) 만.

표본: run_dir 의 seed `--seed` 산출(5 arm 전부, 모든 라운드)에서 **그 라운드에 새로 만든 answered sub-answer**(legacy 는 status 가
없으므로 본문·유효 인용이 있는 것)의 문장. 단위 = (토픽, 문장, 그 sub-answer 의 고유 인용 청크 순서열), 중복 제거 뒤 정렬하고
`default_rng(4503)` 으로 최대 300개.

전제 바꾸기 (원 전제와 **같은 청크 수 k**, 단일 문서 = 청크 본문을 MiniCheckDocEstimator.SEP 로 이어붙임)
  (a) off-topic: 토픽을 id 순으로 놓고 토픽 i 의 문장 → 토픽 (i+1) mod n 의 **토픽 풀**에서 k개.
      토픽 풀 = 그 토픽의 이 seed 산출 전체에서 인용된 고유 청크(자기 토픽 풀과 겹치는 청크는 뺀다). 같은 토픽 배정 0건을 assert.
      기준(실행 전 고정): 위양성 점추정 ≤ 0.05 **그리고** Wilson 95% 상한 ≤ 0.10.
  (b) near-miss (보고만): 같은 토픽 풀에서 이 sub-answer 가 인용하지 않은 청크 k개 (모자라면 있는 만큼, 0이면 건너뜀).
      실제로 지지할 수도 있으므로 위양성의 **상한**으로만 보고한다.
등록 문구 "토픽 풀(고유 청크, 단일 문서)" 의 해석: 풀 전체를 한 문서로 이으면 입력 토큰 가드(7,500)를 넘으므로 원 전제와 같은
청크 수를 풀에서 뽑는다. 이 해석은 결과 전 고정이다(코드 커밋 = C1-C2). 청크 추출 rng = default_rng([4503, 1|2, 토픽 순번]).
문장 < 100 이면 status = "remeasure_on_main_s1" (등록 경로: 본실험 s1 에서 같은 규칙으로 다시 잰다).

  python experiments/c2_negative_control.py --run-dir results/c2/pilot3 --seed 1
  python experiments/c2_negative_control.py --run-dir results/c2/heldout25 --seed 1    # 문장 < 100 일 때만
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import r2_analyze as RA  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED, N_MAX, MIN_N = 4503, 300, 100
FP_POINT, FP_UPPER = 0.05, 0.10


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (0.0 if k == 0 else max(0.0, c - h), 1.0 if k == n else min(1.0, c + h))


def _answered(sa: dict) -> bool:
    st = sa.get("status")
    if st is not None:
        return st == "answered" and bool(sa.get("citations"))
    return bool(sa.get("citations")) and bool((sa.get("text") or "").strip())


def collect(run_dir: Path, seed: int, chunk_text: dict[str, str]) -> tuple[list[tuple[str, str, tuple[str, ...]]], dict[str, list[str]]]:
    runs = RA.load_run(run_dir)
    units: set[tuple[str, str, tuple[str, ...]]] = set()
    pool: dict[str, set[str]] = {}
    for (arm, s), recs in runs.items():
        if s != seed:
            continue
        for qid, rec in recs.items():
            P = pool.setdefault(qid, set())
            for rr in rec["rounds"]:
                for br in rr["branches"]:
                    for sa in br.get("sub_answers") or []:
                        P.update(c for c in sa.get("citations") or [] if c in chunk_text)
                    for sa in RA.generated(br, rr["round"]):
                        if not _answered(sa):
                            continue
                        cites = tuple(c for c in dict.fromkeys(sa.get("citations") or []) if c in chunk_text)
                        if not cites:
                            continue
                        for sent in split_sentences(sa.get("text") or ""):
                            units.add((qid, sent, cites))
    return sorted(units), {q: sorted(v) for q, v in pool.items()}


def build_pairs(units, pool, n_max: int = N_MAX, seed: int = SEED):
    rng = np.random.default_rng(seed)
    if len(units) > n_max:
        units = [units[i] for i in sorted(int(x) for x in rng.choice(len(units), size=n_max, replace=False))]
    topics = sorted(pool)
    pos = {t: i for i, t in enumerate(topics)}
    off, near = [], []
    rngs = {}
    for qid, sent, cites in units:
        i = pos[qid]
        other = topics[(i + 1) % len(topics)]
        assert other != qid, "같은 토픽 배정 — 토픽이 하나뿐이다"
        k = len(cites)
        r_off = rngs.setdefault(("off", i), np.random.default_rng([seed, 1, i]))
        cand = [c for c in pool[other] if c not in set(pool[qid])]
        if cand:
            pick = [cand[int(j)] for j in r_off.choice(len(cand), size=min(k, len(cand)), replace=False)]
            off.append((qid, other, sent, tuple(pick)))
        r_near = rngs.setdefault(("near", i), np.random.default_rng([seed, 2, i]))
        cand = [c for c in pool[qid] if c not in set(cites)]
        if cand:
            pick = [cand[int(j)] for j in r_near.choice(len(cand), size=min(k, len(cand)), replace=False)]
            near.append((qid, qid, sent, tuple(pick)))
    return units, off, near


def score(est, pairs, chunk_text) -> list[bool]:
    return list(est.grounded_many([(s, [chunk_text[c] for c in p]) for _q, _o, s, p in pairs]))


def summarize(flags: list[bool]) -> dict[str, Any]:
    n, k = len(flags), int(sum(flags))
    lo, hi = wilson(k, n)
    return {"n": n, "false_positive": k, "rate": (k / n) if n else None, "wilson95": [lo, hi]}


def run(run_dir: Path, seed: int, chunk_text: dict[str, str], est, n_max: int = N_MAX) -> dict[str, Any]:
    units, pool = collect(run_dir, seed, chunk_text)
    n_all = len(units)
    units, off, near = build_pairs(units, pool, n_max)
    assert all(q != o for q, o, _s, _p in off)
    t0 = time.time()
    fo = score(est, off, chunk_text)
    fn = score(est, near, chunk_text)
    a, b = summarize(fo), summarize(fn)
    a["pass"] = bool(a["n"] > 0 and a["rate"] <= FP_POINT and a["wilson95"][1] <= FP_UPPER)
    a["criterion"] = "rate <= %.2f and Wilson95 upper <= %.2f" % (FP_POINT, FP_UPPER)
    b["note"] = "보고만 — 같은 토픽의 인용 안 한 청크는 실제로 지지할 수 있으므로 위양성의 상한"
    status = "measured" if len(units) >= MIN_N else "remeasure_on_main_s1"
    return {"spec": "PREREG_R3 §4.6 C-G6", "run_dir": str(run_dir), "seed_run": seed, "rng_seed": SEED,
            "n_units_available": n_all, "n_units_sampled": len(units), "n_topics": len(pool), "status": status,
            "offtopic": a, "near_miss": b, "score_seconds": round(time.time() - t0, 1),
            "premise_rule": "원 전제와 같은 청크 수 k, 단일 문서(SEP 이음)",
            "interpretation": "C-G6(a) 실패면 grounding 결과는 'operator not validated on this corpus' 로 보고만, C-H3·C-H4 는 coverage 만"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C-G6 연산자 음성 통제 (LLM 0)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--config", default=str(ROOT / "config_c2.yaml"))
    ap.add_argument("--n-max", type=int, default=N_MAX)
    ap.add_argument("--out", default=None, help="기본 <run_dir>/NEGCTL_C2.json")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    from lfm.config import Config, load_config
    from lfm.estimator import load_estimator
    base = load_config(args.config)
    d = base.as_dict(); d.setdefault("estimator", {})["kind"] = "minicheck_doc"
    est = load_estimator(Config(d, source="c2_negctl"))
    chunk_text = RA.load_chunk_text(ROOT / base["paths.index_dir"])
    run_dir = Path(args.run_dir) if Path(args.run_dir).is_absolute() else ROOT / args.run_dir
    res = run(run_dir, args.seed, chunk_text, est, args.n_max)
    res["config"] = str(args.config)
    out = Path(args.out or (run_dir / "NEGCTL_C2.json"))
    io.open(out, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in ("status", "n_units_sampled", "offtopic", "near_miss")}, ensure_ascii=False, indent=1))
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
