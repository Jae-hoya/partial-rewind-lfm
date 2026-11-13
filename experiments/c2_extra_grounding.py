"""코퍼스2 grounding 기권 처리 민감도 (PREREG_R3 §4.8 '민감도: grounding 짝 제외 읽기'). API 0, 새 실행 0.

`r2_extra.block_c` 와 같은 정의를 코퍼스2 본실험(`results/c2/heldout25`)의 E-R1 − E-GR 에 적용한다.
등록 코드(r2_analyze, r2_extra)는 고치지 않고 import 만 한다. 0 규약 값이 ANALYSIS.json 과 같은지 assert.
  drop_both   = 두 arm 모두 무문장인 (topic, seed) 셀 제외
  drop_either = 어느 한 arm 이라도 무문장인 셀 제외
  cell / topic = 셀 단위 제외(2단 부트스트랩 뒤 남은 셀 평균) / 그 조건이 한 seed 라도 성립한 토픽을 모든 seed 에서 제외

  python experiments/c2_extra_grounding.py   # -> results/c2/heldout25/EXTRA_GROUNDING.json
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import r2_analyze as RA  # noqa: E402
import r2_extra as RX  # noqa: E402

RUN = ROOT / "results/c2/heldout25"


def main() -> int:
    an = RX.jload(RUN / "ANALYSIS.json")
    runs = RA.with_r1(RA.load_run(RUN))
    summ = {}
    for key, pq in an["per_query"].items():
        arm, s = key.rsplit(RX.SEP, 1)
        summ[(arm, int(s[1:]))] = {"per_query": pq}
    nsent = {(arm, s): {q: RX.n_sentences_final(rec) for q, rec in recs.items()} for (arm, s), recs in runs.items()}
    for k in summ:
        for q, n in nsent[k].items():
            if n == 0:
                assert summ[k]["per_query"][q]["grounding"] == 0.0, (k, q)
    a, b = RA.R1, "E-GR"
    filt = lambda t: True  # noqa: E731
    zr = RA.noninf_compare(summ, a, b, "grounding", RA.DELTA_GRD, (), filt, RA.N_BOOT)
    reg = an["all_topics"]["noninferiority"]["E-R1_vs_E-GR"]["grounding"]
    assert RX.jnorm(zr) == RX.jnorm(reg), "0 규약이 ANALYSIS.json 과 다르다"
    topics, seeds = RA._common(summ, a, b, filt)
    A = np.array([[summ[(a, s)]["per_query"][t]["grounding"] for s in seeds] for t in topics])
    B = np.array([[summ[(b, s)]["per_query"][t]["grounding"] for s in seeds] for t in topics])
    Ea = np.array([[nsent[(a, s)][t] == 0 for s in seeds] for t in topics])
    Eb = np.array([[nsent[(b, s)][t] == 0 for s in seeds] for t in topics])
    res = {"spec": "PREREG_R3 §4.8 sensitivity: grounding pair exclusion (definitions of r2_extra.block_c)",
           "run_dir": "results/c2/heldout25", "comparison": "E-R1_vs_E-GR", "n_boot": RA.N_BOOT, "boot_seed": RA.BOOT_SEED,
           "zero_rule": {k: zr[k] for k in ("n_topics", "seeds", "diff", "ci90", "delta", "noninferior", "per_seed")},
           "zero_rule_reproduces_ANALYSIS": True,
           "empty_queries": {"%s·s%d" % k: sum(1 for n in v.values() if n == 0) for k, v in sorted(nsent.items()) if k[0] in (a, b)}}
    for var, drop in (("drop_both", Ea & Eb), ("drop_either", Ea | Eb)):
        M = ~drop
        pt, boots = RX.masked_two_stage(A, B, M, RA.N_BOOT)
        lo, hi = RX.pct(boots, [5, 95])
        cell = {"point": pt, "ci90": [lo, hi], "noninferior": bool(lo > -RA.DELTA_GRD), "n_cells_kept": int(M.sum()),
                "n_cells": int(M.size), "nan_resamples": int((~np.isfinite(boots)).sum())}
        bad_t = {t for i, t in enumerate(topics) if drop[i].any()}
        tp = RA.noninf_compare(summ, a, b, "grounding", RA.DELTA_GRD, (), lambda t, bt=bad_t: t not in bt, RA.N_BOOT)
        topic = {k: tp.get(k) for k in ("n_topics", "diff", "ci90", "noninferior")} if tp.get("available") else {"available": False}
        res[var] = {"cell": cell, "topic": topic}
    out = RUN / "EXTRA_GROUNDING.json"
    io.open(out, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
