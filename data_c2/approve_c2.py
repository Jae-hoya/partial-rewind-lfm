"""PREREG_R3 §4.2 승인 — `data/approve_topics.py` 를 고치지 않고 감싼다 (`approval.mode: delegated`, 자동 기계 검사).

기존 검사 5종(approve_topics.check + 같은 계열 + near-duplicate)에 **archive ≥ topics.min_archives** 를 더한다.
archive 검사를 먼저 걸러 낸 후보 목록으로 approve_topics.main 을 부르는 것은 원 검사 순서와 같은 결과다 — approve_topics 는
탈락한 후보를 kept 목록에 넣지 않으므로(뒤 후보의 near-duplicate 판정에 영향 없음) 사전 탈락과 사후 탈락이 구분되지 않는다.
통과 후보 중 topics.py 정렬 순서 **앞의 28개** (`--n 28`).

모자라면(§4.2): 후보 120 으로 한 번만 다시 만든다(이탈) → 그래도 모자라면 `--allow-short` 로 승인된 수로 진행하되
23 미만(heldout 20 미만)이면 approved: false 로 두고 실행하지 않는다.

  python data_c2/approve_c2.py --config config_c2.yaml
  python data_c2/approve_c2.py --config config_c2.yaml --allow-short     # 120 후보 재생성 뒤에만
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

import approve_topics as AT  # noqa: E402  (data/approve_topics.py)
from lfm.config import load_config  # noqa: E402

N_APPROVE = 28
MIN_APPROVE = 23
DELEGATED_BY = "corresponding author (2026-11-10 chat; PREREG_R3 §4.1 automated, criteria-based topic approval)"


def archive_filter(candidates: list[dict], min_archives: int) -> tuple[list[dict], list[dict]]:
    keep, rej = [], []
    for c in candidates:
        n = int(c.get("n_archives", 0))
        if n < min_archives:
            rej.append({"id": c["id"], "target_article": c.get("target_article"),
                        "reasons": ["이웃 archive %d < %d" % (n, min_archives)]})
        else:
            keep.append(c)
    return keep, rej


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="코퍼스2 토픽 승인 (PREREG_R3 §4.2)")
    ap.add_argument("--config", default=str(ROOT / "config_c2.yaml"))
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=N_APPROVE)
    ap.add_argument("--allow-short", action="store_true", help="120 후보 재생성 뒤에만: 23~27 이면 그 수로 승인")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths.data_dir"])
    cand_path = Path(args.candidates or (data_dir / "topics_pilot50_candidates.yaml"))
    out = Path(args.out or (data_dir / "topics_pilot50.yaml"))
    doc = yaml.safe_load(io.open(cand_path, encoding="utf-8"))
    min_arch = int(cfg.get("topics.min_archives", 2))
    keep, arch_rej = archive_filter(doc["candidates"], min_arch)

    with tempfile.TemporaryDirectory() as td:
        filt = Path(td) / "candidates_archive_ok.yaml"
        yaml.safe_dump({**doc, "candidates": keep}, io.open(filt, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
        rc = AT.main(["--config", str(args.config), "--candidates", str(filt), "--out", str(out),
                      "--n", str(args.n), "--delegated-by", DELEGATED_BY])

    res = yaml.safe_load(io.open(out, encoding="utf-8"))
    n_ok = len(res.get("topics") or [])
    res["approval"]["source_candidates"] = str(cand_path)
    res["approval"]["n_candidates"] = len(doc["candidates"])
    res["approval"]["n_rejected"] = res["approval"].get("n_rejected", 0) + len(arch_rej)
    res["approval"]["criteria_checked"].append("이웃 archive >= topics.min_archives (%d) (C2 추가, 다른 검사보다 먼저)" % min_arch)
    res["approval"]["spec"] = "PREREG_R3 §4.2 — 통과 후보 중 topics.py 정렬 순서 앞의 %d개" % args.n
    order = {c["id"]: i for i, c in enumerate(doc["candidates"])}
    res["rejected"] = sorted(arch_rej + (res.get("rejected") or []), key=lambda r: order.get(r["id"], 10 ** 9))
    if n_ok < args.n:
        if args.allow_short and n_ok >= MIN_APPROVE:
            res["approved"] = True
            res["approval"]["shortfall"] = {"approved": n_ok, "wanted": args.n, "rule": "§4.2: 120 후보 재생성 뒤 승인된 수로 진행 (이탈 기록)"}
            rc = 0
        else:
            res["approved"] = False
            res["approval"]["shortfall"] = {"approved": n_ok, "wanted": args.n,
                                            "rule": "모자라면 후보 120 으로 한 번만 다시 (이탈). 23 미만이면 실행하지 않고 보고"}
            rc = 1
    yaml.safe_dump(res, io.open(out, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False, width=100)
    print("C2 승인 %d (archive 탈락 %d) -> %s · approved=%s" % (n_ok, len(arch_rej), out, res["approved"]))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
