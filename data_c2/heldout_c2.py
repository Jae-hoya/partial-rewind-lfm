"""PREREG_R3 §4.2 held-out — `data/heldout.py` 를 고치지 않고 확장한다.

토픽마다 다음 문서를 코퍼스(인덱스)에서 뺀다.
  (0) target — **identifier 로** (제목 매칭이 아니다)
  (a) 정규화 제목(`heldout.normalize_title`)이 target 과 같은 문서
  (b) 초록 내용어(`topics.tokens(abstract, 4)` = 길이 ≥ 4, STOPWORDS 제외) 자카드 ≥ heldout.near_dup_jaccard(0.6) 인 문서
임베딩 근접 중복 감사는 하지 않는다(공시). 출력은 `data/heldout.py` 와 같은 키(`heldout_identifiers`, `by_topic`)에
토픽별 제거 수를 더한 `data_c2/heldout_ids.json` — `data/chunk_index.py --config config_c2.yaml` 이 그대로 읽는다.

  python data_c2/heldout_c2.py --config config_c2.yaml
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from heldout import load_approved, normalize_title  # noqa: E402  (data/heldout.py)
from lfm.config import load_config  # noqa: E402
from topics import tokens  # noqa: E402  (data/topics.py)

MIN_LEN = 4


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def compute_heldout(rows: list[tuple[int, str, str]], topics: list[dict], thr: float) -> dict[str, Any]:
    """rows = (identifier, name, abstract). 한 번 훑으며 모든 토픽과 비교한다."""
    by_id = {int(r[0]): r for r in rows}
    targets: dict[str, dict[str, Any]] = {}
    unmatched = []
    for t in topics:
        tid = str(t["id"])
        ident = t.get("target_identifier")
        if ident is None or int(ident) not in by_id:
            unmatched.append(tid)
            continue
        _i, name, abstract = by_id[int(ident)]
        targets[tid] = {"identifier": int(ident), "norm_title": normalize_title(name), "terms": tokens(abstract, MIN_LEN),
                        "title_dups": [], "jaccard_dups": []}
    for ident, name, abstract in rows:
        ident = int(ident)
        nt = None
        terms = None
        for tid, tg in targets.items():
            if ident == tg["identifier"]:
                continue
            nt = nt if nt is not None else normalize_title(name)
            if nt and nt == tg["norm_title"]:
                tg["title_dups"].append(ident)
                continue
            terms = terms if terms is not None else tokens(abstract, MIN_LEN)
            if jaccard(terms, tg["terms"]) >= thr:
                tg["jaccard_dups"].append(ident)
    held: dict[str, Any] = {}
    all_ids: set[int] = set()
    for t in topics:
        tid = str(t["id"])
        if tid not in targets:
            continue
        tg = targets[tid]
        ids = sorted({tg["identifier"], *tg["title_dups"], *tg["jaccard_dups"]})
        all_ids.update(ids)
        held[tid] = {"topic": t.get("topic", ""), "target_article": t.get("target_article"),
                     "target_identifier": tg["identifier"], "heldout_identifiers": ids,
                     "removed": {"target": 1, "title_identical": len(tg["title_dups"]),
                                 "abstract_jaccard_ge_%.2f" % thr: len(tg["jaccard_dups"])}}
    return {"by_topic": held, "heldout_identifiers": sorted(all_ids), "unmatched_targets": unmatched}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="코퍼스2 held-out (PREREG_R3 §4.2)")
    ap.add_argument("--config", default=str(ROOT / "config_c2.yaml"))
    ap.add_argument("--topics", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    data_dir = Path(cfg["paths.data_dir"])
    doc = load_approved(Path(args.topics or (data_dir / "topics_pilot50.yaml")))
    out = Path(args.out or (data_dir / "heldout_ids.json"))
    thr = float(cfg["heldout.near_dup_jaccard"])

    import duckdb
    t0 = time.time()
    con = duckdb.connect()
    rows = con.execute("SELECT identifier, name, abstract FROM read_parquet('%s')"
                       % str(data_dir / "corpus.parquet").replace("\\", "/")).fetchall()
    con.close()
    res = compute_heldout(rows, doc["topics"], thr)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "spec": "PREREG_R3 §4.2",
        "match_mode": "identifier + normalized_title_identity + abstract_term_jaccard",
        "near_dup_jaccard": thr, "term_rule": "data/topics.py tokens(abstract, 4): 길이 >= 4, STOPWORDS 제외",
        "embedding_near_dup_audit": "하지 않음 (공시)",
        "n_topics": len(doc["topics"]), "n_heldout_documents": len(res["heldout_identifiers"]),
        **res, "elapsed_s": round(time.time() - t0, 2),
    }
    io.open(out, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=2))
    print("토픽 %d개, held-out 문서 %d건 -> %s" % (len(doc["topics"]), len(res["heldout_identifiers"]), out))
    if res["unmatched_targets"]:
        print("FAIL — target_identifier 가 코퍼스에 없다: %r" % res["unmatched_targets"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
