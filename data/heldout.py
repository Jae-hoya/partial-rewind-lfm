"""SPEC §2.5 — 승인된 각 토픽의 target article 을 코퍼스에서 빼내 정답으로 보관한다.

이 스냅샷에는 **redirect 필드가 없다** (MANIFEST 의 heldout_match_mode 참조).
그래서 title 정규화 매칭으로 대체하고, 그 사실을 산출물에 기록한다 (SPEC v1.1 §2.5).

정규화: NFKD -> 발음구별기호 제거 -> 소문자 -> 괄호 한정어 제거 -> 문장부호를 공백으로
-> 선행 관사(the/a/an) 제거 -> 공백 정규화.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lfm.config import load_config  # noqa: E402

_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_ARTICLE = re.compile(r"^(the|a|an)\s+")


def normalize_title(title: str) -> str:
    t = unicodedata.normalize("NFKD", title or "")
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower().strip()
    t = _PAREN.sub("", t)
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    t = _ARTICLE.sub("", t)
    return t


def load_approved(path: Path) -> dict[str, Any]:
    """SPEC §2.4 승인 게이트. approved: true 가 없으면 에러로 종료한다."""
    if not path.exists():
        raise SystemExit(
            "승인된 토픽 파일이 없다: " + str(path) + "\n"
            "topics.py 가 만든 후보를 사람이 검토해 topics_pilot50.yaml 로 저장하고 "
            "최상단에 approved: true 를 넣어야 한다 (SPEC §2.4)."
        )
    with io.open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict) or doc.get("approved") is not True:
        raise SystemExit(
            str(path) + " 에 approved: true 가 없다. 경고 후 진행이 아니라 에러 종료다 "
            "(SPEC §2.4)."
        )
    if not doc.get("topics"):
        raise SystemExit(str(path) + " 에 topics 가 비어 있다.")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="held-out 확정 (SPEC §2.5)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--topics", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config or (repo_root / "config.yaml"))
    data_dir = Path(cfg["paths.data_dir"])
    topics_path = Path(args.topics or (data_dir / "topics_pilot50.yaml"))
    out = Path(args.out or (data_dir / "heldout_ids.json"))

    doc = load_approved(topics_path)
    topics = doc["topics"]

    import duckdb

    con = duckdb.connect()
    rows = con.execute(
        "SELECT identifier, name FROM read_parquet('"
        + str(data_dir / "corpus.parquet").replace("\\", "/")
        + "')"
    ).fetchall()
    con.close()

    by_norm: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for ident, name in rows:
        by_norm[normalize_title(name)].append((int(ident), name))

    t0 = time.time()
    held: dict[str, dict[str, Any]] = {}
    all_ids: set[int] = set()
    unmatched: list[str] = []

    for t in topics:
        tid = str(t["id"])
        target = str(t["target_article"])
        key = normalize_title(target)
        matches = by_norm.get(key, [])
        if not matches:
            unmatched.append(tid + ": " + target)
        ids = [m[0] for m in matches]
        # 승인 파일이 target_identifier 를 들고 있으면 그것도 반드시 포함한다
        if t.get("target_identifier") is not None:
            ids.append(int(t["target_identifier"]))
        ids = sorted(set(ids))
        all_ids.update(ids)
        held[tid] = {
            "topic": t.get("topic", ""),
            "target_article": target,
            "normalized_title": key,
            "heldout_identifiers": ids,
            "matched_titles": [m[1] for m in matches],
            "n_matched": len(matches),
        }

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "spec": "SPEC_M1_TestA_v1.1.md §2.5",
        "match_mode": "title_normalization",
        "match_mode_reason": (
            "이 스냅샷에는 redirect 필드가 없다 (MANIFEST.verdicts.redirect_field_present=false). "
            "SPEC v1.1 §2.2 가 지정한 폴백 경로다."
        ),
        "n_topics": len(topics),
        "n_heldout_documents": len(all_ids),
        "unmatched_targets": unmatched,
        "heldout_identifiers": sorted(all_ids),
        "by_topic": held,
        "elapsed_s": round(time.time() - t0, 2),
    }
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    dupes = {k: v["n_matched"] for k, v in held.items() if v["n_matched"] > 1}
    print("토픽 %d개, held-out 문서 %d건 -> %s" % (len(topics), len(all_ids), out))
    print("매칭 방식: title_normalization (redirect 필드 없음)")
    if dupes:
        print("제목 정규화가 2건 이상 잡은 토픽: " + repr(dupes))
    if unmatched:
        print("FAIL — target article 을 코퍼스에서 찾지 못했다:")
        for u in unmatched:
            print("  " + u)
        return 1
    print("PASS — 전 토픽의 target article 을 찾았다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
