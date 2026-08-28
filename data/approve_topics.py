"""SPEC §2.4 승인 게이트 — 기계 검증 후 `topics_pilot50.yaml` 을 쓴다.

원래 이 단계는 사람이 한다. 사용자가 2026-08-29 대화에서 승인 권한을 위임했으므로
**대신 수행하되, 승인의 실질인 검증을 생략하지 않는다.** HANDOFF §4 의 경고는
"여기서 대충 승인하면 낭비율이 토픽 난이도의 산물이 된다" 이다. 통과 도장이 게이트가
아니라 검사가 게이트다.

검사 항목 (하나라도 걸리면 그 후보는 탈락, 예비 후보로 대체):
  1. 토픽 문구가 있고 길이가 타당한가
  2. 이웃 문서 수 >= topics.min_related_docs
  3. 이웃의 infobox 유형 >= topics.min_infobox_types  (교차 도메인성)
  4. target_article 이 코퍼스에 실재하는가 (제목 정규화 일치)
  5. 다른 후보와 near-duplicate 가 아닌가 (변별 용어 자카드 + 동일 계열 target)

토픽 문구가 target 제목을 얼마나 재현하는지(title_overlap)는 **기록만** 한다.
held-out 의 통제 수단은 인덱스에서 문서를 빼는 것이지 질문에서 이름을 숨기는 것이
아니다. 자세한 근거는 title_overlap() 주석.

산출 파일에 승인 경위와 검사 결과를 함께 남긴다.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from heldout import normalize_title  # noqa: E402
from lfm.config import load_config  # noqa: E402
from topics import STOPWORDS, tokens  # noqa: E402


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def title_overlap(target: str, topic: str) -> float:
    """target 제목의 내용어 중 몇 할이 토픽 문구에 나오는가. **기록용 지표이지
    탈락 사유가 아니다.**

    처음에는 이걸 'held-out 누수'로 보고 탈락시켰다. 잘못된 판정이었다. held-out 의
    통제 수단은 **target article 을 인덱스에서 제거하는 것**이지 질문에서 이름을
    숨기는 것이 아니다. 주제를 지목하지 않는 질문은 held-out reference 과제로서
    오히려 부자연스럽다. 게다가 제목이 한 단어인 경우("Bharath (actor)" -> "bharath")
    주제를 언급하기만 해도 비율이 1.0 이 되어 지표 자체가 의미를 잃는다.

    그래도 값은 남긴다 — 검색 난이도 해석에 쓸 수 있다.
    """
    t_tokens = tokens(normalize_title(target), 3)
    if not t_tokens:
        return 0.0
    q_tokens = tokens(normalize_title(topic), 3)
    return len(t_tokens & q_tokens) / len(t_tokens)


def family_key(target: str) -> str:
    """연도만 다른 같은 계열(예: Eurovision 2014/2022/2023)을 한 묶음으로 본다."""
    toks = [t for t in normalize_title(target).split()
            if not t.isdigit() and t not in STOPWORDS and len(t) >= 3]
    return " ".join(toks[:5])


def check(cand: dict[str, Any], titles: dict[str, str], cfg) -> list[str]:
    """탈락 사유 목록. 비어 있으면 통과."""
    bad: list[str] = []
    min_rel = int(cfg.get("topics.min_related_docs", 30))
    min_types = int(cfg.get("topics.min_infobox_types", 3))

    topic = (cand.get("topic") or "").strip()
    if not topic:
        bad.append("토픽 문구 없음")
    elif not (30 <= len(topic) <= 400):
        bad.append("토픽 문구 길이 이상 (%d자)" % len(topic))

    if int(cand.get("related_docs", 0)) < min_rel:
        bad.append("이웃 문서 %d < %d" % (cand.get("related_docs", 0), min_rel))
    if int(cand.get("n_infobox_types", 0)) < min_types:
        bad.append("infobox 유형 %d < %d" % (cand.get("n_infobox_types", 0), min_types))

    target = str(cand.get("target_article", ""))
    norm = normalize_title(target)
    if norm not in titles:
        bad.append("target 이 코퍼스에 없음: " + target)

    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="토픽 승인 게이트 (SPEC §2.4)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-jaccard", type=float, default=0.6)
    ap.add_argument("--delegated-by", default="user (2026-08-29 세션에서 위임)")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config or (repo_root / "config.yaml"))
    data_dir = Path(cfg["paths.data_dir"])
    cand_path = Path(args.candidates or (data_dir / "topics_pilot50_candidates.yaml"))
    out = Path(args.out or (data_dir / "topics_pilot50.yaml"))

    with io.open(cand_path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    candidates = doc["candidates"]

    import duckdb

    con = duckdb.connect()
    rows = con.execute(
        "SELECT name FROM read_parquet('"
        + str(data_dir / "corpus.parquet").replace("\\", "/") + "')"
    ).fetchall()
    con.close()
    titles = {normalize_title(r[0]): r[0] for r in rows}

    approved: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    kept_terms: list[set[str]] = []
    kept_families: dict[str, str] = {}

    for cand in candidates:
        reasons = check(cand, titles, cfg)
        terms = set(cand.get("terms") or [cand.get("term", "")])
        fam = family_key(str(cand.get("target_article", "")))
        if not reasons:
            if fam and fam in kept_families:
                reasons.append("기존 후보 %s 와 같은 계열 (%s)" % (kept_families[fam], fam))
            else:
                dup = next((i for i, prev in enumerate(kept_terms)
                            if jaccard(terms, prev) > args.max_jaccard), None)
                if dup is not None:
                    reasons.append("기존 후보 %s 와 near-duplicate" % approved[dup]["id"])
        if reasons:
            rejected.append({"id": cand["id"], "target_article": cand.get("target_article"),
                             "reasons": reasons})
            continue
        kept_terms.append(terms)
        if fam:
            kept_families[fam] = cand["id"]
        cand = dict(cand)
        cand["title_overlap"] = round(
            title_overlap(str(cand.get("target_article", "")), str(cand.get("topic", ""))), 3
        )
        approved.append(cand)
        if len(approved) >= args.n:
            break

    payload = {
        "approved": len(approved) >= args.n,
        "approval": {
            "mode": "delegated",
            "delegated_by": args.delegated_by,
            "performed_by": "Claude Code",
            "approved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "criteria_checked": [
                "토픽 문구 존재·길이",
                "이웃 문서 수 >= topics.min_related_docs",
                "이웃 infobox 유형 >= topics.min_infobox_types",
                "target_article 이 코퍼스에 실재 (제목 정규화 일치)",
                "변별 용어 자카드 <= %.2f (near-duplicate 배제)" % args.max_jaccard,
                "연도만 다른 동일 계열 target 배제",
            ],
            "source_candidates": str(cand_path),
            "n_candidates": len(candidates),
            "n_approved": len(approved),
            "n_rejected": len(rejected),
        },
        "criteria": doc.get("criteria", {}),
        "llm": doc.get("llm", {}),
        "rejected": rejected,
        "topics": approved,
    }
    with io.open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False, width=100)

    print("승인 %d / 후보 %d -> %s" % (len(approved), len(candidates), out))
    if rejected:
        print("탈락 %d건:" % len(rejected))
        for r in rejected[:20]:
            print("  %s  %s" % (r["id"], "; ".join(r["reasons"])))
    if len(approved) < args.n:
        print()
        print("FAIL — 승인 %d개 < 요구 %d개. approved: false 로 기록했다."
              % (len(approved), args.n))
        print("topics.n_candidates 를 올려 후보를 더 뽑아야 한다.")
        return 1
    print("PASS — approved: true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
