"""SPEC §2.2 — 다운로드 직후 필수 스키마 검사 + MANIFEST.json 기록. P1 게이트.

CLAUDE.md 재현성 규칙: 스키마 검사 실패 시 실험 시작 금지.

검사는 두 층으로 한다. **필드가 스키마에 있다**와 **필드에 값이 있다**는 다른 문제이고,
이 스냅샷에서는 실제로 갈린다 (예: version.scores.referencerisk 는 컬럼으로 존재하지만
채워진 행이 1% 미만이다). 존재만 보고 통과시키면 M2 Test C-2 설계가 빈 필드 위에 선다.

  HARD  없거나 비면 실험 시작 금지: identifier, name, url, abstract, sections
  SOFT  없으면 citation_mode 를 chunk 로 낮추고 계속: references
  INFO  채움률만 기록: referencerisk, referenceneed, infoboxes, tables
  REDIRECT  redirect 필드 부재 시 heldout 은 title 정규화 매칭으로 대체 (SPEC v1.1 §2.2)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lfm.config import load_config  # noqa: E402

HARD_FIELDS = ("identifier", "name", "url", "abstract", "sections")
SOFT_FIELDS = ("references",)
INFO_FIELDS = ("infoboxes", "tables", "description", "image")
# 중첩 신뢰도 신호. (표시명, duckdb 경로)
CREDIBILITY_FIELDS = {
    "referencerisk": "version.scores.referencerisk",
    "referenceneed": "version.scores.referenceneed",
    "revertrisk": "version.scores.revertrisk.probability.true",
}
# redirect 정보가 담길 만한 후보 이름들. 하나도 없으면 title 정규화로 간다.
REDIRECT_CANDIDATES = ("redirects", "redirect", "is_redirect", "redirect_target")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with io.open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def flatten_schema(schema) -> list[str]:
    """중첩 struct 를 점 표기 경로로 편다. 존재 검사는 이 목록으로 한다."""
    out: list[str] = []

    def walk(field, prefix: str) -> None:
        name = prefix + field.name
        out.append(name)
        if hasattr(field.type, "num_fields") and str(field.type).startswith("struct"):
            for i in range(field.type.num_fields):
                walk(field.type.field(i), name + ".")

    for f in schema:
        walk(f, "")
    return out


def arrow_type_of(schema, dotted: str):
    """점 표기 경로의 arrow 타입. 없으면 None."""
    parts = dotted.split(".")
    try:
        field = schema.field(parts[0])
    except KeyError:
        return None
    for part in parts[1:]:
        typ = field.type
        if not hasattr(typ, "num_fields") or typ.num_fields == 0:
            return None
        for i in range(typ.num_fields):
            if typ.field(i).name == part:
                field = typ.field(i)
                break
        else:
            return None
    return field.type


def _non_empty_predicate(quoted: str, typ) -> str | None:
    """타입별 '비어 있지 않음' 술어.

    숫자 컬럼에 문자열 길이 술어를 쓰면 한 자리 정수가 전부 빈 값으로 집계된다.
    """
    if typ is None:
        return None
    t = str(typ)
    if t.startswith("list"):
        return "len(" + quoted + ") > 0"
    if t in ("string", "large_string", "utf8"):
        # JSON 직렬화 컬럼의 빈 값('[]', '{}')도 빈 것으로 본다
        return (
            "trim(" + quoted + ") NOT IN ('', '[]', '{}', 'null')"
        )
    return None  # 숫자·시각·구조체는 non-null 을 그대로 쓴다


def fill_rates(parquet_path: Path, paths: dict[str, str], schema=None) -> dict[str, dict[str, Any]]:
    """duckdb 로 컬럼별 non-null / non-empty 비율을 센다."""
    import duckdb

    con = duckdb.connect()
    src = "read_parquet('" + str(parquet_path).replace("\\", "/") + "')"
    total = con.execute("SELECT count(*) FROM " + src).fetchone()[0]

    result: dict[str, dict[str, Any]] = {}
    for label, expr in paths.items():
        quoted = ".".join('"' + p + '"' for p in expr.split("."))
        try:
            n_notnull = con.execute(
                "SELECT count(*) FROM " + src + " WHERE " + quoted + " IS NOT NULL"
            ).fetchone()[0]
        except Exception as exc:  # 컬럼 자체가 없는 경우
            result[label] = {"present": False, "error": type(exc).__name__}
            continue
        fill = round(n_notnull / total, 4) if total else 0.0
        row: dict[str, Any] = {"present": True, "non_null": n_notnull, "fill_rate": fill}

        typ = arrow_type_of(schema, expr) if schema is not None else None
        pred = _non_empty_predicate(quoted, typ)
        if pred is None:
            row["non_empty"] = n_notnull
            row["non_empty_rate"] = fill
        else:
            n_ne = con.execute(
                "SELECT count(*) FROM " + src + " WHERE " + quoted
                + " IS NOT NULL AND " + pred
            ).fetchone()[0]
            row["non_empty"] = n_ne
            row["non_empty_rate"] = round(n_ne / total, 4) if total else 0.0
        result[label] = row
    con.close()
    result["_total_rows"] = {"rows": total}
    return result


def upstream_provenance(cfg, shards_read: list[str]) -> list[dict[str, Any]]:
    """상류 파일의 LFS SHA256. 37GB 를 받지 않고도 무결성 근거를 남긴다."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(
        cfg["data.hf_dataset"], revision=cfg["data.hf_revision"], files_metadata=True
    )
    by_name = {s.rfilename: s for s in info.siblings}
    out = []
    for path in shards_read:
        s = by_name.get(path)
        lfs = getattr(s, "lfs", None) if s else None
        out.append(
            {
                "path": path,
                "size": getattr(s, "size", None) if s else None,
                "sha256": (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None))
                if lfs
                else None,
                "blob_id": getattr(s, "blob_id", None) if s else None,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="스키마 검사 + MANIFEST 기록 (SPEC §2.2)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--corpus", default=None, help="기본 data/corpus.parquet")
    ap.add_argument("--out", default=None, help="기본 data/MANIFEST.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths.data_dir"])
    corpus = Path(args.corpus or (data_dir / "corpus.parquet"))
    manifest_path = Path(args.out or (data_dir / "MANIFEST.json"))
    stats_path = data_dir / "download_stats.json"

    if not corpus.exists():
        print("코퍼스가 없다: " + str(corpus) + " — download.py 를 먼저 돌려라")
        return 1

    t0 = time.time()
    pf = pq.ParquetFile(corpus)
    schema_paths = flatten_schema(pf.schema_arrow)
    n_rows = pf.metadata.num_rows

    # --- 존재 검사 ---------------------------------------------------
    present = set(schema_paths)
    hard_missing = [f for f in HARD_FIELDS if f not in present]
    soft_missing = [f for f in SOFT_FIELDS if f not in present]
    cred_missing = [k for k, p in CREDIBILITY_FIELDS.items() if p not in present]
    redirect_found = [c for c in REDIRECT_CANDIDATES if c in present]

    # --- 채움률 ------------------------------------------------------
    probe = {f: f for f in HARD_FIELDS + SOFT_FIELDS + INFO_FIELDS if f in present}
    probe.update({k: v for k, v in CREDIBILITY_FIELDS.items() if v in present})
    rates = fill_rates(corpus, probe, schema=pf.schema_arrow)
    total_rows = rates.pop("_total_rows")["rows"]

    def rate(label: str) -> float:
        r = rates.get(label, {})
        return float(r.get("non_empty_rate", r.get("fill_rate", 0.0)))

    hard_empty = [f for f in HARD_FIELDS if f in present and rate(f) < 0.5]

    # --- 판정 --------------------------------------------------------
    min_ref_rate = float(cfg.get("data.min_reference_fill_rate", 0.5))
    min_cred_rate = float(cfg.get("data.min_credibility_fill_rate", 0.5))
    ref_rate = rate("references")
    citation_mode = (
        "references" if ("references" in present and ref_rate >= min_ref_rate) else "chunk"
    )
    cred_rate = max((rate(k) for k in CREDIBILITY_FIELDS if k in rates), default=0.0)
    referencerisk_rate = rate("referencerisk")
    credibility_usable = referencerisk_rate >= min_cred_rate

    passed = not hard_missing and not hard_empty

    shards_read = []
    download_summary: dict[str, Any] = {}
    if stats_path.exists():
        with io.open(stats_path, encoding="utf-8") as fh:
            download_summary = json.load(fh)
        shards_read = [s["shard"] for s in download_summary.get("shards", [])]

    print("코퍼스 SHA256 계산 중 ...", flush=True)
    corpus_sha = sha256_file(corpus)

    manifest: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "spec": "SPEC_M1_TestA_v1.1.md §2.2",
        "snapshot": {
            "repo": cfg["data.hf_dataset"],
            "revision": cfg["data.hf_revision"],
            "config_name": cfg["data.hf_config"],
            "language": cfg["data.language"],
            "license": "CC BY-SA 4.0",
            "attribution": "Wikimedia Enterprise / Wikipedia contributors (CC BY-SA 4.0)",
        },
        "corpus": {
            "path": str(corpus),
            "sha256": corpus_sha,
            "bytes": corpus.stat().st_size,
            "n_documents": n_rows,
            "row_groups": pf.metadata.num_row_groups,
            "sampling": {
                "method": "seeded stratified sample over all upstream shards "
                          "(row-group range reads)",
                "seed": cfg["seed"],
                "row_groups_per_shard": cfg.get("data.row_groups_per_shard", None),
                "min_abstract_chars": cfg.get("data.min_abstract_chars", None),
                "n_shards": download_summary.get("n_shards"),
                "shortfall": download_summary.get("shortfall"),
            },
        },
        "upstream_files": upstream_provenance(cfg, shards_read),
        "schema": {
            "n_columns": len(pf.schema_arrow),
            "flattened_paths": schema_paths,
        },
        "field_check": {
            "hard_required": list(HARD_FIELDS),
            "hard_missing": hard_missing,
            "hard_present_but_empty": hard_empty,
            "soft_missing": soft_missing,
            "credibility_missing": cred_missing,
            "fill_rates": rates,
        },
        "verdicts": {
            "passed": passed,
            "citation_mode": citation_mode,
            "references_fill_rate": ref_rate,
            "referencerisk_fill_rate": referencerisk_rate,
            "credibility_signal_max_fill_rate": cred_rate,
            "credibility_signal_usable": credibility_usable,
            "redirect_field_present": bool(redirect_found),
            "redirect_fields_found": redirect_found,
            "heldout_match_mode": "redirect_field" if redirect_found else "title_normalization",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pyarrow": __import__("pyarrow").__version__,
            "duckdb": __import__("duckdb").__version__,
        },
        "elapsed_s": None,
    }
    manifest["elapsed_s"] = round(time.time() - t0, 1)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    # --- 보고 --------------------------------------------------------
    print()
    print("문서 " + str(total_rows) + "건 · 컬럼 " + str(len(pf.schema_arrow)) + "개")
    print("SHA256 " + corpus_sha)
    print()
    print("필드            존재   채움률")
    for f in HARD_FIELDS + SOFT_FIELDS + INFO_FIELDS:
        mark = "O" if f in present else "X"
        print("  %-14s %s     %s" % (f, mark, ("%.4f" % rate(f)) if f in rates else "-"))
    for k in CREDIBILITY_FIELDS:
        mark = "O" if CREDIBILITY_FIELDS[k] in present else "X"
        print("  %-14s %s     %s" % (k, mark, ("%.4f" % rate(k)) if k in rates else "-"))
    print()
    print("citation_mode           = " + citation_mode)
    print("redirect 필드           = " + (", ".join(redirect_found) if redirect_found else "없음"))
    print("heldout_match_mode      = " + manifest["verdicts"]["heldout_match_mode"])
    print("credibility_signal_usable = " + str(credibility_usable)
          + "  (referencerisk 채움률 %.4f, 기준 %.2f)" % (referencerisk_rate, min_cred_rate))
    print()
    print("MANIFEST -> " + str(manifest_path))

    if not passed:
        print()
        print("FAIL — 필수 필드 문제. CLAUDE.md 재현성 규칙상 실험을 시작하면 안 된다.")
        if hard_missing:
            print("  스키마에 없음: " + repr(hard_missing))
        if hard_empty:
            print("  존재하나 채움률 50% 미만: " + repr(hard_empty))
        return 1
    print("PASS — 필수 필드 충족.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
