"""PREREG_R3 §4.2 — 코퍼스2(arXiv 제목 + 초록)를 코퍼스1 스키마의 `data_c2/corpus.parquet` 로 만든다.

**C0·C1-C2 시점에는 실행하지 않는다.** 실행은 C1-C2 커밋 뒤 D0 의 첫 코퍼스2 단계다.

  python data_c2/download_c2.py --config config_c2.yaml                       # 받기 + 만들기 + MANIFEST
  python data_c2/download_c2.py --config config_c2.yaml --build-only --raw-dir <dir>   # 받은 파일로 만들기만 (네트워크 없음)

절차 (§4.2, seed 4500 — config 의 `seed` 42 가 아니다)
  1. 원천 revision 을 고정한다: config 의 data.hf_revision 이 40자리 해시면 그것, 자리표시면 지금 시점의 해시를
     `HfApi().dataset_info` 로 한 번 읽어 고정하고 MANIFEST 에 적는다(이후 `c2_config_diff.py --write` 가 config 에 옮긴다).
  2. `hf_file_glob` 에 맞는 parquet 만 그 revision 으로 받는다(`snapshot_download(allow_patterns=…)`), `data_c2/raw/`.
  3. 1차 통과(필요 열만): arXiv id 당 한 행(같은 id 가 여러 행이면 update_date 가 가장 늦은 행, 같으면 뒤 파일·뒤 행) →
     초록(공백만 정규화) 400자 미만 · "withdrawn" 으로 시작(대소문자 무시) · 제목/초록 빈 행 제외.
  4. 적격 행을 arXiv id 사전순으로 놓고 `default_rng(4500).choice(N, 150000, replace=False)` 단순 무작위.
  5. 뽑힌 행에 arXiv id 사전순으로 identifier 1.. 을 준다(`data_c2/id_map.parquet`). 출력 행은 identifier 순.

필드 대응 (코퍼스1 스키마 — chunk_index.py·topics.py·heldout.py 를 바꾸지 않으려고)
  identifier int64 · name 제목(공백 정규화) · url https://arxiv.org/abs/{id} · abstract 초록(공백 정규화, LaTeX 그대로)
  sections  JSON 문자열 [{"type":"section","name":"Abstract","has_parts":[{"type":"paragraph","value":초록}]}]
  infoboxes JSON 문자열 [{"name": 분류}, …] (categories 공백 분리 순서 그대로; 첫 분류 = 주 분류)
  references list<string> 빈 목록 (topics.load_meta 가 len("references") 를 읽는다)
  arxiv_id, categories, license(원천의 문서별 license — **본문 라이선스이지 메타데이터 라이선스가 아니다**) 는 추가 열.
archive = 분류의 점 앞 접두사 (`cs.LG` → `cs`, `hep-th` → `hep-th`, `cond-mat.mes-hall` → `cond-mat`).

원천 스키마는 Kaggle `arxiv-metadata-oai-snapshot` 과 같다고 **가정**한다(id, title, abstract, categories, update_date,
license, …). 받기 전에는 확인할 수 없으므로 1차 통과 첫 파일에서 필수 열 존재를 assert 한다.
열 이름은 같지만 형식 하나가 다르다: HF `librarian-bots/arxiv-metadata-snapshot` parquet 의 update_date 는 Kaggle JSON 의
문자열이 아니라 `timestamp[s]` 다. 1차 통과는 이를 문자열 'YYYY-MM-DD HH:MM:SS'(null → '')로 바꿔(`cast(pa.string())`)
id 중복 비교에 쓴다. 이 문자열은 시간 순서대로 정렬되므로 "update_date 가 가장 늦은 행" 규칙이 그대로 성립한다
(tests/test_c2_pipeline.py 의 timestamp 시험).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lfm.config import load_config  # noqa: E402

SEED = 4500
N_SAMPLE = 150_000
REQUIRED_COLS = ("id", "title", "abstract", "categories")
OPTIONAL_COLS = ("update_date", "license")
_WS = re.compile(r"\s+")


def archive_of(category: str) -> str:
    return category.split(".", 1)[0]


def norm_ws(s: str | None) -> str:
    return _WS.sub(" ", s or "").strip()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with io.open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- 받기 (네트워크)
def resolve_revision(repo: str, revision: str) -> tuple[str, dict[str, Any]]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo, revision=None if not re.fullmatch(r"[0-9a-f]{40}", revision) else revision,
                                files_metadata=True)
    card = getattr(info, "card_data", None)
    meta = {"resolved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "last_modified": str(info.last_modified),
            "card_license": (card.get("license") if card else None),
            "files": {s.rfilename: s.size for s in info.siblings}}
    return info.sha, meta


def fetch(repo: str, revision: str, glob: str, raw_dir: Path) -> list[Path]:
    from huggingface_hub import snapshot_download

    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, repo_type="dataset", revision=revision, allow_patterns=[glob],
                      local_dir=str(raw_dir))
    return raw_files(raw_dir, glob)


def raw_files(raw_dir: Path, glob: str) -> list[Path]:
    files = sorted(p for p in Path(raw_dir).glob(glob) if p.is_file())
    if not files:
        raise SystemExit("원천 parquet 이 없다: %s / %s" % (raw_dir, glob))
    return files


# --------------------------------------------------------------------------- 만들기 (오프라인)
def _norm_col(col: pa.ChunkedArray) -> pa.ChunkedArray:
    col = pc.fill_null(col.cast(pa.string()), "")
    return pc.utf8_trim_whitespace(pc.replace_substring_regex(col, pattern=r"\s+", replacement=" "))


def pass1(files: list[Path], min_abs: int) -> tuple[list[tuple[str, str, int, int]], dict[str, Any]]:
    """적격 행의 (arxiv_id, update_date, file_idx, row_idx). id 중복은 여기서 하나로 줄인다."""
    excl = Counter()
    n_raw = 0
    best: dict[str, tuple[str, int, int, bool]] = {}     # id -> (update_date, file_idx, row_idx, eligible)
    cols_seen: list[str] = []
    for fi, f in enumerate(files):
        names = pq.read_schema(f).names
        if fi == 0:
            cols_seen = list(names)
            missing = [c for c in REQUIRED_COLS if c not in names]
            assert not missing, "원천 스키마에 필수 열이 없다: %s (있는 열 %s)" % (missing, names)
        cols = list(REQUIRED_COLS) + [c for c in ("update_date",) if c in names]
        t = pq.read_table(f, columns=cols)
        n_raw += t.num_rows
        ids = pc.utf8_trim_whitespace(pc.fill_null(t.column("id").cast(pa.string()), "")).to_pylist()
        title = _norm_col(t.column("title"))
        abstract = _norm_col(t.column("abstract"))
        alen = pc.utf8_length(abstract).to_pylist()
        tlen = pc.utf8_length(title).to_pylist()
        withdrawn = pc.starts_with(pc.utf8_lower(abstract), "withdrawn").to_pylist()
        upd = (pc.fill_null(t.column("update_date").cast(pa.string()), "").to_pylist()
               if "update_date" in t.column_names else [""] * t.num_rows)
        for ri in range(t.num_rows):
            aid = ids[ri]
            if not aid:
                excl["empty_id"] += 1
                continue
            ok = True
            if tlen[ri] == 0 or alen[ri] == 0:
                excl["empty_title_or_abstract"] += 1; ok = False
            elif withdrawn[ri]:
                excl["withdrawn"] += 1; ok = False
            elif alen[ri] < min_abs:
                excl["abstract_lt_%d" % min_abs] += 1; ok = False
            cur = best.get(aid)
            cand = (upd[ri], fi, ri, ok)
            if cur is not None:
                excl["duplicate_id_rows"] += 1
            if cur is None or (cand[0], cand[1], cand[2]) >= (cur[0], cur[1], cur[2]):
                best[aid] = cand
    elig = sorted((aid, v[0], v[1], v[2]) for aid, v in best.items() if v[3])
    return elig, {"n_rows_raw": n_raw, "n_unique_ids": len(best), "n_eligible": len(elig),
                  "excluded_row_counts": dict(excl), "columns_seen": cols_seen,
                  "dedup_rule": "arXiv id 당 update_date 최대 행(같으면 뒤 파일·뒤 행). 적격 판정은 남은 행으로"}


def sample(elig: list, n: int, seed: int = SEED) -> list:
    if len(elig) < n:
        raise SystemExit("적격 %d건 < 표집 %d건" % (len(elig), n))
    rng = np.random.default_rng(seed)
    idx = sorted(int(i) for i in rng.choice(len(elig), size=n, replace=False))
    return [elig[i] for i in idx]                       # elig 가 id 사전순이므로 결과도 id 사전순


def corpus_record(identifier: int, aid: str, title: str, abstract: str, categories: str, license_: str | None) -> dict:
    cats = [c for c in (categories or "").split() if c]
    return {
        "identifier": identifier,
        "name": title,
        "url": "https://arxiv.org/abs/" + aid,
        "abstract": abstract,
        "sections": json.dumps([{"type": "section", "name": "Abstract",
                                 "has_parts": [{"type": "paragraph", "value": abstract}]}], ensure_ascii=False),
        "infoboxes": json.dumps([{"name": c} for c in cats], ensure_ascii=False),
        "references": [],
        "arxiv_id": aid,
        "categories": " ".join(cats),
        "license": license_ or "",
    }


SCHEMA = pa.schema([
    ("identifier", pa.int64()), ("name", pa.string()), ("url", pa.string()), ("abstract", pa.string()),
    ("sections", pa.string()), ("infoboxes", pa.string()), ("references", pa.list_(pa.string())),
    ("arxiv_id", pa.string()), ("categories", pa.string()), ("license", pa.string()),
])


def pass2(files: list[Path], picked: list) -> list[dict]:
    """뽑힌 행을 읽어 코퍼스 레코드로. identifier = id 사전순 1.."""
    ident = {aid: i + 1 for i, (aid, _u, _f, _r) in enumerate(picked)}
    by_file: dict[int, list[tuple[int, str]]] = {}
    for aid, _u, fi, ri in picked:
        by_file.setdefault(fi, []).append((ri, aid))
    recs: list[dict] = []
    for fi, rows in sorted(by_file.items()):
        names = pq.read_schema(files[fi]).names
        cols = list(REQUIRED_COLS) + [c for c in ("license",) if c in names]
        t = pq.read_table(files[fi], columns=cols)
        rows.sort()
        take = t.take(pa.array([r for r, _ in rows], type=pa.int64()))
        title = _norm_col(take.column("title")).to_pylist()
        abstract = _norm_col(take.column("abstract")).to_pylist()
        cats = pc.fill_null(take.column("categories").cast(pa.string()), "").to_pylist()
        lic = (take.column("license").cast(pa.string()).to_pylist() if "license" in take.column_names else [None] * take.num_rows)
        for j, (_r, aid) in enumerate(rows):
            recs.append(corpus_record(ident[aid], aid, title[j], abstract[j], cats[j], lic[j]))
    recs.sort(key=lambda r: r["identifier"])
    return recs


def build(files: list[Path], out_dir: Path, *, min_abs: int, n_sample: int = N_SAMPLE, seed: int = SEED) -> dict[str, Any]:
    t0 = time.time()
    elig, s1 = pass1(files, min_abs)
    picked = sample(elig, n_sample, seed)
    recs = pass2(files, picked)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(recs, schema=SCHEMA), out_dir / "corpus.parquet", compression="zstd")
    pq.write_table(pa.table({"identifier": [r["identifier"] for r in recs], "arxiv_id": [r["arxiv_id"] for r in recs]}),
                   out_dir / "id_map.parquet")
    prim = Counter(archive_of(r["categories"].split()[0]) if r["categories"] else "(none)" for r in recs)
    anyc = Counter(a for r in recs for a in {archive_of(c) for c in r["categories"].split()})
    lic = Counter(r["license"] or "(empty)" for r in recs)
    return {**s1, "seed": seed, "n_sampled": len(recs),
            "sample_rule": "적격 행 arXiv id 사전순 → default_rng(%d).choice(N, %d, replace=False)" % (seed, n_sample),
            "identifier_rule": "뽑힌 행의 arXiv id 사전순 1..N (data_c2/id_map.parquet)",
            "archive_definition": "분류의 점 앞 접두사 (cs.LG→cs, hep-th→hep-th)",
            "archive_distribution_primary": dict(prim.most_common()),
            "archive_distribution_any": dict(anyc.most_common()),
            "per_document_license_field": {"note": "원천의 문서별 license 는 본문 라이선스이지 메타데이터(초록) 라이선스가 아니다",
                                           "distribution": dict(lic.most_common())},
            "corpus_sha256": sha256_file(out_dir / "corpus.parquet"),
            "id_map_sha256": sha256_file(out_dir / "id_map.parquet"),
            "build_seconds": round(time.time() - t0, 1)}


def write_manifest(path: Path, payload: dict) -> None:
    io.open(path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="코퍼스2 다운로드·구성 (PREREG_R3 §4.2)")
    ap.add_argument("--config", default=str(ROOT / "config_c2.yaml"))
    ap.add_argument("--build-only", action="store_true", help="받지 않고 --raw-dir 의 파일로 만든다")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out-dir", default=None, help="기본 config 의 paths.data_dir")
    ap.add_argument("--n-sample", type=int, default=N_SAMPLE)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    repo, rev, glob = cfg["data.hf_dataset"], str(cfg["data.hf_revision"]), cfg["data.hf_file_glob"]
    out_dir = Path(args.out_dir or cfg["paths.data_dir"])
    raw_dir = Path(args.raw_dir or cfg["paths.raw_dir"])
    min_abs = int(cfg["data.min_abstract_chars"])
    manifest_path = out_dir / "MANIFEST_C2.json"

    t0 = time.time()
    src: dict[str, Any] = {"hf_dataset": repo, "hf_config": cfg["data.hf_config"], "hf_file_glob": glob}
    if args.build_only:
        files = raw_files(raw_dir, glob)
        src.update({"hf_revision": rev, "fetched": False})
    else:
        sha, meta = resolve_revision(repo, rev)
        if re.fullmatch(r"[0-9a-f]{40}", rev) and sha != rev:
            raise SystemExit("revision 불일치: config %s, 원천 %s" % (rev, sha))
        print("원천 %s @ %s (card license %s)" % (repo, sha, meta["card_license"]), flush=True)
        files = fetch(repo, sha, glob, raw_dir)
        src.update({"hf_revision": sha, "fetched": True, "source_url": "https://huggingface.co/datasets/%s/tree/%s" % (repo, sha),
                    **meta, "download_seconds": round(time.time() - t0, 1)})
    src["raw_files"] = {str(p.relative_to(raw_dir)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                        for p in files}
    print("원천 파일 %d개, %.2f GB" % (len(files), sum(p.stat().st_size for p in files) / 1e9), flush=True)
    stats = build(files, out_dir, min_abs=min_abs, n_sample=args.n_sample)
    payload = {
        "spec": "PREREG_R3 §4.2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": src,
        "corpus": stats,
        "field_mapping": {
            "identifier": "arXiv id 사전순 정수 (chunk_id d{n}s0p0)", "name": "title (공백 정규화)",
            "url": "https://arxiv.org/abs/{id}", "abstract": "abstract (공백만 정규화, LaTeX 그대로)",
            "sections": "[Abstract 섹션 1개, 문단 1개 = 초록]", "infoboxes": "[{name: 분류}] categories 순서",
            "references": "빈 목록 열 (topics.load_meta 호환)"},
        "license_gate": {"status": "pending",
                         "rule": "arXiv 공식 문서에서 메타데이터(초록 포함)가 CC0 임을 확인해 기록한다. 기록 전에는 토픽 생성·실행 금지",
                         "recorded_by_script": "data_c2/license_gate_c2.py"},
        "elapsed_s": round(time.time() - t0, 1),
    }
    write_manifest(manifest_path, payload)
    print("문서 %d건 -> %s\nMANIFEST -> %s" % (stats["n_sampled"], out_dir / "corpus.parquet", manifest_path))
    print("다음: C-L 기록 (data_c2/license_gate_c2.py), 그다음 experiments/c2_config_diff.py --write --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
