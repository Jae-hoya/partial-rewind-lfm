"""SPEC §2.6 — 섹션 단위 청킹 + BM25 인덱스.

**`heldout_ids.json` 이 없으면 에러로 종료한다** (SPEC 변경 #3). 순서를 어기면
held-out 누수가 생긴다: 정답 문서가 인덱스에 남은 채로 검색되면 낭비율이 아니라
정답 유출을 측정하게 된다.

- 청킹: 구조화 JSON 의 section 경계 그대로. 1,200 토큰 초과 시 문단 분할.
- 토크나이저: tiktoken `o200k_base`. `cl100k_base` 는 한국어에서 1.57배 부풀린다
  (여기 코퍼스는 영어지만 계측 일관성을 위해 통일한다).
- 인덱스: sparse lexical BM25 (`bm25s`). **dense 인덱스를 만들지 말 것** — M1 에서
  retrieval 은 전 조건 공통 고정 변인이다.
- chunk_id = `d{doc_id}s{section_idx}p{part}`. 재빌드해도 같은 id 가 나온다.
  캐시된 답안이 chunk_id 를 인용하므로 id 가 흔들리면 grounding 이 엉뚱한 본문에
  대해 평가된다.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lfm.config import load_config  # noqa: E402

SKIP_SECTIONS = {
    "references", "external links", "further reading", "see also", "notes",
    "bibliography", "sources", "citations", "footnotes",
}


def walk_sections(node: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, str]]:
    """(section_path, text) 를 결정론적 순회 순서로 낸다."""
    if isinstance(node, list):
        for item in node:
            yield from walk_sections(item, path)
        return
    if not isinstance(node, dict):
        return

    ntype = node.get("type")
    if ntype == "section":
        name = str(node.get("name") or "").strip()
        new_path = path + (name,) if name else path
        if name.lower() in SKIP_SECTIONS:
            return
        paragraphs: list[str] = []
        children: list[Any] = []
        for part in node.get("has_parts") or []:
            if isinstance(part, dict) and part.get("type") == "paragraph":
                v = str(part.get("value") or "").strip()
                if v:
                    paragraphs.append(v)
            else:
                children.append(part)
        if paragraphs:
            yield (" > ".join(new_path) or "Body", "\n\n".join(paragraphs))
        for child in children:
            yield from walk_sections(child, new_path)


def split_by_tokens(text: str, enc, max_tokens: int) -> list[str]:
    """섹션이 max_tokens 를 넘으면 문단 경계로 나눈다. 문단 하나가 넘으면 토큰으로 자른다."""
    if len(enc.encode(text)) <= max_tokens:
        return [text]
    out: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for para in text.split("\n\n"):
        n = len(enc.encode(para))
        if n > max_tokens:
            if buf:
                out.append("\n\n".join(buf))
                buf, buf_len = [], 0
            ids = enc.encode(para)
            for i in range(0, len(ids), max_tokens):
                out.append(enc.decode(ids[i : i + max_tokens]))
            continue
        if buf_len + n > max_tokens and buf:
            out.append("\n\n".join(buf))
            buf, buf_len = [], 0
        buf.append(para)
        buf_len += n
    if buf:
        out.append("\n\n".join(buf))
    return out


def build_chunks(corpus: Path, heldout: set[int], cfg) -> tuple[list[dict[str, Any]], dict]:
    import pyarrow.parquet as pq
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    max_tokens = int(cfg.get("retrieval.chunk_max_tokens", 1200))
    max_chunks = int(cfg.get("retrieval.max_chunks", 4000000))

    chunks: list[dict[str, Any]] = []
    n_docs = n_skipped = 0
    total_tokens = 0
    pf = pq.ParquetFile(corpus)

    for batch in pf.iter_batches(
        batch_size=2000, columns=["identifier", "name", "url", "abstract", "sections"]
    ):
        for row in batch.to_pylist():
            doc_id = int(row["identifier"])
            if doc_id in heldout:
                n_skipped += 1
                continue
            n_docs += 1
            try:
                sections = json.loads(row["sections"]) if row["sections"] else []
            except (json.JSONDecodeError, TypeError):
                sections = []
            for sec_idx, (section_path, text) in enumerate(walk_sections(sections)):
                for part_idx, piece in enumerate(split_by_tokens(text, enc, max_tokens)):
                    ntok = len(enc.encode(piece))
                    total_tokens += ntok
                    chunks.append(
                        {
                            "chunk_id": "d%ds%dp%d" % (doc_id, sec_idx, part_idx),
                            "doc_id": doc_id,
                            "title": row["name"],
                            "section_path": section_path,
                            "text": piece,
                            "source_url": row["url"] or "",
                            "n_tokens": ntok,
                        }
                    )
            if len(chunks) > max_chunks:
                raise SystemExit(
                    "청크 수가 retrieval.max_chunks(%d)를 넘었다. 중단하고 보고한다 "
                    "(SPEC §2.6)." % max_chunks
                )

    stats = {
        "n_documents_indexed": n_docs,
        "n_documents_heldout_skipped": n_skipped,
        "n_chunks": len(chunks),
        "total_tokens": total_tokens,
        "mean_chunk_tokens": round(total_tokens / len(chunks), 2) if chunks else 0.0,
        "tokenizer": "o200k_base",
        "chunk_max_tokens": max_tokens,
    }
    return chunks, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="섹션 청킹 + BM25 인덱스 (SPEC §2.6)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-coverage", action="store_true", help="커버리지 검증 생략(디버그용)")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config or (repo_root / "config.yaml"))
    data_dir = Path(cfg["paths.data_dir"])
    index_dir = Path(cfg["paths.index_dir"])
    corpus = data_dir / "corpus.parquet"
    heldout_path = data_dir / "heldout_ids.json"

    # SPEC 변경 #3 — 순서 강제
    if not heldout_path.exists():
        print("heldout_ids.json 이 없다: " + str(heldout_path))
        print("held-out 확정 전에 인덱스를 만들면 정답 문서가 검색된다. 에러로 종료한다 "
              "(SPEC §2.6).")
        return 1
    with io.open(heldout_path, encoding="utf-8") as fh:
        heldout_doc = json.load(fh)
    heldout = set(int(i) for i in heldout_doc["heldout_identifiers"])
    print("held-out 문서 %d건 제외" % len(heldout))

    t0 = time.time()
    chunks, stats = build_chunks(corpus, heldout, cfg)
    t_chunk = time.time() - t0
    print("청킹 완료: 문서 %d, 청크 %d, 평균 %.1f 토큰, %.1fs"
          % (stats["n_documents_indexed"], stats["n_chunks"],
             stats["mean_chunk_tokens"], t_chunk))

    import bm25s

    t1 = time.time()
    texts = [c["title"] + " " + c["section_path"] + " " + c["text"] for c in chunks]
    tokenized = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokenized, show_progress=False)
    t_index = time.time() - t1

    index_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(index_dir), show_progress=False)
    meta_path = index_dir / "chunks.jsonl"
    with io.open(meta_path, "w", encoding="utf-8", newline="\n") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    index_bytes = sum(p.stat().st_size for p in index_dir.rglob("*") if p.is_file())

    # --- 토픽 커버리지 검증 (SPEC §2.6) --------------------------------
    coverage: dict[str, int] = {}
    below: list[str] = []
    if not args.skip_coverage:
        import yaml

        with io.open(data_dir / "topics_pilot50.yaml", encoding="utf-8") as fh:
            topics = yaml.safe_load(fh)["topics"]
        min_needed = int(cfg["data.min_chunks_per_topic"])
        k = max(min_needed * 3, 60)
        queries = [t["topic"] or t["target_article"] for t in topics]
        qtok = bm25s.tokenize(queries, stopwords="en", show_progress=False)
        res, scores = retriever.retrieve(qtok, k=k, show_progress=False)
        for i, t in enumerate(topics):
            n_hit = int(sum(1 for s in scores[i] if s > 0))
            coverage[str(t["id"])] = n_hit
            if n_hit < min_needed:
                below.append(str(t["id"]))

    stats.update(
        {
            "chunking_seconds": round(t_chunk, 1),
            "indexing_seconds": round(t_index, 1),
            "index_bytes": index_bytes,
            "index_dir": str(index_dir),
            "backend": cfg["retrieval.backend"],
            "coverage_per_topic": coverage,
            "topics_below_min": below,
            "min_chunks_per_topic": cfg["data.min_chunks_per_topic"],
        }
    )
    with io.open(data_dir / "index_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    print("인덱스 빌드 %.1fs, 크기 %.1f MB -> %s"
          % (t_index, index_bytes / 1e6, index_dir))
    print("통계 -> " + str(data_dir / "index_stats.json"))
    if below:
        print()
        print("FAIL — 관련 청크가 min_chunks_per_topic(%d) 미만인 토픽 %d개: %s"
              % (cfg["data.min_chunks_per_topic"], len(below), below))
        print("중단하고 보고한다 (SPEC §2.6).")
        return 1
    print("PASS — 전 토픽이 커버리지 기준을 넘었다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
