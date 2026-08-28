"""BM25 검색기. M1 에서 retrieval 은 전 조건 공통 고정 변인이다.

dense 인덱스를 만들지 않는다 (CLAUDE.md STOP RULE / SPEC §2.6).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Sequence

from .types import Chunk

__all__ = ["BM25Retriever", "ListRetriever"]


class BM25Retriever:
    """`chunk_index.py` 가 저장한 인덱스를 읽어 쓴다."""

    def __init__(self, cfg) -> None:
        import bm25s

        index_dir = Path(cfg["paths.index_dir"])
        if not index_dir.exists():
            raise SystemExit(
                "인덱스가 없다: " + str(index_dir) + " — data/chunk_index.py 를 먼저 돌려라"
            )
        self._bm25s = bm25s
        self.retriever = bm25s.BM25.load(str(index_dir), load_corpus=False)
        self.chunks: list[Chunk] = []
        with io.open(index_dir / "chunks.jsonl", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                self.chunks.append(
                    Chunk(
                        chunk_id=d["chunk_id"],
                        doc_id=int(d["doc_id"]),
                        title=d["title"],
                        section_path=d["section_path"],
                        text=d["text"],
                        source_url=d.get("source_url", ""),
                    )
                )
        self.n_searches = 0

    def search(self, query: str, top_k: int) -> list[Chunk]:
        self.n_searches += 1
        tok = self._bm25s.tokenize([query], stopwords="en", show_progress=False)
        idx, scores = self.retriever.retrieve(tok, k=min(top_k, len(self.chunks)),
                                              show_progress=False)
        out: list[Chunk] = []
        for j, s in zip(idx[0], scores[0]):
            if s <= 0:
                continue
            out.append(self.chunks[int(j)])
        return out


class ListRetriever:
    """테스트·스모크용. 주어진 청크에서 어휘 겹침 상위 k 개를 돌려준다 — 결정론적."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.n_searches = 0

    def search(self, query: str, top_k: int) -> list[Chunk]:
        self.n_searches += 1
        q = set(query.lower().split())
        scored = []
        for i, c in enumerate(self.chunks):
            words = set((c.title + " " + c.text).lower().split())
            overlap = len(q & words)
            if overlap:
                scored.append((-overlap, i, c))
        scored.sort(key=lambda t: (t[0], t[1]))
        return [c for _, _, c in scored[:top_k]]
