"""chunk_index.py — 청킹·중복 제거·id 안정성."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import chunk_index as ci  # noqa: E402


class FakeEnc:
    """1 단어 = 1 토큰. 손계산이 되도록."""

    def encode(self, t):
        return t.split()

    def decode(self, ids):
        return " ".join(ids)


def section(name, *paragraphs, children=()):
    return {
        "type": "section",
        "name": name,
        "has_parts": [{"type": "paragraph", "value": p} for p in paragraphs] + list(children),
    }


# --- 섹션 순회 -----------------------------------------------------------


def test_walk_yields_section_path_and_text():
    got = list(ci.walk_sections([section("Abstract", "first para", "second para")]))
    assert got == [("Abstract", "first para\n\nsecond para")]


def test_walk_recurses_into_nested_sections():
    doc = [section("History", "top level", children=[section("Early years", "nested text")])]
    got = dict(ci.walk_sections(doc))
    assert got["History"] == "top level"
    assert got["History > Early years"] == "nested text"


def test_reference_sections_are_skipped():
    """참고문헌 목록은 본문이 아니다. 인덱스에 넣으면 검색이 서지정보로 오염된다."""
    doc = [section("Body", "real text"), section("References", "[1] some citation")]
    paths = [p for p, _ in ci.walk_sections(doc)]
    assert paths == ["Body"]


def test_skipping_a_section_also_skips_its_children():
    doc = [section("See also", "x", children=[section("Sub", "y")])]
    assert list(ci.walk_sections(doc)) == []


def test_empty_sections_produce_nothing():
    assert list(ci.walk_sections([section("Empty")])) == []


def test_walk_is_deterministic():
    doc = [section("A", "a"), section("B", "b"), section("C", "c")]
    assert list(ci.walk_sections(doc)) == list(ci.walk_sections(doc))


# --- 토큰 분할 -----------------------------------------------------------


def test_short_section_is_one_chunk():
    assert ci.split_by_tokens("one two three", FakeEnc(), 10) == ["one two three"]


def test_long_section_splits_on_paragraph_boundaries():
    text = "\n\n".join(["a a a", "b b b", "c c c"])
    parts = ci.split_by_tokens(text, FakeEnc(), 6)
    assert parts == ["a a a\n\nb b b", "c c c"]


def test_paragraph_longer_than_limit_is_hard_split():
    parts = ci.split_by_tokens("w " * 10, FakeEnc(), 4)
    assert all(len(p.split()) <= 4 for p in parts)
    assert sum(len(p.split()) for p in parts) == 10


# --- 코퍼스 빌드 ----------------------------------------------------------


def corpus_file(tmp_path, rows) -> Path:
    tbl = pa.table(
        {
            "identifier": pa.array([r[0] for r in rows], pa.int64()),
            "name": pa.array([r[1] for r in rows]),
            "url": pa.array(["u" for _ in rows]),
            "abstract": pa.array(["abs" for _ in rows]),
            "sections": pa.array([json.dumps(r[2]) for r in rows]),
        }
    )
    p = tmp_path / "corpus.parquet"
    pq.write_table(tbl, p)
    return p


def test_heldout_documents_are_excluded(tmp_path, real_config):
    rows = [(1, "Keep", [section("Body", "keep me")]),
            (2, "Drop", [section("Body", "drop me")])]
    chunks, stats = ci.build_chunks(corpus_file(tmp_path, rows), {2}, real_config)
    assert stats["n_documents_heldout_skipped"] == 1
    assert {c["doc_id"] for c in chunks} == {1}


def test_duplicate_rows_are_indexed_once(tmp_path, real_config):
    """상류 샤드가 겹쳐 같은 문서가 여러 행으로 들어온다 (실측 4.7%).
    그대로 두면 chunk_id 가 충돌한다."""
    rows = [(1, "Dup", [section("Body", "same text")])] * 3
    chunks, stats = ci.build_chunks(corpus_file(tmp_path, rows), set(), real_config)
    assert stats["n_documents_indexed"] == 1
    assert stats["n_rows_duplicate_skipped"] == 2
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_chunk_ids_are_unique_and_stable(tmp_path, real_config):
    rows = [(7, "Doc", [section("A", "alpha"), section("B", "beta")])]
    a, _ = ci.build_chunks(corpus_file(tmp_path, rows), set(), real_config)
    b, _ = ci.build_chunks(corpus_file(tmp_path, rows), set(), real_config)
    ids = [c["chunk_id"] for c in a]
    assert ids == [c["chunk_id"] for c in b], "재빌드해도 같은 id 여야 한다"
    assert len(set(ids)) == len(ids)
    assert ids[0].startswith("d7s")


def test_malformed_sections_json_does_not_crash(tmp_path, real_config):
    tbl = pa.table(
        {
            "identifier": pa.array([1], pa.int64()),
            "name": pa.array(["Broken"]),
            "url": pa.array(["u"]),
            "abstract": pa.array(["a"]),
            "sections": pa.array(["{not json"]),
        }
    )
    p = tmp_path / "c.parquet"
    pq.write_table(tbl, p)
    chunks, stats = ci.build_chunks(p, set(), real_config)
    assert chunks == [] and stats["n_documents_indexed"] == 1
