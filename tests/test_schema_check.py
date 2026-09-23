from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import schema_check as sc  # noqa: E402


def make_corpus(path: Path, *, n=8, with_references=True, credibility_filled=0,
                empty_abstract=False, with_redirects=False) -> Path:
    """상류 스키마를 축소 재현한 합성 parquet."""
    scores = pa.array(
        [
            {"referencerisk": (0.5 if i < credibility_filled else None), "referenceneed": None}
            for i in range(n)
        ],
        type=pa.struct([("referencerisk", pa.float64()), ("referenceneed", pa.float64())]),
    )
    cols = {
        "identifier": pa.array(list(range(n)), pa.int64()),
        "name": pa.array(["Article " + str(i) for i in range(n)]),
        "url": pa.array(["https://en.wikipedia.org/wiki/A" + str(i) for i in range(n)]),
        "abstract": pa.array(["" if empty_abstract else ("요약 " * 60) for _ in range(n)]),
        "sections": pa.array([json.dumps([{"name": "Intro", "value": "x"}]) for _ in range(n)]),
        "infoboxes": pa.array([json.dumps([{"name": "ib"}]) for _ in range(n)]),
        "tables": pa.array(["[]" for _ in range(n)]),
        "description": pa.array(["desc" for _ in range(n)]),
        "version": pa.StructArray.from_arrays([scores], names=["scores"]),
    }
    if with_references:
        cols["references"] = pa.array(
            [[{"identifier": "r1", "text": "ref"}] for _ in range(n)],
            type=pa.list_(pa.struct([("identifier", pa.string()), ("text", pa.string())])),
        )
    if with_redirects:
        cols["redirects"] = pa.array([["Alt " + str(i)] for i in range(n)], pa.list_(pa.string()))
    pq.write_table(pa.table(cols), path)
    return path


@pytest.fixture
def cfg_file(tmp_path, repo_root) -> Path:
    data = yaml.safe_load(io.open(repo_root / "config.yaml", encoding="utf-8"))
    data["paths"]["data_dir"] = str(tmp_path)
    p = tmp_path / "config.yaml"
    with io.open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True)
    return p


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """MANIFEST 의 상류 SHA 조회는 네트워크다. 단위 테스트에서는 끊는다."""
    monkeypatch.setattr(sc, "upstream_provenance", lambda cfg, shards: [])


def run(cfg_file, corpus, out) -> tuple[int, dict]:
    code = sc.main(["--config", str(cfg_file), "--corpus", str(corpus), "--out", str(out)])
    manifest = json.loads(io.open(out, encoding="utf-8").read()) if out.exists() else {}
    return code, manifest


# --- 스키마 평탄화 -------------------------------------------------------


def test_flatten_schema_reaches_nested_struct_paths(tmp_path):
    p = make_corpus(tmp_path / "c.parquet")
    paths = sc.flatten_schema(pq.ParquetFile(p).schema_arrow)
    assert "version.scores.referencerisk" in paths
    assert "version" in paths and "abstract" in paths


def test_sha256_is_stable_and_content_sensitive(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"hello")
    b.write_bytes(b"hello!")
    assert sc.sha256_file(a) == sc.sha256_file(a)
    assert sc.sha256_file(a) != sc.sha256_file(b)


# --- 판정 ---------------------------------------------------------------


def test_healthy_corpus_passes_and_writes_manifest(tmp_path, cfg_file):
    corpus = make_corpus(tmp_path / "corpus.parquet")
    out = tmp_path / "MANIFEST.json"
    code, m = run(cfg_file, corpus, out)
    assert code == 0
    assert m["verdicts"]["passed"] is True
    assert m["corpus"]["n_documents"] == 8
    assert len(m["corpus"]["sha256"]) == 64
    assert m["snapshot"]["revision"]


def test_empty_hard_field_fails_the_gate(tmp_path, cfg_file):
    """CLAUDE.md: 스키마 검사 실패 시 실험 시작 금지."""
    corpus = make_corpus(tmp_path / "corpus.parquet", empty_abstract=True)
    code, m = run(cfg_file, corpus, tmp_path / "MANIFEST.json")
    assert code == 1
    assert m["verdicts"]["passed"] is False
    assert "abstract" in m["field_check"]["hard_present_but_empty"]


def test_presence_alone_does_not_pass_a_field():
    """스키마에 있는 것과 값이 있는 것은 다르다 — 이 구분이 이 스크립트의 존재 이유다."""
    assert "abstract" in sc.HARD_FIELDS


def test_missing_references_falls_back_to_chunk_citation_mode(tmp_path, cfg_file):
    corpus = make_corpus(tmp_path / "corpus.parquet", with_references=False)
    code, m = run(cfg_file, corpus, tmp_path / "MANIFEST.json")
    assert code == 0, "references 부재는 SOFT — 실험은 계속된다"
    assert m["verdicts"]["citation_mode"] == "chunk"
    assert "references" in m["field_check"]["soft_missing"]


def test_present_references_keep_reference_citation_mode(tmp_path, cfg_file):
    corpus = make_corpus(tmp_path / "corpus.parquet", with_references=True)
    _, m = run(cfg_file, corpus, tmp_path / "MANIFEST.json")
    assert m["verdicts"]["citation_mode"] == "references"


def test_sparse_credibility_signal_is_flagged_unusable(tmp_path, cfg_file):
    """referencerisk 는 컬럼으로 존재해도 채움률이 낮으면 M2 Test C-2 근거가 못 된다."""
    corpus = make_corpus(tmp_path / "corpus.parquet", n=100, credibility_filled=1)
    _, m = run(cfg_file, corpus, tmp_path / "MANIFEST.json")
    assert m["verdicts"]["credibility_signal_usable"] is False
    assert m["verdicts"]["referencerisk_fill_rate"] == pytest.approx(0.01)


def test_dense_credibility_signal_is_usable(tmp_path, cfg_file):
    corpus = make_corpus(tmp_path / "corpus.parquet", n=100, credibility_filled=90)
    _, m = run(cfg_file, corpus, tmp_path / "MANIFEST.json")
    assert m["verdicts"]["credibility_signal_usable"] is True


def test_absent_redirect_field_selects_title_normalization(tmp_path, cfg_file):
    corpus = make_corpus(tmp_path / "corpus.parquet", with_redirects=False)
    _, m = run(cfg_file, corpus, tmp_path / "MANIFEST.json")
    assert m["verdicts"]["redirect_field_present"] is False
    assert m["verdicts"]["heldout_match_mode"] == "title_normalization"


def test_present_redirect_field_selects_redirect_mode(tmp_path, cfg_file):
    corpus = make_corpus(tmp_path / "corpus.parquet", with_redirects=True)
    _, m = run(cfg_file, corpus, tmp_path / "MANIFEST.json")
    assert m["verdicts"]["redirect_field_present"] is True
    assert m["verdicts"]["heldout_match_mode"] == "redirect_field"


def test_missing_corpus_returns_error(tmp_path, cfg_file):
    assert sc.main(["--config", str(cfg_file), "--corpus", str(tmp_path / "nope.parquet")]) == 1


def test_manifest_records_license_attribution(tmp_path, cfg_file):
    """CC BY-SA 저작자표시는 제출 전 체크리스트 항목이다."""
    corpus = make_corpus(tmp_path / "corpus.parquet")
    _, m = run(cfg_file, corpus, tmp_path / "MANIFEST.json")
    assert "CC BY-SA" in m["snapshot"]["license"]
    assert m["snapshot"]["attribution"]
