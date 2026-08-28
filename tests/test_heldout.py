"""heldout.py — 제목 정규화 매칭과 승인 게이트."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import heldout as ho  # noqa: E402


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Bharath (actor)", "bharath"),
        ("The Beginning After the End", "beginning after the end"),
        ("Ćustić", "custic"),
        ("Newton N. Minow", "newton n minow"),
        ("  Welsh   Labour  ", "welsh labour"),
        ("A Tale of Two Cities", "tale of two cities"),
        ("An Officer", "officer"),
    ],
)
def test_title_normalization(raw, expected):
    assert ho.normalize_title(raw) == expected


def test_normalization_is_idempotent():
    once = ho.normalize_title("The Café (disambiguation)")
    assert ho.normalize_title(once) == once


def test_diacritics_are_folded_not_dropped():
    """NFKD 후 결합문자 제거 — 글자 자체는 남아야 한다."""
    assert ho.normalize_title("Jurislav Korenić") == "jurislav korenic"
    assert ho.normalize_title("Stéphanie Frappart") == "stephanie frappart"


def write(tmp_path, doc) -> Path:
    p = tmp_path / "topics.yaml"
    with io.open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, allow_unicode=True)
    return p


def test_missing_topics_file_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        ho.load_approved(tmp_path / "nope.yaml")


def test_unapproved_file_is_an_error(tmp_path):
    """SPEC §2.4: 경고 후 진행이 아니라 에러 종료다."""
    with pytest.raises(SystemExit):
        ho.load_approved(write(tmp_path, {"approved": False, "topics": [{"id": "t0"}]}))


def test_missing_approved_key_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        ho.load_approved(write(tmp_path, {"topics": [{"id": "t0"}]}))


def test_approved_true_but_no_topics_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        ho.load_approved(write(tmp_path, {"approved": True, "topics": []}))


def test_approved_file_loads(tmp_path):
    doc = ho.load_approved(write(tmp_path, {"approved": True, "topics": [{"id": "t0"}]}))
    assert doc["topics"][0]["id"] == "t0"


def test_string_true_does_not_count_as_approval(tmp_path):
    """approved: "true" (문자열) 는 승인이 아니다. is True 로 본다."""
    with pytest.raises(SystemExit):
        ho.load_approved(write(tmp_path, {"approved": "true", "topics": [{"id": "t0"}]}))
