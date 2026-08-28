"""approve_topics.py — 승인 게이트의 검사 항목.

이 게이트가 느슨하면 낭비율이 토픽 난이도의 산물이 된다 (HANDOFF §4).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import approve_topics as ap  # noqa: E402

from lfm.config import Config  # noqa: E402

CFG = Config({"topics": {"min_related_docs": 30, "min_infobox_types": 3}})
TITLES = {"welsh labour": "Welsh Labour", "bharath": "Bharath (actor)"}


def cand(**over):
    base = {
        "id": "t000",
        "topic": "How has the party developed organisationally and politically over recent decades?",
        "target_article": "Welsh Labour",
        "related_docs": 120,
        "n_infobox_types": 12,
        "terms": ["welsh", "labour", "senedd"],
    }
    base.update(over)
    return base


def test_healthy_candidate_passes():
    assert ap.check(cand(), TITLES, CFG) == []


def test_missing_topic_phrase_fails():
    assert ap.check(cand(topic=""), TITLES, CFG)


def test_too_few_related_docs_fails():
    reasons = ap.check(cand(related_docs=5), TITLES, CFG)
    assert any("이웃 문서" in r for r in reasons)


def test_too_few_infobox_types_fails():
    """교차 도메인성이 없으면 토픽이 아니다."""
    reasons = ap.check(cand(n_infobox_types=1), TITLES, CFG)
    assert any("infobox" in r for r in reasons)


def test_target_absent_from_corpus_fails():
    """target 이 없으면 heldout.py 가 뺄 대상이 없다."""
    reasons = ap.check(cand(target_article="Nonexistent Article"), TITLES, CFG)
    assert any("코퍼스에 없음" in r for r in reasons)


def test_title_overlap_is_recorded_but_not_a_rejection():
    """held-out 의 통제 수단은 인덱스에서 문서를 빼는 것이지 질문에서 이름을
    숨기는 것이 아니다. 제목이 한 단어면 주제를 언급만 해도 1.0 이 된다."""
    c = cand(topic="How did Welsh Labour shape devolution policy and internal debate?")
    assert ap.check(c, TITLES, CFG) == []
    assert ap.title_overlap(c["target_article"], c["topic"]) == 1.0


def test_title_overlap_of_unrelated_phrasing_is_low():
    assert ap.title_overlap("Welsh Labour", "How did the regional party evolve?") == 0.0


# --- 계열 중복 ------------------------------------------------------------


def test_family_key_ignores_years():
    a = ap.family_key("Lithuania in the Eurovision Song Contest 2014")
    b = ap.family_key("Lithuania in the Eurovision Song Contest 2023")
    assert a == b


def test_family_key_separates_different_subjects():
    assert ap.family_key("Welsh Labour") != ap.family_key("Scottish Labour")


def test_jaccard_bounds():
    assert ap.jaccard(set(), {"a"}) == 0.0
    assert ap.jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert ap.jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
