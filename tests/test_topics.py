"""topics.py — 후보 선별 로직. 실 코퍼스·LLM 없이 합성 문서로 검증한다."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import topics as tp  # noqa: E402

from lfm.config import Config  # noqa: E402


def cfg(**over):
    base = {
        "topics": {
            "min_related_docs": 3,
            "min_infobox_types": 2,
            "min_term_hits": 2,
            "n_terms": 5,
            "df_lo": 2,
            "df_hi": 100,
            "min_target_abstract_chars": 10,
            "min_target_section_chars": 10,
            "min_target_references": 1,
            "max_per_infobox_type": 3,
            "n_candidates": 10,
            "min_term_len": 4,
        }
    }
    base["topics"].update(over)
    return Config(base)


def doc(name, abstract, types):
    return {"identifier": abs(hash(name)) % 10**6, "name": name, "url": "u",
            "abstract": abstract, "infobox_types": types}


def test_tokens_drop_stopwords_and_short_words():
    got = tp.tokens("The quantum of Solar energy is at the CORE", 4)
    assert "quantum" in got and "solar" in got and "energy" in got
    assert "the" not in got and "of" not in got and "is" not in got


def test_stopwords_cover_the_generic_words_that_broke_the_first_attempt():
    """eight/open/point/make 류가 후보를 채웠던 실패의 재발 방지."""
    generic = tp.tokens("world war american state series list total area region", 4)
    assert generic == set(), generic


def test_distinctive_terms_prefer_rare_over_common():
    docs = [doc("D%d" % i, "common common photosynthesis" if i == 0 else "common common",
                ["Infobox a"]) for i in range(10)]
    post = tp.build_postings(docs, 4)
    terms = tp.distinctive_terms(docs[0], post, len(docs), cfg(df_lo=1, df_hi=100))
    assert terms[0] == "photosynthesis", terms


def test_generic_word_is_excluded_by_df_ceiling():
    docs = [doc("D%d" % i, "ubiquitous term everywhere", ["Infobox a"]) for i in range(50)]
    post = tp.build_postings(docs, 4)
    terms = tp.distinctive_terms(docs[0], post, len(docs), cfg(df_lo=2, df_hi=10))
    assert "ubiquitous" not in terms


def build(docs, thick_names=None):
    """thick_names 를 주면 그 문서만 target 후보 자격(두꺼운 본문)을 갖는다.
    이웃까지 target 후보가 되면 무엇을 검증하는지 흐려진다."""
    post = tp.build_postings(docs, 4)
    meta = {
        d["identifier"]: ((99999, 99) if thick_names is None or d["name"] in thick_names
                          else (0, 0))
        for d in docs
    }
    return post, meta


def test_candidate_requires_enough_related_docs():
    docs = [doc("Solar Panel Physics", "photovoltaic silicon junction " * 8, ["Infobox tech"])]
    docs += [doc("Other %d" % i, "unrelated content here entirely", ["Infobox b"])
             for i in range(20)]
    post, meta = build(docs)
    assert tp.find_candidates(docs, post, meta, cfg()) == []


def test_candidate_requires_multiple_infobox_types():
    """이웃이 전부 같은 유형이면 교차 도메인이 아니다."""
    docs = [doc("Hub", "photovoltaic silicon junction wafer " * 8, ["Infobox tech"])]
    docs += [doc("N%d" % i, "photovoltaic silicon junction wafer", ["Infobox same"])
             for i in range(8)]
    post, meta = build(docs, thick_names={"Hub"})
    assert tp.find_candidates(docs, post, meta, cfg(min_infobox_types=2)) == []
    assert tp.find_candidates(docs, post, meta, cfg(min_infobox_types=1)), "유형 1종이면 통과"


def test_cross_domain_candidate_is_found():
    docs = [doc("Hub", "photovoltaic silicon junction wafer " * 8, ["Infobox tech"])]
    docs += [doc("N%d" % i, "photovoltaic silicon junction wafer",
                 ["Infobox " + "abcdefgh"[i]]) for i in range(8)]
    post, meta = build(docs, thick_names={"Hub"})
    got = tp.find_candidates(docs, post, meta, cfg())
    assert got and got[0]["related_docs"] >= 3
    assert got[0]["n_infobox_types"] >= 2


def test_disambiguation_and_list_pages_are_excluded():
    for bad in ("Mercury (disambiguation)", "List of solar plants", "Index of physics"):
        docs = [doc(bad, "photovoltaic silicon junction wafer " * 8, ["Infobox tech"])]
        docs += [doc("N%d" % i, "photovoltaic silicon junction wafer",
                     ["Infobox " + "abcdefgh"[i]]) for i in range(8)]
        post, meta = build(docs, thick_names={bad})
        names = [docs[c["target_idx"]]["name"] for c in tp.find_candidates(docs, post, meta, cfg())]
        assert bad not in names


def test_thin_target_is_excluded():
    docs = [doc("Hub", "photovoltaic silicon junction wafer " * 8, ["Infobox tech"])]
    docs += [doc("N%d" % i, "photovoltaic silicon junction wafer",
                 ["Infobox " + "abcdefgh"[i]]) for i in range(8)]
    post, _ = build(docs)
    thin = {d["identifier"]: (0, 0) for d in docs}   # 본문·참고문헌 없음
    assert tp.find_candidates(docs, post, thin, cfg()) == []


def test_per_type_cap_enforces_diversity():
    """같은 주 유형이 후보를 독식하면 안 된다 (축구선수 50개 방지)."""
    docs = []
    for h in range(6):
        docs.append(doc("Hub%d" % h, "photovoltaic silicon junction wafer " * 8,
                        ["Infobox samekind"]))
    docs += [doc("N%d" % i, "photovoltaic silicon junction wafer",
                 ["Infobox " + "abcdefgh"[i]]) for i in range(8)]
    post, meta = build(docs, thick_names={"Hub%d" % h for h in range(6)})
    got = tp.find_candidates(docs, post, meta, cfg(max_per_infobox_type=2))
    assert len([c for c in got if c["primary_type"] == "Infobox samekind"]) == 2


def test_targets_are_unique():
    docs = [doc("Hub", "photovoltaic silicon junction wafer " * 8, ["Infobox tech"])]
    docs += [doc("N%d" % i, "photovoltaic silicon junction wafer",
                 ["Infobox " + "abcdefgh"[i]]) for i in range(8)]
    post, meta = build(docs, thick_names={"Hub"})
    got = tp.find_candidates(docs, post, meta, cfg())
    assert len({c["target_idx"] for c in got}) == len(got)
