"""evidence.py — 손계산 고정 케이스 포함 (Acceptance Criteria)."""

from __future__ import annotations

import math

import pytest

from lfm.evidence import (
    DIVERSITY_K,
    EvidenceState,
    coverage,
    grounding,
    source_diversity,
    sufficient,
    support_strength,
)
from lfm.nli import MockNLI
from lfm.types import BranchOutput, Chunk, Perspective, SubAnswer


def chunk(cid: str, doc_id: int, text: str = "the sky is blue today") -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc_id, title="T" + str(doc_id),
                 section_path="Body", text=text)


def branch(sub_answers, chunks, claims=()) -> BranchOutput:
    p = Perspective(role="overview_factual", question="q",
                    sub_questions=tuple("sq%d" % i for i in range(len(sub_answers))))
    return BranchOutput(perspective=p, sub_answers=list(sub_answers),
                        claims=list(claims), chunks=list(chunks))


# --- coverage -----------------------------------------------------------


def test_coverage_counts_sub_questions_with_valid_citations():
    chunks = [chunk("c1", 1), chunk("c2", 2)]
    out = branch(
        [
            SubAnswer(0, "sq0", "yes [c1]", ["c1"]),
            SubAnswer(1, "sq1", "no citation here", []),
            SubAnswer(2, "sq2", "yes [c2]", ["c2"]),
            SubAnswer(3, "sq3", "", []),
        ],
        chunks,
    )
    assert coverage(out) == 0.5  # 4개 중 2개


def test_coverage_ignores_citations_to_unknown_chunks():
    """존재하지 않는 chunk_id 를 인용하면 근거가 아니다 — 환각 인용 방어."""
    out = branch([SubAnswer(0, "sq0", "claim [c99]", ["c99"])], [chunk("c1", 1)])
    assert coverage(out) == 0.0


def test_coverage_of_empty_branch_is_zero():
    assert coverage(branch([], [])) == 0.0


# --- source_diversity (손계산) --------------------------------------------


def test_single_source_diversity_is_exactly_zero():
    """CLAUDE.md 확정 스펙: 단일 소스 -> 0."""
    out = branch(
        [SubAnswer(0, "s", "a [c1] b [c2]", ["c1", "c2"])],
        [chunk("c1", 7), chunk("c2", 7)],  # 같은 문서의 다른 청크
    )
    assert source_diversity(out) == 0.0


def test_no_citations_diversity_is_zero():
    assert source_diversity(branch([SubAnswer(0, "s", "x", [])], [chunk("c1", 1)])) == 0.0


def test_two_equal_sources_matches_hand_calculation():
    """p=(0.5,0.5) -> H=ln2=0.693147 -> H/ln5=0.430677."""
    out = branch(
        [SubAnswer(0, "s", "a [c1] b [c2]", ["c1", "c2"])],
        [chunk("c1", 1), chunk("c2", 2)],
    )
    assert source_diversity(out) == pytest.approx(math.log(2) / math.log(5), abs=1e-9)
    assert source_diversity(out) == pytest.approx(0.4306765581, abs=1e-9)


def test_five_equal_sources_saturates_to_one():
    """K=5 이고 균등 5개 소스면 H=ln5 -> D=1.0 (cap 도달)."""
    cids = ["c%d" % i for i in range(5)]
    out = branch(
        [SubAnswer(0, "s", " ".join("[%s]" % c for c in cids), cids)],
        [chunk(c, i) for i, c in enumerate(cids)],
    )
    assert source_diversity(out) == pytest.approx(1.0, abs=1e-9)


def test_more_than_k_sources_is_capped_at_one():
    cids = ["c%d" % i for i in range(9)]
    out = branch(
        [SubAnswer(0, "s", " ".join("[%s]" % c for c in cids), cids)],
        [chunk(c, i) for i, c in enumerate(cids)],
    )
    assert source_diversity(out) == 1.0
    assert DIVERSITY_K == 5


def test_uneven_two_sources_matches_hand_calculation():
    """인용 3회/1회 -> p=(0.75,0.25), H=0.562335 -> /ln5=0.349434."""
    out = branch(
        [SubAnswer(0, "s", "[c1][c1][c1][c2]", ["c1", "c1", "c1", "c2"])],
        [chunk("c1", 1), chunk("c2", 2)],
    )
    h = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert source_diversity(out) == pytest.approx(h / math.log(5), abs=1e-9)


# --- grounding ----------------------------------------------------------


def test_grounding_uses_cited_chunk_as_premise():
    nli = MockNLI(fixed=0.9)
    out = branch([SubAnswer(0, "s", "The sky is blue. [c1]", ["c1"])], [chunk("c1", 1)])
    assert grounding(out, nli, 0.5) == 1.0


def test_sentence_without_citation_is_not_grounded():
    """인용 없는 문장은 근거가 없다. 후하게 세면 grounding 이 의미를 잃는다."""
    nli = MockNLI(fixed=1.0)
    out = branch([SubAnswer(0, "s", "Unsupported sentence.", [])], [chunk("c1", 1)])
    assert grounding(out, nli, 0.5) == 0.0


def test_grounding_respects_tau_threshold():
    out = branch([SubAnswer(0, "s", "A sentence. [c1]", ["c1"])], [chunk("c1", 1)])
    assert grounding(out, MockNLI(fixed=0.49), 0.5) == 0.0
    assert grounding(out, MockNLI(fixed=0.50), 0.5) == 1.0


def test_grounding_is_a_fraction_of_sentences():
    nli = MockNLI(fixed=1.0)
    out = branch(
        [SubAnswer(0, "s", "First one. [c1] Second one. Third one.", ["c1"])],
        [chunk("c1", 1)],
    )
    # 3문장 전부 같은 인용 목록을 물려받으므로 전부 grounded
    assert grounding(out, nli, 0.5) == 1.0


def test_grounding_of_empty_answer_is_zero():
    assert grounding(branch([], []), MockNLI(fixed=1.0), 0.5) == 0.0


# --- support_strength ---------------------------------------------------


def test_support_strength_averages_max_entailment_per_claim():
    nli = MockNLI(fixed=0.8)
    out = branch(
        [SubAnswer(0, "s", "x [c1]", ["c1"])],
        [chunk("c1", 1)],
        claims=["claim one", "claim two"],
    )
    assert support_strength(out, nli) == pytest.approx(0.8)


def test_support_strength_without_claims_is_zero():
    out = branch([SubAnswer(0, "s", "x [c1]", ["c1"])], [chunk("c1", 1)], claims=[])
    assert support_strength(out, MockNLI(fixed=1.0)) == 0.0


def test_support_strength_without_citations_is_zero():
    """인용이 없으면 전제가 없다 -> 0. 검색 청크로 몰래 대체하지 않는다."""
    out = branch([SubAnswer(0, "s", "x", [])], [chunk("c1", 1)], claims=["a claim"])
    assert support_strength(out, MockNLI(fixed=1.0)) == 0.0


# --- sufficient ---------------------------------------------------------


def test_sufficient_uses_strict_inequality():
    """SPEC §4.3: sufficient(b) := grounding > τ AND coverage > τ. 이상이 아니라 초과."""
    ev = EvidenceState(coverage=0.8, grounding=0.8, source_diversity=0.5, support_strength=0.5)
    assert sufficient(ev, 0.8) is False
    assert sufficient(ev, 0.79) is True


def test_sufficient_needs_both_metrics():
    assert sufficient(EvidenceState(0.9, 0.5, 0.0, 0.0), 0.7) is False
    assert sufficient(EvidenceState(0.5, 0.9, 0.0, 0.0), 0.7) is False
    assert sufficient(EvidenceState(0.9, 0.9, 0.0, 0.0), 0.7) is True


def test_sufficient_accepts_dict_form():
    assert sufficient({"grounding": 0.9, "coverage": 0.9}, 0.8) is True
