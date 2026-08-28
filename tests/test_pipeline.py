"""파이프라인 · 컨트롤러 · 라운드 교란 테스트. 실 API 호출 없음 (mock LLM)."""

from __future__ import annotations

import json

import pytest

from lfm.branch import perturbation_params, run_branch
from lfm.config import Config
from lfm.controller import GlobalRetry
from lfm.controller.base import Action, State
from lfm.llm import LLM, cache_key
from lfm.nli import MockNLI
from lfm.perspectives import ROLES, decompose
from lfm.pipeline import run_query
from lfm.retriever import ListRetriever
from lfm.trace import TraceWriter, new_run_id
from lfm.types import Chunk, Perspective
from lfm.verifier import global_verify
from lfm.evidence import EvidenceState


# --- mock LLM ------------------------------------------------------------


class ScriptedTransport:
    """tag 가 아니라 프롬프트 내용으로 분기한다 — llm.py 는 tag 를 transport 에
    넘기지 않기 때문. 프롬프트를 전부 기록해 교란 검증에 쓴다."""

    def __init__(self, *, answer_citations=("c1", "c2"), n_sub=3) -> None:
        self.prompts: list[str] = []
        self.answer_citations = list(answer_citations)
        self.n_sub = n_sub

    def __call__(self, *, model, system, prompt, max_tokens, reasoning_effort=None):
        self.prompts.append(prompt)
        if "Decompose it into exactly" in prompt:
            body = json.dumps(
                [
                    {
                        "role": r,
                        "question": "Q for " + r,
                        "sub_questions": ["sub %s %d blue sky" % (r, i)
                                          for i in range(self.n_sub)],
                    }
                    for r in ROLES
                ]
            )
        elif "atomic factual claims" in prompt:
            body = json.dumps(["claim one blue", "claim two sky"])
        elif "Rewrite each one" in prompt:
            n = prompt.count("\n0.") + prompt.count("\n1.") + prompt.count("\n2.")
            items = [line.split(". ", 1)[1] for line in prompt.strip().split("\n")
                     if line[:1].isdigit() and ". " in line]
            body = json.dumps(["rephrased " + s for s in items])
        else:  # answer
            cites = " ".join("[%s]" % c for c in self.answer_citations)
            body = json.dumps(
                [{"index": i, "text": "The sky is blue. %s" % cites} for i in range(self.n_sub)]
            )
        return (body, 100, 50, "stop")


@pytest.fixture
def mock_llm(tmp_config):
    return LLM(tmp_config, transport=ScriptedTransport()), tmp_config


@pytest.fixture
def chunks():
    return [
        Chunk("c1", 1, "Sky", "Body", "The sky is blue. Rayleigh scattering."),
        Chunk("c2", 2, "Light", "Body", "The sky is blue because of scattering."),
        Chunk("c3", 3, "Air", "Body", "Air molecules scatter blue light."),
    ]


# --- perspectives --------------------------------------------------------


def test_decompose_returns_the_four_fixed_roles_in_order(mock_llm):
    llm, cfg = mock_llm
    ps = decompose("why is the sky blue", llm, cfg)
    assert [p.role for p in ps] == list(ROLES)
    assert all(3 <= len(p.sub_questions) <= 5 for p in ps)


def test_decompose_coerces_a_bad_model_response(tmp_config):
    """모델이 role 을 빠뜨리거나 개수를 바꿔도 branch 수는 4개로 강제된다."""

    def junk(*, model, system, prompt, max_tokens, reasoning_effort=None):
        return (json.dumps([{"role": "overview_factual", "question": "q",
                             "sub_questions": ["only one"]}]), 1, 1, "stop")

    ps = decompose("q", LLM(tmp_config, transport=junk), tmp_config)
    assert len(ps) == 4
    assert [p.role for p in ps] == list(ROLES)
    assert all(len(p.sub_questions) >= 3 for p in ps), "부족한 sub-question 은 채워진다"


# --- branch --------------------------------------------------------------


def test_run_branch_parses_citations_and_claims(mock_llm, chunks):
    llm, cfg = mock_llm
    p = Perspective("overview_factual", "q", ("blue sky", "scattering light", "air"))
    out = run_branch(p, ListRetriever(chunks), llm, cfg)
    assert len(out.sub_answers) == 3
    assert out.citations
    assert out.claims == ["claim one blue", "claim two sky"]
    assert out.degraded is False
    assert out.retrievals == 3


def test_branch_marks_degraded_when_answer_is_unparseable(tmp_config, chunks):
    def broken(*, model, system, prompt, max_tokens, reasoning_effort=None):
        if "Decompose it into exactly" in prompt:
            return (json.dumps([{"role": r, "question": "q",
                                 "sub_questions": ["a", "b", "c"]} for r in ROLES]), 1, 1, "stop")
        return ("not json at all", 1, 1, "stop")

    p = Perspective("overview_factual", "q", ("a", "b", "c"))
    out = run_branch(p, ListRetriever(chunks), LLM(tmp_config, transport=broken), tmp_config)
    assert out.degraded is True
    assert out.answer == ""


def test_truncated_answer_triggers_one_bumped_retry(tmp_config, chunks):
    seen: list[int] = []

    def truncating(*, model, system, prompt, max_tokens, reasoning_effort=None):
        seen.append(max_tokens)
        return ("", 10, max_tokens, "length")

    p = Perspective("overview_factual", "q", ("a", "b", "c"))
    out = run_branch(p, ListRetriever(chunks), LLM(tmp_config, transport=truncating), tmp_config)
    assert len(seen) == 2, "잘리면 예산을 올려 한 번 더 부른다"
    assert seen[1] > seen[0]
    assert out.truncated is True and out.degraded is True


# --- 라운드 교란 (Acceptance Criteria) -------------------------------------


def test_round_2_prompt_hash_differs_from_round_1(tmp_config, chunks):
    """캐시 회피 검증. 같으면 3라운드가 동일 복제가 되고 낭비율은 항등식이 된다."""
    t = ScriptedTransport()
    llm = LLM(tmp_config, transport=t)
    p = Perspective("overview_factual", "q", ("blue sky", "scattering", "air light"))

    keys = []
    for r in (0, 1, 2):
        t.prompts.clear()
        run_branch(p, ListRetriever(chunks), llm, tmp_config, round_=r)
        answer_prompt = next(x for x in t.prompts if "Context chunks:" in x)
        keys.append(
            cache_key(
                model=llm.model,
                system="",
                prompt=answer_prompt,
                max_tokens=tmp_config["llm.max_tokens"],
                reasoning_effort=llm.reasoning_effort or "",
                cache_nonce="r" + str(r) if r else "",
            )
        )
    assert len(set(keys)) == 3, "라운드마다 캐시 키가 갈려야 한다"


def test_round_perturbation_widens_top_k(tmp_config, chunks):
    r = ListRetriever(chunks)
    llm = LLM(tmp_config, transport=ScriptedTransport())
    p = Perspective("overview_factual", "q", ("blue sky",))
    assert tmp_config.top_k_for_round(0) < tmp_config.top_k_for_round(1)
    params = perturbation_params(tmp_config, 1)
    assert params["top_k"] == tmp_config["retrieval.top_k"] * 2
    assert params["cache_nonce"] == "r1"
    assert params["paraphrase"] is True


def test_paraphrase_keeps_sub_question_count(tmp_config, chunks):
    """개수가 바뀌면 coverage 분모가 흔들린다. 개수 불일치 시 교란을 포기한다."""

    def bad_paraphrase(*, model, system, prompt, max_tokens, reasoning_effort=None):
        if "Rewrite each one" in prompt:
            return (json.dumps(["only one"]), 1, 1, "stop")
        return ScriptedTransport()(model=model, system=system, prompt=prompt,
                                   max_tokens=max_tokens)

    p = Perspective("overview_factual", "q", ("a blue", "b sky", "c air"))
    out = run_branch(p, ListRetriever(chunks), LLM(tmp_config, transport=bad_paraphrase),
                     tmp_config, round_=1)
    assert len(out.sub_answers) == 3


# --- verifier ------------------------------------------------------------


def test_verify_uses_mean_grounding_and_min_coverage(tmp_config):
    """SPEC 변경 #2 의 비대칭. grounding 은 평균, coverage 는 최솟값."""
    ev = {
        "a": EvidenceState(coverage=1.0, grounding=1.0, source_diversity=0, support_strength=0),
        "b": EvidenceState(coverage=0.5, grounding=0.7, source_diversity=0, support_strength=0),
    }
    res = global_verify(ev, tmp_config)
    assert res.agg["mean_grounding"] == pytest.approx(0.85)
    assert res.agg["min_coverage"] == 0.5
    assert res.passed is False, "min_coverage 0.5 < 0.80"


def test_verify_passes_when_both_thresholds_met(tmp_config):
    ev = {r: EvidenceState(0.9, 0.9, 0.5, 0.5) for r in ("a", "b")}
    assert global_verify(ev, tmp_config).passed is True


def test_verify_of_no_branches_fails(tmp_config):
    assert global_verify({}, tmp_config).passed is False


# --- controller ----------------------------------------------------------


def make_state(round_, passed, cfg):
    return State(query_id="q1", round=round_, max_rounds=int(cfg["max_rounds"]),
                 evidence={r: None for r in ROLES}, verify_passed=passed)


def test_global_retry_stops_on_pass(tmp_config):
    a = GlobalRetry(tmp_config).step(make_state(0, True, tmp_config))
    assert a.type == "STOP" and a.params["reason"] == "verify_passed"


def test_global_retry_retries_every_branch(tmp_config):
    """선택적 재실행을 넣으면 그건 GlobalRetry 가 아니다."""
    a = GlobalRetry(tmp_config).step(make_state(0, False, tmp_config))
    assert a.type == "RETRY_ALL"
    assert sorted(a.target) == sorted(ROLES)


def test_global_retry_stops_at_max_rounds(tmp_config):
    last = int(tmp_config["max_rounds"]) - 1
    a = GlobalRetry(tmp_config).step(make_state(last, False, tmp_config))
    assert a.type == "STOP" and a.params["reason"] == "max_rounds"


# --- end-to-end (mock) ---------------------------------------------------


def run(cfg, chunks, nli, transport=None):
    llm = LLM(cfg, transport=transport or ScriptedTransport())
    return run_query(
        query_id="q0001",
        query="why is the sky blue",
        retriever=ListRetriever(chunks),
        llm=llm,
        nli=nli,
        cfg=cfg,
        controller=GlobalRetry(cfg),
        run_id=new_run_id(),
    )


def test_end_to_end_stops_early_when_verify_passes(tmp_config, chunks):
    res = run(tmp_config, chunks, MockNLI(fixed=1.0))
    assert res.passed is True
    assert len(res.rounds) == 1
    assert res.rounds[0].action.type == "STOP"
    assert res.failed_round_0 is False


def test_end_to_end_exhausts_rounds_when_verify_fails(tmp_config, chunks):
    res = run(tmp_config, chunks, MockNLI(fixed=0.0))
    assert res.passed is False
    assert len(res.rounds) == int(tmp_config["max_rounds"])
    assert res.failed_round_0 is True
    assert res.n_retry_rounds == int(tmp_config["max_rounds"]) - 1
    assert [r.action.type for r in res.rounds] == ["RETRY_ALL", "RETRY_ALL", "STOP"]


def test_perspectives_are_decomposed_once_not_per_round(tmp_config, chunks):
    t = ScriptedTransport()
    llm = LLM(tmp_config, transport=t)
    run_query(query_id="q1", query="why is the sky blue", retriever=ListRetriever(chunks),
              llm=llm, nli=MockNLI(fixed=0.0), cfg=tmp_config,
              controller=GlobalRetry(tmp_config), run_id=new_run_id())
    n_decompose = sum(1 for p in t.prompts if "Decompose it into exactly" in p)
    assert n_decompose == 1, "관점이 라운드마다 바뀌면 branch 정체성이 무너진다"


def test_trace_is_written_per_round_and_validates(tmp_path, tmp_config, chunks):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import trace_validate as tv

    path = tmp_path / "trace.jsonl"
    with TraceWriter(path) as w:
        llm = LLM(tmp_config, transport=ScriptedTransport())
        run_query(query_id="q0007", query="why is the sky blue",
                  retriever=ListRetriever(chunks), llm=llm, nli=MockNLI(fixed=0.0),
                  cfg=tmp_config, controller=GlobalRetry(tmp_config),
                  run_id=new_run_id(), trace_writer=w)
    n, errs = tv.validate_file(path)
    assert n == int(tmp_config["max_rounds"])
    assert errs == []


def test_trace_records_round_perturbation_params(tmp_path, tmp_config, chunks):
    from lfm.trace import read_trace

    path = tmp_path / "trace.jsonl"
    with TraceWriter(path) as w:
        llm = LLM(tmp_config, transport=ScriptedTransport())
        run_query(query_id="q0007", query="why is the sky blue",
                  retriever=ListRetriever(chunks), llm=llm, nli=MockNLI(fixed=0.0),
                  cfg=tmp_config, controller=GlobalRetry(tmp_config),
                  run_id=new_run_id(), trace_writer=w)
    recs = list(read_trace(path))
    retry = [r for r in recs if r["action"]["type"] == "RETRY_ALL"]
    assert retry, "실패했으면 RETRY_ALL 이 있어야 한다"
    assert retry[0]["action"]["params"]["top_k"] == tmp_config["retrieval.top_k"] * 2
    assert retry[0]["action"]["params"]["cache_nonce"] == "r1"
