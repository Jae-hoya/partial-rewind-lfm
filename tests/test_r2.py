"""R2 재실험 코드 테스트 (재실험설계_확정안_R2 §3 mock). 실 API·GPU 없음."""

from __future__ import annotations

import json
import re

import pytest

from lfm.branch import run_branch
from lfm.config import Config
from lfm.controller import FreezeRetry, GlobalRetry
from lfm.controller.base import Action, Controller
from lfm.estimator import MemoEstimator, NLIMaxEstimator
from lfm.evidence import coverage, subq_targets
from lfm.llm import LLM, BudgetExceeded, CacheMiss
from lfm.nli import MockNLI
from lfm.perspectives import ROLES, decompose
from lfm.pipeline import run_query
from lfm.retriever import ListRetriever
from lfm.trace import new_run_id
from lfm.types import Chunk, Perspective

N_SUB = 3


class R2Transport:
    """프롬프트에서 role 과 sub-question 번호를 읽어 policy(role, idx, round>0) 대로 답한다."""

    def __init__(self, policy=None):
        self.prompts: list[str] = []
        self.policy = policy or (lambda role, i, retry: {"status": "answered", "text": "The sky is blue [c1]."})

    def __call__(self, *, model, system, prompt, max_tokens, reasoning_effort=None):
        self.prompts.append(prompt)
        if "Decompose it into exactly" in prompt:
            body = json.dumps([{"role": r, "question": "Q for " + r,
                                "sub_questions": ["sub %s %d blue sky" % (r, i) for i in range(N_SUB)]}
                               for r in ROLES])
        elif "atomic factual claims" in prompt:
            body = json.dumps(["the sky is blue"])
        elif "Rewrite each one" in prompt:
            items = [ln.split(". ", 1)[1] for ln in prompt.strip().split("\n") if ln[:1].isdigit() and ". " in ln]
            body = json.dumps(["rephrased " + s for s in items])
        elif "Perspective role:" not in prompt:
            body = "ok"
        else:
            role = re.search(r"Perspective role: (\S+)", prompt).group(1)
            subs = prompt.split("Sub-questions:\n", 1)[1]
            idx = [int(m) for m in re.findall(r"^(\d+)\. ", subs, re.M)]
            retry = "IMPORTANT:" in prompt
            status_mode = "Status rules" in prompt
            rows = []
            for i in idx:
                r = dict(self.policy(role, i, retry))
                if not status_mode:
                    r.pop("status", None)
                rows.append({"index": i, **r})
            body = json.dumps(rows)
        return (body, 100, 50, "stop")

    def answer_prompts(self):
        return [p for p in self.prompts if "Perspective role:" in p]


def cfg_with(tmp_config, **over) -> Config:
    d = tmp_config.as_dict()
    for dotted, v in over.items():
        sec, key = dotted.split(".")
        d.setdefault(sec, {})[key] = v
    return Config(d, source="test_r2")


@pytest.fixture
def chunks():
    return [Chunk("c1", 1, "Sky", "Body", "The sky is blue. Rayleigh scattering."),
            Chunk("c2", 2, "Light", "Body", "The sky is blue because of scattering.")]


def run(cfg, chunks, transport, controller, nli=None):
    llm = LLM(cfg, transport=transport)
    return run_query(query_id="q1", query="why is the sky blue", retriever=ListRetriever(chunks), llm=llm,
                     nli=nli or MockNLI(fixed=1.0), cfg=cfg, controller=controller, run_id=new_run_id())


# --- ①②③ seed 와 nonce ----------------------------------------------------------------------


def test_seed_null_leaves_nonce_unchanged(tmp_config):
    llm = LLM(tmp_config, transport=R2Transport())
    assert llm.effective_nonce("answer", "r1") == "r1"
    assert llm.effective_nonce("answer", "") == ""


def test_seed_prefix_differs_by_seed_and_skips_decompose(tmp_config):
    a = LLM(cfg_with(tmp_config, **{"run.seed": 1, "run.family": "E"}), transport=R2Transport())
    b = LLM(cfg_with(tmp_config, **{"run.seed": 2, "run.family": "E"}), transport=R2Transport())
    assert a.effective_nonce("answer", "r1") == "s1|E|r1"
    assert a.effective_nonce("answer", "") != b.effective_nonce("answer", "")     # 라운드 0 도 seed 별로 갈린다
    assert a.effective_nonce("decompose", "") == "" == b.effective_nonce("decompose", "")


def test_seeds_share_the_decomposition_but_not_the_answers(tmp_config, chunks):
    t = R2Transport()
    for seed in (1, 2):
        cfg = cfg_with(tmp_config, **{"run.seed": seed, "run.family": "E", "run.branch_workers": 1})
        run(cfg, chunks, t, GlobalRetry(cfg))
    n_decomp = sum("Decompose it into exactly" in p for p in t.prompts)
    assert n_decomp == 1                                  # 두 번째 seed 는 캐시
    assert len(t.answer_prompts()) == 2 * 4 * 1            # verify 통과 → 라운드 0 만, seed 마다 4 branch


# --- ④ status 파서 ----------------------------------------------------------------------------


def test_status_parser_drops_uncited_and_flags_failures(tmp_config, chunks):
    def policy(role, i, retry):
        return [{"status": "answered", "text": "Fact one [c1]. Uncited sentence. Bad id [zz9]."},
                {"status": "insufficient", "text": "Leaked text [c1]."},
                {"text": "No status here [c1]."}][i]
    cfg = cfg_with(tmp_config, **{"branch.prompt_mode": "status"})
    llm = LLM(cfg, transport=R2Transport(policy))
    p = Perspective("overview_factual", "Q", tuple("sub %d blue sky" % i for i in range(N_SUB)))
    out = run_branch(p, ListRetriever(chunks), llm, cfg)
    s = out.sub_answers
    assert (s[0].status, s[0].text, s[0].citations) == ("answered", "Fact one [c1].", ["c1"])
    assert (s[1].status, s[1].text) == ("insufficient", "")
    assert (s[2].status, s[2].text) == ("format_fail", "")
    st = out.answer_stats
    assert st["n_dropped_uncited_sentences"] == 2 and st["n_fabricated_id"] == 1
    assert st["n_insufficient"] == 1 and st["n_insufficient_with_text"] == 1 and st["n_format_fail"] == 1
    assert coverage(out) == pytest.approx(1 / 3)


def test_status_prompt_never_changes_legacy_prompt(tmp_config, chunks):
    t = R2Transport()
    p = Perspective("overview_factual", "Q", tuple("sub %d blue sky" % i for i in range(N_SUB)))
    run_branch(p, ListRetriever(chunks), LLM(tmp_config, transport=t), tmp_config)
    assert "Status rules" not in t.prompts[0] and "leave the\n  text empty" in t.prompts[0]


# --- ⑤ FREEZE: 동결된 branch 는 다음 라운드에 호출하지 않는다 ---------------------------------------


def test_freeze_retry_reruns_only_unfrozen_branches(tmp_config, chunks):
    empty = {"impact_application", "controversy_limitation"}
    t = R2Transport(lambda role, i, retry: {"text": "" if role in empty else "The sky is blue [c1]."})
    cfg = cfg_with(tmp_config, **{"run.branch_workers": 1})
    qr = run(cfg, chunks, t, FreezeRetry(cfg, granularity="branch"))
    assert [r.action.type for r in qr.rounds] == ["RERUN", "RERUN", "STOP"]
    assert qr.rounds[0].action.params["frozen"] == ["overview_factual", "technical_mechanism"]
    assert qr.rounds[1].executed == sorted(empty) and qr.rounds[2].executed == sorted(empty)
    assert len(t.answer_prompts()) == 4 + 2 + 2
    assert qr.rounds[1].cost["llm_calls"] < qr.rounds[0].cost["llm_calls"]     # 물려받은 branch 는 비용 0


# --- ⑥ sub-question 재실행: 대상만 다시, 나머지는 바이트 그대로 ----------------------------------------


def test_subq_rerun_only_regenerates_targets_and_carries_the_rest(tmp_config, chunks):
    def policy(role, i, retry):
        if role == "overview_factual" and not retry and i > 0:
            return {"status": "insufficient", "text": ""}
        return {"status": "answered", "text": "The sky is blue [c1]."}
    t = R2Transport(policy)
    cfg = cfg_with(tmp_config, **{"branch.prompt_mode": "status", "run.branch_workers": 1})
    qr = run(cfg, chunks, t, FreezeRetry(cfg, granularity="subq"))
    a0 = qr.rounds[0]
    assert a0.action.type == "RERUN" and a0.action.target == ["overview_factual"]
    assert a0.action.params["subqs"] == {"overview_factual": [1, 2]}
    r1_prompt = t.answer_prompts()[-1]
    listed = [int(m) for m in re.findall(r"^(\d+)\. ", r1_prompt.split("Sub-questions:\n", 1)[1], re.M)]
    assert listed == [1, 2]
    before, after = a0.outputs["overview_factual"].sub_answers[0], qr.rounds[1].outputs["overview_factual"].sub_answers[0]
    assert (after.text, after.citations, after.produced_round) == (before.text, before.citations, 0)
    assert [s.produced_round for s in qr.rounds[1].outputs["overview_factual"].sub_answers] == [0, 1, 1]
    assert coverage(qr.rounds[1].outputs["overview_factual"]) == 1.0
    assert qr.rounds[1].action.type == "STOP" and qr.passed


def test_subq_targets_are_missing_or_unsupported(tmp_config, chunks):
    t = R2Transport(lambda role, i, retry: [{"text": "The sky is blue [c1]."}, {"text": ""},
                                            {"text": "Zebras fly [c2]."}][i])
    p = Perspective("overview_factual", "Q", tuple("sub %d blue sky" % i for i in range(N_SUB)))
    out = run_branch(p, ListRetriever(chunks), LLM(tmp_config, transport=t), tmp_config)
    assert subq_targets(out, MockNLI(), 0.5) == [1, 2]       # 1 은 미커버, 2 는 어휘 겹침 0 → 비지지


def test_memo_estimator_scores_each_pair_once():
    nli = MockNLI()
    m = MemoEstimator(NLIMaxEstimator(nli, 0.5))
    items = [("the sky is blue", ["The sky is blue."])] * 3
    assert m.grounded_many(items) == [True] * 3 and nli.n_pairs == 1 and m.hits == 2


# --- ⑦⑧ VERIFY/PRUNE 차단, 라운드 상한 -------------------------------------------------------------


class _VerifyController(Controller):
    name = "verify_bot"

    def step(self, state):
        return Action("VERIFY", target=sorted(state.evidence))


def test_verify_and_prune_are_refused(tmp_config, chunks):
    t = R2Transport(lambda role, i, retry: {"text": ""})
    with pytest.raises(AssertionError):
        run(tmp_config, chunks, t, _VerifyController())


@pytest.mark.parametrize("make", [lambda c: GlobalRetry(c), lambda c: FreezeRetry(c, "branch"),
                                  lambda c: FreezeRetry(c, "subq")])
def test_rounds_never_exceed_max_rounds(tmp_config, chunks, make):
    t = R2Transport(lambda role, i, retry: {"status": "insufficient", "text": ""})
    cfg = cfg_with(tmp_config, **{"branch.prompt_mode": "status", "run.branch_workers": 1})
    qr = run(cfg, chunks, t, make(cfg))
    assert len(qr.rounds) <= int(cfg["max_rounds"]) and qr.rounds[-1].action.type == "STOP"


# --- ⑨ cache_only, 예산 ---------------------------------------------------------------------------


def test_cache_only_raises_instead_of_calling_the_api(tmp_config):
    t = R2Transport()
    llm = LLM(cfg_with(tmp_config, **{"llm.cache_policy": "cache_only"}), transport=t)
    with pytest.raises(CacheMiss):
        llm.call("hello", tag="answer")
    assert t.prompts == []


def test_budget_stops_before_the_next_billed_call(tmp_config):
    llm = LLM(cfg_with(tmp_config, **{"budget.max_billed_in": 150}), transport=R2Transport())
    llm.call("one", tag="answer")
    llm.call("two", tag="answer")          # 누적 100 → 아직 상한 밑이라 부른다 (200)
    with pytest.raises(BudgetExceeded):
        llm.call("three", tag="answer")
    llm.call("one", tag="answer")          # 캐시 적중은 청구가 아니므로 막지 않는다
