"""질의 1건을 컨트롤러 아래에서 라운드 루프로 돌린다. 전 라운드 trace 기록.

M1 의 컨트롤러는 GlobalRetry 하나뿐이지만 루프는 컨트롤러에 의존하지 않는다.
M2 의 LFM controller 가 같은 루프에 들어와야 비교가 공정하다 (CLAUDE.md).

관점 분해는 **라운드 0에서 한 번만** 한다. 라운드 교란은 branch 수준에서 일어난다
(SPEC §3.6): 검색 폭, sub-question paraphrase, retry directive, cache_nonce.
관점 자체를 라운드마다 새로 뽑으면 branch 정체성이 바뀌어 "같은 branch 를 다시
돌렸다"는 낭비율의 전제가 무너진다.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .branch import perturbation_params, run_branch
from .controller.base import Action, Controller, State
from .estimator import MemoEstimator, NLIMaxEstimator
from .evidence import EvidenceState, evaluate, subq_targets
from .graph import Graph
from .perspectives import decompose
from .trace import budget_left, make_action, make_cost, make_record, make_state
from .types import BranchOutput, Perspective
from .verifier import VerifyResult, global_verify

__all__ = ["RoundResult", "QueryResult", "run_query"]


@dataclass
class RoundResult:
    round: int
    evidence: dict[str, EvidenceState]
    verify: VerifyResult
    action: Action
    outputs: dict[str, BranchOutput] = field(default_factory=dict)
    cost: dict[str, int] = field(default_factory=dict)
    # 이 라운드에 실제로 실행한 branch. 나머지 outputs 는 이전 라운드에서 물려받은 것이다 (비용 0)
    executed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "evidence": {k: v.to_dict() for k, v in self.evidence.items()},
            "verify": {"passed": self.verify.passed, "agg": self.verify.agg},
            "action": self.action.to_dict(),
            "cost": dict(self.cost),
        }


@dataclass
class QueryResult:
    query_id: str
    query: str
    perspectives: list[Perspective]
    rounds: list[RoundResult]
    run_id: str

    @property
    def passed(self) -> bool:
        return bool(self.rounds) and self.rounds[-1].verify.passed

    @property
    def failed_round_0(self) -> bool:
        """SPEC §4.2 — 실패 케이스 = 1라운드 verify 불통과 쿼리."""
        return bool(self.rounds) and not self.rounds[0].verify.passed

    @property
    def n_retry_rounds(self) -> int:
        return sum(1 for r in self.rounds if r.action.type == "RETRY_ALL")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "run_id": self.run_id,
            "perspectives": [p.to_dict() for p in self.perspectives],
            "passed": self.passed,
            "failed_round_0": self.failed_round_0,
            "n_rounds": len(self.rounds),
            "rounds": [r.to_dict() for r in self.rounds],
        }


def _cost_of(outputs: dict[str, BranchOutput]) -> dict[str, int]:
    return make_cost(
        tokens_in=sum(o.tokens_in for o in outputs.values()),
        tokens_out=sum(o.tokens_out for o in outputs.values()),
        llm_calls=sum(o.llm_calls for o in outputs.values()),
        retrievals=sum(o.retrievals for o in outputs.values()),
        latency_ms=sum(o.latency_ms for o in outputs.values()),
    )


def run_query(
    *,
    query_id: str,
    query: str,
    retriever,
    llm,
    nli,
    cfg,
    controller: Controller,
    run_id: str,
    trace_writer=None,
    search_anchor: str = "",
    est=None,
) -> QueryResult:
    """`est` 를 주면 grounding 을 그 연산자로 잰다 (SPEC_M1prime §1).

    주지 않으면 종전대로 `nli` 로 잰다 — Test A 원본 경로가 그대로 재현된다.

    R2: 컨트롤러가 RERUN 을 내면 다음 라운드에 그 target branch 만 실행하고, 나머지 branch 는
    직전 산출과 evidence 를 그대로 물려받는다(호출 0, 비용 0). params["subqs"] 가 있으면 그 branch 는
    대상 sub-question 만 다시 돈다. VERIFY / PRUNE 은 평가 시스템에서 제외했으므로 나오면 멈춘다.
    GlobalRetry(RETRY_ALL)의 경로는 예전과 같다.
    """
    max_rounds = int(cfg["max_rounds"])

    perspectives = decompose(query, llm, cfg)
    branch_ids = [p.role for p in perspectives]
    graph = Graph(branch_ids)
    rounds: list[RoundResult] = []

    workers = int(cfg.get("run.branch_workers", 1))
    needs_view = bool(getattr(controller, "needs_subq_view", False))
    if needs_view:
        # sub-question 대상 계산과 evaluate 가 같은 문장을 채점한다 — 한 번만 계산한다
        est = MemoEstimator(est if est is not None else NLIMaxEstimator(nli, float(cfg.get("nli.tau_ground", 0.5))))

    prev_action: Action | None = None
    prev_outputs: dict[str, BranchOutput] = {}
    prev_evidence: dict[str, EvidenceState] = {}
    frozen: list[str] = []

    for r in range(max_rounds):
        if prev_action is None or prev_action.type == "RETRY_ALL":
            run_set, subqs = list(perspectives), {}
        else:  # RERUN
            run_set = [p for p in perspectives if p.role in set(prev_action.target)]
            subqs = prev_action.params.get("subqs") or {}

        def go(p):
            sel = subqs.get(p.role)
            return run_branch(p, retriever, llm, cfg, round_=r, search_anchor=search_anchor,
                              subq_filter=set(sel) if sel is not None else None,
                              carry=prev_outputs[p.role] if sel is not None else None)

        # 한 라운드의 branch 들은 서로 독립이다. 병렬로 돌려도 결과는 같다 —
        # 캐시 키가 결정론이고 호출 회계는 스레드-로컬 스코프로 센다.
        if workers > 1 and len(run_set) > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {p.role: ex.submit(go, p) for p in run_set}
                ran = {role: f.result() for role, f in futures.items()}
        else:
            ran = {p.role: go(p) for p in run_set}
        outputs = {p.role: ran.get(p.role, prev_outputs.get(p.role)) for p in perspectives}
        evidence = {bid: (evaluate(o, nli, cfg, est) if bid in ran else prev_evidence[bid])
                    for bid, o in outputs.items()}
        verify = global_verify(evidence, cfg)

        state = State(
            query_id=query_id,
            round=r,
            max_rounds=max_rounds,
            evidence=evidence,
            verify_passed=verify.passed,
            verify_agg=verify.agg,
            graph=graph.snapshot(),
            frozen=list(frozen),
        )
        if needs_view:
            tau_g = float(cfg.get("nli.tau_ground", 0.5))
            state.subq_view = {b: subq_targets(o, est, tau_g) for b, o in outputs.items() if b not in frozen}
        action = controller.step(state)
        if action.type in ("VERIFY", "PRUNE"):
            raise AssertionError("VERIFY/PRUNE 은 R2 평가 시스템에서 제외했다: " + action.type)
        frozen = list(action.params.get("frozen", frozen))

        params = dict(action.params)
        if action.type in ("RETRY_ALL", "RERUN"):
            params.update(perturbation_params(cfg, r + 1))
        params["search_anchor"] = search_anchor
        params["verify"] = {
            "mean_grounding": verify.agg.get("mean_grounding"),
            "min_coverage": verify.agg.get("min_coverage"),
            "passed": verify.passed,
        }
        params["degraded_branches"] = sorted(b for b, o in outputs.items() if o.degraded)

        cost = _cost_of(ran)                      # 이번 라운드에 실행한 것만. 물려받은 branch 는 0
        rounds.append(
            RoundResult(round=r, evidence=evidence, verify=verify, action=action,
                        outputs=outputs, cost=cost, executed=sorted(ran))
        )

        if trace_writer is not None:
            trace_writer.write(
                make_record(
                    run_id=run_id,
                    query_id=query_id,
                    round_=r,
                    state=make_state(
                        evidence={b: e.to_dict() for b, e in evidence.items()},
                        budget=budget_left(r, max_rounds),
                        vertices=graph.V,
                        status=graph.S,
                        frozen=frozen,
                    ),
                    action=make_action(action.type, target=list(action.target), params=params),
                    cost=cost,
                )
            )

        if action.type not in ("RETRY_ALL", "RERUN"):
            break
        prev_action, prev_outputs, prev_evidence = action, outputs, evidence

    return QueryResult(
        query_id=query_id, query=query, perspectives=perspectives, rounds=rounds, run_id=run_id
    )
