"""동결 trace 스키마 검증기 (CLAUDE.md).

사용: python experiments/trace_validate.py results/test_a/<ts>/trace.jsonl [...]

검사 항목
  - 최상위 필수 키 8개 존재 및 타입
  - state 하위 키 6개 (E, R, budget_left, frozen, verify_count, graph) — M1 에서 비어도 존재
  - graph 하위 (V, E, S), S 값이 허용 라벨
  - reward 는 항상 null
  - action.params 존재 (라운드 교란 파라미터 기록 위치)
필드 추가는 허용이므로 여분 키는 오류가 아니다.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lfm.trace import EVIDENCE_KEYS, STATE_LABELS  # noqa: E402

TOP_LEVEL: dict[str, type | tuple[type, ...]] = {
    "run_id": str,
    "query_id": str,
    "round": int,
    "state": dict,
    "action": dict,
    "cost": dict,
    "next_state": dict,
}
STATE_KEYS = ("E", "R", "budget_left", "frozen", "verify_count", "graph")
ACTION_KEYS = ("type", "target", "params")
COST_KEYS = ("tokens_in", "tokens_out", "llm_calls", "retrievals", "latency_ms")


def validate_record(rec: dict[str, Any], where: str) -> list[str]:
    errs: list[str] = []

    def bad(msg: str) -> None:
        errs.append(where + ": " + msg)

    for key, typ in TOP_LEVEL.items():
        if key not in rec:
            bad("최상위 키 누락 " + key)
        elif not isinstance(rec[key], typ) or isinstance(rec[key], bool):
            bad("최상위 키 타입 오류 " + key + " -> " + type(rec[key]).__name__)
    if "reward" not in rec:
        bad("reward 키 누락 (동결 스키마상 항상 존재해야 한다)")
    elif rec["reward"] is not None:
        bad("reward 는 항상 null 이어야 한다 (backfill 은 M1 범위 밖)")

    state = rec.get("state")
    if isinstance(state, dict):
        for key in STATE_KEYS:
            if key not in state:
                bad("state 키 누락 " + key)
        E = state.get("E")
        if isinstance(E, dict):
            for node, metrics in E.items():
                if not isinstance(metrics, dict):
                    bad("state.E[" + node + "] 가 매핑이 아니다")
                    continue
                for m in EVIDENCE_KEYS:
                    if m not in metrics:
                        bad("state.E[" + node + "] 에 " + m + " 없음")
        elif E is not None:
            bad("state.E 가 매핑이 아니다")
        if not isinstance(state.get("R"), list):
            bad("state.R 은 리스트여야 한다 (M1 은 빈 리스트)")
        if not isinstance(state.get("verify_count"), dict):
            bad("state.verify_count 는 매핑이어야 한다 (M1 은 빈 매핑)")
        if not isinstance(state.get("frozen"), list):
            bad("state.frozen 은 리스트여야 한다")
        graph = state.get("graph")
        if isinstance(graph, dict):
            for key in ("V", "E", "S"):
                if key not in graph:
                    bad("state.graph 키 누락 " + key)
            S = graph.get("S")
            if isinstance(S, dict):
                for node, label in S.items():
                    if label not in STATE_LABELS:
                        bad("허용되지 않은 상태 라벨 " + repr(label) + " (" + node + ")")
                missing = [v for v in graph.get("V", []) if v not in S]
                if missing:
                    bad("graph.V 에 있으나 graph.S 에 없는 노드: " + repr(missing))
            else:
                bad("state.graph.S 가 매핑이 아니다")
        else:
            bad("state.graph 가 매핑이 아니다")

    action = rec.get("action")
    if isinstance(action, dict):
        for key in ACTION_KEYS:
            if key not in action:
                bad("action 키 누락 " + key)

    cost = rec.get("cost")
    if isinstance(cost, dict):
        for key in COST_KEYS:
            if key not in cost:
                bad("cost 키 누락 " + key)
            elif not isinstance(cost[key], int) or isinstance(cost[key], bool):
                bad("cost." + key + " 는 정수여야 한다")

    return errs


def validate_file(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    n = 0
    with io.open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            where = path.name + ":" + str(lineno)
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(where + ": JSON 파싱 실패 " + str(exc))
                continue
            if not isinstance(rec, dict):
                errors.append(where + ": 레코드가 객체가 아니다")
                continue
            n += 1
            errors.extend(validate_record(rec, where))
    return n, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="동결 trace 스키마 검증")
    ap.add_argument("paths", nargs="+", help="trace JSONL 경로")
    args = ap.parse_args(argv)

    total, all_errors = 0, []
    for raw in args.paths:
        p = Path(raw)
        if not p.exists():
            all_errors.append(str(p) + ": 파일 없음")
            continue
        n, errs = validate_file(p)
        total += n
        all_errors.extend(errs)
        print("checked " + str(p) + ": " + str(n) + " records, " + str(len(errs)) + " errors")

    if all_errors:
        print("\nFAIL — " + str(len(all_errors)) + "건")
        for e in all_errors[:50]:
            print("  " + e)
        if len(all_errors) > 50:
            print("  ... " + str(len(all_errors) - 50) + "건 더")
        return 1
    print("OK — " + str(total) + " records, 동결 스키마 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
