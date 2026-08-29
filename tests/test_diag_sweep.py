"""diag_sweep.py — 조건·판정 기준이 코드에 고정돼 있는지."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import diag_sweep as ds  # noqa: E402


def test_conditions_are_exactly_the_four_in_the_spec():
    """D4 규칙 2: 조건은 실행 전에 확정하고 중간에 추가하지 않는다."""
    assert set(ds.CONDITIONS) == {"C1_baseline", "C2_topk15", "C3_partial_heldout", "C4_both"}
    assert ds.CONDITIONS["C2_topk15"]["top_k"] == 15
    assert ds.CONDITIONS["C3_partial_heldout"]["index"] == "partial"
    assert ds.CONDITIONS["C4_both"] == {"top_k": 15, "index": "partial"}


def test_gates_match_spec_v11():
    assert ds.GATES["intersection_rate"][1] == 0.10
    assert ds.GATES["cov1_grounding_median"][1] == 0.50
    assert ds.GATES["cov1_rate"][1] == 0.30
    assert ds.GATES["empty_rate"][1] == 0.15
    assert ds.GATES["empty_rate"][2] == "le"


def test_length_biased_metric_is_record_only():
    """v1.1 변경 #1: grounding 단독은 판정에 쓰지 않는다."""
    assert "grounding_gt08_rate" not in ds.GATES
    assert "grounding_gt08_rate" in ds.RECORD_ONLY
    assert "grounded_count_zero_rate" in ds.RECORD_ONLY


def test_module_does_not_import_waste_ratio_machinery():
    """D4 규칙 1: 낭비율을 계산하지 않는다.

    주석·docstring 의 단어가 아니라 실제 import 와 호출을 본다.
    """
    import ast

    tree = ast.parse(Path(ds.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(a.name for a in node.names)
    assert not any("test_a" in m for m in imported), imported
    assert "sufficient" not in imported

    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "sufficient" not in called
    assert not hasattr(ds, "waste_by_query") and not hasattr(ds, "bootstrap_ci")


def test_pilot_topics_are_excluded_and_selection_is_seeded():
    topics = [{"id": "t%03d" % i, "topic": "q", "terms": ["a"]} for i in range(50)]
    a = ds.pick_topics(topics, 15, 42)
    b = ds.pick_topics(topics, 15, 42)
    c = ds.pick_topics(topics, 15, 43)
    assert [t["id"] for t in a] == [t["id"] for t in b], "같은 시드면 같은 토픽"
    assert [t["id"] for t in a] != [t["id"] for t in c]
    assert len(a) == 15
    assert not ({t["id"] for t in a} & set(ds.PILOT_IDS)), "파일럿 토픽은 빠져야 한다"


def row(cov, grd, n_sent=5, cited=4, entail=None):
    return {"coverage": cov, "grounding_ratio": grd, "n_sentences": n_sent,
            "n_cited_sub_questions": cited,
            "grounded_count": (entail if entail is not None else grd * n_sent) / max(1, cited)}


def test_metrics_hand_calculation():
    rows = [row(1.0, 1.0), row(1.0, 0.2), row(0.5, 0.9), row(0.0, 0.0, n_sent=0)]
    m = ds.metrics(rows)
    assert m["intersection_rate"] == 0.25          # 1/4
    assert m["cov1_rate"] == 0.5                   # 2/4
    assert m["cov1_grounding_median"] == 0.6       # (1.0+0.2)/2
    assert m["empty_rate"] == 0.25                 # 1/4
    assert m["grounding_gt08_rate"] == 0.5         # 2/4


def test_judge_requires_all_four_gates():
    good = {"intersection_rate": 0.2, "cov1_grounding_median": 0.6,
            "cov1_rate": 0.4, "empty_rate": 0.1}
    ok, _ = ds.judge(good)
    assert ok is True
    for k, bad in [("intersection_rate", 0.05), ("cov1_grounding_median", 0.4),
                   ("cov1_rate", 0.2), ("empty_rate", 0.5)]:
        ok, _ = ds.judge({**good, k: bad})
        assert ok is False, k


def test_current_baseline_would_fail_every_gate_but_one():
    """본실험 라운드 0 실측값. 스윕이 의미 있으려면 기준선은 떨어져야 한다."""
    now = {"intersection_rate": 0.005, "cov1_grounding_median": 0.231,
           "cov1_rate": 0.255, "empty_rate": 0.335}
    ok, _ = ds.judge(now)
    assert ok is False
