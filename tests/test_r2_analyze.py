"""r2_analyze / r2_sheets — 합성 run 디렉터리(2 arm × 2 seed × 3 토픽)로 검사한다. LLM·GPU·인덱스 없음."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import r2_analyze as ra  # noqa: E402
import r2_sheets as rs  # noqa: E402

from lfm.branch import format_context  # noqa: E402
from lfm.types import Chunk  # noqa: E402

TOPICS = ("t000", "t007", "t031")
BR = ("overview_factual", "technical_mechanism")
SEEDS = (1, 2)
R0_STATUS = {"overview_factual": ["answered", "insufficient", "format_fail"],
             "technical_mechanism": ["answered", "answered", "insufficient"]}
R1_STATUS = {"overview_factual": ["answered", "insufficient", "answered"],
             "technical_mechanism": ["answered", "answered", "insufficient"]}


def sa(i, status, rnd, b, q, seed):
    text = ("Fact %s %s %d s%d supported [c%d]." % (q, b, i, seed, i)) if status == "answered" else ""
    return {"index": i, "sub_question": "What about %s item %d in %s?" % (b, i, q), "text": text,
            "citations": ["c%d" % i] if text else [], "status": status, "produced_round": rnd}


def branch(b, rnd, sas, executed=True, tin=100, tout=50, subq=3, sha=""):
    return {"round": rnd, "branch": b, "executed": executed,
            "evidence": {"coverage": sum(1 for s in sas if s["citations"]) / len(sas), "grounding": 1.0,
                         "source_diversity": 0.0, "support_strength": 0.0},
            "tokens_in": tin, "tokens_out": tout, "llm_calls": 2,
            "billing": {"api_calls": 2 if executed else 0, "billed_in": tin, "billed_out": tout,
                        "by_tag": {"answer": {"calls": 1, "api_calls": 1, "tokens_in": tin, "tokens_out": tout},
                                   "claims": {"calls": 1, "api_calls": 1, "tokens_in": 0, "tokens_out": 0}}},
            "answer_stats": {"n_format_fail": sum(1 for s in sas if s["status"] == "format_fail"),
                             "n_dropped_uncited_sentences": 0},
            "subq_executed": subq, "context_sha256": sha, "degraded": False, "truncated": False,
            "sub_answers": sas, "n_chunks": 3}


def rnd(r, branches, action, frozen=()):
    return {"round": r, "action": {"type": action, "target": [], "params": {"round": r, "frozen": list(frozen)}},
            "verify": {"passed": False, "agg": {"pass_grounding": True, "pass_coverage": False}},
            "executed": [b["branch"] for b in branches if b["executed"]], "cost": {}, "branches": branches}


def round0(q, seed, status_mode=True):
    out = []
    for b in BR:
        sas = [sa(i, s, 0, b, q, seed) for i, s in enumerate(R0_STATUS[b])]
        if not status_mode:
            for x in sas:
                x["status"] = None
        out.append(branch(b, 0, sas))
    return out


def topic_gr(q, seed, status_mode=True):
    r1 = []
    for b in BR:
        sas = [sa(i, s, 1, b, q, seed) for i, s in enumerate(R1_STATUS[b])]
        if not status_mode:
            for x in sas:
                x["status"] = None
        r1.append(branch(b, 1, sas, tin=120, tout=60))
    return {"query_id": q, "run_tag": "syn", "passed": False, "n_rounds": 2,
            "rounds": [rnd(0, round0(q, seed, status_mode), "RETRY_ALL"), rnd(1, r1, "STOP")]}


def topic_sq(q, seed):
    r0 = round0(q, seed)
    a_prev = r0[0]["sub_answers"]
    a = [copy.deepcopy(a_prev[0]), sa(1, "answered", 1, BR[0], q, seed), sa(2, "insufficient", 1, BR[0], q, seed)]
    # 동결된 branch: 실행하지 않았다. 토큰 칸에 값이 있어도 세면 안 된다 (executed 플래그로 거른다)
    frozen_b = branch(BR[1], 1, copy.deepcopy(r0[1]["sub_answers"]), executed=False, tin=999, tout=999)
    return {"query_id": q, "run_tag": "syn", "passed": False, "n_rounds": 2,
            "rounds": [rnd(0, r0, "RERUN", frozen=[BR[1]]),
                       rnd(1, [branch(BR[0], 1, a, tin=40, tout=20, subq=2), frozen_b], "STOP", frozen=[BR[1]])]}


def write_topic(root, arm, seed, rec):
    d = root / arm / ("s%d" % seed) / "topics"
    d.mkdir(parents=True, exist_ok=True)
    io.open(d / (rec["query_id"] + ".json"), "w", encoding="utf-8").write(json.dumps(rec))


@pytest.fixture
def run_dir(tmp_path):
    root = tmp_path / "syn"
    ledger = []
    for seed in SEEDS:
        for arm, make in (("E-GR", topic_gr), ("E-SQ", topic_sq)):
            bin_ = bout = calls = 0
            for q in TOPICS:
                rec = make(q, seed)
                write_topic(root, arm, seed, rec)
                for rr in rec["rounds"]:
                    for b in rr["branches"]:
                        if b["executed"]:
                            ledger.append({"tag": "answer", "nonce": "s%d|E|r%d" % (seed, rr["round"]),
                                           "tokens_in": b["tokens_in"], "tokens_out": b["tokens_out"]})
                            bin_ += b["tokens_in"]; bout += b["tokens_out"]; calls += 1
            io.open(root / arm / ("s%d" % seed) / "usage.json", "w", encoding="utf-8").write(json.dumps(
                [{"finished_at": "x", "usage": {"calls": calls, "cache_hits": 0, "api_calls": calls,
                                                "billed_tokens_in": bin_, "billed_tokens_out": bout,
                                                "by_tag": {"decompose": {"calls": 1, "cache_hits": 1}}}}]))
    with io.open(root / "billing_ledger.jsonl", "w", encoding="utf-8") as fh:
        for r in ledger:
            fh.write(json.dumps(r) + "\n")
    return root


# --------------------------------------------------------------------------- 분석
def test_accounting_excludes_non_executed_branches(run_dir):
    runs = ra.with_r1(ra.load_run(run_dir))
    summ = ra.armseed_summary(runs[("E-SQ", 1)], ra.Support())
    for q in TOPICS:
        m = summ["per_query"][q]
        assert m["accounting_tokens"] == 2 * 150 + 60          # 999 은 세지 않는다
        assert m["billed_tokens"] == 360
        assert m["branch_rounds"] == 3
        assert m["subq_executed"] == 3 + 3 + 2
    gr = ra.armseed_summary(runs[("E-GR", 1)], ra.Support())
    assert gr["per_query"]["t007"]["accounting_tokens"] == 2 * 150 + 2 * 180


def test_e_r1_is_e_gr_round_0(run_dir):
    runs = ra.with_r1(ra.load_run(run_dir))
    for s in SEEDS:
        r1, gr = runs[("E-R1", s)], runs[("E-GR", s)]
        for q in TOPICS:
            assert len(r1[q]["rounds"]) == 1
            assert r1[q]["rounds"][0] == gr[q]["rounds"][0]
        summ = ra.armseed_summary(r1, ra.Support())
        assert all(m["accounting_tokens"] == 300 for m in summ["per_query"].values())
    # derive 가 원본을 건드리지 않는다
    assert len(runs[("E-GR", 1)]["t007"]["rounds"]) == 2


def test_bootstrap_is_deterministic_and_point_estimate(run_dir):
    runs = ra.with_r1(ra.load_run(run_dir))
    summ = {k: ra.armseed_summary(v, ra.Support()) for k, v in runs.items()}
    a = ra.savings_compare(summ, "E-SQ", "E-GR", "accounting_tokens")
    b = ra.savings_compare(summ, "E-SQ", "E-GR", "accounting_tokens")
    assert a == b
    assert a["point"] == pytest.approx(1 - 360 / 660)
    assert set(a["sign_table"]) == {"1", "2"} and a["claim"] in (True, False)
    A = np.arange(6, dtype=float).reshape(3, 2) + 1
    B = A * 2
    assert np.array_equal(ra.boot_savings(A, B, 500), ra.boot_savings(A, B, 500))
    assert np.allclose(ra.boot_savings(A, B, 500), 0.5)
    ni = ra.noninf_compare(summ, "E-R1", "E-GR", "coverage", 0.05, (0.03, 0.10))
    # E-R1 coverage = 라운드 0 (3/6), E-GR 최종 = 4/6
    assert ni["diff"] == pytest.approx(3 / 6 - 4 / 6)
    assert ni["noninferior"] is False


def test_transition_table_sums(run_dir):
    runs = ra.with_r1(ra.load_run(run_dir))
    t = ra.transitions(runs[("E-GR", 1)], ra.Support())
    n = sum(v for row in t["counts"].values() for v in row.values())
    tok = sum(v for row in t["tokens"].values() for v in row.values())
    assert n == t["n_reexecuted_units"] == len(TOPICS) * len(BR) * 3
    assert tok == pytest.approx(len(TOPICS) * 2 * 180)        # 라운드 ≥ 1 회계 토큰 전부
    assert sum(t["round0"]["counts"].values()) == len(TOPICS) * 6
    assert t["counts"]["format_fail"] == {"cited": len(TOPICS)}   # 지지 판정이 없으면 cited 로 합친다
    assert t["counts"]["insufficient"] == {"insufficient": len(TOPICS) * 2}
    assert t["label"].startswith("accounting decomposition")
    sq = ra.transitions(runs[("E-SQ", 1)], ra.Support())
    assert sq["n_reexecuted_units"] == len(TOPICS) * 2
    assert sum(v for row in sq["tokens"].values() for v in row.values()) == pytest.approx(len(TOPICS) * 60)
    assert sq["subq_executed_mismatch"] == 0
    # insufficient(라운드 0 의 1번) → answered(라운드 1)
    assert sq["insufficient_to_answered"]["n_to_answered"] == len(TOPICS)


class KeywordScorer:
    def grounded_many(self, items):
        return ["supported" in s for s, _ in items]


def test_support_splits_cited_into_sufficient_and_unsupported(run_dir):
    runs = ra.with_r1(ra.load_run(run_dir))
    chunks = {"c%d" % i: "premise %d" % i for i in range(3)}
    sup = ra.Support(KeywordScorer(), chunks)
    sup.prime(x for recs in runs.values() for rec in recs.values() for rr in rec["rounds"]
              for b in rr["branches"] for x in b["sub_answers"])
    t = ra.transitions(runs[("E-GR", 1)], sup)
    assert "cited" not in t["states"]
    assert t["counts"]["format_fail"] == {"sufficient": len(TOPICS)}
    assert t["format_recovery_tokens"] == pytest.approx(len(TOPICS) * 60)
    cons = ra.rescore_consistency(runs, sup)
    assert cons["pass"] and cons["n_mismatch"] == 0


def test_analyze_writes_outputs_and_pilot_gates(run_dir):
    res = ra.analyze(run_dir, pilot=True, n_boot=200)
    j, m = ra.write_outputs(res, run_dir)
    assert j.exists() and m.exists()
    data = json.load(io.open(j, encoding="utf-8"))
    assert data["sentence_support"] == "unavailable"
    assert data["sensitivity_30"]["savings"]["E-SQ_vs_E-GR"]["accounting_tokens"]["n_topics"] == 2
    assert data["all_topics"]["savings"]["E-SQ_vs_E-GR"]["accounting_tokens"]["n_topics"] == 3
    assert data["all_topics"]["savings"]["L-BR_vs_L-GR"]["accounting_tokens"]["available"] is False
    G = data["gates"]
    assert G["G1"]["7_ledger_equals_usage"]["pass"] is True
    assert G["G1"]["2_decompose_api_calls"]["pass"] is True
    assert G["G1"]["9_inherited_bytes_identical"]["pass"] is True
    assert G["G1"]["round0_identical_within_family_seed"]["pass"] is True
    assert G["G1"]["5_fabricated_ids_left"]["n"] == 0
    assert G["G3"]["by_seed"]["E-SQ_s1"]["rate"] == 1.0
    assert "회계 분해" in m.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- 시트
class StubRetriever:
    def __init__(self):
        self.chunks = [Chunk("c%d" % i, i, "Title%d" % i, "Sec", ("body %d " % i) * 60) for i in range(3)]

    def search(self, query, top_k):
        return self.chunks[:top_k]


def stub_builder(verify=True):
    return rs.ContextBuilder(StubRetriever(), {}, lambda r: 3, 12000, verify=verify)


ARM_WORDS = ("E-GR", "E-SQ", "E-R1", "E-BR", "L-GR", "L-BR")
STATUS_WORDS = ("insufficient", "answered", "format_fail", "dropped", "legacy")


def _assert_blind(html: str):
    for w in ARM_WORDS + STATUS_WORDS:
        assert w not in html, w


def add_legacy_arm(run_dir):
    for seed in SEEDS:
        for q in TOPICS:
            write_topic(run_dir, "L-GR", seed, topic_gr(q, seed, status_mode=False))


def test_sheets_are_blinded_and_keys_have_strata(run_dir, tmp_path):
    add_legacy_arm(run_dir)
    out = tmp_path / "sheets"
    ph = rs.build_ph(run_dir, out, stub_builder(), n_eb=4, n_ea=4, n_la=3)
    h4 = rs.build_h4(run_dir, out, n_slots=8, seeds=SEEDS)
    h5 = rs.build_h5(run_dir, out, stub_builder(), seeds=SEEDS, n_insufficient=4, n_new_answered=2)
    for r in (ph, h4, h5):
        html = r["html"].read_text(encoding="utf-8")
        _assert_blind(html)
        keys = list(csv.DictReader(io.open(r["key"], encoding="utf-8")))
        assert keys
        assert all(k["stratum"] and k["N_h"] and k["n_h"] for k in keys)
        assert all(k["arm"] and k["arm_hash"] == rs.arm_hash(k["arm"]) for k in keys)
        meta = json.load(io.open(r["key"].parent / ("_meta_%s.json" % r["meta"]["family"]), encoding="utf-8"))
        assert meta["rng_seed"] == rs.SET_SEEDS[meta["family"]]
    ph_keys = list(csv.DictReader(io.open(ph["key"], encoding="utf-8")))
    assert {k["stratum"].split(":")[0] + ":" + k["stratum"].split(":")[1] for k in ph_keys if k["item_id"]} == \
        {"E:insufficient", "E:answered", "L:answered"}
    assert ph["meta"]["n_items"] == {"ph_a": 7, "ph_b": 4}
    # H4: 빈 답은 문항이 아니라 키에만 (UA=0). 질문은 원래(라운드 0) 문구
    h4_keys = list(csv.DictReader(io.open(h4["key"], encoding="utf-8")))
    assert any(k["source"] == "empty:UA=0" and k["item_id"] == "" for k in h4_keys)
    assert {k["arm"] for k in h4_keys} == {"E-SQ", "E-GR", "E-R1"}
    h5_keys = list(csv.DictReader(io.open(h5["key"], encoding="utf-8")))
    assert any(k["stratum"].startswith("new_answered") for k in h5_keys)


def test_sheet_sampling_is_reproducible(run_dir, tmp_path):
    a = rs.build_h4(run_dir, tmp_path / "a", n_slots=6, seeds=SEEDS)
    b = rs.build_h4(run_dir, tmp_path / "b", n_slots=6, seeds=SEEDS)
    assert a["html"].read_bytes() == b["html"].read_bytes()
    assert a["key"].read_bytes() == b["key"].read_bytes()


def test_empty_answer_shown_as_no_answer():
    assert rs.shown_answer("") == rs.NO_ANSWER
    assert rs.shown_answer("A fact [c1].") == "A fact."


def test_context_rebuild_checks_sha(run_dir):
    runs = ra.load_run(run_dir)
    rec = copy.deepcopy(runs[("E-GR", 1)]["t007"])
    ctx = format_context(StubRetriever().chunks, 12000)
    br = rec["rounds"][0]["branches"][0]
    br["context_sha256"] = hashlib.sha256(ctx.encode("utf-8")).hexdigest()
    text = stub_builder().for_subq(rec, br["branch"], 1, 0)
    assert "body 0" in text and "[c0]" not in text
    br["context_sha256"] = "0" * 64
    with pytest.raises(rs.ContextMismatch):
        stub_builder().for_subq(rec, br["branch"], 1, 0)


def test_allocate_and_g5_formula():
    assert rs.allocate({"a": 10, "b": 30}, 8) == {"a": 2, "b": 6}
    assert sum(rs.allocate({"a": 1, "b": 2, "c": 50}, 10).values()) == 10
    assert rs.g5_slots(None) == 200
    assert rs.g5_slots(0.1) == 120
    assert rs.g5_slots(0.5) == int(np.ceil((1.645 + 0.842) ** 2 * 0.25 / 0.01 * 1.2))


def test_h3_builds_without_h2_anchor_files(run_dir, tmp_path):
    r = rs.build_h3(run_dir, tmp_path / "h3", n_ea=5, h2_dir=tmp_path / "nope")
    _assert_blind(r["html"].read_text(encoding="utf-8"))
    keys = list(csv.DictReader(io.open(r["key"], encoding="utf-8")))
    assert len(keys) == 5 and all(k["stratum"].startswith("E:answered:") for k in keys)
    assert r["meta"]["n_anchor"] == 0


def test_ph_key_has_empty_pairs_for_g5(run_dir, tmp_path):
    add_legacy_arm(run_dir)
    ph = rs.build_ph(run_dir, tmp_path / "ph", stub_builder(), n_eb=4, n_ea=4, n_la=3)
    keys = list(csv.DictReader(io.open(ph["key"], encoding="utf-8")))
    labels = tmp_path / "labels.csv"
    with io.open(labels, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "item_id", "human_label"])
        for k in keys:
            if k["item_id"] and k["tab"] == "A":
                w.writerow(["ph_a", k["item_id"], "답함"])
    g = rs.g5_from_labels(ph["key"], labels)
    # E insufficient 슬롯의 L 짝(답 있음, 라벨됨) 또는 L 빈 답(UA=0)으로 짝이 생긴다
    assert g["n_pairs"] >= 1 and g["slots"] >= 120
