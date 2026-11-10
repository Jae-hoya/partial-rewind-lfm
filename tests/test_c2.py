"""코퍼스2 어댑터 테스트 (PREREG_R3 §4). 네트워크·LLM·GPU 없음. 저장소의 data_c2/·results/c2/·.cache/llm_c2 를 만들지 않는다."""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "data", ROOT / "data_c2", ROOT / "experiments"):
    sys.path.insert(0, str(p))

import approve_c2  # noqa: E402
import c2_analyze  # noqa: E402
import c2_config_diff  # noqa: E402
import c2_negative_control as NC  # noqa: E402
import c2_run  # noqa: E402
import download_c2  # noqa: E402
import heldout_c2  # noqa: E402
import topics as T  # noqa: E402
import topics_c2  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.types import CITATION_RE  # noqa: E402

WORDS = ("quantum lattice entropy boson fermion spectral manifold topology kernel gradient stochastic markov "
         "protein genome neuron cortex tensor sparse convex dual photon plasma galaxy redshift neutrino catalyst "
         "polymer graphene enzyme vaccine epidemic climate aerosol glacier seismic fluid turbulence vortex").split()


def _abstract(rng, n=90):
    return " ".join(WORDS[int(i)] for i in rng.integers(0, len(WORDS), size=n)) + "."


def _abstract_doc(i, n=80):
    """문서마다 어휘가 겹치지 않는 초록 (Jaccard·BM25 판정이 뚜렷하게)."""
    return " ".join("term%sq%d" % ("abcdefghij"[i], j % 40) for j in range(n)) + "."


def _fake_raw(tmp: Path) -> tuple[list[Path], dict]:
    """두 파일 원천(Kaggle 스키마). 적격 12건 + 제외 4건(짧음·withdrawn·빈 제목) + 같은 id 중복 1건."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        rows.append({"id": "2101.%05d" % (100 - i), "title": "Paper  %d\n  on things" % i,
                     "abstract": "  " + _abstract(rng) + "\n  more  ",
                     "categories": ["cs.LG stat.ML", "hep-th", "math.CO cs.DM", "cond-mat.mes-hall"][i % 4],
                     "update_date": "2021-01-%02d" % (i + 1), "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"})
    rows.append({"id": "2101.00001", "title": "Short", "abstract": "too short", "categories": "cs.AI", "update_date": "2021-01-01", "license": None})
    rows.append({"id": "2101.00002", "title": "Gone", "abstract": "Withdrawn " + _abstract(rng), "categories": "cs.AI", "update_date": "2021-01-01", "license": None})
    rows.append({"id": "2101.00003", "title": "  ", "abstract": _abstract(rng), "categories": "cs.AI", "update_date": "2021-01-01", "license": None})
    # 같은 id, 더 늦은 update_date → 이 행이 남아야 한다 (제목으로 확인)
    rows.append({"id": "2101.00100", "title": "Paper 0 revised", "abstract": _abstract(rng), "categories": "cs.LG",
                 "update_date": "2022-05-05", "license": None})
    f1, f2 = tmp / "raw" / "data" / "train-00000-of-00002.parquet", tmp / "raw" / "data" / "train-00001-of-00002.parquet"
    f1.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows[:8]), f1)
    pq.write_table(pa.Table.from_pylist(rows[8:]), f2)
    return [f1, f2], {"rows": rows}


def _tmp_cfg(tmp: Path, **over) -> Path:
    d = yaml.safe_load(io.open(ROOT / "config_c2.yaml", encoding="utf-8"))
    d["paths"].update({"data_dir": str(tmp / "data_c2"), "raw_dir": str(tmp / "raw"), "index_dir": str(tmp / "data_c2" / "index"),
                       "cache_dir": str(tmp / "cache"), "results_dir": str(tmp / "results")})
    d["data"]["min_chunks_per_topic"] = 1
    for k, v in over.items():
        sec, key = k.split(".")
        d[sec][key] = v
    p = tmp / "config_c2_test.yaml"
    yaml.safe_dump(d, io.open(p, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    return p


# --------------------------------------------------------------------------- config / 등록 코드
def test_config_c2_matches_table():
    assert c2_config_diff.check() == []
    base = io.open(ROOT / "config.yaml", encoding="utf-8", newline="").read()
    want = c2_config_diff.render(base, c2_config_diff.PLACEHOLDER_HF)
    assert want == io.open(ROOT / "config_c2.yaml", encoding="utf-8", newline="").read()
    # 표 밖의 키를 하나 바꾸면 실패해야 한다
    bad = want.replace("top_k: 5", "top_k: 6")
    tmp = ROOT / "tests" / "_tmp_bad_c2.yaml"
    try:
        io.open(tmp, "w", encoding="utf-8", newline="").write(bad)
        errs = c2_config_diff.check(c2_path=tmp)
        assert any("바이트" in e for e in errs) and any("retrieval.top_k" in e for e in errs)
    finally:
        tmp.unlink(missing_ok=True)
    assert c2_config_diff.check(require_manifest=True, manifest=ROOT / "tests" / "_no_manifest.json")


def test_registered_code_untouched():
    out = subprocess.run(["git", "diff", "HEAD", "--stat", "--", "experiments/r2_run.py", "experiments/r2_analyze.py",
                          "lfm/", "config.yaml", "data/topics.py", "data/heldout.py", "data/approve_topics.py",
                          "data/chunk_index.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    assert out.strip() == ""


def test_prompt_hashes_and_phrase_change():
    c2_run.assert_prompts()
    import lfm.perspectives as P
    assert "encyclopedia" in P.PROMPT                     # 분해 프롬프트는 그대로
    new = topics_c2.c2_phrase_prompt(T.PHRASE_PROMPT)
    assert new.count(topics_c2.PHRASE_NEW) == 1 and topics_c2.PHRASE_OLD not in new
    assert len(new) - len(T.PHRASE_PROMPT) == len(topics_c2.PHRASE_NEW) - len(topics_c2.PHRASE_OLD)


# --------------------------------------------------------------------------- 다운로드 구성 (오프라인)
def test_build_corpus_from_fake_raw(tmp_path):
    files, _ = _fake_raw(tmp_path)
    stats = download_c2.build(files, tmp_path / "data_c2", min_abs=400, n_sample=10)
    t = pq.read_table(tmp_path / "data_c2" / "corpus.parquet")
    assert t.num_rows == 10 and stats["n_eligible"] == 12
    assert stats["excluded_row_counts"]["withdrawn"] == 1 and stats["excluded_row_counts"]["abstract_lt_400"] == 1
    assert stats["excluded_row_counts"]["empty_title_or_abstract"] == 1 and stats["excluded_row_counts"]["duplicate_id_rows"] == 1
    recs = t.to_pylist()
    assert [r["identifier"] for r in recs] == list(range(1, 11))
    assert [r["arxiv_id"] for r in recs] == sorted(r["arxiv_id"] for r in recs)
    for r in recs:
        assert "  " not in r["abstract"] and "\n" not in r["name"] and len(r["abstract"]) >= 400
        sec = json.loads(r["sections"])
        assert sec[0]["name"] == "Abstract" and sec[0]["has_parts"][0]["value"] == r["abstract"]
        assert r["references"] == [] and json.loads(r["infoboxes"])[0]["name"] == r["categories"].split()[0]
    dup = [r for r in recs if r["arxiv_id"] == "2101.00100"]
    assert not dup or dup[0]["name"] == "Paper 0 revised"
    # 같은 입력 → 같은 출력 (seed 4500)
    s2 = download_c2.build(files, tmp_path / "again", min_abs=400, n_sample=10)
    assert s2["corpus_sha256"] == stats["corpus_sha256"]
    assert download_c2.archive_of("cond-mat.mes-hall") == "cond-mat" and download_c2.archive_of("hep-th") == "hep-th"


# --------------------------------------------------------------------------- 10문서 끝까지: held-out → 청킹 → 인덱스 → 검색
def test_fake_parquet_end_to_end_chunk_index(tmp_path):
    import chunk_index
    from lfm.retriever import BM25Retriever

    rng = np.random.default_rng(1)
    dd = tmp_path / "data_c2"; dd.mkdir()
    recs = []
    abstracts = [_abstract_doc(i) for i in range(10)]
    abstracts[9] = abstracts[0].replace("termaq0", "zzzz", 1)          # 문서 1 의 근접 중복 → Jaccard 로 빠져야 한다
    for i in range(10):
        recs.append(download_c2.corpus_record(i + 1, "2201.%05d" % i, "Title %d" % i, abstracts[i],
                                              ["cs.LG", "hep-th", "math.CO"][i % 3], None))
    recs[5]["name"] = "Title 2"                                          # 문서 3 과 같은 제목 → 빠져야 한다
    pq.write_table(pa.Table.from_pylist(recs, schema=download_c2.SCHEMA), dd / "corpus.parquet")
    cfgp = _tmp_cfg(tmp_path)
    topics = [{"id": "t000", "topic": abstracts[0][:120], "target_article": "Title 0", "target_identifier": 1},
              {"id": "t001", "topic": abstracts[2][:120], "target_article": "Title 2", "target_identifier": 3}]
    yaml.safe_dump({"approved": True, "topics": topics}, io.open(dd / "topics_pilot50.yaml", "w", encoding="utf-8"))
    assert heldout_c2.main(["--config", str(cfgp)]) == 0
    h = json.load(io.open(dd / "heldout_ids.json", encoding="utf-8"))
    assert h["by_topic"]["t000"]["heldout_identifiers"] == [1, 10]
    assert h["by_topic"]["t001"]["heldout_identifiers"] == [3, 6]
    assert h["heldout_identifiers"] == [1, 3, 6, 10]

    # 커버리지 검사(k = max(3·min, 60))는 10문서로는 돌 수 없다 — 바꾸지 않은 wiki 코드 그대로이므로 건너뛴다
    rc = chunk_index.main(["--config", str(cfgp), "--skip-coverage", "--stats-out", str(tmp_path / "index_stats.json")])
    assert rc == 0
    st = json.load(io.open(tmp_path / "index_stats.json", encoding="utf-8"))
    assert st["n_documents_indexed"] == 6 and st["n_chunks"] == 6 and st["n_documents_heldout_skipped"] == 4
    chunks = [json.loads(l) for l in io.open(dd / "index" / "chunks.jsonl", encoding="utf-8")]
    assert len(chunks) == 6
    for c in chunks:
        assert re.fullmatch(r"d\d+s0p0", c["chunk_id"]) and c["section_path"] == "Abstract"
        assert CITATION_RE.findall("claim [%s]." % c["chunk_id"]) == [c["chunk_id"]]
        assert c["doc_id"] not in {1, 3, 6, 10}
    r = BM25Retriever(load_config(cfgp))
    hits = r.search(abstracts[4][:200], 3)
    assert hits and hits[0].chunk_id == "d5s0p0"


# --------------------------------------------------------------------------- 토픽 후보 (LLM 없음)
def _docs(n=40, seed=2):
    rng = np.random.default_rng(seed)
    cats = ["cs.LG", "cs.CV", "stat.ML", "hep-th", "hep-ph", "math.CO", "math.PR", "q-bio.NC", "astro-ph.GA"]
    docs = []
    for i in range(n):
        k = int(rng.integers(1, 3))
        docs.append({"identifier": i + 1, "name": "Doc %d" % i, "url": "", "abstract": _abstract(rng, 60),
                     "infobox_types": [cats[int(j)] for j in rng.choice(len(cats), size=k, replace=False)]})
    return docs


def _tcfg(**kw):
    t = {"min_related_docs": 3, "min_infobox_types": 2, "min_term_hits": 2, "n_terms": 5, "df_lo": 2, "df_hi": 30,
         "min_target_abstract_chars": 0, "min_target_section_chars": 0, "min_target_references": 0,
         "max_per_infobox_type": 2, "n_candidates": 60, "min_term_len": 4}
    t.update(kw)
    return Config({"topics": t}, source="test")


def test_find_candidates_c2_reduces_to_original():
    docs = _docs()
    postings = T.build_postings(docs, 4)
    meta = {d["identifier"]: (0, 0) for d in docs}
    a = T.find_candidates(docs, postings, meta, _tcfg())
    b = topics_c2.find_candidates_c2(docs, postings, meta, _tcfg(min_archives=0, max_per_archive=None))
    assert a and [{k: v for k, v in c.items() if k not in ("n_archives", "archives")} for c in b] == a
    c = topics_c2.find_candidates_c2(docs, postings, meta, _tcfg(min_archives=3, max_per_archive=1))
    assert all(x["n_archives"] >= 3 for x in c)
    prim = [topics_c2.archive_of(x["primary_type"]) for x in c]
    assert len(prim) == len(set(prim))


def test_archive_filter():
    keep, rej = approve_c2.archive_filter([{"id": "t0", "n_archives": 1}, {"id": "t1", "n_archives": 2}], 2)
    assert [k["id"] for k in keep] == ["t1"] and rej[0]["id"] == "t0"


# --------------------------------------------------------------------------- split / arm 설정
def test_split_and_arm_cfg(tmp_path):
    topics = [{"id": "t%03d" % i} for i in range(28)]
    s = c2_run.make_split_c2(topics, path=tmp_path / "split.json")
    pos = sorted(int(i) for i in np.random.default_rng(4501).choice(28, 3, replace=False))
    assert s["pilot3"] == ["t%03d" % i for i in pos] and len(s["heldout25"]) == 25
    assert not set(s["pilot3"]) & set(s["heldout25"])
    assert c2_run.make_split_c2([], path=tmp_path / "split.json") == s           # 고정
    red = c2_run.make_reduced(s, 20, path=tmp_path / "red.json")
    assert len(red) == 20 and set(red) <= set(s["heldout25"])
    with pytest.raises(SystemExit):
        c2_run.make_reduced(s, 19, path=tmp_path / "red2.json")
    import argparse
    import r2_run
    base = load_config(ROOT / "config_c2.yaml")
    ns = argparse.Namespace(cache_dir=c2_run.CACHE_C2, cache_only=False, max_billed_in=600_000, max_billed_out=160_000)
    for arm in r2_run.ORDER:
        cfg = r2_run.arm_cfg(base, arm, 1, c2_run.c2_args(ns))
        assert cfg.get("paths.cache_dir_ro", None) is None and cfg["paths.cache_dir"] == ".cache/llm_c2"
        assert cfg["llm.cache_policy"] == "live" and cfg["budget.max_billed_in"] == 600_000
        assert cfg["paths.index_dir"] == "data_c2/index" and cfg.get("llm.decompose_model", None) is None


def test_preflight_blocks_without_manifest():
    if (ROOT / "data_c2" / "MANIFEST_C2.json").exists():
        pytest.skip("MANIFEST 가 이미 있다 (C-L 뒤)")
    with pytest.raises(SystemExit):
        c2_run.preflight(load_config(ROOT / "config_c2.yaml"))


# --------------------------------------------------------------------------- 음성 통제
def _write_run(root: Path):
    texts = {"t000": "Alpha binds beta [c1].", "t001": "Gamma melts delta [c3].", "t002": "Epsilon orbits zeta [c5]."}
    for arm in ("E-GR", "L-GR"):
        for qid, txt in texts.items():
            n = int(qid[-1]) * 2 + 1
            sas = [{"index": 0, "sub_question": "q", "text": txt, "citations": ["c%d" % n], "status": "answered" if arm[0] == "E" else None,
                    "produced_round": 0},
                   {"index": 1, "sub_question": "q", "text": "Other fact here [c%d]." % (n + 1), "citations": ["c%d" % (n + 1)],
                    "status": "answered" if arm[0] == "E" else None, "produced_round": 0},
                   {"index": 2, "sub_question": "q", "text": "", "citations": [], "status": "insufficient" if arm[0] == "E" else None,
                    "produced_round": 0}]
            for sa in sas:
                if sa["status"] is None:
                    sa.pop("status")
            rec = {"query_id": qid, "rounds": [{"round": 0, "action": {}, "verify": {"passed": False, "agg": {}},
                                                "branches": [{"branch": "b0", "executed": True, "sub_answers": sas,
                                                              "evidence": {"grounding": 0.0}}]}]}
            d = root / arm / "s1" / "topics"; d.mkdir(parents=True, exist_ok=True)
            io.open(d / (qid + ".json"), "w", encoding="utf-8").write(json.dumps(rec))


class _KeywordEst:
    """문장의 첫 낱말이 전제에 있으면 지지 — off-topic 에서는 0 이어야 한다."""
    def grounded_many(self, items):
        return [bool(p) and s.split()[0].lower() in " ".join(p).lower() for s, p in items]


def test_negative_control(tmp_path):
    _write_run(tmp_path)
    chunk_text = {"c%d" % i: "chunk %d text about %s" % (i, w) for i, w in
                  zip(range(1, 7), ["alpha", "misc", "gamma", "misc", "epsilon", "misc"])}
    res = NC.run(tmp_path, 1, chunk_text, _KeywordEst())
    assert res["n_units_sampled"] == 6 and res["status"] == "remeasure_on_main_s1"
    assert res["offtopic"]["n"] == 6 and res["offtopic"]["false_positive"] == 0
    assert res["near_miss"]["n"] == 6
    lo, hi = NC.wilson(0, 300)
    assert lo == 0.0 and 0.012 < hi < 0.013


# --------------------------------------------------------------------------- 분석 판정
def test_verdict_rules():
    s = {"available": True, "point": 0.03, "claim": True, "ci95": [0.01, 0.05], "per_seed": {}}
    assert c2_analyze.verdict_savings(s)["verdict"] == "재현"
    assert c2_analyze.verdict_savings({**s, "claim": False})["verdict"] == "방향만 일치"
    assert c2_analyze.verdict_savings({**s, "point": -0.01, "claim": False})["verdict"] == "재현 실패"
    assert c2_analyze.verdict_savings(s, freeze_br=0)["verdict"] == "재현 실패"
    legs = {"coverage": {"available": True, "diff": -0.01, "ci90": [-0.03, 0.01], "noninferior": True},
            "grounding": {"available": True, "diff": -0.08, "ci90": [-0.1, -0.06], "noninferior": False}}
    ok = {"available": True, "sensitive": True, "egr_ci90_lo": 0.3}
    bad = {"available": True, "sensitive": False, "egr_ci90_lo": 0.02}
    assert c2_analyze.verdict_noninf(legs, {"coverage": ok, "grounding": ok}, True, None)["verdict"] == "재현 실패"
    assert c2_analyze.verdict_noninf(legs, {"coverage": ok, "grounding": ok}, False, None)["verdict"] == "재현"
    assert c2_analyze.verdict_noninf(legs, {"coverage": bad, "grounding": ok}, False, None)["verdict"] == "판정 불가"
    assert c2_analyze.verdict_noninf(legs, {"coverage": ok, "grounding": ok}, True, "퇴화")["verdict"] == "판정 불가"


@pytest.mark.skipif(not (ROOT / "results/r2/pilot5").exists(), reason="R2 산출 없음")
def test_c2_analyze_runs_on_r2_pilot(tmp_path):
    """코드 시험: 코퍼스2 래퍼가 실제 r2 산출 형식에서 끝까지 돈다 (결과 아님, 쓰지 않음)."""
    res = c2_analyze.analyze_c2(ROOT / "results/r2/pilot5", rescore=False, pilot=True, n_boot=200,
                                corpus1=None, negctl=None)
    assert "sensitivity_30" not in res and res["gates"]["G4"]["limit_hard"] == [12_600_000, 3_400_000]
    assert "2_decompose_api_calls" in res["gates"]["G1"] and "switch_rule_4_2_1" in res["c2"]
    import r2_analyze as RA
    assert RA.PILOT_DEV == {"t000", "t001", "t002", "t003", "t004"}          # 원복
    res2 = c2_analyze.analyze_c2(ROOT / "results/r2/pilot5", rescore=False, pilot=False, n_boot=200,
                                 corpus1=ROOT / "results/r2/heldout35", negctl=None)
    for h in ("C-H1", "C-H2", "C-H3", "C-H4"):
        assert res2["c2"][h]["verdict"] in ("재현", "방향만 일치", "재현 실패", "판정 불가")
    assert res2["c2"]["corpus1_consistency"]["C-H1"]["available"]
