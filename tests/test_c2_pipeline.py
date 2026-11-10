"""코퍼스2 어댑터의 오프라인 연쇄 시험 (PREREG_R3 §4.2, C1-C2 조건). 네트워크·LLM·GPU 없음.

  1. download_c2.main --build-only (가짜 원천 parquet) → MANIFEST → license_gate_c2 → c2_config_diff (MANIFEST 대조)
  2. 가짜 코퍼스 → topics_c2.main (문구 생성만 가짜로 바꿈, LLM 객체도 가짜 — api 0) → approve_c2.main
     → heldout_c2.main → data/chunk_index.py → BM25Retriever → c2_run.make_split_c2
모든 산출은 tmp_path 에만 쓴다. 저장소의 data_c2/ 산출·results/c2/·.cache/llm_c2 를 만들지 않는 것을 마지막에 assert 한다.
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "data", ROOT / "data_c2", ROOT / "experiments", ROOT / "tests"):
    sys.path.insert(0, str(p))

import approve_c2  # noqa: E402
import c2_config_diff  # noqa: E402
import c2_run  # noqa: E402
import download_c2  # noqa: E402
import heldout_c2  # noqa: E402
import license_gate_c2  # noqa: E402
import topics as T  # noqa: E402
import topics_c2  # noqa: E402
from lfm.types import CITATION_RE  # noqa: E402
from test_c2 import _abstract, _fake_raw, _tmp_cfg  # noqa: E402

REPO_OUTPUTS = (ROOT / "data_c2" / "corpus.parquet", ROOT / "data_c2" / "MANIFEST_C2.json", ROOT / "data_c2" / "index",
                ROOT / "data_c2" / "topics_pilot50.yaml", ROOT / "data_c2" / "split_c2.json",
                ROOT / "results" / "c2", ROOT / ".cache" / "llm_c2")


def _repo_clean() -> list[str]:
    return [str(p) for p in REPO_OUTPUTS if p.exists()]


def test_download_build_only_manifest_license_config(tmp_path):
    before = _repo_clean()
    _fake_raw(tmp_path)
    cfgp = _tmp_cfg(tmp_path)
    out = tmp_path / "data_c2"
    rc = download_c2.main(["--config", str(cfgp), "--build-only", "--raw-dir", str(tmp_path / "raw"),
                           "--out-dir", str(out), "--n-sample", "10"])
    assert rc == 0
    m = json.load(io.open(out / "MANIFEST_C2.json", encoding="utf-8"))
    assert m["corpus"]["n_sampled"] == 10 and m["license_gate"]["status"] == "pending"
    assert len(m["source"]["raw_files"]) == 2 and m["source"]["fetched"] is False
    assert license_gate_c2.gate_status(out / "MANIFEST_C2.json") == "pending"
    # C-L 기록 (이 시험에서는 가짜 URL — 실제 C-L 은 사람이 공식 문서를 읽고 적는다)
    assert license_gate_c2.main(["--pass", "--evidence-url", "https://example.invalid/terms", "--statement", "test",
                                 "--checked-by", "pytest", "--manifest", str(out / "MANIFEST_C2.json")]) == 0
    assert license_gate_c2.gate_status(out / "MANIFEST_C2.json") == "passed"
    # MANIFEST 의 revision 으로 config_c2 를 렌더하면 §4.3 대조를 통과하고, 자리표시 config 는 실패해야 한다
    m = json.load(io.open(out / "MANIFEST_C2.json", encoding="utf-8"))
    m["source"]["hf_revision"] = "0123456789abcdef0123456789abcdef01234567"
    io.open(out / "MANIFEST_C2.json", "w", encoding="utf-8").write(json.dumps(m))
    hf, fm = c2_config_diff.hf_values(out / "MANIFEST_C2.json")
    assert fm and hf["hf_revision"].startswith("0123")
    rendered = tmp_path / "config_c2_rendered.yaml"
    base = io.open(ROOT / "config.yaml", encoding="utf-8", newline="").read()
    io.open(rendered, "w", encoding="utf-8", newline="").write(c2_config_diff.render(base, hf))
    assert c2_config_diff.check(c2_path=rendered, manifest=out / "MANIFEST_C2.json", require_manifest=True) == []
    errs = c2_config_diff.check(manifest=out / "MANIFEST_C2.json", require_manifest=True)   # 저장소 config(자리표시)
    assert any("hf_revision" in e for e in errs)
    assert _repo_clean() == before


def _fake_raw_ts(tmp: Path) -> dict:
    """HF parquet 형식(update_date = timestamp[s]) 두 파일. 같은 id 중복 3쌍:
    X: 앞 파일이 더 늦은 날짜 → 파일 순서가 아니라 날짜로 골라야 한다. Y: 같은 날, 뒤 파일이 더 늦은 시각.
    Z: 뒤 파일 행의 update_date 가 null → 날짜가 있는 앞 파일 행이 남는다."""
    from datetime import datetime
    rng = np.random.default_rng(5)
    row = lambda aid, title, ts: {"id": aid, "title": title, "abstract": _abstract(rng), "categories": "cs.LG",
                                  "update_date": ts, "license": None}
    f0 = [row("2101.%05d" % (500 + i), "Unique %d" % i, datetime(2020, 1, 1 + i)) for i in range(6)]
    f1 = [row("2101.%05d" % (600 + i), "Unique b%d" % i, datetime(2020, 2, 1 + i)) for i in range(6)]
    f0 += [row("2101.00010", "X later", datetime(2023, 3, 1)), row("2101.00020", "Y earlier", datetime(2021, 6, 1)),
           row("2101.00030", "Z dated", datetime(2019, 5, 5))]
    f1 += [row("2101.00010", "X earlier", datetime(2021, 1, 1)), row("2101.00020", "Y later", datetime(2021, 6, 1, 12, 0, 0)),
           row("2101.00030", "Z null", None)]
    schema = pa.schema([("id", pa.string()), ("title", pa.string()), ("abstract", pa.string()), ("categories", pa.string()),
                        ("update_date", pa.timestamp("s")), ("license", pa.string())])
    d = tmp / "raw" / "data"
    d.mkdir(parents=True)
    for k, rows in enumerate((f0, f1)):
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), d / ("train-%05d-of-00002.parquet" % k))
    # parquet 은 초 단위를 따로 두지 않아 다시 읽으면 timestamp[ms] 일 수 있다 — 어느 쪽이든 timestamp 경로를 탄다
    assert pa.types.is_timestamp(pq.read_schema(d / "train-00000-of-00002.parquet").field("update_date").type)
    return {"n_unique": 15}


def test_download_dedup_timestamp_update_date_later_wins(tmp_path):
    """r3fix_0: HF 원천의 update_date 는 timestamp[s]. 문자열로 바꿔 비교해도 같은 id 는 더 늦은 날짜(시각) 행이 남는다."""
    before = _repo_clean()
    info = _fake_raw_ts(tmp_path)
    cfgp = _tmp_cfg(tmp_path)
    out = tmp_path / "data_c2"
    rc = download_c2.main(["--config", str(cfgp), "--build-only", "--raw-dir", str(tmp_path / "raw"),
                           "--out-dir", str(out), "--n-sample", str(info["n_unique"])])
    assert rc == 0
    m = json.load(io.open(out / "MANIFEST_C2.json", encoding="utf-8"))
    assert m["corpus"]["n_unique_ids"] == m["corpus"]["n_eligible"] == info["n_unique"]
    assert m["corpus"]["excluded_row_counts"].get("duplicate_id_rows") == 3
    t = pq.read_table(out / "corpus.parquet", columns=["arxiv_id", "name"]).to_pylist()
    title = {r["arxiv_id"]: r["name"] for r in t}
    assert title["2101.00010"] == "X later"        # 앞 파일이지만 날짜가 더 늦다
    assert title["2101.00020"] == "Y later"        # 같은 날, 시각이 더 늦다
    assert title["2101.00030"] == "Z dated"        # null('') < 날짜
    # 원천 선언 형식 timestamp[s] 그대로의 문자열 변환도 시간 순서를 지킨다(pass1 과 같은 식)
    import pyarrow.compute as pc
    from datetime import datetime
    ts = pa.array([datetime(2021, 6, 1, 12), None, datetime(2021, 6, 1), datetime(2023, 3, 1)], pa.timestamp("s"))
    st = pc.fill_null(ts.cast(pa.string()), "").to_pylist()
    assert st[0] == "2021-06-01 12:00:00" and st[1] == ""
    assert sorted(st) == [st[1], st[2], st[0], st[3]]
    assert _repo_clean() == before


def _corpus(n=40, seed=3):
    """교차 archive 이웃이 생기도록 공통 어휘를 섞은 초록."""
    rng = np.random.default_rng(seed)
    shared = ["lattice", "entropy", "kernel", "gradient", "spectral", "manifold", "photon", "genome", "neuron", "plasma"]
    cats = ["cs.LG", "cs.CV stat.ML", "hep-th", "hep-ph hep-th", "math.CO", "math.PR cs.DM", "q-bio.NC", "astro-ph.GA",
            "cond-mat.mes-hall", "physics.optics"]
    recs = []
    for i in range(n):
        words = [shared[int(j)] for j in rng.integers(0, len(shared), size=25)]
        words += ["uniq%dw%d" % (i, j) for j in range(40)]
        rng.shuffle(words)
        recs.append(download_c2.corpus_record(i + 1, "2301.%05d" % i, "Paper %s about item%d" % ("alpha beta gamma delta epsilon zeta theta kappa lambda sigma".split()[i % 10], i),
                                              " ".join(words) + ".", cats[i % len(cats)], None))
    return recs


class _FakeStats:
    calls = cache_hits = api_calls = billed_tokens_in = billed_tokens_out = 0


class _FakeLLM:
    def __init__(self, cfg, ledger=None):
        self.model, self.reasoning_effort, self.stats = "fake", None, _FakeStats()


class _FakeLedger:
    def __init__(self, path=None):
        self.path = path


def test_topics_approve_heldout_index_split_offline(tmp_path, monkeypatch):
    import chunk_index
    import lfm.llm
    from lfm.config import load_config
    from lfm.retriever import BM25Retriever

    before = _repo_clean()
    dd = tmp_path / "data_c2"; dd.mkdir()
    recs = _corpus()
    pq.write_table(pa.Table.from_pylist(recs, schema=download_c2.SCHEMA), dd / "corpus.parquet")
    io.open(dd / "MANIFEST_C2.json", "w", encoding="utf-8").write(json.dumps({"license_gate": {"status": "passed"}}))
    cfgp = _tmp_cfg(tmp_path, **{"topics.min_related_docs": 5, "topics.min_infobox_types": 3, "topics.df_lo": 3,
                                 "topics.df_hi": 39, "topics.min_target_abstract_chars": 0, "topics.n_candidates": 6,
                                 "topics.max_per_archive": 2})

    # 코퍼스1 로더가 코퍼스2 스키마를 그대로 읽는다 (코드 변경 없음)
    docs = T.load_docs(dd / "corpus.parquet")
    meta = T.load_meta(dd / "corpus.parquet")
    assert len(docs) == 40 and docs[1]["infobox_types"] == ["cs.CV", "stat.ML"]
    assert all(meta[d["identifier"]][1] == 0 for d in docs)                  # references 빈 목록 → 0

    seen_prompt = {}

    def fake_phrase(llm, docs_, picked, batch):
        seen_prompt["p"] = T.PHRASE_PROMPT
        return {i: "Fake research topic phrase number %d about shared things" % i for i in range(len(picked))}

    orig_prompt = T.PHRASE_PROMPT
    monkeypatch.setattr(T, "phrase_topics", fake_phrase)
    monkeypatch.setattr(lfm.llm, "LLM", _FakeLLM)
    monkeypatch.setattr(lfm.llm, "BillingLedger", _FakeLedger)
    assert topics_c2.main(["--config", str(cfgp)]) == 0
    assert topics_c2.PHRASE_NEW in seen_prompt["p"] and T.PHRASE_PROMPT == orig_prompt     # 실행 중에만 바꾸고 원복
    cand = yaml.safe_load(io.open(dd / "topics_pilot50_candidates.yaml", encoding="utf-8"))
    assert cand["llm"]["api_calls"] == 0 and cand["n_candidates"] >= 3
    assert all(c["n_archives"] >= 2 and c["n_infobox_types"] >= 3 for c in cand["candidates"])
    arch = [topics_c2.archive_of(c["primary_infobox_type"]) for c in cand["candidates"]]
    assert max(arch.count(a) for a in set(arch)) <= 2

    assert approve_c2.main(["--config", str(cfgp), "--n", "3"]) == 0
    appr = yaml.safe_load(io.open(dd / "topics_pilot50.yaml", encoding="utf-8"))
    assert appr["approved"] is True and len(appr["topics"]) == 3 and appr["approval"]["mode"] == "delegated"
    assert [t["id"] for t in appr["topics"]] == [c["id"] for c in cand["candidates"]][:3] or appr["rejected"]

    assert heldout_c2.main(["--config", str(cfgp)]) == 0
    h = json.load(io.open(dd / "heldout_ids.json", encoding="utf-8"))
    tids = sorted(t["target_identifier"] for t in appr["topics"])
    assert set(tids) <= set(h["heldout_identifiers"])

    assert chunk_index.main(["--config", str(cfgp), "--skip-coverage", "--stats-out", str(tmp_path / "st.json")]) == 0
    st = json.load(io.open(tmp_path / "st.json", encoding="utf-8"))
    assert st["n_chunks"] == st["n_documents_indexed"] == 40 - len(h["heldout_identifiers"])
    chunks = [json.loads(line) for line in io.open(dd / "index" / "chunks.jsonl", encoding="utf-8")]
    assert all(re.fullmatch(r"d\d+s0p0", c["chunk_id"]) and CITATION_RE.findall("x [%s]." % c["chunk_id"]) for c in chunks)
    assert not {c["doc_id"] for c in chunks} & set(h["heldout_identifiers"])
    hits = BM25Retriever(load_config(cfgp)).search(appr["topics"][0]["topic"] + " lattice entropy", 5)
    assert hits and all(hit.chunk_id.endswith("s0p0") for hit in hits)

    split = c2_run.make_split_c2(appr["topics"], path=tmp_path / "split.json")
    assert len(split["pilot3"]) == 3 and split["heldout25"] == [] and "28 아님" in split["note"]
    assert _repo_clean() == before


class _DecomposeTransport:
    """분해 응답(유효 JSON)을 돌려주고 호출 수를 센다. 실 API 아님."""

    def __init__(self, text: str):
        self.text, self.calls = text, []

    def __call__(self, *, model, system, prompt, max_tokens, reasoning_effort=None):
        self.calls.append(prompt)
        return (self.text, 200, 800, "stop")


def test_live_decompose_once_then_cache_across_arms_and_seeds(tmp_path):
    """§4.4: cache_dir_ro = null 이면 분해가 토픽당 한 번 라이브로 나가고(llm.py:334 차단 없음), 이후 모든 arm·seed 는 캐시 재생.
    대조: wiki 처럼 cache_dir_ro 를 걸면 같은 미스가 CacheMiss 로 막힌다."""
    import argparse
    import r2_run
    from lfm.config import load_config
    from lfm.llm import LLM, BillingLedger, CacheMiss
    from lfm.perspectives import ROLES, decompose

    before = _repo_clean()
    base = load_config(ROOT / "config_c2.yaml")
    rows = [{"role": r, "question": "Q about %s?" % r, "sub_questions": ["a %s?" % r, "b %s?" % r, "c %s?" % r]} for r in ROLES]
    tr = _DecomposeTransport(json.dumps(rows))
    ledger = BillingLedger(tmp_path / "ledger.jsonl")
    ns = argparse.Namespace(cache_dir=str(tmp_path / "llm_c2"), cache_only=False, max_billed_in=600_000, max_billed_out=160_000)
    outs = []
    for arm, seed in [("L-BR", 1), ("L-GR", 1), ("E-SQ", 2), ("E-GR", 3)]:
        cfg = r2_run.arm_cfg(base, arm, seed, c2_run.c2_args(ns))
        llm = LLM(cfg, transport=tr, ledger=ledger)
        outs.append(decompose("What is known about lattice entropy?", llm, cfg))
        assert llm.stats.api_calls == (1 if (arm, seed) == ("L-BR", 1) else 0)
    assert len(tr.calls) == 1 and all(o == outs[0] for o in outs)
    led = [json.loads(line) for line in io.open(tmp_path / "ledger.jsonl", encoding="utf-8")]
    assert [r["tag"] for r in led] == ["decompose"] and led[0]["nonce"] in ("", None)
    assert any((tmp_path / "llm_c2").rglob("*"))
    # 대조: 읽기 전용 캐시를 건 설정(R2 wiki 실행)에서는 같은 미스가 막힌다
    ns_ro = argparse.Namespace(cache_dir=str(tmp_path / "llm_x"), cache_dir_ro=str(tmp_path / "ro_empty"), cache_only=False,
                               model=None, max_billed_in=1, max_billed_out=1)
    (tmp_path / "ro_empty").mkdir()
    cfg_ro = r2_run.arm_cfg(base, "L-BR", 1, ns_ro)
    import pytest
    with pytest.raises(CacheMiss):
        decompose("What is known about lattice entropy?", LLM(cfg_ro, transport=tr), cfg_ro)
    assert len(tr.calls) == 1
    assert _repo_clean() == before
