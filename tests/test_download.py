from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import download as dl  # noqa: E402


def test_quotas_sum_to_total_and_are_balanced():
    q = dl.quotas(30000, 86)
    assert sum(q) == 30000
    assert max(q) - min(q) <= 1
    assert q == sorted(q, reverse=True), "나머지는 앞쪽 bin 이 갖는다 — 결정론적 배분"


def test_quotas_handles_more_bins_than_items():
    q = dl.quotas(3, 5)
    assert sum(q) == 3 and q == [1, 1, 1, 0, 0]


def test_shard_uri_pins_revision():
    uri = dl.shard_uri("wikimedia/structured-wikipedia", "enwiki/data/x_0.parquet", "abc123")
    assert uri == "datasets/wikimedia/structured-wikipedia@abc123/enwiki/data/x_0.parquet"
    assert "@abc123" in uri, "revision 이 빠지면 스냅샷이 고정되지 않는다"


def test_shard_regex_sorts_numerically_not_lexically():
    files = [
        "enwiki/data/enwiki_namespace_0_10.parquet",
        "enwiki/data/enwiki_namespace_0_2.parquet",
        "enwiki/data/enwiki_namespace_0_1.parquet",
    ]
    ordered = sorted(files, key=lambda f: int(dl.SHARD_RE.search(f).group(1)))
    assert [int(dl.SHARD_RE.search(f).group(1)) for f in ordered] == [1, 2, 10]
    assert sorted(files) != ordered, "문자열 정렬이면 10 이 2 앞에 온다"


def test_shard_regex_ignores_non_shard_files():
    assert dl.SHARD_RE.search("README.md") is None
    assert dl.SHARD_RE.search("enwiki/data/enwiki_namespace_0_0.parquet") is not None


def test_row_group_choice_is_reproducible_from_seed():
    """같은 seed + 같은 샤드 수 -> 같은 row group 선택. 재실행하면 같은 코퍼스."""

    def pick(seed: int, n_shards: int, n_rg: int) -> list[list[int]]:
        ss = np.random.SeedSequence(seed).spawn(n_shards)
        return [
            sorted(int(x) for x in np.random.default_rng(s).choice(n_rg, size=2, replace=False))
            for s in ss
        ]

    assert pick(42, 5, 80) == pick(42, 5, 80)
    assert pick(42, 5, 80) != pick(43, 5, 80)


def test_list_shards_raises_when_prefix_matches_nothing(monkeypatch):
    class FakeSibling:
        def __init__(self, name):
            self.rfilename = name

    class FakeApi:
        def dataset_info(self, *a, **k):
            return type("I", (), {"siblings": [FakeSibling("README.md")]})()

    monkeypatch.setitem(sys.modules, "huggingface_hub", type("M", (), {"HfApi": FakeApi})())
    with pytest.raises(SystemExit):
        dl.list_shards("repo", "rev", "enwiki/data/")
