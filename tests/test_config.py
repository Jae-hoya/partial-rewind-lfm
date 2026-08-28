from __future__ import annotations

import io

import pytest
import yaml

from lfm.config import Config, ConfigError, REQUIRED_KEYS, load_config, validate


def test_repo_config_has_all_required_keys(real_config):
    for key in REQUIRED_KEYS:
        assert real_config.get(key, None) is not None, key


def test_spec_pinned_values(real_config):
    """SPEC v1.1 변경 #2, #5 가 config 에 실제로 박혀 있는지."""
    assert real_config["verify.tau_g_global"] == 0.80
    assert real_config["verify.tau_c_global"] == 0.80
    assert real_config["sweep.primary_tau"] == 0.8
    assert real_config["sweep.tau"] == [0.7, 0.8, 0.9]
    assert real_config["sweep.bootstrap_n"] == 1000
    assert real_config["data.max_articles"] == 30000
    assert real_config["retrieval.backend"] == "bm25s"
    assert real_config["retrieval.retry_k_multiplier"] == [1, 2, 3]
    assert real_config["max_rounds"] == 3


def test_korean_directives_roundtrip_utf8(real_config):
    """Windows 기본 인코딩(cp949)으로 읽으면 깨진다 — UTF-8 고정 확인."""
    assert real_config.retry_directive(1).startswith("이전 시도는")
    assert "하위 질문" in real_config.retry_directive(2)


def test_retry_directive_accepts_int_and_str_keys():
    cfg = Config({"retry_directives": {1: "int-key", "2": "str-key"}})
    assert cfg.retry_directive(1) == "int-key"
    assert cfg.retry_directive(2) == "str-key"
    with pytest.raises(ConfigError):
        cfg.retry_directive(3)


def test_top_k_for_round(real_config):
    assert real_config.top_k_for_round(0) == real_config["retrieval.top_k"]
    assert real_config.top_k_for_round(1) == real_config["retrieval.top_k"] * 2
    assert real_config.top_k_for_round(2) == real_config["retrieval.top_k"] * 3
    with pytest.raises(ConfigError):
        real_config.top_k_for_round(3)


def test_missing_key_raises(real_config):
    data = real_config.as_dict()
    del data["llm"]["model"]
    with pytest.raises(ConfigError):
        validate(Config(data))


def test_primary_tau_must_be_in_sweep(real_config):
    data = real_config.as_dict()
    data["sweep"]["primary_tau"] = 0.85
    with pytest.raises(ConfigError):
        validate(Config(data))


def test_retry_multiplier_shorter_than_max_rounds(real_config):
    data = real_config.as_dict()
    data["retrieval"]["retry_k_multiplier"] = [1, 2]
    with pytest.raises(ConfigError):
        validate(Config(data))


def test_config_is_a_copy_not_a_view(real_config):
    d = real_config.as_dict()
    d["seed"] = 999
    assert real_config["seed"] != 999


def test_load_config_from_disk_is_utf8(tmp_path, repo_root):
    src = (repo_root / "config.yaml").read_text(encoding="utf-8")
    p = tmp_path / "config.yaml"
    with io.open(p, "w", encoding="utf-8") as fh:
        fh.write(src)
    cfg = load_config(p)
    assert cfg["llm.model"] == yaml.safe_load(src)["llm"]["model"]
