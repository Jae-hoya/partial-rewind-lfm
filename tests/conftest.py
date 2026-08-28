from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lfm.config import Config, load_config  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def real_config() -> Config:
    """저장소의 config.yaml 자체. 필수 키 검증 포함."""
    return load_config(REPO_ROOT / "config.yaml")


@pytest.fixture
def tmp_config(tmp_path, real_config) -> Config:
    """캐시·결과 경로만 tmp 로 돌린 사본. 테스트가 저장소를 오염시키지 않게."""
    data = real_config.as_dict()
    data["paths"]["cache_dir"] = str(tmp_path / "cache")
    data["paths"]["results_dir"] = str(tmp_path / "results")
    return Config(data, source="tmp_config")
