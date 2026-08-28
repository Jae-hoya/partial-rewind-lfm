"""config.yaml 로더.

CLAUDE.md 규약: 모델명·경로·시드·예산은 전부 config 에서 주입한다. 코드에 하드코딩 금지.
따라서 이 모듈은 값을 알지 못한다 — 키의 존재만 검사한다.
"""

from __future__ import annotations

import copy
import io
import os
from pathlib import Path
from typing import Any

import yaml

# 없으면 실행을 시작하면 안 되는 키. 점 표기 경로.
REQUIRED_KEYS = (
    "seed",
    "paths.cache_dir",
    "paths.results_dir",
    "data.max_articles",
    "data.min_chunks_per_topic",
    "retrieval.backend",
    "retrieval.top_k",
    "retrieval.retry_k_multiplier",
    "retrieval.max_chunks",
    "llm.model",
    "llm.max_tokens",
    "nli.model",
    "nli.tau_ground",
    "verify.tau_g_global",
    "verify.tau_c_global",
    "sweep.tau",
    "sweep.primary_tau",
    "sweep.bootstrap_n",
    "max_rounds",
    "retry_directives",
)

DEFAULT_CONFIG_PATH = "config.yaml"


class ConfigError(KeyError):
    """필수 키 부재 또는 정합성 위반."""


class Config:
    """중첩 dict 를 점 표기로 읽는 얇은 래퍼. 쓰기는 지원하지 않는다."""

    def __init__(self, data: dict[str, Any], source: str | None = None) -> None:
        self._data = copy.deepcopy(data)
        self.source = source

    # --- 읽기 ---------------------------------------------------------
    _MISSING = object()

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is Config._MISSING:
                    raise ConfigError(f"config 키 없음: {dotted} (source={self.source})")
                return default
            node = node[part]
        return copy.deepcopy(node) if isinstance(node, (dict, list)) else node

    def __getitem__(self, dotted: str) -> Any:
        return self.get(dotted)

    def __contains__(self, dotted: str) -> bool:
        return self.get(dotted, None) is not None

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # --- 파생 ---------------------------------------------------------
    def retry_directive(self, round_: int) -> str:
        """YAML 은 `1:` 을 int 키로 읽는다. 문자열 키로 쓴 config 도 받아준다."""
        table = self._data.get("retry_directives") or {}
        for key in (round_, str(round_)):
            if key in table:
                return table[key]
        raise ConfigError(f"retry_directives 에 라운드 {round_} 항목이 없다")

    def top_k_for_round(self, round_: int) -> int:
        mult = self.get("retrieval.retry_k_multiplier")
        if round_ >= len(mult):
            raise ConfigError(
                f"retry_k_multiplier 길이 {len(mult)} < 라운드 {round_}. max_rounds 와 맞춰라"
            )
        return int(self.get("retrieval.top_k") * mult[round_])

    def path(self, dotted: str) -> Path:
        return Path(self.get(dotted))


def validate(cfg: Config) -> None:
    """필수 키 + config 내부 정합성 검사. 실패하면 ConfigError."""
    missing = [k for k in REQUIRED_KEYS if cfg.get(k, None) is None]
    if missing:
        raise ConfigError(f"config 필수 키 누락: {missing} (source={cfg.source})")

    max_rounds = int(cfg.get("max_rounds"))
    mult = cfg.get("retrieval.retry_k_multiplier")
    if len(mult) < max_rounds:
        raise ConfigError(
            f"retrieval.retry_k_multiplier 길이({len(mult)}) < max_rounds({max_rounds})"
        )
    for r in range(1, max_rounds):
        cfg.retry_directive(r)  # 없으면 ConfigError

    primary = float(cfg.get("sweep.primary_tau"))
    if primary not in [float(t) for t in cfg.get("sweep.tau")]:
        raise ConfigError("sweep.primary_tau 가 sweep.tau 목록에 없다")


def load_config(path: str | os.PathLike[str] | None = None, *, strict: bool = True) -> Config:
    p = Path(path or os.environ.get("LFM_CONFIG", DEFAULT_CONFIG_PATH))
    with io.open(p, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"config 최상위가 매핑이 아니다: {p}")
    cfg = Config(data, source=str(p))
    if strict:
        validate(cfg)
    return cfg
