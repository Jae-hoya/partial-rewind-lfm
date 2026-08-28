"""llm.py 테스트 — 실 API 호출 없음. transport 를 mock 으로 끼운다."""

from __future__ import annotations

import json

import pytest

from lfm.llm import LLM, LLMError, LLMResult, cache_key


class CountingTransport:
    """호출마다 다른 텍스트를 돌려준다 — 캐시 히트/미스가 눈에 보이도록."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *, model, system, prompt, max_tokens):
        self.calls.append(
            {"model": model, "system": system, "prompt": prompt, "max_tokens": max_tokens}
        )
        n = len(self.calls)
        return ("resp-" + str(n), 10 + n, 20 + n, "stop")


def make(cfg, transport):
    return LLM(cfg, transport=transport)


# --- 캐시 키 -----------------------------------------------------------


def test_cache_key_covers_all_five_components():
    base = dict(model="m", system="s", prompt="p", max_tokens=100, cache_nonce="")
    k0 = cache_key(**base)
    for field, other in [
        ("model", "m2"),
        ("system", "s2"),
        ("prompt", "p2"),
        ("max_tokens", 101),
        ("cache_nonce", "r1"),
    ]:
        assert cache_key(**{**base, field: other}) != k0, field


def test_cache_key_is_not_string_concatenation():
    """sha256("ab"+"c") == sha256("a"+"bc") 함정. 정규화 JSON 이면 갈린다."""
    a = cache_key(model="m", system="ab", prompt="c", max_tokens=1)
    b = cache_key(model="m", system="a", prompt="bc", max_tokens=1)
    assert a != b


# --- 캐시 히트/미스 ----------------------------------------------------


def test_cache_miss_then_hit(tmp_config):
    t = CountingTransport()
    llm = make(tmp_config, t)
    r1 = llm.call("q", tag="unit")
    r2 = llm.call("q", tag="unit")
    assert len(t.calls) == 1, "두 번째 호출은 캐시에서 와야 한다"
    assert r1.cached is False and r2.cached is True
    assert r2.text == r1.text
    assert (r2.tokens_in, r2.tokens_out, r2.latency_ms) == (
        r1.tokens_in,
        r1.tokens_out,
        r1.latency_ms,
    ), "재실행 결과가 바이트 동일하려면 캐시가 토큰·지연도 그대로 돌려줘야 한다"


def test_nonce_forces_a_new_call(tmp_config):
    """SPEC 변경 #1: 라운드 교란의 유일한 캐시 회피 수단이 cache_nonce 다."""
    t = CountingTransport()
    llm = make(tmp_config, t)
    r0 = llm.call("q", tag="gen", cache_nonce="")
    r1 = llm.call("q", tag="gen", cache_nonce="r1")
    r2 = llm.call("q", tag="gen", cache_nonce="r2")
    assert len(t.calls) == 3
    assert len({r0.text, r1.text, r2.text}) == 3, "라운드마다 응답이 갈려야 한다"


def test_same_nonce_replays_identically(tmp_config):
    """같은 config 2회 실행 = 같은 nonce = 캐시 히트 = 동일 결과 (Acceptance Criteria)."""
    t = CountingTransport()
    first = [make(tmp_config, t).call("q", tag="gen", cache_nonce="r1").text for _ in range(1)]
    second = [make(tmp_config, t).call("q", tag="gen", cache_nonce="r1").text for _ in range(1)]
    assert first == second
    assert len(t.calls) == 1


def test_cache_disabled_always_calls(tmp_config):
    data = tmp_config.as_dict()
    data["llm"]["cache_enabled"] = False
    from lfm.config import Config

    t = CountingTransport()
    llm = make(Config(data), t)
    llm.call("q", tag="unit")
    llm.call("q", tag="unit")
    assert len(t.calls) == 2


def test_cache_file_is_utf8_json(tmp_config):
    class KoreanTransport:
        def __call__(self, *, model, system, prompt, max_tokens):
            return ("한국어 응답", 1, 2, "stop")

    llm = make(tmp_config, KoreanTransport())
    r = llm.call("질의", tag="unit")
    assert r.text == "한국어 응답"
    files = list(llm.cache_dir.rglob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["text"] == "한국어 응답"


# --- 파라미터 전달 ------------------------------------------------------


def test_max_tokens_defaults_from_config_and_is_overridable(tmp_config):
    t = CountingTransport()
    llm = make(tmp_config, t)
    llm.call("a", tag="unit")
    llm.call("b", tag="unit", max_tokens=42)
    assert t.calls[0]["max_tokens"] == tmp_config["llm.max_tokens"]
    assert t.calls[1]["max_tokens"] == 42


def test_system_prompt_is_passed_through(tmp_config):
    t = CountingTransport()
    make(tmp_config, t).call("p", system="sys", tag="unit")
    assert t.calls[0]["system"] == "sys"


# --- 통계 ---------------------------------------------------------------


def test_usage_stats_separates_billed_from_cached(tmp_config):
    t = CountingTransport()
    llm = make(tmp_config, t)
    r = llm.call("q", tag="gen")
    llm.call("q", tag="gen")  # 캐시 히트
    s = llm.stats
    assert (s.calls, s.api_calls, s.cache_hits) == (2, 1, 1)
    assert s.billed_tokens_in == r.tokens_in
    assert s.tokens_in == r.tokens_in * 2
    assert s.by_tag["gen"]["calls"] == 2


# --- 오류 처리 ----------------------------------------------------------


def test_non_retryable_error_raises_immediately(tmp_config):
    calls = []

    def boom(*, model, system, prompt, max_tokens):
        calls.append(1)
        raise ValueError("bad request")

    with pytest.raises(LLMError):
        make(tmp_config, boom).call("q", tag="unit")
    assert len(calls) == 1, "재시도 불가 오류는 즉시 올라와야 한다"


def test_retryable_error_is_retried_then_raises(tmp_config, monkeypatch):
    monkeypatch.setattr("lfm.llm.time.sleep", lambda *_: None)
    calls = []

    class RateLimitError(Exception):
        pass

    def flaky(*, model, system, prompt, max_tokens):
        calls.append(1)
        raise RateLimitError("429")

    with pytest.raises(LLMError):
        make(tmp_config, flaky).call("q", tag="unit")
    assert len(calls) == tmp_config["llm.max_retries"]


def test_retryable_error_recovers(tmp_config, monkeypatch):
    monkeypatch.setattr("lfm.llm.time.sleep", lambda *_: None)
    state = {"n": 0}

    class APITimeoutError(Exception):
        pass

    def flaky(*, model, system, prompt, max_tokens):
        state["n"] += 1
        if state["n"] == 1:
            raise APITimeoutError("timeout")
        return ("ok", 1, 1, "stop")

    assert make(tmp_config, flaky).call("q", tag="unit").text == "ok"


def test_non_openai_provider_is_rejected(tmp_config):
    """CLAUDE.md 모델 정책: 상용 API 는 OpenAI 만."""
    from lfm.config import Config

    data = tmp_config.as_dict()
    data["llm"]["provider"] = "anthropic"
    llm = LLM(Config(data))  # transport 없음 -> OpenAI 경로
    with pytest.raises(LLMError, match="provider"):
        llm.call("q", tag="unit")


def test_llm_result_is_frozen():
    r = LLMResult(text="t", tokens_in=1, tokens_out=2, latency_ms=3)
    with pytest.raises(Exception):
        r.text = "x"  # type: ignore[misc]
