"""LLM 단일 래퍼 (SPEC §3.1).

모든 LLM 호출은 이 모듈을 경유한다. 호출마다 (tokens_in, tokens_out, latency_ms) 를
반환하고 trace 에 기록한다.

디스크 캐시 키 = sha256(canonical_json(
    [model, system, prompt, max_tokens, reasoning_effort, cache_nonce]))
문자열 연결이 아니라 정규화 JSON 을 해시한다. 연결하면 ("ab","c") 와 ("a","bc") 가 충돌한다.

cache_nonce 는 라운드 교란의 유일한 캐시 회피 수단이다 (SPEC 변경 #1).
라운드 r>0 의 모든 호출은 cache_nonce=f"r{r}" 를 전달한다. nonce 가 라운드의 결정론적
함수이므로 같은 config 를 두 번 돌리면 같은 nonce -> 캐시 히트 -> 동일 결과가 된다.
재현성과 라운드 교란이 양립한다.

R2 (재실험설계_확정안_R2 §1, §3): 캐시 키에는 seed 가 없다. 그래서 `run.seed` 가 있으면
nonce 앞에 `s{seed}|{family}|` 를 붙여 seed 마다 독립 표본을 만든다. 분해(decompose)는
붙이지 않는다 — 모든 arm·seed 가 Test A 의 sub-question 을 바이트 그대로 쓰게 하기 위해서다.
seed 가 null 이면 nonce 가 바뀌지 않아 기존 캐시 키와 같다.
`llm.cache_policy: cache_only` 면 캐시 미스를 예외로 막는다. 기본(live)에서는 미스가 나면
조용히 API 를 부른다 — "호출 0" 재분석은 반드시 cache_only 로 돌린다.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

__all__ = ["LLMResult", "LLMError", "CacheMiss", "BudgetExceeded", "LLM", "UsageStats",
           "cache_key", "load_dotenv_if_present", "UNSEEDED_TAGS"]

# seed 접두사를 붙이지 않는 호출. 분해는 모든 arm·seed 에서 Test A 캐시를 그대로 재생한다.
UNSEEDED_TAGS = frozenset({"decompose", "decompose_repair"})


class LLMError(RuntimeError):
    """LLM 호출 실패 (재시도 소진 포함)."""


class CacheMiss(LLMError):
    """cache_only 정책에서 캐시에 없는 호출. 실제 API 를 부르지 않고 멈춘다."""


class BudgetExceeded(LLMError):
    """누적 청구 토큰이 budget.max_billed_* 를 넘었다."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    model: str = ""
    tag: str = ""
    cached: bool = False
    finish_reason: str = ""
    effective_nonce: str = ""        # 캐시 키에 실제로 들어간 nonce (seed 접두사 포함)
    tokens_in_cached: int = 0        # provider 쪽 프롬프트 캐시로 할인된 입력 토큰
    tokens_reasoning: int = 0        # 출력 토큰 중 reasoning 몫
    model_returned: str = ""         # provider 가 실제로 돌려준 모델 id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Transport(Protocol):
    """실제 API 호출부. 테스트·mock 은 이 자리에 함수를 끼워 넣는다."""

    def __call__(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> tuple[str, int, int, str]:
        """returns (text, tokens_in, tokens_out, finish_reason)"""


def _canonical(
    model: str, system: str, prompt: str, max_tokens: int, reasoning_effort: str, cache_nonce: str
) -> str:
    return json.dumps(
        [model, system, prompt, int(max_tokens), reasoning_effort, cache_nonce],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def cache_key(
    *,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int,
    reasoning_effort: str = "",
    cache_nonce: str = "",
) -> str:
    """reasoning_effort 도 키에 들어간다. 응답을 바꾸는 파라미터를 키에서 빼면
    effort 를 조정한 뒤에도 예전 응답이 캐시에서 그대로 나온다."""
    payload = _canonical(model, system, prompt, max_tokens, reasoning_effort, cache_nonce)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class UsageStats:
    """세션 누적. 비용 로그(Acceptance Criteria)의 근거."""

    calls: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    billed_tokens_in: int = 0  # 캐시 히트를 제외한 실제 청구 대상
    billed_tokens_out: int = 0
    billed_tokens_in_cached: int = 0   # 청구분 중 provider 캐시 할인 입력
    billed_tokens_reasoning: int = 0   # 청구분 중 reasoning 출력
    latency_ms: int = 0
    by_tag: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, r: LLMResult) -> None:
        self.calls += 1
        self.tokens_in += r.tokens_in
        self.tokens_out += r.tokens_out
        self.latency_ms += r.latency_ms
        if r.cached:
            self.cache_hits += 1
        else:
            self.api_calls += 1
            self.billed_tokens_in += r.tokens_in
            self.billed_tokens_out += r.tokens_out
            self.billed_tokens_in_cached += r.tokens_in_cached
            self.billed_tokens_reasoning += r.tokens_reasoning
        slot = self.by_tag.setdefault(
            r.tag, {"calls": 0, "cache_hits": 0, "tokens_in": 0, "tokens_out": 0}
        )
        slot["calls"] += 1
        slot["cache_hits"] += int(r.cached)
        slot["tokens_in"] += r.tokens_in
        slot["tokens_out"] += r.tokens_out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLM:
    """config 주입형 LLM 클라이언트.

    transport 를 주면 그것을 쓰고, 없으면 OpenAI 를 지연 초기화한다.
    """

    def __init__(self, cfg, *, transport: Transport | None = None) -> None:
        self.cfg = cfg
        self.model: str = cfg.get("llm.model")
        self.default_max_tokens: int = int(cfg.get("llm.max_tokens"))
        self.cache_dir = Path(cfg.get("paths.cache_dir"))
        self.cache_enabled: bool = bool(cfg.get("llm.cache_enabled", True))
        self.max_retries: int = int(cfg.get("llm.max_retries", 4))
        self.timeout_s: float = float(cfg.get("llm.timeout_s", 120))
        self.temperature = cfg.get("llm.temperature", None)
        self.reasoning_effort = cfg.get("llm.reasoning_effort", None)
        self.max_tokens_param: str = cfg.get("llm.max_tokens_param", "auto")
        self.cache_policy: str = cfg.get("llm.cache_policy", "live") or "live"
        if self.cache_policy not in ("live", "cache_only"):
            raise LLMError("llm.cache_policy 는 live | cache_only: " + str(self.cache_policy))
        ro = cfg.get("paths.cache_dir_ro", None)
        self.cache_dir_ro = Path(ro) if ro else None     # 읽기만 한다. 새 응답은 cache_dir 에만 쓴다
        seed, family = cfg.get("run.seed", None), cfg.get("run.family", None)
        self.nonce_prefix = "" if seed is None else "s%s|%s|" % (seed, family or "")
        self.max_billed_in = cfg.get("budget.max_billed_in", None)
        self.max_billed_out = cfg.get("budget.max_billed_out", None)
        self.stats = UsageStats()
        self._stats_lock = threading.Lock()
        # branch 를 스레드로 병렬 실행하므로, 호출 회계는 스레드-로컬 스코프로 센다.
        # 전역 stats 를 앞뒤로 빼서 세면 스레드 간에 값이 섞인다.
        self._local = threading.local()
        self._transport = transport
        self._client = None
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # --- 캐시 ---------------------------------------------------------
    # --- 스레드-로컬 호출 회계 -------------------------------------------
    _SCOPE_KEYS = ("llm_calls", "tokens_in", "tokens_out", "latency_ms",
                   "api_calls", "billed_in", "billed_out", "billed_in_cached", "billed_reasoning")

    def begin_scope(self) -> None:
        """이 스레드에서 지금부터의 호출을 따로 센다."""
        self._local.scope = dict.fromkeys(self._SCOPE_KEYS, 0)
        self._local.scope["by_tag"] = {}

    def end_scope(self) -> dict[str, Any]:
        scope = getattr(self._local, "scope", None)
        self._local.scope = None
        if scope is None:
            scope = dict.fromkeys(self._SCOPE_KEYS, 0)
            scope["by_tag"] = {}
        return scope

    def _record_scope(self, res: "LLMResult") -> None:
        scope = getattr(self._local, "scope", None)
        if scope is None:
            return
        scope["llm_calls"] += 1
        scope["tokens_in"] += res.tokens_in
        scope["tokens_out"] += res.tokens_out
        scope["latency_ms"] += res.latency_ms
        if not res.cached:
            scope["api_calls"] += 1
            scope["billed_in"] += res.tokens_in
            scope["billed_out"] += res.tokens_out
            scope["billed_in_cached"] += res.tokens_in_cached
            scope["billed_reasoning"] += res.tokens_reasoning
        slot = scope["by_tag"].setdefault(res.tag, {"calls": 0, "api_calls": 0, "tokens_in": 0, "tokens_out": 0})
        slot["calls"] += 1
        slot["api_calls"] += int(not res.cached)
        slot["tokens_in"] += res.tokens_in
        slot["tokens_out"] += res.tokens_out

    @staticmethod
    def is_truncated(res: "LLMResult") -> bool:
        """토큰 상한에 걸려 잘린 응답. 추론 모델은 reasoning 이 예산을 먹어
        finish_reason='length' + 빈 본문이 나올 수 있다."""
        return res.finish_reason == "length"

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / (key + ".json")

    def _cache_read(self, key: str) -> dict[str, Any] | None:
        for root in (self.cache_dir, self.cache_dir_ro):
            if root is None:
                continue
            p = root / key[:2] / (key + ".json")
            if not p.exists():
                continue
            try:
                with io.open(p, encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue  # 손상된 캐시는 미스로 취급
        return None

    def effective_nonce(self, tag: str, cache_nonce: str) -> str:
        """캐시 키에 실제로 들어갈 nonce. seed 가 없거나 분해 호출이면 그대로다."""
        if not self.nonce_prefix or tag in UNSEEDED_TAGS:
            return cache_nonce
        return self.nonce_prefix + cache_nonce

    def _check_budget(self) -> None:
        s = self.stats
        if (self.max_billed_in is not None and s.billed_tokens_in > int(self.max_billed_in)) or \
                (self.max_billed_out is not None and s.billed_tokens_out > int(self.max_billed_out)):
            raise BudgetExceeded("청구 토큰 상한 초과: in %d / out %d" % (s.billed_tokens_in, s.billed_tokens_out))

    def _cache_write(self, key: str, payload: dict[str, Any]) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # 임시 파일 이름은 호출마다 달라야 한다. 고정 이름이면 같은 키를 동시에 쓰는
        # 두 스레드가 같은 tmp 를 잡고, Windows 에서 os.replace 가 PermissionError 로
        # 터진다. 병렬 branch 실행에서 실제로 발생했다.
        tmp = p.with_name(p.name + "." + uuid.uuid4().hex[:8] + ".tmp")
        try:
            with io.open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp, p)
        except OSError:
            # 다른 스레드가 같은 키를 먼저 썼다면 내용이 같으므로 버려도 된다.
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    # --- 호출 ---------------------------------------------------------
    def call(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int | None = None,
        tag: str,
        cache_nonce: str = "",
    ) -> LLMResult:
        mt = int(max_tokens if max_tokens is not None else self.default_max_tokens)
        nonce = self.effective_nonce(tag, cache_nonce)
        key = cache_key(
            model=self.model,
            system=system,
            prompt=prompt,
            max_tokens=mt,
            reasoning_effort=self.reasoning_effort or "",
            cache_nonce=nonce,
        )

        if self.cache_enabled:
            hit = self._cache_read(key)
            if hit is not None:
                # 토큰·지연은 저장값을 그대로 돌려준다 — 재실행 결과가 바이트 동일해야 한다.
                res = LLMResult(
                    text=hit["text"],
                    tokens_in=hit["tokens_in"],
                    tokens_out=hit["tokens_out"],
                    latency_ms=hit["latency_ms"],
                    model=self.model,
                    tag=tag,
                    cached=True,
                    finish_reason=hit.get("finish_reason", ""),
                    effective_nonce=nonce,
                    tokens_in_cached=int(hit.get("tokens_in_cached", 0)),
                    tokens_reasoning=int(hit.get("tokens_reasoning", 0)),
                    model_returned=hit.get("model_returned", ""),
                )
                with self._stats_lock:
                    self.stats.record(res)
                self._record_scope(res)
                return res

        if self.cache_policy == "cache_only":
            raise CacheMiss("캐시 미스 (tag=%s, nonce=%r) — cache_only 라 API 를 부르지 않는다" % (tag, nonce))
        with self._stats_lock:
            self._check_budget()

        t0 = time.perf_counter()
        text, tin, tout, finish, tin_cached, treason, model_ret = self._invoke(
            system=system, prompt=prompt, max_tokens=mt)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        res = LLMResult(
            text=text,
            tokens_in=tin,
            tokens_out=tout,
            latency_ms=latency_ms,
            model=self.model,
            tag=tag,
            cached=False,
            finish_reason=finish,
            effective_nonce=nonce,
            tokens_in_cached=tin_cached,
            tokens_reasoning=treason,
            model_returned=model_ret,
        )
        if self.cache_enabled:
            payload = {
                "text": text,
                "tokens_in": tin,
                "tokens_out": tout,
                "latency_ms": latency_ms,
                "finish_reason": finish,
                "model": self.model,
                "tag": tag,
                "cache_nonce": nonce,
            }
            # 새 필드는 값이 있을 때만 쓴다 — mock 이나 구형 transport 의 payload 는 예전과 같다.
            for k, v in (("tokens_in_cached", tin_cached), ("tokens_reasoning", treason),
                         ("model_returned", model_ret)):
                if v:
                    payload[k] = v
            self._cache_write(key, payload)
        with self._stats_lock:
            self.stats.record(res)
        self._record_scope(res)
        return res

    def _invoke(self, *, system: str, prompt: str, max_tokens: int):
        """(text, tokens_in, tokens_out, finish, tokens_in_cached, tokens_reasoning, model_returned).
        transport 가 예전 4-튜플을 돌려주면 나머지는 0/빈 값으로 채운다 (mock 호환)."""
        transport = self._transport or self._openai_transport
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                out = tuple(transport(
                    model=self.model,
                    system=system,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    reasoning_effort=self.reasoning_effort,
                ))
                return out + (0, 0, "")[len(out) - 4:] if len(out) < 7 else out[:7]
            except Exception as exc:  # noqa: BLE001 — 재시도 후 LLMError 로 감싼다
                if not self._retryable(exc):
                    raise LLMError("LLM 호출 실패 (재시도 불가): " + repr(exc)) from exc
                last = exc
                if attempt < self.max_retries - 1:
                    time.sleep(min(2**attempt, 30))
        raise LLMError(
            "LLM 호출 실패 (" + str(self.max_retries) + "회 재시도 소진): " + repr(last)
        ) from last

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        name = type(exc).__name__
        if name in {
            "RateLimitError",
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
        }:
            return True
        status = getattr(exc, "status_code", None)
        return status is not None and int(status) >= 500

    # --- OpenAI -------------------------------------------------------
    def _ensure_client(self):
        if self._client is None:
            provider = self.cfg.get("llm.provider", "openai")
            if provider != "openai":
                raise LLMError(
                    "허용되지 않은 provider: " + str(provider)
                    + ". CLAUDE.md 모델 정책상 OpenAI 만 쓴다"
                )
            if not os.environ.get("OPENAI_API_KEY"):
                raise LLMError("OPENAI_API_KEY 가 없다. .env 또는 환경변수에 설정하라")
            from openai import OpenAI  # 지연 import — 테스트는 API 없이 돈다

            self._client = OpenAI(timeout=self.timeout_s, max_retries=0)
        return self._client

    def _openai_transport(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        max_tokens: int,
        reasoning_effort: str | None = None,
    ) -> tuple[str, int, int, str]:
        client = self._ensure_client()
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        # 신형 모델은 max_tokens 대신 max_completion_tokens 를 받는다. auto 면 한 번 되짚는다.
        order = {
            "auto": ["max_completion_tokens", "max_tokens"],
            "max_tokens": ["max_tokens"],
            "max_completion_tokens": ["max_completion_tokens"],
        }[self.max_tokens_param]

        resp = None
        for param in order:
            try:
                resp = client.chat.completions.create(**kwargs, **{param: max_tokens})
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if param != order[-1] and (
                    "max_tokens" in msg or "max_completion_tokens" in msg
                ):
                    continue
                raise

        choice = resp.choices[0]
        usage = resp.usage
        pdet = getattr(usage, "prompt_tokens_details", None)
        cdet = getattr(usage, "completion_tokens_details", None)
        return (
            choice.message.content or "",
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
            str(choice.finish_reason or ""),
            int(getattr(pdet, "cached_tokens", 0) or 0),
            int(getattr(cdet, "reasoning_tokens", 0) or 0),
            str(getattr(resp, "model", "") or ""),
        )


def load_dotenv_if_present(path: str = ".env") -> bool:
    """.env 를 환경변수로 올린다. 값은 로그에 남기지 않는다."""
    p = Path(path)
    if not p.exists():
        return False
    with io.open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return True
