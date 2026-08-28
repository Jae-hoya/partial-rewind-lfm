"""LLM 의 구조화 출력 파싱.

캐시와 재시도가 부딪히는 지점이다. 응답이 깨졌다고 **같은 프롬프트로** 다시 부르면
디스크 캐시가 같은 불량 응답을 영원히 돌려준다. 복구 호출은 반드시 프롬프트를 바꿔
(불량 텍스트를 본문에 실어) 새 캐시 키를 만든다.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["strip_fences", "loads_lenient", "parse_structured"]

_FENCE_OPEN = re.compile(r"^```[a-zA-Z0-9_-]*\s*")
_FENCE_CLOSE = re.compile(r"```\s*$")


def strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = _FENCE_OPEN.sub("", t)
        t = _FENCE_CLOSE.sub("", t)
    return t.strip()


def loads_lenient(text: str, expect: type = list) -> Any | None:
    """관대한 파싱. 코드펜스 제거 -> 통짜 파싱 -> 첫 괄호 블록 추출 순."""
    opener, closer = ("[", "]") if expect is list else ("{", "}")
    for candidate in (strip_fences(text), text or ""):
        if not candidate:
            continue
        try:
            v = json.loads(candidate)
            if isinstance(v, expect):
                return v
        except json.JSONDecodeError:
            pass
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                v = json.loads(candidate[start : end + 1])
                if isinstance(v, expect):
                    return v
            except json.JSONDecodeError:
                pass
    return None


REPAIR_SYSTEM = "You fix malformed JSON. Reply with JSON only, no prose, no code fences."

REPAIR_PROMPT = """The text below was supposed to be a JSON {kind} but is malformed or wrapped
in prose. Return the corrected JSON {kind} only.

--- BEGIN ---
{body}
--- END ---"""


def parse_structured(
    llm,
    text: str,
    *,
    expect: type = list,
    tag: str,
    cache_nonce: str = "",
    max_repair: int = 1,
) -> Any | None:
    """관대 파싱 후 실패 시 복구 호출. 복구 프롬프트는 원본과 달라 캐시 키가 갈린다."""
    v = loads_lenient(text, expect)
    if v is not None:
        return v
    if not (text or "").strip():
        # 빈 응답은 고칠 것이 없다. 복구 호출은 토큰만 태운다.
        return None

    body = (text or "")[:4000]
    for attempt in range(max_repair):
        prompt = REPAIR_PROMPT.format(
            kind="array" if expect is list else "object", body=body
        )
        res = llm.call(
            prompt,
            system=REPAIR_SYSTEM,
            tag=tag + "_repair",
            cache_nonce=cache_nonce + ("" if attempt == 0 else ":a" + str(attempt)),
        )
        v = loads_lenient(res.text, expect)
        if v is not None:
            return v
        body = res.text[:4000]
    return None
