"""SPEC §1 — 착수 시 `config.llm.model` 이 실제 API 에서 호출 가능한지 1콜로 검증한다.

실패하면 즉시 중단·보고한다. 모델 ID 가 404 면 다른 모델로 말없이 대체하지 않는다.

캐시를 끄고 호출한다. 캐시 히트로 통과하면 검증이 아니다.
결과는 `results/model_check.json` 에 남긴다 — 어느 날 어떤 모델·키로 무엇을 확인했는지가
비용 귀속의 근거가 된다.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, LLMError, load_dotenv_if_present  # noqa: E402

PROMPT = "Reply with exactly one word: ok"


def key_fingerprint() -> dict[str, object]:
    """키 값은 절대 기록하지 않는다. 길이와 해시 앞 12자리만 남긴다."""
    k = os.environ.get("OPENAI_API_KEY", "")
    if not k:
        return {"present": False}
    return {
        "present": True,
        "length": len(k),
        "sha256_12": hashlib.sha256(k.encode()).hexdigest()[:12],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="모델 호출 가능성 검증 (SPEC §1)")
    ap.add_argument("--config", default=None)
    # 추론 모델은 reasoning 토큰을 이 예산에서 먼저 쓴다. 16 이면 reasoning 만으로
    # 소진돼 finish_reason="length" + 빈 본문이 나오고, 검증이 통과처럼 보인다.
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--note", default="", help="results 에 남길 메모 (예: 용도)")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(str(repo_root / ".env"))

    cfg = load_config(args.config or (repo_root / "config.yaml"))
    data = cfg.as_dict()
    data["llm"]["cache_enabled"] = False  # 캐시 히트로 통과하면 검증이 아니다
    llm = LLM(Config(data, source="model_check"))

    record: dict[str, object] = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": args.note or "LFM M1 Test A",
        "model": llm.model,
        "provider": cfg.get("llm.provider", "openai"),
        "api_key": key_fingerprint(),
    }

    print("model   = " + llm.model)
    print("purpose = " + str(record["purpose"]))
    print("api_key = " + json.dumps(record["api_key"]))
    print("호출 중 ...", flush=True)

    try:
        res = llm.call(PROMPT, max_tokens=args.max_tokens, tag="model_check")
    except LLMError as exc:
        record["ok"] = False
        record["error"] = str(exc)
        _write(cfg, record)
        print()
        print("FAIL — " + str(exc))
        print("모델 ID 가 거부되면 다른 모델로 대체하지 말고 보고한다 (SPEC §1).")
        return 1

    truncated = LLM.is_truncated(res)
    record.update(
        {
            "ok": bool(res.text.strip()) and not truncated,
            "truncated": truncated,
            "tokens_in": res.tokens_in,
            "tokens_out": res.tokens_out,
            "latency_ms": res.latency_ms,
            "finish_reason": res.finish_reason,
            "response_preview": res.text[:120],
        }
    )
    _write(cfg, record)

    print()
    if not record["ok"]:
        print("FAIL — 호출은 됐으나 가시 응답이 비었다 (finish_reason=%s)." % res.finish_reason)
        print("max_tokens 가 reasoning 토큰에 전부 먹혔다. 예산을 올리거나 "
              "llm.reasoning_effort 를 낮춰라.")
        return 1
    print("OK — 응답 %r" % res.text[:60])
    print("tokens in/out = %d/%d, latency %d ms" % (res.tokens_in, res.tokens_out, res.latency_ms))
    return 0


def _write(cfg, record: dict[str, object]) -> None:
    out = Path(cfg["paths.results_dir"]) / "model_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    if out.exists():
        try:
            history = json.load(io.open(out, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            history = []
    history.append(record)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)
    print("기록 -> " + str(out))


if __name__ == "__main__":
    raise SystemExit(main())
