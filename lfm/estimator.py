"""충분성 연산자 g — SPEC_M1prime §1 의 "바꾸는 것 하나".

`evidence.grounding()` 이 문장을 근거에 대고 판정하는 방식을 여기로 뺀다.
verify 규칙(두 절, ≥)·sufficient(>)·τ·라운드 수는 **건드리지 않는다.**

두 구현:

  `nli_max`        현행 g₁. 문장 × 인용 청크 각각을 NLI 에 넣고 **최댓값 ≥ τ**.
                   Test A·D1~D14 가 이걸로 측정됐다. **기본값이며 바뀌면 안 된다.**
  `minicheck_doc`  V′ 에서 사람 라벨 60건으로 고른 연산자. 인용 청크를 이어붙인
                   **단일 문서**를 전제로 MiniCheck 에 넣고 **> 0.5** (엄격).

왜 최댓값을 버리는가는 논문 §VI 전체다 — 최댓값 집계는 전제 풀이 커질수록 오르고
(그래서 풀 범위를 안 적으면 비교가 깨진다), 풀 안의 무관 내용 비율에 눈이 먼다.
단일 문서 전제에는 집계 자유도가 없다.

**assert 를 건다.** MiniCheck scorer 의 내부 청크 크기 기본값이 입력 상한보다 작으면
긴 문서가 조용히 둘로 쪼개지고 max 가 뒷문으로 다시 들어온다. 매 호출마다 내부 청크
수가 1 인지 확인한다 — 이 논문의 주장이 "집계 자유도를 없앴다" 이므로, 그게 깨지면
소리 없이 틀리는 게 아니라 죽어야 한다.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["Estimator", "NLIMaxEstimator", "MiniCheckDocEstimator", "load_estimator"]

DEFAULT_KIND = "nli_max"


@runtime_checkable
class Estimator(Protocol):
    """문장 하나가 전제들에 의해 지지되는가. 판정만 돌려준다."""

    name: str

    def grounded(self, sentence: str, premises: list[str]) -> bool:
        ...


class NLIMaxEstimator:
    """현행 g₁ — 문장 × 인용 청크 최댓값 ≥ τ. **기존 동작 그대로다.**"""

    name = "nli_max"

    def __init__(self, nli, tau_ground: float = 0.5):
        self.nli = nli
        self.tau = float(tau_ground)

    def grounded(self, sentence: str, premises: list[str]) -> bool:
        if not premises:
            return False                      # 인용 없는 문장은 근거가 없는 것이다
        scores = self.nli.entailment([(p, sentence) for p in premises])
        return max(scores) >= self.tau        # 비엄격 — 기존 코드와 같다

    def grounded_many(self, items: list[tuple[str, list[str]]]) -> list[bool]:
        """문장을 한 번에 넣어 배치 이득을 살린다. 결과는 grounded() 와 같아야 한다."""
        pairs: list[tuple[str, str]] = []
        spans: list[tuple[int, int]] = []
        for sent, prem in items:
            start = len(pairs)
            pairs.extend((p, sent) for p in prem)
            spans.append((start, len(pairs)))
        scores = self.nli.entailment(pairs) if pairs else []
        return [end > start and max(scores[start:end]) >= self.tau
                for start, end in spans]


class MiniCheckDocEstimator:
    """확정 연산자 — 인용 청크를 이어붙인 단일 문서 전제, **> 0.5 엄격**.

    `scorer` 는 `score(doc, claim) -> float` 를 갖는 것이면 된다
    (`experiments/d9_minicheck.Scorer`). 주입식이라 이 모듈은 MiniCheck 를
    import 하지 않는다 — 테스트에서 가짜 scorer 로 갈아 끼울 수 있다.
    """

    name = "minicheck_doc"
    SEP = "\n\n"

    def __init__(self, scorer, tau: float = 0.5, *, check_chunks: bool = True):
        self.scorer = scorer
        self.tau = float(tau)
        self.check_chunks = check_chunks

    def _assert_single_chunk(self, doc: str) -> None:
        """scorer 가 전제를 내부에서 쪼갰으면 max 가 되돌아온 것이다. 죽인다."""
        n = getattr(self.scorer, "last_n_chunks", None)
        if n is None or not self.check_chunks:
            return
        if int(n) != 1:
            raise AssertionError(
                "단일 문서 전제가 내부에서 %d 조각으로 쪼개졌다 — 최댓값 집계가 "
                "되돌아왔다. scorer 의 max_model_len 을 올려라. 전제 %d자."
                % (int(n), len(doc)))

    def grounded(self, sentence: str, premises: list[str]) -> bool:
        if not premises:
            return False
        doc = self.SEP.join(premises)
        p = float(self.scorer.score(doc, sentence))
        self._assert_single_chunk(doc)
        return p > self.tau                   # **엄격 부등호** — 스펙 문구 그대로

    def grounded_many(self, items: list[tuple[str, list[str]]]) -> list[bool]:
        return [self.grounded(s, p) for s, p in items]


def load_estimator(cfg, nli=None, scorer=None) -> Estimator:
    """설정에서 연산자를 만든다.

    `estimator.kind` 가 없으면 **`nli_max`** 다. 기본값을 바꾸지 않는 것이 중요하다 —
    Test A 부터 D14 까지 전부 그것으로 측정됐고, 기본값이 바뀌면 재현이 조용히 깨진다.
    M1′ 는 설정에서 명시적으로 `minicheck_doc` 을 켠다.
    """
    kind = str(cfg.get("estimator.kind", DEFAULT_KIND)) if cfg is not None else DEFAULT_KIND
    if kind == "nli_max":
        if nli is None:
            raise ValueError("nli_max 는 NLI 모델이 있어야 한다")
        return NLIMaxEstimator(nli, float(cfg.get("nli.tau_ground", 0.5)) if cfg else 0.5)
    if kind == "minicheck_doc":
        if scorer is None:
            raise ValueError(
                "minicheck_doc 은 scorer 가 있어야 한다. "
                "experiments/d9_minicheck.Scorer() 를 만들어 넘겨라.")
        return MiniCheckDocEstimator(scorer, float(cfg.get("estimator.tau", 0.5)) if cfg else 0.5)
    raise ValueError("모르는 estimator.kind: %r (nli_max | minicheck_doc)" % kind)
