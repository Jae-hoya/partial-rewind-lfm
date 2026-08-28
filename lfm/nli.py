"""NLI 판정기. evidence.py 와 verifier.py 의 유일한 의미 판정 수단이다.

CLAUDE.md 최종 확정 스펙: **전용 NLI 모델. LLM-as-judge 금지.**

라벨 순서를 상수로 박지 않는다. `cross-encoder/nli-deberta-v3-base` 는
{0: contradiction, 1: entailment, 2: neutral} 이지만 모델마다 다르고, 순서를 틀리면
grounding 이 조용히 뒤집힌 채로 실험이 "정상 동작"한다. `config.id2label` 을 읽는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = ["NLIScores", "NLI", "MockNLI", "load_nli"]


@dataclass(frozen=True)
class NLIScores:
    entailment: float
    contradiction: float
    neutral: float


class NLIBase:
    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        raise NotImplementedError

    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [s.entailment for s in self.score(pairs)]

    def max_entailment(self, premises: Iterable[str], hypothesis: str) -> float:
        """가설 하나를 여러 전제에 대해 재고 최댓값을 돌려준다. 전제가 없으면 0.0."""
        prem = list(premises)
        if not prem:
            return 0.0
        return max(self.entailment([(p, hypothesis) for p in prem]))


class NLI(NLIBase):
    """transformers 기반. 배치 추론, eval 모드, grad 없음."""

    def __init__(self, cfg) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self.model_name: str = cfg.get("nli.model")
        self.batch_size: int = int(cfg.get("nli.batch_size", 16))
        self.max_length: int = int(cfg.get("nli.max_length", 512))

        want = str(cfg.get("nli.device", "auto"))
        if want == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif want == "cuda" and not torch.cuda.is_available():
            device = "cpu"  # GPU 없으면 강제 강등 (SPEC §1)
        else:
            device = want
        self.device = device
        self._torch = torch

        conf = AutoConfig.from_pretrained(self.model_name)
        id2label = {int(k): str(v).lower() for k, v in conf.id2label.items()}
        missing = {"entailment", "contradiction", "neutral"} - set(id2label.values())
        if missing:
            raise ValueError(
                "NLI 모델 라벨에 " + repr(sorted(missing)) + " 이 없다: " + repr(conf.id2label)
            )
        self.label_index = {v: k for k, v in id2label.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()
        self.n_pairs = 0

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        if not pairs:
            return []
        torch = self._torch
        out: list[NLIScores] = []
        ie = self.label_index["entailment"]
        ic = self.label_index["contradiction"]
        inu = self.label_index["neutral"]
        with torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                enc = self.tokenizer(
                    [p for p, _ in batch],
                    [h for _, h in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu()
                for row in probs:
                    out.append(
                        NLIScores(
                            entailment=float(row[ie]),
                            contradiction=float(row[ic]),
                            neutral=float(row[inu]),
                        )
                    )
        self.n_pairs += len(pairs)
        return out


class MockNLI(NLIBase):
    """테스트·스모크용. 어휘 겹침으로 entailment 를 흉내 낸다 — 결정론적."""

    def __init__(self, fixed: float | None = None) -> None:
        self.fixed = fixed
        self.n_pairs = 0

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        self.n_pairs += len(pairs)
        out = []
        for premise, hypothesis in pairs:
            if self.fixed is not None:
                e = self.fixed
            else:
                pw = set(premise.lower().split())
                hw = set(hypothesis.lower().split())
                e = len(pw & hw) / len(hw) if hw else 0.0
            e = max(0.0, min(1.0, e))
            out.append(NLIScores(entailment=e, contradiction=0.0, neutral=1.0 - e))
        return out


def load_nli(cfg, *, mock: bool = False) -> NLIBase:
    return MockNLI() if mock else NLI(cfg)
