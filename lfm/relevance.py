"""SPEC_R2 연산자 A·B — 질문-답 관련성, 질문-컨텍스트 답 가능성. 로컬 모델만 쓴다 (API 0).

grounding(문장이 근거에 entail 되는가)과 직교한다. A 는 근거를 보지 않고 (질문, 답) 만 보고,
B 는 생성기 출력을 보지 않고 (질문, 청크) 만 본다. 정의와 선택 규칙은 results/d18/PREREG_D18_AMEND1.md.
"""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .types import CITATION_RE

__all__ = ["strip_ids", "RerankerScorer", "NLITemplateScorer", "NLI_TEMPLATE"]

NLI_TEMPLATE = 'This text answers the question: "{q}"'


def strip_ids(text: str) -> str:
    return re.sub(r"\s+", " ", CITATION_RE.sub("", text or "")).replace(" .", ".").strip()


class _CrossEncoder:
    def __init__(self, model_name: str, max_length: int, batch: int = 16):
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.batch = batch

    @torch.inference_mode()
    def _logits(self, pairs: list[tuple[str, str]]) -> torch.Tensor:
        outs = []
        for i in range(0, len(pairs), self.batch):
            a, b = zip(*pairs[i:i + self.batch])
            enc = self.tok(list(a), list(b), padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            outs.append(self.model(**enc).logits.float().cpu())
        return torch.cat(outs) if outs else torch.empty(0)


class RerankerScorer(_CrossEncoder):
    """bge-reranker: (질문, 텍스트) -> sigmoid(logit)."""

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        lg = self._logits(pairs)
        return torch.sigmoid(lg.view(-1)).tolist() if len(pairs) else []


class NLITemplateScorer(_CrossEncoder):
    """NLI cross-encoder: premise = 텍스트, hypothesis = 질문을 담은 평서문 템플릿 -> entailment 확률."""

    def __init__(self, model_name: str, max_length: int = 512, batch: int = 16):
        super().__init__(model_name, max_length, batch)
        lab = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.ent = lab["entailment"]

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        prem_hyp = [(text, NLI_TEMPLATE.format(q=q)) for q, text in pairs]
        lg = self._logits(prem_hyp)
        return torch.softmax(lg, dim=-1)[:, self.ent].tolist() if len(pairs) else []
