"""SPEC_M2 v1.1 — R_ij 재정의. **관련 pair 로 제한한다.** 생성 호출 0회.

v1.0 이 왜 실패했는지가 이 파일의 전제다. `contradiction` 을 전 pair 의 최댓값으로
잡았더니 **교란(다른 토픽) 쌍이 정상 쌍보다 더 자주 충돌로 잡혔다**(93.3% 대 91.2%).
max 를 걷어내고 pair 단위로 봐도 방향이 같았다(28.3% 대 16.3%). 팽창이 아니라 정의
문제였다 — MNLI 계열은 aboutness 를 모델링하지 않아 주어가 다른 두 사실을 곧잘
contradiction 으로 읽는다. **v1.0 의 contradiction 은 관련성의 역수였다.**

v1.1 의 변경 (SPEC_M2 v1.1):

  contradiction  **관련 pair 안에서만** max. 무관 pair 는 계산하지 않는다.
                 관련 pair 가 0 인 edge 는 "충돌 없음" 이 아니라 **"신호 없음"**.
  agreement      전 pair 상위 25% 평균 -> **관련 pair 평균**.
  redundancy     대칭 점수 + cos -> **비대칭 entailment recall**:
                 "j 가 i 의 claim 을 몇 % 나 덮는가". cos 조건은 뺀다(기록만).
                 PRUNE 은 j 가 i 를 80% 이상 덮고 j 가 sufficient 일 때 i 를 prune.

**교체는 한 번뿐이다. v1.2 는 없다.** v1.1 이 게이트에서 탈락하면 마스터플랜 §9-2 다.
그래서 이 파일에서 임계를 만지작거리지 않는다.

---

**스펙에 없는 단계를 하나 추가했다. 반드시 읽을 것.**

v1.1 은 "관련 pair = sub-question 매핑이 동일·인접인 claim 쌍" 이라 하고, 그 매핑이
complementarity 에 이미 쓰이고 있다고 본다. **그런데 그 매핑이 자료에 없다.**
`branch.py` 는 claim 을 **branch 답안 전체**에서 한 번에 분해하므로(`CLAIM_PROMPT` 의
`{answer}` 는 sub-answer 를 이어붙인 것) claim -> sub-question 링크가 존재하지 않는다.
v1.0 의 Relatedness 는 branch 사이의 sub-question 겹침이었지 claim 단위가 아니었다.

없는 매핑을 지어내는 대신 **결정론적 귀속**을 추가한다: claim 을 각 sub-answer 본문과
BM25 로 견주어 최고점 sub-question 에 귀속시킨다. 호출 0, 모델 0, 시드 없음.
이 프로젝트가 이미 같은 방식을 쓰고 있다(`d8a2_estimator_net.py`, `d14_spec_trunc.py`).

**이건 내 추가이지 스펙 문안이 아니다.** 결과와 함께 보고하고, 승인 없이 확정하지 않는다.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

__all__ = ["RelationV11", "ThresholdsV11", "attribute_claims", "related_pairs",
           "RelationExtractorV11"]

TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ThresholdsV11:
    tau_c: float = 0.7          # contradiction (VERIFY 트리거) — v1.0 과 같다
    tau_g: float = 0.8          # sufficiency
    adjacency: int = 1          # |q_i - q_j| <= adjacency 면 "인접"
    redundancy_cover: float = 0.8   # j 가 i 를 이만큼 덮으면 redundant
    tau_ent: float = 0.5        # claim 하나가 "덮였다"고 볼 entailment 문턱


@dataclass
class RelationV11:
    i: str
    j: str
    contradiction: float = 0.0
    agreement: float = 0.0
    cover_j_of_i: float = 0.0       # j 가 i 를 덮는 비율 (비대칭)
    cover_i_of_j: float = 0.0
    redundancy: int = 0
    argmax_pair: tuple[int, int] | None = None
    n_related: int = 0
    n_total: int = 0
    cos: float = 0.0                # 기록만 한다. 판정에 쓰지 않는다.
    has_signal: bool = False        # 관련 pair 가 0 이면 False = "신호 없음"

    def to_dict(self) -> dict:
        return {"i": self.i, "j": self.j,
                "contradiction": round(self.contradiction, 6),
                "agreement": round(self.agreement, 6),
                "cover_j_of_i": round(self.cover_j_of_i, 6),
                "cover_i_of_j": round(self.cover_i_of_j, 6),
                "redundancy": int(self.redundancy),
                "argmax_pair": list(self.argmax_pair) if self.argmax_pair else None,
                "n_related": self.n_related, "n_total": self.n_total,
                "cos": round(self.cos, 6), "has_signal": bool(self.has_signal)}


def _toks(s: str) -> list[str]:
    return TOKEN.findall(s.lower())


def attribute_claims(claims: Sequence[str], sub_answers: Sequence[str]) -> list[int]:
    """claim 을 sub-answer 에 BM25 로 귀속시킨다. 결정론, 호출 0.

    **스펙에 없는 추가 단계다** (모듈 docstring 참고). sub_answers 가 비면 전부 -1.
    """
    if not sub_answers:
        return [-1] * len(claims)
    docs = [_toks(x) for x in sub_answers]
    n = len(docs)
    avgdl = (sum(len(d) for d in docs) / n) or 1.0
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    out = []
    for c in claims:
        qs = _toks(c)
        best, best_s = -1, 0.0
        for k, d in enumerate(docs):
            tf = Counter(d)
            dl = len(d) or 1
            s = 0.0
            for w in qs:
                if w not in tf:
                    continue
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * dl / avgdl))
            if s > best_s:
                best, best_s = k, s
        out.append(best)
    return out


def related_pairs(attr_i: Sequence[int], attr_j: Sequence[int],
                  adjacency: int = 1) -> list[tuple[int, int]]:
    """sub-question 매핑이 동일·인접인 claim 쌍만. 귀속 실패(-1)는 제외한다."""
    out = []
    for a, qa in enumerate(attr_i):
        if qa < 0:
            continue
        for b, qb in enumerate(attr_j):
            if qb < 0:
                continue
            if abs(qa - qb) <= adjacency:
                out.append((a, b))
    return out


class RelationExtractorV11:
    """v1.1 계산기. `nli3` 는 (premise, hypothesis) -> (ent, con) 목록."""

    def __init__(self, nli3: Callable[[Sequence[tuple[str, str]]],
                                      Sequence[tuple[float, float]]],
                 th: ThresholdsV11 | None = None) -> None:
        self.nli3 = nli3
        self.th = th or ThresholdsV11()

    def relate(self, i: str, j: str,
               claims_i: Sequence[str], claims_j: Sequence[str],
               subq_i: Sequence[str], subq_j: Sequence[str]) -> RelationV11:
        th = self.th
        rel = RelationV11(i=i, j=j, n_total=len(claims_i) * len(claims_j))
        if not claims_i or not claims_j:
            return rel

        ai = attribute_claims(claims_i, subq_i)
        aj = attribute_claims(claims_j, subq_j)
        pairs = related_pairs(ai, aj, th.adjacency)
        rel.n_related = len(pairs)
        if not pairs:
            rel.has_signal = False          # "충돌 없음" 이 아니라 "신호 없음"
            return rel
        rel.has_signal = True

        fwd = [(claims_i[a], claims_j[b]) for a, b in pairs]
        rev = [(claims_j[b], claims_i[a]) for a, b in pairs]
        f = list(self.nli3(fwd))
        r = list(self.nli3(rev))

        ents, cons = [], []
        for k in range(len(pairs)):
            ents.append((float(f[k][0]) + float(r[k][0])) / 2.0)
            cons.append(max(float(f[k][1]), float(r[k][1])))

        best = max(range(len(cons)), key=lambda k: cons[k])
        rel.contradiction = cons[best]
        rel.argmax_pair = pairs[best]
        rel.agreement = sum(ents) / len(ents)          # 관련 pair **평균**

        # 비대칭 커버리지 — "j 가 i 의 claim 을 몇 % 나 덮는가"
        cov_i: dict[int, bool] = {}
        cov_j: dict[int, bool] = {}
        for k, (a, b) in enumerate(pairs):
            if float(f[k][0]) >= th.tau_ent:       # i 의 claim a 를 j 의 b 가 함의
                cov_i[a] = True
            if float(r[k][0]) >= th.tau_ent:
                cov_j[b] = True
        rel.cover_j_of_i = len(cov_i) / len(claims_i)
        rel.cover_i_of_j = len(cov_j) / len(claims_j)
        rel.redundancy = int(max(rel.cover_j_of_i, rel.cover_i_of_j) >= th.redundancy_cover)
        return rel
