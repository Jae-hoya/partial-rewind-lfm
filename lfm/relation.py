"""SPEC_M2 §2 — Relation State R_ij ∈ [0,1]^4. **생성 호출 0회.**

LFM 의 신규성은 여기 있다. 관점별 evidence 를 따로 재는 것이 아니라, **관점 사이의
관계**를 신호로 써서 고칠 곳을 찾는다. 그 관계가 R_ij 네 개다.

  agreement_ij       상위 q=25% pair 의 양방향 entailment 평균
  contradiction_ij   전 pair 의 양방향 contradiction **최댓값** + argmax pair
  complementarity_ij Relatedness x NonRedundancy x CoverageGain
  redundancy_ij      1[agreement > tau_a] x 1[cos(answer_i, answer_j) > tau_r]

**왜 상위 25% 평균인가.** claim 은 branch 당 10여 개이고 pair 대부분은 서로 무관해
0 근처다. 전체 평균을 쓰면 그 0 들이 신호를 덮는다. q 는 **고정이고 스윕 금지**다
(SPEC_M2 §2). 결과를 보고 q 를 고르면 그 순간 이 프로젝트가 D7 에서 저지른 실수를
반복한다.

**왜 contradiction 은 최댓값인가.** VERIFY 트리거라 보수적이어야 한다. 하나라도
정면으로 부딪히면 확인하러 간다. 대신 최댓값은 pair 수가 많을수록 위로 치우치므로
**교란 음성 통제가 필수다** — 이 모듈은 그걸 `negative_control()` 로 같이 제공한다.
D8-A 가 밝힌 팽창을 여기서 또 겪지 않기 위해서다.

**체크포인트.** V′ 로 MiniCheck 가 확정되면 relation 은 `bart-large-mnli` 를 쓴다.
MiniCheck 는 문서 단위 검증기라 3-way(ent/neu/con) 출력이 없어 contradiction 을
못 낸다. 이 대체는 스펙에 고정돼 있고 논문에 명시한다.

claim-claim 은 짧은 전제라 D7-0/D13 이 잡아낸 길이 희석 문제가 작다 — 그것이
관계를 claim 단위로 재는 또 하나의 이유다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

__all__ = ["Thresholds", "PairScore", "Relation", "RelationExtractor"]

# SPEC_M2 §2 — 기본 임계. **고정.** 부록 민감도 분석 외에는 건드리지 않는다.
TAU_A = 0.6      # agreement
TAU_R = 0.85     # redundancy 의 cos 문턱
TAU_C = 0.7      # contradiction (VERIFY 트리거)
TAU_G = 0.8      # sufficiency
Q_TOP = 0.25     # agreement 상위 비율


@dataclass(frozen=True)
class Thresholds:
    tau_a: float = TAU_A
    tau_r: float = TAU_R
    tau_c: float = TAU_C
    tau_g: float = TAU_G
    q_top: float = Q_TOP


@dataclass(frozen=True)
class PairScore:
    """claim 쌍 하나의 양방향 3-way 결과."""

    i_idx: int
    j_idx: int
    ent: float          # (p_ent(a->b) + p_ent(b->a)) / 2
    con: float          # max(p_con(a->b), p_con(b->a))


@dataclass
class Relation:
    """branch 쌍 하나의 R_ij."""

    i: str
    j: str
    agreement: float = 0.0
    contradiction: float = 0.0
    complementarity: float = 0.0
    redundancy: int = 0
    argmax_pair: tuple[int, int] | None = None   # contradiction 최댓값을 낸 claim 쌍
    n_pairs: int = 0
    cos: float = 0.0

    def to_dict(self) -> dict:
        return {"i": self.i, "j": self.j,
                "agreement": round(self.agreement, 6),
                "contradiction": round(self.contradiction, 6),
                "complementarity": round(self.complementarity, 6),
                "redundancy": int(self.redundancy),
                "argmax_pair": list(self.argmax_pair) if self.argmax_pair else None,
                "n_pairs": self.n_pairs, "cos": round(self.cos, 6)}


def _top_q_mean(values: Sequence[float], q: float) -> float:
    """상위 q 비율의 평균. 값이 없으면 0. 최소 1개는 본다."""
    if not values:
        return 0.0
    k = max(1, int(math.ceil(len(values) * q)))
    return sum(sorted(values, reverse=True)[:k]) / k


class RelationExtractor:
    """R_ij 계산기.

    `nli3` 는 (premise, hypothesis) 목록을 받아 (ent, con) 목록을 돌려주는 호출자다.
    3-way 출력이 필요하므로 MiniCheck 는 여기 못 쓴다 (모듈 docstring 참고).

    `embed` 는 텍스트 목록 -> 벡터 목록. redundancy 의 cos 에만 쓴다.
    둘 다 주입식이라 테스트에서 모델 없이 돌릴 수 있다.
    """

    def __init__(self, nli3: Callable[[Sequence[tuple[str, str]]], Sequence[tuple[float, float]]],
                 embed: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
                 th: Thresholds | None = None) -> None:
        self.nli3 = nli3
        self.embed = embed
        self.th = th or Thresholds()

    # --- pair 단위 -----------------------------------------------------
    def pair_scores(self, claims_i: Sequence[str], claims_j: Sequence[str]) -> list[PairScore]:
        """양방향 1패스. 모든 (a, b) 에 p(a->b), p(b->a)."""
        if not claims_i or not claims_j:
            return []
        fwd, rev, idx = [], [], []
        for a, ca in enumerate(claims_i):
            for b, cb in enumerate(claims_j):
                fwd.append((ca, cb))
                rev.append((cb, ca))
                idx.append((a, b))
        f = list(self.nli3(fwd))
        r = list(self.nli3(rev))
        out = []
        for k, (a, b) in enumerate(idx):
            ent = (float(f[k][0]) + float(r[k][0])) / 2.0
            con = max(float(f[k][1]), float(r[k][1]))
            out.append(PairScore(a, b, ent, con))
        return out

    # --- branch 단위 집계 ----------------------------------------------
    def aggregate(self, i: str, j: str, pairs: Sequence[PairScore]) -> Relation:
        rel = Relation(i=i, j=j, n_pairs=len(pairs))
        if not pairs:
            return rel
        rel.agreement = _top_q_mean([p.ent for p in pairs], self.th.q_top)
        best = max(pairs, key=lambda p: p.con)
        rel.contradiction = best.con
        rel.argmax_pair = (best.i_idx, best.j_idx)
        return rel

    def finish(self, rel: Relation, *, answer_i: str = "", answer_j: str = "",
               relatedness: float = 0.0, coverage_gain: float = 0.0) -> Relation:
        """redundancy 와 complementarity 를 채운다.

        relatedness 와 coverage_gain 은 호출자가 sub-question 매핑·인용에서 계산해
        넘긴다. 둘 다 **무호출**이다 (SPEC_M2 §2).
        """
        cos = 0.0
        if self.embed is not None and answer_i and answer_j:
            va, vb = list(self.embed([answer_i, answer_j]))
            na = math.sqrt(sum(x * x for x in va)) or 1.0
            nb = math.sqrt(sum(x * x for x in vb)) or 1.0
            cos = sum(x * y for x, y in zip(va, vb)) / (na * nb)
        rel.cos = cos
        rel.redundancy = int(rel.agreement > self.th.tau_a and cos > self.th.tau_r)
        rel.complementarity = (max(0.0, relatedness)
                               * (1.0 - rel.redundancy)
                               * max(0.0, coverage_gain))
        return rel

    def relate(self, i: str, j: str, claims_i: Sequence[str], claims_j: Sequence[str],
               **kw) -> Relation:
        return self.finish(self.aggregate(i, j, self.pair_scores(claims_i, claims_j)), **kw)

    # --- 교란 음성 통제 (필수) -------------------------------------------
    def negative_control(self, claims_by_topic: dict[str, list[list[str]]],
                         shift: int = 7) -> dict:
        """다른 토픽 branch 쌍에 같은 계산. SPEC_M2 §2.

        보고: 교란 쌍에서 contradiction > tau_c 비율, agreement > tau_a 비율.
        **둘 다 <= 5% 가 아니면 tau 를 올리는 게 아니라 그 사실을 보고한다.**
        임계 조정은 Test C 이후 부록 스윕에서만 (SPEC_M2 §2).
        """
        tids = sorted(claims_by_topic)
        if len(tids) < 2:
            return {"n": 0, "note": "토픽이 2개 미만이라 교란 불가"}
        mapping = {tids[k]: tids[(k + shift) % len(tids)] for k in range(len(tids))}
        assert all(a != b for a, b in mapping.items()), "같은 토픽 배정이 있다"

        con_hits = agr_hits = n = 0
        for src, dst in mapping.items():
            for ci in claims_by_topic[src]:
                for cj in claims_by_topic[dst]:
                    if not ci or not cj:
                        continue
                    rel = self.aggregate(src, dst, self.pair_scores(ci, cj))
                    n += 1
                    con_hits += rel.contradiction > self.th.tau_c
                    agr_hits += rel.agreement > self.th.tau_a
        if not n:
            return {"n": 0, "note": "교란 쌍이 없다"}
        cr, ar = 100.0 * con_hits / n, 100.0 * agr_hits / n
        return {"n": n, "shift": shift,
                "contradiction_gt_tau_c_pct": round(cr, 4),
                "agreement_gt_tau_a_pct": round(ar, 4),
                "pass": bool(cr <= 5.0 and ar <= 5.0),
                "note": ("둘 다 <=5% — 통과" if cr <= 5.0 and ar <= 5.0 else
                         "5% 초과. tau 를 올리지 말고 이 사실을 Test C-1 해석에 반영한다")}
