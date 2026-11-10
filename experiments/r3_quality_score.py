"""PREREG_R3 §3 — 블라인드 답 품질(C·K) 채점: S/H/G/F/D/ΔH, 토픽 군집 부트스트랩, κ_w, 읽기 점검 관문, 판정표 → 원고 문구.
**LLM 0, GPU 0.** C0 에 커밋하고 고치지 않고 돌린다.

  python experiments/r3_quality_score.py                       # results/r3/labels/labels_{A,B}.csv + sealed/_key_q.csv(+_meta_q.json) → quality/RESULT.json
  python experiments/r3_quality_score.py --selftest [--B 400]  # 합성 키 + 합성 라벨 여러 벌(관문 실패·통과 등)로 끝까지. 값은 버린다

정의는 PREREG_R3 §3.4 (S, H), §3.6 (점검), §3.7 (분석), §3.8 (문구표).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "results/r3"
ARMS = ("E-SQ", "E-GR", "E-R1")
COMPS = {"E-SQ": "E-SQ − E-GR", "E-R1": "E-R1 − E-GR"}
SEED_BOOT = 4410
SEED_KW = 4411          # §3.7·§8: κ_w(S) 의 CI (같은 두 단 방식, B = 10,000)
KW_BAR = 0.40
HARM_MARGIN = 0.10
N_SLOTS_ALL, N_UE = 1752, 357
TR_FULL = "근거 전체"


def rd(p: Path) -> list[dict]:
    return list(csv.DictReader(io.open(p, encoding="utf-8-sig")))


def c2(c: str) -> str:
    return "C2" if c in ("C2a", "C2b") else c


def score(C: str, K: str) -> int:
    """§3.4 슬롯 점수(답함). 기권은 호출하지 않는다."""
    c = c2(C)
    if c == "C0":
        return 0
    if c == "C1":
        return 1
    return {"K0": 2, "K1": 3, "K2": 4}[K]


# --------------------------------------------------------------------------- 자료
class Data:
    def __init__(self, keyrows: list[dict], labels: dict[str, dict[str, dict]]):
        self.labels = labels
        main = [r for r in keyrows if r["stage"] == "main" and str(r.get("catch", "0")) in ("0", "")]
        self.catch = [r for r in keyrows if str(r.get("catch", "0")) == "1"]
        self.calib = [r for r in keyrows if r["stage"] == "calib"]
        slots = defaultdict(dict)
        meta = {}
        for r in main:
            sl = (str(r["seed"]), r["query_id"], r["branch"], str(r["sq_index"]))
            slots[sl][r["arm"]] = r
            # §3.3 두 방향 층: stratum = 층 id(g1|토픽, g0|토픽, g0|small), topic = 군집(부트스트랩 1단)
            meta[sl] = {"topic": r.get("topic") or r["query_id"], "cell": r["stratum"], "N_h": int(r["N_h"]), "n_h": int(r["n_h"])}
        self.slots = sorted(slots)
        assert all(set(slots[s]) == set(ARMS) for s in self.slots), "슬롯마다 세 arm 행"
        self.rows = slots
        self.meta = meta
        self.topic = [meta[s]["topic"] for s in self.slots]
        self.cell = [meta[s]["cell"] for s in self.slots]
        self.w = np.array([meta[s]["N_h"] / meta[s]["n_h"] for s in self.slots], float)
        # 층마다 Σw = N_h (표본 슬롯 수 = n_h 확인)
        cnt = Counter(self.cell)
        assert all(cnt[c] == meta[s]["n_h"] for s in self.slots for c in [meta[s]["cell"]]), "층 n_h 불일치"
        self.items = sorted({r["item_id"] for r in main if r["item_id"]})
        self.n_qs = None        # 전체 질의·seed 수(_meta_q.json n_query_seeds_all), analyze 가 넣는다

    def judge_ok(self, j: str) -> bool:
        L = self.labels.get(j, {})
        return bool(L) and all(L.get(i, {}).get("C") and L.get(i, {}).get("K") for i in self.items)

    def S(self, j: str, abstain: int = 2, label_of=None, excl_items: set | None = None):
        """arm 별 (S, H, answered) 배열. label_of(item_id) → (C, K) 대체(기계적 합의 판)."""
        out = {}
        for a in ARMS:
            S_, H_, A_ = [], [], []
            for sl in self.slots:
                r = self.rows[sl][a]
                if str(r["answered"]) == "1":
                    C, K = label_of(r["item_id"]) if label_of else (self.labels[j][r["item_id"]]["C"], self.labels[j][r["item_id"]]["K"])
                    s = score(C, K) if not isinstance(C, (int, float)) else C
                    S_.append(s); H_.append(int(s <= 1)); A_.append(1)
                else:
                    S_.append(abstain); H_.append(0); A_.append(0)
            out[a] = (np.array(S_, float), np.array(H_, float), np.array(A_, int))
        return out


def hajek(x, w, idx=None):
    if idx is not None:
        x, w = x[idx], w[idx]
    return float((w * x).sum() / w.sum())


def stats_of(arr, w, idx=None, qscale: float | None = None, s_abst: float = 2) -> dict:
    """qscale = N_UE / (전체 질의·seed 수): 질의 수준 'S = 4 sub-answer 수 / 질의·seed' 의 Hájek 합 추정(보조 8).
    s_abst = 기권 점수(§3.4·§3.7): G·F 의 기준 arm A0(전부 기권)의 슬롯 점수다. 1차 2, 보조 2(민감도)는 arr 를 만든
    기권 점수와 같은 값(1 또는 3)을 넘긴다 — G = E_w[S_GR − s_abst], F = E_w[(S_a − s_abst) − 0.5 (S_GR − s_abst)].
    D·ΔH·B4·Q4 는 s_abst 와 무관하다."""
    S_gr = arr["E-GR"][0]
    out = {"G_GR": hajek(S_gr - s_abst, w, idx)}
    for a in ("E-SQ", "E-R1"):
        Sa, Ha = arr[a][0], arr[a][1]
        out["F_" + a] = hajek((Sa - s_abst) - 0.5 * (S_gr - s_abst), w, idx)
        out["D_" + a] = hajek(Sa - S_gr, w, idx)
        out["dH_" + a] = hajek(Ha - arr["E-GR"][1], w, idx)
        out["B4_" + a] = hajek((Sa == 4).astype(float) - (S_gr == 4).astype(float), w, idx)
    if qscale is not None:
        for a in ARMS:
            out["Q4_" + a] = qscale * hajek((arr[a][0] == 4).astype(float), w, idx)
        for a in ("E-SQ", "E-R1"):
            out["dQ4_" + a] = out["Q4_" + a] - out["Q4_E-GR"]
    return out


def boot_indices(d: Data, B: int, seed: int) -> list[np.ndarray]:
    """§3.7: 표본에 나타난 토픽(등록 추출에서 26개; U_E 28 중 g0|small 에만 있고 뽑히지 않은 토픽은 빠진다) 군집 복원 재추출 → 뽑힌 토픽마다, 그 토픽 안의 층(g1|토픽, g0|토픽, g0|small 의 그 토픽 몫)별로
    표본 슬롯을 같은 수만큼 복원 재추출. 가중 w = N_h/n_h 는 고정, Hájek 로 다시 계산. 같은 색인을 비교·판정자에 공유한다."""
    rng = np.random.default_rng(seed)
    by = defaultdict(lambda: defaultdict(list))
    for k, (t, c) in enumerate(zip(d.topic, d.cell)):
        by[t][c].append(k)
    topics = sorted(by)
    out = []
    for _ in range(B):
        idx = []
        for t in rng.choice(len(topics), len(topics), replace=True).tolist():
            for c in sorted(by[topics[t]]):
                pool = by[topics[t]][c]
                idx += rng.choice(pool, len(pool), replace=True).tolist()
        out.append(np.array(idx, int))
    return out


def ci(vals, level):
    a = (100 - level) / 2
    v = np.asarray(vals, float)
    return [float(np.percentile(v, a)), float(np.percentile(v, 100 - a))]


def summarize(arr, d: Data, BI, s_abst: float = 2) -> dict:
    """s_abst 는 arr 를 만든 기권 점수와 같아야 한다(§3.7: A0 의 점수 = 기권 점수)."""
    qs = (float(d.w.sum()) / d.n_qs) if d.n_qs else None
    pt = stats_of(arr, d.w, qscale=qs, s_abst=s_abst)
    draws = [stats_of(arr, d.w, idx, qscale=qs, s_abst=s_abst) for idx in BI]
    return {k: {"est": v, "ci90": ci([x[k] for x in draws], 90), "ci95": ci([x[k] for x in draws], 95)} for k, v in pt.items()}


def verdicts(sm: dict, harm_margin: float = HARM_MARGIN) -> dict:
    gate = sm["G_GR"]["ci90"][0] > 0
    out = {"assay_gate": gate}
    for a in ("E-SQ", "E-R1"):
        f = sm["F_" + a]["ci90"]
        p1 = "결론 없음 (assay)" if not gate else ("비열등" if f[0] > 0 else ("열등" if f[1] < 0 else "결론 없음"))
        h = sm["dH_" + a]["ci90"]
        p2 = "비열등" if h[1] < harm_margin else ("열등" if h[0] > harm_margin else "결론 없음")
        out[a] = {"P1": p1, "P2": p2}
    return out


def abs_verdicts(sm: dict) -> dict:
    out = {}
    for a in ("E-SQ", "E-R1"):
        lo = sm["D_" + a]["ci90"][0]
        out[a] = {"D_ni_at_delta": {str(dl): bool(lo > -dl) for dl in (0.25, 0.5, 1.0)},
                  "B4_ni_delta0.10": bool(sm["B4_" + a]["ci90"][0] > -0.10)}
    return out


# --------------------------------------------------------------------------- 일치
def kappa_weighted(x, y, cats, kind="quadratic"):
    x, y = list(x), list(y)
    if not x:
        return None
    K = len(cats)
    ix = {c: k for k, c in enumerate(cats)}
    O = np.zeros((K, K))
    for a, b in zip(x, y):
        O[ix[a], ix[b]] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    Wt = np.array([[((i - j) / (K - 1)) ** 2 if kind == "quadratic" else abs(i - j) / (K - 1) if kind == "linear" else float(i != j)
                    for j in range(K)] for i in range(K)])
    den = (Wt * E).sum()
    return None if den < 1e-12 else float(1 - (Wt * O).sum() / den)


def gwet_ac2(x, y, cats):
    """Gwet AC2, 이차 가중(두 평정자)."""
    K = len(cats)
    ix = {c: k for k, c in enumerate(cats)}
    Wt = np.array([[1 - ((i - j) / (K - 1)) ** 2 for j in range(K)] for i in range(K)])
    n = len(x)
    pa = float(np.mean([Wt[ix[a], ix[b]] for a, b in zip(x, y)]))
    pi = np.array([(sum(a == c for a in x) + sum(b == c for b in y)) / (2 * n) for c in cats])
    pe = Wt.sum() / (K * (K - 1)) * float((pi * (1 - pi)).sum())
    return None if abs(1 - pe) < 1e-12 else (pa - pe) / (1 - pe)


def agreement(d: Data, jA: str, jB: str, B: int) -> dict:
    """판정자 간 일치(답한 본 문항, 점검·보정 제외). κ_w(S) CI = §3.7 과 같은 두 단 부트스트랩(토픽 → 토픽 안 층별 슬롯),
    seed 4411, B 회 — 뽑힌 슬롯의 답한 문항(슬롯 안 같은 글은 한 문항)을 모아 κ_w 를 다시 계산."""
    it = d.items
    LA, LB = d.labels[jA], d.labels[jB]
    sA = [score(LA[i]["C"], LA[i]["K"]) for i in it]
    sB = [score(LB[i]["C"], LB[i]["K"]) for i in it]
    cA, cB = [c2(LA[i]["C"]) for i in it], [c2(LB[i]["C"]) for i in it]
    kA, kB = [LA[i]["K"] for i in it], [LB[i]["K"] for i in it]
    out = {"n_items": len(it), "kw_S": kappa_weighted(sA, sB, [0, 1, 2, 3, 4]), "ac2_S": gwet_ac2(sA, sB, [0, 1, 2, 3, 4]),
           "exact": float(np.mean([a == b for a, b in zip(sA, sB)])), "within1": float(np.mean([abs(a - b) <= 1 for a, b in zip(sA, sB)])),
           "confusion_S": [[sum(1 for a, b in zip(sA, sB) if a == p and b == q) for q in range(5)] for p in range(5)],
           "kappa_linear_C": kappa_weighted(cA, cB, ["C0", "C1", "C2"], "linear"),
           "kappa_linear_K": kappa_weighted(kA, kB, ["K0", "K1", "K2"], "linear"),
           "kappa_harm_boundary": kappa_weighted([int(s <= 1) for s in sA], [int(s <= 1) for s in sB], [0, 1], "nominal"),
           "citation_precision": {j: (lambda L: (sum(L[i]["C"] == "C2a" for i in it) / max(1, sum(L[i]["C"] in ("C2a", "C2b") for i in it))))(d.labels[j]) for j in (jA, jB)}}
    pos = {i: k for k, i in enumerate(it)}
    slot_items = [sorted({pos[d.rows[sl][a]["item_id"]] for a in ARMS if d.rows[sl][a]["item_id"]}) for sl in d.slots]
    ks = []
    for idx in boot_indices(d, B, SEED_KW):
        kk = [k for s_ in idx.tolist() for k in slot_items[s_]]
        v = kappa_weighted([sA[k] for k in kk], [sB[k] for k in kk], [0, 1, 2, 3, 4]) if kk else None
        if v is not None:
            ks.append(v)
    out["kw_S_ci95"] = ci(ks, 95) if ks else [None, None]
    out["kw_S_ci_scheme"] = {"seed": SEED_KW, "B": B, "scheme": "topic cluster -> slots within (topic, cell) -> items of drawn slots"}
    return out


# --------------------------------------------------------------------------- 점검·보정
def catch_gate(d: Data, j: str) -> dict:
    L = d.labels.get(j, {})
    res = {"items": [], "pass": False}
    nc = 0
    for r in d.catch:
        C = L.get(r["item_id"], {}).get("C", "")
        if r["expected_C"] == "C0":
            ok = C == "C0"
        else:
            ok = c2(C) in ("C0", "C1")
        nc += ok
        res["items"].append({"difficulty": r["catch_difficulty"], "type": r["catch_type"], "correct": bool(ok),
                             "lenient_C_le_1": c2(C) in ("C0", "C1")})
    res["n_correct"] = nc
    res["hard_correct"] = sum(x["correct"] for x in res["items"] if x["difficulty"] == "hard")
    res["easy_correct"] = sum(x["correct"] for x in res["items"] if x["difficulty"] == "easy")
    res["pass"] = len(d.catch) == 4 and nc >= 3
    return res


def calib_agreement(d: Data, j: str) -> dict:
    L = d.labels.get(j, {})
    out = []
    for r in d.calib:
        x = L.get(r["item_id"], {})
        out.append({"C_first_match": x.get("first_C", x.get("C")) == r["ref_C"], "K_first_match": x.get("first_K", x.get("K")) == r["ref_K"]})
    return {"n": len(out), "C_match": sum(o["C_first_match"] for o in out), "K_match": sum(o["K_first_match"] for o in out)}


# --------------------------------------------------------------------------- 판정·문구
def combine(vs: list[str]) -> str:
    return vs[0] if len(set(vs)) == 1 else "결론 없음"


TABLE_R1 = {
    ("비열등", "비열등"): ("유지", "In blinded human evaluation, retrying did not improve evidence-grounded correctness/completeness (noninferior; 50 % retention, harm +0.10)."),
    ("비열등", "결론 없음"): ("유지", "In blinded human evaluation, retrying did not improve evidence-grounded correctness/completeness (noninferior; 50 % retention); harm comparison inconclusive."),
    ("비열등", "열등"): ("좁힘", "retrying reduces unsupported/incorrect answers by ΔH at the cost of fewer answers; it does not increase correct answers."),
    ("결론 없음", "비열등"): ("유지", "not demonstrated; human evaluation inconclusive on score"),
    ("결론 없음", "결론 없음"): ("유지", "not demonstrated; human evaluation inconclusive"),
    ("결론 없음", "열등"): ("좁힘", "retrying reduces unsupported/incorrect answers by ΔH at the cost of fewer answers; it does not increase correct answers."),
    ("열등", "비열등"): ("번복", "retrying increases human-judged correct and complete answers (F, D) at cost Y; Partial Rewind reduces that cost."),
    ("열등", "결론 없음"): ("번복", "retrying increases human-judged correct and complete answers (F, D) at cost Y; Partial Rewind reduces that cost. Harm inconclusive."),
    ("열등", "열등"): ("번복+해악", "retrying increases human-judged correct and complete answers (F, D) at cost Y and reduces harm; Partial Rewind reduces that cost."),
}
TABLE_SQ = {
    ("비열등", "비열등"): "Partial Rewind reduces retry cost; blinded human quality was noninferior.",
    ("비열등", "결론 없음"): "Partial Rewind reduces retry cost; quality noninferior on score; harm inconclusive.",
    ("비열등", "열등"): "trade-off: fewer abstentions but more unsupported answers (remove 'quality maintained').",
    ("결론 없음", "비열등"): "savings stand; human evaluation did not establish quality noninferiority; automatic measures only.",
    ("결론 없음", "결론 없음"): "savings stand; human evaluation did not establish quality noninferiority; automatic measures only.",
    ("결론 없음", "열등"): "trade-off: fewer abstentions but more unsupported answers (remove 'quality maintained').",
    ("열등", "비열등"): "savings reported as a quality-for-cost trade-off (remove 'quality maintained').",
    ("열등", "결론 없음"): "savings reported as a quality-for-cost trade-off (remove 'quality maintained').",
    ("열등", "열등"): "savings reported as a quality-for-cost trade-off (remove 'quality maintained').",
}


def base(v: str) -> str:
    return "결론 없음" if v.startswith("결론 없음") else v


def fill_numbers(txt: str, a: str, per_judge: dict, valid: list[str]) -> str:
    """§3.8 문구의 수치 자리: ΔH, (F, D) 는 유효 판정자별 점추정으로 채운다. cost Y 는 이 채점의 밖(R2 토큰 회계)이라
    ⟦cost Y⟧ 표시를 남긴다(원고에서 NUMBERS 값으로 채움)."""
    f3 = lambda x: "%+.3f" % x
    by = lambda k: " / ".join("%s %s" % (j, f3(per_judge[j]["summary"][k + "_" + a]["est"])) for j in valid)
    txt = txt.replace("by ΔH", "by ΔH (%s)" % by("dH"))
    txt = txt.replace("(F, D) at cost Y", "(F: %s; D: %s) at cost ⟦cost Y: R2 accounted-token figure⟧" % (by("F"), by("D")))
    return txt


def decide(per_judge: dict, valid: list[str], kw: float | None) -> dict:
    out = {}
    suffix = []
    if len(valid) == 1:
        suffix.append("(single blinded judge)")
    for a in ("E-SQ", "E-R1"):
        p1s = [per_judge[j]["verdicts"][a]["P1"] for j in valid]
        p2s = [per_judge[j]["verdicts"][a]["P2"] for j in valid]
        p1 = combine([base(x) for x in p1s]) if valid else "결론 없음"
        p2 = combine([base(x) for x in p2s]) if valid else "결론 없음"
        notes = list(suffix)
        # §3.8: assay 표시는 **모든** 유효 판정자의 assay 관문이 실패했을 때만 붙인다(판정자 불일치로 결론 없음이면 붙이지 않음)
        assay = bool(p1s) and all(x == "결론 없음 (assay)" for x in p1s)
        forced = []
        if len(valid) == 2 and (kw is None or kw < KW_BAR):
            if p1 == "비열등":
                p1 = "결론 없음"; forced.append("P1")
            if p2 == "비열등":
                p2 = "결론 없음"; forced.append("P2")
        if p1 == "결론 없음" and assay:
            notes.append("(assay sensitivity not established)")
        if forced:
            notes.append("(inter-judge reliability below 0.40)")
        if a == "E-R1":
            act, txt = TABLE_R1[(p1, p2)]
        else:
            act, txt = "—", TABLE_SQ[(p1, p2)]
        if all("summary" in per_judge[j] for j in valid):
            txt = fill_numbers(txt, a, per_judge, valid)
        out[a] = {"P1": p1, "P2": p2, "forced_by_reliability": forced, "action": act, "wording": txt + (" " + " ".join(notes) if notes else ""),
                  "assay_gate_by_judge": {j: per_judge[j]["verdicts"]["assay_gate"] for j in valid if "assay_gate" in per_judge[j]["verdicts"]},
                  "per_judge": {j: per_judge[j]["verdicts"][a] for j in valid}}
    return out


# --------------------------------------------------------------------------- 주 분석
def post_stage3_fix(labels: dict[str, dict[str, dict]]) -> tuple[dict, dict]:
    """§2.4·§3.7: 3단계가 처음 보인 시각(첫 3단계 shown_at 과 s3_started_at 중 이른 쪽) 뒤에 answered_at 이 찍힌 2단계 행을
    센다. 시트가 그런 저장을 막으므로 0 이어야 한다. 0 이 아니면 그 행은 1차 분석에서 first_C·first_K(3단계 전에 단 첫 답 —
    3단계는 2단계를 다 채워야 열린다)로 바꿔 쓴다. 공시(§3.7): CSV 는 첫 답과 마지막 답만 담으므로 "3단계 전 마지막 답"이
    아니라 첫 답으로 되돌린다 — 3단계 전에 한 정당한 2단계 수정도 함께 되돌려진다. 반환 (바꾼 라벨, 판정자별 수)."""
    out, rep = {}, {}
    for j, L in labels.items():
        t3 = [r.get("shown_at_iso", "") for r in L.values() if str(r.get("stage", "")) == "3" and r.get("shown_at_iso")]
        t3 += [r.get("s3_started_at_iso", "") for r in L.values() if r.get("s3_started_at_iso")]
        first3 = min(t3) if t3 else None
        LL, flagged = {}, []
        for i, r in L.items():
            r = dict(r)
            if first3 and str(r.get("stage", "")) == "2" and r.get("answered_at_iso") and r["answered_at_iso"] > first3:
                flagged.append(i)
                if r.get("first_C") and r.get("first_K"):
                    r["C"], r["K"] = r["first_C"], r["first_K"]
            LL[i] = r
        out[j] = LL
        rep[j] = {"first_stage3_at": first3, "n_stage2_rows_answered_after_stage3": len(flagged),
                  "treatment": "first_C/first_K substituted in the primary analysis (first answer, not the last pre-stage-3 answer)"}
    return out, rep


def analyze(keyrows: list[dict], labels: dict[str, dict[str, dict]], B: int = 10000, n_query_seeds: int | None = None) -> dict:
    labels, post3 = post_stage3_fix(labels)
    d = Data(keyrows, labels)
    d.n_qs = n_query_seeds
    BI = boot_indices(d, B, SEED_BOOT)
    judges = [j for j in ("A", "B") if d.judge_ok(j)]
    res = {"spec": "PREREG_R3 §3.7–§3.8", "n_slots": len(d.slots), "n_items": len(d.items), "judges_complete": judges,
           "answered_slots": {a: int(sum(str(d.rows[s][a]["answered"]) == "1" for s in d.slots)) for a in ARMS},
           "cells": len(set(d.cell)), "sum_w": float(d.w.sum()), "n_query_seeds_all": n_query_seeds,
           "post_stage3_stage2_answers": post3}
    res["partial_completion"] = {j: {"n_labelled_main": sum(1 for i in d.items if labels.get(j, {}).get(i, {}).get("C") and labels[j][i].get("K")),
                                     "n_main": len(d.items)} for j in ("A", "B") if j in labels and labels[j] and j not in judges}
    per = {}
    for j in judges:
        gate = catch_gate(d, j)
        arr = d.S(j)
        sm = summarize(arr, d, BI)
        L = d.labels[j]
        aux = {}
        for ab in (1, 3):
            sm2 = summarize(d.S(j, abstain=ab), d, BI, s_abst=ab)      # §3.7 보조 2: A0 기준도 같은 기권 점수
            aux["abstain_S=%d" % ab] = {"verdicts": verdicts(sm2), "summary": sm2, "s_abst": ab}
        aux["harm_margin_0.05"] = verdicts(sm, 0.05)
        aux["absolute"] = abs_verdicts(sm)
        aux["full_scale_factor"] = N_UE / N_SLOTS_ALL
        # 조건부: 두 arm 모두 답한 슬롯
        cond = {}
        for a in ("E-SQ", "E-R1"):
            both = np.array([str(d.rows[s][a]["answered"]) == "1" and str(d.rows[s]["E-GR"]["answered"]) == "1" for s in d.slots])
            same = np.array([d.rows[s][a]["item_id"] == d.rows[s]["E-GR"]["item_id"] for s in d.slots])
            dif = arr[a][0] - arr["E-GR"][0]
            cond[a] = {"n_both": int(both.sum()),
                       "D_both": hajek(dif, d.w, np.where(both)[0]) if both.any() else None,
                       "D_both_excl_identical": hajek(dif, d.w, np.where(both & ~same)[0]) if (both & ~same).any() else None}
        aux["conditional"] = cond
        dist = {}
        for a in ARMS:
            its = [d.rows[s][a]["item_id"] for s in d.slots if str(d.rows[s][a]["answered"]) == "1"]
            dist[a] = {"C": dict(Counter(L[i]["C"] for i in its)), "K": dict(Counter(L[i]["K"] for i in its)),
                       "mean_S_answered": float(np.mean([score(L[i]["C"], L[i]["K"]) for i in its])) if its else None}
        aux["distributions"] = dist
        aux["m_GR"] = dist["E-GR"]["mean_S_answered"]
        r1_only = [d.rows[s]["E-R1"]["item_id"] for s in d.slots
                   if str(d.rows[s]["E-R1"]["answered"]) == "1" and str(d.rows[s]["E-GR"]["answered"]) == "0"]
        aux["R1_answered_GR_abstained_C"] = {"n": len(r1_only), "C": dict(Counter(L[i]["C"] for i in r1_only))}
        # 토픽·seed 별 점추정, t000~t004 제외
        seeds = sorted({s[0] for s in d.slots})
        aux["by_seed"] = {sd: stats_of(arr, d.w, np.array([k for k, s in enumerate(d.slots) if s[0] == sd])) for sd in seeds}
        fo = [int(L[i].get("ctx_fold_opens") or 0) for i in d.items]
        aux["ctx_fold_opens"] = {"items_opened": int(sum(x > 0 for x in fo)), "n_items": len(fo),
                                 "C2b_when_opened": sum(1 for i, x in zip(d.items, fo) if x > 0 and L[i]["C"] == "C2b"),
                                 "C2b_when_not_opened": sum(1 for i, x in zip(d.items, fo) if x == 0 and L[i]["C"] == "C2b")}
        aux["sign_table_seed"] = {k: [int(np.sign(aux["by_seed"][sd][k])) for sd in seeds] for k in ("F_E-SQ", "F_E-R1", "dH_E-SQ", "dH_E-R1")}
        keep = np.array([k for k, s in enumerate(d.slots) if d.meta[s]["topic"] not in ("t000", "t001", "t002", "t003", "t004")])
        aux["excl_t000_t004"] = stats_of(arr, d.w, keep)
        # 제외 민감도 (해당 문항이 든 슬롯을 뺀다)
        excl = {}
        rowsof = lambda s: [d.rows[s][a] for a in ARMS]
        rules = {
            "excl_h4_overlap": lambda r: str(r.get("h4_overlap", "0")) == "1",
            "excl_active_lt_20s": lambda r: r["item_id"] and float(L[r["item_id"]].get("active_ms") or 0) < 20000,
            "excl_active_lt_1.2tmin": lambda r: r["item_id"] and float(L[r["item_id"]].get("active_ms") or 0) < 1.2 * 1000 * float(L[r["item_id"]].get("t_min_s") or 0),
            "excl_translator_full": lambda r: r["item_id"] and L[r["item_id"]].get("translator_used") == TR_FULL,
            "excl_paraphrase": lambda r: str(r.get("asked_ne_question", "0")) == "1" and str(r.get("produced_round", "0")) not in ("0", ""),
        }
        for nm, f in rules.items():
            kk = np.array([k for k, s in enumerate(d.slots) if not any(f(r) for r in rowsof(s))], int)
            excl[nm] = {"n_slots": int(len(kk)), "est": stats_of(arr, d.w, kk) if len(kk) else None}
        aux["exclusions"] = excl
        # 질의 수준(보조 8): 질의·seed 당 S = 4 sub-answer 수 = (N_UE / 전체 질의·seed 수) × E_w[1(S=4)], arm 별과 차이, CI 는 같은 BI
        aux["query_level_S4_per_query_seed"] = ({k: sm[k] for k in sm if k.startswith(("Q4_", "dQ4_"))}
                                                if d.n_qs else "n_query_seeds_all 없음(_meta_q.json)")
        per[j] = {"catch": gate, "calib": calib_agreement(d, j), "summary": sm, "verdicts": verdicts(sm), "aux": aux}
    res["per_judge"] = per
    valid = [j for j in judges if per[j]["catch"]["pass"]]
    res["valid_judges"] = valid
    kw = None
    if len(judges) == 2:
        ag = agreement(d, "A", "B", B)
        res["agreement"] = ag
        kw = ag["kw_S"]
    if len(valid) == 2:
        status = "충족" if (kw is not None and kw >= KW_BAR) else "충족, 신뢰도 제한"
    elif len(valid) == 1:
        status = "부분 충족 (single blinded judge, reliability not estimable)"
    else:
        status = "미충족"
    res["R1-7"] = status
    res["decision"] = decide(per, valid, kw) if valid else {a: {"P1": "결론 없음", "P2": "결론 없음", "wording": "결론 없음, 수치만 공개"} for a in ("E-SQ", "E-R1")}
    # 보조 9: 기계적 합의 판 (최솟값, 평균)
    if len(judges) == 2:
        mech = {}
        for nm, fn in (("min", min), ("mean", lambda a, b: (a + b) / 2)):
            lab_of = lambda i, fn=fn: (fn(score(labels["A"][i]["C"], labels["A"][i]["K"]), score(labels["B"][i]["C"], labels["B"][i]["K"])), None)
            arr = d.S("A", label_of=lab_of)
            sm = summarize(arr, d, BI)
            mech[nm] = {"verdicts": verdicts(sm), "summary": sm}
        res["mechanical_consensus"] = mech
    return res


def load_labels(p: Path) -> dict[str, dict]:
    return {r["item_id"]: r for r in rd(p)} if Path(p).exists() else {}


SHEET_VERSION = "r3_sheet_v1"
TEST_MARKS = ("-test", "PLACEHOLDER")


def label_path(d: Path, annot: str) -> Path:
    """받은 CSV 이름(§2.4): labels_A.csv / labels_B.csv."""
    return Path(d) / ("labels_%s.csv" % annot)


def check_csv(p: Path, annot: str, version: str = SHEET_VERSION) -> dict:
    """받은 CSV 의 출처 검사(채점 전, 값은 보지 않는다). 시험판(`-test`, `+PLACEHOLDER_REF`)·다른 판정자 파일·중복 id 는 멈춘다.
    `version` 을 바꿔도 `-test`·`PLACEHOLDER` 가 든 판은 **항상** 거부한다(§2.4: 시험 CSV = 받지 않은 것)."""
    assert not any(m in version for m in TEST_MARKS), "--sheet-version %r: 시험판 표시가 든 판은 받지 않는다" % version
    if not Path(p).exists():
        return {"present": False}
    rows = rd(p)
    vers = Counter(r.get("sheet_version", "") for r in rows)
    if any(m in v for v in vers for m in TEST_MARKS):              # §2.4: 시험판 CSV = 받지 않은 것(채점에서 뺀다)
        print("경고: %s 는 시험판 CSV(%s) — 받지 않은 것으로 다룬다" % (p, dict(vers)))
        return {"present": False, "rejected_test_csv": True, "sheet_version": dict(vers)}
    who = Counter(r.get("annotator", "") for r in rows)
    ids = Counter(r.get("item_id", "") for r in rows)
    assert set(vers) == {version}, "%s: sheet_version %s ≠ %s (시험판·자리표시자판은 채점하지 않는다)" % (p, dict(vers), version)
    assert set(who) == {annot}, "%s: annotator %s ≠ %s" % (p, dict(who), annot)
    assert all(v == 1 for v in ids.values()), "%s: 중복 item_id" % p
    return {"present": True, "rows": len(rows), "sheet_version": version, "annotator": annot}


# --------------------------------------------------------------------------- 자체 시험
def synth_key(seed: int = 7) -> list[dict]:
    """§3.3 와 같은 두 방향 층: g1(E-GR 답함) × 토픽(√N_h, 25), g0 × 토픽(N_h ≤ K_SMALL(4) 은 g0|small 하나, 층마다 1).
    28 토픽, 세 arm. 실제 봉인 키를 읽지 않는다."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from r3_sheet import K_SMALL, sqrt_alloc
    rng = np.random.default_rng(seed)
    N1 = {"g1|t%03d" % k: int(rng.integers(1, 12)) for k in range(24)}
    N0t = {"t%03d" % k: int(rng.integers(1, 23)) for k in range(28)}
    N0 = defaultdict(int)
    for t, v in N0t.items():
        N0["g0|" + (t if v > K_SMALL else "small")] += v
    al = {**sqrt_alloc(N1, 25), **{c: 1 for c in N0}}
    Ncell = {**N1, **N0}
    small_topics = sorted(t for t, v in N0t.items() if v <= K_SMALL)
    rows, n = [], 0
    for c in sorted(al):
        g1 = c.startswith("g1|")
        for q in range(al[c]):
            sd = str(1 + q % 3)
            t = c[3:] if c[3:] != "small" else small_topics[int(rng.integers(len(small_topics)))]
            ans = {"E-SQ": rng.random() < 0.78, "E-GR": g1, "E-R1": rng.random() < 0.75}
            if not any(ans.values()):
                ans["E-SQ"] = True
            same = ans["E-SQ"] and ans["E-R1"] and rng.random() < 0.3
            sq_id = ""
            for a in ARMS:
                iid = ""
                if ans[a]:
                    iid = sq_id if (same and a == "E-R1") else "q%04d" % n
                    n += 1
                    if a == "E-SQ":
                        sq_id = iid
                rows.append({"item_id": iid, "stage": "main", "arm": a, "seed": sd, "query_id": t, "branch": c, "sq_index": str(q),
                             "produced_round": "1" if a == "E-GR" else "0", "answered": "1" if ans[a] else "0", "stratum": c,
                             "topic": t, "egr_answered": str(int(g1)),
                             "N_h": str(Ncell[c]), "n_h": str(al[c]), "catch": "0", "h4_overlap": str(int(rng.random() < 0.1)),
                             "asked_ne_question": "1" if a == "E-GR" and rng.random() < 0.3 else "0"})
    for k, (diff, typ, exp) in enumerate((("hard", "contradiction", "C0"), ("hard", "contradiction", "C0"),
                                          ("easy", "evidence_swap", "C<=1"), ("easy", "evidence_swap", "C<=1"))):
        rows.append({"item_id": "qk%d" % k, "stage": "main", "arm": "catch", "catch": "1", "catch_type": typ,
                     "catch_difficulty": diff, "expected_C": exp, "answered": "1"})
    for k in range(3):
        rows.append({"item_id": "qc%d" % k, "stage": "calib", "arm": "E-SQ", "catch": "0", "ref_C": "C2a", "ref_K": "K2", "answered": "1"})
    return rows


def synth_labels(keyrows, scenario: str, judge: str, seed: int) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    arm_of = {}
    for r in keyrows:
        if r["item_id"]:
            arm_of.setdefault(r["item_id"], r["arm"])
    good = {"gate_pass": {"E-SQ": 0.85, "E-GR": 0.95, "E-R1": 0.85}, "gate_fail": {"E-SQ": 0.6, "E-GR": 0.25, "E-R1": 0.6},
            "r1_harmful": {"E-SQ": 0.9, "E-GR": 0.97, "E-R1": 0.3}}[scenario]
    out = {}
    for i, a in arm_of.items():
        if a == "catch":
            C = "C0" if rng.random() < 0.9 else "C2a"
            K = "K2"
        else:
            p = good.get(a, 0.8)
            if rng.random() < p:
                C, K = "C2a", ("K2" if rng.random() < 0.8 else "K1")
            else:
                C, K = ("C1" if rng.random() < 0.6 else "C0"), "K1"
        out[i] = {"item_id": i, "C": C, "K": K, "first_C": C, "first_K": K, "active_ms": str(int(rng.uniform(15000, 150000))),
                  "t_min_s": "60", "translator_used": rng.choice(["안 씀", "문장만", "근거 일부", "근거 전체"]),
                  "stage": "1" if i.startswith("qc") else "2", "shown_at_iso": "2026-11-11T05:00:00.000Z",
                  "answered_at_iso": "2026-11-11T05:01:00.000Z", "ctx_fold_opens": str(int(rng.integers(0, 2)))}
    return out


def noisy(L: dict[str, dict], p: float, seed: int) -> dict[str, dict]:
    """판정자 B = A 라벨에 잡음 p (점검 문항은 그대로)."""
    rng = np.random.default_rng(seed)
    out = {}
    for i, r in L.items():
        r = dict(r)
        if not i.startswith("qk") and rng.random() < p:
            r["C"] = {"C2a": "C1", "C1": "C2b", "C0": "C1", "C2b": "C2a"}[r["C"]]
            r["first_C"] = r["C"]
        out[i] = r
    return out


def write_csv_labels(path: Path, L: dict[str, dict], annot: str):
    cols = ["stage", "item_id", "label", "C", "K", "first_label", "first_C", "first_K", "n_changes", "active_ms", "t_min_s",
            "translator_used", "shown_at_iso", "answered_at_iso", "ctx_fold_opens", "s3_started_at_iso", "sheet_version", "annotator"]
    with io.open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in L.values():
            w.writerow({**r, "label": "", "first_label": "", "n_changes": 0, "sheet_version": "synthetic", "annotator": annot})


def selftest(B: int) -> int:
    import tempfile
    assert score("C0", "K2") == 0 and score("C1", "K0") == 1 and score("C2b", "K0") == 2 and score("C2a", "K1") == 3 and score("C2a", "K2") == 4
    assert abs(kappa_weighted([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) - 1) < 1e-12
    key = synth_key()
    seen = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for sc in ("gate_fail", "gate_pass", "r1_harmful"):
            LA = synth_labels(key, sc, "A", 11); LB = noisy(LA, 0.08, 12)
            write_csv_labels(td / "A.csv", LA, "A"); write_csv_labels(td / "B.csv", LB, "B")
            res = analyze(key, {"A": load_labels(td / "A.csv"), "B": load_labels(td / "B.csv")}, B=B, n_query_seeds=105)
            seen[sc] = res
            print("synthetic %-10s: R1-7 %s | κ_w %.3f | assay A %s | E-R1 %s / %s → %s | E-SQ %s / %s" % (
                sc, res["R1-7"], res["agreement"]["kw_S"], res["per_judge"]["A"]["verdicts"]["assay_gate"],
                res["decision"]["E-R1"]["P1"], res["decision"]["E-R1"]["P2"], res["decision"]["E-R1"]["action"],
                res["decision"]["E-SQ"]["P1"], res["decision"]["E-SQ"]["P2"]))
        # 한 판정자만 점검 통과
        LA = synth_labels(key, "gate_pass", "A", 11); LB = noisy(LA, 0.08, 12)
        for r in key:
            if str(r.get("catch")) == "1":
                LB[r["item_id"]]["C"] = "C2a"
        one = analyze(key, {"A": LA, "B": LB}, B=B)
        print("synthetic one-valid: %s | %s" % (one["R1-7"], one["decision"]["E-R1"]["wording"][-40:]))
        # κ_w 강제: B 를 무작위로
        rng = np.random.default_rng(3)
        LR = {i: {**v, "C": rng.choice(["C0", "C1", "C2a"]), "K": rng.choice(["K0", "K1", "K2"])} if not i.startswith("qk") else v
              for i, v in LA.items()}
        low = analyze(key, {"A": LA, "B": LR}, B=B)
        print("synthetic low-κ: R1-7 %s | κ_w %.3f | E-SQ %s forced %s" % (low["R1-7"], low["agreement"]["kw_S"],
              low["decision"]["E-SQ"]["P1"], low["decision"]["E-SQ"]["forced_by_reliability"]))
    assert seen["gate_fail"]["per_judge"]["A"]["verdicts"]["assay_gate"] is False
    assert "(assay sensitivity not established)" in seen["gate_fail"]["decision"]["E-R1"]["wording"]
    assert seen["gate_pass"]["per_judge"]["A"]["verdicts"]["assay_gate"] is True
    assert seen["r1_harmful"]["decision"]["E-R1"]["P2"] == "열등"
    assert one["R1-7"].startswith("부분 충족") and "(single blinded judge)" in one["decision"]["E-R1"]["wording"]
    assert low["R1-7"] == "충족, 신뢰도 제한"
    assert seen["gate_pass"]["decision"]["E-SQ"]["P1"] == "비열등"
    # 판정자 결합·κ_w 강제 단위 시험
    pj = lambda p1, p2: {"verdicts": {a: {"P1": p1, "P2": p2} for a in ("E-SQ", "E-R1")}}
    dd = decide({"A": pj("비열등", "비열등"), "B": pj("비열등", "비열등")}, ["A", "B"], 0.30)
    assert dd["E-SQ"]["P1"] == "결론 없음" and dd["E-SQ"]["forced_by_reliability"] == ["P1", "P2"]
    assert "(inter-judge reliability below 0.40)" in dd["E-SQ"]["wording"]
    dd = decide({"A": pj("열등", "비열등"), "B": pj("열등", "결론 없음")}, ["A", "B"], 0.30)
    assert dd["E-R1"]["P1"] == "열등" and dd["E-R1"]["P2"] == "결론 없음" and dd["E-R1"]["action"] == "번복"
    dd = decide({"A": pj("비열등", "비열등"), "B": pj("열등", "비열등")}, ["A", "B"], 0.80)
    assert dd["E-SQ"]["P1"] == "결론 없음" and dd["E-SQ"]["P2"] == "비열등"
    # assay 표시: 한 판정자만 assay 실패 → 표시 없음; 둘 다 실패 → 표시
    dd = decide({"A": pj("결론 없음 (assay)", "비열등"), "B": pj("비열등", "비열등")}, ["A", "B"], 0.80)
    assert dd["E-SQ"]["P1"] == "결론 없음" and "(assay sensitivity not established)" not in dd["E-SQ"]["wording"]
    dd = decide({"A": pj("결론 없음 (assay)", "비열등"), "B": pj("결론 없음 (assay)", "비열등")}, ["A", "B"], 0.80)
    assert "(assay sensitivity not established)" in dd["E-SQ"]["wording"]
    # 3단계 뒤 2단계 저장: 표시·first 값으로 대체
    LA = synth_labels(key, "gate_pass", "A", 11)
    some = next(i for i in LA if not i.startswith(("qk", "qc")))
    LA[some] = {**LA[some], "C": "C0", "first_C": "C2a", "answered_at_iso": "2026-11-11T09:00:00.000Z"}
    LA["vp_x"] = {"item_id": "vp_x", "stage": "3", "shown_at_iso": "2026-11-11T08:00:00.000Z", "label": "지지"}
    fixed, rep = post_stage3_fix({"A": LA})
    assert rep["A"]["n_stage2_rows_answered_after_stage3"] == 1 and fixed["A"][some]["C"] == "C2a"
    # §3.7 보조 2: 기권 점수 s_abst 를 G·F 의 A0 기준에도 같이 쓴다
    wt = np.array([1.0, 2.0, 3.0, 4.0])
    for sa in (1, 2, 3):
        arr0 = {"E-GR": (np.full(4, float(sa)), np.zeros(4), np.zeros(4, int)),      # E-GR 전부 기권 = A0
                "E-SQ": (np.array([4.0, 0.0, sa, 3.0]), np.array([0.0, 1.0, 0.0, 0.0]), np.array([1, 1, 0, 1])),
                "E-R1": (np.array([sa, 1.0, 2.0, 4.0]), np.array([0.0, 1.0, 0.0, 0.0]), np.array([0, 1, 1, 1]))}
        st0 = stats_of(arr0, wt, s_abst=sa)
        assert abs(st0["G_GR"]) < 1e-12, (sa, st0["G_GR"])                       # A0 자신의 G = 0
        for a in ("E-SQ", "E-R1"):
            assert abs(st0["F_" + a] - hajek(arr0[a][0] - sa, wt)) < 1e-12, (sa, a)
    dS = Data(key, {"A": synth_labels(key, "gate_pass", "A", 11)})
    a2 = dS.S("A")
    piGR = hajek(a2["E-GR"][2].astype(float), dS.w)
    rS = analyze(key, {"A": synth_labels(key, "gate_pass", "A", 11)}, B=20)["per_judge"]["A"]
    for sa in (1, 3):
        g_s = rS["aux"]["abstain_S=%d" % sa]["summary"]["G_GR"]["est"]
        assert rS["aux"]["abstain_S=%d" % sa]["s_abst"] == sa
        assert abs(g_s - hajek(dS.S("A", abstain=sa)["E-GR"][0] - sa, dS.w)) < 1e-12
        assert abs(g_s - (rS["summary"]["G_GR"]["est"] - (sa - 2) * piGR)) < 1e-9, (sa, g_s)   # G(s) = G(2) − (s − 2)·π_GR
    print("s_abst: A0 G = 0 at s = 1/2/3; G(1) %.3f > G(2) %.3f > G(3) %.3f" % (
        rS["aux"]["abstain_S=1"]["summary"]["G_GR"]["est"], rS["summary"]["G_GR"]["est"], rS["aux"]["abstain_S=3"]["summary"]["G_GR"]["est"]))
    # 질의 수준 추정과 차이(n_query_seeds 를 받는다, 105 하드코딩 없음)
    q = seen["gate_pass"]["per_judge"]["A"]["aux"]["query_level_S4_per_query_seed"]
    assert {"Q4_E-SQ", "Q4_E-GR", "Q4_E-R1", "dQ4_E-SQ", "dQ4_E-R1"} <= set(q), q
    assert seen["gate_pass"]["agreement"]["kw_S_ci_scheme"]["seed"] == 4411
    # 시험판 CSV 거부
    for bad in ("r3_sheet_v1-test", "r3_sheet_v1+PLACEHOLDER_REF"):
        try:
            check_csv(Path(__file__), "A", bad)
            raise RuntimeError("시험판 판을 받아들였다")
        except AssertionError:
            pass
    print("selftest OK (values from synthetic labels are discarded)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--labels-dir", default=str(R3 / "labels"))
    ap.add_argument("--key", default=str(R3 / "sealed/_key_q.csv"))
    ap.add_argument("--out", default=str(R3 / "quality/RESULT.json"))
    ap.add_argument("--sheet-version", default=SHEET_VERSION,
                    help="받아들일 sheet_version (배관 시험용). '-test'·'PLACEHOLDER' 가 든 값은 항상 거부한다")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.B)
    ld = Path(a.labels_dir)
    prov = {x: check_csv(label_path(ld, x), x, a.sheet_version) for x in ("A", "B")}
    meta_p = Path(a.key).parent / "_meta_q.json"
    nqs = json.load(io.open(meta_p, encoding="utf-8")).get("n_query_seeds_all") if meta_p.exists() else None
    res = analyze(rd(Path(a.key)), {x: (load_labels(label_path(ld, x)) if prov[x]["present"] else {}) for x in ("A", "B")},
                  B=a.B, n_query_seeds=nqs)
    res["csv_provenance"] = prov
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    io.open(a.out, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print("-> %s: R1-7 %s; E-R1 %s; E-SQ %s" % (a.out, res["R1-7"], res["decision"]["E-R1"]["wording"], res["decision"]["E-SQ"]["wording"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
