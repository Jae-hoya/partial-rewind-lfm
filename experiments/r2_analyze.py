"""R2 사후 분석 — run_tag 디렉터리 하나를 읽어 ANALYSIS.json 과 ANALYSIS.md 를 낸다. **LLM 호출 0.**

  python experiments/r2_analyze.py results/r2/pilot5 --pilot          # 파일럿: G1/G3/G4 관문 포함
  python experiments/r2_analyze.py results/r2/heldout35 --rescore     # 문장 단위 MiniCheck 재채점 (GPU, LLM 0)

근거: PREREG_R2 §6(관문), §8(분석). 입력 형식은 experiments/r2_run.py 가 쓰는 그대로다.
  <run_dir>/<arm>[@model]/s<seed>/topics/<qid>.json   rounds[].branches[] (branch_record)
  <run_dir>/<arm>/s<seed>/usage.json                  프로세스별 usage 의 **목록** (재개마다 한 줄)
  <run_dir>/billing_ledger.jsonl                      run_tag 전체에 하나. 행에 arm 이 없다 (nonce = s{seed}|{family}|r..)

형식에서 오는 제약 (ANALYSIS.md 에도 적는다)
  - arm 별 청구 토큰은 원장으로 나눌 수 없다. executed branch 의 billing.billed_in/out 합으로 낸다.
    공통 난수 때문에 뒤에 도는 arm(GR)은 앞 arm(BR)의 캐시를 맞혀 청구가 작게 나온다 — 청구는 실행 순서에
    의존하고 회계 토큰(tokens_in+out, executed 만)은 의존하지 않는다. 절감 주장은 회계 토큰으로 한다.
  - 토픽 JSON 에는 문장 단위 지지 판정과 청크 본문이 없다. cited-unsupported 와 sufficient 를 가르려면
    --rescore(또는 scorer 주입)가 필요하다. 없으면 두 상태를 'cited' 하나로 합치고 그렇게 표시한다.
  - E-R1 은 실행하지 않는다. E-GR 의 라운드 0 을 최종 출력으로 삼아 여기서 만든다.

R1-8 상태 전이표는 **회계 분해이고 인과가 아니다.** 토큰은 branch 실행 토큰을 그 실행에서 새로 만든
sub-answer 수로 균등하게 나눠 붙인다.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lfm.types import CITATION_RE, split_sentences  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# experiments/r2_run.py 의 ARMS / ORDER 를 그대로 옮겼다 (r2_run 은 import 하면 NLI·검색기를 끌어온다)
ARMS = {
    "L-GR": ("L", "legacy", "global_retry", "branch"),
    "L-BR": ("L", "legacy", "freeze_retry", "branch"),
    "E-GR": ("E", "status", "global_retry", "branch"),
    "E-SQ": ("E", "status", "freeze_retry", "subq"),
    "E-BR": ("E", "status", "freeze_retry", "branch"),
}
ORDER = ["L-BR", "L-GR", "E-SQ", "E-BR", "E-GR"]
R1 = "E-R1"
# PREREG §8 절감(1차) 비교 셋. (A, B) — 절감 = 1 - ΣA / ΣB
SAVINGS = [("E-SQ", "E-GR"), ("L-BR", "L-GR"), ("E-BR", "E-SQ")]
# PREREG §8 비열등성 (A - B)
NONINF = [("E-SQ", "E-GR"), (R1, "E-GR")]
PILOT_DEV = {"t000", "t001", "t002", "t003", "t004"}   # PREREG §1: 30토픽 민감도에서 뺀다
N_BOOT = 10_000
BOOT_SEED = 42
BIG_EFFECT = 0.20                     # PREREG §8 사전등록 대효과
DELTA_COV, DELTA_COV_SENS = 0.05, (0.03, 0.10)
DELTA_GRD = 0.05
ABSTAIN = ("format_fail", "insufficient", "dropped")
STATES = ("format_fail", "insufficient", "dropped", "cited-unsupported", "sufficient")
METRICS = ("accounting_tokens", "billed_tokens", "branch_rounds", "subq_executed")


# --------------------------------------------------------------------------- 읽기
def parse_arm(name: str) -> tuple[str, str | None]:
    arm, _, model = name.partition("@")
    return arm, (model or None)


def load_run(run_dir: Path) -> dict[tuple[str, int], dict[str, dict]]:
    """{(arm 이름(@model 포함), seed): {qid: topic record}}. TESTA-replay 는 건너뛴다."""
    runs: dict[tuple[str, int], dict[str, dict]] = {}
    for arm_dir in sorted(p for p in Path(run_dir).iterdir() if p.is_dir()):
        arm, _ = parse_arm(arm_dir.name)
        if arm not in ARMS:
            continue
        for sd in sorted(p for p in arm_dir.iterdir() if p.is_dir() and re.fullmatch(r"s\d+", p.name)):
            tdir = sd / "topics"
            if not tdir.exists():
                continue
            recs = {}
            for f in sorted(tdir.glob("*.json")):
                rec = json.load(io.open(f, encoding="utf-8"))
                recs[str(rec["query_id"])] = rec
            runs[(arm_dir.name, int(sd.name[1:]))] = recs
    return runs


def load_ledger(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "billing_ledger.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def load_usage(run_dir: Path) -> dict[tuple[str, int], dict[str, int]]:
    out = {}
    for f in Path(run_dir).glob("*/s*/usage.json"):
        hist = json.load(io.open(f, encoding="utf-8"))
        tot = Counter()
        for h in hist:
            u = h.get("usage", {})
            for k in ("calls", "cache_hits", "api_calls", "billed_tokens_in", "billed_tokens_out"):
                tot[k] += int(u.get(k, 0))
            for tag, slot in (u.get("by_tag") or {}).items():
                tot["decompose_api_calls"] += (int(slot.get("calls", 0)) - int(slot.get("cache_hits", 0))) \
                    if tag.startswith("decompose") else 0
        out[(f.parent.parent.name, int(f.parent.name[1:]))] = dict(tot)
    return out


def derive_r1(egr: dict[str, dict]) -> dict[str, dict]:
    """E-R1 = E-GR 라운드 0. 재시도 없는 시스템이므로 라운드 0 이 최종이다."""
    out = {}
    for qid, rec in egr.items():
        r = copy.deepcopy(rec)
        r["rounds"] = r["rounds"][:1]
        r["n_rounds"] = 1
        r["passed"] = bool(r["rounds"][0]["verify"]["passed"]) if r["rounds"] else False
        r["derived_from"] = "E-GR round 0"
        out[qid] = r
    return out


def with_r1(runs: dict) -> dict:
    runs = dict(runs)
    for (arm, seed), recs in list(runs.items()):
        a, model = parse_arm(arm)
        if a == "E-GR":
            runs[(R1 + ("@" + model if model else ""), seed)] = derive_r1(recs)
    return runs


# --------------------------------------------------------------------------- 문장 지지 (선택)
class Support:
    """sub-answer 단위 MiniCheck 지지. scorer 는 grounded_many(items) 를 가진 Estimator, chunk_text 는 id->본문.

    한 번에 모아 채점하고 (문장, 전제) 쌍으로 메모한다 — 같은 family·seed 의 라운드 0 은 arm 마다 같다.
    """

    def __init__(self, scorer=None, chunk_text: dict[str, str] | None = None):
        self.scorer, self.chunk_text = scorer, chunk_text or {}
        self.memo: dict[tuple[str, tuple[str, ...]], bool] = {}
        self.missing_chunks: set[str] = set()

    @property
    def available(self) -> bool:
        return self.scorer is not None

    def _items(self, sa: dict) -> list[tuple[str, tuple[str, ...]]]:
        prem = []
        for c in dict.fromkeys(sa.get("citations") or []):
            if c in self.chunk_text:
                prem.append(self.chunk_text[c])
            else:
                self.missing_chunks.add(c)
        # evidence.cited_texts 와 같은 규칙: 문장마다 이 sub-answer 가 인용한 **고유** 청크, 첫 인용 순서
        return [(s, tuple(prem)) for s in split_sentences(sa.get("text") or "")]

    def prime(self, sub_answers: Iterable[dict]) -> None:
        if not self.available:
            return
        todo = []
        for sa in sub_answers:
            for it in self._items(sa):
                if it not in self.memo and it[1]:
                    todo.append(it)
        todo = list(dict.fromkeys(todo))
        if todo:
            flags = self.scorer.grounded_many([(s, list(p)) for s, p in todo])
            self.memo.update(zip(todo, (bool(f) for f in flags)))

    def sentence_flags(self, sa: dict) -> list[bool] | None:
        if not self.available:
            return None
        return [bool(p) and self.memo.get((s, p), False) for s, p in self._items(sa)]

    def supported(self, sa: dict) -> bool | None:
        flags = self.sentence_flags(sa)
        if flags is None:
            return None
        return bool(flags) and all(flags)


def load_chunk_text(index_dir: Path) -> dict[str, str]:
    """data/index/chunks.jsonl 을 직접 읽는다 (BM25Retriever 를 만들지 않는다)."""
    out = {}
    with io.open(Path(index_dir) / "chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            out[d["chunk_id"]] = d["text"]
    return out


# --------------------------------------------------------------------------- sub-answer 상태
def generated(br: dict, rnd: int) -> list[dict]:
    """이 branch 실행이 새로 만든 sub-answer. produced_round 가 없으면(Test A 재생) 실행 branch 전부."""
    if not br.get("executed"):
        return []
    sas = br.get("sub_answers") or []
    if any(sa.get("produced_round") is not None for sa in sas):
        return [sa for sa in sas if sa.get("produced_round") == rnd]
    return list(sas)


def sa_state(sa: dict, sup: Support) -> str:
    st = sa.get("status")
    if st in ABSTAIN:
        return st
    if not sa.get("citations") or not (sa.get("text") or "").strip():
        # legacy: 빈 답 또는 유효 인용 없음. status 가 없으므로 기권 이유를 모른다 — insufficient 로 센다
        return "insufficient"
    s = sup.supported(sa)
    if s is None:
        return "cited"
    return "sufficient" if s else "cited-unsupported"


def final_round(rec: dict) -> dict:
    return rec["rounds"][-1]


def br_tokens(br: dict) -> int:
    return int(br.get("tokens_in", 0)) + int(br.get("tokens_out", 0)) if br.get("executed") else 0


# --------------------------------------------------------------------------- arm × seed 집계
def query_metrics(rec: dict, sup: Support) -> dict[str, Any]:
    acc_in = acc_out = bill_in = bill_out = api = calls = brr = sq = 0
    by_tag: Counter = Counter()
    for rr in rec["rounds"]:
        for br in rr["branches"]:
            if not br.get("executed"):
                continue                      # 물려받은 branch: 비용 0 (값이 있어도 세지 않는다)
            acc_in += int(br.get("tokens_in", 0))
            acc_out += int(br.get("tokens_out", 0))
            b = br.get("billing") or {}
            bill_in += int(b.get("billed_in", 0))
            bill_out += int(b.get("billed_out", 0))
            api += int(b.get("api_calls", 0))
            calls += int(br.get("llm_calls", 0))
            brr += 1
            sq += int(br.get("subq_executed") or len(br.get("sub_answers") or []))
            for tag, slot in (b.get("by_tag") or {}).items():
                by_tag[tag + ":calls"] += int(slot.get("calls", 0))
                by_tag[tag + ":tokens"] += int(slot.get("tokens_in", 0)) + int(slot.get("tokens_out", 0))
    fr = final_round(rec)
    subs = [sa for br in fr["branches"] for sa in br.get("sub_answers") or []]
    n_cov = sum(1 for sa in subs if sa.get("citations"))
    # grounding: 문장 가중 풀링. 문장이 0 인 branch 는 evidence 상 0.0 이지만 가중치도 0 이다
    sent = [(len(split_sentences(sa.get("text") or "")), br) for br in fr["branches"] for sa in br.get("sub_answers") or []]
    n_sent = defaultdict(int)
    for n, br in sent:
        n_sent[br["branch"]] += n
    if sup.available:
        flags = [f for sa in subs for f in (sup.sentence_flags(sa) or [])]
        grd = (sum(flags) / len(flags)) if flags else 0.0
    else:
        tot = sum(n_sent.values())
        grd = (sum(float(br["evidence"]["grounding"]) * n_sent[br["branch"]] for br in fr["branches"]) / tot) if tot else 0.0
    agg = (fr.get("verify") or {}).get("agg") or {}
    frozen = (fr.get("action") or {}).get("params", {}).get("frozen") or []
    return {
        "accounting_in": acc_in, "accounting_out": acc_out, "accounting_tokens": acc_in + acc_out,
        "billed_in": bill_in, "billed_out": bill_out, "billed_tokens": bill_in + bill_out,
        "api_calls": api, "llm_calls": calls, "branch_rounds": brr, "subq_executed": sq,
        "coverage": (n_cov / len(subs)) if subs else 0.0, "grounding": grd,
        "passed": bool(fr["verify"]["passed"]), "pass_grounding": bool(agg.get("pass_grounding", False)),
        "pass_coverage": bool(agg.get("pass_coverage", False)),
        "n_rounds": len(rec["rounds"]), "n_frozen_final": len(frozen), "by_tag": dict(by_tag),
    }


def armseed_summary(recs: dict[str, dict], sup: Support) -> dict[str, Any]:
    q = {qid: query_metrics(rec, sup) for qid, rec in sorted(recs.items())}
    tot = Counter()
    by_tag = Counter()
    for m in q.values():
        for k in ("accounting_in", "accounting_out", "accounting_tokens", "billed_in", "billed_out",
                  "billed_tokens", "api_calls", "llm_calls", "branch_rounds", "subq_executed",
                  "passed", "pass_grounding", "pass_coverage", "n_frozen_final"):
            tot[k] += int(m[k])
        by_tag.update(m["by_tag"])
    gen_status = Counter()
    fin_status = Counter()
    for rec in recs.values():
        for rr in rec["rounds"]:
            for br in rr["branches"]:
                for sa in generated(br, rr["round"]):
                    gen_status[sa.get("status") or "legacy"] += 1
        for br in final_round(rec)["branches"]:
            for sa in br.get("sub_answers") or []:
                fin_status[sa.get("status") or "legacy"] += 1
    ng, nf = sum(gen_status.values()), sum(fin_status.values())
    n = len(q)
    return {
        "n_topics": n, "totals": dict(tot),
        "verify": {"passed": tot["passed"], "pass_grounding": tot["pass_grounding"],
                   "pass_coverage": tot["pass_coverage"], "n": n},
        "frozen_branches_final": tot["n_frozen_final"],
        "by_tag": dict(by_tag),
        "generated_status": dict(gen_status),
        "generated_rates": {s: (gen_status[s] / ng if ng else None) for s in ABSTAIN + ("answered",)},
        "final_rates": {s: (fin_status[s] / nf if nf else None) for s in ABSTAIN + ("answered",)},
        "mean_coverage": float(np.mean([m["coverage"] for m in q.values()])) if q else None,
        "mean_grounding": float(np.mean([m["grounding"] for m in q.values()])) if q else None,
        "per_query": q,
    }


# --------------------------------------------------------------------------- 부트스트랩
def _matrix(per: dict[int, dict[str, float]], topics: list[str], seeds: list[int]) -> np.ndarray:
    return np.array([[per[s][t] for s in seeds] for t in topics], dtype=float)


def hier_indices(n_t: int, n_s: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """topic × seed 2단 계층: 토픽을 복원추출하고, 뽑힌 토픽마다 seed 를 복원추출한다."""
    rng = np.random.default_rng(seed)
    ti = rng.integers(0, n_t, size=(n_boot, n_t))
    si = rng.integers(0, n_s, size=(n_boot, n_t, n_s))
    return ti, si


def boot_savings(A: np.ndarray, B: np.ndarray, n_boot=N_BOOT, seed=BOOT_SEED) -> np.ndarray:
    ti, si = hier_indices(A.shape[0], A.shape[1], n_boot, seed)
    a = A[ti[:, :, None], si].sum(axis=(1, 2))
    b = B[ti[:, :, None], si].sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - a / b


def boot_meandiff(A: np.ndarray, B: np.ndarray, n_boot=N_BOOT, seed=BOOT_SEED) -> np.ndarray:
    ti, si = hier_indices(A.shape[0], A.shape[1], n_boot, seed)
    return (A - B)[ti[:, :, None], si].mean(axis=(1, 2))


def var_components(D: np.ndarray) -> dict[str, float | None]:
    """일원 랜덤효과(요인 = seed) ANOVA 추정. D 는 topic × seed 짝 차이."""
    n_t, n_s = D.shape
    if n_s < 2 or n_t < 2:
        return {"sigma2_seed": None, "sigma2_within": None}
    gm = D.mean()
    msb = n_t * ((D.mean(axis=0) - gm) ** 2).sum() / (n_s - 1)
    msw = ((D - D.mean(axis=0)) ** 2).sum() / (n_s * (n_t - 1))
    return {"sigma2_seed": float(max(0.0, (msb - msw) / n_t)), "sigma2_within": float(msw)}


def _common(summ: dict, a: str, b: str, topic_filter: Callable[[str], bool]):
    seeds = sorted({s for (x, s) in summ if x == a} & {s for (x, s) in summ if x == b})
    if not seeds:
        return [], []
    topics = None
    for s in seeds:
        ts = set(summ[(a, s)]["per_query"]) & set(summ[(b, s)]["per_query"])
        topics = ts if topics is None else topics & ts
    return sorted(t for t in topics if topic_filter(t)), seeds


def savings_compare(summ, a, b, metric, topic_filter=lambda t: True, n_boot=N_BOOT) -> dict:
    topics, seeds = _common(summ, a, b, topic_filter)
    if not topics:
        return {"a": a, "b": b, "metric": metric, "available": False}
    per = lambda arm: {s: {t: summ[(arm, s)]["per_query"][t][metric] for t in topics} for s in seeds}
    A, B = _matrix(per(a), topics, seeds), _matrix(per(b), topics, seeds)
    point = float(1 - A.sum() / B.sum()) if B.sum() else None
    boots = boot_savings(A, B, n_boot)
    lo, hi = (float(x) for x in np.nanpercentile(boots, [2.5, 97.5]))
    per_seed = {str(s): (float(1 - A[:, j].sum() / B[:, j].sum()) if B[:, j].sum() else None)
                for j, s in enumerate(seeds)}
    signs = {k: (None if v is None else int(np.sign(v))) for k, v in per_seed.items()}
    same_sign = len(set(signs.values())) == 1 and None not in signs.values() and 0 not in signs.values()
    excl0 = lo > 0 or hi < 0
    return {"a": a, "b": b, "metric": metric, "available": True, "n_topics": len(topics), "seeds": seeds,
            "definition": "1 - Σ%s(%s) / Σ%s(%s)" % (metric, a, metric, b),
            "point": point, "ci95": [lo, hi], "per_seed": per_seed, "sign_table": signs,
            # PREREG §8: 청구 토큰은 공통 난수·실행 순서에 의존하므로 보고만 한다. 주장은 회계 토큰 등으로만
            "claim": bool(same_sign and excl0) if metric != "billed_tokens" else None,
            "vs_big_effect": {"threshold": BIG_EFFECT, "point_ge": (point is not None and point >= BIG_EFFECT),
                              "ci_lo_ge": bool(lo >= BIG_EFFECT)},
            "var_components": var_components(A - B)}


def noninf_compare(summ, a, b, metric, delta, sens=(), topic_filter=lambda t: True, n_boot=N_BOOT) -> dict:
    topics, seeds = _common(summ, a, b, topic_filter)
    if not topics:
        return {"a": a, "b": b, "metric": metric, "available": False}
    per = lambda arm: {s: {t: summ[(arm, s)]["per_query"][t][metric] for t in topics} for s in seeds}
    A, B = _matrix(per(a), topics, seeds), _matrix(per(b), topics, seeds)
    boots = boot_meandiff(A, B, n_boot)
    lo, hi = (float(x) for x in np.percentile(boots, [5, 95]))
    res = {"a": a, "b": b, "metric": metric, "available": True, "n_topics": len(topics), "seeds": seeds,
           "diff": float((A - B).mean()), "ci90": [lo, hi], "delta": delta,
           "noninferior": bool(lo > -delta), "equivalent": bool(lo > -delta and hi < delta),
           "per_seed": {str(s): float((A[:, j] - B[:, j]).mean()) for j, s in enumerate(seeds)}}
    res["sensitivity"] = {str(d): {"noninferior": bool(lo > -d), "equivalent": bool(lo > -d and hi < d)} for d in sens}
    return res


# --------------------------------------------------------------------------- R1-8 전이표
def transitions(recs: dict[str, dict], sup: Support) -> dict[str, Any]:
    """재실행 단위(= 라운드 r+1 에서 새로 만든 sub-answer)의 (r → r+1) 상태. 회계 분해, 인과 아님."""
    table: dict[str, dict[str, int]] = defaultdict(Counter)
    tok: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    initial, initial_tok = Counter(), defaultdict(float)
    unit_hist: dict[tuple, list[str]] = defaultdict(list)
    unit_tok: dict[tuple, float] = defaultdict(float)
    repair_tokens = retry_tokens = 0
    n_units = 0
    mismatch_subq_executed = 0
    for qid, rec in recs.items():
        prev: dict[str, dict[int, dict]] = {}
        for rr in rec["rounds"]:
            r = rr["round"]
            cur = {}
            for br in rr["branches"]:
                cur[br["branch"]] = {sa["index"]: sa for sa in br.get("sub_answers") or []}
                gen = generated(br, r)
                if br.get("executed"):
                    for tag, slot in ((br.get("billing") or {}).get("by_tag") or {}).items():
                        t = int(slot.get("tokens_in", 0)) + int(slot.get("tokens_out", 0))
                        if tag.endswith("_repair"):
                            repair_tokens += t
                        elif tag == "answer_retry":
                            retry_tokens += t
                    if br.get("subq_executed") and int(br["subq_executed"]) != len(gen):
                        mismatch_subq_executed += 1
                if not gen:
                    continue
                share = br_tokens(br) / len(gen)
                for sa in gen:
                    st = sa_state(sa, sup)
                    key = (qid, br["branch"], sa["index"])
                    unit_hist[key].append(st)
                    unit_tok[key] += share
                    if r == 0:
                        initial[st] += 1
                        initial_tok[st] += share
                        continue
                    p = prev.get(br["branch"], {}).get(sa["index"])
                    frm = sa_state(p, sup) if p is not None else "missing"
                    table[frm][st] += 1
                    tok[frm][st] += share
                    n_units += 1
            prev = cur
    # 문장 지지 판정이 없으면(--rescore 없이) sufficient 상태가 없어 이 값은 구조적으로 0 이다 — 측정 불가로 둔다
    fmt_rec = sum(v for k, v in tok.get("format_fail", {}).items() if k == "sufficient") if sup.available else None
    unrec = sum(unit_tok[k] for k, h in unit_hist.items() if h and all(s == "insufficient" for s in h))
    from_ins = table.get("insufficient", {})
    n_from_ins = sum(from_ins.values())
    to_answered = sum(v for k, v in from_ins.items() if k not in ABSTAIN)
    return {
        "label": "accounting decomposition (회계 분해, 인과 아님)",
        "states": list(STATES) + (["cited"] if not sup.available else []),
        "counts": {k: dict(v) for k, v in table.items()},
        "tokens": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in tok.items()},
        "n_reexecuted_units": n_units,
        "round0": {"counts": dict(initial), "tokens": {k: round(v, 3) for k, v in initial_tok.items()}},
        "format_recovery_tokens": round(fmt_rec, 3) if fmt_rec is not None else None,
        "format_fail_to_cited_tokens": round(sum(v for k, v in tok.get("format_fail", {}).items()
                                                 if k in ("cited", "sufficient", "cited-unsupported")), 3),
        "format_recovery_upper_bound": {"repair_tokens": repair_tokens, "answer_retry_tokens": retry_tokens,
                                        "sum": repair_tokens + retry_tokens},
        "unrecoverable_tokens": round(unrec, 3),
        "insufficient_to_answered": {"n_from_insufficient": n_from_ins, "n_to_answered": to_answered,
                                     "rate": (to_answered / n_from_ins) if n_from_ins else None},
        "subq_executed_mismatch": mismatch_subq_executed,
    }


def abstention_by_round(recs: dict[str, dict]) -> dict[str, Any]:
    c = defaultdict(Counter)
    for rec in recs.values():
        for rr in rec["rounds"]:
            for br in rr["branches"]:
                for sa in generated(br, rr["round"]):
                    st = sa.get("status")
                    if st is None:
                        st = "insufficient" if not sa.get("citations") else "answered"
                    c[str(rr["round"])][st] += 1
    return {r: {"n": sum(v.values()), "insufficient_rate": v["insufficient"] / sum(v.values()),
                "abstain_rate": sum(v[s] for s in ABSTAIN) / sum(v.values())}
            for r, v in sorted(c.items()) if sum(v.values())}


# --------------------------------------------------------------------------- 관문 (파일럿)
def g1_checks(runs: dict, usage: dict, ledger: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    seeded = [r for r in ledger if not str(r.get("tag", "")).startswith("decompose")]
    out["1_seed_key_disjoint"] = {"status": "manual",
                                  "note": "캐시 키는 산출에 없다. 원장 nonce 의 seed 접두사만 확인한다",
                                  "ledger_rows_without_seed_prefix": sum(1 for r in seeded if not re.match(r"s\d+\|", str(r.get("nonce", ""))))}
    dec_ledger = sum(1 for r in ledger if str(r.get("tag", "")).startswith("decompose"))
    dec_usage = sum(u.get("decompose_api_calls", 0) for u in usage.values())
    out["2_decompose_api_calls"] = {"ledger": dec_ledger, "usage": dec_usage, "pass": dec_ledger == 0 and dec_usage == 0}
    n_gen = n_ff = n_exec = n_trunc = calls = repairs = fab = dropped_sent = kept_sent = 0
    inherit_total = inherit_same = 0
    fab_by_family: dict[str, int] = defaultdict(int)
    for (arm, _s), recs in runs.items():
        if parse_arm(arm)[0] == R1:
            continue
        family = ARMS[parse_arm(arm)[0]][0]
        for rec in recs.values():
            prev = {}
            for rr in rec["rounds"]:
                cur = {}
                for br in rr["branches"]:
                    sas = {sa["index"]: sa for sa in br.get("sub_answers") or []}
                    cur[br["branch"]] = sas
                    if br.get("executed"):
                        n_exec += 1
                        n_trunc += int(bool(br.get("truncated")))
                        st = br.get("answer_stats") or {}
                        dropped_sent += int(st.get("n_dropped_uncited_sentences", 0))
                        for tag, slot in ((br.get("billing") or {}).get("by_tag") or {}).items():
                            calls += int(slot.get("calls", 0))
                            repairs += int(slot.get("calls", 0)) if tag.endswith("_repair") else 0
                    gen = generated(br, rr["round"])
                    gid = {id(x) for x in gen}
                    for sa in gen:
                        n_gen += 1
                        n_ff += int(sa.get("status") == "format_fail")
                        if sa.get("status") == "answered":
                            kept_sent += len(split_sentences(sa.get("text") or ""))
                        nf = len(set(CITATION_RE.findall(sa.get("text") or "")) - set(sa.get("citations") or []))
                        fab += nf
                        fab_by_family[family] += nf
                    if rr["round"] > 0 and br["branch"] in prev:
                        for i, sa in sas.items():
                            if id(sa) in gid:
                                continue
                            p = prev[br["branch"]].get(i)
                            inherit_total += 1
                            inherit_same += int(p is not None and all(p.get(k) == sa.get(k) for k in ("text", "citations", "status")))
                prev = cur
    out["3_answer_format"] = {"format_fail_rate": (n_ff / n_gen) if n_gen else None,
                              "truncated_branch_rate": (n_trunc / n_exec) if n_exec else None,
                              "threshold": 0.01,
                              "pass": (n_gen > 0 and n_ff / n_gen <= 0.01 and (n_trunc / n_exec if n_exec else 0) <= 0.01),
                              "note": "파싱 실패·status 누락은 format_fail 로 합쳐진다. finish=length 는 재시도 후에도 잘린 branch 비율"}
    out["4_repair_calls"] = {"rate": (repairs / calls) if calls else None, "threshold": 0.02,
                             "pass": calls > 0 and repairs / calls <= 0.02}
    # "남은 문장"을 거르는 것은 status 모드(E)뿐이다. legacy(L) 파서는 일부러 바꾸지 않았고, 위조 id 는 유효 인용으로
    # 세지 않을 뿐 문장에 남는다. 그래서 판정은 E 에서 하고 L 은 보고만 한다(DEVIATIONS 부기 1)
    out["5_fabricated_ids_left"] = {"n": fab, "by_family": dict(fab_by_family),
                                    "pass": fab_by_family.get("E", 0) == 0,
                                    "scope": "E (status 모드 필터 대상). L 은 보고만"}
    den = dropped_sent + kept_sent
    out["6_dropped_uncited_sentences"] = {"rate": (dropped_sent / den) if den else None, "threshold": 0.05,
                                          "pass": den > 0 and dropped_sent / den <= 0.05}
    lin = sum(int(r["tokens_in"]) for r in ledger)
    lout = sum(int(r["tokens_out"]) for r in ledger)
    uin = sum(u.get("billed_tokens_in", 0) for u in usage.values())
    uout = sum(u.get("billed_tokens_out", 0) for u in usage.values())
    out["7_ledger_equals_usage"] = {"ledger": [lin, lout, len(ledger)], "usage": [uin, uout,
                                    sum(u.get("api_calls", 0) for u in usage.values())],
                                    "pass": (lin, lout) == (uin, uout) and len(ledger) == sum(u.get("api_calls", 0) for u in usage.values())}
    # arm 별 청구는 executed branch 의 billing 합으로 낸다 — 그 합이 usage 청구(분해 제외)와 같은지 대조한다
    bb_in = bb_out = 0
    for recs in runs.values():
        for rec in recs.values():
            for rr in rec["rounds"]:
                for br in rr["branches"]:
                    if br.get("executed"):
                        bb_in += int((br.get("billing") or {}).get("billed_in", 0))
                        bb_out += int((br.get("billing") or {}).get("billed_out", 0))
    dec_in = sum(int(r["tokens_in"]) for r in ledger if str(r.get("tag", "")).startswith("decompose"))
    dec_out = sum(int(r["tokens_out"]) for r in ledger if str(r.get("tag", "")).startswith("decompose"))
    out["7b_branch_billing_equals_usage"] = {"branch_billing": [bb_in, bb_out], "usage_minus_decompose": [uin - dec_in, uout - dec_out],
                                             "pass": (bb_in, bb_out) == (uin - dec_in, uout - dec_out)}
    out["8_minicheck_cuda_single_chunk"] = {"status": "manual", "pass": None,
                                            "note": "산출에서 알 수 없다. 실행 로그로 확인한다"}
    out["9_inherited_bytes_identical"] = {"n": inherit_total, "same": inherit_same,
                                          "rate": (inherit_same / inherit_total) if inherit_total else None,
                                          "pass": inherit_total == inherit_same}
    out["round0_identical_within_family_seed"] = round0_identity(runs)
    return out


def round0_identity(runs: dict) -> dict[str, Any]:
    """같은 family·seed 의 라운드 0 은 nonce 가 같아 arm 이 달라도 바이트가 같아야 한다."""
    groups = defaultdict(list)
    for (arm, s), recs in runs.items():
        a, model = parse_arm(arm)
        if a == R1:
            continue
        groups[(ARMS[a][0], model, s)].append(recs)
    n = diff = 0
    for _, lst in groups.items():
        if len(lst) < 2:
            continue
        ref = lst[0]
        for other in lst[1:]:
            for qid in set(ref) & set(other):
                sig = lambda rec: [(b["branch"], [(x.get("text"), x.get("citations"), x.get("status")) for x in b["sub_answers"]])
                                   for b in rec["rounds"][0]["branches"]]
                n += 1
                diff += int(sig(ref[qid]) != sig(other[qid]))
    return {"n_pairs": n, "n_differ": diff, "pass": diff == 0}


def g3_check(runs: dict) -> dict[str, Any]:
    out = {}
    tin = defaultdict(list)
    for (arm, s), recs in runs.items():
        a = parse_arm(arm)[0]
        if a not in ("E-SQ", "E-BR"):
            continue
        partial = total = 0
        for rec in recs.values():
            for rr in rec["rounds"][1:]:
                for br in rr["branches"]:
                    if not br.get("executed"):
                        continue
                    total += 1
                    tin[a].append(int(br.get("tokens_in", 0)))
                    partial += int(int(br.get("subq_executed") or 0) < len(br.get("sub_answers") or []))
        if a == "E-SQ":
            out["E-SQ_s%d" % s] = {"partial": partial, "total": total, "rate": (partial / total) if total else None}
    rates = [v["rate"] for v in out.values() if v["rate"] is not None]
    sq_in = float(np.mean(tin["E-SQ"])) if tin["E-SQ"] else None
    br_in = float(np.mean(tin["E-BR"])) if tin["E-BR"] else None
    return {"by_seed": out, "partial_rate_threshold": 0.20,
            "mean_tokens_in_per_rerun": {"E-SQ": sq_in, "E-BR": br_in},
            "pass": bool(bool(rates) and min(rates) >= 0.20 and sq_in is not None and br_in is not None and sq_in < br_in)}


def g4_check(summ: dict, n_topics_main=35, n_seeds_main=3, limit_in=15_000_000, limit_out=5_600_000,
             est_in=9_900_000, est_out=3_700_000, ledger: list[dict] | None = None) -> dict[str, Any]:
    """파일럿 실측(arm 별 토픽당 청구)을 heldout35 × 5 arm × 3 seed 로 외삽한다.
    기준: PREREG §6 G4 "1.5 × 추정 이하(gpt-5-mini 15M in / 5.6M out)", §10 추정 9.9M / 3.7M — 둘 다 보고한다.

    branch billing 외삽은 파일럿이 **자기 이전 실행의 캐시**를 맞히면(크래시 후 재개) 작게 나온다. 그래서 원장 총량
    × (본실험 토픽·seed / 파일럿 토픽·seed) 도 내고, **둘 중 큰 쪽**으로 판정한다(독립 검증 지적, 2026-11-05)."""
    per_arm = defaultdict(list)
    for (arm, s), v in summ.items():
        a, model = parse_arm(arm)
        if a == R1 or model:
            continue
        n = v["n_topics"] or 1
        per_arm[a].append((v["totals"]["billed_in"] / n, v["totals"]["billed_out"] / n,
                           v["totals"]["accounting_in"] / n, v["totals"]["accounting_out"] / n))
    ext = np.zeros(4)
    for a, rows in per_arm.items():
        ext += np.mean(np.array(rows), axis=0) * n_topics_main * n_seeds_main
    led = None
    if ledger:
        pilot_units = max((v["n_topics"] or 0) for v in summ.values()) * len({s for (_a, s) in summ})
        f = n_topics_main * n_seeds_main / pilot_units if pilot_units else None
        if f:
            led = [sum(int(r["tokens_in"]) for r in ledger) * f, sum(int(r["tokens_out"]) for r in ledger) * f]
    use = [max(ext[0], led[0]), max(ext[1], led[1])] if led else [ext[0], ext[1]]
    return {"extrapolated_billed": [float(ext[0]), float(ext[1])],
            "extrapolated_ledger": led,
            "extrapolated_used": [float(use[0]), float(use[1])],
            "extrapolated_accounting_upper": [float(ext[2]), float(ext[3])],
            "arms_used": sorted(per_arm), "limit_hard": [limit_in, limit_out],
            "limit_1p5x_estimate": [1.5 * est_in, 1.5 * est_out],
            "pass": bool(bool(per_arm) and use[0] <= limit_in and use[1] <= limit_out),
            "gpu_hours": None, "note": "GPU 시간은 산출에서 알 수 없다(수동). 청구는 CRN·실행 순서를 본실험과 같게 두고 외삽한다"}


# --------------------------------------------------------------------------- 전체
def analyze(run_dir: Path, *, sup: Support | None = None, pilot: bool = False, n_boot: int = N_BOOT) -> dict[str, Any]:
    run_dir = Path(run_dir)
    sup = sup or Support()
    raw = load_run(run_dir)
    runs = with_r1(raw)
    if sup.available:
        sup.prime(sa for recs in runs.values() for rec in recs.values()
                  for rr in rec["rounds"] for br in rr["branches"] for sa in br.get("sub_answers") or [])
    summ = {k: armseed_summary(v, sup) for k, v in runs.items()}
    usage = load_usage(run_dir)
    ledger = load_ledger(run_dir)
    by_nonce = defaultdict(lambda: [0, 0, 0])
    for r in ledger:
        m = re.match(r"(s\d+\|[A-Z]*)\|", str(r.get("nonce", "")))
        k = m.group(1) if m else "unseeded"
        by_nonce[k][0] += int(r["tokens_in"]); by_nonce[k][1] += int(r["tokens_out"]); by_nonce[k][2] += 1

    def comps(filt):
        return {
            "savings": {"%s_vs_%s" % (a, b): {m: savings_compare(summ, a, b, m, filt, n_boot) for m in METRICS}
                        for a, b in SAVINGS},
            "noninferiority": {"%s_vs_%s" % (a, b): {
                "coverage": noninf_compare(summ, a, b, "coverage", DELTA_COV, DELTA_COV_SENS, filt, n_boot),
                "grounding": noninf_compare(summ, a, b, "grounding", DELTA_GRD, (), filt, n_boot)}
                for a, b in NONINF},
        }

    res: dict[str, Any] = {
        "run_dir": str(run_dir), "n_boot": n_boot, "boot_seed": BOOT_SEED,
        "sentence_support": "minicheck_rescored" if sup.available else "unavailable",
        "notes": [
            "accounting_tokens = executed branch 의 tokens_in + tokens_out (물려받은 branch 제외)",
            "billed_tokens = executed branch 의 billing.billed_in + billed_out. 원장 행에는 arm 이 없어 arm 별 분해는 branch billing 으로 한다",
            "CRN: 같은 family·seed 에서 뒤에 도는 arm 은 앞 arm 의 캐시를 맞혀 청구가 작다 — 청구 비교는 실행 순서에 의존한다",
            "E-R1 = E-GR 라운드 0 (실행하지 않음)",
            "coverage = 최종 라운드 sub-answer 중 유효 인용이 있는 비율(질의 단위 풀링). grounding = 문장 가중 풀링",
            "R1-8 전이표는 회계 분해이고 인과가 아니다. 토큰은 branch 실행 토큰을 새로 만든 sub-answer 수로 균등 분할했다",
            "legacy(L) arm 은 status 가 없다. 빈 답·유효 인용 없는 답을 전이표와 기권률에서 insufficient 로 센다 — L 이 기권 status 를 냈다는 뜻이 아니다",
        ],
        "arm_seed": {"%s·s%d" % k: {kk: vv for kk, vv in v.items() if kk != "per_query"} for k, v in sorted(summ.items())},
        "per_query": {"%s·s%d" % k: v["per_query"] for k, v in sorted(summ.items())},
        "ledger": {"rows": len(ledger), "by_seed_family": {k: {"billed_in": v[0], "billed_out": v[1], "api_calls": v[2]}
                                                           for k, v in sorted(by_nonce.items())}},
        "usage": {"%s·s%d" % k: v for k, v in sorted(usage.items())},
        "all_topics": comps(lambda t: True),
        "sensitivity_30": comps(lambda t: t not in PILOT_DEV),
        "transitions": {"%s·s%d" % k: transitions(v, sup) for k, v in sorted(runs.items())},
        "gaming": {"abstention_by_round": {"%s·s%d" % k: abstention_by_round(v) for k, v in sorted(runs.items())}},
    }
    res["gaming"]["insufficient_to_answered"] = {k: v["insufficient_to_answered"] for k, v in res["transitions"].items()}
    if sup.available:
        res["rescore_consistency"] = rescore_consistency(runs, sup)
        if sup.missing_chunks:
            res["rescore_missing_chunks"] = sorted(sup.missing_chunks)[:50]
    if pilot:
        res["gates"] = {"G1": g1_checks(raw, usage, ledger), "G3": g3_check(raw), "G4": g4_check(summ, ledger=ledger),
                        "G2": "사람 라벨(P-H) 필요 — r2_sheets.py", "G5": "P-H 라벨 후 r2_sheets.g5_from_labels"}
    return res


def rescore_consistency(runs: dict, sup: Support) -> dict[str, Any]:
    """offline 재채점 grounding 이 루프 안 evidence.grounding 과 같은지 (PREREG §8 'offline 재채점')."""
    n = bad = 0
    examples = []
    for (arm, s), recs in runs.items():
        if parse_arm(arm)[0] == R1:
            continue
        for qid, rec in recs.items():
            for rr in rec["rounds"]:
                for br in rr["branches"]:
                    flags = [f for sa in br.get("sub_answers") or [] for f in sup.sentence_flags(sa) or []]
                    g = (sum(flags) / len(flags)) if flags else 0.0
                    n += 1
                    if abs(round(g, 6) - float(br["evidence"]["grounding"])) > 1e-6:
                        bad += 1
                        if len(examples) < 10:
                            examples.append([arm, s, qid, rr["round"], br["branch"], g, br["evidence"]["grounding"]])
    return {"n_branch_records": n, "n_mismatch": bad, "pass": bad == 0, "examples": examples}


# --------------------------------------------------------------------------- 마크다운
def _f(x, d=3):
    return "—" if x is None else (("%." + str(d) + "f") % x if isinstance(x, float) else str(x))


def to_markdown(res: dict) -> str:
    L = ["# R2 분석 요약 — `%s`" % res["run_dir"], "",
         "자동 생성(`experiments/r2_analyze.py`). 수치는 모두 같은 디렉터리의 `ANALYSIS.json` 에서 왔다. "
         "부트스트랩 %d회, seed %d, topic × seed 2단 계층. 문장 지지: **%s**." % (res["n_boot"], res["boot_seed"], res["sentence_support"]), ""]
    L += ["## 형식에서 오는 제약", ""] + ["- " + n for n in res["notes"]] + [""]
    L += ["## arm × seed", "", "| arm·seed | 토픽 | 회계 토큰 | 청구 토큰 | api | branch-round | subq 실행 | verify 통과 (g / c) | 최종 동결 | 생성 insufficient / format_fail / dropped |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["arm_seed"].items():
        t, g = v["totals"], v["generated_rates"]
        L.append("| %s | %d | %d | %d | %d | %d | %d | %d (%d / %d) | %d | %s / %s / %s |" % (
            k, v["n_topics"], t["accounting_tokens"], t["billed_tokens"], t["api_calls"], t["branch_rounds"],
            t["subq_executed"], v["verify"]["passed"], v["verify"]["pass_grounding"], v["verify"]["pass_coverage"],
            v["frozen_branches_final"], _f(g["insufficient"]), _f(g["format_fail"]), _f(g["dropped"])))
    for scope, title in (("all_topics", "전체 토픽"), ("sensitivity_30", "30토픽 민감도 (t000~t004 제외)")):
        c = res[scope]
        L += ["", "## 절감 — %s" % title, "", "절감 = 1 − ΣA/ΣB. 주장 조건: seed 부호 일치 ∧ 95% CI 가 0 제외.", "",
              "| 비교 | 지표 | 점추정 | 95% CI | seed별 | 주장 |", "|---|---|---|---|---|---|"]
        for name, ms in c["savings"].items():
            for m, s in ms.items():
                if not s.get("available"):
                    L.append("| %s | %s | 짝 없음 | | | |" % (name, m)); continue
                L.append("| %s | %s | %s | [%s, %s] | %s | %s |" % (
                    name, m, _f(s["point"]), _f(s["ci95"][0]), _f(s["ci95"][1]),
                    ", ".join("s%s %s" % (k, _f(v)) for k, v in s["per_seed"].items()),
                    "보고만" if s["claim"] is None else ("예" if s["claim"] else "아니오")))
        L += ["", "## 자동 비열등성 — %s" % title, "", "차이 = A − B (질의 단위 평균). 90% CI 하한 > −δ.", "",
              "| 비교 | 지표 | 차이 | 90% CI | δ | 비열등 | 동등 |", "|---|---|---|---|---|---|---|"]
        for name, ms in c["noninferiority"].items():
            for m, s in ms.items():
                if not s.get("available"):
                    L.append("| %s | %s | 짝 없음 | | | | |" % (name, m)); continue
                L.append("| %s | %s | %s | [%s, %s] | %s | %s | %s |" % (
                    name, m, _f(s["diff"]), _f(s["ci90"][0]), _f(s["ci90"][1]), s["delta"],
                    "예" if s["noninferior"] else "아니오", "예" if s["equivalent"] else "아니오"))
    L += ["", "## R1-8 상태 전이 — 회계 분해 (인과 아님)", ""]
    for k, t in res["transitions"].items():
        if not t["n_reexecuted_units"]:
            continue
        fr = t["format_recovery_tokens"]
        fr_txt = _f(fr, 0) if fr is not None else "산출 불가(--rescore 필요; format_fail→cited %s)" % _f(t["format_fail_to_cited_tokens"], 0)
        L += ["### %s" % k, "", "재실행 단위 %d. 형식 회복 몫 %s 토큰 (상한 %d). 회복 불가 지출 %s 토큰. insufficient→answered %s." % (
            t["n_reexecuted_units"], fr_txt, t["format_recovery_upper_bound"]["sum"],
            _f(t["unrecoverable_tokens"], 0), _f(t["insufficient_to_answered"]["rate"])), "",
            "| r → r+1 | " + " | ".join(t["states"]) + " |", "|---|" + "---|" * len(t["states"])]
        for frm, row in t["counts"].items():
            L.append("| %s | " % frm + " | ".join(str(row.get(s, 0)) for s in t["states"]) + " |")
        L.append("")
    if "gates" in res:
        G = res["gates"]
        L += ["## 파일럿 관문", "", "| 관문 | 항목 | 통과 |", "|---|---|---|"]
        for k, v in G["G1"].items():
            L.append("| G1 | %s | %s%s |" % (k, v.get("pass") if isinstance(v, dict) else v,
                     (" (%s)" % v["by_family"]) if isinstance(v, dict) and "by_family" in v else ""))
        L.append("| G3 | SQ 부분 재실행 ≥ 20%% ∧ SQ 입력 < BR | %s |" % G["G3"]["pass"])
        L.append("| G4 | 외삽 청구(branch billing·원장 중 큰 쪽) %s / %s ≤ %s / %s | %s |" % (
            _f(G["G4"]["extrapolated_used"][0], 0), _f(G["G4"]["extrapolated_used"][1], 0),
            G["G4"]["limit_hard"][0], G["G4"]["limit_hard"][1], G["G4"]["pass"]))
        L += ["", "G2·G5 는 사람 라벨이 필요하다. 효과 관문은 없다(PREREG §6)."]
    return "\n".join(L) + "\n"


def write_outputs(res: dict, run_dir: Path) -> tuple[Path, Path]:
    j, m = Path(run_dir) / "ANALYSIS.json", Path(run_dir) / "ANALYSIS.md"
    io.open(j, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    io.open(m, "w", encoding="utf-8").write(to_markdown(res))
    return j, m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R2 사후 분석 (LLM 호출 0)")
    ap.add_argument("run_dir")
    ap.add_argument("--pilot", action="store_true", help="G1/G3/G4 관문 점검을 함께 낸다")
    ap.add_argument("--rescore", action="store_true", help="문장 단위 MiniCheck 재채점 (GPU, LLM 0)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = ROOT / run_dir
    sup = Support()
    if args.rescore:
        from lfm.config import Config, load_config
        from lfm.estimator import load_estimator
        base = load_config(ROOT / "config.yaml")
        d = base.as_dict(); d.setdefault("estimator", {})["kind"] = "minicheck_doc"
        sup = Support(load_estimator(Config(d, source="r2_analyze")), load_chunk_text(ROOT / base["paths.index_dir"]))
    res = analyze(run_dir, sup=sup, pilot=args.pilot, n_boot=args.n_boot)
    j, m = write_outputs(res, run_dir)
    print("-> %s\n-> %s" % (j, m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
