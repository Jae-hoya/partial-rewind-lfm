"""R2 사후 추가 분석 — API 0원, 새 실행 0. 등록된 코드(r2_analyze, r2_human)는 고치지 않고 import 만 한다.

  python experiments/r2_extra.py                 # MiniCheck(GPU) 로 H4 지지 채점 포함 (~10분)
  python experiments/r2_extra.py --no-support    # 빠른 경로: H4 는 사람 라벨만(human_only 와 대조)

블록 (심사 답변서 ③ 항목 대응)
  A  H4 사람 UA — PREREG §8 머리의 topic × seed 2단 부트스트랩(등록 서술)으로 다시 낸다. 1단(r2_human.h4_compare,
     §8 1 "topic 군집")은 HUMAN_RESULT.json 과 바이트 수준으로 같은지 assert 하고 나란히 둔다. 두 라벨러(동료·저자).
  B  일치도 보강 — 원 일치율, Cohen κ(HUMAN_RESULT 와 assert), PABAK, Gwet AC1, 범주별 주변분포.
     V′50(저자 vs 동료), H3 앵커 25(저자 vs 동료), H4 36(저자 vs 동료).
  C  grounding 기권 처리 민감도 — 0 규약(ANALYSIS.json 재현 assert), 양쪽 무문장 짝 제외, 한쪽이라도 무문장 제외.
     셀(topic, seed) 단위 제외와 토픽 단위 제외를 모두 낸다. DEVIATIONS 15 수치를 어느 변형이 재현하는지 표시한다.
  D  라운드 절단 비용–품질 곡선 — 관측 궤적을 라운드 k 에서 자른 **에뮬레이션**(별도 실행 아님).
     k = 최대에서 ANALYSIS.json 최종값을 재현하는지 assert.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import r2_analyze as RA  # noqa: E402
import r2_human as RH  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

HO = ROOT / "results/r2/heldout35"
NANO = ROOT / "results/r2/heldout35_nano"
OUT = ROOT / "results/r2/extra"
LAB_COLLEAGUE = RH.LABELS
LAB_AUTHOR_MERGED = ROOT / "results/r2/human_author/labels_merged_authorH4.csv"
LAB_AUTHOR_H4 = ROOT / "results/r2/human_author/labels_h4_filled_author.csv"
HR_COLLEAGUE = ROOT / "results/r2/human/HUMAN_RESULT.json"
HR_AUTHOR = ROOT / "results/r2/human_author/HUMAN_RESULT.json"
SEP = "·"   # ANALYSIS.json 키 "arm·s1"


def jnorm(x):
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


def jload(p: Path):
    return json.load(io.open(p, encoding="utf-8"))


def pct(x, q):
    return RH.pct(np.asarray(x, float), q)


# =========================================================================== A. H4 2단 부트스트랩
def h4_two_stage(slots, meta, a: str, b: str, excl_items=(), n_boot=RA.N_BOOT, seed=RA.BOOT_SEED) -> dict:
    """PREREG §8 머리: topic × seed 2단 계층. 토픽을 복원추출 → 뽑힌 토픽마다 seed(1..3)를 복원추출.
    슬롯은 seed × 관점 층에서 뽑혀 (토픽, seed) 셀이 빌 수 있다 — 빈 셀은 분자·분모에 0 을 더한다.
    Hájek: Σ w·ua / Σ w (w = N_h / n_h). 슬롯 제외 규칙은 h4_compare 와 같다(두 arm 의 문항만 본다)."""
    ss = [s for s in sorted(slots) if not ({slots[s][a]["item"], slots[s][b]["item"]} & set(excl_items))]
    topics = sorted({s[1] for s in ss})
    seeds = sorted({k["seed"] for k in RH.read_csv(RH.SEALED / "_key_h4.csv")}, key=int)  # 설계 seed 전부
    ti_of, si_of = {t: i for i, t in enumerate(topics)}, {s: j for j, s in enumerate(seeds)}
    cols = ("w", "wa", "wb", "wne", "wane", "wbne")
    cell = {c: np.zeros((len(topics), len(seeds))) for c in cols}
    for s in ss:
        w = meta[s]["N_h"] / meta[s]["n_h"]
        ua, ub = slots[s][a]["ua"], slots[s][b]["ua"]
        ne = float(slots[s][a]["nonempty"] or slots[s][b]["nonempty"])
        i, j = ti_of[s[1]], si_of[s[0]]
        for c, v in zip(cols, (w, w * ua, w * ub, w * ne, w * ua * ne, w * ub * ne)):
            cell[c][i, j] += v
    ti, si = RA.hier_indices(len(topics), len(seeds), n_boot, seed)
    S = {c: m[ti[:, :, None], si].sum(axis=(1, 2)) for c, m in cell.items()}
    ra, rb = S["wa"] / S["w"], S["wb"] / S["w"]
    diff = ra - rb
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ra / rb
        cond = (S["wane"] - S["wbne"]) / S["wne"]
    W = cell["w"].sum()
    pa, pb = float(cell["wa"].sum() / W), float(cell["wb"].sum() / W)
    lo, hi = pct(diff, [5, 95])
    rlo, rhi = pct(ratio, [5, 95])
    return {
        "a": a, "b": b, "bootstrap": "topic x seed two-stage (PREREG §8 header), n_boot %d, seed %d" % (n_boot, seed),
        "n_slots": len(ss), "n_topics": len(topics), "seeds": seeds,
        "n_empty_cells": int((cell["w"] == 0).sum()), "n_cells": int(cell["w"].size),
        "ua_a": pa, "ua_b": pb, "diff": pa - pb, "ci90": [lo, hi], "delta": RH.DELTA,
        "noninferior": bool(lo is not None and lo > -RH.DELTA),
        "equivalent": bool(lo is not None and lo > -RH.DELTA and hi < RH.DELTA),
        "sensitivity": {str(d): {"noninferior": bool(lo > -d), "equivalent": bool(lo > -d and hi < d)} for d in RH.SENS},
        "ratio": (pa / pb) if pb else None, "ratio_ci90": [rlo, rhi], "ratio_bar": RH.RATIO_BAR,
        "ratio_pass": bool(rlo is not None and rlo >= RH.RATIO_BAR),
        "ratio_nan_resamples": int((~np.isfinite(ratio)).sum()),
        "conditional_diff_ci90": pct(cond, [5, 95]),
    }


def _slim(r: dict) -> dict:
    keys = ("n_slots", "n_topics", "ua_a", "ua_b", "diff", "ci90", "noninferior", "equivalent", "sensitivity",
            "ratio", "ratio_ci90", "ratio_pass", "ratio_nan_resamples")
    return {k: r.get(k) for k in keys}


def block_a(use_support: bool, n_boot: int) -> dict:
    keys = RH.read_csv(RH.SEALED / "_key_h4.csv")
    note = jload(RH.SEALED / "_note_heldout_blinding.json")
    inferable = sorted(set(note["h4_inferable_from_h3"]) | set(note["h4_inferable_from_h5"]))
    hr = {"colleague": jload(HR_COLLEAGUE), "author": jload(HR_AUTHOR)}
    labs = {"colleague": RH.load_labels(LAB_COLLEAGUE), "author": RH.load_labels(LAB_AUTHOR_MERGED)}
    support, sinfo = RH.final_support(keys, use_support)          # 한 번만 — 키에만 의존한다
    asserts = []
    if use_support:
        for who in hr:
            ref = hr[who]["h4_support"]
            assert sinfo["n_supported"] == ref["n_supported"], (who, sinfo["n_supported"], ref["n_supported"])
            assert jnorm(sinfo["sentences"]) == jnorm(ref["sentences"]), who
        asserts.append("final_support n_supported %d 및 문장별 판정이 두 HUMAN_RESULT 와 같다" % sinfo["n_supported"])
    out: dict[str, Any] = {"support": "minicheck_doc (r2_human.final_support)" if use_support else "none (human only)",
                           "n_supported": sinfo.get("n_supported"), "inferable_items": inferable, "by_labeller": {}}
    for who, lab in labs.items():
        slots, meta = RH.h4_table(lab, support)
        res = {}
        for a in ("E-SQ", RA.R1):
            name = "%s_vs_E-GR" % a
            one_all = RH.h4_compare(slots, meta, a, "E-GR", n_boot=n_boot)
            one_ex = RH.h4_compare(slots, meta, a, "E-GR", inferable, n_boot=n_boot)
            if n_boot == RA.N_BOOT:
                if use_support:
                    ref = hr[who]["h4"]["comparisons"][name]
                    assert jnorm(one_all) == jnorm(ref["all"]), (who, name, "all")
                    assert jnorm(one_ex) == jnorm(ref["excl_inferable"]), (who, name, "excl")
                else:
                    assert jnorm(one_all) == jnorm(hr[who]["h4"]["human_only"][name]), (who, name, "human_only")
                asserts.append("%s %s 1단 결과 == HUMAN_RESULT" % (who, name))
            res[name] = {
                "all": {"one_stage_topic_cluster": _slim(one_all),
                        "two_stage_topic_x_seed": h4_two_stage(slots, meta, a, "E-GR", (), n_boot)},
                "excl_inferable": {"one_stage_topic_cluster": _slim(one_ex),
                                   "two_stage_topic_x_seed": h4_two_stage(slots, meta, a, "E-GR", inferable, n_boot)},
            }
        if use_support:   # 사람 라벨만(자동 절반 없이)도 2단으로
            slots_h, meta_h = RH.h4_table(lab, {k: None for k in support})
            res["human_only"] = {"%s_vs_E-GR" % a: h4_two_stage(slots_h, meta_h, a, "E-GR", (), n_boot)
                                 for a in ("E-SQ", RA.R1)}
        out["by_labeller"][who] = res
    out["asserts"] = asserts
    return out


# =========================================================================== B. 일치도
def agreement(x: list[str], y: list[str], cats: list[str], pos: str, n_boot: int) -> dict:
    kb = RH.kappa_block(x, y, cats, pos, n_boot=n_boot)
    n = len(x)

    def stats(xx, yy, cc):
        k = len(cc)
        po = sum(a == b for a, b in zip(xx, yy)) / n
        cx, cy = Counter(xx), Counter(yy)
        pi = {c: (cx[c] + cy[c]) / (2 * n) for c in cc}
        pe_ac1 = sum(p * (1 - p) for p in pi.values()) / (k - 1)
        return {"k": k, "p_o": po, "kappa": RH.cohen_kappa(xx, yy, cc),
                "pabak": (k * po - 1) / (k - 1),
                "gwet_ac1": (po - pe_ac1) / (1 - pe_ac1) if pe_ac1 != 1 else None, "pe_ac1": pe_ac1,
                "marginals": {c: {"author": cx[c], "colleague": cy[c],
                                  "author_p": cx[c] / n, "colleague_p": cy[c] / n} for c in cc}}

    xb = ["pos" if v == pos else "neg" for v in x]
    yb = ["pos" if v == pos else "neg" for v in y]
    s3, s2 = stats(x, y, cats), stats(xb, yb, ["pos", "neg"])
    assert abs(s3["kappa"] - kb["kappa3"]) < 1e-12 and abs(s2["kappa"] - kb["kappa2"]) < 1e-12
    return {"n": n, "categories": cats, "binary_positive": pos, "three_class": s3, "binary": s2,
            "kappa3_ci95": kb["kappa3_ci95"], "kappa2_ci95": kb["kappa2_ci95"],
            "confusion_rows_author_cols_colleague": kb["confusion_rows_author_cols_colleague"],
            "_kb": kb}


def block_b(n_boot: int) -> dict:
    col = RH.load_labels(LAB_COLLEAGUE)
    hr = jload(HR_COLLEAGUE)["kappa"]
    asserts = []
    # V′50 — kappa_report 와 같은 짝
    key = RH.read_csv(RH.V / "c50_sealed/_key_vc50.csv")
    auth = RH.vprime_author()
    v = agreement([auth[k["source_item"]] for k in key], [col[k["item_id"]] for k in key],
                  ["지지", "불지지", "판단불가"], "지지", n_boot)
    # H3 앵커 25
    k3 = [k for k in RH.read_csv(RH.SEALED / "_key_h3.csv") if k["arm"] == "H2-anchor"]
    d18 = {(r["set"], r["item_id"]): r["human_label"].strip() for r in RH.read_csv(RH.D18 / "labels_all_filled.csv")}
    anc = agreement([d18[("h2hoA", k["source"].split(":", 1)[1])] for k in k3], [col[k["item_id"]] for k in k3],
                    ["답함", "일부", "답안함"], "답함", n_boot)
    for name, blk in (("vprime50", v), ("anchor25", anc)):
        kb, ref = blk.pop("_kb"), hr[name]
        for f in ("kappa3", "kappa2", "agree3", "agree2", "n"):
            assert kb[f] == ref[f], (name, f, kb[f], ref[f])
        if n_boot == RA.N_BOOT:
            assert jnorm(kb["kappa3_ci95"]) == jnorm(ref["kappa3_ci95"]), name
        asserts.append("%s κ3 %.4f κ2 %.4f 일치율 == HUMAN_RESULT" % (name, kb["kappa3"], kb["kappa2"]))
    # H4 36 — 저자 원본 CSV vs 동료 묶음 h4_a
    a_rows = {r["item_id"]: r["human_label"].strip() for r in RH.read_csv(LAB_AUTHOR_H4) if r["set"] == "h4_a"}
    merged = RH.load_labels(LAB_AUTHOR_MERGED)
    assert all(merged[i] == a_rows[i] for i in a_rows), "merged CSV 의 H4 가 저자 원본과 다르다"
    ids = sorted(a_rows)
    assert len(ids) == 36 and all(i in col for i in ids)
    h4 = agreement([a_rows[i] for i in ids], [col[i] for i in ids], ["답함", "일부", "답안함"], "답함", n_boot)
    h4.pop("_kb")
    h4["disagreements"] = [{"item": i, "author": a_rows[i], "colleague": col[i]} for i in ids if a_rows[i] != col[i]]
    h4["note"] = "HUMAN_RESULT 에 H4 κ 저장값은 없다. DEVIATIONS 21: 33/36, κ 0.36 / 0.36 과 대조(assert 아님)"
    return {"vprime50": v, "anchor25": anc, "h4_36": h4, "asserts": asserts,
            "formulas": {"pabak": "(k·p_o − 1)/(k − 1)",
                         "gwet_ac1": "(p_o − p_e)/(1 − p_e), p_e = Σ_c π_c(1 − π_c)/(k − 1), π_c = 두 라벨러 주변 비율의 평균",
                         "binary": "binary_positive 범주 vs 나머지 (kappa_block 과 같다)"}}


# =========================================================================== C. grounding 기권 민감도
def n_sentences_final(rec: dict) -> int:
    fr = RA.final_round(rec)
    return sum(len(split_sentences(sa.get("text") or "")) for br in fr["branches"] for sa in br.get("sub_answers") or [])


def masked_two_stage(A, B, M, n_boot, seed=RA.BOOT_SEED):
    """셀 단위 제외: M(valid) 인 셀만 평균. 2단 추출은 hier_indices 그대로."""
    D = np.where(M, A - B, 0.0)
    ti, si = RA.hier_indices(A.shape[0], A.shape[1], n_boot, seed)
    num = D[ti[:, :, None], si].sum(axis=(1, 2))
    den = M.astype(float)[ti[:, :, None], si].sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = num / den
    return float(D.sum() / M.sum()), boots


def block_c(n_boot: int) -> dict:
    out: dict[str, Any] = {"definition": {
        "zero_rule": "최종 출력에 문장이 없는 질의 grounding = 0 (r2_analyze 1차, DEVIATIONS 15)",
        "drop_both": "두 arm 모두 무문장인 (topic, seed) 셀을 뺀다",
        "drop_either": "어느 한 arm 이라도 무문장인 (topic, seed) 셀을 뺀다",
        "cell": "셀 단위로 빼고 2단 부트스트랩 추출 뒤 남은 셀만 평균",
        "topic": "그 조건이 한 seed 에서라도 성립하는 토픽을 세 seed 모두에서 뺀다(r2_analyze.noninf_compare 그대로, 균형)"},
        "runs": {}, "asserts": []}
    dev15 = {"E-SQ_vs_E-GR": (0.082, 0.033, 0.134), "E-R1_vs_E-GR": (-0.059, -0.122, 0.006)}
    for run_dir, comps in ((HO, [("E-SQ", "E-GR"), (RA.R1, "E-GR"), ("E-SQ", "E-BR"), ("L-BR", "L-GR")]),
                           (NANO, [("E-SQ", "E-GR"), (RA.R1, "E-GR")])):
        an = jload(run_dir / "ANALYSIS.json")
        runs = RA.with_r1(RA.load_run(run_dir))
        summ = {}
        for key, pq in an["per_query"].items():
            arm, s = key.rsplit(SEP, 1)
            summ[(arm, int(s[1:]))] = {"per_query": pq}
        nsent = {(arm, s): {q: n_sentences_final(rec) for q, rec in recs.items()} for (arm, s), recs in runs.items()}
        # 0 규약의 무문장 ↔ grounding 0 이어야 한다 (문장이 있는데 0 인 질의는 있을 수 있다)
        for k in summ:
            for q, n in nsent[k].items():
                if n == 0:
                    assert summ[k]["per_query"][q]["grounding"] == 0.0, (k, q)
        model = sorted({RA.parse_arm(a)[1] or "" for (a, _s) in summ})
        assert len(model) == 1
        tag = ("@" + model[0]) if model[0] else ""
        res_run: dict[str, Any] = {"empty_counts": {}, "comparisons": {}}
        for (arm, s) in sorted(nsent):
            res_run["empty_counts"]["%s%ss%d" % (arm, SEP, s)] = sum(1 for n in nsent[(arm, s)].values() if n == 0)
        for a0, b0 in comps:
            a, b = a0 + tag, b0 + tag
            name = "%s_vs_%s%s" % (a0, b0, tag)
            if not any(k[0] == a for k in summ) or not any(k[0] == b for k in summ):
                continue
            entry: dict[str, Any] = {}
            for scope, filt in (("all_topics", lambda t: True), ("sensitivity_30", lambda t: t not in RA.PILOT_DEV)):
                zr = RA.noninf_compare(summ, a, b, "grounding", RA.DELTA_GRD, (), filt, n_boot)
                reg = an[scope]["noninferiority"].get(name, {}).get("grounding")
                if reg is not None and n_boot == RA.N_BOOT:
                    assert jnorm(zr) == jnorm(reg), (run_dir.name, scope, name)
                    out["asserts"].append("%s %s %s 0 규약 == ANALYSIS.json" % (run_dir.name, scope, name))
                topics, seeds = RA._common(summ, a, b, filt)
                A = np.array([[summ[(a, s)]["per_query"][t]["grounding"] for s in seeds] for t in topics])
                B = np.array([[summ[(b, s)]["per_query"][t]["grounding"] for s in seeds] for t in topics])
                Ea = np.array([[nsent[(a, s)][t] == 0 for s in seeds] for t in topics])
                Eb = np.array([[nsent[(b, s)][t] == 0 for s in seeds] for t in topics])
                sc: dict[str, Any] = {"zero_rule": {k: zr[k] for k in ("n_topics", "seeds", "diff", "ci90", "delta",
                                                                       "noninferior", "equivalent", "per_seed")},
                                      "zero_rule_reproduces_ANALYSIS": reg is not None}
                for var, drop in (("drop_both", Ea & Eb), ("drop_either", Ea | Eb)):
                    M = ~drop
                    pt, boots = masked_two_stage(A, B, M, n_boot)
                    lo, hi = pct(boots, [5, 95])
                    cell = {"point": pt, "ci90": [lo, hi], "noninferior": bool(lo > -RA.DELTA_GRD),
                            "n_cells_kept": int(M.sum()), "n_cells": int(M.size),
                            "nan_resamples": int((~np.isfinite(boots)).sum()),
                            "dropped_by_seed": {str(s): int(drop[:, j].sum()) for j, s in enumerate(seeds)},
                            "per_seed": {str(s): (float((A[M[:, j], j] - B[M[:, j], j]).mean()) if M[:, j].any() else None)
                                         for j, s in enumerate(seeds)}}
                    bad_t = {t for i, t in enumerate(topics) if drop[i].any()}
                    tp = RA.noninf_compare(summ, a, b, "grounding", RA.DELTA_GRD, (),
                                           lambda t, f=filt, bt=bad_t: f(t) and t not in bt, n_boot)
                    topic = {k: tp.get(k) for k in ("n_topics", "diff", "ci90", "noninferior", "per_seed")} \
                        if tp.get("available") else {"available": False}
                    sc[var] = {"cell": cell, "topic": topic}
                entry[scope] = sc
            # 이탈 15 수치 재현 표시
            if run_dir == HO and name in dev15:
                tgt = dev15[name]
                hits, cand = [], {}
                for var in ("drop_both", "drop_either"):
                    for lvl in ("cell", "topic"):
                        r = entry["all_topics"][var][lvl]
                        if r.get("available") is False:
                            continue
                        vals = (r.get("point", r.get("diff")), r["ci90"][0], r["ci90"][1])
                        cand["%s/%s" % (var, lvl)] = {"values": list(vals),
                                                      "abs_diff": [abs(v - t) for v, t in zip(vals, tgt)]}
                        if all(abs(round(v, 3) - t) < 1e-9 for v, t in zip(vals, tgt)):
                            hits.append("%s/%s" % (var, lvl))
                best = min(cand, key=lambda c: max(cand[c]["abs_diff"]))
                entry["deviations15"] = {"target": list(tgt), "reproduced_by_3dp": hits, "closest": best,
                                         "closest_values": cand[best]["values"],
                                         "closest_max_abs_diff": max(cand[best]["abs_diff"]), "candidates": cand}
            res_run["comparisons"][name] = entry
        out["runs"][run_dir.name] = res_run
    return out


# =========================================================================== D. 라운드 절단 비용–품질
def truncate(rec: dict, k: int) -> dict:
    r = copy.deepcopy(rec)
    r["rounds"] = r["rounds"][:k + 1]
    r["n_rounds"] = len(r["rounds"])
    return r


def query_quality(rec: dict) -> dict:
    """최종 라운드 sub-answer 의 기권 지표. status_insufficient = armseed_summary.final_rates 와 같은 정의,
    abstain_any = sa_state ∈ ABSTAIN (legacy 는 빈 답·무인용 → insufficient, r2_analyze 규약)."""
    subs = [sa for br in RA.final_round(rec)["branches"] for sa in br.get("sub_answers") or []]
    sup = RA.Support()
    return {"n_sub": len(subs), "n_status_ins": sum(1 for sa in subs if (sa.get("status") or "legacy") == "insufficient"),
            "n_abstain_any": sum(1 for sa in subs if RA.sa_state(sa, sup) in RA.ABSTAIN)}


def block_d(n_boot: int) -> dict:
    out: dict[str, Any] = {"runs": {}, "asserts": [], "max_abs_grounding_dev_vs_ANALYSIS": 0.0}
    for run_dir in (HO, NANO):
        an = jload(run_dir / "ANALYSIS.json")
        raw = RA.load_run(run_dir)
        arms = sorted({a for (a, _s) in raw})
        res_run = {}
        for arm in arms:
            seeds = sorted(s for (a, s) in raw if a == arm)
            topics = sorted(set.intersection(*(set(raw[(arm, s)]) for s in seeds)))
            kmax = max(len(rec["rounds"]) for s in seeds for rec in raw[(arm, s)].values()) - 1
            curve = []
            for k in range(kmax + 1):
                M = {m: np.zeros((len(topics), len(seeds))) for m in
                     ("accounting_tokens", "coverage", "grounding", "n_sub", "n_status_ins", "n_abstain_any", "trunc")}
                per_seed = {}
                for j, s in enumerate(seeds):
                    tr = {q: truncate(raw[(arm, s)][q], k) for q in topics}
                    summ = RA.armseed_summary(tr, RA.Support())
                    for i, q in enumerate(topics):
                        pq = summ["per_query"][q]
                        qq = query_quality(tr[q])
                        M["accounting_tokens"][i, j] = pq["accounting_tokens"]
                        M["coverage"][i, j] = pq["coverage"]
                        M["grounding"][i, j] = pq["grounding"]
                        for f in ("n_sub", "n_status_ins", "n_abstain_any"):
                            M[f][i, j] = qq[f]
                        M["trunc"][i, j] = float(len(raw[(arm, s)][q]["rounds"]) > k + 1)
                    per_seed[str(s)] = {"mean_accounting_tokens": float(M["accounting_tokens"][:, j].mean()),
                                        "mean_coverage": summ["mean_coverage"], "mean_grounding": summ["mean_grounding"],
                                        "final_insufficient_rate": summ["final_rates"]["insufficient"]}
                    if k == kmax:   # 최종값 재현
                        key = "%s%ss%d" % (arm, SEP, s)
                        ref_q, ref_s = an["per_query"][key], an["arm_seed"][key]
                        for i, q in enumerate(topics):
                            assert summ["per_query"][q]["accounting_tokens"] == ref_q[q]["accounting_tokens"], (key, q)
                            assert summ["per_query"][q]["coverage"] == ref_q[q]["coverage"], (key, q)
                            dev = abs(summ["per_query"][q]["grounding"] - ref_q[q]["grounding"])
                            out["max_abs_grounding_dev_vs_ANALYSIS"] = max(out["max_abs_grounding_dev_vs_ANALYSIS"], dev)
                            assert dev < 1e-6, (key, q, dev)
                        assert summ["final_rates"] == ref_s["final_rates"], key
                        assert summ["totals"]["accounting_tokens"] == ref_s["totals"]["accounting_tokens"], key
                        assert summ["mean_coverage"] == ref_s["mean_coverage"], key
                ti, si = RA.hier_indices(len(topics), len(seeds), n_boot, RA.BOOT_SEED)
                pick = lambda X: X[ti[:, :, None], si]
                row = {"k": k, "n_queries_cut_short": int(M["trunc"].sum()), "per_seed": per_seed}
                for m in ("accounting_tokens", "coverage", "grounding"):
                    row["mean_" + m] = float(M[m].mean())
                    row["mean_" + m + "_ci90"] = pct(pick(M[m]).mean(axis=(1, 2)), [5, 95])
                for f, nm in (("n_status_ins", "final_insufficient_rate"), ("n_abstain_any", "final_abstain_rate")):
                    row[nm] = float(M[f].sum() / M["n_sub"].sum())
                    with np.errstate(divide="ignore", invalid="ignore"):
                        row[nm + "_ci90"] = pct(pick(M[f]).sum(axis=(1, 2)) / pick(M["n_sub"]).sum(axis=(1, 2)), [5, 95])
                curve.append(row)
            res_run[arm] = {"seeds": seeds, "n_topics": len(topics), "k_max": kmax, "curve": curve}
            out["asserts"].append("%s %s k=%d == ANALYSIS 최종값(토큰·coverage 정확, grounding |Δ|<1e-6, final_rates 정확)"
                                  % (run_dir.name, arm, kmax))
        # E-R1 = E-GR k=0 대조
        for arm in arms:
            if RA.parse_arm(arm)[0] == "E-GR":
                r1 = RA.R1 + arm[len("E-GR"):]
                for s in sorted(s for (a, s) in raw if a == arm):
                    ref = an["arm_seed"]["%s%ss%d" % (r1, SEP, s)]
                    tr = {q: truncate(rec, 0) for q, rec in raw[(arm, s)].items()}
                    sm = RA.armseed_summary(tr, RA.Support())
                    assert sm["totals"]["accounting_tokens"] == ref["totals"]["accounting_tokens"]
                    assert sm["final_rates"] == ref["final_rates"]
                out["asserts"].append("%s %s k=0 == ANALYSIS %s" % (run_dir.name, arm, r1))
        out["runs"][run_dir.name] = res_run
    return out


# =========================================================================== 마크다운
def f3(x):
    return "—" if x is None else "%.3f" % x


def ci(c):
    return "[%s, %s]" % (f3(c[0]), f3(c[1]))


def to_md(res: dict) -> str:
    L = ["# R2 사후 추가 분석 (API 0원, 새 실행 0)", "",
         "자동 생성(`experiments/r2_extra.py`, SHA-256 `%s`). 수치는 같은 폴더 `EXTRA_RESULT.json`. "
         "등록 코드(`r2_analyze.py`, `r2_human.py`)는 고치지 않고 import 했다. 부트스트랩 %d회, seed %d. 실행 %s (HEAD %s)."
         % (res["code_sha256"][:16], res["n_boot"], RA.BOOT_SEED, res["run_at"], res["git_head"]), ""]
    A = res["A_h4_two_stage"]
    L += ["## A. H4 사람 UA — 1단(실제 수행) vs 2단(PREREG §8 머리 서술)", "",
          "UA 지지: %s. 2단 = 토픽 복원추출 → 뽑힌 토픽마다 seed 1–3 복원추출. 슬롯은 seed×관점 층에서 뽑혀 "
          "(토픽, seed) 셀이 빌 수 있고, 빈 셀은 Hájek 분자·분모에 0 을 더한다. 1단 수치는 HUMAN_RESULT.json 과 같음을 assert 했다."
          % A["support"], "",
          "| 라벨러 | 비교 | 범위 | 차이 | 1단 90% CI | 2단 90% CI | 비열등(δ 0.10) 1단 / 2단 | 비율 | 1단 비율 CI | 2단 비율 CI | 비율 ≥ 0.75 1단 / 2단 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for who, r in A["by_labeller"].items():
        for name in ("E-SQ_vs_E-GR", "E-R1_vs_E-GR"):
            for scope in ("all", "excl_inferable"):
                o, t = r[name][scope]["one_stage_topic_cluster"], r[name][scope]["two_stage_topic_x_seed"]
                L.append("| %s | %s | %s | %+.3f | %s | %s | %s / %s | %s | %s | %s | %s / %s |" % (
                    who, name, scope, o["diff"], ci(o["ci90"]), ci(t["ci90"]), o["noninferior"], t["noninferior"],
                    f3(o["ratio"]), ci(o["ratio_ci90"]), ci(t["ratio_ci90"]), o["ratio_pass"], t["ratio_pass"]))
    L += ["", "민감도 δ(2단): " + "; ".join(
        "%s %s %s" % (who, name, {d: v["noninferior"] for d, v in r[name]["all"]["two_stage_topic_x_seed"]["sensitivity"].items()})
        for who, r in A["by_labeller"].items() for name in ("E-SQ_vs_E-GR", "E-R1_vs_E-GR")), ""]
    B = res["B_agreement"]
    L += ["## B. 일치도 보강", "", "| 세트 | n | 범주 | 일치율 | Cohen κ | PABAK | Gwet AC1 |", "|---|---|---|---|---|---|---|"]
    for nm in ("vprime50", "anchor25", "h4_36"):
        for lv in ("three_class", "binary"):
            s = B[nm][lv]
            L.append("| %s | %d | %s | %.3f | %s | %s | %s |" % (nm, B[nm]["n"], "3범주" if lv == "three_class" else
                     "2범주(%s vs 나머지)" % B[nm]["binary_positive"], s["p_o"], f3(s["kappa"]), f3(s["pabak"]), f3(s["gwet_ac1"])))
    L += ["", "주변분포(저자 / 동료):"]
    for nm in ("vprime50", "anchor25", "h4_36"):
        L.append("- %s: " % nm + ", ".join("%s %d / %d" % (c, v["author"], v["colleague"])
                                           for c, v in B[nm]["three_class"]["marginals"].items()))
    L += ["", "PABAK·AC1 은 유병률 편향이 큰 표에서 κ 를 보완하는 지표다. PREREG §9 의 판정 규칙(κ ≥ 0.60)은 κ 로 등록됐고, "
          "이 지표로 판정을 바꾸지 않는다.", ""]
    C = res["C_grounding_abstain"]
    L += ["## C. grounding 기권 처리 민감도", "", "0 규약 값은 ANALYSIS.json 과 같음을 assert 했다(등록된 두 비교).", "",
          "| run | 비교 | 0 규약 | 양쪽 무문장 제외(셀) | 한쪽 무문장 제외(셀) | 양쪽(토픽) | 한쪽(토픽) |", "|---|---|---|---|---|---|---|"]
    for rn, rr in C["runs"].items():
        for name, e in rr["comparisons"].items():
            a = e["all_topics"]
            cell = lambda v: "%+.3f %s %s" % (a[v]["cell"]["point"], ci(a[v]["cell"]["ci90"]), "NI" if a[v]["cell"]["noninferior"] else "not NI")
            top = lambda v: ("%+.3f %s (%d토픽)" % (a[v]["topic"]["diff"], ci(a[v]["topic"]["ci90"]), a[v]["topic"]["n_topics"])
                             if a[v]["topic"].get("diff") is not None else "—")
            L.append("| %s | %s | %+.3f %s %s | %s | %s | %s | %s |" % (
                rn, name, a["zero_rule"]["diff"], ci(a["zero_rule"]["ci90"]), "NI" if a["zero_rule"]["noninferior"] else "not NI",
                cell("drop_both"), cell("drop_either"), top("drop_both"), top("drop_either")))
    L += [""]
    for rn, rr in C["runs"].items():
        for name, e in rr["comparisons"].items():
            if "deviations15" in e:
                d = e["deviations15"]
                L.append("- DEVIATIONS 15 %s %s — 소수 셋째 자리 일치: %s; 가장 가까운 변형 %s = %s (최대 |Δ| %.4f)" % (
                    name, d["target"], ", ".join(d["reproduced_by_3dp"]) or "없음", d["closest"],
                    ", ".join("%.4f" % v for v in d["closest_values"]), d["closest_max_abs_diff"]))
        L.append("- %s 무문장 질의 수: %s" % (rn, ", ".join("%s %d" % kv for kv in rr["empty_counts"].items())))
    D = res["D_round_truncation"]
    L += ["", "## D. 라운드 절단 비용–품질 곡선 (에뮬레이션)", "",
          "**관측 궤적을 라운드 k 에서 자른 에뮬레이션이다 — 라운드 예산이 작은 별도 실행이 아니다.** 각 질의의 상태는 라운드 k 끝 "
          "상태(그 전에 멈춘 질의는 최종 상태), 토큰은 라운드 ≤ k 의 회계 토큰 누계다. 라운드 ≤ k 의 결정(FREEZE 포함)은 관측 그대로이고, "
          "라운드 > k 의 FREEZE·재시도 결정은 잘린 궤적에 영향을 주지 않는다. 실제로 더 작은 라운드 예산으로 돌렸을 때 결정이 같았을지는 "
          "이 에뮬레이션으로 알 수 없다. "
          "k = 최대는 ANALYSIS.json 최종값을 재현함을 assert 했다(grounding 은 evidence 경로, 최대 |Δ| %.1e). "
          "E-R1 = E-GR k=0 (assert). '최종 insufficient' 는 status 기준(ANALYSIS final_rates 와 같음)이라 status 가 없는 L 은 0 이다; "
          "'최종 기권(전체 규약)' 은 sa_state 규약(legacy 빈 답·무인용 → insufficient)이다." % D["max_abs_grounding_dev_vs_ANALYSIS"], "",
          "| run | arm | k | 토큰/질의 (90% CI) | coverage | grounding (0 규약) | 최종 insufficient | 최종 기권(전체 규약) |",
          "|---|---|---|---|---|---|---|---|"]
    for rn, rr in D["runs"].items():
        for arm, v in rr.items():
            for row in v["curve"]:
                L.append("| %s | %s | %d | %.0f [%.0f, %.0f] | %.3f %s | %.3f %s | %.3f | %.3f |" % (
                    rn, arm, row["k"], row["mean_accounting_tokens"], row["mean_accounting_tokens_ci90"][0],
                    row["mean_accounting_tokens_ci90"][1], row["mean_coverage"], ci(row["mean_coverage_ci90"]),
                    row["mean_grounding"], ci(row["mean_grounding_ci90"]), row["final_insufficient_rate"],
                    row["final_abstain_rate"]))
    L += ["", "## assert", ""] + ["- " + s for blk in ("A_h4_two_stage", "B_agreement", "C_grounding_abstain", "D_round_truncation")
                                  for s in res[blk]["asserts"]]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-support", action="store_true", help="H4 MiniCheck 없이(빠른 점검)")
    ap.add_argument("--n-boot", type=int, default=RA.N_BOOT)
    ap.add_argument("--blocks", default="ABCD")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args(argv)
    code = Path(__file__).read_bytes()
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        head = None
    res: dict[str, Any] = {"code_sha256": hashlib.sha256(code).hexdigest(), "git_head": head,
                           "run_at": _dt.datetime.now().isoformat(timespec="seconds"),
                           "n_boot": args.n_boot, "boot_seed": RA.BOOT_SEED, "no_support": args.no_support,
                           "api_calls": 0, "new_runs": 0}
    if "B" in args.blocks:
        res["B_agreement"] = block_b(args.n_boot); print("B done", flush=True)
    if "C" in args.blocks:
        res["C_grounding_abstain"] = block_c(args.n_boot); print("C done", flush=True)
    if "D" in args.blocks:
        res["D_round_truncation"] = block_d(args.n_boot); print("D done", flush=True)
    if "A" in args.blocks:
        res["A_h4_two_stage"] = block_a(not args.no_support, args.n_boot); print("A done", flush=True)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    suffix = "" if (args.blocks == "ABCD" and not args.no_support and args.n_boot == RA.N_BOOT) else "_check"
    io.open(out / ("EXTRA_RESULT%s.json" % suffix), "w", encoding="utf-8").write(
        json.dumps(res, ensure_ascii=False, indent=1, default=float))
    if args.blocks == "ABCD":
        io.open(out / ("EXTRA_RESULT%s.md" % suffix), "w", encoding="utf-8").write(to_md(res))
    print("-> %s" % (out / ("EXTRA_RESULT%s.json" % suffix)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
