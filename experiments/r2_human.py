"""R2 사람 라벨 채점 — H4 사람 UA 비열등성, H5, H3, 저자–2nd annotator κ (PREREG_R2 §8·§9, DEVIATIONS 17). LLM 0.

PREREG §9: "사람 UA 비열등성 분석 코드는 라벨이 들어온 뒤 이 등록 규칙대로 작성한다. 작성 시점과 커밋을
DEVIATIONS.md 에 적는다." 이 파일은 묶음 라벨(`results/r2/bundle/labels/labels_r2_bundle_filled.csv`, 커밋 f669e45)이
들어온 뒤 2026-11-07 에 썼고, **라벨에 돌리기 전에** 커밋한다(DEVIATIONS 20).

  python experiments/r2_human.py                 # MiniCheck(GPU) 로 H4 최종 출력 문장 지지를 채점한다
  python experiments/r2_human.py --no-support    # 사람 라벨만(UA 의 자동 절반 없이) — 점검용

등록 규칙 (그대로 옮긴다)
  H4  비교 (E-SQ − E-GR), (E-R1 − E-GR). 슬롯 = (seed, 질의, 관점, 번호). 최종 출력 = 슬롯별 마지막 생성본.
      UA = 사람 "답함" ∧ 모든 문장이 MiniCheck 지지. 기권(빈 답·insufficient·dropped·format_fail)은 0.
      텍스트가 같은 짝은 차이 0. Hájek 가중 w = N_h / n_h. topic 군집 부트스트랩 90% CI 하한 > −0.10.
      공동 보고: 비율 하한 0.75. 민감도 δ 0.05, 0.20. 게이밍 감시: 불일치 짝 수, 최소 한쪽이 비지 않은 슬롯의
      조건부 UA, arm 별 기권률. 추론 가능 문항(봉인 노트)은 포함·제외로 함께 낸다.
  H5  E-SQ 최종 insufficient(30) 과 라운드 ≥ 1 새 answered(10) — 층을 합치지 않고 따로 낸다. 빈 컨텍스트 문항 포함·제외.
  H3  held-out 라운드 0 의 G2 (a)(b) 재현: (a) H2-B 100 재사용 — E 라운드 0 insufficient 인 문항의 저자 B 라벨
      '답있음' 비율(≤ 15%), (b) E answered '답함' 비율 − L(H2-A 앵커) '답함' 비율(≥ 0.10).
  κ   저자–라벨러: V′ 50(`_key_vc50.csv`) + H3 안의 H2-A 앵커 25. Cohen κ(3범주, 2범주), 일치율, 혼동표.
      κ ≥ 0.60 이면 조정 라벨, 미달이면 라벨러별 결과를 함께 내고 "신뢰도 제한"(PREREG §9).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from r2_analyze import BOOT_SEED, N_BOOT, R1, final_round, load_run, with_r1  # noqa: E402
from r2_sheets import strip_cites  # noqa: E402

HO = ROOT / "results/r2/heldout35"
SEALED = HO / "sheets_sealed"
LABELS = ROOT / "results/r2/bundle/labels/labels_r2_bundle_filled.csv"
D18 = ROOT / "results/d18"
V = ROOT / "results/vprime"
OUT = ROOT / "results/r2/human"
ABSTAIN = ("insufficient", "format_fail", "dropped", "legacy_empty")
DELTA, SENS, RATIO_BAR = 0.10, (0.05, 0.20), 0.75
KAPPA_BAR = 0.60
EXPECTED = {"h4_a": 36, "h5_b": 40, "h3_a": 60, "vc_a": 50}


def read_csv(p: Path) -> list[dict]:
    return list(csv.DictReader(io.open(p, encoding="utf-8-sig")))


# --------------------------------------------------------------------------- 라벨
def load_labels(path: Path = LABELS) -> dict[str, str]:
    rows = read_csv(path)
    n = Counter(r["set"] for r in rows)
    assert dict(n) == EXPECTED, n
    lab = {r["item_id"]: r["human_label"].strip() for r in rows}
    assert len(lab) == len(rows), "item_id 중복"
    empty = [i for i, v in lab.items() if not v]
    assert not empty, "빈 라벨 %s" % empty[:5]
    return lab


def check_keys_match(lab: dict[str, str]) -> None:
    """키에 있는 비어 있지 않은 문항과 라벨 문항이 정확히 같아야 한다."""
    want = set()
    for f in ("h4", "h5", "h3"):
        want |= {k["item_id"] for k in read_csv(SEALED / ("_key_%s.csv" % f)) if k["item_id"]}
    want |= {k["item_id"] for k in read_csv(V / "c50_sealed/_key_vc50.csv")}
    assert want == set(lab), (sorted(want - set(lab))[:5], sorted(set(lab) - want)[:5])


# --------------------------------------------------------------------------- 가중 비율과 군집 부트스트랩
def cluster_boot(clusters: list[str], cols: dict[str, np.ndarray], n_boot=N_BOOT, seed=BOOT_SEED):
    """행마다 군집 id 와 값 열들. 군집(토픽)을 복원추출해 열별 합을 낸다. 반환 {열: (n_boot,) 합}."""
    ids = sorted(set(clusters))
    pos = {c: i for i, c in enumerate(ids)}
    ci = np.array([pos[c] for c in clusters])
    sums = {k: np.bincount(ci, weights=v, minlength=len(ids)) for k, v in cols.items()}
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    cnt = np.zeros((n_boot, len(ids)))
    np.add.at(cnt, (np.repeat(np.arange(n_boot), len(ids)), draw.ravel()), 1)
    return {k: cnt @ s for k, s in sums.items()}, len(ids)


def pct(x: np.ndarray, q) -> list[float | None]:
    x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, q)] if len(x) else [None] * len(q)


# --------------------------------------------------------------------------- H4
def final_support(keys: list[dict], use_support: bool) -> tuple[dict[tuple, bool | None], dict]:
    """비어 있지 않은 H4 키 행마다 그 arm·seed 최종 출력 sub-answer 의 모든 문장이 MiniCheck 지지인가.
    E-R1 = E-GR 라운드 0 (with_r1). 같은 Support 구성(단일 문서, 고유 인용 청크, p > 0.5)을 쓴다."""
    runs = with_r1(load_run(HO))
    need = [k for k in keys if k["item_id"]]
    sas = {}
    for k in need:
        rec = runs[(k["arm"], int(k["seed"]))][k["query_id"]]
        fr = final_round(rec)
        sa = next(sa for br in fr["branches"] if br["branch"] == k["branch"]
                  for sa in br["sub_answers"] if sa["index"] == int(k["sq_index"]))
        assert strip_cites(sa.get("text") or ""), k
        sas[(k["arm"], k["seed"], k["query_id"], k["branch"], k["sq_index"])] = sa
    info = {"n_rows": len(sas)}
    if not use_support:
        return {s: None for s in sas}, info
    from r2_analyze import Support, load_chunk_text
    from lfm.config import Config, load_config
    from lfm.estimator import load_estimator
    base = load_config(ROOT / "config.yaml")
    d = base.as_dict(); d.setdefault("estimator", {})["kind"] = "minicheck_doc"
    est = load_estimator(Config(d, source="r2_human:h4"))
    sup = Support(est, load_chunk_text(ROOT / base["paths.index_dir"]))
    sup.prime(sas.values())
    out = {s: bool(sup.supported(sa)) for s, sa in sas.items()}
    info.update({"n_supported": sum(out.values()), "missing_chunks": sorted(sup.missing_chunks),
                 "sentences": {"/".join(s): sup.sentence_flags(sa) for s, sa in sas.items()}})
    return out, info


def h4_table(lab: dict[str, str], support: dict[tuple, bool | None]) -> tuple[dict, dict]:
    """슬롯 → {arm: (UA, human, nonempty)}, 슬롯 메타."""
    keys = read_csv(SEALED / "_key_h4.csv")
    slots: dict[tuple, dict[str, dict]] = defaultdict(dict)
    meta: dict[tuple, dict] = {}
    for k in keys:
        s = (k["seed"], k["query_id"], k["branch"], k["sq_index"])
        m = {"stratum": k["stratum"], "N_h": int(k["N_h"]), "n_h": int(k["n_h"])}
        assert meta.setdefault(s, m) == m, s
        assert k["arm"] not in slots[s], (s, k["arm"])
        if not k["item_id"]:
            assert k["source"] == "empty:UA=0"
            slots[s][k["arm"]] = {"ua": 0, "human": None, "nonempty": False, "item": None, "status": k["status"]}
            continue
        if k["status"] in ABSTAIN:                       # 등록: 기권은 0 (글자가 있어도)
            slots[s][k["arm"]] = {"ua": 0, "human": lab[k["item_id"]], "nonempty": True, "item": k["item_id"],
                                  "status": k["status"]}
            continue
        h = lab[k["item_id"]]
        sp = support[(k["arm"], k["seed"], k["query_id"], k["branch"], k["sq_index"])]
        ua = int(h == "답함" and (sp is None or sp))
        slots[s][k["arm"]] = {"ua": ua, "human": h, "nonempty": True, "item": k["item_id"], "sup": sp,
                              "status": k["status"]}
    # 텍스트가 같은 짝은 차이 0 — 같은 문항(item)을 공유하는 arm 은 UA 가 같아야 한다
    conflicts = []
    for s, arms in slots.items():
        by_item = defaultdict(set)
        for a, v in arms.items():
            if v["item"]:
                by_item[v["item"]].add(v["ua"])
        conflicts += [(s, it) for it, u in by_item.items() if len(u) > 1]
    assert not conflicts, conflicts
    assert all(len(v) == 3 for v in slots.values())
    return slots, meta


def h4_compare(slots, meta, a: str, b: str, excl_items=(), n_boot=N_BOOT) -> dict[str, Any]:
    ss = [s for s in sorted(slots) if not ({slots[s][a]["item"], slots[s][b]["item"]} & set(excl_items))]
    w = np.array([meta[s]["N_h"] / meta[s]["n_h"] for s in ss])
    ua = np.array([slots[s][a]["ua"] for s in ss], float)
    ub = np.array([slots[s][b]["ua"] for s in ss], float)
    ne = np.array([slots[s][a]["nonempty"] or slots[s][b]["nonempty"] for s in ss], float)
    sums, n_c = cluster_boot([s[1] for s in ss], {"w": w, "wa": w * ua, "wb": w * ub, "wne": w * ne,
                                                    "wane": w * ua * ne, "wbne": w * ub * ne}, n_boot)
    ra, rb = sums["wa"] / sums["w"], sums["wb"] / sums["w"]
    diff = ra - rb
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ra / rb
    pa, pb = float((w * ua).sum() / w.sum()), float((w * ub).sum() / w.sum())
    lo, hi = pct(diff, [5, 95])
    rlo, rhi = pct(ratio, [5, 95])
    seeds = sorted({s[0] for s in ss})
    per_seed = {}
    for sd in seeds:
        m = np.array([s[0] == sd for s in ss])
        per_seed[sd] = float((w[m] * (ua[m] - ub[m])).sum() / w[m].sum())
    nem = ne.astype(bool)
    return {
        "a": a, "b": b, "n_slots": len(ss), "n_topics": n_c, "excluded_items": sorted(excl_items),
        "ua_a": pa, "ua_b": pb, "diff": pa - pb, "ci90": [lo, hi], "delta": DELTA,
        "noninferior": bool(lo is not None and lo > -DELTA),
        "equivalent": bool(lo is not None and lo > -DELTA and hi < DELTA),
        "sensitivity": {str(d): {"noninferior": bool(lo > -d), "equivalent": bool(lo > -d and hi < d)} for d in SENS},
        "ratio": (pa / pb) if pb else None, "ratio_ci90": [rlo, rhi], "ratio_bar": RATIO_BAR,
        "ratio_pass": bool(rlo is not None and rlo >= RATIO_BAR),
        "ratio_nan_resamples": int((~np.isfinite(ratio)).sum()),
        "unweighted": {"ua_a": float(ua.mean()), "ua_b": float(ub.mean()), "diff": float((ua - ub).mean())},
        "per_seed_diff": per_seed,
        "gaming": {
            "discordant_pairs": int((ua != ub).sum()), "a_only": int(((ua == 1) & (ub == 0)).sum()),
            "b_only": int(((ua == 0) & (ub == 1)).sum()), "n_slots_nonempty_either": int(nem.sum()),
            "conditional_ua_a": float((w * ua)[nem].sum() / w[nem].sum()) if nem.any() else None,
            "conditional_ua_b": float((w * ub)[nem].sum() / w[nem].sum()) if nem.any() else None,
            "conditional_diff_ci90": pct((sums["wane"] - sums["wbne"]) / sums["wne"], [5, 95]),
        },
    }


def h4_report(lab, support, note) -> dict[str, Any]:
    slots, meta = h4_table(lab, support)
    inferable = sorted(set(note["h4_inferable_from_h3"]) | set(note["h4_inferable_from_h5"]))
    arms = ("E-SQ", "E-GR", R1)
    abst = {}
    for a in arms:
        w = np.array([meta[s]["N_h"] / meta[s]["n_h"] for s in slots])
        e = np.array([not slots[s][a]["nonempty"] or slots[s][a]["status"] in ABSTAIN for s in slots], float)
        abst[a] = {"abstain_rate_weighted": float((w * e).sum() / w.sum()), "n_abstain": int(e.sum()),
                   "n_nonempty": int(len(e) - e.sum())}
    human = {a: Counter(v[a]["human"] for v in slots.values() if v[a]["human"]) for a in arms}
    sup_fail = {a: sum(1 for v in slots.values() if v[a].get("human") == "답함" and v[a].get("sup") is False)
                for a in arms}
    out = {"n_slots": len(slots), "arm_abstention": abst, "human_labels_by_arm": {a: dict(c) for a, c in human.items()},
           "answered_but_not_supported": sup_fail, "inferable_items": inferable, "comparisons": {}}
    for a in ("E-SQ", R1):
        out["comparisons"]["%s_vs_E-GR" % a] = {"all": h4_compare(slots, meta, a, "E-GR"),
                                                 "excl_inferable": h4_compare(slots, meta, a, "E-GR", inferable)}
    # 사람 라벨만(자동 절반 없이) — 보고용
    slots_h, _ = h4_table(lab, {k: None for k in support})
    out["human_only"] = {"%s_vs_E-GR" % a: h4_compare(slots_h, meta, a, "E-GR") for a in ("E-SQ", R1)}
    return out


# --------------------------------------------------------------------------- H5
def h5_report(lab, note) -> dict[str, Any]:
    keys = read_csv(SEALED / "_key_h5.csv")
    no_ctx = set(note["h5_no_context"])
    out = {}
    for grp in ("ins", "new_answered"):
        for excl in (False, True):
            ks = [k for k in keys if k["stratum"].startswith(grp + ":") and not (excl and k["item_id"] in no_ctx)]
            w = np.array([int(k["N_h"]) / int(k["n_h"]) for k in ks])
            res = {"n": len(ks), "counts": dict(Counter(lab[k["item_id"]] for k in ks))}
            for lv in ("답있음", "일부", "답없음"):
                x = np.array([lab[k["item_id"]] == lv for k in ks], float)
                sums, _ = cluster_boot([k["query_id"] for k in ks], {"w": w, "wx": w * x})
                res[lv] = {"weighted": float((w * x).sum() / w.sum()), "unweighted": float(x.mean()),
                           "ci95": pct(sums["wx"] / sums["w"], [2.5, 97.5])}
            out["%s%s" % (grp, "_excl_no_context" if excl else "")] = res
    out["no_context_items"] = sorted(no_ctx)
    return out


# --------------------------------------------------------------------------- H3
def h3_report(lab) -> dict[str, Any]:
    keys = read_csv(SEALED / "_key_h3.csv")
    author = {(r["set"], r["item_id"]): r["human_label"].strip() for r in read_csv(D18 / "labels_all_filled.csv")}
    e = [k for k in keys if k["arm"] == "E-GR"]
    anc = [k for k in keys if k["arm"] == "H2-anchor"]
    assert len(e) == 35 and len(anc) == 25

    def rate(ks, labf, lv="답함"):
        w = np.array([int(k["N_h"]) / int(k["n_h"]) for k in ks])
        x = np.array([labf(k) == lv for k in ks], float)
        return {"n": len(ks), "weighted": float((w * x).sum() / w.sum()), "unweighted": float(x.mean()),
                "counts": dict(Counter(labf(k) for k in ks))}

    col = lambda k: lab[k["item_id"]]
    auth = lambda k: author[("h2hoA", k["source"].split(":", 1)[1])]
    b = {"E_answered_colleague": rate(e, col), "L_anchor_colleague": rate(anc, col), "L_anchor_author": rate(anc, auth)}
    b["diff_colleague_weighted"] = b["E_answered_colleague"]["weighted"] - b["L_anchor_colleague"]["weighted"]
    b["diff_colleague_unweighted"] = b["E_answered_colleague"]["unweighted"] - b["L_anchor_colleague"]["unweighted"]
    b["diff_mixed_unweighted"] = b["E_answered_colleague"]["unweighted"] - b["L_anchor_author"]["unweighted"]
    b["bar"] = 0.10
    b["pass_colleague"] = b["diff_colleague_unweighted"] >= 0.10
    b["note"] = ("앵커 가중은 H3 안의 앵커 표집(N_h = H2-A 층 크기)이다. E 와 L 의 모집단이 달라(held-out s1 라운드 0 E vs "
                 "D18 H2-A L) 가중 차이는 참고로만 낸다. 판정은 G2(b) 와 같은 비가중")
    ru = read_csv(SEALED / "_key_h3_reuseB.csv")
    ins = [r for r in ru if r["found"] == "1" and r["e_status_r0"] == "insufficient"]
    yes = sum(1 for r in ins if author[("h2hoB", r["h2_item"])] == "답있음")
    a = {"n_reuse": len(ru), "n_found": sum(r["found"] == "1" for r in ru), "n_E_insufficient": len(ins),
         "답있음": yes, "일부": sum(1 for r in ins if author[("h2hoB", r["h2_item"])] == "일부"),
         "rate": yes / len(ins) if ins else None, "bar": 0.15,
         "pass": bool(ins) and yes / len(ins) <= 0.15,
         "labeler": "저자(H2-B 라벨 재사용, 새 라벨 0)",
         "status_counts": dict(Counter(r["e_status_r0"] for r in ru if r["found"] == "1"))}
    return {"a": a, "b": b}


# --------------------------------------------------------------------------- κ
def cohen_kappa(x: list[str], y: list[str], cats: list[str]) -> float | None:
    n = len(x)
    if not n:
        return None
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = Counter(x), Counter(y)
    pe = sum(px[c] * py[c] for c in cats) / n ** 2
    return None if pe == 1 else (po - pe) / (1 - pe)


def kappa_block(x: list[str], y: list[str], cats: list[str], pos: str, n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    xb = [("pos" if v == pos else "neg") for v in x]
    yb = [("pos" if v == pos else "neg") for v in y]
    rng = np.random.default_rng(seed)
    n = len(x)
    bk3, bk2 = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        k3 = cohen_kappa([x[j] for j in i], [y[j] for j in i], cats)
        k2 = cohen_kappa([xb[j] for j in i], [yb[j] for j in i], ["pos", "neg"])
        bk3.append(np.nan if k3 is None else k3); bk2.append(np.nan if k2 is None else k2)
    conf = {a: {b: sum(1 for u, v in zip(x, y) if u == a and v == b) for b in cats} for a in cats}
    return {"n": n, "kappa3": cohen_kappa(x, y, cats), "kappa3_ci95": pct(np.array(bk3), [2.5, 97.5]),
            "kappa2": cohen_kappa(xb, yb, ["pos", "neg"]), "kappa2_ci95": pct(np.array(bk2), [2.5, 97.5]),
            "binary_positive": pos, "agree3": sum(a == b for a, b in zip(x, y)) / n,
            "agree2": sum(a == b for a, b in zip(xb, yb)) / n,
            "confusion_rows_author_cols_colleague": conf,
            "marginals": {"author": dict(Counter(x)), "colleague": dict(Counter(y))}}


def vprime_author() -> dict[str, str]:
    lab = {r["item_id"]: r["human_label"].strip() for r in read_csv(V / "labels_stage1_filled.csv")}
    for r in read_csv(D18 / "labels_all_filled.csv"):
        if r["set"] == "stage2":
            lab[r["item_id"]] = r["human_label"].strip()
    assert len(lab) == 100
    return lab


def vprime_pr(items: list[dict], labf, pred, w) -> dict:
    """MiniCheck 정밀도·재현율(판단불가 제외). w = 층 가중."""
    out = {}
    for weighted in (False, True):
        tp = fp = fn = 0.0
        for it in items:
            v = labf(it)
            if v not in ("지지", "불지지"):
                continue
            ww = w[it["cell"]] if weighted else 1.0
            p, h = pred(it), v == "지지"
            tp += ww * (p and h); fp += ww * (p and not h); fn += ww * (h and not p)
        out["weighted" if weighted else "unweighted"] = {
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None}
    return out


def kappa_report(lab) -> dict[str, Any]:
    key = read_csv(V / "c50_sealed/_key_vc50.csv")
    auth = vprime_author()
    vkey = {}
    for st in ("stage1", "stage2"):
        for r in read_csv(V / ("_key_%s.csv" % st)):
            vkey[r["item_id"]] = r
    changed = {c["item_id"]: c for c in json.load(io.open(V / "VPRIME_DEDUP_CHECK.json", encoding="utf-8"))["changed"]}
    cats_v = ["지지", "불지지", "판단불가"]
    x = [auth[k["source_item"]] for k in key]
    y = [lab[k["item_id"]] for k in key]
    v = kappa_block(x, y, cats_v, "지지")
    v["by_cell"] = {}
    for c in sorted({k["cell"] for k in key}):
        idx = [i for i, k in enumerate(key) if k["cell"] == c]
        v["by_cell"][c] = {"n": len(idx), "agree": sum(x[i] == y[i] for i in idx) / len(idx),
                           "author": dict(Counter(x[i] for i in idx)), "colleague": dict(Counter(y[i] for i in idx))}
    # V′ 50 에서 라벨러별 MiniCheck 정밀도·재현율 (같은 50건, 층 가중 = 원 층 크기 / 50건 안 층 표본)
    strata = json.load(io.open(V / "stage2_strata.json", encoding="utf-8"))["strata"]
    n50 = Counter(k["cell"] for k in key)
    w50 = {c: strata[c]["N_h"] / n50[c] for c in n50}
    items = [{**k, "author": a, "colleague": b} for k, a, b in zip(key, x, y)]
    pred_reg = lambda it: vkey[it["source_item"]]["lab_mc"] == "1"
    pred_dedup = lambda it: (changed[it["source_item"]]["p_new"] > 0.5) if it["source_item"] in changed else pred_reg(it)
    v["minicheck_on_50"] = {
        lbl: {"registered_operator": vprime_pr(items, lambda it, l=lbl: it[l], pred_reg, w50),
              "dedup_operator": vprime_pr(items, lambda it, l=lbl: it[l], pred_dedup, w50)}
        for lbl in ("author", "colleague")}
    # 앵커 25
    k3 = [k for k in read_csv(SEALED / "_key_h3.csv") if k["arm"] == "H2-anchor"]
    author = {(r["set"], r["item_id"]): r["human_label"].strip() for r in read_csv(D18 / "labels_all_filled.csv")}
    ax = [author[("h2hoA", k["source"].split(":", 1)[1])] for k in k3]
    ay = [lab[k["item_id"]] for k in k3]
    anc = kappa_block(ax, ay, ["답함", "일부", "답안함"], "답함")
    pooled2 = cohen_kappa([("pos" if a == "지지" else "neg") for a in x] + [("pos" if a == "답함" else "neg") for a in ax],
                          [("pos" if a == "지지" else "neg") for a in y] + [("pos" if a == "답함" else "neg") for a in ay],
                          ["pos", "neg"])
    kmin = min(v["kappa3"], anc["kappa3"])
    return {"vprime50": v, "anchor25": anc, "pooled_binary_kappa_75": pooled2, "bar": KAPPA_BAR,
            "rule": ("κ ≥ 0.60 → 조정 라벨" if kmin >= KAPPA_BAR else
                     "κ < 0.60 → 라벨러별 결과를 함께 내고 '신뢰도 제한'(PREREG §9)"),
            "min_kappa3": kmin, "reliability_limited": bool(kmin < KAPPA_BAR)}


# --------------------------------------------------------------------------- 자동 비열등 (기존 분석 결과를 그대로 옮긴다)
def auto_noninf() -> dict[str, Any]:
    an = json.load(io.open(HO / "ANALYSIS.json", encoding="utf-8"))
    return {"%s:%s" % (c, m): {k: v.get(k) for k in ("a", "b", "metric", "diff", "ci90", "delta", "noninferior")}
            for c, ms in an["all_topics"]["noninferiority"].items() for m, v in ms.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-support", action="store_true", help="MiniCheck 없이 사람 라벨만 (점검용)")
    ap.add_argument("--labels", default=str(LABELS), help="점검용 가짜 라벨 CSV 를 줄 수 있다")
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args(argv)
    lab = load_labels(Path(args.labels))
    check_keys_match(lab)
    note = json.load(io.open(SEALED / "_note_heldout_blinding.json", encoding="utf-8"))
    res: dict[str, Any] = {"labels_csv": args.labels, "n_boot": N_BOOT, "boot_seed": BOOT_SEED}
    res["kappa"] = kappa_report(lab)                    # κ 먼저 — 보고 규칙을 정한다
    support, sinfo = final_support(read_csv(SEALED / "_key_h4.csv"), not args.no_support)
    res["h4_support"] = sinfo
    res["h4"] = h4_report(lab, support, note)
    res["h5"] = h5_report(lab, note)
    res["h3"] = h3_report(lab)
    res["auto_noninferiority"] = auto_noninf()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    name = "HUMAN_RESULT%s.json" % ("_nosupport" if args.no_support else "")
    io.open(out / name, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    k = res["kappa"]
    print("κ V′50 3범주 %.3f / 2범주 %.3f · 앵커25 3범주 %.3f / 2범주 %.3f → %s" % (
        k["vprime50"]["kappa3"], k["vprime50"]["kappa2"], k["anchor25"]["kappa3"], k["anchor25"]["kappa2"], k["rule"]))
    for c, r in res["h4"]["comparisons"].items():
        a = r["all"]
        print("H4 %s: UA %.3f vs %.3f, 차 %+.3f, 90%% CI [%.3f, %.3f] → 비열등 %s · 비율 %s CI %s" % (
            c, a["ua_a"], a["ua_b"], a["diff"], a["ci90"][0], a["ci90"][1], a["noninferior"], a["ratio"], a["ratio_ci90"]))
    print("-> %s" % (out / name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
