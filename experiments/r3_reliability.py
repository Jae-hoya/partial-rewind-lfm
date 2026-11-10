"""PREREG_R3 §2 — V′ C50 의 저자·A·B 신뢰도, 기계적 다수결, 연산자 precision/recall. **LLM 0, GPU 0.** C0 에 커밋하고 고치지 않고 돌린다.

  python experiments/r3_reliability.py                      # results/r3/labels/labels_{A,B}.csv → R3_RELIABILITY.{json,md}
  python experiments/r3_reliability.py --selftest [--B 500]  # 실제 저자 라벨로 V1·C50-v1 재현 + 합성 라벨 두 벌로 끝까지

입력: 봉인 키 `results/r3/sealed/_key_r3_v.csv`(sheet_item_id, stage, source_item, cell, doc_tokens, len_bin, t_min_s),
저자 라벨(`labels_stage1_filled.csv` e9f6652, `d18/labels_all_filled.csv` stage2 aa75e82), A-v1(`r2/bundle/labels`),
점수열 `rescore_singledoc_rows.csv`(registered_dup), `VPRIME_DEDUP_CHECK.json`(dedup), `_key_stage{1,2}.csv`(g1).
정의는 PREREG_R3 §2.5~§2.7.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "results/vprime"
R3 = ROOT / "results/r3"
CATS = ["지지", "불지지", "판단불가"]
BARS = {"precision": 0.85, "recall": 0.70, "recovery": 0.50}
SEED_KAPPA, SEED_OP = 4310, 4311
KAPPA_BAR = 0.60
TR_FULL = "근거 전체"


def rd(p: Path) -> list[dict]:
    return list(csv.DictReader(io.open(p, encoding="utf-8-sig")))


def b(x: str) -> int:
    """이진: 지지 = 1, 불지지 ∪ 판단불가 = 0."""
    return int(x == "지지")


# --------------------------------------------------------------------------- 일치 통계
def cohen(a, c, cats, w=None) -> float | None:
    a, c = list(a), list(c)
    if not a:
        return None
    w = np.ones(len(a)) if w is None else np.asarray(w, float)
    W = w.sum()
    po = sum(wi for wi, x, y in zip(w, a, c) if x == y) / W
    pe = sum((sum(wi for wi, x in zip(w, a) if x == k) / W) * (sum(wi for wi, y in zip(w, c) if y == k) / W) for k in cats)
    return None if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)


def agree(a, c) -> float:
    return float(np.mean([x == y for x, y in zip(a, c)])) if a else float("nan")


def pabak(a, c, K) -> float:
    po = agree(a, c)
    return (K * po - 1) / (K - 1)


def ac1(a, c, cats) -> float | None:
    po = agree(a, c)
    K = len(cats)
    pi = {k: (sum(x == k for x in a) + sum(y == k for y in c)) / (2 * len(a)) for k in cats}
    pe = sum(pi[k] * (1 - pi[k]) for k in cats) / (K - 1)
    return None if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)


def fleiss(rows: list[list], cats) -> float | None:
    """rows: 문항마다 평정자 라벨 목록(같은 수)."""
    n = len(rows[0])
    N = len(rows)
    cnt = [[sum(x == k for x in r) for k in cats] for r in rows]
    P_i = [(sum(c * c for c in r) - n) / (n * (n - 1)) for r in cnt]
    p_j = [sum(r[j] for r in cnt) / (N * n) for j in range(len(cats))]
    Pb, Pe = float(np.mean(P_i)), sum(p * p for p in p_j)
    return None if abs(1 - Pe) < 1e-12 else (Pb - Pe) / (1 - Pe)


def table(a, c, cats) -> dict:
    return {x: {y: sum(1 for p, q in zip(a, c) if p == x and q == y) for y in cats} for x in cats}


def wilson(k: int, n: int, z: float = 1.959964) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]


def strat_boot(cells: list[str], fn, B: int, seed: int):
    """cell 층화 문항 부트스트랩(cell 안 복원 재추출), 백분위 95%. fn(idx) → 값 또는 dict 값."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, c in enumerate(cells):
        by[c].append(i)
    keys = sorted(by)
    draws = []
    for _ in range(B):
        idx = [j for k in keys for j in rng.choice(by[k], len(by[k]), replace=True).tolist()]
        draws.append(fn(idx))
    return draws


def pct(vals, lo=2.5, hi=97.5):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if len(v) else [None, None]


# --------------------------------------------------------------------------- 자료
def load_static() -> dict:
    key = {}
    for st in ("stage1", "stage2"):
        for r in rd(V / ("_key_%s.csv" % st)):
            key[r["item_id"]] = {**r, "stage": st}
    lab = {r["item_id"]: r["human_label"].strip() for r in rd(V / "labels_stage1_filled.csv")}
    for r in rd(ROOT / "results/d18/labels_all_filled.csv"):
        if r["set"] == "stage2":
            assert r["item_id"] not in lab
            lab[r["item_id"]] = r["human_label"].strip()
    assert set(lab) == set(key) and len(lab) == 100
    pnew = {r["item_id"]: float(r["p_mc_new"]) for r in rd(V / "rescore_singledoc_rows.csv")}
    dd = json.load(io.open(V / "VPRIME_DEDUP_CHECK.json", encoding="utf-8"))
    pdedup = dict(pnew)
    for c in dd["changed"]:
        pdedup[c["item_id"]] = float(c["p_new"])
    strata = json.load(io.open(V / "stage2_strata.json", encoding="utf-8"))["strata"]
    N_h = {c: strata[c]["N_h"] for c in strata}
    c50 = {r["item_id"]: r["source_item"] for r in rd(V / "c50_sealed/_key_vc50.csv")}
    av1 = {r["item_id"]: r["human_label"].strip() for r in rd(ROOT / "results/r2/bundle/labels/labels_r2_bundle_filled.csv")
           if r["set"] == "vc_a"}
    pl = {r["item_id"]: int(r["doc_tokens"]) for r in rd(V / "premise_lengths.csv")}
    return {"key": key, "lab": lab, "pnew": pnew, "pdedup": pdedup, "N_h": N_h, "c50": c50, "av1": av1, "pl": pl}


def len_bin(t: int) -> str:
    return "short" if t < 613 else ("long" if t >= 2073 else "mid")


OPS = {"MiniCheck (registered_dup, p>0.5)": lambda S, i: S["pnew"][i] > 0.5,
       "MiniCheck (dedup, p>0.5)": lambda S, i: S["pdedup"][i] > 0.5,
       "g1 (p>=0.5)": lambda S, i: S["key"][i]["lab_g1"] == "1"}


def op_metrics(S, items: list[str], y: dict[str, int], op, weighted: bool) -> dict:
    """vprime_final_score 와 같은 가중(Hájek, w = N_h/n_h, n_h = 그 판의 cell 문항 수)."""
    n_h = Counter(S["key"][i]["cell"] for i in items)
    acc = {m: [0.0, 0.0] for m in BARS}
    for i in items:
        c = S["key"][i]["cell"]
        ww = S["N_h"][c] / n_h[c] if weighted else 1.0
        pred, pos = op(S, i), y[i] == 1
        if pred:
            acc["precision"][1] += ww; acc["precision"][0] += ww * pos
        if pos:
            acc["recall"][1] += ww; acc["recall"][0] += ww * pred
            if S["key"][i]["lab_g1"] == "0":
                acc["recovery"][1] += ww; acc["recovery"][0] += ww * pred
    return {m: (k / n if n else None) for m, (k, n) in acc.items()} | ({"_raw": acc} if not weighted else {})


def op_version(S, items, y, B, seed) -> dict:
    out = {"n": len(items), "n_h": dict(sorted(Counter(S["key"][i]["cell"] for i in items).items())),
           "human_support": sum(y[i] for i in items)}
    cells = [S["key"][i]["cell"] for i in items]
    for name, op in OPS.items():
        raw = op_metrics(S, items, y, op, False)
        wt = op_metrics(S, items, y, op, True)
        o = {}
        for m in BARS:
            k, n = raw["_raw"][m]
            o[m] = {"k": int(k), "n": int(n), "rate": raw[m], "wilson": wilson(int(k), int(n)), "weighted": wt[m], "bar": BARS[m]}
        draws = strat_boot(cells, lambda idx: op_metrics(S, [items[j] for j in idx], y, op, True), B, seed)
        for m in BARS:
            o[m]["weighted_ci95"] = pct([d[m] for d in draws])
            o[m]["weighted_upper95"] = o[m]["weighted_ci95"][1]
        out[name] = o
    return out


def _row(p, up, bar) -> int:
    if p is None:
        return 3
    if p < bar - 0.10 or (p < bar and up is not None and up < bar):
        return 3
    return 2 if p < bar else 1


def wording_row(o: dict) -> dict:
    """§2.7 문구 행: 3 > 2 > 1, precision·recall 중 더 나쁜 행.
    **판정 = 비가중 점추정과 Wilson 95% 상한**(SPEC_M1Vprime §4 와 같은 규칙, PREREG_R3 §1.3·§2.7).
    Hájek 가중 점추정·부트스트랩 상한으로 같은 규칙을 적용한 행은 함께 보고만 한다(판정에 쓰지 않음)."""
    rows, rows_w = {}, {}
    for m in ("precision", "recall"):
        rows[m] = _row(o[m]["rate"], o[m]["wilson"][1], BARS[m])
        rows_w[m] = _row(o[m]["weighted"], o[m]["weighted_upper95"], BARS[m])
    return {"rule": "unweighted point estimate + Wilson 95% upper (SPEC §4)", "by_metric": rows, "row": max(rows.values()),
            "co_report_weighted": {"by_metric": rows_w, "row": max(rows_w.values()),
                                   "rule": "Hajek-weighted point estimate + bootstrap 95% upper (report only)"}}


ROW_TEXT = {3: "validity not confirmed by independent annotators.",
            2: "below the bar on {labels} labels but within uncertainty; selection stands as registered",
            1: "confirmed on majority labels of three annotators"}


# --------------------------------------------------------------------------- 주 분석
def read_labels(p: Path) -> dict[str, dict]:
    return {r["item_id"]: r for r in rd(p)} if p and Path(p).exists() else {}


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


def pair_stats(x, y, cells, w_item, B, seed) -> dict:
    xb, yb = [b(v) for v in x], [b(v) for v in y]
    out = {"n": len(x), "kappa_bin": cohen(xb, yb, [0, 1]), "kappa_bin_weighted": cohen(xb, yb, [0, 1], w_item),
           "kappa_3cat": cohen(x, y, CATS), "agree_bin": agree(xb, yb), "agree_3cat": agree(x, y),
           "pabak_bin": pabak(xb, yb, 2), "pabak_3cat": pabak(x, y, 3), "ac1_bin": ac1(xb, yb, [0, 1]),
           "ac1_3cat": ac1(x, y, CATS), "support_rate": [float(np.mean(xb)), float(np.mean(yb))],
           "table_3cat": table(x, y, CATS)}
    draws = strat_boot(cells, lambda idx: cohen([xb[j] for j in idx], [yb[j] for j in idx], [0, 1]), B, seed)
    out["kappa_bin_ci95"] = pct(draws)
    return out


def analyze(lab_paths: dict[str, Path], key_path: Path, B: int = 10000, S: dict | None = None) -> dict:
    S = S or load_static()
    keyv = rd(key_path)
    test = [r for r in keyv if r["stage"] == "test"]
    prac = [r for r in keyv if r["stage"] == "practice"]
    assert len(test) == 50 and len(prac) == 10
    assert {r["sheet_item_id"]: r["source_item"] for r in test} == S["c50"], "C50 키 불일치"
    L = {a: read_labels(p) for a, p in lab_paths.items()}
    done = {a: (len(L[a]) > 0 and all(L[a].get(r["sheet_item_id"], {}).get("label", "") in CATS for r in test)) for a in L}
    n_lab = {a: sum(L[a].get(r["sheet_item_id"], {}).get("label", "") in CATS for r in test) for a in L}
    ids = [r["sheet_item_id"] for r in test]
    src = {r["sheet_item_id"]: r["source_item"] for r in test}
    cell = {r["sheet_item_id"]: r["cell"] for r in test}
    tok = {r["sheet_item_id"]: int(r["doc_tokens"]) for r in test}
    n_h = Counter(cell.values())
    assert dict(n_h) == {"P1": 13, "P2": 21, "N1": 12, "N2": 4}
    w_item = [S["N_h"][cell[i]] / n_h[cell[i]] for i in ids]
    cells = [cell[i] for i in ids]
    author = [S["lab"][src[i]] for i in ids]
    raters = {"author": author}
    for a in ("A", "B"):
        if done.get(a):
            raters[a] = [L[a][i]["label"] for i in ids]
    raters["A-v1"] = [S["av1"][i] for i in ids]
    res = {"spec": "PREREG_R3 §2.5–§2.7", "completed": done, "n_test": 50, "n_labelled_test": n_lab}
    # §7.3 받은 데까지: 50문항 미완 판정자는 판정·다수결에 넣지 않고, 단 문항에서 기술 통계만 낸다
    partial = {}
    for a in L:
        if not done.get(a) and n_lab.get(a):
            J = [j for j, i in enumerate(ids) if L[a].get(i, {}).get("label", "") in CATS]
            xa = [b(author[j]) for j in J]
            ya = [b(L[a][ids[j]]["label"]) for j in J]
            partial[a] = {"n_labelled": len(J), "agree_bin": agree(xa, ya), "kappa_bin": cohen(xa, ya, [0, 1]),
                          "agree_3cat": agree([author[j] for j in J], [L[a][ids[j]]["label"] for j in J]),
                          "support_rate": float(np.mean(ya)) if J else None,
                          "note": "descriptive only (partial completion, §7.3); not used for the verdict"}
    res["partial_completion"] = partial
    pairs = {}
    for x, y, sd in (("author", "A", SEED_KAPPA), ("author", "B", SEED_KAPPA), ("A", "B", SEED_KAPPA),
                     ("author", "A-v1", SEED_KAPPA), ("A-v1", "A", SEED_KAPPA), ("A-v1", "B", SEED_KAPPA)):
        if x in raters and y in raters:
            pairs["%s–%s" % (x, y)] = pair_stats(raters[x], raters[y], cells, w_item, B, sd)
    res["pairs"] = pairs
    if all(k in raters for k in ("author", "A", "B")):
        res["fleiss_bin"] = fleiss([[b(raters[r][j]) for r in ("author", "A", "B")] for j in range(50)], [0, 1])
        res["fleiss_3cat"] = fleiss([[raters[r][j] for r in ("author", "A", "B")] for j in range(50)], CATS)
    # 판정
    newc = [a for a in ("A", "B") if a in raters]
    if len(newc) == 2:
        ka, kb = pairs["author–A"]["kappa_bin"], pairs["author–B"]["kappa_bin"]
        ok = ka is not None and kb is not None and ka >= KAPPA_BAR and kb >= KAPPA_BAR
        res["verdict"] = {"status": "신뢰도 확보" if ok else "신뢰도 제한", "kappa_author_A": ka, "kappa_author_B": kb,
                          "ci95_author_A": pairs["author–A"]["kappa_bin_ci95"], "ci95_author_B": pairs["author–B"]["kappa_bin_ci95"],
                          "R1-3c": "충족" if ok else "부분 충족 (reliability-limited)"}
    elif len(newc) == 1:
        res["verdict"] = {"status": "single second annotator", "R1-3c": "부분 충족 (single second annotator)",
                          "kappa_author_X": pairs["author–%s" % newc[0]]["kappa_bin"]}
    else:
        res["verdict"] = {"status": "no new annotator", "R1-3c": "미충족"}
    # 장치 진단·민감도
    diag = {}
    for a in newc:
        rows = [L[a][i] for i in ids]
        act = np.array([float(r["active_ms"] or 0) for r in rows])
        tm = np.array([float(r["t_min_s"] or 0) for r in rows])
        bins = [len_bin(tok[i]) for i in ids]
        d = {"by_len_bin": {}, "active_over_tmin_quantiles": [float(q) for q in np.percentile(act / np.maximum(tm * 1000, 1), [10, 25, 50, 75, 90])],
             "hidden_gt_active": int(sum(float(r["hidden_ms"] or 0) > float(r["active_ms"] or 0) for r in rows)),
             "translator": dict(Counter(r["translator_used"] for r in rows)),
             "translator_tool": rows[0].get("translator_tool", ""), "proficiency": rows[0].get("proficiency", "")}
        for bn in ("short", "mid", "long"):
            J = [j for j in range(50) if bins[j] == bn]
            xa, ya = [b(author[j]) for j in J], [b(raters[a][j]) for j in J]
            d["by_len_bin"][bn] = {"n": len(J), "agree_bin": agree(xa, ya), "kappa_bin": cohen(xa, ya, [0, 1]),
                                   "support_rate": float(np.mean(ya)) if J else None,
                                   "active_ms_median": float(np.median([act[j] for j in J])) if J else None}
        by_tr = {}
        for t in sorted(set(r["translator_used"] for r in rows)):
            J = [j for j in range(50) if rows[j]["translator_used"] == t]
            by_tr[t] = {"n": len(J), "agree_bin": agree([b(author[j]) for j in J], [b(raters[a][j]) for j in J])}
        d["agree_by_translator"] = by_tr
        sens = {}
        for name, keep in (("excl_active_lt_1.2tmin", [j for j in range(50) if act[j] >= 1.2 * tm[j] * 1000]),
                           ("excl_translator_full", [j for j in range(50) if rows[j]["translator_used"] != TR_FULL])):
            sens[name] = {"n": len(keep), "kappa_bin": cohen([b(author[j]) for j in keep], [b(raters[a][j]) for j in keep], [0, 1])}
        fl = [rows[j]["first_label"] or rows[j]["label"] for j in range(50)]
        sens["first_label"] = {"n": 50, "kappa_bin": cohen([b(x) for x in author], [b(x) for x in fl], [0, 1])}
        d["sensitivity"] = sens
        # 연습 일치 (첫 답)
        pr = [L[a].get(r["sheet_item_id"], {}) for r in prac]
        pa = [S["lab"][r["source_item"]] for r in prac]
        pf = [x.get("first_label", "") for x in pr]
        d["practice"] = {"n": sum(1 for x in pf if x), "agree_bin": agree([b(x) for x in pa], [b(x) for x in pf]),
                         "agree_3cat": agree(pa, pf),
                         "by_len_bin": {bn: agree([b(pa[j]) for j in range(10) if prac[j]["len_bin"] == bn],
                                                  [b(pf[j]) for j in range(10) if prac[j]["len_bin"] == bn])
                                        for bn in ("short", "mid", "long") if any(p["len_bin"] == bn for p in prac)}}
        diag[a] = d
    # A-v1 을 같은 경계로 (§2.5)
    bins = [len_bin(tok[i]) for i in ids]
    av = {}
    for bn in ("short", "mid", "long"):
        J = [j for j in range(50) if bins[j] == bn]
        xa, ya = [b(author[j]) for j in J], [b(raters["A-v1"][j]) for j in J]
        av[bn] = {"n": len(J), "support": sum(ya), "agree_bin": agree(xa, ya), "kappa_bin": cohen(xa, ya, [0, 1])}
    diag["A-v1"] = {"by_len_bin": av, "bins": "613 / 2,073 (294-population doc_tokens), same as A and B"}
    res["diagnostics"] = diag
    # 다수결
    if len(newc) == 2:
        maj = [int(b(author[j]) + b(raters["A"][j]) + b(raters["B"][j]) >= 2) for j in range(50)]
        flips = Counter()
        for j in range(50):
            ab = b(author[j])
            if maj[j] != ab:
                flips["지지→불지지" if ab == 1 else "불지지→지지"] += 1
        res["majority"] = {"author_flipped": dict(flips), "author_flipped_n": sum(flips.values()),
                           "A_minority": sum(b(raters["A"][j]) != maj[j] for j in range(50)),
                           "B_minority": sum(b(raters["B"][j]) != maj[j] for j in range(50)),
                           "unanimous_rate": float(np.mean([b(author[j]) == b(raters["A"][j]) == b(raters["B"][j]) for j in range(50)])),
                           "flips_by_len_bin": dict(Counter(bins[j] for j in range(50) if maj[j] != b(author[j]))),
                           "flips_by_translator": {a: dict(Counter(L[a][ids[j]]["translator_used"] for j in range(50)
                                                                   if maj[j] != b(author[j]))) for a in ("A", "B")}}
        if res["majority"]["author_flipped_n"] == 0:
            res["majority"]["sentence"] = "the majority coincided with the author on every item"
    # 연산자 P/R
    vers = {}
    v100 = sorted(S["key"])
    yA = {i: b(S["lab"][i]) for i in v100}
    vers["V1"] = op_version(S, v100, yA, B, SEED_OP)
    c50src = [src[i] for i in ids]
    vers["V1c"] = op_version(S, c50src, yA, B, SEED_OP)
    for a, nm in (("A", "V3a"), ("B", "V3b")):
        if a in raters:
            vers[nm] = op_version(S, c50src, {src[i]: b(raters[a][j]) for j, i in enumerate(ids)}, B, SEED_OP)
    if len(newc) == 2:
        ym = {src[i]: maj[j] for j, i in enumerate(ids)}
        vers["V4"] = op_version(S, c50src, ym, B, SEED_OP)
        y5 = dict(yA); y5.update(ym)
        vers["V5"] = op_version(S, v100, y5, B, SEED_OP)
    res["operator"] = vers
    REG = "MiniCheck (registered_dup, p>0.5)"
    if len(newc) == 2 and res["verdict"]["status"] == "신뢰도 확보":
        prim = ["V4"]
    elif len(newc) == 2:
        prim = ["V1c", "V3a", "V3b"]
    elif len(newc) == 1:
        prim = ["V1c", "V3a" if newc[0] == "A" else "V3b"]
    else:
        prim = ["V1c"]
    rows = {v: wording_row(vers[v][REG]) for v in prim}
    worst = max(r["row"] for r in rows.values())
    limited = res["verdict"]["status"] != "신뢰도 확보"
    txt = ROW_TEXT[worst].format(labels="majority" if prim == ["V4"] else "per-annotator")
    if worst == 1 and limited:
        txt = "confirmed on each annotator's labels"
    res["operator_wording"] = {"primary_versions": prim, "rows": rows, "row": worst,
                               "rule": "unweighted point estimate + Wilson 95% upper (SPEC §4); weighted co-reported",
                               "row_if_weighted_co_report": max(r["co_report_weighted"]["row"] for r in rows.values()),
                               "text": ("reliability-limited: " if limited else "") + txt}
    return res


# --------------------------------------------------------------------------- 출력
def write_md(res: dict, p: Path):
    L = ["# R3 §2 reliability (PREREG_R3)", "", "판정: **%s** — %s" % (res["verdict"]["status"], res["verdict"]["R1-3c"]), ""]
    L.append("| 쌍 | n | κ 이진 | 95% CI | 가중 κ | 3범주 κ | PABAK | AC1 | 일치 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    f = lambda x: "—" if x is None else "%.3f" % x
    for k, v in res["pairs"].items():
        L.append("| %s | %d | %s | [%s, %s] | %s | %s | %s | %s | %s |" % (k, v["n"], f(v["kappa_bin"]), f(v["kappa_bin_ci95"][0]),
                 f(v["kappa_bin_ci95"][1]), f(v["kappa_bin_weighted"]), f(v["kappa_3cat"]), f(v["pabak_bin"]), f(v["ac1_bin"]), f(v["agree_bin"])))
    L += ["", "연산자 문구: row %d — %s" % (res["operator_wording"]["row"], res["operator_wording"]["text"]), ""]
    L.append("| 판 | n | precision (가중) | recall (가중) | 회수 (가중) |")
    L.append("|---|---|---|---|---|")
    for v, o in res["operator"].items():
        m = o["MiniCheck (registered_dup, p>0.5)"]
        L.append("| %s | %d | %s (%s) | %s (%s) | %s (%s) |" % (v, o["n"], f(m["precision"]["rate"]), f(m["precision"]["weighted"]),
                 f(m["recall"]["rate"]), f(m["recall"]["weighted"]), f(m["recovery"]["rate"]), f(m["recovery"]["weighted"])))
    io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- 자체 시험
def synth_csv(path: Path, keyv: list[dict], S: dict, annot: str, p_flip: float, seed: int):
    rng = np.random.default_rng(seed)
    cols = ["stage", "item_id", "label", "first_label", "n_changes", "active_ms", "total_ms", "hidden_ms", "t_min_s",
            "translator_used", "translator_tool", "proficiency", "sheet_version", "annotator"]
    rows = []
    for r in keyv:
        a = S["lab"][r["source_item"]]
        lab = a if rng.random() > p_flip else rng.choice([c for c in CATS if c != a])
        tm = float(r["t_min_s"])
        rows.append({"stage": 3 if r["stage"] == "practice" else 4, "item_id": r["sheet_item_id"], "label": lab, "first_label": lab,
                     "n_changes": 0, "active_ms": int(tm * 1000 * rng.uniform(0.9, 3)), "total_ms": 0, "hidden_ms": int(rng.integers(0, 3000)),
                     "t_min_s": tm, "translator_used": rng.choice(["안 씀", "문장만", "근거 일부", "근거 전체"]),
                     "translator_tool": "synthetic", "proficiency": "synthetic", "sheet_version": "synthetic", "annotator": annot})
    with io.open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(rows)


def selftest(B: int) -> int:
    S = load_static()
    # (1) V1 재현 (VPRIME_FINAL: 0.861/0.756, 가중 0.850/0.782) — 실제 저자 라벨
    v100 = sorted(S["key"])
    y = {i: b(S["lab"][i]) for i in v100}
    op = OPS["MiniCheck (registered_dup, p>0.5)"]
    raw, wt = op_metrics(S, v100, y, op, False), op_metrics(S, v100, y, op, True)
    print("V1 precision %.3f recall %.3f / weighted %.3f %.3f" % (raw["precision"], raw["recall"], wt["precision"], wt["recall"]))
    assert (round(raw["precision"], 3), round(raw["recall"], 3), round(wt["precision"], 3), round(wt["recall"], 3)) == (0.861, 0.756, 0.850, 0.782)
    v1 = op_version(S, v100, y, min(B, 500), SEED_OP)["MiniCheck (registered_dup, p>0.5)"]
    wr = wording_row(v1)
    print("V1 wording row (unweighted + Wilson) %d; weighted co-report row %d" % (wr["row"], wr["co_report_weighted"]["row"]))
    assert wr["row"] == 1, wr                     # 0.861 ≥ 0.85, 0.756 ≥ 0.70 → 행 1 (§2.7 V1 재진술)
    # (2) C50-v1 κ 0.024 (저자 vs A-v1, 이진 비가중)
    k = cohen([b(S["lab"][S["c50"][i]]) for i in sorted(S["c50"])], [b(S["av1"][i]) for i in sorted(S["c50"])], [0, 1])
    print("C50-v1 kappa %.4f" % k)
    assert round(k, 3) == 0.024
    # (3) 공식 단위 시험
    assert abs(fleiss([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]], [0, 1]) - 1.0) < 1e-12
    # 위키 예(Fleiss 1971 형식 축약): 2범주 3평정자 손 계산
    rows = [[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0]]
    cnt = [[r.count(0), r.count(1)] for r in rows]
    Pi = [(c0 * c0 + c1 * c1 - 3) / 6 for c0, c1 in cnt]
    pj = [sum(c[0] for c in cnt) / 15, sum(c[1] for c in cnt) / 15]
    hand = (np.mean(Pi) - sum(p * p for p in pj)) / (1 - sum(p * p for p in pj))
    assert abs(fleiss(rows, [0, 1]) - hand) < 1e-12
    assert abs(cohen([1, 1, 0, 0], [1, 1, 0, 0], [0, 1]) - 1) < 1e-12 and pabak([1, 0], [1, 1], 2) == 0.0
    # (4) 합성 CSV 두 벌로 끝까지 (키는 r3_sheet.build_v() 로 다시 만든다 — 봉인 키 파일을 읽지 않는다)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from r3_sheet import build_v
    keyv = [{k2: str(v) for k2, v in r.items()} for r in build_v()["key"]]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        kp = td / "_key_r3_v.csv"
        with io.open(kp, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(keyv[0])); w.writeheader(); w.writerows(keyv)
        out = {}
        for tag, (pa, pb) in {"high_agreement": (0.03, 0.05), "low_agreement": (0.45, 0.5)}.items():
            synth_csv(td / ("A_%s.csv" % tag), keyv, S, "A", pa, 1)
            synth_csv(td / ("B_%s.csv" % tag), keyv, S, "B", pb, 2)
            r = analyze({"A": td / ("A_%s.csv" % tag), "B": td / ("B_%s.csv" % tag)}, kp, B=B, S=S)
            out[tag] = (r["verdict"]["status"], r["operator_wording"]["primary_versions"], r["operator_wording"]["row"])
            print("synthetic %s: verdict %s, κ A %.3f B %.3f, Fleiss %.3f, primary %s, row %d, author flips %d" % (
                tag, r["verdict"]["status"], r["verdict"]["kappa_author_A"], r["verdict"]["kappa_author_B"], r["fleiss_bin"],
                r["operator_wording"]["primary_versions"], r["operator_wording"]["row"], r["majority"]["author_flipped_n"]))
        r1 = analyze({"A": td / "A_high_agreement.csv", "B": td / "missing.csv"}, kp, B=B, S=S)
        print("synthetic one-annotator: %s, primary %s" % (r1["verdict"]["status"], r1["operator_wording"]["primary_versions"]))
        assert out["high_agreement"][0] == "신뢰도 확보" and out["high_agreement"][1] == ["V4"]
        assert out["low_agreement"][0] == "신뢰도 제한" and out["low_agreement"][1] == ["V1c", "V3a", "V3b"]
        assert r1["verdict"]["status"] == "single second annotator"
        # 부분 완료: B 가 50문항 중 20만 단 CSV → 판정은 single, 받은 데까지 기술 통계
        rows = rd(td / "B_high_agreement.csv")
        test_ids = [k["sheet_item_id"] for k in keyv if k["stage"] == "test"]
        for r in rows:
            if r["item_id"] in test_ids[20:]:
                r["label"] = ""
        with io.open(td / "B_partial.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        rp = analyze({"A": td / "A_high_agreement.csv", "B": td / "B_partial.csv"}, kp, B=B, S=S)
        print("synthetic partial B: %s, partial %s" % (rp["verdict"]["status"], {k: v["n_labelled"] for k, v in rp["partial_completion"].items()}))
        assert rp["verdict"]["status"] == "single second annotator" and rp["partial_completion"]["B"]["n_labelled"] == 20
        # 시험판 CSV 는 --sheet-version 으로도 받지 않는다
        for bad in ("r3_sheet_v1-test", "r3_sheet_v1+PLACEHOLDER_REF"):
            try:
                check_csv(td / "A_high_agreement.csv", "A", bad)
                raise RuntimeError("시험판 판을 받아들였다")
            except AssertionError:
                pass
        rows = rd(td / "A_high_agreement.csv")
        for r in rows:
            r["sheet_version"] = "r3_sheet_v1-test"
        with io.open(td / "A_test.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        pv = check_csv(td / "A_test.csv", "A", "r3_sheet_v1")
        assert pv["present"] is False and pv["rejected_test_csv"], pv
    print("selftest OK (values from synthetic labels are discarded)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--labels-dir", default=str(R3 / "labels"))
    ap.add_argument("--key", default=str(R3 / "sealed/_key_r3_v.csv"))
    ap.add_argument("--out", default=str(R3 / "R3_RELIABILITY"))
    ap.add_argument("--sheet-version", default=SHEET_VERSION,
                    help="받아들일 sheet_version (배관 시험용). '-test'·'PLACEHOLDER' 가 든 값은 항상 거부한다")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.B)
    ld = Path(a.labels_dir)
    prov = {x: check_csv(label_path(ld, x), x, a.sheet_version) for x in ("A", "B")}
    use = {x: (label_path(ld, x) if prov[x]["present"] else ld / ("_not_received_%s.csv" % x)) for x in ("A", "B")}
    res = analyze(use, Path(a.key), B=a.B)
    res["csv_provenance"] = prov
    io.open(a.out + ".json", "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    write_md(res, Path(a.out + ".md"))
    print("-> %s.{json,md}: %s / %s" % (a.out, res["verdict"]["status"], res["operator_wording"]["text"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
