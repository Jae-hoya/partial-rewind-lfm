"""PREREG_R2 §7 drift 보고 — L-GR s1 vs Test A, 같은 35토픽 (API 0, GPU 0, 저장값만).

등록 문구(results/r2/PREREG_R2.md:123-125): L-GR s1과 Test A의 같은 35토픽에서 degraded, verify,
coverage 차이를 보고한다. ±10pp를 넘으면 drift로 보고하고 계속 진행한다. Test A는 역사 기록이다.

정의(코드):
  degraded  = truncated or not any(sa.text for sa in sub_answers)   lfm/branch.py:311 (Test A 시점 b9d2afc:201 동일)
  coverage  = 유효 인용이 붙은 sub-answer 비율                       lfm/evidence.py:39-46 (b9d2afc:38-45 동일)
  verify    = mean_g >= tau_g(0.8) and min_c >= tau_c(0.8)          lfm/verifier.py:38-42 (b9d2afc:42 동일)

단위(등록 문구가 정하지 않아 여기서 고정한다):
  degraded  = degraded 인 branch-round 비율 (35 x 3 라운드 x 4 branch = 420)
  verify    = 최종 verify 통과 토픽 비율 (35). 보조: 라운드 상태 통과 비율 (105)
  coverage  = branch-round legacy coverage 평균 (420)
차이 = L-GR s1 - Test A, pp(x100). |차이| > 10pp 이면 drift.

lfm 을 import 하지 않는다(NLI 스택·GPU 회피). csv/json 만 읽는다. 재채점 없음.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TA_DIR = ROOT / "results" / "test_a" / "20260829-041445"
LG_DIR = ROOT / "results" / "r2" / "heldout35" / "L-GR" / "s1"
OUT_DIR = ROOT / "results" / "r2" / "extra"
THRESH_PP = 10.0
ROUNDS = (0, 1, 2)


def read_jsonl(p: Path):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_lgr():
    """branch-round 행과 라운드 verify. 출처: topics/*.json, trace.jsonl 로 교차검증."""
    rows, rverify, final = [], {}, {}
    for f in sorted((LG_DIR / "topics").glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))
        q = t["query_id"]
        assert [r["round"] for r in t["rounds"]] == list(ROUNDS), (q, "rounds")
        final[q] = bool(t["passed"])
        for r in t["rounds"]:
            rverify[(q, r["round"])] = bool(r["verify"]["passed"])
            assert len(r["branches"]) == 4, (q, r["round"])
            for b in r["branches"]:
                assert b["executed"], (q, r["round"], b["branch"], "not executed")
                rows.append({"query_id": q, "round": r["round"], "branch": b["branch"],
                             "degraded": int(bool(b["degraded"])),
                             "coverage": float(b["evidence"]["coverage"]),
                             "grounding": float(b["evidence"]["grounding"])})
        assert final[q] == rverify[(q, 2)] or final[q], q
    # trace 교차검증
    tr = read_jsonl(LG_DIR / "trace.jsonl")
    deg_tr = {(x["query_id"], x["round"]): set(x["action"]["params"]["degraded_branches"]) for x in tr}
    ver_tr = {(x["query_id"], x["round"]): bool(x["action"]["params"]["verify"]["passed"]) for x in tr}
    cov_tr = {(x["query_id"], x["round"], b): v["coverage"] for x in tr for b, v in x["state"]["E"].items()}
    for r in rows:
        k = (r["query_id"], r["round"])
        assert (r["branch"] in deg_tr[k]) == bool(r["degraded"]), ("L-GR degraded trace mismatch", k, r["branch"])
        assert abs(cov_tr[k + (r["branch"],)] - r["coverage"]) < 1e-9, ("L-GR coverage trace mismatch", k)
    for k, v in rverify.items():
        assert ver_tr[k] == v, ("L-GR verify trace mismatch", k)
    return rows, rverify, final


def load_testa(qids: set[str]):
    """branch-round 행: waste_ratio.csv. 라운드 verify: trace.jsonl params.verify.passed."""
    with open(TA_DIR / "waste_ratio.csv", encoding="utf-8") as fh:
        allrows = list(csv.DictReader(fh))
    assert len(allrows) == 600, len(allrows)
    n_deg_all = sum(int(r["degraded"]) for r in allrows)
    res = json.loads((TA_DIR / "results.json").read_text(encoding="utf-8"))
    assert n_deg_all == res["context"]["n_degraded"] == 215, (n_deg_all, res["context"]["n_degraded"])
    assert res["context"]["n_passed"] == 0
    rows = [{"query_id": r["query_id"], "round": int(r["round"]), "branch": r["branch"],
             "degraded": int(r["degraded"]), "coverage": float(r["coverage"]),
             "grounding": float(r["grounding"]), "re_executed": int(r["re_executed"])}
            for r in allrows if r["query_id"] in qids]
    # re_executed = "이 라운드 뒤에 다시 돌렸다"(라운드 2 는 0). 라운드 0·1 이 전부 1 이면 라운드 1·2 는 전부 실행이다.
    assert all(r["re_executed"] == (1 if r["round"] < 2 else 0) for r in rows), "Test A: 실행 안 된 branch-round 존재"
    tr = [x for x in read_jsonl(TA_DIR / "trace.jsonl") if x["query_id"] in qids]
    rverify = {(x["query_id"], x["round"]): bool(x["action"]["params"]["verify"]["passed"]) for x in tr}
    deg_tr = {(x["query_id"], x["round"]): set(x["action"]["params"]["degraded_branches"]) for x in tr}
    cov_tr = {(x["query_id"], x["round"], b): v["coverage"] for x in tr for b, v in x["state"]["E"].items()}
    for r in rows:
        k = (r["query_id"], r["round"])
        assert (r["branch"] in deg_tr[k]) == bool(r["degraded"]), ("Test A degraded trace mismatch", k, r["branch"])
        assert abs(cov_tr[k + (r["branch"],)] - r["coverage"]) < 1e-6, ("Test A coverage trace mismatch", k)
    for q in qids:
        assert sorted(rd for (qq, rd) in rverify if qq == q) == list(ROUNDS), (q, "Test A rounds")
    final = {q: any(rverify[(q, rd)] for rd in ROUNDS) for q in qids}  # GR 은 통과 시 STOP → 마지막 = 임의 라운드 통과
    return rows, rverify, final


def metrics(rows, rverify, final, rounds=ROUNDS):
    br = [r for r in rows if r["round"] in rounds]
    rv = [v for (q, rd), v in rverify.items() if rd in rounds]
    out = {
        "n_branch_rounds": len(br),
        "degraded": sum(r["degraded"] for r in br) / len(br),
        "n_degraded": sum(r["degraded"] for r in br),
        "coverage": sum(r["coverage"] for r in br) / len(br),
        "verify_round": sum(rv) / len(rv),
        "n_round_states": len(rv),
        "n_verify_round_pass": sum(rv),
        # 보조(§7 지표 아님): grounding — verify 가 의존한다
        "grounding_mean": sum(r["grounding"] for r in br) / len(br),
    }
    if tuple(rounds) == ROUNDS:
        out["verify"] = sum(final.values()) / len(final)
        out["n_topics"] = len(final)
        out["n_verify_pass"] = sum(final.values())
    return out


def diff_block(a, b, keys):
    d = {}
    for k in keys:
        pp = (a[k] - b[k]) * 100.0
        d[k] = {"L-GR_s1": round(a[k], 6), "TestA": round(b[k], 6), "diff_pp": round(pp, 3),
                "drift": abs(pp) > THRESH_PP}
    return d


def main():
    lg_rows, lg_rv, lg_final = load_lgr()
    qids = set(lg_final)
    assert len(qids) == 35
    ta_ids = {r["query_id"] for r in csv.DictReader(open(TA_DIR / "waste_ratio.csv", encoding="utf-8"))}
    assert qids <= ta_ids, sorted(qids - ta_ids)
    ta_rows, ta_rv, ta_final = load_testa(qids)
    assert len(lg_rows) == len(ta_rows) == 420
    assert {(r["query_id"], r["round"], r["branch"]) for r in lg_rows} == \
           {(r["query_id"], r["round"], r["branch"]) for r in ta_rows}

    prim_keys = ["degraded", "verify", "coverage"]
    lg_all, ta_all = metrics(lg_rows, lg_rv, lg_final), metrics(ta_rows, ta_rv, ta_final)
    primary = diff_block(lg_all, ta_all, prim_keys)
    aux = diff_block(lg_all, ta_all, ["verify_round"])
    supp = diff_block(lg_all, ta_all, ["grounding_mean"])
    by_round = {}
    for rd in ROUNDS:
        a, b = metrics(lg_rows, lg_rv, lg_final, (rd,)), metrics(ta_rows, ta_rv, ta_final, (rd,))
        by_round[str(rd)] = diff_block(a, b, ["degraded", "verify_round", "coverage", "grounding_mean"])

    # 보조: 라운드 mean_grounding / min_coverage (verify 의 두 성분) 평균
    def comp(trace_path):
        tr = [x for x in read_jsonl(trace_path) if x["query_id"] in qids]
        mg = [x["action"]["params"]["verify"]["mean_grounding"] for x in tr]
        mc = [x["action"]["params"]["verify"]["min_coverage"] for x in tr]
        return {"mean_grounding": sum(mg) / len(mg), "min_coverage": sum(mc) / len(mc), "n": len(tr)}
    vc = {"L-GR_s1": comp(LG_DIR / "trace.jsonl"), "TestA": comp(TA_DIR / "trace.jsonl")}

    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        head = None
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    any_drift = any(v["drift"] for v in primary.values())

    caveats = [
        "Test A(run_id 9448e3b1, 2026-08-29 04:14:45 시작)는 v1.0 시기 코드(9ea222c 04:19 직전, b9d2afc 계열)로 돌았다. "
        "세 지표의 정의는 b9d2afc 와 HEAD 가 글자 그대로 같다: degraded branch.py:201→311, coverage evidence.py:38-45→39-46, "
        "verify verifier.py:42→42. coverage·degraded 는 출력이 주어지면 결정론이다.",
        "저장값만 썼다(재채점 없음). 양쪽 grounding 은 실행 당시 nli_max(cross-encoder/nli-deberta-v3-base, τ 0.5)로 저장됐다. "
        "M1′/R2 MiniCheck 재채점값은 쓰지 않았다. 이탈 2(fb5e6ea, 전제 고유 청크)는 nli_max 에 무관(최댓값). "
        "D15 는 Test A 를 후속 코드로 재생해 degraded 600/600 일치를 확인했다(results/d15/D15_RESULT.md:9).",
        "분해는 같다(Test A 캐시 .cache/llm 재생, PREREG §4). 생성 모델 gpt-5-mini·reasoning_effort low·temperature null, "
        "prompt_mode legacy, retry_directives, max_rounds 3, τ 0.8/0.8 이 두 config 에서 같다. 그러나 L-GR s1 생성은 "
        "seed nonce(s1|L|r{round}) 아래 새 표본이고, 8월→11월 사이 제공자 측 모델이 바뀌었을 수 있다(model_returned 비교는 하지 않음).",
        "verify 는 양쪽 모두 0/35 라 차이 0pp 는 정보가 적다. 라운드 통과 비율(105 상태)도 함께 냈다.",
        "grounding 은 §7 지표가 아니다(보조 행, drift 판정에 넣지 않음). verify 는 mean_g 에 의존한다. "
        "grounding 은 +31pp 로 크게 다르다. 계산식은 같다(b9d2afc evidence.grounding = HEAD estimator.py NLIMaxEstimator: "
        "(청크, 문장) 쌍 최댓값 ≥ 0.5). D18 은 Test A 를 후속 코드로 재생해 grounding 이 waste_ratio.csv 와 일치함을 가드로 확인했고"
        "(experiments/d18_textdump.py:10), P0 는 Test A pilot5 재생이 60/60 바이트 일치했다(432bec7). 따라서 차이는 채점 코드가 아니라 "
        "생성 출력(새 표본·제공자 측 변화)에서 온다고 본다 — 원인은 이 스크립트가 진단하지 않는다.",
        "단위(branch-round 평균, 토픽 비율)는 등록 문구가 정하지 않아 이 스크립트에서 고정했다. coverage 는 branch 비율의 "
        "평균이다 — Test A csv 에 sub-answer 개수가 없어 sub-answer 풀링 평균은 계산할 수 없다.",
        "Test A 50토픽 중 heldout35 의 35토픽만 비교한다(나머지 15는 제외). Test A 는 역사 기록이며 결론은 같은 시기 arm 끼리만 비교한다(PREREG §7).",
    ]

    result = {
        "rule": "PREREG_R2 §7 (results/r2/PREREG_R2.md:123-125)",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": "experiments/r2_drift.py", "script_sha256_16": sha, "git_head": head,
        "sources": {"L-GR_s1": str(LG_DIR.relative_to(ROOT)).replace("\\", "/"),
                    "TestA": str(TA_DIR.relative_to(ROOT)).replace("\\", "/")},
        "topics": sorted(qids),
        "threshold_pp": THRESH_PP, "diff_sign": "L-GR_s1 - TestA",
        "definitions": {
            "degraded": "branch-round 비율; degraded = truncated or not any(sa.text) — lfm/branch.py:311",
            "coverage": "branch-round 평균 legacy coverage — lfm/evidence.py:39-46",
            "verify": "최종 verify 통과 토픽 비율; mean_g>=0.8 and min_c>=0.8 — lfm/verifier.py:38-42",
        },
        "primary": primary, "any_drift": any_drift,
        "aux_verify_round": aux, "supplementary_not_in_rule": supp,
        "verify_components_round_mean": vc,
        "by_round": by_round,
        "counts": {"L-GR_s1": lg_all, "TestA": ta_all},
        "caveats": caveats,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "DRIFT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def row(name, d, unit_pct=True):
        f = (lambda x: "%.1f%%" % (100 * x)) if unit_pct else (lambda x: "%.3f" % x)
        return "| %s | %s | %s | %+.1f | %s |" % (name, f(d["L-GR_s1"]), f(d["TestA"]), d["diff_pp"],
                                                 "**drift**" if d["drift"] else "—")
    L = ["# R2 drift — L-GR s1 vs Test A (PREREG §7)", "",
         "자동 생성(`experiments/r2_drift.py`, SHA-256 `%s`), HEAD `%s`, %s. API 0, GPU 0, 저장값만." % (sha, head, result["generated_at"]),
         "", "같은 35토픽, branch-round 420개(35×3라운드×4branch), 라운드 상태 105개. 차이 = L-GR s1 − Test A (pp). 기준 ±%.0fpp." % THRESH_PP,
         "", "## 등록 지표", "",
         "| 지표 | L-GR s1 | Test A | 차이(pp) | 판정 |", "|---|---|---|---|---|",
         row("degraded (branch-round 비율)", primary["degraded"]),
         row("verify (최종 통과 토픽 비율, %d/%d vs %d/%d)" % (lg_all["n_verify_pass"], 35, ta_all["n_verify_pass"], 35), primary["verify"]),
         row("coverage (branch-round 평균)", primary["coverage"]),
         "", "**drift: %s**" % ("있음 — 보고하고 계속 진행(§7)" if any_drift else "없음"), "",
         "## 보조(판정 밖)", "",
         "| 지표 | L-GR s1 | Test A | 차이(pp) | ±10pp 초과 |", "|---|---|---|---|---|",
         row("verify 라운드 통과 비율(105)", aux["verify_round"]),
         row("grounding 평균(nli_max 저장값, §7 아님)", supp["grounding_mean"]),
         "", "라운드 verify 성분 평균: mean_grounding L-GR %.3f / Test A %.3f; min_coverage L-GR %.3f / Test A %.3f." % (
             vc["L-GR_s1"]["mean_grounding"], vc["TestA"]["mean_grounding"], vc["L-GR_s1"]["min_coverage"], vc["TestA"]["min_coverage"]),
         "", "## 라운드별", "",
         "| 라운드 | degraded L / A (pp) | coverage L / A (pp) | grounding L / A (pp) | verify 라운드 L / A |", "|---|---|---|---|---|"]
    for rd in ROUNDS:
        b = by_round[str(rd)]
        g = lambda k: "%.1f / %.1f (%+.1f)" % (100 * b[k]["L-GR_s1"], 100 * b[k]["TestA"], b[k]["diff_pp"])
        L.append("| %d | %s | %s | %s | %s |" % (rd, g("degraded"), g("coverage"), g("grounding_mean"), g("verify_round")))
    L += ["", "## 정의", ""] + ["- %s: %s" % kv for kv in result["definitions"].items()]
    L += ["", "## 주의", ""] + ["- " + c for c in caveats]
    L += ["", "검증(assert): 35토픽 ⊂ Test A 50; 양쪽 (토픽, 라운드, branch) 집합 동일·전부 실행; Test A csv 600행·degraded 215 = results.json; "
          "양쪽 topics/csv 의 degraded·coverage·verify 가 각자 trace.jsonl 과 일치."]
    (OUT_DIR / "DRIFT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({"primary": primary, "any_drift": any_drift, "aux": aux, "supp": supp,
                      "by_round": by_round, "vc": vc}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
