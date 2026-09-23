"""부록 A·B·D·E·F 생성. 호출 0. (C 감사 축약은 공저자 몫이라 자리만 둔다.)

공저자 지시 순서 — "검토자가 재현하려고 열 때의 순서":
  A 프롬프트 전문 → B 토픽 예시 3개 → C 추정자 감사 축약 → D 임계 민감도
  → E leakage audit → F 재현 절차

**프롬프트는 손으로 옮기지 않는다.** `lfm/*.py` 와 `prompts/*.txt` 에서 직접 읽어
쓴다. 옮겨 적으면 코드와 원고가 갈라지고, 그게 오늘 내내 잡은 오류의 부류다.

숫자도 전부 원행에서 다시 센다. 요약 CSV 를 읽지 않는다.

**토픽 예시 선정은 시드 고정 무작위**로 한다. "잘 된 것/빈 답안/충돌 있는 것"을
성격별로 고르면 독자에게 친절하지만 그건 제가 고른 것이고, 부록의 목적은 대표성이
아니라 재현이다. 성격별 선정을 원하시면 규칙을 주십시오 — 그때 이유를 부록에 적겠다.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import statistics as st
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/appendix"
SEED = 42


def rd(p: str):
    q = ROOT / p
    return list(csv.DictReader(io.open(q, encoding="utf-8"))) if q.exists() else []


def f(r, k):
    return float(r[k])


def pct(rows, pred):
    return 100.0 * sum(1 for r in rows if pred(r)) / len(rows) if rows else 0.0


def source_strings(path: str, names: list[str]) -> dict[str, str]:
    """모듈 소스에서 상수 문자열을 **AST 로** 뽑는다. import 하지 않는다."""
    tree = ast.parse(io.open(ROOT / path, encoding="utf-8").read())
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in names:
                try:
                    out[t.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return out


def appendix_a() -> str:
    L = ["## Appendix A. Prompts",
         "",
         "All prompts are frozen for the study and reproduced here verbatim from the "
         "source; they are not transcribed. Placeholders in braces are filled at "
         "call time.", ""]
    blocks: list[tuple[str, str, str]] = []
    for path, names in [
        ("lfm/perspectives.py", ["SYSTEM", "PROMPT"]),
        ("lfm/branch.py", ["ANSWER_SYSTEM", "ANSWER_PROMPT", "CLAIM_SYSTEM",
                           "CLAIM_PROMPT", "PARAPHRASE_SYSTEM", "PARAPHRASE_PROMPT"]),
        ("lfm/jsonio.py", ["REPAIR_SYSTEM", "REPAIR_PROMPT"]),
    ]:
        got = source_strings(path, names)
        for n in names:
            if n in got:
                blocks.append((path, n, got[n]))
    for p in sorted((ROOT / "prompts").glob("*.txt")):
        blocks.append(("prompts/" + p.name, p.stem, io.open(p, encoding="utf-8").read()))

    for src, name, body in blocks:
        L += ["### %s (`%s`)" % (name, src), "", "```", body.rstrip(), "```", ""]
    L.append("Total: %d prompts." % len(blocks))
    L.append("")
    return "\n".join(L)


def appendix_b() -> str:
    topics = yaml.safe_load(io.open(ROOT / "data/topics_pilot50.yaml",
                                    encoding="utf-8"))["topics"]
    rng = np.random.default_rng(SEED)
    idx = sorted(rng.choice(len(topics), size=3, replace=False).tolist())
    L = ["## Appendix B. Example topics",
         "",
         "Three of the %d approved topics, drawn with a fixed seed (%d); no topic was "
         "chosen for its outcome. Each shows the research question, the target article "
         "held out of the index, and the search anchor terms."
         % (len(topics), SEED), ""]
    for k in idx:
        t = topics[k]
        L += ["### %s" % t["id"], "",
              "**Question.** %s" % t["topic"], "",
              "| | |", "|---|---|",
              "| Target article | %s |" % t["target_article"],
              "| Identifier | %s |" % t["target_identifier"],
              "| Anchor terms | %s |" % ", ".join(t.get("terms") or []),
              "| Related documents in corpus | %s |" % t.get("related_docs"),
              "| Primary infobox type | %s |" % t.get("primary_infobox_type"), ""]
    return "\n".join(L)


def appendix_d() -> str:
    g = rd("results/diag/d7/pool_unit_grid.csv")
    sp = rd("results/diag/d7/pool_unit_grid_sentprem.csv")
    cfgs = [("Sentence x chunk (deployed)", [f(r, "p_cited") for r in g
                                             if r["unit"] == "sentence"]),
            ("Claim x chunk", [f(r, "p_cited") for r in g if r["unit"] == "claim"]),
            ("Sentence x chunk-sentence", [f(r, "p_cited") for r in sp
                                           if r["unit"] == "sentence"]),
            ("Claim x chunk-sentence", [f(r, "p_cited") for r in sp
                                        if r["unit"] == "claim"])]
    taus = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    L = ["## Appendix D. Threshold sensitivity",
         "",
         "Grounded fraction as $\\tau_g$ is swept over the full range, for the four "
         "estimator configurations of Section VI. Counted from the raw per-sentence "
         "scores, not from rounded summaries.", "",
         "| Configuration | " + " | ".join("%.2f" % t for t in taus) + " | Range |",
         "|---" * (len(taus) + 2) + "|"]
    for name, v in cfgs:
        row = [100.0 * sum(1 for x in v if x >= t) / len(v) for t in taus]
        L.append("| %s | " % name + " | ".join("%.1f" % x for x in row)
                 + " | %.1f pp |" % (row[0] - row[-1]))
    L += ["",
          "Moving the threshold across its entire range changes the grounded fraction "
          "by 10 to 19 points. Changing the estimator configuration changes it by 41.5. "
          "Threshold calibration is not the lever.", ""]
    return "\n".join(L)


def appendix_e() -> str:
    g = rd("results/diag/d7/pool_unit_grid.csv")
    S = [r for r in g if r["unit"] == "sentence"]
    low = [r for r in S if f(r, "p_cited") < 0.05]
    rem = [r for r in low if f(r, "p_retrieved") < 0.05]
    up = [r for r in rem if f(r, "p_target") >= 0.5]
    still = [r for r in rem if f(r, "p_target") < 0.05]
    L = ["## Appendix E. Parametric leakage audit",
         "",
         "The generator has seen Wikipedia in pretraining, so it may state true facts "
         "from memory that lie in parts of the target article the retriever never "
         "returned. We bound that by scoring the same sentences against **all** chunks "
         "of the held-out target article.", "",
         "| | Sentences |", "|---|---|",
         "| Scoring below 0.05 against cited chunks | %d |" % len(low),
         "| Still below 0.05 against every retrieved chunk | %d (%.1f%%) |"
         % (len(rem), 100.0 * len(rem) / len(low)),
         "| Of those, reaching 0.5 against the full target article | **%d (%.1f%%)** |"
         % (len(up), 100.0 * len(up) / len(rem)),
         "| Of those, still below 0.05 | %d (%.1f%%) |"
         % (len(still), 100.0 * len(still) / len(rem)), "",
         "Leakage exists and is small: about one sentence in ten that no retrieved "
         "chunk supports is supported somewhere in the target article. It does not "
         "account for the bulk of the unsupported mass.", ""]
    return "\n".join(L)


def appendix_f() -> str:
    return "\n".join([
        "## Appendix F. Reproduction",
        "",
        "### Environment",
        "",
        "```",
        "python 3.11 · transformers 4.42.3 · torch 2.3.1+cu118 · numpy 1.26.4",
        "bm25s · tiktoken (o200k_base) · pyarrow",
        "```",
        "",
        "### Models",
        "",
        "| Role | Model |",
        "|---|---|",
        "| Generation | `gpt-5-mini`, reasoning_effort=low, max_tokens=2000 |",
        "| Judge (gates only) | `gpt-5.4-nano` |",
        "| Standard operator $g_1$ | `cross-encoder/nli-deberta-v3-base`, "
        "$\\tau_{\\mathrm{ground}}=0.5$ |",
        "| Accepted operator | `lytang/MiniCheck-Flan-T5-Large`, single-document "
        "premise, threshold $>0.5$ (strict) |",
        "| Relation (3-way) | `facebook/bart-large-mnli` |",
        "",
        "### Installing MiniCheck — a name collision to avoid",
        "",
        "```bash",
        'pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git"',
        "```",
        "",
        "**Do not run `pip install minicheck`.** The package of that name on PyPI is a "
        "different project — an explicit-state model checker. It installs cleanly and "
        "`import minicheck` succeeds, so the mistake surfaces only later, when "
        "`minicheck.minicheck` is not found. The repository install adds "
        "`minicheck-0.1.0` alone and does not alter transformers, torch, or numpy; it "
        "downloads the nltk `punkt` tokenizer on first import.",
        "",
        "### Corpus",
        "",
        "| | |",
        "|---|---|",
        "| Source | HuggingFace `wikimedia/structured-wikipedia` |",
        "| Revision | `417c267bb457fa645c22eb3b5c77764963194c70` |",
        "| Rows loaded | 150,000 → 142,972 unique (4.69% duplicates) |",
        "| Full index | 142,922 documents / 685,683 chunks |",
        "| Partial held-out index | 142,972 / 686,221 |",
        "| Chunking | one chunk per section; sections over 1,200 tokens split at "
        "paragraph boundaries |",
        "",
        "### Numerical conventions",
        "",
        "- Every percentage-point difference is computed from raw per-row rates, never "
        "by subtracting two rounded values.",
        "- Summary CSVs store four decimals; scripts read the per-row files instead, "
        "because re-rounding a rounded value shifts figures (15.6463% became 15.7% "
        "this way once).",
        "- Proportions carry Wilson 95% intervals; means over queries carry "
        "query-level bootstrap intervals (B=10,000, seed 42).",
        "- Every grounded fraction is reported with its premise pool scope stated, "
        "because max-aggregation rises with pool size.",
        "",
        "### Verification",
        "",
        "```bash",
        "python experiments/verify_numbers.py           # 112 reported figures vs raw data",
        "python experiments/verify_manuscript.py        # manuscript VII/VIII vs raw data",
        "python experiments/verify_manuscript_part1.py  # manuscript VI vs raw data",
        "python -m pytest -q                            # unit tests",
        "```",
        "",
        "Each exits non-zero on any mismatch and reports, rather than silently skips, "
        "any figure it cannot locate in the manuscript.",
        "",
    ])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = [("A_prompts", appendix_a()), ("B_topics", appendix_b()),
             ("D_threshold_sensitivity", appendix_d()),
             ("E_leakage_audit", appendix_e()), ("F_reproduction", appendix_f())]
    for name, body in parts:
        io.open(OUT / ("appendix_%s.md" % name), "w", encoding="utf-8").write(body)
        print("  %-28s %5d 자" % (name, len(body)))

    combined = ["# Appendices", "",
                "> Appendix C (condensed estimator audit) is written by the co-author "
                "and is not included here.", ""]
    for name, body in parts:
        combined.append(body)
        if name == "B_topics":
            combined += ["## Appendix C. Condensed estimator audit", "",
                         "[[공저자 작성 — v0.9 에서 본문에 안 들어간 것만]]", ""]
    io.open(OUT / "appendices.md", "w", encoding="utf-8").write("\n".join(combined))
    print("")
    print("합본 -> %s" % (OUT / "appendices.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
