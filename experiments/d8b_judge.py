"""D8-B — 판정 표본 검증. **이 파일만 생성 LLM 호출이 허용된다** (CLAUDE.md D8-D).

D8-A 에서 게이트를 통과한 구성은 현행 추정자(E1: 문장 가설 × 인용 청크 전제)뿐이다.
따라서 스펙 v1.1 §D8-B 의 "A 통과 구성이 없으면 현행" 조항이 아니라 "A 를 통과한
구성" 조항이 적용되고, 표본 단위는 **문장**이다 (E1 이 판정하는 단위가 문장이므로).

  G층  max_p >= 0.5   precision 을 잰다 — "지지" 판정이 옳은가
  N층  max_p < 0.05   **false negative rate 를 잰다 — 이것이 D8-G 게이트다**
  M층  0.05~0.5       기록만

통과 기준 (실행 전 고정, 사후 변경 금지)
  G층 precision >= 0.85
  N층 지지 비율(FNR) <= 0.25
D8-G 분기: N층 FNR >= 0.5 이면 추정자 계열 교체 경로, < 0.5 이면 논문 절단.

판정자 gpt-5.4-nano (CLAUDE.md 모델 정책). temperature 미지정(모델 기본).
프롬프트는 prompts/d8b_judge.txt 에 고정하고 실행 후 수정하지 않는다.

사람 게이트: 사용자가 25건(G12/N9/M4)을 판정자 결과를 **보지 않고** 직접 판정한다.
이 스크립트는 그 서식을 `d8b_human25.csv` 로 내보내고, 채워지면 `--score-human` 으로
일치율을 계산한다.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_sweep import CONDITIONS, pick_topics  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

STRATA = {"G": (0.5, 1.01, 60), "N": (0.0, 0.05, 40), "M": (0.05, 0.5, 20)}
HUMAN_N = {"G": 12, "N": 9, "M": 4}
GATE_G_PRECISION = 0.85
GATE_N_FNR = 0.25
LABELS = ("supported", "unsupported", "unclear")


def parse_label(text: str) -> tuple[str, str]:
    lab, reason = "unclear", ""
    for line in (text or "").splitlines():
        m = re.match(r"\s*LABEL\s*:\s*(\w+)", line, re.I)
        if m and m.group(1).lower() in LABELS:
            lab = m.group(1).lower()
        m = re.match(r"\s*REASON\s*:\s*(.+)", line, re.I)
        if m:
            reason = m.group(1).strip()
    return lab, reason


def collect(cfg, ccfg, picked) -> list[dict]:
    """캐시 재생으로 문장 단위 표본 모집단을 만든다. 생성 호출 0."""
    nli = load_nli(ccfg)
    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    pop: list[dict] = []
    for n, t in enumerate(picked):
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            for sa in o.sub_answers:
                cited = [cmap[c] for c in sa.citations if c in cmap]
                if not cited:
                    continue
                for sent in split_sentences(sa.text):
                    scores = nli.entailment([(c.text, sent) for c in cited])
                    b = max(range(len(scores)), key=lambda i: scores[i])
                    pop.append({
                        "id": "%s:%s:%d" % (tid, p.role, len(pop)),
                        "topic": tid, "branch": p.role, "sentence": sent,
                        "max_p": round(float(scores[b]), 6),
                        "best_chunk": cited[b].text,
                        "all_cited": "\n\n".join(c.text for c in cited),
                        "claims": " | ".join(o.claims[:20]),
                    })
        print("  [%2d/%2d] %s  누적 %d" % (n + 1, len(picked), tid, len(pop)), flush=True)
    return pop


def stratify(pop: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for name, (lo, hi, want) in STRATA.items():
        pool = [r for r in pop if lo <= r["max_p"] < hi]
        take = min(want, len(pool))
        idx = sorted(rng.choice(len(pool), size=take, replace=False).tolist()) if pool else []
        for i in idx:
            r = dict(pool[i])
            r["stratum"] = name
            out.append(r)
        print("  %s층: 모집단 %d -> 표본 %d (목표 %d)" % (name, len(pool), take, want))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D8-B 판정 표본 (판정 호출만 허용)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--judge", default="gpt-5.4-nano")
    ap.add_argument("--outdir", default="results/diag/d8")
    ap.add_argument("--score-human", action="store_true",
                    help="사람이 채운 d8b_human25.csv 로 게이트를 채점한다")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(str(repo_root / ".env"))
    cfg = load_config(args.config or (repo_root / "config.yaml"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.score_human:
        return score_human(outdir)

    spec = CONDITIONS[args.condition]
    data = cfg.as_dict()
    if spec["top_k"] is not None:
        data["retrieval"]["top_k"] = spec["top_k"]
    if spec["index"] == "partial":
        data["paths"]["index_dir"] = data["paths"]["index_dir_partial"]
    ccfg = Config(data, source=args.condition)

    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))

    print("표본 모집단 수집 (캐시 재생, 생성 호출 0) ...")
    pop = collect(cfg, ccfg, picked)
    print("문장 %d개" % len(pop))
    print()
    sample = stratify(pop, int(cfg["seed"]))

    # --- 판정 ---------------------------------------------------------
    jdata = cfg.as_dict()
    jdata["llm"]["model"] = args.judge
    jdata["llm"]["max_tokens"] = 256
    judge_llm = LLM(Config(jdata, source="d8b_judge"))
    template = io.open(repo_root / "prompts/d8b_judge.txt", encoding="utf-8").read()

    rows = []
    for i, r in enumerate(sample):
        evidence = r["best_chunk"] if r["stratum"] == "G" else r["all_cited"]
        res = judge_llm.call(
            template.format(evidence=evidence[:6000], claim=r["sentence"]),
            tag="d8b_judge", max_tokens=256,
        )
        lab, reason = parse_label(res.text)
        rows.append({"claim_id": r["id"], "stratum": r["stratum"], "topic": r["topic"],
                     "max_p": r["max_p"], "judge_label": lab, "judge_reason": reason,
                     "sentence": r["sentence"]})
        if (i + 1) % 20 == 0:
            print("  판정 %d/%d" % (i + 1, len(sample)), flush=True)

    with io.open(outdir / "d8b_judge_sample.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- 사람 판정용 25건 (판정 결과 미포함) ------------------------------
    rng = np.random.default_rng(int(cfg["seed"]) + 1)
    human = []
    for st, k in HUMAN_N.items():
        pool = [r for r in sample if r["stratum"] == st]
        idx = sorted(rng.choice(len(pool), size=min(k, len(pool)), replace=False).tolist())
        for i in idx:
            r = pool[i]
            human.append({"claim_id": r["id"], "stratum": r["stratum"],
                          "evidence": (r["best_chunk"] if st == "G" else r["all_cited"])[:4000],
                          "claim": r["sentence"], "human_label": ""})
    with io.open(outdir / "d8b_human25.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(human[0].keys()))
        w.writeheader()
        w.writerows(human)

    # --- 집계 ---------------------------------------------------------
    print()
    print("=== D8-B 판정 결과 ===")
    print("| 층 | n | supported | unsupported | unclear | 비율 |")
    print("|---|---|---|---|---|---|")
    stats = {}
    for st in ("G", "N", "M"):
        sub = [r for r in rows if r["stratum"] == st]
        if not sub:
            continue
        c = Counter(r["judge_label"] for r in sub)
        rate = c["supported"] / len(sub)
        stats[st] = rate
        print("| %s | %d | %d | %d | %d | %.1f%% |"
              % (st, len(sub), c["supported"], c["unsupported"], c["unclear"], 100 * rate))
    print()
    g_prec, n_fnr = stats.get("G", 0.0), stats.get("N", 0.0)
    print("G층 precision = %.3f   (기준 >= %.2f)  %s"
          % (g_prec, GATE_G_PRECISION, "PASS" if g_prec >= GATE_G_PRECISION else "FAIL"))
    print("N층 FNR       = %.3f   (기준 <= %.2f)  %s"
          % (n_fnr, GATE_N_FNR, "PASS" if n_fnr <= GATE_N_FNR else "FAIL"))
    print()
    print("D8-G 분기: N층 FNR %.3f -> %s"
          % (n_fnr, "추정자 계열 교체 경로 (>=0.5)" if n_fnr >= 0.5
             else "논문 절단 — 답안이 검색 근거에서 나오지 않는다 (<0.5)"))
    print()
    print("판정 호출 %d (캐시 %d), 청구 토큰 in/out = %d/%d, 모델 %s"
          % (judge_llm.stats.calls, judge_llm.stats.cache_hits,
             judge_llm.stats.billed_tokens_in, judge_llm.stats.billed_tokens_out, args.judge))
    print()
    print("사람 게이트 25건 -> %s  (human_label 열을 supported/unsupported/unclear 로 채운 뒤"
          % (outdir / "d8b_human25.csv"))
    print("  --score-human 으로 일치율을 계산한다. 판정자 결과는 그 파일에 없다.)")
    return 0


def score_human(outdir: Path) -> int:
    hp = outdir / "d8b_human25.csv"
    jp = outdir / "d8b_judge_sample.csv"
    human = {r["claim_id"]: r["human_label"].strip().lower()
             for r in csv.DictReader(io.open(hp, encoding="utf-8"))}
    judge = {r["claim_id"]: r["judge_label"] for r in csv.DictReader(io.open(jp, encoding="utf-8"))}
    filled = {k: v for k, v in human.items() if v in LABELS}
    if not filled:
        print("human_label 이 비어 있다. 채운 뒤 다시 실행하라.")
        return 1
    agree = sum(1 for k, v in filled.items() if judge.get(k) == v)
    rate = agree / len(filled)
    print("사람 판정 %d건 중 판정자와 일치 %d (%.1f%%)  기준 >= 90%%  -> %s"
          % (len(filled), agree, 100 * rate, "PASS" if rate >= 0.9 else "FAIL"))
    if rate < 0.9:
        print("게이트 미달. 스펙대로 판정자를 gpt-5-mini 로 교체해 1회 재시행한다.")
    return 0 if rate >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
