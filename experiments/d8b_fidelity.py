"""D8-B 부수 검사 — claim 분해가 원문의 사실을 보존하는가. 판정 호출만 허용.

**스펙과 단위가 다르다. 바꾼 이유를 먼저 적는다.**

스펙은 "문장 30개 → 그 문장의 claim 집합"을 재라고 했다. 그런데 `lfm/branch.py`
는 claim 을 **문장이 아니라 branch 답안 전체**에서 한 번에 분해한다(`CLAIM_PROMPT`
의 `{answer}` 는 `answer_text`). 따라서 문장 → claim 매핑이 자료에 **존재하지 않는다.**
없는 매핑을 만들어 내면 그건 측정이 아니라 창작이다. 그래서 자료가 지지하는
단위인 **branch 답안 → claim 집합**으로 잰다. 이 변경은 결과와 함께 보고한다.

이 단위가 오히려 논문에 더 맞다. 표 4·6 의 claim 행 546개가 바로 이렇게 만들어진
claim 이므로, 그 claim 집합이 답안을 보존하는지가 그 표의 전제이기 때문이다.

판정자·프롬프트는 D8-B 와 같다. `prompts/d8b_fidelity.txt` 는 실행 후 수정하지 않는다.
표본 30건, 시드 고정. 사전 등록된 통과 기준은 없다 — 부수 검사이므로 기록만 한다.
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
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

LABELS = ("complete", "omission", "addition")
SAMPLE_N = 30


def parse_label(text: str) -> tuple[str, str]:
    lab, reason = "", ""
    for line in (text or "").splitlines():
        m = re.match(r"\s*LABEL\s*:\s*(\w+)", line, re.I)
        if m and m.group(1).lower() in LABELS:
            lab = m.group(1).lower()
        m = re.match(r"\s*REASON\s*:\s*(.+)", line, re.I)
        if m:
            reason = m.group(1).strip()
    return (lab or "unparsed"), reason


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="claim 충실도 부수 검사 (판정 호출만)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--judge", default="gpt-5.4-nano")
    ap.add_argument("--outdir", default="results/diag/d8")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(str(root / ".env"))
    cfg = load_config(root / "config.yaml")
    spec = CONDITIONS[args.condition]
    data = cfg.as_dict()
    if spec["top_k"] is not None:
        data["retrieval"]["top_k"] = spec["top_k"]
    if spec["index"] == "partial":
        data["paths"]["index_dir"] = data["paths"]["index_dir_partial"]
    ccfg = Config(data, source=args.condition)

    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))

    print("모집단 수집 (캐시 재생, 생성 호출 0) ...")
    pop: list[dict] = []
    for n, t in enumerate(picked):
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            answer = "\n".join(sa.text for sa in o.sub_answers).strip()
            if not answer or not o.claims:
                continue
            pop.append({"id": "%s:%s" % (tid, p.role), "topic": tid, "branch": p.role,
                        "answer": answer, "claims": list(o.claims),
                        "n_sentences": len(split_sentences(answer)),
                        "n_claims": len(o.claims)})
        print("  [%2d/%2d] %s  누적 %d" % (n + 1, len(picked), tid, len(pop)), flush=True)

    if not pop:
        print("모집단이 비었다."); return 1
    rng = np.random.default_rng(int(cfg["seed"]))
    take = min(SAMPLE_N, len(pop))
    idx = sorted(rng.choice(len(pop), size=take, replace=False).tolist())
    sample = [pop[i] for i in idx]
    print("모집단 %d branch -> 표본 %d" % (len(pop), take))

    jdata = cfg.as_dict()
    jdata["llm"]["model"] = args.judge
    jdata["llm"]["max_tokens"] = 256
    judge = LLM(Config(jdata, source="d8b_fidelity"))
    template = io.open(root / "prompts/d8b_fidelity.txt", encoding="utf-8").read()

    rows = []
    for i, r in enumerate(sample):
        res = judge.call(
            template.format(sentence=r["answer"][:6000],
                            claims="\n".join("- " + c for c in r["claims"])),
            tag="d8b_fidelity", max_tokens=256)
        lab, reason = parse_label(res.text)
        rows.append({"branch_id": r["id"], "topic": r["topic"], "branch": r["branch"],
                     "n_sentences": r["n_sentences"], "n_claims": r["n_claims"],
                     "label": lab, "reason": reason})
        if (i + 1) % 10 == 0:
            print("  판정 %d/%d" % (i + 1, take), flush=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "d8b_fidelity.csv"
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    c = Counter(r["label"] for r in rows)
    print()
    print("=== claim 충실도 (단위: branch 답안, 스펙의 '문장'이 아니다) ===")
    print("| 판정 | n | 비율 |")
    print("|---|---|---|")
    for k in LABELS + ("unparsed",):
        if c[k]:
            print("| %s | %d | %.1f%% |" % (k, c[k], 100 * c[k] / len(rows)))
    print()
    print("complete %.1f%% — 나머지는 omission %d · addition %d"
          % (100 * c["complete"] / len(rows), c["omission"], c["addition"]))
    print("표본의 문장/claim: 문장 중앙 %d, claim 중앙 %d"
          % (int(np.median([r["n_sentences"] for r in rows])),
             int(np.median([r["n_claims"] for r in rows]))))
    print()
    print("판정 호출 %d (캐시 %d), 청구 %d/%d, 모델 %s"
          % (judge.stats.calls, judge.stats.cache_hits,
             judge.stats.billed_tokens_in, judge.stats.billed_tokens_out, args.judge))
    print("-> " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
