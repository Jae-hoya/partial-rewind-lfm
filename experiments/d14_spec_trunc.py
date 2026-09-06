"""D14 — 스펙의 `trunc` 절차를 실제로 구현한다. 스펙 검증의 마지막 미결. **호출 0회.**

`D7_FINDINGS.md` §미결 1번:

> 스펙의 `trunc`(BM25 상위 L=64 토큰)와 여기 쓴 문장 분할은 다른 절차다.
> `union`/`single` 대칭 적용과 `premise_tokens_after_tokenizer` 중앙값 10% 이내
> 일치 검증은 하지 않았다. 본격 재측정 전에 스펙대로 한 번 더 맞춰야 한다.

D7-2 는 전제를 **문장으로 쪼개** 최댓값을 잡았다. 스펙이 요구한 것은 다르다 —
가설과의 BM25 점수로 전제를 **L=64 토큰까지 줄이는** 것이다. 취지는 같으나 절차가
다르고, 절차가 다르면 수치도 다르다. 여기서 스펙대로 맞춘다.

  single  청크마다 따로 trunc(L) 하고 각각 채점해 **최댓값**
  union   trunc 한 청크들을 **이어붙여 하나의 전제**로 만들고 1회 채점
          -> union 은 최댓값 집계가 없다. D8-A 가 밝힌 팽창 경로가 원천 차단된다.

**대칭 검증**: 두 조건의 `premise_tokens_after_tokenizer` 중앙값이 10% 이내로
일치해야 비교가 성립한다. 안 맞으면 그 사실을 보고한다 — 맞추려고 L 을 조정하지 않는다.

**D13 이 세운 예측**: 절벽은 181 토큰이고 L=64 는 그 한참 아래다. 따라서 trunc 는
길이 희석을 원천 제거해야 한다. **이 예측을 여기서 검증한다.** 스펙이 이 값을 고른
것은 D13 이전이므로 사후 맞춤이 아니다.

L 스윕(32/64/128/256)은 **부차**다. 스펙 값은 64 이고 판정은 64 로만 한다.
스윕은 절벽 곡선과의 정합을 보는 기술 통계다.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from d10_checkpoint_family import collect  # noqa: E402
from diag_sweep import CONDITIONS, pick_topics  # noqa: E402

from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

SPEC_L = 64
L_SWEEP = (32, 64, 128, 256)
TAU = 0.5
TOKEN = re.compile(r"[a-z0-9]+")


def toks(s: str) -> list[str]:
    return TOKEN.findall(s.lower())


def bm25_rank(query: str, docs: list[str]) -> list[int]:
    """가설과의 BM25 점수 내림차순 인덱스. 결정론, 모델 없음. d8a2 와 같은 식."""
    dts = [toks(d) for d in docs]
    n = len(dts) or 1
    avgdl = (sum(len(d) for d in dts) / n) or 1.0
    df: Counter[str] = Counter()
    for d in dts:
        df.update(set(d))
    q = toks(query)
    scored = []
    for i, d in enumerate(dts):
        tf = Counter(d)
        dl = len(d) or 1
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * dl / avgdl))
        scored.append((-s, i))
    scored.sort()
    return [i for _, i in scored]


def trunc(chunk: str, hypothesis: str, L: int, tokenizer) -> str:
    """청크를 가설과의 BM25 상위 문장으로 L 토큰까지 줄인다. 원문 순서는 보존한다."""
    sents = split_sentences(chunk) or [chunk]
    if len(sents) == 1:
        ids = tokenizer.encode(sents[0], truncation=False)
        if len(ids) <= L:
            return sents[0]
        return tokenizer.decode(ids[:L], skip_special_tokens=True)
    order = bm25_rank(hypothesis, sents)
    keep, used = set(), 0
    for i in order:
        t = len(tokenizer.encode(sents[i], truncation=False))
        if used + t > L and keep:
            break
        keep.add(i)
        used += t
        if used >= L:
            break
    return " ".join(sents[i] for i in sorted(keep)) if keep else sents[order[0]]


def main(argv=None):
    ap = argparse.ArgumentParser(description="스펙 trunc 구현 · 대칭 검증 (호출 0회)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--outdir", default="results/diag/d14")
    ap.add_argument("--checkpoints",
                    default="cross-encoder/nli-deberta-v3-base,facebook/bart-large-mnli")
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

    load_nli(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))
    print("자료 수집 (캐시 재생, 호출 0) ...")
    grid, _ = collect(cfg, ccfg, picked)
    print("문장 %d" % len(grid))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []

    for ck in [x.strip() for x in args.checkpoints.split(",") if x.strip()]:
        print("")
        print("=" * 74)
        print("체크포인트: %s" % ck, flush=True)
        d2 = ccfg.as_dict()
        d2["nli"]["model"] = ck
        try:
            nli = load_nli(Config(d2, source="d14:" + ck))
        except Exception as e:                                  # noqa: BLE001
            print("  적재 실패: %s" % e, flush=True)
            continue
        tk = nli.tokenizer
        ntok = lambda s: len(tk.encode(s, truncation=False))    # noqa: E731

        conds = [("whole_max", None), ("sentsplit_max", None)]
        conds += [("single_L%d" % L, L) for L in L_SWEEP]
        conds += [("union_L%d" % L, L) for L in L_SWEEP]

        for cname, L in conds:
            ps, tl = [], []
            for r in grid:
                h = r["sentence"]
                if cname == "whole_max":
                    prem = r["premises"]
                    p_ = max(nli.entailment([(x, h) for x in prem]))
                    tl.append(st.median([ntok(x) for x in prem]))
                elif cname == "sentsplit_max":
                    prem = []
                    for c in r["premises"]:
                        prem.extend(split_sentences(c) or [c])
                    p_ = max(nli.entailment([(x, h) for x in prem]))
                    tl.append(st.median([ntok(x) for x in prem]))
                elif cname.startswith("single"):
                    prem = [trunc(c, h, L, tk) for c in r["premises"]]
                    p_ = max(nli.entailment([(x, h) for x in prem]))
                    tl.append(st.median([ntok(x) for x in prem]))
                else:
                    prem = " ".join(trunc(c, h, L, tk) for c in r["premises"])
                    p_ = nli.entailment([(prem, h)])[0]
                    tl.append(ntok(prem))
                ps.append(p_)
                rows.append({"checkpoint": ck, "condition": cname, "topic": r["topic"],
                             "premise_tokens_after_tokenizer": int(tl[-1]),
                             "p_entail": round(p_, 6)})
            g = 100.0 * sum(1 for x in ps if x >= TAU) / len(ps)
            med_t = st.median(tl)
            summary.append({"checkpoint": ck, "condition": cname, "n": len(ps),
                            "premise_tokens_median": round(med_t, 1),
                            "grounded_at_05": round(g, 4),
                            "median_p": round(st.median(ps), 4),
                            "below_005": round(100.0 * sum(1 for x in ps if x < 0.05) / len(ps), 4)})
            print("  %-16s 전제토큰중앙 %5.0f   grounded %5.1f%%   중앙 p %.3f"
                  % (cname, med_t, g, st.median(ps)), flush=True)
        del nli

    keys = sorted({k for r in summary for k in r})
    with io.open(outdir / "d14_summary.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(summary)
    keys = sorted({k for r in rows for k in r})
    with io.open(outdir / "d14_rows.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    # --- 대칭 검증: single_L64 vs union_L64 의 전제 토큰 중앙값 10% 이내 ---------
    print("")
    print("=== 스펙 대칭 검증 — single_L64 vs union_L64 전제 토큰 중앙값 ===")
    print("| 체크포인트 | single | union | 차이 | 10%% 이내 |")
    print("|---|---|---|---|---|")
    for ck in sorted({s["checkpoint"] for s in summary}):
        a = next((s for s in summary if s["checkpoint"] == ck
                  and s["condition"] == "single_L%d" % SPEC_L), None)
        b = next((s for s in summary if s["checkpoint"] == ck
                  and s["condition"] == "union_L%d" % SPEC_L), None)
        if not a or not b:
            continue
        x, y = a["premise_tokens_median"], b["premise_tokens_median"]
        d = abs(x - y) / max(x, y) * 100 if max(x, y) else 0.0
        print("| %s | %.0f | %.0f | %.1f%% | %s |"
              % (ck, x, y, d, "예" if d <= 10 else "**아니오**"))
    print("")
    print("union 은 청크를 이어붙이므로 single 보다 길 수밖에 없다. 10%% 를 넘으면")
    print("그건 절차 오류가 아니라 **두 조건이 애초에 다른 길이라는 사실**이다.")
    print("맞추려고 L 을 조정하지 않는다 — 사실대로 보고한다.")
    print("")
    print("-> " + str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
