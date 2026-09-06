"""D14-N — trunc 조건의 **음성 통제**. 이걸 안 하면 또 D7 을 반복한다. 호출 0회.

D14 는 `union_L256` 에서 **현행 g1 이 grounded 67.0%** 를 낸다고 말한다. 청크 통째로는
15.6% 였다. 모델을 안 바꾸고 전제 구성만 바꿔서 51%p 가 올랐다.

**그 수만 보고 채택하면 안 된다.** 그리고 여기엔 D8-A 때보다 더 뚜렷한 의심이 있다:
`trunc` 는 **가설과 BM25 점수가 높은 문장을 골라 남긴다.** 교란 배정(다른 토픽 전제)에
걸어도 BM25 는 그 안에서 **가장 어휘가 겹치는 문장**을 골라낸다. 즉 trunc 자체가
위양성을 만들어 낼 수 있는 구조다. 표면 유사성을 최대화하도록 설계된 절차이기 때문이다.

교란은 D8-A/D10-N/D9-A 와 **같은 규칙** (i -> i+7 mod 15).
**게이트도 같다: 교란 grounded <= 5%.** 실행 전 고정, 사후 변경 금지.
판정은 **순증**으로만 한다.

---

**전제 범위 정정 (2026-09-06, 1차 실행 후).**

1차 실행은 정상·교란 양쪽에 **토픽 단위 풀**(중앙 7 청크)을 썼다. 그런데 D14 의 정상
값은 **sub-answer 인용 청크**(중앙 3)로 잰 것이다. 범위가 다르니 D14 의 57.8% 와
1차 D14-N 의 8.84% 가 어긋났다 — union 은 청크를 이어붙이므로 풀이 커지면 전제가
길어지고 희석된다. **E2 의 57.1 vs 62.3 과 같은 부류의 오류를 두 번째로 저질렀다.**

이 판에서는 **범위를 맞춘다.**

  정상   그 문장이 실제로 인용한 청크 (D14 와 동일)
  교란   **같은 개수**의 청크를 다른 토픽 풀에서 결정론적으로 뽑는다

개수를 맞추지 않으면 최댓값 집계와 union 길이가 양쪽에서 달라져 비교가 성립하지
않는다. 뽑기는 시드 없는 결정론이다 — 교란 토픽 풀을 정렬 순서 그대로 앞에서부터,
문장 인덱스만큼 회전시켜 겹치지 않게 고른다.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from d14_spec_trunc import trunc  # noqa: E402
from d8a_negative_control import SHIFT  # noqa: E402
from diag_sweep import CONDITIONS, pick_topics  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

TAU = 0.5
GATE = 5.0
CONDS = [("whole_max", None, "max"), ("single_L64", 64, "max"), ("union_L64", 64, "union"),
         ("union_L128", 128, "union"), ("union_L256", 256, "union"),
         ("single_L128", 128, "max"), ("single_L256", 256, "max")]


def main(argv=None):
    ap = argparse.ArgumentParser(description="trunc 음성 통제 (호출 0회)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--outdir", default="results/diag/d14")
    ap.add_argument("--checkpoints", default="cross-encoder/nli-deberta-v3-base")
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
    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))

    print("자료 수집 (캐시 재생, 호출 0) ...")
    # 문장마다 **그 문장이 실제로 인용한 청크**를 들고 다닌다 (D14 와 같은 범위).
    items_by: dict[str, list[dict]] = {}
    pool_by: dict[str, list[str]] = {}
    for t in picked:
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        items, pool = [], []
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            for c in o.cited_chunks():
                pool.append(c.text)
            for sa in o.sub_answers:
                cited = [cmap[c].text for c in sa.citations if c in cmap]
                if not cited:
                    continue
                for sent in split_sentences(sa.text):
                    items.append({"sentence": sent, "premises": cited})
        items_by[tid], pool_by[tid] = items, pool
        print("  %s  문장 %d · 토픽 풀 %d" % (tid, len(items), len(pool)), flush=True)

    tids = [str(t["id"]) for t in picked]
    mapping = {tids[i]: tids[(i + SHIFT) % len(tids)] for i in range(len(tids))}
    assert all(a != b for a, b in mapping.items())
    print("")
    print("교란 배정 (i -> i+%d mod %d), 같은 토픽 0건 확인" % (SHIFT, len(tids)))
    print("**전제 개수를 맞춘다** — 교란도 정상과 같은 수의 청크를 다른 토픽에서 뽑는다.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []

    for ck in [x.strip() for x in args.checkpoints.split(",") if x.strip()]:
        print("")
        print("=" * 74)
        print("체크포인트: %s" % ck, flush=True)
        d2 = ccfg.as_dict()
        d2["nli"]["model"] = ck
        nli = load_nli(Config(d2, source="d14n:" + ck))
        tk = nli.tokenizer

        for cname, L, agg in CONDS:
            def score(hyp: str, chunks: list[str]) -> float:
                if L is None:
                    return max(nli.entailment([(c, hyp) for c in chunks]))
                cut = [trunc(c, hyp, L, tk) for c in chunks]
                if agg == "union":
                    return nli.entailment([(" ".join(cut), hyp)])[0]
                return max(nli.entailment([(c, hyp) for c in cut]))

            norm, shuf, ntok_n, ntok_s = [], [], [], []
            for tid in tids:
                other = pool_by[mapping[tid]]
                if not items_by[tid] or not other:
                    continue
                for k, it in enumerate(items_by[tid]):
                    own = it["premises"]
                    m = len(own)
                    # 같은 개수를 다른 토픽 풀에서 결정론적으로 회전 선택
                    fake = [other[(k * m + j) % len(other)] for j in range(m)]
                    a = score(it["sentence"], own)
                    b = score(it["sentence"], fake)
                    norm.append(a); shuf.append(b)
                    ntok_n.append(sum(len(tk.encode(x, truncation=False)) for x in own))
                    ntok_s.append(sum(len(tk.encode(x, truncation=False)) for x in fake))
                    rows.append({"checkpoint": ck, "condition": cname, "src_topic": tid,
                                 "n_chunks": m, "p_normal": round(a, 6),
                                 "p_shuffled": round(b, 6)})
            gn = 100.0 * sum(1 for x in norm if x >= TAU) / len(norm)
            gs = 100.0 * sum(1 for x in shuf if x >= TAU) / len(shuf)
            ok = gs <= GATE
            summary.append({"checkpoint": ck, "condition": cname, "n": len(norm),
                            "grounded_normal": round(gn, 4), "grounded_shuffled": round(gs, 4),
                            "net": round(gn - gs, 4), "gate_pass": ok})
            print("  %-14s n=%d  정상 %5.2f%%  교란 %5.2f%%  **순증 %6.2f%%p**  %s"
                  % (cname, len(norm), gn, gs, gn - gs, "PASS" if ok else "**FAIL**"),
                  flush=True)
        del nli

    for name, d_ in [("d14n_negative.csv", rows), ("d14n_summary.csv", summary)]:
        keys = sorted({k for r in d_ for k in r})
        with io.open(outdir / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(d_)

    print("")
    print("=== 판정 — 순증으로만 ===")
    print("| 구성 | 정상 | 교란(위양성) | **순증** | 게이트(<=5%) |")
    print("|---|---|---|---|---|")
    for s_ in summary:
        print("| %s | %.2f%% | %.2f%% | **%.2f%%p** | %s |"
              % (s_["condition"], s_["grounded_normal"], s_["grounded_shuffled"],
                 s_["net"], "PASS" if s_["gate_pass"] else "**FAIL**"))
    print("")
    print("대조 — D14 의 정상값(같은 범위): whole 15.6 · single_L64 31.6 · union_L64 34.0")
    print("      · union_L128 57.8 · union_L256 67.0 · single_L128 42.2 · single_L256 30.3")
    print("정상값이 이와 맞아야 범위가 같다는 확인이 된다.")
    print("-> " + str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
