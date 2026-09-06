"""D10-N — 체크포인트 4종의 **음성 통제**. 생성 호출 0회.

D10 은 bart-large-mnli 가 grounded 60.5% 를 낸다고만 말한다. **그 수만 보고 채택하면
D7 에서 저지른 실수를 그대로 반복하는 것이다.** D7 은 57.1% 를 보고 "고쳤다"고 했고,
D8-A 가 그 구성의 위양성률이 50.5% 임을 밝혀 철회됐다.

그러므로 판정 지표는 grounded 가 아니라 **순증 = 정상 grounded − 교란 grounded** 다.

교란은 D8-A 와 **같은 규칙**이다: 토픽 i 의 문장에 토픽 (i+7) mod 15 의 전제 풀을 준다.
같은 토픽 배정이 0건임을 실행 전 assert 로 확인한다.

전제 풀은 정상·교란 양쪽 모두 **토픽 단위 인용 청크**로 맞춘다. 풀 크기가 다르면
최댓값 집계 때문에 비교가 성립하지 않는다 (D8-A 가 밝힌 바로 그 함정).
따라서 여기 '정상' 값은 D10 격자(sub-answer 단위 풀)와 다를 수 있고, 그건 정상이다.

**게이트**: D8-A 와 동일하게 교란 grounded <= 5%. 실행 전 고정, 사후 변경 금지.
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from d10_checkpoint_family import CHECKPOINTS  # noqa: E402
from d8a_negative_control import SHIFT, spearman  # noqa: E402
from diag_sweep import CONDITIONS, pick_topics  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

PASS_THRESHOLD = 5.0   # 교란 grounded 백분율 상한. D8-A 와 동일.


def main(argv=None):
    ap = argparse.ArgumentParser(description="체크포인트 4종 음성 통제 (생성 호출 0회)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--outdir", default="results/diag/d10")
    ap.add_argument("--checkpoints", default=",".join(CHECKPOINTS))
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

    load_nli(ccfg)            # torch/CUDA 를 먼저 (뒤에 올리면 세그폴트)
    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))

    print("자료 수집 (캐시 재생, 생성 호출 0) ...")
    sents_by_topic: dict[str, list[str]] = {}
    pool_by_topic: dict[str, list[str]] = {}
    for n, t in enumerate(picked):
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        sents, pool = [], []
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            for c in o.cited_chunks():
                pool.append(c.text)
            for sa in o.sub_answers:
                if any(c in cmap for c in sa.citations):
                    sents.extend(split_sentences(sa.text))
        sents_by_topic[tid] = sents
        pool_by_topic[tid] = pool
        print("  [%2d/%2d] %s  문장 %d · 전제 청크 %d"
              % (n + 1, len(picked), tid, len(sents), len(pool)), flush=True)

    tids = [str(t["id"]) for t in picked]
    mapping = {tids[i]: tids[(i + SHIFT) % len(tids)] for i in range(len(tids))}
    assert all(a != b for a, b in mapping.items()), "같은 토픽 배정이 있다"
    print("")
    print("교란 배정 (i -> i+%d mod %d), 같은 토픽 0건 확인" % (SHIFT, len(tids)))
    print("전제 풀은 정상·교란 모두 **토픽 단위** 인용 청크 (풀 크기를 맞춘다)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []

    for ck in [x.strip() for x in args.checkpoints.split(",") if x.strip()]:
        print("")
        print("=" * 70)
        print("체크포인트: %s" % ck, flush=True)
        d2 = ccfg.as_dict()
        d2["nli"]["model"] = ck
        try:
            nli = load_nli(Config(d2, source="d10n:" + ck))
        except Exception as e:                                  # noqa: BLE001
            print("  적재 실패: %s" % e, flush=True)
            summary.append({"checkpoint": ck, "status": "load_failed"})
            continue

        norm, shuf, psize = [], [], []
        for tid in tids:
            sents = sents_by_topic[tid]
            own, other = pool_by_topic[tid], pool_by_topic[mapping[tid]]
            if not sents or not own or not other:
                continue
            for s_ in sents:
                a = max(nli.entailment([(p, s_) for p in own]))
                b = max(nli.entailment([(p, s_) for p in other]))
                norm.append(a)
                shuf.append(b)
                psize.append(float(len(own)))
                rows.append({"checkpoint": ck, "src_topic": tid,
                             "premise_topic": mapping[tid], "n_premise_chunks": len(own),
                             "p_normal": round(a, 6), "p_shuffled": round(b, 6)})
        gn = 100.0 * sum(1 for x in norm if x >= 0.5) / len(norm)
        gs = 100.0 * sum(1 for x in shuf if x >= 0.5) / len(shuf)
        rho = spearman(psize, norm)
        ok = gs <= PASS_THRESHOLD
        print("  n=%d  정상 %.2f%%  교란 %.2f%%  **순증 %.2f%%p**  rho(풀,p) %+.3f  게이트 %s"
              % (len(norm), gn, gs, gn - gs, rho, "PASS" if ok else "FAIL"), flush=True)
        summary.append({"checkpoint": ck, "status": "ok", "n": len(norm),
                        "pool_median": st.median(psize),
                        "grounded_normal": round(gn, 4), "grounded_shuffled": round(gs, 4),
                        "net": round(gn - gs, 4), "rho_poolsize_p": round(rho, 3),
                        "gate_pass": ok})
        del nli

    for name, data_ in [("d10n_negative_by_checkpoint.csv", rows),
                        ("d10n_summary.csv", summary)]:
        if not data_:
            continue
        keys = sorted({k for r in data_ for k in r})
        with io.open(outdir / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(data_)

    print("")
    print("=== 음성 통제 요약 — 판정 지표는 순증이다 ===")
    print("| 체크포인트 | n | 풀 중앙 | 정상 | 교란(위양성) | **순증** | rho | 게이트(<=5%) |")
    print("|---|---|---|---|---|---|---|---|")
    for s_ in summary:
        if s_.get("status") != "ok":
            print("| %s | 적재 실패 |" % s_["checkpoint"])
            continue
        print("| %s | %d | %.0f | %.1f%% | %.1f%% | **%.1f%%p** | %+.3f | %s |"
              % (s_["checkpoint"], s_["n"], s_["pool_median"], s_["grounded_normal"],
                 s_["grounded_shuffled"], s_["net"], s_["rho_poolsize_p"],
                 "PASS" if s_["gate_pass"] else "FAIL"))
    okl = [s_ for s_ in summary if s_.get("gate_pass")]
    print("")
    if okl:
        best = max(okl, key=lambda s_: s_["net"])
        print("게이트 통과 중 순증 최대: %s (%.1f%%p)" % (best["checkpoint"], best["net"]))
    else:
        print("게이트를 통과한 체크포인트가 없다.")
    print("-> " + str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
