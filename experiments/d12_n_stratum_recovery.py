"""D12 — **결론을 가르는 칸.** N층 회수율. 생성 호출 0회, 판정 호출 0회.

D8-B 는 현행 추정자가 `max_p < 0.05` 로 버린 40 문장 중 **29개가 실제로는 지지된다**고
했다(FNR 72.5%). 논문의 결론 문장 "표준 추정자가 근거를 못 본다"는 이 29건에 걸려 있다.

여기서 묻는다: **다른 추정자는 그 29건을 되찾는가?**

  되찾는다  -> "못 본다"가 확증된다. 근거는 거기 있었고 base 가 못 봤을 뿐이다.
  못 되찾는다 -> 판정자가 후했거나 근거가 실제로 없다. **논문 결론이 뒤집힌다.**

이건 D9 스펙이 MiniCheck 의 핵심 칸으로 잡은 것과 같은 질문인데, NLI 체크포인트에
대해서는 **지금 당장 호출 0회로** 답할 수 있다. 같은 문장, 같은 전제, 모델만 다르다.

**대조군을 반드시 같이 본다.** N층만 보면 관대한 모델이 무조건 이긴다. 그래서
D8-B 가 `unsupported` 로 판정한 11건(N층)과 G층 42건을 함께 본다.

  회수율   = N층 supported 29건 중 되찾은 비율   (높을수록 좋다)
  오회수율 = N층 unsupported 11건 중 되찾은 비율 (낮을수록 좋다 — 판정자 기준 위양성)
  유지율   = G층 supported 42건 중 계속 잡는 비율 (높을수록 좋다)

세 값이 같이 움직이면 그냥 관대한 것이고, 회수율만 오르면 진짜 개선이다.
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

from d10_checkpoint_family import CHECKPOINTS  # noqa: E402
from diag_sweep import CONDITIONS, pick_topics  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

TAU = 0.5


def main(argv=None):
    ap = argparse.ArgumentParser(description="N층 회수율 (호출 0회)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--outdir", default="results/diag/d12")
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

    load_nli(ccfg)          # torch/CUDA 를 먼저
    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))

    # --- D8-B 표본과 같은 방식으로 모집단을 재구성해 전제를 되살린다 ----------
    # d8b_judge.py 의 collect() 와 동일한 순서·동일한 id 규칙을 쓴다.
    print("모집단 재구성 (캐시 재생, 호출 0) ...")
    pop: dict[str, dict] = {}
    n_seen = 0
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
                    key = "%s:%s:%d" % (tid, p.role, n_seen)
                    pop[key] = {"sentence": sent,
                                "premises": [c.text for c in cited]}
                    n_seen += 1
        print("  [%2d/%2d] %s  누적 %d" % (n + 1, len(picked), tid, n_seen), flush=True)

    jb = list(csv.DictReader(io.open(root / "results/diag/d8/d8b_judge_sample.csv",
                                     encoding="utf-8")))
    matched, missing = [], []
    for r in jb:
        src = pop.get(r["claim_id"])
        if src is None or src["sentence"] != r["sentence"]:
            missing.append(r["claim_id"])
            continue
        matched.append({**r, "premises": src["premises"]})
    print("")
    print("D8-B 판정 표본 %d건 중 전제 복원 %d건, 실패 %d건"
          % (len(jb), len(matched), len(missing)))
    if missing:
        print("  복원 실패 id (앞 5개):", missing[:5])
    if len(matched) < 0.9 * len(jb):
        print("복원율이 90%% 미만이다. id 규칙이 어긋났을 수 있으므로 중단한다.")
        return 1

    groups = {
        "N층 supported (회수 대상)":
            [r for r in matched if r["stratum"] == "N" and r["judge_label"] == "supported"],
        "N층 unsupported (오회수 대조)":
            [r for r in matched if r["stratum"] == "N" and r["judge_label"] == "unsupported"],
        "G층 supported (유지 대조)":
            [r for r in matched if r["stratum"] == "G" and r["judge_label"] == "supported"],
        "M층 supported (참고)":
            [r for r in matched if r["stratum"] == "M" and r["judge_label"] == "supported"],
    }
    for k, v in groups.items():
        print("  %-30s n=%d" % (k, len(v)))

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
            nli = load_nli(Config(d2, source="d12:" + ck))
        except Exception as e:                                  # noqa: BLE001
            print("  적재 실패: %s" % e, flush=True)
            continue
        s_ = {"checkpoint": ck}
        for gname, items in groups.items():
            if not items:
                continue
            hit = 0
            for r in items:
                p_ = max(nli.entailment([(pr, r["sentence"]) for pr in r["premises"]]))
                if p_ >= TAU:
                    hit += 1
                rows.append({"checkpoint": ck, "claim_id": r["claim_id"],
                             "stratum": r["stratum"], "judge_label": r["judge_label"],
                             "base_max_p": r["max_p"], "new_max_p": round(p_, 6),
                             "grounded": int(p_ >= TAU)})
            rate = 100.0 * hit / len(items)
            s_[gname] = round(rate, 4)
            print("  %-30s %2d/%2d = %5.1f%%" % (gname, hit, len(items), rate), flush=True)
        summary.append(s_)
        del nli

    for name, d_ in [("d12_recovery_rows.csv", rows), ("d12_summary.csv", summary)]:
        if not d_:
            continue
        keys = sorted({k for r in d_ for k in r})
        with io.open(outdir / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(d_)

    print("")
    print("=== N층 회수율 — 논문 결론을 가르는 표 ===")
    hdr = list(groups)
    print("| 체크포인트 | " + " | ".join("%s (n=%d)" % (h, len(groups[h])) for h in hdr) + " |")
    print("|" + "---|" * (len(hdr) + 1))
    for s_ in summary:
        print("| %s | " % s_["checkpoint"]
              + " | ".join("%.1f%%" % s_.get(h, float("nan")) for h in hdr) + " |")
    print("")
    print("읽는 법: 1열이 오르면서 2열이 안 오르면 진짜 개선이다.")
    print("         두 열이 같이 오르면 그냥 관대한 것이고 D8-A 순증에서 걸러진다.")
    print("-> " + str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
