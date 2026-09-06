"""D10 — 계열 일반화. 같은 자료를 NLI 체크포인트 4종에 건다. **생성 호출 0회.**

논문은 "sentence-pair NLI 계열"이라고 계열 단위로 주장하는데 실제로 잰 것은
`cross-encoder/nli-deberta-v3-base` 하나뿐이다. 게다가 **base 크기**다. 심사자는
"large 면 274토큰에서 안 무너질 수 있다"고 쓸 것이고 지금 우리는 답이 없다.
D7-0 의 희석, D8-A 의 정밀, D8-B 의 맹목 — 세 기둥이 전부 체크포인트 하나 위에 있다.

여기서 재는 것은 딱 둘이고, 둘 다 이미 확정된 절차를 그대로 다시 돌리는 것이다.

  P. 양성 통제 재현 — D7-0 의 padding_control 표를 체크포인트마다 다시 만든다.
     전제 길이 희석이 **계열 성질**인지 이 모델 버릇인지가 여기서 갈린다.
  G. 격자 재현 — 294 문장 × 인용 청크 전제로 grounded@0.5 를 다시 잰다.

**사전 등록 성격**: 게이트 없음. 판정 기준을 새로 만들지 않는다. 기존 측정의
robustness check 이므로 어떤 결과가 나와도 채택·탈락 결정에 쓰지 않는다.
쓰임은 하나뿐이다 — 논문의 "계열" 주장이 근거를 얻거나 잃는다.

양성 쌍은 **현행 체크포인트 기준**으로 고르고 그 집합을 고정해 전 체크포인트에
같은 쌍을 건다. 그래야 "현행이 양성이라 한 것을 남들은 어떻게 보는가"가 성립한다.
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

from diag_sweep import CONDITIONS, pick_topics  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

# 문헌에서 흔히 쓰는 sentence-pair NLI 체크포인트. 크기를 섞는다.
CHECKPOINTS = [
    "cross-encoder/nli-deberta-v3-base",     # 현행 (base)
    "cross-encoder/nli-deberta-v3-large",    # 같은 계열 large — "large 면 다르다" 반론 정면
    "roberta-large-mnli",                    # 다른 사전학습 · 다른 학습 데이터
    "facebook/bart-large-mnli",              # 다른 구조
]
MULTIPLIERS = (1, 2, 3, 4)
POS_THRESHOLD = 0.90   # D7-0 과 동일: 원래 p_entail > 0.90 인 쌍만 양성 통제에 쓴다


def collect(cfg, ccfg, picked):
    """캐시 재생으로 (전제, 가설) 쌍과 패딩 재료를 모은다. 생성 호출 0."""
    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    grid = []      # 문장 가설 × 인용 청크 전제
    filler = []    # 패딩용 무관 청크
    for n, t in enumerate(picked):
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            for c in o.cited_chunks():
                filler.append(c.text)
            for sa in o.sub_answers:
                cited = [cmap[c].text for c in sa.citations if c in cmap]
                if not cited:
                    continue
                for sent in split_sentences(sa.text):
                    grid.append({"topic": tid, "branch": p.role, "sentence": sent,
                                 "premises": cited})
        print("  [%2d/%2d] %s  문장 %d" % (n + 1, len(picked), tid, len(grid)), flush=True)
    return grid, filler


def main(argv=None):
    ap = argparse.ArgumentParser(description="NLI 체크포인트 계열 일반화 (생성 호출 0회)")
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

    nli0 = load_nli(ccfg)     # torch/CUDA 를 먼저 (뒤에 올리면 세그폴트)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))

    print("자료 수집 (캐시 재생, 생성 호출 0) ...")
    grid, filler = collect(cfg, ccfg, picked)
    print("문장 %d · 패딩 재료 청크 %d" % (len(grid), len(filler)))

    print("양성 쌍 선정 (현행 체크포인트 기준, p_entail > %.2f) ..." % POS_THRESHOLD)
    pos = []
    for r in grid:
        scores = nli0.entailment([(p, r["sentence"]) for p in r["premises"]])
        b = max(range(len(scores)), key=lambda i: scores[i])
        if scores[b] > POS_THRESHOLD:
            pos.append((r["premises"][b], r["sentence"]))
    print("양성 쌍 %d개" % len(pos))
    del nli0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pad_rows, grid_rows, summary = [], [], []

    for ck in [x.strip() for x in args.checkpoints.split(",") if x.strip()]:
        print("")
        print("=" * 70)
        print("체크포인트: %s" % ck)
        print("=" * 70, flush=True)
        d2 = ccfg.as_dict()
        d2["nli"]["model"] = ck
        try:
            nli = load_nli(Config(d2, source="d10:" + ck))
        except Exception as e:                                  # noqa: BLE001
            print("  적재 실패, 건너뛴다: %s" % e, flush=True)
            summary.append({"checkpoint": ck, "status": "load_failed", "detail": str(e)[:200]})
            continue

        # --- P. 양성 통제 (내용 고정, 전제 길이만 늘린다) ----------------------
        keep = {}
        for m in MULTIPLIERS:
            ps, ntok = [], []
            for i, (prem, hyp) in enumerate(pos):
                ext = prem
                for k in range(m - 1):
                    ext = ext + "\n\n" + filler[(i * 7 + k) % len(filler)]
                sc = nli.entailment([(ext, hyp)])[0]
                ps.append(sc)
                ntok.append(len(nli.tokenizer.encode(ext, truncation=False)))
                pad_rows.append({"checkpoint": ck, "multiplier": m, "pair_idx": i,
                                 "premise_tokens": ntok[-1], "p_entail": round(sc, 6)})
            keep[m] = 100.0 * sum(1 for x in ps if x >= 0.5) / len(ps)
            print("  %dx  전제토큰 중앙 %4d  p 중앙 %.3f  >=0.5 유지 %.1f%%"
                  % (m, int(st.median(ntok)), st.median(ps), keep[m]), flush=True)

        # --- G. 격자 (문장 가설 × 인용 청크 전제) ------------------------------
        gs = []
        for r in grid:
            gs.append(max(nli.entailment([(p, r["sentence"]) for p in r["premises"]])))
        for r, s_ in zip(grid, gs):
            grid_rows.append({"checkpoint": ck, "topic": r["topic"], "branch": r["branch"],
                              "n_premises": len(r["premises"]), "max_p": round(s_, 6)})
        grounded = 100.0 * sum(1 for x in gs if x >= 0.5) / len(gs)
        below = 100.0 * sum(1 for x in gs if x < 0.05) / len(gs)
        mid = 100.0 * sum(1 for x in gs if 0.1 <= x <= 0.9) / len(gs)
        print("  격자: grounded@0.5 %.1f%%  <0.05 %.1f%%  중간대 %.1f%%  중앙 %.3f"
              % (grounded, below, mid, st.median(gs)), flush=True)

        summary.append({"checkpoint": ck, "status": "ok", "n_pos_pairs": len(pos),
                        "keep_1x": round(keep[1], 1), "keep_2x": round(keep[2], 1),
                        "keep_3x": round(keep[3], 1), "keep_4x": round(keep[4], 1),
                        "grid_n": len(gs), "grounded_at_05": round(grounded, 2),
                        "below_005": round(below, 2), "mid_band": round(mid, 2),
                        "median_max_p": round(st.median(gs), 4)})
        del nli

    for name, rows in [("d10_padding_by_checkpoint.csv", pad_rows),
                       ("d10_grid_by_checkpoint.csv", grid_rows),
                       ("d10_summary.csv", summary)]:
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        with io.open(outdir / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    print("")
    print("=== 계열 일반화 요약 ===")
    print("| 체크포인트 | 1x | 2x | 3x | 4x | grounded@0.5 | <0.05 | 중간대 |")
    print("|---|---|---|---|---|---|---|---|")
    for s_ in summary:
        if s_.get("status") != "ok":
            print("| %s | 적재 실패 |" % s_["checkpoint"])
            continue
        print("| %s | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.1f%% | %.1f%% | %.1f%% |"
              % (s_["checkpoint"], s_["keep_1x"], s_["keep_2x"], s_["keep_3x"], s_["keep_4x"],
                 s_["grounded_at_05"], s_["below_005"], s_["mid_band"]))
    print("")
    print("-> " + str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
