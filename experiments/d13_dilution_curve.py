"""D13 — 희석은 어디서 시작되는가. 곡선과 위치 효과. **호출 0회.**

D7-0 은 점 **네 개**(1x~4x)다. "긴 전제에서 무너진다"까지만 말하고 **어디서** 무너지는지
말하지 못한다. 실무 규칙("전제는 N토큰 이하로")을 주려면 곡선이 필요하다.

세 가지를 잰다.

  A. **절벽 위치.** 패딩 배수를 촘촘히(1.0~6.0) 늘려 전제 토큰 수 대 p_entail 을
     체크포인트별로 그린다. `p >= 0.5` 를 지키는 마지막 토큰 수가 실무 상한이다.
  B. **위치 효과.** 같은 길이의 패딩을 **앞**에 붙일 때와 **뒤**에 붙일 때가 다른가.
     다르면 그 자체가 결과다 — 희석이 아니라 절단·주의 편향이라는 뜻이 된다.
  C. **자연 상태의 곡선.** 인위적 패딩이 아니라 **실제 (청크, 문장) 쌍** 전체에서
     전제 토큰 수와 p_entail 의 관계를 본다. A 가 인공물인지 실제 현상인지 가른다.

**사전 등록 성격**: 게이트 없음. 판정 기준을 만들지 않는다. D7-0 을 점에서 곡선으로
늘리는 기술 통계다. 어떤 결과가 나와도 채택·탈락 결정에 쓰지 않는다.

C 는 A·B 와 달리 교란 요인이 있다. 긴 청크는 내용도 다르다(긴 섹션 = 개요·목록일
가능성). **따라서 C 는 상관이지 인과가 아니다**. A 가 인과 쪽이고 C 는 그 인과가
실제 분포에서도 보이는지를 확인하는 용도다. 이 구분을 보고서에 명시한다.
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

from d10_checkpoint_family import CHECKPOINTS, collect  # noqa: E402
from diag_sweep import CONDITIONS, pick_topics  # noqa: E402

from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402

POS_THRESHOLD = 0.90                  # D7-0 과 동일한 양성 쌍 기준
STEPS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
TAU = 0.5


def pad(premise: str, filler: list[str], mult: float, i: int, where: str) -> str:
    """전제 길이를 mult 배로 늘린다. 내용(원 전제)은 그대로 둔다."""
    if mult <= 1.0:
        return premise
    want = int(len(premise) * (mult - 1.0))
    got, k = [], 0
    n = 0
    while n < want and k < len(filler):
        piece = filler[(i * 7 + k) % len(filler)]
        got.append(piece)
        n += len(piece)
        k += 1
    extra = "\n\n".join(got)[:want]
    return (extra + "\n\n" + premise) if where == "front" else (premise + "\n\n" + extra)


def main(argv=None):
    ap = argparse.ArgumentParser(description="희석 곡선 · 위치 효과 (호출 0회)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--outdir", default="results/diag/d13")
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

    nli0 = load_nli(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))

    print("자료 수집 (캐시 재생, 호출 0) ...")
    grid, filler = collect(cfg, ccfg, picked)
    print("문장 %d · 패딩 재료 %d" % (len(grid), len(filler)))

    print("양성 쌍 선정 (현행 기준 p > %.2f) ..." % POS_THRESHOLD)
    pos = []
    for r in grid:
        sc = nli0.entailment([(p, r["sentence"]) for p in r["premises"]])
        b = max(range(len(sc)), key=lambda i: sc[i])
        if sc[b] > POS_THRESHOLD:
            pos.append((r["premises"][b], r["sentence"]))
    print("양성 쌍 %d개" % len(pos))
    del nli0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    curve_rows, nat_rows = [], []

    for ck in [x.strip() for x in args.checkpoints.split(",") if x.strip()]:
        print("")
        print("=" * 72)
        print("체크포인트: %s" % ck, flush=True)
        d2 = ccfg.as_dict()
        d2["nli"]["model"] = ck
        try:
            nli = load_nli(Config(d2, source="d13:" + ck))
        except Exception as e:                                  # noqa: BLE001
            print("  적재 실패: %s" % e, flush=True)
            continue

        # --- A + B. 곡선과 위치 효과 -----------------------------------------
        print("  배수    토큰중앙  뒤에붙임 p중앙/유지   앞에붙임 p중앙/유지", flush=True)
        for m in STEPS:
            out = {}
            for where in ("back", "front"):
                ps, tk = [], []
                for i, (prem, hyp) in enumerate(pos):
                    ext = pad(prem, filler, m, i, where)
                    p_ = nli.entailment([(ext, hyp)])[0]
                    ps.append(p_)
                    tk.append(len(nli.tokenizer.encode(ext, truncation=False)))
                    curve_rows.append({"checkpoint": ck, "multiplier": m, "where": where,
                                       "pair_idx": i, "premise_tokens": tk[-1],
                                       "p_entail": round(p_, 6)})
                out[where] = (st.median(tk), st.median(ps),
                              100.0 * sum(1 for x in ps if x >= TAU) / len(ps))
            print("  %.2fx   %5d    %.3f / %5.1f%%      %.3f / %5.1f%%"
                  % (m, int(out["back"][0]), out["back"][1], out["back"][2],
                     out["front"][1], out["front"][2]), flush=True)

        # --- C. 자연 상태 ------------------------------------------------------
        for r in grid:
            for prem in r["premises"]:
                p_ = nli.entailment([(prem, r["sentence"])])[0]
                nat_rows.append({"checkpoint": ck, "topic": r["topic"],
                                 "premise_tokens": len(nli.tokenizer.encode(prem,
                                                                            truncation=False)),
                                 "p_entail": round(p_, 6)})
        sub = [r for r in nat_rows if r["checkpoint"] == ck]
        print("  자연 상태 %d 쌍 — 전제 토큰 구간별 grounded@0.5" % len(sub))
        edges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 600), (600, 10 ** 9)]
        for lo, hi in edges:
            b = [r for r in sub if lo <= r["premise_tokens"] < hi]
            if not b:
                continue
            print("    %4d-%-5s n=%4d  중앙 p %.3f  >=0.5 %5.1f%%"
                  % (lo, ("%d" % hi) if hi < 10 ** 8 else "+", len(b),
                     st.median([r["p_entail"] for r in b]),
                     100.0 * sum(1 for r in b if r["p_entail"] >= TAU) / len(b)), flush=True)
        del nli

    for name, rows in [("d13_padding_curve.csv", curve_rows),
                       ("d13_natural_pairs.csv", nat_rows)]:
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        with io.open(outdir / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    # --- 실무 상한: 유지율이 처음 50% 아래로 떨어지는 토큰 수 --------------------
    print("")
    print("=== 실무 상한 — 유지율이 50% 아래로 처음 내려가는 지점 ===")
    print("| 체크포인트 | 붙이는 위치 | 마지막 안전 배수 | 그때 토큰 중앙 |")
    print("|---|---|---|---|")
    for ck in sorted({r["checkpoint"] for r in curve_rows}):
        for where in ("back", "front"):
            last = None
            for m in STEPS:
                b = [r for r in curve_rows if r["checkpoint"] == ck
                     and r["multiplier"] == m and r["where"] == where]
                if not b:
                    continue
                keep = 100.0 * sum(1 for r in b if r["p_entail"] >= TAU) / len(b)
                if keep >= 50.0:
                    last = (m, st.median([r["premise_tokens"] for r in b]))
                else:
                    break
            print("| %s | %s | %s | %s |"
                  % (ck, where,
                     ("%.2fx" % last[0]) if last else "1x 에서 이미 미달",
                     ("%d" % int(last[1])) if last else "-"))
    print("")
    print("-> " + str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
