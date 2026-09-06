"""M1-V′ 1단계 라벨 시트 생성. SPEC_M1V′ §1. 생성·판정 호출 0회.

관문 P 가 순환이었던 이유는 양성 집합이 `g1 의 p>0.90` 이었기 때문이다. V′ 는
**사람 라벨**을 기준으로 삼아 순환을 없앤다. 순환 제거는 표본 크기가 아니라
기준이 사람이라는 사실에서 나오므로, 크기보다 **층**이 중요하다.

층은 **MiniCheck(후보 1) × g1(현행)** 라벨의 4칸이다. 스펙 §1 그대로:

  P1  MiniCheck 지지 · g1 지지    15   양쪽 합의 양성의 정밀도
  P2  MiniCheck 지지 · g1 미지지  25   **되찾은 근거의 정밀도 — 관문의 핵심 칸**
  N1  MiniCheck 미지지 · g1 미지지 15   양쪽 합의 음성의 recall
  N2  MiniCheck 미지지 · g1 지지    5   불일치 (D9-P 의 8건 부류)

LFM 에서 위험한 오류는 **잘못된 FREEZE** 다. 잘못된 재실행은 비용만 버리지만
잘못된 FREEZE 는 틀린 답을 고정한다. 그래서 표본이 추정자 **양성 쪽**에 몰려 있다.
"N층 중심 50건" 을 택하지 않은 이유가 이것이다.

시트 규칙 (스펙 §1):
  - 문장 + **인용 청크 전문**, 모든 층 동일 범위 (D8-B 의 층별 비대칭을 반복하지 않는다)
  - 청크는 인용 순서
  - 추정자 라벨·판정자 라벨·층·점수는 **시트에 없다**
  - 순서는 시드 고정 셔플
  - 라벨 3값: 지지 / 불지지 / 판단불가
  - **유형 열은 넣지 않는다** (감사용이었고 LFM 에는 불필요)

정답 키는 `_key_stage1.csv` 로 따로 쓴다. 채점 때만 쓰고 사용자에게 주지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import io
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

CELLS = [("P1", True, True, 15), ("P2", True, False, 25),
         ("N1", False, False, 15), ("N2", False, True, 5)]
RULES = ("지지: 문장의 모든 사실이 근거에 명시돼 있거나 직접 따라 나온다. / "
         "불지지: 사실 중 하나라도 근거에 없거나 어긋난다. 부분 지지도 불지지. / "
         "판단불가: 문장 자체가 모호해 사실을 특정할 수 없을 때만. 근거 부족은 불지지.")
BART = "facebook/bart-large-mnli"


def main(argv=None):
    ap = argparse.ArgumentParser(description="V' 1단계 라벨 시트 (호출 0회)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--outdir", default="results/vprime")
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

    nli = load_nli(ccfg)                     # g1 — torch/CUDA 를 먼저
    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))

    print("모집단 수집 (캐시 재생, 호출 0) ...")
    pop = []
    for n, t in enumerate(picked):
        tid = str(t["id"])
        anchor = " ".join(t.get("terms") or [])
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            for sa in o.sub_answers:
                cited_ids = [c for c in sa.citations if c in cmap]
                if not cited_ids:
                    continue
                cited = [cmap[c].text for c in cited_ids]      # 인용 순서 보존
                for sent in split_sentences(sa.text):
                    pop.append({"item_id": "v%03d" % len(pop), "topic": tid,
                                "branch": p.role, "sentence": sent, "premises": cited})
        print("  [%2d/%2d] %s  누적 %d" % (n + 1, len(picked), tid, len(pop)), flush=True)
    print("모집단 %d 문장" % len(pop))

    print("")
    print("g1 채점 (문장 × 인용 청크 통째, 최댓값 >= 0.5) ...", flush=True)
    for r in pop:
        r["p_g1"] = max(nli.entailment([(c, r["sentence"]) for c in r["premises"]]))
        r["lab_g1"] = r["p_g1"] >= 0.5
    del nli

    print("bart 채점 (같은 구성) ...", flush=True)
    d2 = ccfg.as_dict()
    d2["nli"]["model"] = BART
    nb = load_nli(Config(d2, source="vprime:bart"))
    for r in pop:
        r["p_bart"] = max(nb.entailment([(c, r["sentence"]) for c in r["premises"]]))
        r["lab_bart"] = r["p_bart"] >= 0.5
    del nb

    print("MiniCheck 채점 (인용 청크를 이어붙인 단일 문서, > 0.5) ...", flush=True)
    from d9_minicheck import Scorer
    sc = Scorer()
    for i, r in enumerate(pop):
        r["p_mc"] = sc.score("\n\n".join(r["premises"]), r["sentence"])
        r["lab_mc"] = r["p_mc"] > 0.5
        if (i + 1) % 60 == 0:
            print("  %d/%d" % (i + 1, len(pop)), flush=True)

    print("")
    print("칸별 모집단:")
    buckets = {}
    for name, mc, g1, want in CELLS:
        b = [r for r in pop if r["lab_mc"] == mc and r["lab_g1"] == g1]
        buckets[name] = b
        print("  %s  MiniCheck %-4s · g1 %-4s   모집단 %3d  (목표 %d)%s"
              % (name, "지지" if mc else "미지지", "지지" if g1 else "미지지",
                 len(b), want, "  ** 부족 **" if len(b) < want else ""))

    rng = np.random.default_rng(int(cfg["seed"]))
    sample = []
    for name, mc, g1, want in CELLS:
        b = buckets[name]
        take = min(want, len(b))
        idx = sorted(rng.choice(len(b), size=take, replace=False).tolist()) if b else []
        for i in idx:
            r = dict(b[i])
            r["cell"] = name
            sample.append(r)
    print("")
    print("표본 %d 건 (목표 60)" % len(sample))
    if len(sample) < 60:
        print("  일부 칸이 모집단보다 작다. 스펙 §1 의 비율을 유지할 수 없으므로")
        print("  칸별 실제 수를 그대로 보고한다. 채우려고 다른 칸에서 당겨오지 않는다.")

    order = rng.permutation(len(sample)).tolist()
    sample = [sample[i] for i in order]

    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    sheet = outdir / "labels_stage1.csv"
    with io.open(sheet, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "판정규칙", "근거", "문장", "human_label"])
        w.writeheader()
        for r in sample:
            w.writerow({"item_id": r["item_id"], "판정규칙": RULES,
                        "근거": "\n\n".join(r["premises"]),
                        "문장": r["sentence"], "human_label": ""})

    key = outdir / "_key_stage1.csv"
    with io.open(key, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "cell", "topic", "branch",
                                           "lab_g1", "lab_mc", "lab_bart",
                                           "p_g1", "p_mc", "p_bart"])
        w.writeheader()
        for r in sample:
            w.writerow({"item_id": r["item_id"], "cell": r["cell"], "topic": r["topic"],
                        "branch": r["branch"], "lab_g1": int(r["lab_g1"]),
                        "lab_mc": int(r["lab_mc"]), "lab_bart": int(r["lab_bart"]),
                        "p_g1": round(r["p_g1"], 6), "p_mc": round(r["p_mc"], 6),
                        "p_bart": round(r["p_bart"], 6)})

    print("")
    print("bart 양성 칸 점검 (스펙 §1: 40건 미만이면 2단계에서 보충)")
    nb_pos = sum(1 for r in sample if r["lab_bart"])
    print("  표본 60건 중 bart 지지 %d건%s"
          % (nb_pos, "  ** 40 미만 — 2단계 보충 대상 **" if nb_pos < 40 else ""))
    print("")
    print("칸 분포:", dict(Counter(r["cell"] for r in sample)))
    print("시트 -> %s   (라벨·층·점수 없음, 시드 고정 셔플)" % sheet)
    print("키   -> %s   (채점 전용, 사용자에게 주지 않는다)" % key)
    print("MiniCheck 호출 %d · 내부 청크 분할 위반 %d" % (sc.calls, sc.violations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
