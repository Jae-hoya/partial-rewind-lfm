"""V′ 60건에 deberta-large·roberta 를 얹어 사람 기준 회수표를 4체크포인트로 채운다.

**API 호출 0.** 로컬 NLI 추론만. 공저자 지시 2026-09-06 — "돌리되 부록으로".

왜 필요한가: `D12_HUMAN.md` §A 는 V′ ∩ D12 겹침(28건)에 기대서 회수 칸이 n=9 다.
같은 질문을 V′ 60건 전체에서 물으면 n=33 인데(§C), 정답키에 있는 점수가
g₁·MiniCheck·bart 셋뿐이라 나머지 두 체크포인트 행이 빈다. 그 둘을 여기서 채운다.

**전제를 어떻게 되찾는가.** 처음에는 `근거` 열(= `"\\n\\n".join(premises)`)을 그대로
되쪼개려 했다. **틀렸다 — 청크 본문에 빈 줄이 있어 60건 중 36건이 과분할됐다.**
과분할은 전제를 짧게 만들고, D13 이 보인 대로 짧은 전제는 max_p 를 **올린다**:
`v209` 는 0.0014 가 0.6071 로 뛰어 미지지가 지지로 뒤집혔다. 그대로 갔으면 회수율이
부풀었을 것이다.

그래서 `vprime_make_sheet.py` 의 모집단 구성을 **그대로 재생**해 전제를 받는다.
`item_id` 가 `"v%03d" % len(pop)` 로 결정적이라 정확히 맞는다. 캐시 재생이므로
생성 호출은 0 이다.

검증은 그대로 둔다: 재생한 전제로 **deberta-base 의 `p_g1` 을 다시 계산해 정답키와
60/60 일치**해야 진행한다. 하나라도 어긋나면 중단한다. 절차를 원하는 답에 맞춰
손보지 않는다 — 위 36건을 잡아낸 것이 이 대조다.
"""

from __future__ import annotations

import csv
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml                                          # noqa: E402

from diag_sweep import CONDITIONS, pick_topics       # noqa: E402

from lfm.branch import run_branch                    # noqa: E402
from lfm.config import Config, load_config           # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present      # noqa: E402
from lfm.nli import load_nli                         # noqa: E402
from lfm.perspectives import decompose               # noqa: E402
from lfm.retriever import BM25Retriever              # noqa: E402
from lfm.types import split_sentences                # noqa: E402

OUT = ROOT / "results/vprime"
POS, NEG = "지지", "불지지"
TAU = 0.5
BASE = "cross-encoder/nli-deberta-v3-base"
EXTRA = ["cross-encoder/nli-deberta-v3-large", "roberta-large-mnli"]


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * max(0.0, (c - h) / d), 100 * min(1.0, (c + h) / d))


def cell(k, n):
    if not n:
        return "—"
    lo, hi = wilson(k, n)
    return "%d/%d = %.1f%% [%.1f, %.1f]" % (k, n, 100.0 * k / n, lo, hi)


def main() -> int:
    lab = list(csv.DictReader(io.open(OUT / "labels_stage1.csv", encoding="utf-8-sig")))
    key = {r["item_id"]: r for r in
           csv.DictReader(io.open(OUT / "_key_stage1.csv", encoding="utf-8-sig"))}

    load_dotenv_if_present(str(ROOT / ".env"))
    cfg = load_config(ROOT / "config.yaml")
    spec = CONDITIONS["C3_partial_heldout"]
    data = cfg.as_dict()
    if spec["top_k"] is not None:
        data["retrieval"]["top_k"] = spec["top_k"]
    if spec["index"] == "partial":
        data["paths"]["index_dir"] = data["paths"]["index_dir_partial"]
    ccfg = Config(data, source="C3_partial_heldout")

    def nli_for(name):
        d = ccfg.as_dict()
        d["nli"]["model"] = name
        return load_nli(Config(d, source="vprime_extra:" + name))

    # ---- 모집단 재생 — `vprime_make_sheet.py` 와 동일한 순서·동일한 id 규칙 ----
    nli = nli_for(BASE)                      # torch/CUDA 를 먼저 (make_sheet 와 같은 순서)
    llm = LLM(ccfg)
    retriever = BM25Retriever(ccfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, 15, int(cfg["seed"]))

    print("모집단 재생 (캐시 재생, 생성 호출 0) ...")
    pop: dict[str, list[str]] = {}
    for n, t in enumerate(picked):
        anchor = " ".join(t.get("terms") or [])
        for p in decompose(t["topic"], llm, ccfg):
            o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
            cmap = o.chunk_map()
            for sa in o.sub_answers:
                cited_ids = [c for c in sa.citations if c in cmap]
                if not cited_ids:
                    continue
                cited = [cmap[c].text for c in cited_ids]
                for _ in split_sentences(sa.text):
                    pop["v%03d" % len(pop)] = cited
        print("  [%2d/%2d] %s  누적 %d" % (n + 1, len(picked), t["id"], len(pop)), flush=True)

    items, nofind = [], []
    for x in lab:
        h = x["human_label"].strip()
        k = key.get(x["item_id"])
        if h not in (POS, NEG) or not k:
            continue
        prem = pop.get(x["item_id"])
        if prem is None:
            nofind.append(x["item_id"])
            continue
        items.append({"id": x["item_id"], "human": h, "key": k,
                      "sent": x["문장"], "prem": prem})
    print("")
    print("항목 %d건 · 전제 복원 실패 %d · 전제 개수 중앙 %d · 최대 %d"
          % (len(items), len(nofind),
             sorted(len(i["prem"]) for i in items)[len(items) // 2],
             max(len(i["prem"]) for i in items)))

    # ---- 검증: 쪼갠 전제로 p_g1 을 재현하는가 -------------------------------
    print("")
    print("검증 — 재생한 전제로 %s 를 다시 돌려 정답키와 대조한다." % BASE.split("/")[-1])
    bad = []
    for i in items:
        p = max(nli.entailment([(pr, i["sent"]) for pr in i["prem"]]))
        i["chk"] = p
        if abs(p - float(i["key"]["p_g1"])) > 1e-3:
            bad.append((i["id"], p, float(i["key"]["p_g1"]), len(i["prem"])))
    del nli
    print("  불일치 %d / %d" % (len(bad), len(items)))
    for b in bad[:6]:
        print("    %s  재계산 %.4f vs 정답키 %.4f  (전제 %d개)" % b)
    if bad or nofind:
        print("")
        print("**전제 복원이 실패했다. 중단한다.** 모집단 재생 순서가 어긋났다.")
        print("`vprime_make_sheet.py` 와 조건·시드·토픽수가 같은지 확인하라.")
        return 1
    print("  -> 전제 복원이 정확하다. 새 체크포인트를 돌린다.")

    # ---- 새 체크포인트 ------------------------------------------------------
    for ck in EXTRA:
        print("")
        print("체크포인트: %s" % ck, flush=True)
        try:
            nli = nli_for(ck)
        except Exception as e:                                    # noqa: BLE001
            print("  적재 실패: %s" % e)
            for i in items:
                i[ck] = None
            continue
        for i in items:
            i[ck] = max(nli.entailment([(pr, i["sent"]) for pr in i["prem"]]))
        del nli
        print("  완료 — grounded %d/%d"
              % (sum(1 for i in items if i[ck] >= TAU), len(items)))

    # ---- 표 ----------------------------------------------------------------
    miss = [i for i in items if i["key"]["lab_g1"] == "0"]
    tgt = [i for i in miss if i["human"] == POS]
    fls = [i for i in miss if i["human"] == NEG]
    keep = [i for i in items if i["key"]["lab_g1"] == "1" and i["human"] == POS]

    def pred(i, src):
        if src[0] == "key":
            return i["key"][src[1]] == "1"
        return i[src[1]] is not None and i[src[1]] >= TAU

    ests = [("현행 g₁ (deberta-v3-base)", ("key", "lab_g1")),
            ("deberta-v3-large", ("ck", EXTRA[0])),
            ("roberta-large-mnli", ("ck", EXTRA[1])),
            ("bart-large-mnli", ("key", "lab_bart")),
            ("**MiniCheck**", ("key", "lab_mc"))]

    L = ["# V′ 사람 기준 회수표 — 5개 추정자 (부록)", "",
         "API 호출 0 · 로컬 NLI 추론. `vprime_extra_checkpoints.py`.", "",
         "`D12_HUMAN.md` §C 와 같은 표인데 deberta-large·roberta 행이 채워졌다. "
         "전제는 `vprime_make_sheet.py` 의 모집단을 재생해 복원했고(캐시 재생, 생성 호출 0), "
         "deberta-base 로 `p_g1` 을 재현해 "
         "**%d/%d 일치**를 확인한 뒤에 돌렸다." % (len(items) - len(bad), len(items)), "",
         "| 추정자 | 회수 (n=%d) | **오회수** (n=%d) | 유지 (n=%d) |"
         % (len(tgt), len(fls), len(keep)),
         "|---|---|---|---|"]
    rows = []
    for name, src in ests:
        if src[0] == "ck" and items[0].get(src[1]) is None:
            L.append("| %s | 적재 실패 | | |" % name)
            continue
        v = [sum(1 for i in s if pred(i, src)) for s in (tgt, fls, keep)]
        L.append("| %s | %s | %s | %s |" % (name, cell(v[0], len(tgt)),
                                            cell(v[1], len(fls)), cell(v[2], len(keep))))
        rows.append({"estimator": name.replace("*", ""),
                     "recovery_k": v[0], "recovery_n": len(tgt),
                     "false_recovery_k": v[1], "false_recovery_n": len(fls),
                     "retention_k": v[2], "retention_n": len(keep)})
    L += ["",
          "회수만 보면 관대한 모델이 이긴다. **오회수·유지 칸과 같이 읽어야 한다.**",
          "분모가 7 과 16 이라 그 두 칸은 순위를 정하지 못한다 — 방향만 본다.", ""]

    io.open(OUT / "VPRIME_RECOVERY_5.md", "w", encoding="utf-8").write("\n".join(L))
    with io.open(OUT / "vprime_recovery_5.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    io.open(OUT / "vprime_recovery_5.json", "w", encoding="utf-8").write(
        json.dumps({"premise_reconstruction_mismatches": len(bad),
                    "n_items": len(items), "rows": rows}, ensure_ascii=False, indent=1))
    print("")
    print("\n".join(L))
    print("-> %s" % (OUT / "VPRIME_RECOVERY_5.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
