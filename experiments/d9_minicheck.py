"""D9 — 검증기 계열 교체. MiniCheck. **생성 호출 0회, 판정 호출 0회.**

사전 등록된 설계(공저자가 `SPEC_M1D9` 로 발급, 내 권고를 그대로 채택):

  검증기   `lytang/MiniCheck-Flan-T5-Large` (MIT, 입력 상한 2048)
  가설     **문장** — 우리 D7 의 claim 결론이 아니라 **MiniCheck 공식 지침**을 따른다.
           문헌 기본 설정을 수정 없이 쓴다는 원칙이고, 자유도를 더 만들지 않기 위해서다.
  전제     branch 인용 청크를 **이어붙인 단일 문서**. 2048 안에 들어가므로 내부 분할과
           최댓값 풀링이 일어나지 않는다 — **집계 자유도를 0 으로 만든다.**
           초과 시 **꼬리 절단**(앞에서부터 남긴다).
  임계     `> 0.5` **엄격 부등호** (MiniCheck 소스의 `1 if final_score > 0.5 else 0`)

세 관문. 실행 전 고정, 사후 변경 금지.

  D9-P 양성 통제. D7-0/D13 과 **같은 35쌍·같은 패딩**을 MiniCheck 에 건다.
       통과: 1x 에서 32/35 이상 동의, 4x 에서 유지율 80% 이상 (g1 은 11%).
       1x 동의부터 안 되면 g1 의 양성 집합조차 안 받는 것이므로 거기서 중단.
  D9-A 음성 통제. D8-A/D10-N 과 같은 교란 (i -> i+7 mod 15).
       통과: 교란 grounded <= 5%. 판정은 **순증**으로만.
  D9-V 회수. D8-B 의 N층 지지 29건을 되찾는가. 대조군은 D12 와 동일.

**assert**: 단일 문서 전제에서 내부 청크 수가 1 이어야 한다. scorer 의 chunk 크기
기본값이 입력 상한보다 작으면 640토큰 문서도 둘로 쪼개져 max 가 다시 들어온다.
의도가 아니라 코드로 막는다 — **위반 1건이면 실행 무효.**

> **재현 함정.** `pip install minicheck` 는 동명이인(explicit-state model checker)이다.
> `pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git"` 를 써야 한다.
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

from d10_checkpoint_family import POS_THRESHOLD, collect  # noqa: E402
from d13_dilution_curve import pad  # noqa: E402
from d8a_negative_control import SHIFT, spearman  # noqa: E402
from diag_sweep import CONDITIONS, pick_topics  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.nli import load_nli  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402
from lfm.types import split_sentences  # noqa: E402

MODEL = "flan-t5-large"          # = lytang/MiniCheck-Flan-T5-Large
MAX_LEN = 2048
TAU = 0.5                        # 엄격 부등호로 쓴다 (> 0.5)
MULTIPLIERS = (1, 2, 3, 4)
GATE_P_1X = 32                   # 35쌍 중
GATE_P_4X = 80.0                 # %
GATE_A_FPR = 5.0                 # %


class Scorer:
    """MiniCheck 래퍼. 내부 청크 수 == 1 을 강제한다."""

    def __init__(self):
        from minicheck.minicheck import MiniCheck
        self.m = MiniCheck(model_name=MODEL, max_model_len=MAX_LEN, batch_size=8)
        self.violations = 0
        self.calls = 0
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("google/flan-t5-large")

    def truncate(self, doc: str) -> str:
        """2048 초과 시 꼬리를 자른다 (앞에서부터 남긴다)."""
        ids = self.tok.encode(doc, truncation=False)
        if len(ids) <= MAX_LEN - 64:      # 프롬프트 여유
            return doc
        return self.tok.decode(ids[: MAX_LEN - 64], skip_special_tokens=True)

    def score(self, doc: str, claim: str) -> float:
        doc = self.truncate(doc)
        _, probs, _, _ = self.m.score(docs=[doc], claims=[claim])
        self.calls += 1
        return float(probs[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description="D9 MiniCheck (호출 0회)")
    ap.add_argument("--condition", default="C3_partial_heldout")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--outdir", default="results/diag/d9")
    ap.add_argument("--stages", default="P,A,V")
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

    print("양성 쌍 선정 (g1 기준 p > %.2f) ..." % POS_THRESHOLD)
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
    stages = [x.strip().upper() for x in args.stages.split(",") if x.strip()]

    print("")
    print("MiniCheck 적재 (%s, 상한 %d) ..." % (MODEL, MAX_LEN), flush=True)
    sc = Scorer()
    verdict = {}

    # ---------------- D9-P 양성 통제 -------------------------------------
    if "P" in stages:
        print("")
        print("=" * 70)
        print("D9-P 양성 통제 — D7-0/D13 과 같은 35쌍·같은 패딩")
        print("=" * 70, flush=True)
        rows = []
        keep = {}
        for m in MULTIPLIERS:
            ps = []
            for i, (prem, hyp) in enumerate(pos):
                ext = pad(prem, filler, float(m), i, "back")
                p_ = sc.score(ext, hyp)
                ps.append(p_)
                rows.append({"stage": "P", "multiplier": m, "pair_idx": i,
                             "p_support": round(p_, 6)})
            keep[m] = 100.0 * sum(1 for x in ps if x > TAU) / len(ps)
            print("  %dx  p 중앙 %.3f  >0.5 유지 %5.1f%%   (g1: 1x 100 / 2x 14 / 3x 14 / 4x 11)"
                  % (m, st.median(ps), keep[m]), flush=True)
        n1 = int(round(keep[1] * len(pos) / 100))
        ok_p = (n1 >= GATE_P_1X) and (keep[4] >= GATE_P_4X)
        print("")
        print("  1x 동의 %d/%d (기준 >=%d)  ·  4x 유지 %.1f%% (기준 >=%.0f%%)  ->  %s"
              % (n1, len(pos), GATE_P_1X, keep[4], GATE_P_4X, "PASS" if ok_p else "FAIL"))
        verdict["P"] = ok_p
        with io.open(outdir / "d9p_padding.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        if not ok_p and n1 < GATE_P_1X:
            print("  1x 동의부터 미달이다. 스펙대로 여기서 중단한다.")
            return 1

    # ---------------- D9-A 음성 통제 -------------------------------------
    if "A" in stages:
        print("")
        print("=" * 70)
        print("D9-A 음성 통제 — 교란 i -> i+%d mod 15, 단일 문서 전제" % SHIFT)
        print("=" * 70, flush=True)
        llm = LLM(ccfg)
        retriever = BM25Retriever(ccfg)
        sents_by, pool_by = {}, {}
        for t in picked:
            tid = str(t["id"])
            anchor = " ".join(t.get("terms") or [])
            s_, p_ = [], []
            for p in decompose(t["topic"], llm, ccfg):
                o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
                cmap = o.chunk_map()
                for c in o.cited_chunks():
                    p_.append(c.text)
                for sa in o.sub_answers:
                    if any(c in cmap for c in sa.citations):
                        s_.extend(split_sentences(sa.text))
            sents_by[tid], pool_by[tid] = s_, p_
        tids = [str(t["id"]) for t in picked]
        mapping = {tids[i]: tids[(i + SHIFT) % len(tids)] for i in range(len(tids))}
        assert all(a != b for a, b in mapping.items())
        norm, shuf, psz, rows = [], [], [], []
        for tid in tids:
            own = "\n\n".join(pool_by[tid])
            other = "\n\n".join(pool_by[mapping[tid]])
            if not sents_by[tid] or not own or not other:
                continue
            for s_ in sents_by[tid]:
                a = sc.score(own, s_)
                b = sc.score(other, s_)
                norm.append(a); shuf.append(b); psz.append(float(len(pool_by[tid])))
                rows.append({"stage": "A", "src_topic": tid, "premise_topic": mapping[tid],
                             "n_premise_chunks": len(pool_by[tid]),
                             "p_normal": round(a, 6), "p_shuffled": round(b, 6)})
        gn = 100.0 * sum(1 for x in norm if x > TAU) / len(norm)
        gs = 100.0 * sum(1 for x in shuf if x > TAU) / len(shuf)
        ok_a = gs <= GATE_A_FPR
        print("  n=%d  정상 %.2f%%  교란 %.2f%%  **순증 %.2f%%p**  rho(풀,p) %+.3f  ->  %s"
              % (len(norm), gn, gs, gn - gs, spearman(psz, norm), "PASS" if ok_a else "FAIL"))
        print("  대조: g1 15.65/0.00/15.65 · bart 60.88/1.02/59.86 · roberta 19.73/2.72/17.01")
        verdict["A"] = ok_a
        with io.open(outdir / "d9a_negative.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    # ---------------- D9-V 회수 -------------------------------------------
    if "V" in stages:
        print("")
        print("=" * 70)
        print("D9-V 회수 — D8-B N층 지지 29건 (대조군은 D12 와 동일)")
        print("=" * 70, flush=True)
        llm = LLM(ccfg)
        retriever = BM25Retriever(ccfg)
        pop, n_seen = {}, 0
        for t in picked:
            tid = str(t["id"])
            anchor = " ".join(t.get("terms") or [])
            for p in decompose(t["topic"], llm, ccfg):
                o = run_branch(p, retriever, llm, ccfg, round_=0, search_anchor=anchor)
                cmap = o.chunk_map()
                for sa in o.sub_answers:
                    cited = [cmap[c].text for c in sa.citations if c in cmap]
                    if not cited:
                        continue
                    for sent in split_sentences(sa.text):
                        pop["%s:%s:%d" % (tid, p.role, n_seen)] = {
                            "sentence": sent, "doc": "\n\n".join(cited)}
                        n_seen += 1
        jb = list(csv.DictReader(io.open(root / "results/diag/d8/d8b_judge_sample.csv",
                                         encoding="utf-8")))
        groups = {"N층 supported (회수)": [], "N층 unsupported (오회수)": [],
                  "G층 supported (유지)": [], "M층 supported (참고)": []}
        for r in jb:
            src = pop.get(r["claim_id"])
            if src is None or src["sentence"] != r["sentence"]:
                continue
            k = None
            if r["stratum"] == "N" and r["judge_label"] == "supported":
                k = "N층 supported (회수)"
            elif r["stratum"] == "N" and r["judge_label"] == "unsupported":
                k = "N층 unsupported (오회수)"
            elif r["stratum"] == "G" and r["judge_label"] == "supported":
                k = "G층 supported (유지)"
            elif r["stratum"] == "M" and r["judge_label"] == "supported":
                k = "M층 supported (참고)"
            if k:
                groups[k].append({**r, "doc": src["doc"]})
        rows = []
        res = {}
        for gname, items in groups.items():
            if not items:
                continue
            hit = 0
            for r in items:
                p_ = sc.score(r["doc"], r["sentence"])
                if p_ > TAU:
                    hit += 1
                rows.append({"stage": "V", "claim_id": r["claim_id"], "group": gname,
                             "base_max_p": r["max_p"], "p_support": round(p_, 6),
                             "grounded": int(p_ > TAU)})
            res[gname] = 100.0 * hit / len(items)
            print("  %-26s %2d/%2d = %5.1f%%" % (gname, hit, len(items), res[gname]), flush=True)
        print("")
        print("  대조 (D12): base 0.0/0.0/100.0 · bart 62.1/70.0/88.1 · roberta 17.2/0.0/50.0")
        with io.open(outdir / "d9v_recovery.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    print("")
    print("=== D9 관문 ===")
    for k in ("P", "A"):
        if k in verdict:
            print("  D9-%s  %s" % (k, "PASS" if verdict[k] else "FAIL"))
    print("  내부 청크 분할 위반: %d 건 (0 이어야 유효)" % sc.violations)
    print("  MiniCheck 호출 %d (생성 0 · 판정 0)" % sc.calls)
    print("-> " + str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
