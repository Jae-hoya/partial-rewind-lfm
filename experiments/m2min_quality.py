"""M2-min 품질 재측정 — **선택 기준에 안 들어간 지표로.** 호출 0회.

공저자 지적이 정확하다:

> 얼릴 branch 를 고르는 기준이 `sufficient = coverage=1.0 ∧ grounding>τ` 인데,
> 개선됐다고 보고한 두 지표가 정확히 coverage 와 grounding 이다. 그 둘이 높은
> branch 만 골라 보존했으니 평균이 오르는 것은 **부분적으로 정의상** 그렇다.

메커니즘("라운드 0 의 좋은 것을 나쁜 재시도가 덮어쓴다")은 D11 이 독립적으로 보였고
방향도 맞다. 그러나 표만 보면 심사자는 "선택 기준으로 쓴 지표가 올랐다고 말하는 것
아니냐" 고 쓰고, **그게 반박이 안 된다.**

그래서 **선택에 안 쓰인 지표** 둘로 다시 잰다. 둘 다 결정론이고 추정자와 무관하다.

  entity recall        답안이 target article 의 엔티티를 얼마나 담는가
  heading soft recall  target article 섹션 제목 대비 답안 커버리지

통제 변수로 답안 길이와 sub-answer 수도 같이 낸다 — 길이가 늘어서 오른 것인지
가려야 하기 때문이다.

두 궤적:
  A  GlobalRetry — 전 branch 의 **최종 라운드** artifact
  B  축소 컨트롤러 — 얼린 branch 는 **얼린 라운드**, 나머지는 최종 라운드

**엔티티 정의는 결정론이다.** 대문자로 시작하는 연속 토큰열(고유명사 후보)과 4자리
연도·숫자. 모델을 쓰지 않는다. 문서에서 2회 이상 등장하는 것만 남겨 잡음을 줄인다.
이 규칙은 실행 전에 고정하고 결과를 보고 바꾸지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from d7_probe import target_chunks  # noqa: E402
from diag_sweep import pick_topics  # noqa: E402

from lfm.branch import run_branch  # noqa: E402
from lfm.config import load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402
from lfm.perspectives import decompose  # noqa: E402
from lfm.retriever import BM25Retriever  # noqa: E402

TAU = 0.8
MAX_ROUNDS = 3
BOOT = 10000
SEED = 42

# 결정론적 엔티티 규칙 — 실행 전 고정
ENT = re.compile(r"\b(?:[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+)*|\d{4}|\d[\d,\.]*)\b")
STOP = {"The", "A", "An", "In", "On", "At", "It", "This", "That", "These", "Those",
        "He", "She", "They", "We", "I", "His", "Her", "Their", "Its", "As", "By",
        "For", "From", "To", "Of", "And", "But", "Or", "However", "Although"}
WORD = re.compile(r"[a-z0-9]+")


def entities(text: str, min_count: int = 1) -> set[str]:
    c: dict[str, int] = {}
    for m in ENT.findall(text or ""):
        s = m.strip()
        if not s or s in STOP or len(s) < 3:
            continue
        c[s] = c.get(s, 0) + 1
    return {k for k, v in c.items() if v >= min_count}


def heading_recall(headings: list[str], answer: str) -> float:
    """제목의 내용어가 답안에 얼마나 나타나는가. 제목별 부분 점수의 평균."""
    if not headings:
        return float("nan")
    aw = set(WORD.findall(answer.lower()))
    scores = []
    for h in headings:
        hw = [w for w in WORD.findall(h.lower()) if len(w) > 2]
        if not hw:
            continue
        scores.append(sum(1 for w in hw if w in aw) / len(hw))
    return st.mean(scores) if scores else float("nan")


def boot_ci(vals, n=BOOT, seed=SEED):
    v = [x for x in vals if x == x]
    if not v:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    a = np.array(v, dtype=float)
    b = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return (float(lo), float(hi))


def main(argv=None):
    ap = argparse.ArgumentParser(description="M2-min 품질 재측정 (호출 0회)")
    ap.add_argument("--n-topics", type=int, default=50)
    ap.add_argument("--live", default="results/m1c/m2min_live.csv")
    ap.add_argument("--outdir", default="results/m1c")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(str(root / ".env"))
    cfg = load_config(root / "config.yaml")
    llm = LLM(cfg)
    retriever = BM25Retriever(cfg)
    with io.open(Path(cfg["paths.data_dir"]) / "topics_pilot50.yaml", encoding="utf-8") as fh:
        topics = yaml.safe_load(fh)["topics"]
    picked = pick_topics(topics, args.n_topics, int(cfg["seed"]))

    # 라이브 비교에 실제로 들어간 질의만 (분모 일치)
    used = {r["query_id"] for r in csv.DictReader(io.open(root / args.live, encoding="utf-8"))}
    print("라이브 비교에 쓰인 질의 %d개로 제한" % len(used))

    print("target article 적재 ...", flush=True)
    tchunks = target_chunks(cfg)

    # 섹션 제목 — walk_sections 의 경로 마지막 이름
    import pyarrow.parquet as pq
    from chunk_index import walk_sections
    heldout = json.load(io.open(Path(cfg["paths.data_dir"]) / "heldout_ids.json",
                                encoding="utf-8"))
    tid_of = {}
    for k, v in heldout["by_topic"].items():
        for i in v["heldout_identifiers"]:
            tid_of[int(i)] = k
    heads: dict[str, list[str]] = {}
    pf = pq.ParquetFile(Path(cfg["paths.data_dir"]) / "corpus.parquet")
    for batch in pf.iter_batches(batch_size=2000, columns=["identifier", "sections"]):
        for row in batch.to_pylist():
            did = int(row["identifier"])
            k = tid_of.get(did)
            if not k or k in heads:
                continue
            # sections 는 JSON 문자열이다. walk_sections 의 경로도 튜플이 아니라
            # " > " 로 이어붙인 문자열이다. 둘 다 앞 판에서 틀렸다.
            raw = row["sections"]
            if isinstance(raw, str):
                raw = json.loads(raw)
            hs = [str(path).split(" > ")[-1].strip()
                  for path, _ in walk_sections(raw) if path]
            heads[k] = sorted({h for h in hs if h and h != "Body"})
    med = int(st.median([len(v) for v in heads.values()]) or 0) if heads else 0
    print("섹션 제목 수집: 토픽 %d, 제목 중앙 %d" % (len(heads), med))
    # 조용한 실패 방지 — 앞 판에서 제목 0 개인데 계산이 그냥 진행됐다.
    assert med > 0, "섹션 제목을 하나도 못 뽑았다. sections 파싱을 확인하라."

    from d9_minicheck import Scorer
    print("MiniCheck 적재 (얼리기 판정용) ...", flush=True)
    sc = Scorer()
    from m2min_live import evaluate

    rows = []
    for t in picked:
        tid = str(t["id"])
        if tid not in used or tid not in tchunks:
            continue
        tgt = "\n".join(tchunks[tid])
        tgt_ent = entities(tgt, min_count=2)
        if not tgt_ent:
            continue
        hs = heads.get(tid, [])
        anchor = " ".join(t.get("terms") or [])
        persps = decompose(t["topic"], llm, cfg)

        ev, art = {}, {}
        for p in persps:
            for r_ in range(MAX_ROUNDS):
                o = run_branch(p, retriever, llm, cfg, round_=r_, search_anchor=anchor)
                ev[(p.role, r_)] = evaluate(o, sc)
                art[(p.role, r_)] = "\n".join(sa.text for sa in o.sub_answers)

        roles = [p.role for p in persps]
        # B 궤적 재현 — 라이브 비교와 같은 규칙
        frozen: dict[str, int] = {}
        b_state = {b: 0 for b in roles}
        for r_ in range(MAX_ROUNDS):
            active = [b for b in roles if b not in frozen]
            if r_ > 0 and not active:
                break
            for b in (roles if r_ == 0 else active):
                b_state[b] = r_
            for b in roles:
                if b in frozen:
                    continue
                cov, grd, _, _ = ev[(b, b_state[b])]
                if cov == 1.0 and grd > TAU:
                    frozen[b] = b_state[b]

        a_text = "\n".join(art[(b, MAX_ROUNDS - 1)] for b in roles)
        b_text = "\n".join(art[(b, frozen.get(b, b_state[b]))] for b in roles)
        ae, be = entities(a_text), entities(b_text)
        rows.append({
            "query_id": tid, "n_target_entities": len(tgt_ent), "n_headings": len(hs),
            "A_entity_recall": len(ae & tgt_ent) / len(tgt_ent),
            "B_entity_recall": len(be & tgt_ent) / len(tgt_ent),
            "A_heading_recall": heading_recall(hs, a_text),
            "B_heading_recall": heading_recall(hs, b_text),
            "A_chars": len(a_text), "B_chars": len(b_text),
            "frozen": len(frozen)})
        print("  [%2d] %s  엔티티 %d · 제목 %d" % (len(rows), tid, len(tgt_ent), len(hs)),
              flush=True)

    if not rows:
        print("계산할 질의가 없다.")
        return 1

    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    with io.open(outdir / "m2min_quality.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("")
    print("=== 표 16 추가 행 — 선택 기준에 안 들어간 지표 (질의 %d) ===" % len(rows))
    print("| 지표 | Global retry | 축소 컨트롤러 | 차이 (95%% CI) |")
    print("|---|---|---|---|")
    out = {"n_queries": len(rows), "provisional": True, "metrics": {}}
    for name, ka, kb in [("entity recall", "A_entity_recall", "B_entity_recall"),
                         ("heading soft recall", "A_heading_recall", "B_heading_recall"),
                         ("답안 길이(자)", "A_chars", "B_chars")]:
        d = [r[kb] - r[ka] for r in rows if r[ka] == r[ka] and r[kb] == r[kb]]
        lo, hi = boot_ci(d)
        ma = st.mean([r[ka] for r in rows if r[ka] == r[ka]])
        mb = st.mean([r[kb] for r in rows if r[kb] == r[kb]])
        direction = "better" if lo > 0 else ("worse" if hi < 0 else "same")
        out["metrics"][name] = {"A": round(ma, 6), "B": round(mb, 6),
                                "delta": round(st.mean(d), 6),
                                "ci": [round(lo, 6), round(hi, 6)],
                                "direction": direction, "n": len(d)}
        fmt = "%.1f" if "길이" in name else "%.4f"
        print(("| %s | " + fmt + " | " + fmt + " | %+.4f [%+.4f, %+.4f] **%s** |")
              % (name, ma, mb, st.mean(d), lo, hi, direction))

    io.open(outdir / "m2min_quality.json", "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=1))

    e = out["metrics"]["entity recall"]["direction"]
    h = out["metrics"]["heading soft recall"]["direction"]
    print("")
    print("판정: entity recall **%s** · heading soft recall **%s**" % (e, h))
    if "better" in (e, h) and "worse" not in (e, h):
        print("  -> 선택 기준 밖 지표에서도 같은 방향. 순환 반박이 성립하지 않는다.")
    elif e == h == "same":
        print("  -> 선택 기준 밖 지표는 차이 없음. \"비용은 줄고 품질은 선택 기준")
        print("     지표에서만 오른다\" 로 정직하게 쓴다.")
    else:
        print("  -> 방향이 엇갈린다. 표에 그대로 싣고 해석은 공저자에게 넘긴다.")
    print("")
    print("LLM 호출 %d (캐시 %d), 청구 %d/%d"
          % (llm.stats.calls, llm.stats.cache_hits,
             llm.stats.billed_tokens_in, llm.stats.billed_tokens_out))
    print("-> " + str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
