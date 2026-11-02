"""D19 — LLM judge(gpt-5.4-nano)를 D18 과 같은 관문에 올린다. 등록: results/d19/PREREG_D19.md (커밋 360b917)

judge 는 사람 라벨러가 본 시트 문항을 그대로 받는다 — 같은 입력에서 사람과 비교한다.
  --smoke  1건만 불러 형식·토큰을 확인한다
  (기본)   dev(A 43, B 86)와 held-out(A 100, B 100) 전부. held-out 은 이 한 번이 전부다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from d18_gate import gate, stats  # noqa: E402
from lfm.config import Config, load_config  # noqa: E402
from lfm.llm import LLM, load_dotenv_if_present  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-5.4-nano"
MAX_TOKENS = 512
TOKEN_CAP = 1_000_000
LABELS = {"A": ("full", "partial", "none"), "B": ("yes", "partial", "no")}
SETS = [  # (이름, 연산자, 시트, 키, 라벨 소스, 라벨 세트명, 역할)
    ("h2dev", "A", "results/d18/labeler_h2dev.html", "results/d18/_key_h2dev.csv", "all", "h2dev", "dev"),
    ("h1", "B", "results/d16/labeler_d16.html", "results/d16/_key_d16.csv", "results/d16/labels_d16_filled.csv", None, "dev"),
    ("h2hoA", "A", "results/d18/labeler_h2hoA.html", "results/d18/_key_h2hoA.csv", "all", "h2hoA", "held_out"),
    ("h2hoB", "B", "results/d18/labeler_h2hoB.html", "results/d18/_key_h2hoB.csv", "all", "h2hoB", "held_out"),
]


def items_of(path):
    h = io.open(ROOT / path, encoding="utf-8").read()
    return {x["id"]: x for x in json.loads(re.search(r"const ITEMS = (\[.*?\]);\n", h, re.S).group(1))}


def parse(text, kind):
    lines = (text or "").splitlines()
    for line in lines:
        m = re.match(r"\s*\**LABEL\**\s*:\s*\**\s*([a-z]+)", line, re.I)
        if m and m.group(1).lower() in LABELS[kind]:
            return m.group(1).lower()
    # 스모크에서 nano 가 "LABEL:" 접두어 없이 첫 줄에 라벨만 쓴 것을 봤다 (PREREG_D19 부기 1)
    m = re.match(r"\s*\**([a-z]+)\**\s*$", lines[0], re.I) if lines else None
    if m and m.group(1).lower() in LABELS[kind]:
        return m.group(1).lower()
    return "unparsed"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    load_dotenv_if_present(str(ROOT / ".env"))
    cfg = load_config(ROOT / "config.yaml")
    data = cfg.as_dict()
    data["llm"]["model"] = MODEL
    data["llm"]["max_tokens"] = MAX_TOKENS
    llm = LLM(Config(data, source="d19_judge"))
    tmpl = {k: io.open(ROOT / ("prompts/d19_judge_%s.txt" % k), encoding="utf-8").read() for k in "AB"}
    all_labels = {}
    for r in csv.DictReader(io.open(ROOT / "results/d18/labels_all_filled.csv", encoding="utf-8-sig")):
        all_labels.setdefault(r["set"], {})[r["item_id"]] = r["human_label"]

    out = {"model": MODEL, "max_tokens": MAX_TOKENS, "sets": {}}
    rows_all = []
    for name, kind, sheet, keyf, labsrc, labset, role in SETS:
        items = items_of(sheet)
        keys = list(csv.DictReader(io.open(ROOT / keyf, encoding="utf-8")))
        if labsrc == "all":
            labs = all_labels[labset]
        else:
            labs = {r["item_id"]: r["human_label"] for r in csv.DictReader(io.open(ROOT / labsrc, encoding="utf-8-sig"))}
        rows = []
        for k in keys[: 1 if args.smoke else None]:
            it = items[k["item_id"]]
            prompt = tmpl[kind].format(question=it["claim"], **({"answer": it["ev"]} if kind == "A" else {"context": it["ev"]}))
            res = llm.call(prompt, tag="d19_judge_" + kind, max_tokens=MAX_TOKENS)
            jl = parse(res.text, kind)
            h = labs[k["item_id"]]
            rows.append({"set": name, "item_id": k["item_id"], "topic": k["query_id"],
                         "w": float(k["N_h"]) / float(k["n_h"]) if "N_h" in k else 1.0,
                         "judge": jl, "human": h, "finish": res.finish_reason,
                         "p": int(jl == "full") if kind == "A" else int(jl in ("yes", "partial")),
                         "p_loose": int(jl in ("full", "partial")) if kind == "A" else None,
                         "y": int(h == "답함") if kind == "A" else int(h != "답없음"),
                         "y_loose": int(h != "답안함") if kind == "A" else None,
                         "reason": res.text.strip()[-300:]})
            if llm.stats.billed_tokens_in + llm.stats.billed_tokens_out > TOKEN_CAP:
                print("토큰 상한 초과 — 중단"); return 3
        if args.smoke:
            print(json.dumps({"set": name, "row": rows[0], "stats": llm.stats.to_dict()}, ensure_ascii=False, indent=1))
            return 0
        res = {"role": role, "n": len(rows), "judge_counts": dict(Counter(r["judge"] for r in rows)),
               "crosstab": {"%s|%s" % k: v for k, v in sorted(Counter((r["human"], r["judge"]) for r in rows).items())},
               "primary": gate(rows) if role == "held_out" else stats(rows)}
        if kind == "A":
            loose = [{**r, "p": r["p_loose"], "y": r["y_loose"]} for r in rows]
            res["sensitivity_partial_as_answer"] = gate(loose) if role == "held_out" else stats(loose)
        out["sets"][name] = res
        rows_all += rows
        print("%-6s %s n=%d %s" % (name, role, len(rows), json.dumps(res["judge_counts"], ensure_ascii=False)), flush=True)

    out["usage"] = llm.stats.to_dict()
    d = ROOT / "results/d19"
    with io.open(d / "d19_judge_rows.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_all[0].keys()))
        w.writeheader(); w.writerows(rows_all)
    io.open(d / "d19_gate.json", "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({n: {k: v for k, v in s.items() if k in ("primary", "sensitivity_partial_as_answer")}
                      for n, s in out["sets"].items()}, ensure_ascii=False, indent=1))
    print("usage", {k: out["usage"][k] for k in ("calls", "api_calls", "billed_tokens_in", "billed_tokens_out")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
