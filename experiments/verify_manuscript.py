"""원고의 결과 숫자를 원자료와 대조한다. 읽기 전용, 호출 0.

`verify_numbers.py` 는 **내가 보고한 값**을 원자료와 맞춰 봤다. 이건 다르다 —
**원고에 실제로 적힌 문자열**을 뽑아 원자료와 맞춘다. 옮겨 적는 과정에서 생기는
오류는 그쪽에서만 잡힌다.

오늘 이 대조가 잡아낸 것들: 15.7 이중 반올림, %p 를 반올림끼리 뺀 것 세 건,
E2 의 풀 범위 불일치, verify 정의에서 coverage 절 누락. 전부 "옮겨 적기" 단계였다.

원고에서 숫자를 정규식으로 뽑고 JSON 산출물과 맞춘다. 원고 문구가 바뀌어 패턴이
안 맞으면 **찾지 못했다고 보고**한다 — 조용히 넘어가지 않는다. 그게 오늘 heading
파싱에서 겪은 실패다.
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "specs_from_chat/manuscript_PartialRewind_v0.5.md"

fails: list[str] = []
missing: list[str] = []


def load(p: str):
    q = ROOT / p
    return json.load(io.open(q, encoding="utf-8")) if q.exists() else None


def find(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text)
    return m.group(1) if m else None


def num(s: str | None) -> float | None:
    if s is None:
        return None
    return float(s.replace(",", "").replace("−", "-").replace("–", "-"))


def chk(label: str, got, want, tol=0.005):
    if got is None:
        missing.append(label)
        print("  [못 찾음] %-46s (원고에서 패턴이 안 맞는다)" % label)
        return
    ok = abs(float(got) - float(want)) <= tol
    if not ok:
        fails.append(label)
    print("  [%s] %-46s 원고 %-12s 자료 %s"
          % ("OK  " if ok else "불일치", label, ("%g" % got), ("%g" % want)))


def main() -> int:
    if not MS.exists():
        print("원고를 못 찾았다: %s" % MS)
        return 1
    t = io.open(MS, encoding="utf-8").read()

    live = load("results/m1c/m2min_live.json")
    qual = load("results/m1c/m2min_quality.json")
    m1c = load("results/m1c/m1c_summary.json")
    sav = load("results/m1c/m2min_savings.json")
    n1 = load("results/m2/relation/gate_n1.json")
    p1p2 = load("results/m2/relation/gate_p1p2.json")

    print("=" * 78)
    print("표 16 — 궤적 선택 비교")
    print("=" * 78)
    if live:
        chk("호출 A", num(find(t, r"Generation calls[^|]*\|\s*([\d,]+)\s*\|")),
            live["calls"]["A"], 0.5)
        chk("호출 B", num(find(t, r"Generation calls[^|]*\|[^|]*\|\s*([\d,]+)\s*\|")),
            live["calls"]["B"], 0.5)
        chk("호출 차이(%)", num(find(t, r"Generation calls[^|]*\|[^|]*\|[^|]*\|\s*\*\*([−\-\d\.]+)%")),
            live["calls"]["delta_pct"], 0.01)
        chk("토큰 A", num(find(t, r"\| Tokens \|\s*([\d,]+)")), live["tokens"]["A"], 0.5)
        chk("토큰 B", num(find(t, r"\| Tokens \|[^|]*\|\s*([\d,]+)")), live["tokens"]["B"], 0.5)
        chk("토큰 차이(%)", num(find(t, r"\| Tokens \|[^|]*\|[^|]*\|\s*\*\*([−\-\d\.]+)%")),
            live["tokens"]["delta_pct"], 0.01)
        chk("sub-question A", num(find(t, r"Sub-questions re-executed \|\s*([\d,]+)")),
            live["subq"]["A"], 0.5)
        chk("sub-question B",
            num(find(t, r"Sub-questions re-executed \|[^|]*\|\s*([\d,]+)")),
            live["subq"]["B"], 0.5)
        chk("얼린 branch", num(find(t, r"Branches frozen \|[^|]*\|\s*(\d+)")),
            live["frozen_branches"], 0.5)
        chk("coverage A", num(find(t, r"\| Coverage \|\s*([\d\.]+)")),
            live["coverage"]["A"], 0.0005)
        chk("coverage B", num(find(t, r"\| Coverage \|[^|]*\|\s*([\d\.]+)")),
            live["coverage"]["B"], 0.0005)
        chk("grounding A", num(find(t, r"\| Grounding \|\s*([\d\.]+)")),
            live["grounding"]["A"], 0.0005)
        chk("grounding B", num(find(t, r"\| Grounding \|[^|]*\|\s*([\d\.]+)")),
            live["grounding"]["B"], 0.0005)

    if qual:
        m = qual["metrics"]
        chk("entity recall A", num(find(t, r"Entity recall \|\s*([\d\.]+)")),
            m["entity recall"]["A"], 0.0005)
        chk("entity recall B", num(find(t, r"Entity recall \|[^|]*\|\s*([\d\.]+)")),
            m["entity recall"]["B"], 0.0005)
        chk("heading recall A", num(find(t, r"Heading soft recall \|\s*([\d\.]+)")),
            m["heading soft recall"]["A"], 0.0005)
        chk("heading recall B", num(find(t, r"Heading soft recall \|[^|]*\|\s*([\d\.]+)")),
            m["heading soft recall"]["B"], 0.0005)
        chk("답안 길이 A", num(find(t, r"Answer length[^|]*\|\s*([\d,\.]+)")),
            m["답안 길이(자)"]["A"], 0.6)
        chk("답안 길이 B", num(find(t, r"Answer length[^|]*\|[^|]*\|\s*([\d,\.]+)")),
            m["답안 길이(자)"]["B"], 0.6)

    print("")
    print("=" * 78)
    print("본문 서술 — 표와 같은 값을 쓰는가")
    print("=" * 78)
    if live:
        chk("본문 '7.8% fewer calls'", num(find(t, r"([\d\.]+)% fewer generation calls")),
            abs(live["calls"]["delta_pct"]), 0.05)
        chk("본문 '9.4% fewer tokens'", num(find(t, r"([\d\.]+)% fewer tokens")),
            abs(live["tokens"]["delta_pct"]), 0.05)
        chk("본문 '29 branches preserved'",
            num(find(t, r"(\d+) branches preserved")), live["frozen_branches"], 0.5)

    if m1c:
        s = m1c["sufficient"]
        chk("§VIII-A 충족 branch(tau=0.8)",
            num(find(t, r"from one branch in 180 to (\w+)")
                and {"nineteen": "19"}.get(find(t, r"from one branch in 180 to (\w+)"))),
            s["0.8"], 0.5)
    if sav:
        # 산문이 아니라 **표 15 의 행**을 직접 맞춘다. 산문 문구는 판마다 바뀐다.
        for tau in ("0.7", "0.8", "0.9"):
            b = sav["by_tau"][tau]
            row = find(t, r"\| " + tau.replace(".", r"\.") + r" \| (\d+) / (\d+) \|[^|]*\|\s*\*?\*?([−\-\d\.]+)%")
            m2 = __import__("re").search(
                r"\| " + tau.replace(".", r"\.") +
                r" \| (\d+) / (\d+) \| (\d+) / \d+ \| \*?\*?([−\-\d\.]+)%\*?\*? \| ([−\-\d\.]+)% \| ([−\-\d\.]+)%", t)
            if not m2:
                missing.append("표 15 tau=" + tau)
                print("  [못 찾음] %-46s" % ("표 15 tau=" + tau))
                continue
            chk("표15 tau=%s 얼린(확정)" % tau, num(m2.group(1)), b["frozen_branches"], 0.5)
            chk("표15 tau=%s 얼린(g1)" % tau, num(m2.group(2)), b["g1_frozen"], 0.5)
            chk("표15 tau=%s 절감 단위" % tau, num(m2.group(3)), b["saved_branch_rounds"], 0.5)
            chk("표15 tau=%s 호출(%%)" % tau, num(m2.group(4)), -b["pct_branch_rounds"], 0.01)
            chk("표15 tau=%s 토큰(%%)" % tau, num(m2.group(5)), -b["pct_tokens"], 0.01)
            chk("표15 tau=%s g1 호출(%%)" % tau, num(m2.group(6)),
                -b["g1_pct_branch_rounds"], 0.01)

    print("")
    print("=" * 78)
    print("§VII 관계 — 통제 수치")
    print("=" * 78)
    if n1:
        chk("N1 위양성(%)", num(find(t, r"(\d+\.?\d*)% of compatible fact pairs")),
            n1["gt_tau_c_pct"], 0.1)
    if p1p2:
        chk("P1 검출(%)", num(find(t, r"synthetic contradictions \((\d+\.?\d*)%")),
            p1p2["P1"]["detect_pct"], 0.1)
        chk("P2 함의(%)", num(find(t, r"paraphrases \((\d+\.?\d*)%")),
            p1p2["P2"]["entail_pct"], 0.1)

    print("")
    print("=" * 78)
    n_prov = t.count("[[prov.]]")
    n_todo = len(re.findall(r"\[\[(?!prov\.)", t))
    print("잠정 표기 [[prov.]] %d곳 · 나머지 채움 자리 %d곳" % (n_prov, n_todo))
    if missing:
        print("")
        print("**패턴을 못 찾은 항목 %d개** — 원고 문구가 바뀌었거나 값이 빠졌다:" % len(missing))
        for x in missing:
            print("  - " + x)
    if fails:
        print("")
        print("**불일치 %d건:**" % len(fails))
        for x in fails:
            print("  - " + x)
        return 1
    if missing:
        print("")
        print("불일치는 없으나 못 찾은 항목이 있다. 위 목록을 눈으로 확인하라.")
        return 0
    print("")
    print("원고 수치가 원자료와 전부 일치한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
