"""PREREG_R3 §4.2 라이선스 관문 C-L — 다운로드 직후, 토픽 생성 전에 `MANIFEST_C2.json` 에 확인 결과를 적는다.

이 스크립트는 **판단하지 않는다.** 확인한 사람(또는 확인을 승인한 교신저자)이 출처 URL·확인 일시·문구를 넘기면 기록만 한다.
기록이 없거나 status 가 passed 가 아니면 `experiments/c2_run.py` 와 `data_c2/topics_c2.py` 가 멈춘다.

  python data_c2/license_gate_c2.py --pass --evidence-url https://info.arxiv.org/help/... \
      --statement "<공식 문서의 해당 문장 요지>" --checked-by "<이름/역할>"
  python data_c2/license_gate_c2.py --fail --statement "<이유>" --checked-by "..."    # → §4.2.1 대체 코퍼스
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data_c2" / "MANIFEST_C2.json"


def gate_status(manifest: Path = MANIFEST) -> str:
    if not Path(manifest).exists():
        return "no_manifest"
    m = json.load(io.open(manifest, encoding="utf-8"))
    return str((m.get("license_gate") or {}).get("status", "pending"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C-L 기록 (PREREG_R3 §4.2)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pass", dest="passed", action="store_true")
    g.add_argument("--fail", dest="failed", action="store_true")
    ap.add_argument("--evidence-url", default="")
    ap.add_argument("--statement", required=True)
    ap.add_argument("--checked-by", required=True)
    ap.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args(argv)
    path = Path(args.manifest)
    m = json.load(io.open(path, encoding="utf-8"))
    if args.passed and not args.evidence_url.startswith("https://"):
        raise SystemExit("--pass 에는 공식 문서 https URL 이 필요하다")
    m["license_gate"] = {
        **(m.get("license_gate") or {}),
        "status": "passed" if args.passed else "failed",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "evidence_url": args.evidence_url,
        "statement": args.statement,
        "checked_by": args.checked_by,
        "scope": "메타데이터(제목·초록) 라이선스. 문서별 license 필드(본문 라이선스)는 판단 근거가 아니다",
        "revision": (m.get("source") or {}).get("hf_revision"),
    }
    io.open(path, "w", encoding="utf-8").write(json.dumps(m, ensure_ascii=False, indent=1))
    print("C-L %s -> %s" % (m["license_gate"]["status"], path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
