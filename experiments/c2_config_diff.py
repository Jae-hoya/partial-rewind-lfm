"""PREREG_R3 §4.3 — `config_c2.yaml` 은 `config.yaml` 에서 표의 키만 바꾼 사본이다. 만들고(--write) 대조한다(--check).

대조는 두 층이다.
  1. **바이트**: `config_c2.yaml` == render(config.yaml, MANIFEST). render 는 표의 키 줄만 값(과 그 줄의 주석)을 바꾸고,
     새 키 세 줄(topics.min_archives / max_per_archive, heldout.near_dup_jaccard)과 머리 주석 한 줄을 넣는다.
     나머지 줄은 config.yaml 의 바이트 그대로다.
  2. **의미**: 두 YAML 을 점 표기로 펴서 값이 다른 키 집합 == 표의 키 집합(값 같은 n_candidates·cache_dir_ro 포함 검사).
     data.hf_* 네 키는 C-L 에서 정한다 — `data_c2/MANIFEST_C2.json` 이 있으면 그 값과 같아야 하고, 없으면(C-L 전)
     자리표시 값이어야 한다. `--require-manifest` 면 MANIFEST 가 없을 때 실패한다(c2_run.py 가 이것으로 부른다).

  python experiments/c2_config_diff.py --write     # config_c2.yaml 생성 (MANIFEST 가 있으면 그 값으로)
  python experiments/c2_config_diff.py --check     # 대조. 실패하면 종료 코드 1
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config.yaml"
C2 = ROOT / "config_c2.yaml"
MANIFEST = ROOT / "data_c2" / "MANIFEST_C2.json"

HEADER = "# C2 (PREREG_R3 §4.3): config.yaml 에서 §4.3 표의 키만 바꾼 사본. 손으로 고치지 말고 experiments/c2_config_diff.py --write 로 만든다.\n"

# C-L 전 자리표시. download_c2.py 가 revision 을 고정해 MANIFEST 에 쓰면 --write 가 그 값으로 바꾼다
PLACEHOLDER_HF = {
    "hf_dataset": "librarian-bots/arxiv-metadata-snapshot",
    "hf_config": "default",
    "hf_revision": "PENDING_C_L",
    "hf_file_glob": "data/train-*.parquet",
}

# (section, key) -> (new value as YAML scalar text, new comment or None = 원래 주석 유지)
FIXED: dict[tuple[str, str], tuple[str, str | None]] = {
    ("paths", "data_dir"): ("data_c2", None),
    ("paths", "raw_dir"): ("data_c2/raw", None),
    ("paths", "index_dir"): ("data_c2/index", None),
    ("paths", "index_dir_partial"): ("null", "# C2: partial 진단 없음"),
    ("paths", "cache_dir"): (".cache/llm_c2", None),
    ("paths", "cache_dir_ro"): ("null", "# C2: 읽기 전용 캐시 없음 — 분해는 토픽마다 한 번 라이브 (PREREG_R3 §4.4)"),
    ("data", "min_abstract_chars"): ("400", "# C2: 초록 400자 미만 제외 (§4.2)"),
    ("data", "n_topics"): ("28", "# C2: pilot3 + heldout25"),
    ("topics", "min_target_abstract_chars"): ("1000", "# C2"),
    ("topics", "min_target_section_chars"): ("0", "# C2: 본문 없음 (초록 = 유일한 섹션)"),
    ("topics", "min_target_references"): ("0", "# C2: references 는 빈 목록 열"),
    ("topics", "max_per_infobox_type"): ("2", "# C2: 주 분류당 최대 2"),
    ("topics", "n_candidates"): ("60", "# C2: 28개 승인 + 예비 (모자라면 120 으로 한 번, 이탈)"),
}
HF_KEYS = ("hf_dataset", "hf_config", "hf_revision", "hf_file_glob")
# 새 키: (section, 앞 줄의 key) 뒤에 넣을 줄들
INSERT_AFTER: dict[tuple[str, str], list[str]] = {
    ("topics", "max_per_infobox_type"): [
        "  min_archives: 2                 # C2 신규: 이웃 문서의 서로 다른 archive(분류의 점 앞 접두사) 하한\n",
        "  max_per_archive: 8              # C2 신규: target 주 분류의 archive 당 후보 상한\n",
    ],
}
APPEND_TO_SECTION: dict[str, list[str]] = {
    "heldout": ["  near_dup_jaccard: 0.6           # C2 신규: 초록 내용어 자카드 >= 이 값인 문서도 held-out (§4.2)\n"],
}
# 의미 대조용: 값이 달라야(또는 새로 생겨야) 하는 키 전체
EXPECTED_CHANGED = {
    "paths.data_dir", "paths.raw_dir", "paths.index_dir", "paths.index_dir_partial", "paths.cache_dir",
    "data.min_abstract_chars", "data.n_topics",
    "topics.min_target_abstract_chars", "topics.min_target_section_chars", "topics.min_target_references",
    "topics.max_per_infobox_type", "topics.min_archives", "topics.max_per_archive", "heldout.near_dup_jaccard",
    "data.hf_dataset", "data.hf_config", "data.hf_revision", "data.hf_file_glob",
}
# 표에 있지만 값이 R2 와 같은 키 (같아야 한다)
EXPECTED_SAME = {"paths.cache_dir_ro": None, "topics.n_candidates": 60}

_KEY_LINE = re.compile(r"^(  )([A-Za-z_][A-Za-z0-9_]*):(\s*)(.*?)(\s+#.*)?(\r?\n)?$")
_SECTION = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):")


def hf_values(manifest: Path | None = MANIFEST) -> tuple[dict[str, str], bool]:
    """(값, MANIFEST 에서 왔는가)."""
    if manifest is not None and Path(manifest).exists():
        m = json.load(io.open(manifest, encoding="utf-8"))
        src = m["source"]
        return {k: str(src[k]) for k in HF_KEYS}, True
    return dict(PLACEHOLDER_HF), False


def _yaml_scalar(v: str) -> str:
    # 위키 config 처럼 따옴표 없이 쓴다. YAML 이 다른 타입으로 읽을 값이면 따옴표
    t = yaml.safe_load(v) if v not in ("",) else None
    return v if isinstance(t, str) and t == v else json.dumps(v)


def render(base_text: str, hf: dict[str, str]) -> str:
    out = [HEADER]
    section = None
    lines = base_text.splitlines(keepends=True)
    pending_append: list[str] = []
    for i, line in enumerate(lines):
        ms = _SECTION.match(line)
        if ms:
            if section in APPEND_TO_SECTION:
                # 앞 섹션 끝(빈 줄들 앞)에 새 키를 넣는다
                j = len(out)
                while j > 0 and out[j - 1].strip() == "":
                    j -= 1
                out[j:j] = APPEND_TO_SECTION[section]
            section = ms.group(1)
        mk = _KEY_LINE.match(line)
        if mk and section is not None:
            key = mk.group(2)
            nl = mk.group(6) or ""
            if (section, key) in FIXED:
                val, com = FIXED[(section, key)]
                com = com if com is not None else (mk.group(5) or "").strip()
                line = "  %s:%s%s%s%s" % (key, mk.group(3) or " ", val, ("   " + com) if com else "", nl)
            elif section == "data" and key in HF_KEYS:
                line = "  %s:%s%s   # C2: C-L 에서 고정, data_c2/MANIFEST_C2.json 과 같아야 한다%s" % (
                    key, mk.group(3) or " ", _yaml_scalar(hf[key]), nl)
            out.append(line)
            if (section, key) in INSERT_AFTER:
                # 같은 키의 주석 연속 줄(들여쓴 '#' 줄)이 바로 뒤에 오면 그 뒤에 넣는다
                pending_append = INSERT_AFTER[(section, key)]
                k = i + 1
                if not (k < len(lines) and re.match(r"^\s{20,}#", lines[k])):
                    out.extend(pending_append); pending_append = []
            continue
        out.append(line)
        if pending_append and not (i + 1 < len(lines) and re.match(r"^\s{20,}#", lines[i + 1])):
            out.extend(pending_append); pending_append = []
    if section in APPEND_TO_SECTION:
        out.extend(APPEND_TO_SECTION[section])
    return "".join(out)


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = prefix + str(k)
            if isinstance(v, dict) and k != "retry_directives":
                out.update(flatten(v, key + "."))
            else:
                out[key] = v
    return out


def check(*, base_path: Path = BASE, c2_path: Path = C2, manifest: Path | None = MANIFEST,
          require_manifest: bool = False) -> list[str]:
    """실패 사유 목록. 비면 통과."""
    errs: list[str] = []
    hf, from_manifest = hf_values(manifest)
    if require_manifest and not from_manifest:
        errs.append("MANIFEST_C2.json 이 없다 (C-L 전). 실행 경로는 MANIFEST 를 요구한다: %s" % manifest)
    if not Path(c2_path).exists():
        return errs + ["config_c2.yaml 이 없다: %s" % c2_path]
    base_text = io.open(base_path, encoding="utf-8", newline="").read()
    c2_text = io.open(c2_path, encoding="utf-8", newline="").read()
    want = render(base_text, hf)
    if c2_text != want:
        a, b = c2_text.splitlines(), want.splitlines()
        first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
        errs.append("바이트 대조 실패: config_c2.yaml != render(config.yaml%s), 첫 차이 줄 %d"
                    % (", MANIFEST" if from_manifest else ", 자리표시", first + 1))
    A = flatten(yaml.safe_load(base_text))
    B = flatten(yaml.safe_load(c2_text))
    changed = {k for k in set(A) | set(B) if A.get(k, "<absent>") != B.get(k, "<absent>")}
    if changed != EXPECTED_CHANGED:
        errs.append("값이 다른 키 집합이 §4.3 표와 다르다: 남는 %s / 빠진 %s"
                    % (sorted(changed - EXPECTED_CHANGED), sorted(EXPECTED_CHANGED - changed)))
    for k, v in EXPECTED_SAME.items():
        if B.get(k, "<absent>") != v:
            errs.append("%s 는 %r 이어야 한다 (지금 %r)" % (k, v, B.get(k)))
    for k in HF_KEYS:
        if str(B.get("data." + k)) != hf[k]:
            errs.append("data.%s = %r 가 %s 값 %r 와 다르다" % (k, B.get("data." + k), "MANIFEST" if from_manifest else "자리표시", hf[k]))
    if from_manifest and not re.fullmatch(r"[0-9a-f]{40}", hf["hf_revision"]):
        errs.append("MANIFEST 의 hf_revision 이 40자리 커밋 해시가 아니다: %r" % hf["hf_revision"])
    try:
        sys.path.insert(0, str(ROOT))
        from lfm.config import load_config
        load_config(c2_path)
    except Exception as e:  # noqa: BLE001
        errs.append("lfm.config.load_config 실패: %r" % e)
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="config_c2.yaml 생성·대조 (PREREG_R3 §4.3)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--require-manifest", action="store_true")
    args = ap.parse_args(argv)
    if args.write:
        hf, fm = hf_values()
        text = render(io.open(BASE, encoding="utf-8", newline="").read(), hf)
        io.open(C2, "w", encoding="utf-8", newline="").write(text)
        print("-> %s (data.hf_* = %s)" % (C2, "MANIFEST" if fm else "자리표시, C-L 전"))
    errs = check(require_manifest=args.require_manifest)
    for e in errs:
        print("FAIL " + e)
    if not errs:
        print("PASS config_c2.yaml = config.yaml + §4.3 표 (%d개 키)" % len(EXPECTED_CHANGED))
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
