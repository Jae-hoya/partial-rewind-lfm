# P1 보고 — 코퍼스 구축 + 스키마 게이트

산출: `data/corpus.parquet`, `data/MANIFEST.json`, `data/download_stats.json`
판정: **PASS** (필수 필드 충족, `schema_check.py` exit 0)

## 스냅샷

| 항목 | 값 |
|---|---|
| repo | `wikimedia/structured-wikipedia` |
| revision (고정) | `417c267bb457fa645c22eb3b5c77764963194c70` (2026-05-19) |
| config | `enwiki_namespace_0` (영어판 namespace 0) |
| 라이선스 | CC BY-SA 4.0 — Wikimedia Enterprise / Wikipedia contributors |
| 상류 규모 | 86 샤드 / 37.2 GB |
| 코퍼스 | 30,000 문서 / 293.5 MB / 19 컬럼 |
| SHA256 | `a6aacbbb4848a1fe50c4420c5bc0909da893dedda76d0cf84ae4a70d6cc2b4bf` |
| 소요 | 다운로드 673.1s, 스키마 검사 ~90s |
| 부족분 | 0 |

MANIFEST에 상류 86개 샤드의 LFS SHA256을 전부 기록했다. 37GB를 받지 않고도 무결성
근거가 남는다.

## 표집 방법 — 왜 샤드 하나로는 안 되는가

상류 행은 `name` 기준으로 느슨하게 정렬돼 있다. 샤드 0은 `"Good day, fellow!"` ~
`Ćustić` 구간에 몰려 있어, 그 안에서 3만 건을 뽑으면 D 이후 문자로 시작하는 주제가
통째로 빠진다. P2의 교차 도메인 토픽 생성이 성립하지 않는다.

그래서 **전 샤드 층화 표집**을 한다:

1. 86개 샤드 각각에 대해 시드(`config.seed=42`)로 row group 2개를 고른다.
2. HTTP range read로 그 row group만 읽는다 (샤드 전체 다운로드 없음).
3. `abstract ≥ 200자` 필터 후 샤드 할당량(349/348건)만큼 시드 표집한다.

같은 seed + 같은 revision이면 같은 (샤드, row group, 행)이 선택된다. 재실행하면
같은 코퍼스다.

결과 첫 글자 분포는 자연스럽다: S 2,293 · M 2,063 · A 1,993 · T 1,933 · C 1,897 ·
L 1,751 · B 1,552 · P 1,446 …

## 필드 판정 (30,000건 기준, 채움률 = non-null AND non-empty)

| 필드 | 존재 | 채움률 | 등급 |
|---|---|---|---|
| identifier / name / url / abstract / sections | O | 1.0000 | HARD — 통과 |
| references | O | 0.9282 | SOFT |
| infoboxes | O | 0.7593 | INFO |
| description | O | 0.8903 | INFO |
| image | O | 0.4632 | INFO |
| tables | O | 0.3255 | INFO |
| `version.scores.referencerisk` | O | **0.9063** | 신뢰도 신호 |
| `version.scores.referenceneed` | O | 0.8835 | 신뢰도 신호 |
| `version.scores.revertrisk` | O | 0.9809 | 신뢰도 신호 |
| redirect 계열 (`redirects`/`redirect`/…) | **X** | — | 부재 |

판정값:
- `citation_mode = "references"` (부재 시 `chunk` 폴백이었으나 92.8%로 충족)
- `credibility_signal_usable = True`
- `heldout_match_mode = "title_normalization"` — **redirect 필드가 스키마에 없다.**
  SPEC v1.1 §2.2가 예상한 폴백 경로다. P3 `heldout.py`는 title 정규화 매칭을 쓴다.

## referencerisk — 샤드 간 이질성 (HANDOFF §4 판정 지점)

**샤드 0만 보면 referencerisk 채움률은 0.47%다. 전 코퍼스로는 90.63%다.**
`abstract ≥ 200자` 필터로는 설명되지 않는다 (샤드 0 내 긴 문서만 봐도 0.53%).
row group 1개씩 직접 재본 값:

| 샤드 | referencerisk 채움률 |
|---|---|
| 0 | 0.0005 |
| 1 | 0.0073 |
| 2 | 0.0044 |
| 40 | 0.9912 |
| 85 | 0.9912 |

저번호 샤드에만 점수가 비어 있다. 상류 채점 파이프라인의 산물로 보이며 원인은
확인하지 않았다.

**결론: M2 Test C-2(referencerisk 상위 문서로 자연 conflict 후보 선별)는 이 스냅샷에서
성립한다.** 단, 코퍼스를 저번호 샤드에서만 뽑으면 성립하지 않는다. M2에서
`max_articles`를 올릴 때도 층화 표집을 유지해야 한다.

## 미결

- `data/raw/enwiki/data/enwiki_namespace_0_0.parquet` (371 MB): 초기 스키마 탐색용으로
  받은 샤드 전체. 층화 표집 경로에서는 쓰지 않는다. 위 이질성 진단 재현에만 쓰이므로
  지워도 무방하다.
