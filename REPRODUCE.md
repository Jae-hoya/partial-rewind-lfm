# 재현 절차

전 단계가 시드 고정이다. 같은 `config.yaml` + 같은 스냅샷 revision → 같은 결과.
LLM 응답은 디스크 캐시에서 나오므로 2회차 실행은 API 호출 0회로 끝난다.

## 0. 환경

```bash
C:\Users\skyop\anaconda3\python.exe -m pip install -e ".[dev]"
```

파이썬은 반드시 위 절대경로로 실행한다. 그냥 `python` 은 Windows 스토어 껍데기라
아무것도 안 하고 exit 0 으로 끝난다.

`.env` 에 `OPENAI_API_KEY` 를 넣는다 (`.env.example` 참고). `.gitignore` 1번 줄에 있다.

한국어 로그가 깨지면 `PYTHONIOENCODING=utf-8` 을 준다.

## 1. 순서 — 어기면 held-out 이 샌다

```
download → schema_check → topics(후보) → [승인] → heldout → chunk_index → test_a
```

`chunk_index.py` 는 `heldout_ids.json` 이 없으면 에러로 종료한다. 승인 없이 다음
단계를 돌리면 `heldout.py` 가 에러로 종료한다. 순서는 코드가 강제한다.

```bash
# P1 — 코퍼스 (약 16분, 전송 수백 MB)
C:\Users\skyop\anaconda3\python.exe data/download.py
C:\Users\skyop\anaconda3\python.exe data/schema_check.py      # PASS 여야 계속

# P2 — 토픽 후보 (LLM 6콜) → 승인 게이트
C:\Users\skyop\anaconda3\python.exe data/topics.py            # 후보 출력 후 강제 종료
C:\Users\skyop\anaconda3\python.exe data/approve_topics.py    # 기계 검증 후 approved:true

# P3 — held-out 확정 → 인덱스 (약 6분)
C:\Users\skyop\anaconda3\python.exe data/heldout.py
C:\Users\skyop\anaconda3\python.exe data/chunk_index.py

# P4 — Test A
C:\Users\skyop\anaconda3\python.exe experiments/test_a.py
C:\Users\skyop\anaconda3\python.exe experiments/waste_supplement.py results/test_a/<ts> --append
```

## 2. 검증

```bash
C:\Users\skyop\anaconda3\python.exe -m pytest                                     # 205개
C:\Users\skyop\anaconda3\python.exe experiments/trace_validate.py results/test_a/<ts>/trace.jsonl
C:\Users\skyop\anaconda3\python.exe experiments/model_check.py                    # SPEC §1, 1콜
```

**결정성 확인**: 같은 config 로 두 번 돌리고 `waste_ratio_ci.csv` 를 diff 한다.
2회차는 전부 캐시 히트여야 하고 청구 토큰이 0 이어야 한다.

## 3. 스냅샷·모델 고정

| 항목 | 값 |
|---|---|
| 데이터 | HF `wikimedia/structured-wikipedia` |
| revision | `417c267bb457fa645c22eb3b5c77764963194c70` (2026-05-19) |
| config | `enwiki_namespace_0` (영어판 namespace 0) |
| 라이선스 | CC BY-SA 4.0 — Wikimedia Enterprise / Wikipedia contributors |
| 코퍼스 SHA256 | `7fa93410a5e301b9ca48048c396acff03daa1f7c0ff047c6fa93e4280e308816` |
| 생성 모델 | `gpt-5-mini`, `reasoning_effort=low`, `max_tokens=2000` |
| NLI | `cross-encoder/nli-deberta-v3-base` (CUDA), `tau_ground=0.5` |
| 토크나이저 | `o200k_base` |
| seed | 42 |

상류 86개 샤드의 LFS SHA256 은 `data/MANIFEST.json` 에 있다.

## 4. 재현되지 않는 것

- **`latency_ms`**: 캐시 미스일 때만 실측값이다. 캐시 히트는 최초 측정값을 그대로
  돌려준다 — 재실행 결과를 바이트 동일하게 유지하려는 의도적 선택이다.
- **`gpt-5-mini` 응답 자체**: 캐시를 지우면 같은 프롬프트라도 응답이 달라질 수 있다.
  재현의 단위는 캐시를 포함한 실행 디렉토리다. 캐시 없이 재현하려면 수치가 아니라
  결론(τ별 낭비율 밴드)이 재현되는지를 본다.
- **인덱스 빌드 시간·메모리**: 하드웨어 의존.

## 5. 산출물 지도

| 경로 | 내용 |
|---|---|
| `data/MANIFEST.json` | 스냅샷 식별자·SHA256·필드 판정·중복률 |
| `data/download_stats.json` | 샤드별 표집 내역 (row group, 할당량, 소요) |
| `data/topics_pilot50_candidates.yaml` | 토픽 후보 60개 + 생성 비용 |
| `data/topics_pilot50.yaml` | 승인 50개 + 승인 경위 + 탈락 사유 |
| `data/heldout_ids.json` | 토픽별 held-out 문서 id + 매칭 방식 |
| `data/index_stats.json` | 청크 통계 · 토픽 커버리지 |
| `results/test_a/<ts>/waste_ratio.csv` | (query, round, branch) 원자료 |
| `results/test_a/<ts>/waste_ratio_ci.csv` | τ별 점추정 + bootstrap CI |
| `results/test_a/<ts>/waste_ratio_supplement.csv` | 파일럿 제외 보조 추정 |
| `results/test_a/<ts>/report.md` | 판정 문장 포함 리포트 |
| `results/test_a/<ts>/trace.jsonl` | 전 라운드 trace (동결 스키마) |
| `results/test_a/<ts>/usage.json` | 태그별 호출·토큰 (비용 근거) |
| `results/test_a/_pilot5{,b,c}/` | 파일럿 3회 — 측정 결함 수정의 증거 |

## MiniCheck 설치 — 재현 함정 (2026-09-06)

```
pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git"
```

**`pip install minicheck` 를 그냥 치면 안 된다.** PyPI 의 `minicheck` (0.4.0) 은
동명이인이고 사실검증기가 아니라 **explicit-state model checker** 다. 설치는 성공하고
`import minicheck` 도 되지만 `minicheck.minicheck` 가 없어서 그때 가서야 깨진다.

공식 저장소 설치는 `minicheck-0.1.0` 만 추가하며 transformers/torch/numpy 를
건드리지 않는다 (확인: transformers 4.42.3 · torch 2.3.1+cu118 유지).
첫 import 시 nltk `punkt` 를 내려받는다.
