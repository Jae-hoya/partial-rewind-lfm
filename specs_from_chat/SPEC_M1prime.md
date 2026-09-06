# SPEC M1′ — Test A 신규 실행 (확정 추정자 verifier 탑재) · 논문 Fig.1

발급: 2026-09-11 · 착수 조건: V′ 확정 추정자 + M1-C 완료(GO/STOP 무관 — §3 참조) · `SPEC_M1_TestA` 코드 경로 재사용

## 1. 무엇을 바꾸고 무엇을 안 바꾸나
- **바꾸는 것 하나**: `lfm/verifier.py`와 `sufficient()`의 grounding이 `estimator.py`(V′ 확정 쌍)로 계산된다. verify 규칙(두 절, ≥)·sufficient(>)·τ·라운드 수(3)·GlobalRetry 코드는 그대로.
- **안 바꾸는 것**: 프롬프트(동결), 검색(bm25s k=5), 코퍼스·색인, 토픽 50개, 생성 모델(`gpt-5-mini`, reasoning_effort=low, max_tokens=2000). 미인용 답안도 그대로 둔다(파이프라인 행동이며 Partial Rewind의 regen 분기가 다루는 대상).

## 2. 실행
- 50토픽 × GlobalRetry 3라운드 × **3회 반복**. 생성 API가 시드를 받으면 seed ∈ {42, 43, 44}; 받지 않으면 동일 설정 3회 실행으로 실행 간 분산을 잰다(어느 쪽인지 MANIFEST에 기록).
- 산출: `results/m1prime/<seed>/` — Test A와 동일 스키마(trace, evidence, verify 두 절, sufficient@τ 3개, 토큰·시간). 원본 `results/test_a/20260829-041445/`는 읽기 전용.
- 예상 비용: 1회 ≈ 입력 1.9M / 출력 0.7M 토큰, 68분. 3회 ≈ 3.5시간.

## 3. 산출표 (논문 표 2 세 번째 열, Fig.1)
| 항목 | 값 |
|---|---|
| verify 통과 질의 (라운드 0 / 어느 라운드든), 절별 | 시드별 + 평균 |
| sufficient@0.8 branch 수 (전 라운드) | 시드별 + 평균 |
| $W_{0.7}/W_{0.8}/W_{0.9}$ | 시드별, 질의 부트스트랩 CI, 시드 간 표준편차 |
| 재시도 라운드 호출·토큰 비중 | |
| coverage 라운드0→최종 paired Δ(CI, 부호검정), 미인용 비율 라운드별 | |

**판정(마스터플랜 §6 원안)**: $W_{0.8}$ 평균 ≥60% 그대로 진행 / 40–60 진행 / 20–40 더 어려운 토픽셋 검토 / <20% → 논문 결과 문장 B("headroom is small; Sections VI-B–D test the mechanism"). **어느 경우에도 M2·M3는 진행한다** — 사용자 지시(논문 완성)에 따라 낭비율이 낮은 결과는 연구 중단이 아니라 프레이밍 변경으로 처리한다.

## 4. CLAUDE.md 3줄
- 마일스톤: `M1′ — Test A 신규 실행(확정 추정자)`
- STOP RULE: `시드 1회 완료 후 표 산출·중단·보고 → 승인 후 나머지 2회 → 집계 후 중단·보고`
- 금지: `프롬프트·검색·토픽·모델 변경 금지. 추정자는 estimator.py 경유만. 원본 Test A 디렉토리 쓰기 금지.`
