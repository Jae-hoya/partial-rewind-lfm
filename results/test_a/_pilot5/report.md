# Test A — GlobalRetry 낭비 재실행 측정

생성: 2026-08-29T03:50:47+0900 · run_id `3591f74f-10bb-49e4-9e0c-92c6a23b627a`

## 판정

**τ=0.80 기준 낭비율 0.0% [95% CI 0.0–0.0] → 20% 미만 — 전제 오류. 연구 질문 재설계**

낭비율 = (재실행된 branch 중 재실행 시점에 이미 sufficient 였던 수) / (재실행된 branch 총수).
`sufficient(b) := grounding(b) > τ AND coverage(b) > τ` — 엄격 부등호 (SPEC §4.3).

## 실행 규모

| 항목 | 값 |
|---|---|
| 토픽(질의) 수 | 5 |
| 1라운드 verify 불통과 (실패 케이스) | 5 |
| 최종 통과 | 0 |
| 총 라운드 수 | 15 |
| 재실행된 branch 총수 | 40 |
| degraded branch (절단·파싱 실패) | 0 |
| 모델 | `gpt-5-mini` (reasoning_effort=low) |
| NLI | `cross-encoder/nli-deberta-v3-base` (cuda) |
| 소요 | 7.6 분 |

## τ 스윕 — 전역 검증 임계와 병기

낭비율은 전역 검증 임계 τ_global 과 사후 라벨링 임계 τ 의 **상대 위치**에 직접
좌우된다. 두 값을 같은 표에 둔다 (SPEC §4.5).

| τ | τ_global | 낭비율 | 95% CI | 낭비 branch | 재실행 branch | 실패 쿼리 |
|---|---|---|---|---|---|---|
| 0.70 | 0.80 | 0.0% | 0.0–0.0% | 0 | 40 | 5 |
| 0.80 **(primary)** | 0.80 | 0.0% | 0.0–0.0% | 0 | 40 | 5 |
| 0.90 | 0.80 | 0.0% | 0.0–0.0% | 0 | 40 | 5 |

CI 는 실패 쿼리 집합에서 **쿼리 단위** 복원추출 1000회 (seed=42). 행 단위로 뽑으면
한 쿼리의 4개 branch 가 독립인 것처럼 취급돼 CI 가 좁아진다. LLM 재호출은 0회다.

## 검증 규칙 (SPEC 변경 #2)

```
passed  <=>  mean_b grounding(b) >= 0.80
        AND  min_b  coverage(b)  >= 0.80
```

grounding 은 branch 평균, coverage 는 branch 최솟값이다. 이 비대칭은 의도된 것이다.

## 라운드별 토큰

| 라운드 | 쿼리 수 | LLM 호출 | 입력 토큰 | 출력 토큰 |
|---|---|---|---|---|
| 0 | 5 | 40 | 68299 | 30101 |
| 1 | 5 | 60 | 78583 | 42554 |
| 2 | 5 | 60 | 80646 | 43896 |

## 비용

| 항목 | 값 |
|---|---|
| LLM 호출 | 165 (캐시 히트 67) |
| 입력 토큰 (총/청구) | 228514 / 133808 |
| 출력 토큰 (총/청구) | 119977 / 67860 |

출력 토큰에는 `gpt-5-mini` 의 reasoning 토큰이 포함된다 (MODEL_NOTES.md 참고).

## 산출물

- `waste_ratio.csv` — (query, round, branch) 단위 원자료
- `waste_ratio_ci.csv` — τ별 점추정과 CI
- `waste_ratio.png` — τ별 낭비율 막대 (CI 에러바)
- `trace.jsonl` — 전 라운드 trace (동결 스키마)
- `config.yaml` — 이 실행에 쓰인 설정 사본
- `results.json` — 쿼리별 라운드 상세
