# R2 분석 요약 — `results\r2\heldout35_nano`

자동 생성(`experiments/r2_analyze.py`). 수치는 모두 같은 디렉터리의 `ANALYSIS.json` 에서 왔다. 부트스트랩 10000회, seed 42, topic × seed 2단 계층. 문장 지지: **minicheck_rescored**.

## 형식에서 오는 제약

- accounting_tokens = executed branch 의 tokens_in + tokens_out (물려받은 branch 제외)
- billed_tokens = executed branch 의 billing.billed_in + billed_out. 원장 행에는 arm 이 없어 arm 별 분해는 branch billing 으로 한다
- CRN: 같은 family·seed 에서 뒤에 도는 arm 은 앞 arm 의 캐시를 맞혀 청구가 작다 — 청구 비교는 실행 순서에 의존한다
- E-R1 = E-GR 라운드 0 (실행하지 않음)
- coverage = 최종 라운드 sub-answer 중 유효 인용이 있는 비율(질의 단위 풀링). grounding = 문장 가중 풀링
- R1-8 전이표는 회계 분해이고 인과가 아니다. 토큰은 branch 실행 토큰을 새로 만든 sub-answer 수로 균등 분할했다
- legacy(L) arm 은 status 가 없다. 빈 답·유효 인용 없는 답을 전이표와 기권률에서 insufficient 로 센다 — L 이 기권 status 를 냈다는 뜻이 아니다

## arm × seed

| arm·seed | 토픽 | 회계 토큰 | 청구 토큰 | api | branch-round | subq 실행 | verify 통과 (g / c) | 최종 동결 | 생성 insufficient / format_fail / dropped |
|---|---|---|---|---|---|---|---|---|---|
| E-GR@gpt-5.4-nano·s1 | 35 | 1564783 | 1123342 | 596 | 420 | 1752 | 0 (0 / 0) | 0 | 0.938 / 0.001 / 0.011 |
| E-R1@gpt-5.4-nano·s1 | 35 | 441441 | 0 | 0 | 140 | 584 | 0 (0 / 0) | 0 | 0.909 / 0.002 / 0.021 |
| E-SQ@gpt-5.4-nano·s1 | 35 | 1540735 | 1540735 | 750 | 420 | 1683 | 0 (0 / 0) | 0 | 0.949 / 0.001 / 0.012 |

## 절감 — 전체 토픽

절감 = 1 − ΣA/ΣB. 주장 조건: seed 부호 일치 ∧ 95% CI 가 0 제외.

| 비교 | 지표 | 점추정 | 95% CI | seed별 | 주장 |
|---|---|---|---|---|---|
| E-SQ_vs_E-GR@gpt-5.4-nano | accounting_tokens | 0.015 | [0.007, 0.024] | s1 0.015 | 예 |
| E-SQ_vs_E-GR@gpt-5.4-nano | billed_tokens | -0.372 | [-0.393, -0.348] | s1 -0.372 | 보고만 |
| E-SQ_vs_E-GR@gpt-5.4-nano | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| E-SQ_vs_E-GR@gpt-5.4-nano | subq_executed | 0.039 | [0.025, 0.055] | s1 0.039 | 예 |
| L-BR_vs_L-GR@gpt-5.4-nano | accounting_tokens | 짝 없음 | | | |
| L-BR_vs_L-GR@gpt-5.4-nano | billed_tokens | 짝 없음 | | | |
| L-BR_vs_L-GR@gpt-5.4-nano | branch_rounds | 짝 없음 | | | |
| L-BR_vs_L-GR@gpt-5.4-nano | subq_executed | 짝 없음 | | | |
| E-BR_vs_E-SQ@gpt-5.4-nano | accounting_tokens | 짝 없음 | | | |
| E-BR_vs_E-SQ@gpt-5.4-nano | billed_tokens | 짝 없음 | | | |
| E-BR_vs_E-SQ@gpt-5.4-nano | branch_rounds | 짝 없음 | | | |
| E-BR_vs_E-SQ@gpt-5.4-nano | subq_executed | 짝 없음 | | | |

## 자동 비열등성 — 전체 토픽

차이 = A − B (질의 단위 평균). 90% CI 하한 > −δ.

| 비교 | 지표 | 차이 | 90% CI | δ | 비열등 | 동등 |
|---|---|---|---|---|---|---|
| E-SQ_vs_E-GR@gpt-5.4-nano | coverage | 0.045 | [0.028, 0.063] | 0.05 | 예 | 아니오 |
| E-SQ_vs_E-GR@gpt-5.4-nano | grounding | 0.308 | [0.179, 0.443] | 0.05 | 예 | 아니오 |
| E-R1_vs_E-GR@gpt-5.4-nano | coverage | 0.029 | [0.013, 0.045] | 0.05 | 예 | 예 |
| E-R1_vs_E-GR@gpt-5.4-nano | grounding | 0.199 | [0.067, 0.334] | 0.05 | 예 | 아니오 |

## 절감 — 30토픽 민감도 (t000~t004 제외)

절감 = 1 − ΣA/ΣB. 주장 조건: seed 부호 일치 ∧ 95% CI 가 0 제외.

| 비교 | 지표 | 점추정 | 95% CI | seed별 | 주장 |
|---|---|---|---|---|---|
| E-SQ_vs_E-GR@gpt-5.4-nano | accounting_tokens | 0.014 | [0.006, 0.023] | s1 0.014 | 예 |
| E-SQ_vs_E-GR@gpt-5.4-nano | billed_tokens | -0.370 | [-0.393, -0.345] | s1 -0.370 | 보고만 |
| E-SQ_vs_E-GR@gpt-5.4-nano | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| E-SQ_vs_E-GR@gpt-5.4-nano | subq_executed | 0.039 | [0.023, 0.056] | s1 0.039 | 예 |
| L-BR_vs_L-GR@gpt-5.4-nano | accounting_tokens | 짝 없음 | | | |
| L-BR_vs_L-GR@gpt-5.4-nano | billed_tokens | 짝 없음 | | | |
| L-BR_vs_L-GR@gpt-5.4-nano | branch_rounds | 짝 없음 | | | |
| L-BR_vs_L-GR@gpt-5.4-nano | subq_executed | 짝 없음 | | | |
| E-BR_vs_E-SQ@gpt-5.4-nano | accounting_tokens | 짝 없음 | | | |
| E-BR_vs_E-SQ@gpt-5.4-nano | billed_tokens | 짝 없음 | | | |
| E-BR_vs_E-SQ@gpt-5.4-nano | branch_rounds | 짝 없음 | | | |
| E-BR_vs_E-SQ@gpt-5.4-nano | subq_executed | 짝 없음 | | | |

## 자동 비열등성 — 30토픽 민감도 (t000~t004 제외)

차이 = A − B (질의 단위 평균). 90% CI 하한 > −δ.

| 비교 | 지표 | 차이 | 90% CI | δ | 비열등 | 동등 |
|---|---|---|---|---|---|---|
| E-SQ_vs_E-GR@gpt-5.4-nano | coverage | 0.048 | [0.032, 0.064] | 0.05 | 예 | 아니오 |
| E-SQ_vs_E-GR@gpt-5.4-nano | grounding | 0.342 | [0.209, 0.483] | 0.05 | 예 | 아니오 |
| E-R1_vs_E-GR@gpt-5.4-nano | coverage | 0.034 | [0.017, 0.051] | 0.05 | 예 | 아니오 |
| E-R1_vs_E-GR@gpt-5.4-nano | grounding | 0.215 | [0.069, 0.361] | 0.05 | 예 | 아니오 |

## R1-8 상태 전이 — 회계 분해 (인과 아님)

### E-GR@gpt-5.4-nano·s1

재실행 단위 1168. 형식 회복 몫 0 토큰 (상한 582). 회복 불가 지출 1362001 토큰. insufficient→answered 0.016.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 1066 | 5 | 3 | 14 |
| sufficient | 0 | 29 | 1 | 3 | 14 |
| dropped | 0 | 10 | 1 | 1 | 3 |
| cited-unsupported | 0 | 6 | 1 | 8 | 2 |
| format_fail | 0 | 1 | 0 | 0 | 0 |

### E-SQ@gpt-5.4-nano·s1

재실행 단위 1099. 형식 회복 몫 0 토큰 (상한 515). 회복 불가 지출 1387340 토큰. insufficient→answered 0.015.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 1 | 1046 | 6 | 5 | 11 |
| cited-unsupported | 0 | 8 | 0 | 7 | 0 |
| dropped | 0 | 11 | 2 | 0 | 1 |
| format_fail | 0 | 1 | 0 | 0 | 0 |

