# R2 분석 요약 — `results\r2\pilot5`

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
| E-BR·s1 | 5 | 265323 | 186227 | 90 | 60 | 264 | 0 (0 / 0) | 0 | 0.841 / 0.000 / 0.000 |
| E-GR·s1 | 5 | 265323 | 0 | 0 | 60 | 264 | 0 (0 / 0) | 0 | 0.841 / 0.000 / 0.000 |
| E-R1·s1 | 5 | 79096 | 0 | 0 | 20 | 88 | 0 (0 / 0) | 0 | 0.739 / 0.000 / 0.000 |
| E-SQ·s1 | 5 | 254906 | 254906 | 116 | 60 | 236 | 0 (0 / 0) | 0 | 0.864 / 0.004 / 0.000 |
| L-BR·s1 | 5 | 294468 | 162684 | 77 | 60 | 264 | 0 (1 / 0) | 0 | 0.000 / 0.000 / 0.000 |
| L-GR·s1 | 5 | 294468 | 0 | 0 | 60 | 264 | 0 (1 / 0) | 0 | 0.000 / 0.000 / 0.000 |

## 절감 — 전체 토픽

절감 = 1 − ΣA/ΣB. 주장 조건: seed 부호 일치 ∧ 95% CI 가 0 제외.

| 비교 | 지표 | 점추정 | 95% CI | seed별 | 주장 |
|---|---|---|---|---|---|
| E-SQ_vs_E-GR | accounting_tokens | 0.039 | [-0.003, 0.091] | s1 0.039 | 아니오 |
| E-SQ_vs_E-GR | billed_tokens | — | [nan, nan] | s1 — | 보고만 |
| E-SQ_vs_E-GR | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| E-SQ_vs_E-GR | subq_executed | 0.106 | [0.042, 0.183] | s1 0.106 | 예 |
| L-BR_vs_L-GR | accounting_tokens | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| L-BR_vs_L-GR | billed_tokens | — | [nan, nan] | s1 — | 보고만 |
| L-BR_vs_L-GR | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| L-BR_vs_L-GR | subq_executed | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| E-BR_vs_E-SQ | accounting_tokens | -0.041 | [-0.101, 0.003] | s1 -0.041 | 아니오 |
| E-BR_vs_E-SQ | billed_tokens | 0.269 | [0.231, 0.305] | s1 0.269 | 보고만 |
| E-BR_vs_E-SQ | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| E-BR_vs_E-SQ | subq_executed | -0.119 | [-0.224, -0.043] | s1 -0.119 | 예 |

## 자동 비열등성 — 전체 토픽

차이 = A − B (질의 단위 평균). 90% CI 하한 > −δ.

| 비교 | 지표 | 차이 | 90% CI | δ | 비열등 | 동등 |
|---|---|---|---|---|---|---|
| E-SQ_vs_E-GR | coverage | 0.085 | [0.010, 0.165] | 0.05 | 예 | 아니오 |
| E-SQ_vs_E-GR | grounding | 0.210 | [-0.350, 0.760] | 0.05 | 아니오 | 아니오 |
| E-R1_vs_E-GR | coverage | 0.128 | [0.025, 0.227] | 0.05 | 예 | 아니오 |
| E-R1_vs_E-GR | grounding | 0.180 | [-0.420, 0.780] | 0.05 | 아니오 | 아니오 |

## 절감 — 30토픽 민감도 (t000~t004 제외)

절감 = 1 − ΣA/ΣB. 주장 조건: seed 부호 일치 ∧ 95% CI 가 0 제외.

| 비교 | 지표 | 점추정 | 95% CI | seed별 | 주장 |
|---|---|---|---|---|---|
| E-SQ_vs_E-GR | accounting_tokens | 0.039 | [-0.003, 0.091] | s1 0.039 | 아니오 |
| E-SQ_vs_E-GR | billed_tokens | — | [nan, nan] | s1 — | 보고만 |
| E-SQ_vs_E-GR | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| E-SQ_vs_E-GR | subq_executed | 0.106 | [0.042, 0.183] | s1 0.106 | 예 |
| L-BR_vs_L-GR | accounting_tokens | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| L-BR_vs_L-GR | billed_tokens | — | [nan, nan] | s1 — | 보고만 |
| L-BR_vs_L-GR | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| L-BR_vs_L-GR | subq_executed | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| E-BR_vs_E-SQ | accounting_tokens | -0.041 | [-0.101, 0.003] | s1 -0.041 | 아니오 |
| E-BR_vs_E-SQ | billed_tokens | 0.269 | [0.231, 0.305] | s1 0.269 | 보고만 |
| E-BR_vs_E-SQ | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| E-BR_vs_E-SQ | subq_executed | -0.119 | [-0.224, -0.043] | s1 -0.119 | 예 |

## 자동 비열등성 — 30토픽 민감도 (t000~t004 제외)

차이 = A − B (질의 단위 평균). 90% CI 하한 > −δ.

| 비교 | 지표 | 차이 | 90% CI | δ | 비열등 | 동등 |
|---|---|---|---|---|---|---|
| E-SQ_vs_E-GR | coverage | 0.085 | [0.010, 0.165] | 0.05 | 예 | 아니오 |
| E-SQ_vs_E-GR | grounding | 0.210 | [-0.350, 0.760] | 0.05 | 아니오 | 아니오 |
| E-R1_vs_E-GR | coverage | 0.128 | [0.025, 0.227] | 0.05 | 예 | 아니오 |
| E-R1_vs_E-GR | grounding | 0.180 | [-0.420, 0.780] | 0.05 | 아니오 | 아니오 |

## R1-8 상태 전이 — 회계 분해 (인과 아님)

### E-BR·s1

재실행 단위 176. 형식 회복 몫 0 토큰 (상한 1075). 회복 불가 지출 166149 토큰. insufficient→answered 0.075.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| sufficient | 0 | 13 | 0 | 2 | 3 |
| insufficient | 0 | 136 | 0 | 5 | 6 |
| cited-unsupported | 0 | 8 | 0 | 1 | 2 |

### E-GR·s1

재실행 단위 176. 형식 회복 몫 0 토큰 (상한 1075). 회복 불가 지출 166149 토큰. insufficient→answered 0.075.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| sufficient | 0 | 13 | 0 | 2 | 3 |
| insufficient | 0 | 136 | 0 | 5 | 6 |
| cited-unsupported | 0 | 8 | 0 | 1 | 2 |

### E-SQ·s1

재실행 단위 148. 형식 회복 몫 0 토큰 (상한 0). 회복 불가 지출 190932 토큰. insufficient→answered 0.036.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 1 | 131 | 0 | 4 | 1 |
| cited-unsupported | 0 | 8 | 0 | 3 | 0 |

### L-BR·s1

재실행 단위 176. 형식 회복 몫 0 토큰 (상한 6243). 회복 불가 지출 88585 토큰. insufficient→answered 0.218.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| sufficient | 0 | 24 | 0 | 7 | 14 |
| insufficient | 0 | 79 | 0 | 7 | 15 |
| cited-unsupported | 0 | 7 | 0 | 15 | 8 |

### L-GR·s1

재실행 단위 176. 형식 회복 몫 0 토큰 (상한 6243). 회복 불가 지출 88585 토큰. insufficient→answered 0.218.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| sufficient | 0 | 24 | 0 | 7 | 14 |
| insufficient | 0 | 79 | 0 | 7 | 15 |
| cited-unsupported | 0 | 7 | 0 | 15 | 8 |

## 파일럿 관문

| 관문 | 항목 | 통과 |
|---|---|---|
| G1 | 1_seed_key_disjoint | None |
| G1 | 2_decompose_api_calls | True |
| G1 | 3_answer_format | True |
| G1 | 4_repair_calls | True |
| G1 | 5_fabricated_ids_left | True ({'E': 0, 'L': 10}) |
| G1 | 6_dropped_uncited_sentences | True |
| G1 | 7_ledger_equals_usage | False |
| G1 | 7b_branch_billing_equals_usage | True |
| G1 | 8_minicheck_cuda_single_chunk | None |
| G1 | 9_inherited_bytes_identical | True |
| G1 | round0_identical_within_family_seed | True |
| G3 | SQ 부분 재실행 ≥ 20% ∧ SQ 입력 < BR | True |
| G4 | 외삽 청구(branch billing·원장 중 큰 쪽) 12084912 / 3362709 ≤ 15000000 / 5600000 | True |

G2·G5 는 사람 라벨이 필요하다. 효과 관문은 없다(PREREG §6).
