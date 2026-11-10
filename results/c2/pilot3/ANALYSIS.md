# R2 분석 요약 — `C:\Users\skyop\Desktop\나의 rag\03_LFM\LFM\m1\results\c2\pilot3`

자동 생성(`experiments/r2_analyze.py`). 수치는 모두 같은 디렉터리의 `ANALYSIS.json` 에서 왔다. 부트스트랩 10000회, seed 42, topic × seed 2단 계층. 문장 지지: **minicheck_rescored**.

## 형식에서 오는 제약

- accounting_tokens = executed branch 의 tokens_in + tokens_out (물려받은 branch 제외)
- billed_tokens = executed branch 의 billing.billed_in + billed_out. 원장 행에는 arm 이 없어 arm 별 분해는 branch billing 으로 한다
- CRN: 같은 family·seed 에서 뒤에 도는 arm 은 앞 arm 의 캐시를 맞혀 청구가 작다 — 청구 비교는 실행 순서에 의존한다
- E-R1 = E-GR 라운드 0 (실행하지 않음)
- coverage = 최종 라운드 sub-answer 중 유효 인용이 있는 비율(질의 단위 풀링). grounding = 문장 가중 풀링
- R1-8 전이표는 회계 분해이고 인과가 아니다. 토큰은 branch 실행 토큰을 새로 만든 sub-answer 수로 균등 분할했다
- legacy(L) arm 은 status 가 없다. 빈 답·유효 인용 없는 답을 전이표와 기권률에서 insufficient 로 센다 — L 이 기권 status 를 냈다는 뜻이 아니다
- 코퍼스2: R2 의 30토픽 민감도(t000~t004 제외)는 해당 없음 — 블록을 지웠다

## arm × seed

| arm·seed | 토픽 | 회계 토큰 | 청구 토큰 | api | branch-round | subq 실행 | verify 통과 (g / c) | 최종 동결 | 생성 insufficient / format_fail / dropped |
|---|---|---|---|---|---|---|---|---|---|
| E-BR·s1 | 3 | 171141 | 118623 | 58 | 36 | 156 | 0 (0 / 0) | 0 | 0.840 / 0.000 / 0.000 |
| E-GR·s1 | 3 | 171141 | 0 | 0 | 36 | 156 | 0 (0 / 0) | 0 | 0.840 / 0.000 / 0.000 |
| E-R1·s1 | 3 | 52518 | 0 | 0 | 12 | 52 | 0 (0 / 0) | 0 | 0.712 / 0.000 / 0.000 |
| E-SQ·s1 | 3 | 177831 | 177831 | 81 | 36 | 153 | 0 (0 / 0) | 0 | 0.778 / 0.033 / 0.000 |
| L-BR·s1 | 3 | 211375 | 211375 | 97 | 36 | 156 | 0 (0 / 0) | 0 | 0.000 / 0.000 / 0.000 |
| L-GR·s1 | 3 | 211375 | 0 | 0 | 36 | 156 | 0 (0 / 0) | 0 | 0.000 / 0.000 / 0.000 |

## 절감 — 전체 토픽

절감 = 1 − ΣA/ΣB. 주장 조건: seed 부호 일치 ∧ 95% CI 가 0 제외.

| 비교 | 지표 | 점추정 | 95% CI | seed별 | 주장 |
|---|---|---|---|---|---|
| E-SQ_vs_E-GR | accounting_tokens | -0.039 | [-0.073, -0.005] | s1 -0.039 | 예 |
| E-SQ_vs_E-GR | billed_tokens | — | [nan, nan] | s1 — | 보고만 |
| E-SQ_vs_E-GR | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| E-SQ_vs_E-GR | subq_executed | 0.019 | [0.000, 0.050] | s1 0.019 | 아니오 |
| L-BR_vs_L-GR | accounting_tokens | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| L-BR_vs_L-GR | billed_tokens | — | [nan, nan] | s1 — | 보고만 |
| L-BR_vs_L-GR | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| L-BR_vs_L-GR | subq_executed | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| E-BR_vs_E-SQ | accounting_tokens | 0.038 | [0.005, 0.068] | s1 0.038 | 예 |
| E-BR_vs_E-SQ | billed_tokens | 0.333 | [0.326, 0.343] | s1 0.333 | 보고만 |
| E-BR_vs_E-SQ | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| E-BR_vs_E-SQ | subq_executed | -0.020 | [-0.053, 0.000] | s1 -0.020 | 아니오 |

## 자동 비열등성 — 전체 토픽

차이 = A − B (질의 단위 평균). 90% CI 하한 > −δ.

| 비교 | 지표 | 차이 | 90% CI | δ | 비열등 | 동등 |
|---|---|---|---|---|---|---|
| E-SQ_vs_E-GR | coverage | 0.079 | [-0.042, 0.200] | 0.05 | 예 | 아니오 |
| E-SQ_vs_E-GR | grounding | -0.173 | [-0.284, -0.062] | 0.05 | 아니오 | 아니오 |
| E-R1_vs_E-GR | coverage | 0.229 | [0.167, 0.292] | 0.05 | 예 | 아니오 |
| E-R1_vs_E-GR | grounding | 0.111 | [-0.120, 0.341] | 0.05 | 아니오 | 아니오 |

## 절감 — 30토픽 민감도 — 코퍼스2 해당 없음(전체 토픽과 같은 값)

절감 = 1 − ΣA/ΣB. 주장 조건: seed 부호 일치 ∧ 95% CI 가 0 제외.

| 비교 | 지표 | 점추정 | 95% CI | seed별 | 주장 |
|---|---|---|---|---|---|
| E-SQ_vs_E-GR | accounting_tokens | -0.039 | [-0.073, -0.005] | s1 -0.039 | 예 |
| E-SQ_vs_E-GR | billed_tokens | — | [nan, nan] | s1 — | 보고만 |
| E-SQ_vs_E-GR | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| E-SQ_vs_E-GR | subq_executed | 0.019 | [0.000, 0.050] | s1 0.019 | 아니오 |
| L-BR_vs_L-GR | accounting_tokens | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| L-BR_vs_L-GR | billed_tokens | — | [nan, nan] | s1 — | 보고만 |
| L-BR_vs_L-GR | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| L-BR_vs_L-GR | subq_executed | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| E-BR_vs_E-SQ | accounting_tokens | 0.038 | [0.005, 0.068] | s1 0.038 | 예 |
| E-BR_vs_E-SQ | billed_tokens | 0.333 | [0.326, 0.343] | s1 0.333 | 보고만 |
| E-BR_vs_E-SQ | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000 | 아니오 |
| E-BR_vs_E-SQ | subq_executed | -0.020 | [-0.053, 0.000] | s1 -0.020 | 아니오 |

## 자동 비열등성 — 30토픽 민감도 — 코퍼스2 해당 없음(전체 토픽과 같은 값)

차이 = A − B (질의 단위 평균). 90% CI 하한 > −δ.

| 비교 | 지표 | 차이 | 90% CI | δ | 비열등 | 동등 |
|---|---|---|---|---|---|---|
| E-SQ_vs_E-GR | coverage | 0.079 | [-0.042, 0.200] | 0.05 | 예 | 아니오 |
| E-SQ_vs_E-GR | grounding | -0.173 | [-0.284, -0.062] | 0.05 | 아니오 | 아니오 |
| E-R1_vs_E-GR | coverage | 0.229 | [0.167, 0.292] | 0.05 | 예 | 아니오 |
| E-R1_vs_E-GR | grounding | 0.111 | [-0.120, 0.341] | 0.05 | 아니오 | 아니오 |

## R1-8 상태 전이 — 회계 분해 (인과 아님)

### E-BR·s1

재실행 단위 104. 형식 회복 몫 0 토큰 (상한 1545). 회복 불가 지출 105563 토큰. insufficient→answered 0.085.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 75 | 0 | 7 | 0 |
| cited-unsupported | 0 | 18 | 0 | 3 | 0 |
| sufficient | 0 | 1 | 0 | 0 | 0 |

### E-GR·s1

재실행 단위 104. 형식 회복 몫 0 토큰 (상한 1545). 회복 불가 지출 105563 토큰. insufficient→answered 0.085.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 75 | 0 | 7 | 0 |
| cited-unsupported | 0 | 18 | 0 | 3 | 0 |
| sufficient | 0 | 1 | 0 | 0 | 0 |

### E-SQ·s1

재실행 단위 101. 형식 회복 몫 0 토큰 (상한 1783). 회복 불가 지출 96612 토큰. insufficient→answered 0.093.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 3 | 65 | 0 | 7 | 0 |
| cited-unsupported | 2 | 15 | 0 | 3 | 1 |
| format_fail | 0 | 2 | 0 | 3 | 0 |

### L-BR·s1

재실행 단위 104. 형식 회복 몫 0 토큰 (상한 3173). 회복 불가 지출 8709 토큰. insufficient→answered 0.300.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| sufficient | 0 | 4 | 0 | 6 | 2 |
| cited-unsupported | 0 | 17 | 0 | 63 | 2 |
| insufficient | 0 | 7 | 0 | 3 | 0 |

### L-GR·s1

재실행 단위 104. 형식 회복 몫 0 토큰 (상한 3173). 회복 불가 지출 8709 토큰. insufficient→answered 0.300.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| sufficient | 0 | 4 | 0 | 6 | 2 |
| cited-unsupported | 0 | 17 | 0 | 63 | 2 |
| insufficient | 0 | 7 | 0 | 3 | 0 |

## 파일럿 관문

| 관문 | 항목 | 통과 |
|---|---|---|
| G1 | 1_seed_key_disjoint | None |
| G1 | 2_decompose_api_calls | True |
| G1 | 3_answer_format | True |
| G1 | 4_repair_calls | False |
| G1 | 5_fabricated_ids_left | True ({'E': 0, 'L': 28}) |
| G1 | 6_dropped_uncited_sentences | True |
| G1 | 7_ledger_equals_usage | True |
| G1 | 7b_branch_billing_equals_usage | True |
| G1 | 8_minicheck_cuda_single_chunk | None |
| G1 | 9_inherited_bytes_identical | True |
| G1 | round0_identical_within_family_seed | True |
| G3 | SQ 부분 재실행 ≥ 20% ∧ SQ 입력 < BR | False |
| G4 | 외삽 청구(branch billing·원장 중 큰 쪽) 8991600 / 3777125 ≤ 12600000 / 3400000 | False |

G2·G5 는 사람 라벨이 필요하다. 효과 관문은 없다(PREREG §6).
