# R2 분석 요약 — `C:\Users\skyop\Desktop\나의 rag\03_LFM\LFM\m1\results\c2\heldout25`

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
| E-BR·s1 | 22 | 1190457 | 1190457 | 546 | 264 | 1182 | 0 (0 / 0) | 0 | 0.849 / 0.000 / 0.001 |
| E-BR·s2 | 22 | 1189462 | 1189462 | 543 | 264 | 1182 | 0 (0 / 0) | 0 | 0.855 / 0.000 / 0.003 |
| E-BR·s3 | 22 | 1198055 | 1198055 | 547 | 264 | 1182 | 0 (0 / 0) | 0 | 0.844 / 0.000 / 0.006 |
| E-GR·s1 | 22 | 1190457 | 0 | 0 | 264 | 1182 | 0 (0 / 0) | 0 | 0.849 / 0.000 / 0.001 |
| E-GR·s2 | 22 | 1189462 | 0 | 0 | 264 | 1182 | 0 (0 / 0) | 0 | 0.855 / 0.000 / 0.003 |
| E-GR·s3 | 22 | 1198055 | 0 | 0 | 264 | 1182 | 0 (0 / 0) | 0 | 0.844 / 0.000 / 0.006 |
| E-R1·s1 | 22 | 357835 | 0 | 0 | 88 | 394 | 0 (0 / 0) | 0 | 0.739 / 0.000 / 0.003 |
| E-R1·s2 | 22 | 359264 | 0 | 0 | 88 | 394 | 0 (0 / 0) | 0 | 0.744 / 0.000 / 0.005 |
| E-R1·s3 | 22 | 358911 | 0 | 0 | 88 | 394 | 0 (0 / 0) | 0 | 0.726 / 0.000 / 0.018 |
| L-BR·s1 | 22 | 1504790 | 1504790 | 701 | 264 | 1182 | 0 (0 / 3) | 0 | 0.000 / 0.000 / 0.000 |
| L-BR·s2 | 22 | 1511043 | 1511043 | 699 | 264 | 1182 | 0 (0 / 5) | 0 | 0.000 / 0.000 / 0.000 |
| L-BR·s3 | 22 | 1517421 | 1517421 | 703 | 262 | 1174 | 0 (0 / 4) | 1 | 0.000 / 0.000 / 0.000 |
| L-GR·s1 | 22 | 1504790 | 0 | 0 | 264 | 1182 | 0 (0 / 3) | 0 | 0.000 / 0.000 / 0.000 |
| L-GR·s2 | 22 | 1511043 | 0 | 0 | 264 | 1182 | 0 (0 / 5) | 0 | 0.000 / 0.000 / 0.000 |
| L-GR·s3 | 22 | 1528431 | 11010 | 6 | 264 | 1182 | 0 (0 / 4) | 0 | 0.000 / 0.000 / 0.000 |

## 절감 — 전체 토픽

절감 = 1 − ΣA/ΣB. 주장 조건: seed 부호 일치 ∧ 95% CI 가 0 제외.

| 비교 | 지표 | 점추정 | 95% CI | seed별 | 주장 |
|---|---|---|---|---|---|
| E-SQ_vs_E-GR | accounting_tokens | 짝 없음 | | | |
| E-SQ_vs_E-GR | billed_tokens | 짝 없음 | | | |
| E-SQ_vs_E-GR | branch_rounds | 짝 없음 | | | |
| E-SQ_vs_E-GR | subq_executed | 짝 없음 | | | |
| L-BR_vs_L-GR | accounting_tokens | 0.002 | [0.000, 0.010] | s1 0.000, s2 0.000, s3 0.007 | 아니오 |
| L-BR_vs_L-GR | billed_tokens | -410.740 | [nan, -99.391] | s1 —, s2 —, s3 -136.822 | 보고만 |
| L-BR_vs_L-GR | branch_rounds | 0.003 | [0.000, 0.010] | s1 0.000, s2 0.000, s3 0.008 | 아니오 |
| L-BR_vs_L-GR | subq_executed | 0.002 | [0.000, 0.009] | s1 0.000, s2 0.000, s3 0.007 | 아니오 |
| E-BR_vs_E-SQ | accounting_tokens | 짝 없음 | | | |
| E-BR_vs_E-SQ | billed_tokens | 짝 없음 | | | |
| E-BR_vs_E-SQ | branch_rounds | 짝 없음 | | | |
| E-BR_vs_E-SQ | subq_executed | 짝 없음 | | | |

## 자동 비열등성 — 전체 토픽

차이 = A − B (질의 단위 평균). 90% CI 하한 > −δ.

| 비교 | 지표 | 차이 | 90% CI | δ | 비열등 | 동등 |
|---|---|---|---|---|---|---|
| E-SQ_vs_E-GR | coverage | 짝 없음 | | | | |
| E-SQ_vs_E-GR | grounding | 짝 없음 | | | | |
| E-R1_vs_E-GR | coverage | 0.165 | [0.123, 0.208] | 0.05 | 예 | 아니오 |
| E-R1_vs_E-GR | grounding | 0.092 | [0.006, 0.177] | 0.05 | 예 | 아니오 |

## 절감 — 30토픽 민감도 — 코퍼스2 해당 없음(전체 토픽과 같은 값)

절감 = 1 − ΣA/ΣB. 주장 조건: seed 부호 일치 ∧ 95% CI 가 0 제외.

| 비교 | 지표 | 점추정 | 95% CI | seed별 | 주장 |
|---|---|---|---|---|---|
| E-SQ_vs_E-GR | accounting_tokens | 짝 없음 | | | |
| E-SQ_vs_E-GR | billed_tokens | 짝 없음 | | | |
| E-SQ_vs_E-GR | branch_rounds | 짝 없음 | | | |
| E-SQ_vs_E-GR | subq_executed | 짝 없음 | | | |
| L-BR_vs_L-GR | accounting_tokens | 0.002 | [0.000, 0.010] | s1 0.000, s2 0.000, s3 0.007 | 아니오 |
| L-BR_vs_L-GR | billed_tokens | -410.740 | [nan, -99.391] | s1 —, s2 —, s3 -136.822 | 보고만 |
| L-BR_vs_L-GR | branch_rounds | 0.003 | [0.000, 0.010] | s1 0.000, s2 0.000, s3 0.008 | 아니오 |
| L-BR_vs_L-GR | subq_executed | 0.002 | [0.000, 0.009] | s1 0.000, s2 0.000, s3 0.007 | 아니오 |
| E-BR_vs_E-SQ | accounting_tokens | 짝 없음 | | | |
| E-BR_vs_E-SQ | billed_tokens | 짝 없음 | | | |
| E-BR_vs_E-SQ | branch_rounds | 짝 없음 | | | |
| E-BR_vs_E-SQ | subq_executed | 짝 없음 | | | |

## 자동 비열등성 — 30토픽 민감도 — 코퍼스2 해당 없음(전체 토픽과 같은 값)

차이 = A − B (질의 단위 평균). 90% CI 하한 > −δ.

| 비교 | 지표 | 차이 | 90% CI | δ | 비열등 | 동등 |
|---|---|---|---|---|---|---|
| E-SQ_vs_E-GR | coverage | 짝 없음 | | | | |
| E-SQ_vs_E-GR | grounding | 짝 없음 | | | | |
| E-R1_vs_E-GR | coverage | 0.165 | [0.123, 0.208] | 0.05 | 예 | 아니오 |
| E-R1_vs_E-GR | grounding | 0.092 | [0.006, 0.177] | 0.05 | 예 | 아니오 |

## R1-8 상태 전이 — 회계 분해 (인과 아님)

### E-BR·s1

재실행 단위 788. 형식 회복 몫 0 토큰 (상한 669). 회복 불가 지출 777858 토큰. insufficient→answered 0.054.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 612 | 0 | 33 | 2 |
| cited-unsupported | 0 | 94 | 0 | 34 | 2 |
| sufficient | 0 | 7 | 0 | 3 | 0 |
| dropped | 0 | 0 | 0 | 1 | 0 |

### E-BR·s2

재실행 단위 788. 형식 회복 몫 0 토큰 (상한 4451). 회복 불가 지출 782349 토큰. insufficient→answered 0.048.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 616 | 1 | 30 | 1 |
| sufficient | 0 | 17 | 0 | 1 | 1 |
| cited-unsupported | 0 | 83 | 0 | 32 | 3 |
| dropped | 0 | 2 | 0 | 1 | 0 |

### E-BR·s3

재실행 단위 788. 형식 회복 몫 0 토큰 (상한 6848). 회복 불가 지출 752951 토큰. insufficient→answered 0.065.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 606 | 0 | 39 | 3 |
| cited-unsupported | 0 | 88 | 0 | 32 | 0 |
| sufficient | 0 | 12 | 0 | 1 | 0 |
| dropped | 0 | 6 | 0 | 1 | 0 |

### E-GR·s1

재실행 단위 788. 형식 회복 몫 0 토큰 (상한 669). 회복 불가 지출 777858 토큰. insufficient→answered 0.054.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 612 | 0 | 33 | 2 |
| cited-unsupported | 0 | 94 | 0 | 34 | 2 |
| sufficient | 0 | 7 | 0 | 3 | 0 |
| dropped | 0 | 0 | 0 | 1 | 0 |

### E-GR·s2

재실행 단위 788. 형식 회복 몫 0 토큰 (상한 4451). 회복 불가 지출 782349 토큰. insufficient→answered 0.048.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 616 | 1 | 30 | 1 |
| sufficient | 0 | 17 | 0 | 1 | 1 |
| cited-unsupported | 0 | 83 | 0 | 32 | 3 |
| dropped | 0 | 2 | 0 | 1 | 0 |

### E-GR·s3

재실행 단위 788. 형식 회복 몫 0 토큰 (상한 6848). 회복 불가 지출 752951 토큰. insufficient→answered 0.065.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 606 | 0 | 39 | 3 |
| cited-unsupported | 0 | 88 | 0 | 32 | 0 |
| sufficient | 0 | 12 | 0 | 1 | 0 |
| dropped | 0 | 6 | 0 | 1 | 0 |

### L-BR·s1

재실행 단위 788. 형식 회복 몫 0 토큰 (상한 34759). 회복 불가 지출 91614 토큰. insufficient→answered 0.497.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| cited-unsupported | 0 | 89 | 0 | 405 | 38 |
| sufficient | 0 | 9 | 0 | 49 | 17 |
| insufficient | 0 | 91 | 0 | 82 | 8 |

### L-BR·s2

재실행 단위 788. 형식 회복 몫 0 토큰 (상한 22202). 회복 불가 지출 77873 토큰. insufficient→answered 0.473.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| cited-unsupported | 0 | 70 | 0 | 463 | 39 |
| sufficient | 0 | 16 | 0 | 41 | 9 |
| insufficient | 0 | 79 | 0 | 66 | 5 |

### L-BR·s3

재실행 단위 780. 형식 회복 몫 0 토큰 (상한 34258). 회복 불가 지출 77833 토큰. insufficient→answered 0.451.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| cited-unsupported | 0 | 67 | 0 | 454 | 43 |
| sufficient | 0 | 9 | 0 | 54 | 9 |
| insufficient | 0 | 79 | 0 | 60 | 5 |

### L-GR·s1

재실행 단위 788. 형식 회복 몫 0 토큰 (상한 34759). 회복 불가 지출 91614 토큰. insufficient→answered 0.497.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| cited-unsupported | 0 | 89 | 0 | 405 | 38 |
| sufficient | 0 | 9 | 0 | 49 | 17 |
| insufficient | 0 | 91 | 0 | 82 | 8 |

### L-GR·s2

재실행 단위 788. 형식 회복 몫 0 토큰 (상한 22202). 회복 불가 지출 77873 토큰. insufficient→answered 0.473.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| cited-unsupported | 0 | 70 | 0 | 463 | 39 |
| sufficient | 0 | 16 | 0 | 41 | 9 |
| insufficient | 0 | 79 | 0 | 66 | 5 |

### L-GR·s3

재실행 단위 788. 형식 회복 몫 0 토큰 (상한 34258). 회복 불가 지출 77833 토큰. insufficient→answered 0.455.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| cited-unsupported | 0 | 67 | 0 | 457 | 43 |
| sufficient | 0 | 10 | 0 | 56 | 10 |
| insufficient | 0 | 79 | 0 | 61 | 5 |

