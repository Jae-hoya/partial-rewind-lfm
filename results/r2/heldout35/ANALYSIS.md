# R2 분석 요약 — `results\r2\heldout35`

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
| E-BR·s1 | 35 | 1773141 | 1274678 | 620 | 418 | 1744 | 0 (0 / 0) | 1 | 0.901 / 0.000 / 0.001 |
| E-BR·s2 | 35 | 1749148 | 1245979 | 608 | 418 | 1744 | 0 (0 / 0) | 1 | 0.901 / 0.000 / 0.001 |
| E-BR·s3 | 35 | 1761077 | 1260090 | 614 | 420 | 1752 | 0 (0 / 0) | 0 | 0.905 / 0.000 / 0.000 |
| E-GR·s1 | 35 | 1785108 | 11967 | 6 | 420 | 1752 | 0 (0 / 0) | 0 | 0.900 / 0.000 / 0.001 |
| E-GR·s2 | 35 | 1759781 | 10633 | 6 | 420 | 1752 | 0 (0 / 0) | 0 | 0.900 / 0.000 / 0.001 |
| E-GR·s3 | 35 | 1761077 | 0 | 0 | 420 | 1752 | 0 (0 / 0) | 0 | 0.905 / 0.000 / 0.000 |
| E-R1·s1 | 35 | 498463 | 0 | 0 | 140 | 584 | 0 (2 / 0) | 0 | 0.856 / 0.000 / 0.000 |
| E-R1·s2 | 35 | 503169 | 0 | 0 | 140 | 584 | 0 (2 / 0) | 0 | 0.837 / 0.000 / 0.002 |
| E-R1·s3 | 35 | 500987 | 0 | 0 | 140 | 584 | 0 (2 / 0) | 0 | 0.844 / 0.000 / 0.000 |
| E-SQ·s1 | 35 | 1711162 | 1711162 | 795 | 418 | 1619 | 0 (4 / 0) | 1 | 0.917 / 0.000 / 0.000 |
| E-SQ·s2 | 35 | 1713854 | 1713854 | 796 | 418 | 1620 | 0 (3 / 0) | 1 | 0.917 / 0.000 / 0.001 |
| E-SQ·s3 | 35 | 1729917 | 1729917 | 799 | 420 | 1627 | 0 (3 / 0) | 0 | 0.916 / 0.001 / 0.000 |
| L-BR·s1 | 35 | 1757774 | 1690857 | 842 | 384 | 1608 | 1 (6 / 1) | 18 | 0.000 / 0.000 / 0.000 |
| L-BR·s2 | 35 | 1794292 | 1794292 | 888 | 390 | 1633 | 0 (5 / 0) | 18 | 0.000 / 0.000 / 0.000 |
| L-BR·s3 | 35 | 1823750 | 1823750 | 901 | 393 | 1647 | 1 (6 / 1) | 16 | 0.000 / 0.000 / 0.000 |
| L-GR·s1 | 35 | 1955504 | 197730 | 104 | 420 | 1752 | 0 (4 / 0) | 0 | 0.000 / 0.000 / 0.000 |
| L-GR·s2 | 35 | 1935924 | 153876 | 78 | 416 | 1734 | 1 (5 / 1) | 0 | 0.000 / 0.000 / 0.000 |
| L-GR·s3 | 35 | 1988244 | 164494 | 81 | 420 | 1752 | 0 (2 / 0) | 0 | 0.000 / 0.000 / 0.000 |

## 절감 — 전체 토픽

절감 = 1 − ΣA/ΣB. 주장 조건: seed 부호 일치 ∧ 95% CI 가 0 제외.

| 비교 | 지표 | 점추정 | 95% CI | seed별 | 주장 |
|---|---|---|---|---|---|
| E-SQ_vs_E-GR | accounting_tokens | 0.028 | [0.014, 0.044] | s1 0.041, s2 0.026, s3 0.018 | 예 |
| E-SQ_vs_E-GR | billed_tokens | -227.094 | [nan, -62.280] | s1 -141.990, s2 -160.183, s3 — | 보고만 |
| E-SQ_vs_E-GR | branch_rounds | 0.003 | [0.000, 0.011] | s1 0.005, s2 0.005, s3 0.000 | 아니오 |
| E-SQ_vs_E-GR | subq_executed | 0.074 | [0.052, 0.098] | s1 0.076, s2 0.075, s3 0.071 | 예 |
| L-BR_vs_L-GR | accounting_tokens | 0.086 | [0.049, 0.124] | s1 0.101, s2 0.073, s3 0.083 | 예 |
| L-BR_vs_L-GR | billed_tokens | -9.287 | [-17.402, -5.787] | s1 -7.551, s2 -10.661, s3 -10.087 | 보고만 |
| L-BR_vs_L-GR | branch_rounds | 0.071 | [0.040, 0.104] | s1 0.086, s2 0.062, s3 0.064 | 예 |
| L-BR_vs_L-GR | subq_executed | 0.067 | [0.038, 0.099] | s1 0.082, s2 0.058, s3 0.060 | 예 |
| E-BR_vs_E-SQ | accounting_tokens | -0.025 | [-0.040, -0.011] | s1 -0.036, s2 -0.021, s3 -0.018 | 예 |
| E-BR_vs_E-SQ | billed_tokens | 0.267 | [0.249, 0.282] | s1 0.255, s2 0.273, s3 0.272 | 보고만 |
| E-BR_vs_E-SQ | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000, s2 0.000, s3 0.000 | 아니오 |
| E-BR_vs_E-SQ | subq_executed | -0.077 | [-0.104, -0.052] | s1 -0.077, s2 -0.077, s3 -0.077 | 예 |

## 자동 비열등성 — 전체 토픽

차이 = A − B (질의 단위 평균). 90% CI 하한 > −δ.

| 비교 | 지표 | 차이 | 90% CI | δ | 비열등 | 동등 |
|---|---|---|---|---|---|---|
| E-SQ_vs_E-GR | coverage | 0.088 | [0.061, 0.118] | 0.05 | 예 | 아니오 |
| E-SQ_vs_E-GR | grounding | 0.238 | [0.148, 0.331] | 0.05 | 예 | 아니오 |
| E-R1_vs_E-GR | coverage | 0.083 | [0.057, 0.111] | 0.05 | 예 | 아니오 |
| E-R1_vs_E-GR | grounding | 0.137 | [0.045, 0.233] | 0.05 | 예 | 아니오 |

## 절감 — 30토픽 민감도 (t000~t004 제외)

절감 = 1 − ΣA/ΣB. 주장 조건: seed 부호 일치 ∧ 95% CI 가 0 제외.

| 비교 | 지표 | 점추정 | 95% CI | seed별 | 주장 |
|---|---|---|---|---|---|
| E-SQ_vs_E-GR | accounting_tokens | 0.027 | [0.011, 0.045] | s1 0.044, s2 0.023, s3 0.014 | 예 |
| E-SQ_vs_E-GR | billed_tokens | -192.882 | [nan, -52.956] | s1 -120.336, s2 -136.141, s3 — | 보고만 |
| E-SQ_vs_E-GR | branch_rounds | 0.004 | [0.000, 0.013] | s1 0.006, s2 0.006, s3 0.000 | 아니오 |
| E-SQ_vs_E-GR | subq_executed | 0.065 | [0.042, 0.089] | s1 0.069, s2 0.066, s3 0.059 | 예 |
| L-BR_vs_L-GR | accounting_tokens | 0.090 | [0.049, 0.133] | s1 0.111, s2 0.070, s3 0.091 | 예 |
| L-BR_vs_L-GR | billed_tokens | -8.654 | [-16.992, -5.269] | s1 -6.672, s2 -11.070, s3 -9.034 | 보고만 |
| L-BR_vs_L-GR | branch_rounds | 0.073 | [0.040, 0.109] | s1 0.092, s2 0.059, s3 0.069 | 예 |
| L-BR_vs_L-GR | subq_executed | 0.069 | [0.037, 0.105] | s1 0.088, s2 0.055, s3 0.065 | 예 |
| E-BR_vs_E-SQ | accounting_tokens | -0.023 | [-0.039, -0.007] | s1 -0.038, s2 -0.016, s3 -0.014 | 예 |
| E-BR_vs_E-SQ | billed_tokens | 0.265 | [0.247, 0.283] | s1 0.251, s2 0.274, s3 0.272 | 보고만 |
| E-BR_vs_E-SQ | branch_rounds | 0.000 | [0.000, 0.000] | s1 0.000, s2 0.000, s3 0.000 | 아니오 |
| E-BR_vs_E-SQ | subq_executed | -0.065 | [-0.093, -0.041] | s1 -0.068, s2 -0.065, s3 -0.063 | 예 |

## 자동 비열등성 — 30토픽 민감도 (t000~t004 제외)

차이 = A − B (질의 단위 평균). 90% CI 하한 > −δ.

| 비교 | 지표 | 차이 | 90% CI | δ | 비열등 | 동등 |
|---|---|---|---|---|---|---|
| E-SQ_vs_E-GR | coverage | 0.075 | [0.047, 0.105] | 0.05 | 예 | 아니오 |
| E-SQ_vs_E-GR | grounding | 0.230 | [0.128, 0.339] | 0.05 | 예 | 아니오 |
| E-R1_vs_E-GR | coverage | 0.072 | [0.046, 0.101] | 0.05 | 예 | 아니오 |
| E-R1_vs_E-GR | grounding | 0.130 | [0.025, 0.240] | 0.05 | 예 | 아니오 |

## R1-8 상태 전이 — 회계 분해 (인과 아님)

### E-BR·s1

재실행 단위 1160. 형식 회복 몫 0 토큰 (상한 11267). 회복 불가 지출 1364649 토큰. insufficient→answered 0.041.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 994 | 0 | 20 | 23 |
| sufficient | 0 | 50 | 1 | 6 | 20 |
| cited-unsupported | 0 | 26 | 0 | 11 | 8 |
| dropped | 0 | 1 | 0 | 0 | 0 |

### E-BR·s2

재실행 단위 1160. 형식 회복 몫 0 토큰 (상한 1244). 회복 불가 지출 1350024 토큰. insufficient→answered 0.038.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 993 | 1 | 16 | 23 |
| cited-unsupported | 0 | 35 | 0 | 4 | 7 |
| sufficient | 0 | 53 | 0 | 4 | 23 |
| dropped | 0 | 1 | 0 | 0 | 0 |

### E-BR·s3

재실행 단위 1168. 형식 회복 몫 0 토큰 (상한 7381). 회복 불가 지출 1366486 토큰. insufficient→answered 0.027.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 1010 | 0 | 10 | 18 |
| sufficient | 0 | 53 | 0 | 6 | 21 |
| cited-unsupported | 0 | 30 | 0 | 11 | 9 |

### E-GR·s1

재실행 단위 1168. 형식 회복 몫 0 토큰 (상한 11267). 회복 불가 지출 1364649 토큰. insufficient→answered 0.042.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 996 | 0 | 20 | 24 |
| sufficient | 0 | 52 | 1 | 7 | 20 |
| cited-unsupported | 0 | 27 | 0 | 12 | 8 |
| dropped | 0 | 1 | 0 | 0 | 0 |

### E-GR·s2

재실행 단위 1168. 형식 회복 몫 0 토큰 (상한 1244). 회복 불가 지출 1350024 토큰. insufficient→answered 0.038.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 996 | 1 | 16 | 23 |
| cited-unsupported | 0 | 35 | 0 | 4 | 7 |
| sufficient | 0 | 56 | 0 | 4 | 25 |
| dropped | 0 | 1 | 0 | 0 | 0 |

### E-GR·s3

재실행 단위 1168. 형식 회복 몫 0 토큰 (상한 7381). 회복 불가 지출 1366486 토큰. insufficient→answered 0.027.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 1010 | 0 | 10 | 18 |
| sufficient | 0 | 53 | 0 | 6 | 21 |
| cited-unsupported | 0 | 30 | 0 | 11 | 9 |

### E-SQ·s1

재실행 단위 1035. 형식 회복 몫 0 토큰 (상한 0). 회복 불가 지출 1428364 토큰. insufficient→answered 0.037.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 962 | 0 | 12 | 25 |
| cited-unsupported | 0 | 23 | 0 | 10 | 3 |

### E-SQ·s2

재실행 단위 1036. 형식 회복 몫 0 토큰 (상한 7667). 회복 불가 지출 1436781 토큰. insufficient→answered 0.025.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 967 | 0 | 8 | 17 |
| cited-unsupported | 0 | 29 | 0 | 9 | 5 |
| dropped | 0 | 1 | 0 | 0 | 0 |

### E-SQ·s3

재실행 단위 1043. 형식 회복 몫 0 토큰 (상한 4607). 회복 불가 지출 1453234 토큰. insufficient→answered 0.025.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 1 | 970 | 0 | 8 | 17 |
| cited-unsupported | 0 | 27 | 0 | 12 | 7 |
| format_fail | 0 | 1 | 0 | 0 | 0 |

### L-BR·s1

재실행 단위 1024. 형식 회복 몫 0 토큰 (상한 12714). 회복 불가 지출 698107 토큰. insufficient→answered 0.158.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 570 | 0 | 46 | 61 |
| sufficient | 0 | 80 | 0 | 49 | 77 |
| cited-unsupported | 0 | 48 | 0 | 52 | 41 |

### L-BR·s2

재실행 단위 1049. 형식 회복 몫 0 토큰 (상한 10559). 회복 불가 지출 690344 토큰. insufficient→answered 0.176.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 549 | 0 | 36 | 81 |
| sufficient | 0 | 74 | 0 | 52 | 105 |
| cited-unsupported | 0 | 33 | 0 | 70 | 49 |

### L-BR·s3

재실행 단위 1063. 형식 회복 몫 0 토큰 (상한 21928). 회복 불가 지출 641988 토큰. insufficient→answered 0.197.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 557 | 0 | 48 | 89 |
| sufficient | 0 | 79 | 0 | 38 | 102 |
| cited-unsupported | 0 | 42 | 0 | 54 | 54 |

### L-GR·s1

재실행 단위 1168. 형식 회복 몫 0 토큰 (상한 12714). 회복 불가 지출 698107 토큰. insufficient→answered 0.168.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 579 | 0 | 52 | 65 |
| sufficient | 0 | 116 | 0 | 66 | 126 |
| cited-unsupported | 0 | 53 | 0 | 57 | 54 |

### L-GR·s2

재실행 단위 1150. 형식 회복 몫 0 토큰 (상한 10559). 회복 불가 지출 689164 토큰. insufficient→answered 0.178.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 553 | 0 | 36 | 84 |
| sufficient | 0 | 100 | 0 | 63 | 137 |
| cited-unsupported | 0 | 34 | 0 | 79 | 64 |

### L-GR·s3

재실행 단위 1168. 형식 회복 몫 0 토큰 (상한 21928). 회복 불가 지출 641988 토큰. insufficient→answered 0.201.

| r → r+1 | format_fail | insufficient | dropped | cited-unsupported | sufficient |
|---|---|---|---|---|---|
| insufficient | 0 | 559 | 0 | 50 | 91 |
| sufficient | 0 | 89 | 0 | 65 | 152 |
| cited-unsupported | 0 | 45 | 0 | 58 | 59 |

