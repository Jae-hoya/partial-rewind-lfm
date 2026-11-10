# R2 사후 추가 분석 (API 0원, 새 실행 0)

자동 생성(`experiments/r2_extra.py`, SHA-256 `3887504bdffadcdb`). 수치는 같은 폴더 `EXTRA_RESULT.json`. 등록 코드(`r2_analyze.py`, `r2_human.py`)는 고치지 않고 import 했다. 부트스트랩 10000회, seed 42. 실행 2026-11-10T23:40:53 (HEAD 6cbf12a9c22b90a5faa5674090567401e4468749).

## A. H4 사람 UA — 1단(실제 수행) vs 2단(PREREG §8 머리 서술)

UA 지지: minicheck_doc (r2_human.final_support). 2단 = 토픽 복원추출 → 뽑힌 토픽마다 seed 1–3 복원추출. 슬롯은 seed×관점 층에서 뽑혀 (토픽, seed) 셀이 빌 수 있고, 빈 셀은 Hájek 분자·분모에 0 을 더한다. 1단 수치는 HUMAN_RESULT.json 과 같음을 assert 했다.

| 라벨러 | 비교 | 범위 | 차이 | 1단 90% CI | 2단 90% CI | 비열등(δ 0.10) 1단 / 2단 | 비율 | 1단 비율 CI | 2단 비율 CI | 비율 ≥ 0.75 1단 / 2단 |
|---|---|---|---|---|---|---|---|---|---|---|
| colleague | E-SQ_vs_E-GR | all | +0.074 | [0.032, 0.119] | [0.016, 0.137] | True / True | 2.469 | [1.543, 4.719] | [1.255, 6.275] | True / True |
| colleague | E-SQ_vs_E-GR | excl_inferable | +0.050 | [0.008, 0.093] | [-0.002, 0.109] | True / True | 1.973 | [1.183, 3.540] | [0.970, 4.504] | True / True |
| colleague | E-R1_vs_E-GR | all | +0.025 | [-0.017, 0.064] | [-0.032, 0.080] | True / True | 1.488 | [0.744, 3.290] | [0.547, 4.422] | False / False |
| colleague | E-R1_vs_E-GR | excl_inferable | +0.008 | [-0.029, 0.044] | [-0.045, 0.058] | True / True | 1.154 | [0.503, 2.388] | [0.332, 3.083] | False / False |
| author | E-SQ_vs_E-GR | all | +0.066 | [0.028, 0.105] | [0.016, 0.119] | True / True | 2.308 | [1.487, 4.308] | [1.284, 5.649] | True / True |
| author | E-SQ_vs_E-GR | excl_inferable | +0.042 | [0.008, 0.077] | [-0.001, 0.089] | True / True | 1.812 | [1.146, 2.986] | [0.980, 3.954] | True / True |
| author | E-R1_vs_E-GR | all | +0.025 | [-0.017, 0.065] | [-0.028, 0.077] | True / True | 1.490 | [0.747, 3.272] | [0.601, 4.361] | False / False |
| author | E-R1_vs_E-GR | excl_inferable | +0.008 | [-0.029, 0.044] | [-0.040, 0.057] | True / True | 1.156 | [0.502, 2.440] | [0.400, 3.026] | False / False |

민감도 δ(2단): colleague E-SQ_vs_E-GR {'0.05': True, '0.2': True}; colleague E-R1_vs_E-GR {'0.05': True, '0.2': True}; author E-SQ_vs_E-GR {'0.05': True, '0.2': True}; author E-R1_vs_E-GR {'0.05': True, '0.2': True}

## B. 일치도 보강

| 세트 | n | 범주 | 일치율 | Cohen κ | PABAK | Gwet AC1 |
|---|---|---|---|---|---|---|
| vprime50 | 50 | 3범주 | 0.360 | 0.024 | 0.040 | 0.147 |
| vprime50 | 50 | 2범주(지지 vs 나머지) | 0.360 | 0.024 | -0.280 | -0.278 |
| anchor25 | 25 | 3범주 | 0.600 | 0.053 | 0.400 | 0.506 |
| anchor25 | 25 | 2범주(답함 vs 나머지) | 0.600 | 0.038 | 0.200 | 0.370 |
| h4_36 | 36 | 3범주 | 0.917 | 0.365 | 0.875 | 0.911 |
| h4_36 | 36 | 2범주(답함 vs 나머지) | 0.917 | 0.357 | 0.833 | 0.904 |

주변분포(저자 / 동료):
- vprime50: 지지 40 / 12, 불지지 10 / 38, 판단불가 0 / 0
- anchor25: 답함 15 / 23, 일부 8 / 2, 답안함 2 / 0
- h4_36: 답함 33 / 34, 일부 2 / 2, 답안함 1 / 0

PABAK·AC1 은 유병률 편향이 큰 표에서 κ 를 보완하는 지표다. PREREG §9 의 판정 규칙(κ ≥ 0.60)은 κ 로 등록됐고, 이 지표로 판정을 바꾸지 않는다.

## C. grounding 기권 처리 민감도

0 규약 값은 ANALYSIS.json 과 같음을 assert 했다(등록된 두 비교).

| run | 비교 | 0 규약 | 양쪽 무문장 제외(셀) | 한쪽 무문장 제외(셀) | 양쪽(토픽) | 한쪽(토픽) |
|---|---|---|---|---|---|---|
| heldout35 | E-SQ_vs_E-GR | +0.238 [0.148, 0.331] NI | +0.321 [0.206, 0.437] NI | +0.082 [0.033, 0.134] NI | +0.301 [0.180, 0.427] (24토픽) | +0.083 [0.013, 0.162] (11토픽) |
| heldout35 | E-R1_vs_E-GR | +0.137 [0.045, 0.233] NI | +0.182 [0.060, 0.306] NI | -0.059 [-0.122, 0.005] not NI | +0.169 [0.050, 0.293] (25토픽) | -0.029 [-0.100, 0.049] (11토픽) |
| heldout35 | E-SQ_vs_E-BR | +0.238 [0.148, 0.331] NI | +0.321 [0.205, 0.437] NI | +0.081 [0.033, 0.134] NI | +0.300 [0.180, 0.427] (24토픽) | +0.082 [0.011, 0.161] (11토픽) |
| heldout35 | L-BR_vs_L-GR | +0.021 [0.004, 0.041] NI | +0.023 [0.005, 0.044] NI | +0.023 [0.005, 0.044] NI | +0.025 [0.005, 0.048] (30토픽) | +0.025 [0.005, 0.048] (30토픽) |
| heldout35_nano | E-SQ_vs_E-GR@gpt-5.4-nano | +0.308 [0.179, 0.443] NI | +0.468 [0.289, 0.646] NI | +0.064 [-0.087, 0.219] not NI | +0.468 [0.289, 0.649] (23토픽) | +0.064 [-0.081, 0.210] (12토픽) |
| heldout35_nano | E-R1_vs_E-GR@gpt-5.4-nano | +0.199 [0.067, 0.334] NI | +0.331 [0.117, 0.539] NI | -0.005 [-0.138, 0.128] not NI | +0.331 [0.115, 0.544] (21토픽) | -0.005 [-0.133, 0.123] (11토픽) |

- DEVIATIONS 15 E-SQ_vs_E-GR [0.082, 0.033, 0.134] — 소수 셋째 자리 일치: drop_either/cell; 가장 가까운 변형 drop_either/cell = 0.0817, 0.0331, 0.1339 (최대 |Δ| 0.0003)
- DEVIATIONS 15 E-R1_vs_E-GR [-0.059, -0.122, 0.006] — 소수 셋째 자리 일치: 없음; 가장 가까운 변형 drop_either/cell = -0.0589, -0.1220, 0.0055 (최대 |Δ| 0.0005)
- heldout35 무문장 질의 수: E-BR·s1 17, E-BR·s2 15, E-BR·s3 18, E-GR·s1 17, E-GR·s2 15, E-GR·s3 18, E-R1·s1 10, E-R1·s2 10, E-R1·s3 9, E-SQ·s1 9, E-SQ·s2 9, E-SQ·s3 11, L-BR·s1 3, L-BR·s2 2, L-BR·s3 2, L-GR·s1 3, L-GR·s2 2, L-GR·s3 2
- heldout35_nano 무문장 질의 수: E-GR@gpt-5.4-nano·s1 23, E-R1@gpt-5.4-nano·s1 15, E-SQ@gpt-5.4-nano·s1 12

## D. 라운드 절단 비용–품질 곡선 (에뮬레이션)

**관측 궤적을 라운드 k 에서 자른 에뮬레이션이다 — 라운드 예산이 작은 별도 실행이 아니다.** 각 질의의 상태는 라운드 k 끝 상태(그 전에 멈춘 질의는 최종 상태), 토큰은 라운드 ≤ k 의 회계 토큰 누계다. 라운드 ≤ k 의 결정(FREEZE 포함)은 관측 그대로이고, 라운드 > k 의 FREEZE·재시도 결정은 잘린 궤적에 영향을 주지 않는다. 실제로 더 작은 라운드 예산으로 돌렸을 때 결정이 같았을지는 이 에뮬레이션으로 알 수 없다. k = 최대는 ANALYSIS.json 최종값을 재현함을 assert 했다(grounding 은 evidence 경로, 최대 |Δ| 3.8e-07). E-R1 = E-GR k=0 (assert). '최종 insufficient' 는 status 기준(ANALYSIS final_rates 와 같음)이라 status 가 없는 L 은 0 이다; '최종 기권(전체 규약)' 은 sa_state 규약(legacy 빈 답·무인용 → insufficient)이다.

| run | arm | k | 토큰/질의 (90% CI) | coverage | grounding (0 규약) | 최종 insufficient | 최종 기권(전체 규약) |
|---|---|---|---|---|---|---|---|
| heldout35 | E-BR | 0 | 14311 [13606, 14995] | 0.156 [0.119, 0.194] | 0.592 [0.487, 0.695] | 0.846 | 0.846 |
| heldout35 | E-BR | 1 | 32180 [31143, 33251] | 0.072 [0.051, 0.095] | 0.413 [0.309, 0.520] | 0.928 | 0.929 |
| heldout35 | E-BR | 2 | 50318 [48812, 51952] | 0.076 [0.054, 0.099] | 0.456 [0.349, 0.566] | 0.925 | 0.925 |
| heldout35 | E-GR | 0 | 14311 [13606, 14995] | 0.156 [0.119, 0.194] | 0.592 [0.487, 0.695] | 0.846 | 0.846 |
| heldout35 | E-GR | 1 | 32276 [31238, 33345] | 0.068 [0.048, 0.090] | 0.411 [0.307, 0.517] | 0.932 | 0.932 |
| heldout35 | E-GR | 2 | 50533 [49039, 52159] | 0.073 [0.053, 0.094] | 0.456 [0.349, 0.566] | 0.928 | 0.928 |
| heldout35 | E-SQ | 0 | 14311 [13606, 14995] | 0.156 [0.119, 0.194] | 0.592 [0.487, 0.695] | 0.846 | 0.846 |
| heldout35 | E-SQ | 1 | 31797 [30772, 32851] | 0.143 [0.109, 0.178] | 0.657 [0.537, 0.771] | 0.859 | 0.860 |
| heldout35 | E-SQ | 2 | 49095 [47776, 50498] | 0.161 [0.121, 0.202] | 0.694 [0.579, 0.805] | 0.842 | 0.842 |
| heldout35 | L-BR | 0 | 15581 [14753, 16388] | 0.432 [0.364, 0.502] | 0.746 [0.675, 0.811] | 0.000 | 0.575 |
| heldout35 | L-BR | 1 | 33808 [32347, 35353] | 0.420 [0.352, 0.491] | 0.741 [0.671, 0.807] | 0.000 | 0.587 |
| heldout35 | L-BR | 2 | 51198 [48898, 53588] | 0.436 [0.367, 0.503] | 0.749 [0.680, 0.813] | 0.000 | 0.572 |
| heldout35 | L-GR | 0 | 15581 [14753, 16388] | 0.432 [0.364, 0.502] | 0.746 [0.675, 0.811] | 0.000 | 0.575 |
| heldout35 | L-GR | 1 | 35883 [34296, 37543] | 0.400 [0.335, 0.465] | 0.731 [0.660, 0.797] | 0.000 | 0.607 |
| heldout35 | L-GR | 2 | 55997 [53691, 58433] | 0.397 [0.334, 0.460] | 0.728 [0.659, 0.791] | 0.000 | 0.609 |
| heldout35_nano | E-GR@gpt-5.4-nano | 0 | 12613 [12045, 13151] | 0.070 [0.047, 0.095] | 0.463 [0.330, 0.597] | 0.909 | 0.932 |
| heldout35_nano | E-GR@gpt-5.4-nano | 1 | 28570 [27893, 29252] | 0.042 [0.025, 0.061] | 0.233 [0.129, 0.348] | 0.954 | 0.959 |
| heldout35_nano | E-GR@gpt-5.4-nano | 2 | 44708 [43821, 45623] | 0.041 [0.024, 0.060] | 0.264 [0.157, 0.386] | 0.950 | 0.959 |
| heldout35_nano | E-SQ@gpt-5.4-nano | 0 | 12613 [12045, 13151] | 0.070 [0.047, 0.095] | 0.463 [0.330, 0.597] | 0.909 | 0.932 |
| heldout35_nano | E-SQ@gpt-5.4-nano | 1 | 28314 [27609, 29011] | 0.077 [0.056, 0.099] | 0.559 [0.429, 0.692] | 0.921 | 0.925 |
| heldout35_nano | E-SQ@gpt-5.4-nano | 2 | 44021 [43156, 44931] | 0.086 [0.063, 0.111] | 0.572 [0.439, 0.705] | 0.904 | 0.916 |

## assert

- final_support n_supported 33 및 문장별 판정이 두 HUMAN_RESULT 와 같다
- colleague E-SQ_vs_E-GR 1단 결과 == HUMAN_RESULT
- colleague E-R1_vs_E-GR 1단 결과 == HUMAN_RESULT
- author E-SQ_vs_E-GR 1단 결과 == HUMAN_RESULT
- author E-R1_vs_E-GR 1단 결과 == HUMAN_RESULT
- vprime50 κ3 0.0244 κ2 0.0244 일치율 == HUMAN_RESULT
- anchor25 κ3 0.0530 κ2 0.0385 일치율 == HUMAN_RESULT
- heldout35 all_topics E-SQ_vs_E-GR 0 규약 == ANALYSIS.json
- heldout35 sensitivity_30 E-SQ_vs_E-GR 0 규약 == ANALYSIS.json
- heldout35 all_topics E-R1_vs_E-GR 0 규약 == ANALYSIS.json
- heldout35 sensitivity_30 E-R1_vs_E-GR 0 규약 == ANALYSIS.json
- heldout35_nano all_topics E-SQ_vs_E-GR@gpt-5.4-nano 0 규약 == ANALYSIS.json
- heldout35_nano sensitivity_30 E-SQ_vs_E-GR@gpt-5.4-nano 0 규약 == ANALYSIS.json
- heldout35_nano all_topics E-R1_vs_E-GR@gpt-5.4-nano 0 규약 == ANALYSIS.json
- heldout35_nano sensitivity_30 E-R1_vs_E-GR@gpt-5.4-nano 0 규약 == ANALYSIS.json
- heldout35 E-BR k=2 == ANALYSIS 최종값(토큰·coverage 정확, grounding |Δ|<1e-6, final_rates 정확)
- heldout35 E-GR k=2 == ANALYSIS 최종값(토큰·coverage 정확, grounding |Δ|<1e-6, final_rates 정확)
- heldout35 E-SQ k=2 == ANALYSIS 최종값(토큰·coverage 정확, grounding |Δ|<1e-6, final_rates 정확)
- heldout35 L-BR k=2 == ANALYSIS 최종값(토큰·coverage 정확, grounding |Δ|<1e-6, final_rates 정확)
- heldout35 L-GR k=2 == ANALYSIS 최종값(토큰·coverage 정확, grounding |Δ|<1e-6, final_rates 정확)
- heldout35 E-GR k=0 == ANALYSIS E-R1
- heldout35_nano E-GR@gpt-5.4-nano k=2 == ANALYSIS 최종값(토큰·coverage 정확, grounding |Δ|<1e-6, final_rates 정확)
- heldout35_nano E-SQ@gpt-5.4-nano k=2 == ANALYSIS 최종값(토큰·coverage 정확, grounding |Δ|<1e-6, final_rates 정확)
- heldout35_nano E-GR@gpt-5.4-nano k=0 == ANALYSIS E-R1@gpt-5.4-nano
