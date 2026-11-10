# R2 drift — L-GR s1 vs Test A (PREREG §7)

자동 생성(`experiments/r2_drift.py`, SHA-256 `e79190ab2d2fa140`), HEAD `2f7f1b621f8d73ee1d24cca286dbfe381e0c79c1`, 2026-11-11T04:46:19+09:00. API 0, GPU 0, 저장값만.

같은 35토픽, branch-round 420개(35×3라운드×4branch), 라운드 상태 105개. 차이 = L-GR s1 − Test A (pp). 기준 ±10pp.

## 등록 지표

| 지표 | L-GR s1 | Test A | 차이(pp) | 판정 |
|---|---|---|---|---|
| degraded (branch-round 비율) | 36.2% | 35.5% | +0.7 | — |
| verify (최종 통과 토픽 비율, 0/35 vs 0/35) | 0.0% | 0.0% | +0.0 | — |
| coverage (branch-round 평균) | 39.1% | 42.4% | -3.3 | — |

**drift: 없음**

## 보조(판정 밖)

| 지표 | L-GR s1 | Test A | 차이(pp) | ±10pp 초과 |
|---|---|---|---|---|
| verify 라운드 통과 비율(105) | 0.0% | 0.0% | +0.0 | — |
| grounding 평균(nli_max 저장값, §7 아님) | 49.4% | 18.3% | +31.1 | **drift** |

라운드 verify 성분 평균: mean_grounding L-GR 0.494 / Test A 0.183; min_coverage L-GR 0.099 / Test A 0.144.

## 라운드별

| 라운드 | degraded L / A (pp) | coverage L / A (pp) | grounding L / A (pp) | verify 라운드 L / A |
|---|---|---|---|---|
| 0 | 32.1 / 32.9 (-0.7) | 43.8 / 45.5 (-1.8) | 54.1 / 20.5 (+33.5) | 0.0 / 0.0 (+0.0) |
| 1 | 37.9 / 36.4 (+1.4) | 38.5 / 41.7 (-3.2) | 46.5 / 17.2 (+29.3) | 0.0 / 0.0 (+0.0) |
| 2 | 38.6 / 37.1 (+1.4) | 35.1 / 40.1 (-4.9) | 47.6 / 17.2 (+30.5) | 0.0 / 0.0 (+0.0) |

## 정의

- degraded: branch-round 비율; degraded = truncated or not any(sa.text) — lfm/branch.py:311
- coverage: branch-round 평균 legacy coverage — lfm/evidence.py:39-46
- verify: 최종 verify 통과 토픽 비율; mean_g>=0.8 and min_c>=0.8 — lfm/verifier.py:38-42

## 주의

- Test A(run_id 9448e3b1, 2026-08-29 04:14:45 시작)는 v1.0 시기 코드(9ea222c 04:19 직전, b9d2afc 계열)로 돌았다. 세 지표의 정의는 b9d2afc 와 HEAD 가 글자 그대로 같다: degraded branch.py:201→311, coverage evidence.py:38-45→39-46, verify verifier.py:42→42. coverage·degraded 는 출력이 주어지면 결정론이다.
- 저장값만 썼다(재채점 없음). 양쪽 grounding 은 실행 당시 nli_max(cross-encoder/nli-deberta-v3-base, τ 0.5)로 저장됐다. M1′/R2 MiniCheck 재채점값은 쓰지 않았다. 이탈 2(fb5e6ea, 전제 고유 청크)는 nli_max 에 무관(최댓값). D15 는 Test A 를 후속 코드로 재생해 degraded 600/600 일치를 확인했다(results/d15/D15_RESULT.md:9).
- 분해는 같다(Test A 캐시 .cache/llm 재생, PREREG §4). 생성 모델 gpt-5-mini·reasoning_effort low·temperature null, prompt_mode legacy, retry_directives, max_rounds 3, τ 0.8/0.8 이 두 config 에서 같다. 그러나 L-GR s1 생성은 seed nonce(s1|L|r{round}) 아래 Test A와 다른 표본이다. 다만 L-GR s1 LLM 호출 1,007 중 903은 R2 캐시(.cache/llm_r2) 적중이고 API로 간 것은 104(라운드 0은 0)라, 대부분의 출력이 언제 표집됐는지는 캐시를 채운 시점에 달려 있다(독립 재계산 지적). 제공자 측 모델 변화 여부는 알 수 없다(model_returned 비교는 하지 않음).
- verify 는 양쪽 모두 0/35 라 차이 0pp 는 정보가 적다. 라운드 통과 비율(105 상태)도 함께 냈다.
- grounding 은 §7 지표가 아니다(보조 행, drift 판정에 넣지 않음). verify 는 mean_g 에 의존한다. grounding 은 +31pp 로 크게 다르다. 계산식은 같다(b9d2afc evidence.grounding = HEAD estimator.py NLIMaxEstimator: (청크, 문장) 쌍 최댓값 ≥ 0.5). D18 은 Test A 를 후속 코드로 재생해 grounding 이 waste_ratio.csv 와 일치함을 가드로 확인했고(experiments/d18_textdump.py:10), P0 는 Test A pilot5 재생이 60/60 바이트 일치했다(432bec7). 따라서 차이는 채점 코드가 아니라 생성 출력(다른 표본; 표집 시점·제공자 측 변화는 미상)에서 온다고 본다 — 원인은 이 스크립트가 진단하지 않는다.
- 단위(branch-round 평균, 토픽 비율)는 등록 문구가 정하지 않아 이 스크립트에서 고정했다. coverage 는 branch 비율의 평균이다 — Test A csv 에 sub-answer 개수가 없어 sub-answer 풀링 평균은 계산할 수 없다.
- Test A 50토픽 중 heldout35 의 35토픽만 비교한다(나머지 15는 제외). Test A 는 역사 기록이며 결론은 같은 시기 arm 끼리만 비교한다(PREREG §7).

검증(assert): 35토픽 ⊂ Test A 50; 양쪽 (토픽, 라운드, branch) 집합 동일·전부 실행; Test A csv 600행·degraded 215 = results.json; 양쪽 topics/csv 의 degraded·coverage·verify 가 각자 trace.jsonl 과 일치.
