# PREREG D21 — 기존 파일 재분석 두 건 (R1-5a 단위 재계수, R3 빈 branch 처리)

작성 2026-11-05. **계산 전에 작성한다.** 실행 스크립트 `experiments/d21_reanalysis.py` 가 시작 시
이 파일의 SHA-256 을 계산해 `results/d21/d21_summary.json` 에 기록한다. 실행 후 이 파일은 수정하지 않는다.
API/LLM 호출 0, 캐시 재생도 하지 않는다 (아래 파일만 읽는다). 추적 파일은 수정하지 않는다.

대응 심사 의견 (출처 `심사_2026-11-01/reviews_Access-2026-47134.md`):
- R1-5a: N1 26.4% 를 pair 단위로만 세었다 — 단위에 따라 의미가 달라진다.
- R3: "A third of branch executions uncited (all 215 had chunks). Contaminates coverage, grounding, sufficient population."

공통 bootstrap: 군집 복원추출, B = 2000, `numpy.random.default_rng(42)`, percentile 95% CI.
비율/평균은 ratio-of-sums (군집별 분자합 / 분모합), 개수는 군집별 개수의 합.

---

## A. R1-5a — N1 양성률의 단위 재계수

### 자료
- `results/d20/d20_scores.csv` (set=N1, checkpoint × direction ab/ba 의 contradiction 확률)
- `results/d20/n1_pairs.csv` (쌍 키·텍스트, 823 쌍)
- `results/m2/relation/gate_n1.csv` (원본 bart 양방향 max, 재현 대조용)

### 선행 재현 (실패 시 이후 수치를 "같은 쌍"의 결과로 보고하지 않는다)
`facebook/bart-large-mnli` 의 ab/ba 에서 양방향 max 를 다시 만들어 `gate_n1.csv` 의 `contradiction` 과
`(topic, branch, subq, a, b)` 로 조인: 키 823 개 전부 일치, 최대 절대차 ≤ 1e-3,
pair 단위 `> 0.70` 비율 = 217/823 = 26.367% (D20 기록과 동일).

### 단위 정의 (판정은 원본과 같이 엄격 부등호 `score > τ`, 점수 = 양방향 max)
- **pair**: `(topic, branch, subq, a, b)` 한 쌍. 양성 = score > τ.
- **claim**: `(topic, branch, subq, idx)` — idx 는 묶음 안의 claim 위치(a 또는 b 값).
  `m2_gate_n1.py` 에서 claim 은 하나의 sub-question 에만 귀속되므로 이 키는 claim 하나를 가리킨다.
  양성 = 그 claim 이 들어간 쌍 중 하나라도 양성. 분모 = `n1_pairs.csv` 에 나타나는 서로 다른 claim
  (= claim 2 개 이상인 묶음의 claim). **단독 claim 묶음의 claim 은 파일에 없으므로 분모에서 빠진다.**
- **bundle**: `(topic, branch, subq)` 묶음. 양성 = 묶음 안의 쌍 중 하나라도 양성. 묶음 수 114 를 확인한다.

### 임계
주: τ = 0.70 (원 논문). 부: τ = 0.95.

### 군집
topic (15 개). 군집이 15 개뿐이므로 CI 가 넓을 것이다 — 그대로 보고한다.

### 부 보고
- 다른 3-way 체크포인트(roberta-large-mnli, nli-deberta-v3-large, nli-deberta-v3-base)의 max 집계 같은 표.
  MiniCheck 는 모순 트리거가 아니므로 제외 (PREREG_D20 §3).
- 묶음 크기(claim 수) 분포. claim·bundle 단위 양성률은 묶음이 클수록 기계적으로 오른다
  (쌍이 많을수록 "하나라도" 확률이 커진다) — 이 점을 함께 적는다.
- 해석 제약: N1 은 라벨이 없다. 어느 단위의 수치도 **양성률이지 위양성률이 아니다** (PREREG_D20 §6).

---

## B. R3 — 빈(degraded) branch 처리에 따른 지표 민감도

### 자료
- 실행 1: `results/test_a/20260829-041445/waste_ratio.csv` — `results.json` context 기록
  `estimator` 미기재(기본 nli_max), `nli_model = cross-encoder/nli-deberta-v3-base` (원고의 g₁).
- 실행 2: `results/m1prime/seed42/waste_ratio.csv` — `results.json` context `estimator = minicheck_doc`.
- 두 파일 모두 600 행 = 50 query × 3 round × 4 branch. 키 `(query_id, round, branch)`.

### 집합 정의
- **D215** = `degraded == 1` (모든 sub-answer 텍스트가 비었거나 절단; `lfm/branch.py` 311행).
- **Z219** = `coverage == 0`. D215 ⊂ Z219 이고 나머지 4 개는 텍스트가 있으나 유효 인용이 없는 branch
  (답은 했으나 미인용). 이 4 개는 기권이 아니다.

### 지표 (기존 정의 그대로)
- coverage 평균, grounding 평균: 해당 처리에서 "정의된" branch-round 행에 대한 평균 (600 행 전체 / round 0 만 둘 다 보고).
- verify 통과: query-round (150 개) 마다 `mean_g ≥ 0.8 AND min_c ≥ 0.8` (`lfm/verifier.py`, `≥`).
  mean_g 는 그 round 의 정의된 grounding 의 평균, min_c 는 정의된 coverage 의 최솟값.
  **정의된 branch 가 하나도 없는 query-round 는 불통과**로 세고, 그런 round 수를 따로 보고한다.
  통과 수와 함께 grounding 절 단독·coverage 절 단독 통과 수도 보고한다.
- sufficient@0.8 개수: `grounding > 0.8 AND coverage > 0.8` (`lfm/evidence.py sufficient`, 엄격 `>`), 600 행 중.
  정의되지 않은 값이 있는 branch 는 sufficient 가 아니다.
- **W0.8** = `test_a.py waste_by_query` 와 같은 정의: re_executed == 1 인 행 중 sufficient@0.8 비율.

### 처리
- **(a) as is**: 기록된 값 그대로 (빈 branch 는 coverage 0, grounding 0 — grounding 0 은
  `evidence.py grounding()` 이 문장이 없을 때 돌려주는 관례값이지 측정값이 아니다).
- **(b) excluded**: 해당 집합의 branch-round 행을 모든 지표의 모집단에서 제거한다
  (평균, verify 의 mean_g/min_c, sufficient 모집단, W 의 분자·분모). 두 변형: **(b-215)** D215 제거, **(b-219)** Z219 제거.
- **(c) abstention (주)**: D215 를 기권으로 본다. 기권 branch 의 coverage 는 분모(답한 sub-question)가 0 이 되어
  **정의되지 않으므로** coverage 평균과 min_c 에서만 뺀다. grounding 은 기록값(0)을 그대로 쓴다
  (과제 문언 "coverage denominators only"). 기권 branch 는 재실행된 branch 로 W 분모에 남고 sufficient 가 아니다.
- **(c′) abstention, grounding 도 미정의**: (c) 에 더해 D215 의 grounding 도 0/0 으로 보아 grounding 평균과 mean_g 에서 뺀다.
  W 분모에는 남는다.
- (c)/(c′) 는 D215 에만 적용한다 (Z219 의 추가 4 개는 기권이 아니므로).

### 사전에 알 수 있는 구조 (확인 대상)
D215 는 coverage 0, grounding 0 이므로 (a) 에서 이미 sufficient 가 아니다. 따라서 sufficient 개수와
W 의 분자는 모든 처리에서 같고, W 는 (b) 에서 분모만 줄어 커지며 (c)/(c′) 에서는 (a) 와 같다.
coverage·grounding 평균과 verify 는 처리에 따라 달라질 수 있다.

### 군집 bootstrap
query (50 개). 평균·W 는 ratio-of-sums, 통과·sufficient 는 개수 합.

### 고정점 확인 (하나라도 어긋나면 결과 문서 맨 앞에 표시)
- 실행 1 (a): W0.8 = 2/400 = 0.5%, sufficient@0.8 = 3/600, verify 통과 0/150.
- 실행 2 (a): W0.8 = 34/400 = 8.5%.
- round 0 의 D215 비율 = 67/200 = 33.5%, round 0 coverage 평균 0.474 (원고 수치).

### 한계 (사전 명시)
sub-question 단위 기권(비어 있지 않은 branch 안의 빈 sub-answer 를 coverage 분모에서 빼기)은
`waste_ratio.csv`·`results.json` 에 sub-answer 텍스트가 없어 이 재분석에서 계산하지 않는다.

---

## 출력
`experiments/d21_reanalysis.py` → `results/d21/d21_summary.json`, `results/d21/D21_RESULT.md` (한국어).
