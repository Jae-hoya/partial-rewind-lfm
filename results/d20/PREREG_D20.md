# PREREG D20 — 모순 통제 쌍의 다중 체크포인트 오프라인 재채점

작성 2026-11-05. **채점 전에 작성한다.** 이 파일의 SHA-256 은 실행 스크립트가 시작 시 계산해
`results/d20/d20_summary.json` 에 기록한다. 실행 후 이 파일은 수정하지 않는다.

대응 심사 의견: R1-5 ("같은 답안/sub-question 의 claim 이 반드시 양립하는 것은 아니므로 26.4% 를
위양성률로 읽을 수 없다; '어떤 임계/집계도 적합한 트리거를 주지 않는다'는 주장을 시험한 구성으로 한정하라"),
R2 ("모순 검출 결론이 단일 NLI 체크포인트에 기대고 있다"). 출처: `reviews_Access-2026-47134.md`.

## 1. 자료 (새 생성 없음, API 호출 0)

- **N1** (같은 branch·같은 sub-question claim 쌍, 원래 "음성"으로 가정된 통제):
  `experiments/m2_gate_n1.py` 의 묶음 생성 루프(decompose → run_branch → attribute_claims →
  combinations)를 **`llm.cache_policy = cache_only`** 로 재생해 쌍 텍스트를 복원한다.
  `.env` 는 읽지 않는다. 캐시 미스(`CacheMiss`)는 잡지 않고 실행을 중단시킨다.
  조건 `C3_partial_heldout`, 토픽 15개(원본과 같은 `pick_topics`, 같은 seed), branch 당 claim 상한 12.
  복원한 쌍 텍스트는 `results/d20/n1_pairs.csv` 로 저장한다.
- **P1** (claim 과 그 부정문, 양성 통제) / **P2** (claim 과 그 paraphrase):
  `results/m2/relation/gate_p1p2.csv` 의 텍스트를 그대로 읽는다. 원본과 같은 사용 가능 필터
  (`neg_usable`, `par_usable`). 원본 기록상 P1 n=46, P2 n=50 (`gate_p1p2.json`).
- P1 부정문은 `gpt-5.4-nano` 가 만든 기계적 부정("not" 삽입 등)이므로 **P1 재현율은 쉬운 상한**이다.
  실제 관점 간 모순은 이보다 미묘하다.
- P2 는 이 실험에서 **모순 트리거의 추가 음성**으로도 쓴다(paraphrase 쌍의 contradiction 양성률).

## 2. 재현 검증 (선행 조건)

새 체크포인트 채점 전에 `facebook/bart-large-mnli` 로 복원 N1 쌍을 다시 채점해
`results/m2/relation/gate_n1.csv` 와 `(topic, branch, subq, a, b)` 로 조인한다.
요구: 쌍 수 823, 키 전부 일치, contradiction 최대 절대차 ≤ 1e-3, `> 0.70` 비율이 26.367% 재현.
P1 도 bart 로 `gate_p1p2.csv` 의 `p1_contradiction` 과 대조한다.
**하나라도 실패하면 이후 결과를 "같은 쌍"의 결과로 보고하지 않고 실패를 보고한다.**

## 3. 체크포인트 (전부 로컬, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`)

3-way NLI (트리거 판정 대상):
1. `facebook/bart-large-mnli` (원본)
2. `roberta-large-mnli`
3. `cross-encoder/nli-deberta-v3-large`
4. `cross-encoder/nli-deberta-v3-base`

라벨 순서는 `lfm.nli.NLI` 가 `config.id2label` 에서 읽는다. max_length 512, 배치 16.

보조 (기술 통계만, **트리거 표에 넣지 않는다**):
5. `lytang/MiniCheck-Flan-T5-Large` — P(doc 가 claim 을 지지)만 낸다. 지지 안 됨 ≠ 모순
   (중립일 수 있다)이므로 모순 트리거가 될 수 없다. 양방향 support 의 min/max 를 N1/P1/P2 에 대해 보고만 한다.

## 4. 집계와 임계

쌍 (a, b) 마다 두 방향 (a→b), (b→a) 의 contradiction 확률 c1, c2.
- **주 집계: 양방향 max** (원본과 동일).
- 부 집계: 양방향 min, 양방향 mean. (사전에 이 셋으로 한정한다.)

임계: 고정 **{0.70, 0.95}** (엄격 부등호 `>`, 원본과 동일) + 스윕 0.50–0.99 (0.01 간격).

## 5. 판정 정의

- (체크포인트, 집계, 임계) 한 구성의 **P1 재현율** = P1 쌍 중 점수 > 임계 비율,
  **N1 양성률** = N1 쌍 중 점수 > 임계 비율, **P2 양성률** = P2 쌍 중 점수 > 임계 비율.
- **"쓸 만한 트리거"** = 같은 구성에서 **P1 재현율 ≥ 80% 이고 N1 양성률 ≤ 5%** (원본 게이트 기준을 그대로 가져온다).
- **"결론이 일반화된다"** = 시험한 4 체크포인트 × 3 집계 × 스윕 임계 중 위 조건을 만족하는 구성이 **하나도 없다**.
  하나라도 있으면 "단일 체크포인트 결론은 일반화되지 않는다"로 보고한다.
- 부 보고: 체크포인트·집계마다 P1 ≥ 80% 를 만족하는 임계 중 N1 양성률이 최소인 지점(실패해도 보고).

## 6. 해석 제약 (R1-5)

N1 쌍은 "같은 답안에서 나왔으니 양립"이라고 **가정**했을 뿐 라벨이 없다. 따라서 N1 양성률은
**양성률(positive rate)이지 위양성률이 아니다.** 이 실험은 어떤 라벨도 검증하지 않는다.
결론은 위 구성(4 체크포인트, 3 집계, 이 임계 격자, N1 823 쌍·P1 46·P2 50, 15 토픽, C3_partial_heldout)으로 한정한다.

## 7. 탐색 항목 (사전 명시)

- 체크포인트 간 N1 플래그 일치: 임계 0.95·주 집계에서 4 개 3-way 체크포인트가 모두 플래그한 쌍 수,
  체크포인트 쌍별 Jaccard. 전원 일치 쌍의 텍스트는 `results/d20/n1_consensus_flagged.csv` 로 저장한다
  (사용자가 직접 검토할 자료이며 이 실험에서 라벨을 붙이지 않는다).
  일치가 강하면 N1 안에 실제 모순이 섞였을 가능성(R1-5)을, 약하면 체크포인트별 잡음을 시사한다 — 둘 다 가설일 뿐이다.

## 8. 출력

`experiments/d20_contradiction_rescore.py` →
`results/d20/n1_pairs.csv`, `results/d20/d20_scores.csv` (쌍 × 체크포인트 × 방향),
`results/d20/d20_summary.json`, `results/d20/n1_consensus_flagged.csv`. 해석은 `results/d20/D20_RESULT.md`.
