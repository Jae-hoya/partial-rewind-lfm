# D18 등록 수정 1 — θ 규칙과 연산자 입력 확정

2026-11-03. **연산자 점수를 하나도 계산하기 전에 커밋한다.**
이미 본 것은 PREREG_D18 목록에 더해 H2 사람 라벨 집계(`H2_HUMAN_RESULT.md`, `VPRIME_FINAL.md`)다. 연산자 점수는 아직 아무것도 보지 않았다.

## 왜 고치나

SPEC_R2 §2의 θ 규칙("precision ≥ 0.85 범위에서 cov_A가 가장 낮아지는 값")은 퇴화한다.
precision은 보통 θ가 오를수록 오르므로, 이 규칙은 양성이 한두 개만 남는 최댓값 근처의 θ를 고르게 된다. 그러면 cov_A가 0에 가깝게 된다.
원래 규칙("가장 낮은 θ")은 반대로 결과에 유리한 쪽이었다.
그래서 **결과 방향과 무관한 기준으로 고정한다.**

## 확정 규칙

**이진 목표 (사람 라벨)**
- A (질문-답 관련성): 양성 = `답함`. `일부`와 `답안함`은 음성. 민감도 분석에서는 `일부`도 양성.
- B (질문-컨텍스트 답 가능성): 양성 = `답있음` 또는 `일부`. 음성 = `답없음`.

**dev 데이터**
- A: H2-dev 43건
- B: H1 86건
- 둘 다 D16 15토픽이다. held-out 라벨(H2-ho-A, H2-ho-B)은 θ를 커밋한 뒤에만 연다.

**후보와 입력**

| 후보 | A 입력 | B 입력 | 점수 |
|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` | (질문, 답 본문) | (질문, 청크) | sigmoid(logit) |
| `BAAI/bge-reranker-base` | (질문, 답 본문) | (질문, 청크) | sigmoid(logit) |
| `cross-encoder/nli-deberta-v3-large` | premise = 답 본문, hypothesis = `This text answers the question: "<질문>"` | premise = 청크, 같은 hypothesis | entailment 확률 |
| MiniCheck 단일 문서 (**음성 대조, A 전용, 선택 대상 아님**) | doc = 답이 인용한 청크들(실린 그대로), claim = 답 본문 | — | 현행 grounding 연산자 |

- 답 본문은 [chunk_id] 표기를 지운 텍스트다.
- B의 청크는 그 질문 자기 검색 결과 중 컨텍스트에 200자 이상 실린 것(실린 그대로)이고, 점수는 청크별 max다.
- fp16, 배치 순서 고정, 모델 리비전은 HF 캐시 스냅샷 해시를 기록한다.

**선택과 θ**
1. A와 B 각각, 선택 대상 후보 3개 중 **dev AUROC가 가장 높은 것**을 고른다.
2. θ = dev에서 **Youden J(TPR − FPR)를 최대로 하는 임계값**이다. 인접한 정렬 점수의 중점으로 둔다. 동점이면 동점 임계값들의 중앙값을 쓴다.
3. 민감도: SPEC_R2 원 규칙(precision 제약)으로 고른 θ도 함께 보고한다.
4. AUROC, θ, dev 혼동표, 15토픽 leave-one-topic-out의 θ 분포를 `operator_dev.json`에 쓰고 **커밋한 뒤** 다음 단계로 간다.

**held-out 관문** (한 번만 연다. SPEC_R2 §2 그대로)
- 일치율 ≥ 0.80
- Cohen κ ≥ 0.60
- 토픽 군집 부트스트랩 κ 95% CI 하한 ≥ 0.40
- 음성 예측의 Hájek 가중 precision ≥ 0.80, CI 하한 ≥ 0.70. 음성 예측은 A에서 '질문에 답 안 함', B에서 '답 없음'이다.

**실패 시**: 그 연산자의 지표(coverage_answered 또는 coverage_answerable)는 민감도 지표로 내리고 legacy coverage를 주지표로 둔다. 2순위 후보로 held-out을 다시 열지 않는다.

**음성 대조 예측**: MiniCheck의 A dev AUROC는 0.5에 가까울 것이다(grounding ≠ relevance). 이 예측은 결과와 무관하게 보고한다.
