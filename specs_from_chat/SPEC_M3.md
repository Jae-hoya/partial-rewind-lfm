# SPEC M3 — 주 실험: Test B(Pareto) · Test D(Ablation) · Test E(그래프 분석)

발급: 2026-09-11 · 착수 조건: M2 Test C-1 산출·보고 완료(통과/탈락 무관 — 탈락 시 관계 모듈 비활성 상태로 진행) · 마스터플랜 v2.4 §6 M3

## 0. 원칙
- **동일 하네스**: 전 시스템이 같은 생성기(`gpt-5-mini`)·검색기·코퍼스·토픽·확정 추정자·프롬프트 템플릿을 쓴다. 비교는 제어 흐름의 차이만 반영한다.
- **baseline은 "-like" 재구현**이며 논문에 그렇게 표기한다(원 논문의 학습된 구성요소는 재현하지 않는다). 각 baseline의 정의를 `baselines/README.md`에 5줄 이내로 고정하고 실행 후 수정하지 않는다.
- 임계·예산·판정자 프로토콜은 이 스펙 값. 결과를 본 뒤 바꾸지 않는다.

## 1. Test B — 품질·비용 Pareto
**시스템** (토픽 50, 시드 42 고정; Partial Rewind와 GlobalRetry는 M1′ 시드 3회 재사용 가능):

| 시스템 | 정의 (5줄 규칙) |
|---|---|
| Naive RAG | 관점 없음. 질의 1회 검색(k=5) → 답안 1회 |
| Self-RAG-like | Naive → 자기비평 1회(근거 충분한가 예/아니오, 같은 모델) → '아니오'면 질의 재작성 1회 검색·재생성. 최대 2패스 |
| Adaptive-RAG-like | 무호출 복잡도 휴리스틱(sub-question 수·엔티티 수) → {no-retrieval, single, iterative(2)} 중 하나 |
| STORM-like | 4관점 분해 + 관점별 sub-question·검색·답안, 재시도 없음(= Test A 라운드 0) |
| GlobalRetry | R=3(= M1′) |
| Partial Rewind | Algorithm 1, B ∈ {0.5, 1, 2}×B₀, B₀ = GlobalRetry 3라운드 호출 수. actions_per_round=1 기본, =3 부록 |

**지표**: 생성 호출(주), 토큰, 검색 호출, wall-clock / coverage, grounding(확정 추정자), citation accuracy(구조화 참조 대비), entity recall(target article 대비), heading soft recall / 판정자 쌍대 승률(GlobalRetry 기준).

**판정자 프로토콜**: `gpt-5.4-nano`, 시스템명 블라인드, 위치 교환 2회 평균, temperature 0, 3회 반복 자기일치율 보고. **게이트**: 합성 clean-vs-corrupted 쌍 40개(정답 문장 삭제·거짓 문장 삽입)에서 정확도 ≥ 90%여야 사용. 미달 시 `gpt-5-mini`로 1회 교체 후 재게이트, 그래도 미달이면 판정자 열을 논문에서 뺀다. 판정 차원: coverage·faithfulness·coherence·conflict resolution·source diversity — 각 차원 점수는 부록, 본문은 종합 승률.

**산출**: `results/m3/testb/<system>/`, 집계표 `results/m3/testb/summary.csv`(질의 부트스트랩 CI), Pareto 그림 데이터 `pareto.csv`.

**결과 문장(고정)**: 통과 — "At the global-retry budget Partial Rewind matches global retry on every quality metric within [[Δq]]; at half that budget it retains [[…]]." / 부정 — "Partial Rewind does not dominate global retry at any budget." 어느 쪽이든 그대로 쓴다.

**비용 추정**: 50토픽 × (1 + 3 + 2 + 20 + 60 + 30 + 60 + 120) ≈ 15k 생성 호출 ≈ 입력 22M / 출력 9M 토큰. 판정 ≈ 50 × 7쌍 × 2 × 3 ≈ 2,100 nano 호출. 캐시로 M1′ 재사용 시 GlobalRetry·Partial Rewind(B=B₀) 부분 절감.

## 2. Test D — Ablation (B = B₀, 토픽 50)
| 변형 | 구현 |
|---|---|
| − FREEZE | sufficient branch도 매 스텝 재평가·재실행 대상에 포함 |
| − relation | R_ij 계산 안 함 → VERIFY·PRUNE 비활성 (Test C-1 탈락 시 이 변형이 Full이 된다) |
| − EXPAND | coverage 결손 branch를 방치, FREEZE/VERIFY/PRUNE만 |
| − VERIFY | contradiction 무시 |
| − PRUNE | redundancy 무시 |
| 단일 관점 | n=1 |
지표: Test B와 동일 + localization accuracy(C-1 재사용). **판정(고정)**: −relation ≈ Full(전 지표 CI 겹침)이면 마스터플랜 §9-3 → 제목·기여에서 cross-perspective 철회, 논문 결과 문장 B.

## 3. Test E — 그래프 분석 (호출 0, trace에서)
|ΔG| per step 분포, frozen-branch 재사용률, 난이도(질의 길이·sub-question 수 3분위)별 action 분포, 수렴 스텝, UNRESOLVED 비율. 스윕 n ∈ {2,4,8} × pairing ∈ {full, blocked}는 토픽 15개 서브셋(호출 소량). 실패 궤적 3개 선정: 잘못된 FREEZE 1, UNRESOLVED 1, 예산 소진 1 — 부록 그림.

## 4. CLAUDE.md 3줄
- 마일스톤: `M3 — Test B/D/E`
- STOP RULE: `baselines/README.md 고정 + 판정자 게이트 산출 후 중단·보고 → 승인 후 Test B → summary 산출 후 중단·보고 → Test D·E → 산출 후 중단·보고`
- 금지: `baseline 정의·임계·예산·판정자 프로토콜 사후 수정 금지. 판정자는 judge/ 모듈 한 경로. 결과를 본 뒤 시스템 추가 금지.`
