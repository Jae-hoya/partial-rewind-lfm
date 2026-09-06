# SPEC M2 — LFM 본체: Relation State · Localization · Minimal ΔG · Controllers · Test C

> **전사본.** 원본은 공저자가 2026-09-10 에 발급한 아티팩트 `Spec m2 lfm` 이다.
> 이 파일은 저장소에서 참조하기 위한 전사이며, 불일치가 있으면 **원본이 정본**이다.

발급 2026-09-10 · 마일스톤 M2 · **착수 조건: M1-V′ 확정 + M1-C GO (착수 전까지는 읽기만)**
· 마스터플랜 v2.3 §4 준용.

이 스펙이 LFM 이다. 지금 저장소에 컨트롤러는 `GlobalRetry` 하나, `controller/base.py`
인터페이스뿐이다. 관계 추출·국소화·최소 변경 연산 전부 여기서 처음 짠다.
**원칙: 바꾸는 것은 제어 흐름뿐이다.** 생성 프롬프트·검색기·추정자(V′ 확정 쌍)·
코퍼스·토픽은 Test A/M1′ 와 동일. 그래야 Test C·B 의 차이가 컨트롤러에 귀속된다.

## 0. 구성 요소와 파일

| 파일 | 역할 | 상태 |
|---|---|---|
| `lfm/estimator.py` | V′ 확정 (체크포인트, 전제 구성) 쌍으로 grounding·support_strength 계산. g₁ 호출 경로 삭제 | 신규 |
| `lfm/relation.py` | R_ij 4 feature (§2) | 신규 |
| `lfm/localize.py` | (E, R) → 문제 branch·edge·sub-question 집합 (§3) | 신규 |
| `lfm/actions.py` | FREEZE/EXPAND/VERIFY/PRUNE 의 구조 효과 + 실행 (§4) | 신규 |
| `lfm/graph.py` | G_t=(V_t,E_t,S_t), ΔG 적용, \|ΔG\| | 신규 |
| `controller/lfm.py` | 규칙 기반 v1 (§6) | 신규 |
| `controller/oracle.py`, `random.py`, `global_verifier.py` | Test C 비교선 (§7) | 신규(global 은 기존 GlobalRetry 래핑) |
| `tests/test_c1.py`, `test_c2.py` | Test C (§8) | 신규 |
| `trace/` | 스키마 동결 유지, action·ΔG 필드 추가 (§9) | 확장 |

## 1. Evidence State (변경 없음, 추정자만 교체)

`E_i = [coverage, grounding, source_diversity, support_strength]`. 정의는 마스터플랜 §4.2.
grounding·support_strength 는 `estimator.py`(V′ 확정 쌍)로만. coverage 는 결정론.
`sufficient@τ = coverage=1.0 ∧ grounding>τ` (엄격 부등호, τ=0.8 기본).

추정자 교란 통제 유지: `estimator.py` 에 `--negative-control` 모드를 두어 M2 실행 전
1회 재확인(교란 ≤5%). 통과 기록을 MANIFEST 에.

## 2. Relation State — R_ij ∈ [0,1]⁴

입력: branch i, j 의 atomic claim 집합 C_i, C_j (기존 claim 분해 산출물 재사용, 호출 0).

**NLI 양방향 1패스**: 모든 (c_a ∈ C_i, c_b ∈ C_j) 에 p(a→b), p(b→a) — 3-way(ent/neu/con).
체크포인트는 V′ 확정 체크포인트와 동일 계열. **MiniCheck 확정 시 relation 은
`bart-large-mnli` 를 쓴다** — MiniCheck 는 문서 단위 검증기라 3-way 출력이 없다.
이 대체는 스펙에서 고정하고 논문에 명시한다. claim–claim 은 짧은 전제라 희석 문제가 작다.

pair 단위 → branch 단위 집계:

- `agreement_ij` = pair 들의 `(p_ent(a→b)+p_ent(b→a))/2` 중 **상위 q=25% 평균**.
  (대부분 pair 는 무관해 0 근처이므로 전체 평균은 정보가 없다. **q 는 고정, 스윕 금지.**)
- `contradiction_ij` = pair 들의 `max(p_con(a→b), p_con(b→a))` 의 **최댓값**.
  VERIFY 트리거라 보수적. 동시에 **argmax pair (a\*, b\*) 를 저장** — VERIFY 의 표적.
- `complementarity_ij` = Relatedness × NonRedundancy × CoverageGain.
  Relatedness = 두 branch 의 sub-question 매핑 중 동일·인접 비율(무호출),
  NonRedundancy = 1 − redundancy_ij,
  CoverageGain = cov(Q_i∪Q_j 기준 인용) − max(cov_i, cov_j).
- `redundancy_ij` = `1[agreement_ij > τ_a] × 1[cos(answer_i, answer_j) > τ_r]`.
  cos 는 로컬 임베딩(기존 leakage audit 모델 재사용).

**기본 임계 (고정, 부록 민감도만)**: τ_a=0.6, τ_r=0.85, τ_c=0.7(contradiction), τ_g=0.8(sufficiency).

**교란 음성 통제 (필수)**: 다른 토픽 branch 쌍(i → i+7 mod 15)에 같은 계산.
보고: 교란 쌍에서 `contradiction_ij > τ_c` 비율, `agreement_ij > τ_a` 비율.
둘 다 ≤5% 가 아니면 **τ 를 올리는 게 아니라** 그 사실을 보고하고 Test C-1 결과 해석에
반영한다(임계 조정은 Test C 이후 부록 스윕에서만).

복잡도: n=4, m≈15 → 6 pair × 225 × 2 = 2,700 NLI/topic/round.
`pairing="full"` 기본, `"blocked"`(공유 entity ∨ cos>τ_b) 옵션 — Pareto 비교용.

산출: `results/m2/relation/<topic>_<round>.json` — 4 feature, argmax pair, pair 수, 교란 통제값.

## 3. Localization — (E, R) → 무엇을 고칠지

입력: 활성 branch 집합 A, E_i, R_ij, 예산 B.
출력: action 후보 목록 (branch 또는 edge 또는 sub-question 단위, **우선순위 순**).

단계 (결정론):

- **FREEZE 후보**: i ∈ A, `sufficient@τ(E_i)=1`, 그리고 `∀j: contradiction_ij < τ_c`. → FREEZE i.
- **VERIFY 후보**: edge (i,j) with `contradiction_ij ≥ τ_c`, 아직 VERIFY 횟수 < **K=2**.
  표적은 argmax pair (a\*, b\*) 와 그 claim 이 속한 sub-question. → VERIFY (i,j).
  - **국소화 규칙**: 두 branch 중 `support_strength` 가 낮은 쪽을 suspect 로 표시
    (재검색 우선). 같으면 둘 다.
- **PRUNE 후보**: `redundancy_ij = 1` 이고 둘 다 sufficient → `source_diversity` 낮은 쪽 PRUNE.
  둘 다 non-sufficient 면 **PRUNE 안 함**(정보 손실 위험).
- **EXPAND 후보**: i ∈ A, non-sufficient, contradiction 없음. 표적은 coverage 가 빠진
  sub-question 집합 `Q_i^miss` 와 `grounding<τ_g` 인 문장이 속한 sub-question.
  - `coverage=0`(미인용 답안)이면 EXPAND 대신 **REGEN**(같은 sub-question, 인용 형식
    강제 프롬프트 재생성) — 별도 action 이 아니라 EXPAND 의 분기이며
    trace 에 `expand.kind=regen` 으로 기록.

**우선순위: VERIFY > EXPAND > PRUNE > FREEZE** (FREEZE 는 비용 0 이라 매 라운드 마지막에 일괄).

**최소 변경 원칙: 재실행 단위는 sub-question 이다.** branch 전체 재실행은 하지 않는다
(coverage=0 REGEN 제외). 이것이 Test A 의 branch 단위 낭비율보다 **한 단계 더 작은 ΔG** 다.

## 4. Actions — 구조 효과와 실행

| Action | 구조 효과 (ΔG) | 실행 | 호출 |
|---|---|---|---|
| `FREEZE i` | ΔS: i → FROZEN. artifact 보존, 이후 재호출 금지 | 없음 | 0 |
| `EXPAND (i, Q)` | ΔV>0: sub-question 노드 재실행 (검색 폭 k→k_e=10 또는 iterative 1단계 — breadth/depth 파라미터, **기본 breadth**) | 해당 sub-question 만 검색·재생성·재평가 | \|Q\| 생성 |
| `VERIFY (i,j)` | ΔV>0: **verification subgraph 신규 생성** — (a\*,b\*) 의 sub-question 쌍에 대해 표적 재검색(두 claim 의 entity 로 질의 구성) → 두 branch 의 해당 sub-answer 재생성 → R_ij 재계산 | 2 sub-answer 생성 | 2 |
| `PRUNE i` | ΔV<0: i → PRUNED (artifact 보존, 합성 단계에서 제외) | 없음 | 0 |

**재평가**: action 실행 후 영향받은 branch 의 E_i 와, 그 branch 가 낀 모든 R_ij 만 재계산
(다른 pair 는 캐시). 이 "부분 재평가" 도 \|ΔG\| 에 포함.

**UNRESOLVED_CONFLICT**: edge (i,j) VERIFY **2회** 후에도 `contradiction ≥ τ_c` →
S: 두 branch 모두 UNRESOLVED_CONFLICT, disagreement report 출력(관점별 주장·근거 병기,
문구는 "현재 evidence budget 내 미해소").

## 5. Graph — G_t = (V_t, E_t, S_t), ΔG, 종료

V: branch 노드 + sub-question 노드 + verification 노드.
E: branch–sub-question 포함 관계 + relation edge R_ij.
S ∈ {ACTIVE, FROZEN, PRUNED, UNRESOLVED_CONFLICT}.

`ΔG_t = (ΔV, ΔE, ΔS)`. `|ΔG_t| = |ΔV| + |ΔE| + |ΔS|`. trace 에서 직접 계산.

**종료**: (a) A = ∅ (전부 FROZEN/PRUNED/UNRESOLVED) 또는 (b) 예산 B 소진 또는
(c) 적용 가능한 action 없음.
**Proposition 1 의 두 bound 를 코드 assert 로**: `T ≤ ⌊B/c_min⌋`, `VERIFY 총수 ≤ K·|E_R|`.

**예산 B**: **생성 호출 수로 정의**(토큰 아님 — 결정론). 기본 B = GlobalRetry 3라운드와
같은 호출 수(비교 공정). Pareto 용 스윕 B ∈ {0.5, 1, 2}× 기본.

**FREEZE 단조성 assert**: `F_t ⊆ F_{t+1}`.

## 6. Controller v1 — 규칙 기반 (학습형은 후속)

    loop until 종료:
      E ← estimator(active branches)      # 부분 재평가
      R ← relation(active pairs)          # 캐시 + 부분 재계산
      cands ← localize(E, R, budget)
      a ← cands[0]
      ΔG ← apply(a)
      G ← G ⊕ ΔG ; log(trace)
    FREEZE 일괄 (비용 0)
    report: FROZEN branch 합성 + UNRESOLVED 보고

라운드당 action 1개는 Test C 에서 "첫 action 이 오라클 표적을 맞히는가"를 재기 위함이다.

## 7. 비교 컨트롤러 (Test C)

| 컨트롤러 | 규칙 |
|---|---|
| **Oracle** | Test C-1 의 주입 라벨을 보고 정답 action 을 냄 (상한) |
| **LFM** | §6 |
| **GlobalVerifier** | verify 실패 시 전 branch 재실행 (= Test A 의 GlobalRetry) |
| **Random** | 활성 branch·edge 에서 균등 무작위 action (하한) |

**전부 동일 estimator·relation·예산·시드.**

## 8. Test C

### C-1 합성 (오라클 라벨 자동, 생성 호출 소량)

M1′ 산출물(또는 M1-C 재채점 산출물) 라운드 0 상태에서 토픽당 1개 교란:

- **contradiction 주입**: branch j 의 claim c_b 하나를 골라 부정형으로 바꾼다
- **coverage 삭제**: branch i 의 sub-question q_ik 인용을 제거
- **redundancy 주입**: branch j 의 답안을 branch i 답안의 paraphrase 로 교체

지표: **localization accuracy** = 첫 action 이 오라클과 (종류, 표적) 모두 일치하는 비율.
컨트롤러 4개 × 3교란 × 50토픽.

**판정 (고정)**: LFM > Random 유의(부트스트랩 CI 비겹침) **그리고** LFM ≥ 0.5×Oracle.
LFM ≈ Random 이면 마스터플랜 §9-2.

호출: 교란 생성 ≤ 100(nano) + 컨트롤러 첫 action 실행 ≤ 4×150 = 600 sub-answer 생성(mini).
캐시 적극 사용.

### C-2 자연 (현실 신호 존재)

referencerisk 상위 문서에서 파생된 토픽 20개 선별 → 파이프라인 라운드 0 →
`contradiction_ij ≥ τ_c` 인 edge 전부 추출 → **사용자 수작업 판정**(진짜 관점 간 충돌인가:
예/아니오, 20~30건). 정밀도 보고. **관문 아님, 존재 증명.**

### Relation 추출 P/R/F1

C-1 의 contradiction 주입 50건을 양성, 교란 음성 통제 쌍을 음성으로 하여
contradiction 검출 P/R/F1. agreement 는 redundancy 주입 50건으로 동일.

## 9. Trace 스키마 (동결 + 확장)

기존 필드 유지. 추가:
`action{type, target_branch, target_edge, target_subq[], kind}`,
`delta_g{dV,dE,dS,size}`, `state_after{branch: S}`, `relation_snapshot_id`,
`budget_remaining`, `reward=null`(backfill 구조 유지).

## 10. CLAUDE.md 3줄

- 마일스톤: **M2 — Relation·Controllers·Test C**
- **STOP RULE**: `relation.py` + 교란 통제 산출 후 **중단·보고** → 승인 후 controllers + Test C-1
  → 산출 후 **중단·보고** → C-2 는 사용자 판정 대기
- 금지: 생성 호출은 C-1 교란 생성·sub-answer 재생성에만. 임계(τ_a, τ_r, τ_c, τ_g, q, K)
  변경 금지.

## 11. 비용

NLI: 2,700 pair × 50토픽 ≈ 135k (GPU 수십 분).

## 12. 결과 문장 (실행 전 확정)

- **통과**: "관점 간 contradiction 의 위치가 실패한 evidence 채널을 국소화한다."
- **탈락**: "R_ij 는 무작위 이상의 국소화 신호를 주지 않는다." → 마스터플랜 §9
