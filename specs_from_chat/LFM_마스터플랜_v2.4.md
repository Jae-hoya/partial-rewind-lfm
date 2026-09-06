# LFM 마스터플랜 v2.4 (통합 확정판)

작성일: 2026-08-24 · 개정: 2026-09-11 · 이 문서는 **유일한 기준 문서**다. 구 실행계획서(CPE-MGM)와 감사 원고(manuscript_IEEEAccess_v0.9)는 보관용이며 기준이 아니다.

**v2.4 변경점 (위임·확정):** 사용자가 2026-09-11 "네가 생각했을 때의 베스트를 다 정해서 진행"으로 논문 완성까지의 판단을 채팅에 위임했다(§0.1 규칙의 사용자 승인 기록). 이에 따라 확정: ① 시스템명·제목 **Partial Rewind** (본문 약어 없음; 코드 `lfm/` 내부명 유지) ② 기여 3개 재편 — C1 검증된 충분성 연산자를 제어 상태로, C2 최소 그래프 변경 + 종료 증명, C3 품질≈전역 재실행·비용 절감; 관계 모듈은 구성요소+ablation ③ 낭비율 <20%는 연구 중단(§9-1)이 아니라 결과 문장 B(프레이밍 변경)로 처리하고 M2·M3를 진행 ④ 관계 v1.1 통제 탈락 시 관계 비활성으로 진행 ⑤ 실행 명령서 `ORDERS_ClaudeCode.md`, 스펙 `SPEC_M1prime`·`SPEC_M3` 발급, 원고 `manuscript_PartialRewind_v0.1`. §9-1은 "연구 질문 교체"에서 "결과 문장 B"로 완화. §1 제목 항목은 위 ①로 대체.

**v2.3 변경점 (방향 복원):**
① **논문은 LFM 하나다.** 추정자 감사를 주제로 한 별도 논문(IEEE Access v0.9)은 투고 대상에서 제외하고 저장소 내부 보고서로 내린다. 감사 결과 중 LFM 논문에 들어가는 것은 §4.8의 "충분성 연산자 검증" 소절 하나와 부록뿐이다.
② **M1을 둘로 나눈다.** M1-V′(추정자 확정, 사람 라벨) → M1′(확정 추정자로 Test A 재측정) → go/no-go. Test A 1차(2026-08-29)는 추정자 결함으로 판정 불능이었고, 그 사실을 §6에 기록한다.
③ **M2·M3는 원안 유지.** 단, 감사에서 배운 네 가지를 설계에 반영한다(§4.8).
④ 방향이 튄 경위와 재발 방지 규칙을 §0.1에 남긴다.

---

## 0. 문서 체계 — 뭐에 뭐가 필요한가

| 받는 쪽 | 넘기는 것 | 목적 |
|---|---|---|
| **Claude Code** (지금) | `CLAUDE.md` + `SPEC_M1Vprime.md` | 사람 라벨 기반 추정자 관문. 라벨 도착 즉시 채점(호출 0) |
| **Claude Code** (V′ 통과 후) | `SPEC_M1D8` §D8-C 재채점 → GO면 M1′ 신규 실행 스펙(추후) | Test A go/no-go |
| **Claude Code** (GO 시) | `CLAUDE.md` + `SPEC_M2_LFM.md` | relation + controllers + Test C |
| **Claude 채팅** | 이 마스터플랜 + `results/` | 판정, 스펙 발급, 논문 초안 |
| **사람 (사용자)** | 이 마스터플랜 · 라벨 시트 | go/no-go 결정, 블라인드 판정 |
| **지도교수** | `교수보고_LFM_연구제안.docx` | 방향 검토 |

원칙: **구현 지시는 SPEC으로, 판단 기준과 논문 설계는 이 문서로.** `results/`는 실험 종료까지 삭제 금지.

### 0.1 방향 이탈 기록과 재발 방지
- 2026-09-05~09: Test A 판정 불능의 원인(추정자)을 추적하다가 감사 결과로 별도 논문을 쓰는 방향으로 튀었다. 사용자 주제를 바꾸는 판단을 에이전트와 채팅이 사용자 확인 없이 내렸다.
- **규칙:** 논문의 주제·제목·기여를 바꾸는 결정은 사용자가 직접 문장으로 승인한 기록이 있어야 한다. 에이전트가 전달하는 "사용자 지시"는 실험 범위에만 적용하고, 논문 방향에는 적용하지 않는다.
- **규칙:** 진단 실험은 "현재 마일스톤을 통과시키기 위해 필요한가"를 SPEC 발급 전에 한 줄로 적는다. 적을 수 없으면 하지 않는다.

---

## 1. 논문 정체성 (변경 없음)

- **제목:** Localize, Freeze, Mutate: Evidence-Driven Partial Re-execution for Multi-Perspective RAG — 약칭 **LFM**. 약칭 충돌(Liquid Foundation Model) 재확인 후 필요 시 약칭만 교체. 내부명 CPE-MGM은 본문 등장 금지.
- **정체성 문구:** *A stateful control architecture for localized partial re-execution in multi-perspective RAG.*
- **관통 문장:** Prior work uses semantic conflict to trigger further reasoning or retrieval, attributes failures post hoc, or replays computations invalidated by changed inputs. We instead localize semantic inconsistency at the perspective-relation level and selectively mutate only the affected execution subgraph while preserving validated branches.
- **계열 압축식:** MA-RAG: Conflict→Retrieval · AgentLocate: Failure→Attribution · Execution Lineage: InputChange→Invalidation · **LFM: CrossPerspectiveRelation→Localization→Minimal ΔG**

## 2. Contributions (3개 고정)

1. **C1. Cross-Perspective Relation State** — R_ij를 execution graph의 persistent control state로 정의.
2. **C2. Relation-Conditioned Graph Mutation** — (E_i, R_ij) → a ∈ {FREEZE, EXPAND, VERIFY, PRUNE}, G_{t+1} = G_t ⊕ ΔG_t.
3. **C3. Localized Partial Re-execution** — valid branch 보존 + affected subgraph만 재계산으로 Quality ≈ GlobalRetry, Cost < GlobalRetry 실증.

## 3. 신규성 방어 (변경 없음 — v2.2 §3 그대로)

4-way 경계표(MA-RAG Wu ICML'26 / SELENE / Execution Lineage / AgentLocate·VerifyMAS), 핵심 논거(관점 간 contradiction은 evidence 채널의 위치를 가리킨다), 동명 MA-RAG 2편 구분, 독창성 주장 금지 목록 — v2.2와 동일.

## 4. 방법론 확정 스펙

### 4.1~4.7 (v2.2 그대로)
Action 4개(FREEZE/EXPAND/VERIFY/PRUNE) · E_i 4지표 · R_ij 4 feature(양방향 NLI 1패스, contradiction=max, CoverageGain 무호출, Σ≠1) · UNRESOLVED_CONFLICT · G_t=(V_t,E_t,S_t) 형식화 · Proposition 1 · Pareto 서술. 본문은 v2.2 §4.1~4.7 참조(변경 없음).

### 4.8 감사에서 이어받는 것 (v2.3 신설) — M2 설계에 직접 반영

| 배운 것 | 근거 | M2 설계 반영 |
|---|---|---|
| **충분성 판정은 추정자 검증 없이는 성립하지 않는다.** 현행 `nli-deberta-v3-base`는 미지지 판정의 72.5%가 오판(잠정) — Test A에서 FREEZE 가능한 branch가 3/600, 선택적 재실행 0회 | D8-B, 표 1 | E_i.grounding은 **V′를 통과한 (체크포인트, 전제 구성) 쌍**으로만 계산. g₁ 사용 금지. FREEZE는 그 추정자의 S_τ 위에서만 |
| **raw grounded 비율은 타당성 지표가 아니다.** 좋아 보이던 후보가 교란 통제에서 탈락(deberta-large 21%, 문장분할-max 50.5%) | D8-A, D10 | 모든 추정자 결정에 **교란 음성 통제**(다른 토픽 전제, ≤5%) 필수. R_ij의 NLI에도 동일 통제: 다른 토픽 claim 쌍에서 agreement·contradiction 비율 보고 |
| **전제 희석은 길이가 아니라 무관 내용 비율.** 전제 구성은 모델 의존적 | D7-0, D14 | R_ij는 claim–claim 쌍이라 희석 문제가 작다. E_i.grounding의 전제 구성은 V′에서 확정된 것만 사용, 스윕 금지 |
| **빈 답안은 검색 실패가 아니라 미인용.** 215/600, 라운드 반복으로 안 줄어듦 | D11 | coverage=0인 branch는 EXPAND 대상이 아니라 **재생성(인용 형식 강제)** 대상. 별도 action 없이 EXPAND 내부 분기로 처리하되 trace에 구분 기록 |
| **verify는 두 절 논리곱**(grounding 평균 ≥0.80 ∧ coverage 최솟값 ≥0.80, ≥ 비교) · sufficient()는 > 비교 | A-4 | 코드 규칙 그대로. 재실행 판정과 충분성 판정의 부등호 차이를 논문 각주로 |
| 감사 대상을 감사 기준으로 쓰면 관문이 무너진다 | D9-P | 모든 관문의 기준 라벨은 **사람**. 추정자끼리 서로를 검증하지 않는다 |

**논문 반영:** LFM 논문 Method에 "Sufficiency Operator Validation" 소절 1개(≈1.5쪽). 내용: (i) Test A 1차에서 FREEZE가 한 번도 성립하지 않았다는 사실(50질의 전부 매 라운드 4/4 재실행, grounding 절 0/50) (ii) 추정자 후보 비교표(교란 순증: MiniCheck 64.3 / bart 59.9 / roberta 17.0 / 현행 15.7 / deberta-large 탈락) (iii) V′ 사람 라벨 기준 precision·recall과 확정 추정자. 나머지 감사(길이 편향·검색 스윕·관문 결함·전제 구성 축)는 부록 또는 저장소.

## 5. 데이터 (변경 없음)
v2.2 §5 그대로. 실측 확정값: 150,000행 → 고유 142,972, 섹션 청킹(1,200토큰 초과 시 문단 분할), full held-out 색인 142,922 문서/685,683 청크, partial 142,972/686,221. NLI 현행 `cross-encoder/nli-deberta-v3-base`(교체 예정), 검색 bm25s top_k 5, 생성 `gpt-5-mini`(reasoning_effort=low, max_tokens=2000).

## 6. 실험 로드맵

### M1 — Test A (동기 실험)

**1차 (2026-08-29, 종료):** 50토픽, GlobalRetry 3라운드, 600 branch. 낭비율 0.5% [0.0, 1.5], verify 0/50(grounding 절 0/50, coverage 절 2/50), sufficient 3/600. **판정 불능** — 낭비율이 낮아서가 아니라 충분성 판정이 한 번도 서지 않아서. 원인은 추정자(§4.8). 라운드 반복은 coverage를 평균 0.060 낮췄고 grounding·빈답안은 유의하게 안 움직였다. 비용: 입력 1,882,035 / 출력 697,798 토큰, 67.8분.

**M1-V′ — 추정자 확정 (진행 중, `SPEC_M1Vprime.md`):** 사용자 블라인드 라벨(1단계 60건, 필요 시 +40) → 후보 2개(MiniCheck-Flan-T5-Large 단일문서, bart-large-mnli)를 음성 통제 + 사람 라벨 precision/recall/N층 회수로 판정. 통과 후보 없으면 §9-1로.

**M1-C — 재채점 (호출 0, `SPEC_M1D8` §D8-C):** 확정 추정자로 1차 산출물 600 branch 재채점. 낭비율·verify 통과·sufficient 수를 구판과 나란히.

**M1′ — 신규 실행 (GO 시, 스펙 추후):** 확정 추정자를 verifier에 넣고 50토픽 × 시드 3회. 이 값이 논문 Fig.1이다.

**Go/No-Go (내부 전용, τ=0.8, M1-C 재채점 값 기준):**

| 조건 | 판정 |
|---|---|
| 라운드0 verify 통과 ≥ 5/50 **또는** 낭비율 ≥ 20% | GO → M1′ 실행 → 아래 표로 재판정 |
| 둘 다 미달 | §9-1 — 확정 추정자로도 충분성 모집단이 비어 있음 |

M1′ 낭비율 판정(원안): ≥60% 그대로 진행 / 40–60% 진행, 절감 기대치 하향 / 20–40% 더 어려운 토픽셋으로 재측정 / <20% 전제 오류.

### M2 — Relation + Controllers + Test C (`SPEC_M2_LFM.md`, GO 시 발급 즉시 착수)
- 구현: `relation.py`(§4.3), `estimator.py`(V′ 확정 쌍), controllers {lfm(규칙), oracle, random, global_verifier}, REPORT 모드, trace.
- **Test C-1 (합성):** contradiction 주입·coverage 삭제·redundancy 주입 → 오라클 라벨 자동. Oracle > LFM > GlobalVerifier > Random. **LFM≈Random이면 §9-2.**
- **Test C-2 (자연):** referencerisk 상위 문서로 자연 conflict 후보 → 소규모 수작업 검증.
- Relation 추출 P/R/F1 별도 보고 + **교란 음성 통제**(§4.8).

### M3 — 주 실험 + Ablation + 분석 (v2.2 그대로)
Test B Pareto(전 baseline, `gpt-5-mini` 고정) · Test D ablation(−relation ≈ Full이면 §9-3) · Test E 분석 · 평가자 정책(LLM-judge 게이트, self-preference 방어) · 자동 지표. 본문 v2.2 §6 M3 참조.

## 7. 논문 작성 순서 (변경 없음)
Algorithm 1 → Problem Formulation → **M1′ 결과(Fig.1)** → Introduction → Method(+§4.8 검증 소절) → Test B~E → Results·Ablation → Related Work → Abstract·Conclusion.

## 8. 확장 로드맵 (변경 없음)
Paper 2(frozen graph = 세션 메모리, THAW) · 보류 항목 · 본 논문 포함 확장분 — v2.2 §8.
**추가 보류:** 추정자 감사 논문(v0.9). LFM 투고 후 별도 판단. 지금은 저장소 보고서.

## 9. 연구 중단 신호 (셋 중 하나면 방법 수선이 아니라 연구 질문 교체)
1. M1-C·M1′ 낭비율 < 20% (확정 추정자 기준)  2. Test C에서 LFM ≈ Random  3. Test D에서 −relation ≈ Full

## 10. 마스터 체크리스트 (투고 전) — v2.2 §10에 두 줄 추가
- [ ] 추정자는 사람 라벨 관문(V′) 통과본 · 교란 음성 통제 ≤5% · (체크포인트, 전제 구성) 쌍으로 명시
- [ ] Method에 Sufficiency Operator Validation 소절 · Test A 1차 "FREEZE 0회" 사실 명시 · 감사 나머지는 부록
- (이하 v2.2 §10 항목 동일)
