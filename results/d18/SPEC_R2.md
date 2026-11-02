# SPEC_R2 — coverage 재정의와 질문 분해 수정 (확정안)

작성: 2026-11-02.

**만든 과정**
1. 코드와 데이터를 대조했다.
2. 성격이 다른 설계안 3개(지표 중심 / 분해 중심 / 리뷰어 대응 중심)를 따로 만들었다.
3. 셋을 통합했다.
4. AE·방법론 관점에서 검사해 16건을 지적받았고, 그 수정을 반영했다.

원본은 `_워크플로_원본/SPEC_R2_초안.md`(세부 수식, 파일 목록)와 `SPEC_R2_검사.md`(지적 16건)에 있다. 이 문서는 **검사 수정을 반영한 최종본**이다. 두 문서가 충돌하면 이 문서를 따른다.

코드 경로는 `나의 rag\03_LFM\LFM\m1` 기준이다. 표기: `[확인]` = 파일·데이터를 직접 확인함, `[추정]` = 확인하지 않음.

---

## 0. 왜 바꾸는가

| 근거 | 내용 |
|---|---|
| H1 사람 라벨 | 모델 기권 43건 중 답할 수 있었던 것 **0건**. 모델이 답한 43건 중 **16건은 질문에 답하지 않음**(그중 12건은 MiniCheck 지지 = 근거엔 맞고 질문엔 안 맞음) |
| D15~D17 | 빈 답은 절단·형식 실패가 아니라, 답이 없는 sub-question에 대한 올바른 기권 |
| 코드 [확인] | coverage는 **유효 인용이 1개 이상 붙은 sub-answer**를 커버로 센다(`lfm/evidence.py:39-46`). 질문 관련성은 보지 않는다 |
| 코드 [확인] | 분해 프롬프트는 근거를 보지 않고 "answerable from an encyclopedia"만 요구한다(`lfm/perspectives.py:30-31`). 그런데 본실험은 정답 문서를 인덱스에서 뺀다 |

**재정의는 통과율을 올리는 장치가 아니다. 측정 타당성을 고치는 것이다.** 이 점을 Test A 재집계로 확인했다 [확인].

| 항목 | 값 |
|---|---|
| verifier grounding 절 통과 | 0/150 query-round |
| verifier coverage 절 통과 | 2/150 query-round |
| 두 절 동시 통과 | 0 |
| **coverage = 0인 branch를 grounding 평균에서 빼면** | grounding 절 통과 **6/150** |
| 원래 mean_g 최댓값 | 0.7946 (기준 0.80) |

**정직하게 써야 할 점**: grounding 절은 coverage 결함과 독립된 구속 조건이 아니다. 인용 없는 branch는 grounding이 0으로 고정되어 평균을 끌어내린다(cov=0인 219개 branch 전부 grounding 0 [확인]). 그래서 "grounding이 진짜 병목"이라는 방어는 **캐시된 Test A에 한해서만** 쓸 수 있고, 6/150을 함께 공개한다. **mean_g의 집계 대상(어떤 branch를 평균에 넣는지)은 어떤 변형에서도 바꾸지 않는다**는 것을 사전등록 1항으로 둔다.

**숫자 대조** [확인]
- 빈 답 branch(degraded) = **215**
- coverage = 0인 branch = **219** (215 + 본문은 있지만 유효 인용이 없는 4)

원고에서는 두 정의에 이름을 따로 붙여 구분한다.

---

## 1. 새 지표

**원자 술어** (branch b, sub-question i = 1..n_b)
- `answered_i`: 유효 인용이 붙은 sub-answer. legacy coverage와 같은 규칙이다.
- `relevant_i` = answered_i · [연산자 A(질문, 답 본문) ≥ θ_A]. 답이 질문에 답하는가.
- `answerable_i` = [연산자 B(질문, 그 질문의 검색 청크 R_i) ≥ θ_B]. 컨텍스트로 답할 수 있는가. **입력에 생성기 출력은 넣지 않는다.**

**지표**

| 키 | 정의 | 지위 |
|---|---|---|
| `coverage` (legacy) | Σ answered / n_b | 그대로 둔다. 기존 수치 보존 |
| **`coverage_answered`** | Σ relevant / n_b | **주지표.** 항상 legacy 이하이다(cov_A ≤ cov_L) |
| `coverage_answerable` | Σ_{i∈U} relevant / \|U\|, U = 답할 수 있는 sub-question | 보조. **단독 보고 금지.** \|U\| < k_min이면 0(k_min = 2, 민감도 {1, 2, 3}) |

**부속 지표** (항상 함께 보고)
- `off_question` = 근거는 있으나 질문에서 벗어난 답의 비율
- `over_abstain` = 답할 수 있는데 비운 비율
- `abstention_precision` = 비운 것 중 답할 수 없었던 비율 (H1 사람 기준 41/43)
- `answerable_rate ρ` = 관점별로 낸다. 분해 품질 지표다.

**게이밍 방지** (명제로 쓰고 단위 테스트로 강제)
1. 전부 기권하면 cov_A = 0이다.
2. 생성기가 붙인 status('insufficient')는 어떤 식에도 들어가지 않는다.
3. 인용만 붙인 오프토픽 답은 relevant = 0이다.
4. 분모 U는 (질문, 청크)만으로 정해진다. 생성기가 조작할 수 없다.

**grounding은 재정의하지 않는다.** grounding까지 바꾸면 골대 이동 의심을 피할 수 없다.

---

## 2. 관련성 연산자 A·B — 선택과 검증

**후보** (로컬 HF 캐시에 있음, 다운로드 0, API 0, VRAM [추정])

| 후보 | 역할 |
|---|---|
| `BAAI/bge-reranker-v2-m3` | 1순위 |
| `BAAI/bge-reranker-base` | 2순위 |
| `cross-encoder/nli-deberta-v3-large` | 평서문 템플릿 대조. 템플릿은 사전등록 |
| MiniCheck | **음성 대조.** A에서 AUROC가 약 0.5로 나오면 "grounding ≠ relevance"의 기계적 증거가 된다 |

gpt-5.4-nano judge는 이전 게이트(65.6%)를 실패했으므로 제외한다. 다시 쓰려면 같은 게이트를 통과해야 한다.

**순환 방지: 고르는 데이터와 검증하는 데이터를 나눈다**
- **dev (선택용)**: D16의 15토픽만 쓴다.
  - B: H1 86건
  - A: H2-dev 43건. H1에서 '답함'이었던 항목을 질문 + 답으로 다시 라벨한다. **H1의 재판정이라는 점을 공시한다.**
- **held-out (검증용)**: Test A 50토픽 중 나머지 35토픽, 라운드 0. θ를 고정하고 커밋한 **뒤에** 한 번만 연다. 헤드라인 수치는 35토픽으로만 낸다.

**θ 규칙** — 검사 F3을 반영해 결과에 불리한 방향으로 정했다.
- θ_A: dev precision ≥ 0.85를 만족하는 범위에서 **cov_A가 가장 낮아지는 값**
- θ_B: dev precision ≥ 0.90을 만족하는 범위에서 **분모 제외율이 가장 낮아지는 값**
- 결과에 유리한 방향의 θ는 민감도로만 보고한다. 조건을 만족하는 θ가 없으면 그 연산자는 미채택이다.

**채택 관문** (held-out, 사전등록)
- 일치율 ≥ 0.80, Cohen κ ≥ 0.60, 토픽 군집 κ 95% CI 하한 ≥ 0.40
- 핵심 클래스(A의 미커버, B의 답없음)의 Hájek 가중 precision ≥ 0.80, CI 하한 ≥ 0.70

**관문 실패 시**: coverage_answered를 주지표에서 내리고 legacy coverage만 쓴다(검사 F15). 이 선택지를 사전등록에 함께 둔다.

---

## 3. 질문 분해 수정

- **지금 (API 0)**: 고치지 않고 **측정만** 한다. 연산자 B로 기존 sub-question에 답 가능 여부를 사후에 붙이고, 관점별 ρ를 보고한다. 원고 문장: "Decomposition is evidence-blind; we measure, not fix, its unanswerable share."
- **재실험 (공저자 승인 후) — 답변가능성 게이트 G**
  - 위치: 검색 루프(`lfm/branch.py:151-157` [확인]) 뒤, ANSWER 프롬프트 조립(:160) 앞
  - B가 답할 수 없다고 본 질문은 ANSWER에서 빼고 status='insufficient'로 기록한다. 추가 LLM 호출은 0이다.
  - **결정 키는 n_b 분모의 coverage_answered로 고정하고, 게이트가 뺀 질문은 0으로 센다(검사 F4).** 게이트 효과는 토큰·낭비율·off_question으로 주장하고, 통과율로 주장하지 않는다.
- **retrieve-then-decompose는 이번에 하지 않는다.** sub-question이 arm마다 달라져 짝 비교가 깨진다. future work로 둔다.
- **분해기가 쉬운 질문만 만드는 게이밍 방지(F8)**: ρ가 F arm 대비 올라갈 수 있는 상한(+10pp), paraphrase 전후 의미 유사도 하한을 사전등록한다.
- **held-out 유지**: 게이트는 정답 id·키 파일·참조 답안을 읽지 않는다. 같은 full 인덱스를 쓴다. 정답 문서 제목·별칭이 인덱스에 남아 있지 않은지 점검하는 단계를 둔다[추정, 확인 필요].
- `perspectives.py:39`의 `extra = ""`는 config 훅이 아니라 **하드코딩된 자리표시자**다[확인]. 쓰려면 코드를 고쳐야 한다.

---

## 4. verifier · sufficient · 낭비율

- 규칙과 부등호는 바꾸지 않는다: verifier `≥`(`verifier.py:42`), sufficient `>`(`evidence.py:122-129`) [확인]. τ = 0.80 고정, 민감도 {0.7, 0.9}.
- config에 `verify.coverage_key: coverage | coverage_answered | coverage_answerable`를 추가한다. **기본값 `coverage`**이므로 기존 재현이 그대로 유지된다.
- 낭비율 W는 정의를 그대로 두고 술어만 바꿔 W(L)과 W(A)를 나란히 낸다. W(A) ≤ W(L)이다. 이것은 "재정의가 낭비율을 키우지 않는다"는 **보조 문장**으로만 쓴다. R3의 "8.5% vs 20%"에는 따로 답한다(검사 F13).

---

## 5. 0원 소급 분석 (D18) — 다음에 할 일

API $0, GPU 약 1시간 미만[추정].

| # | 작업 | 산출 |
|---|---|---|
| 0 | `PREREG_D18.md` 커밋. 명칭은 **"post-hoc, label-held-out registration"**. 이미 본 결과(Test A 집계 전체, D15~D17, H1)를 목록으로 적는다(검사 F2) | 해시 |
| 0′ | 캐시·인덱스 백업과 MANIFEST.sha256 (둘 다 이 PC에만 있다) | MANIFEST |
| 1 | Test A 캐시 재생으로 텍스트 복원: 질문(paraphrase 포함), 답, 인용, 청크. **R_i는 sub-question별로 재검색해서 따로 구한다(F7: `seen` 중복 제거 때문에 trace에서 복원 불가).** 재검색 결과가 원래 chunk_map과 같은지 assert | `subq_text.jsonl` |
| 1′ | assert: API 0, 캐시 미스 0, degraded 600/600, legacy coverage·grounding이 `waste_ratio.csv`와 일치 | 로그 |
| 2 | H2 라벨 시트(dev 43 / held-out) 생성. 키 봉인 | html |
| 3 | dev 채점 → θ 고정 → **커밋** | `operator_dev.json` |
| 4 | 전체 채점과 재집계 | `D18_RESULT.md` |
| 5 | held-out 관문 (라벨을 채점 전에 커밋) | `gate.json` |

**D18_RESULT 표**
- 변형 {L, A, U}별 통과 수(절별), coverage 분포, sufficient, W, off_question, ρ
- 215/219 branch를 세 방식으로 처리한 민감도(R3 "contaminates")
- 불통과 사유 귀속: 형식 실패 → 답할 수 없음 → 오프토픽 → 근거 비지지
- **절감 비용 귀속은 E arm과 비교해야 의미가 있으므로 재실험 단계에서 낸다(F11)**

**예측** (방향만, 결과와 무관하게 보고)
- 통과율은 0/50 근처를 유지한다.
- W(A) ≤ W(L).
- off_question은 무시할 수 없는 크기다.
- controversy의 ρ가 가장 낮다.

**해석 한계**: 이것은 캐시 산출물의 **retrospective re-scoring**이다. 게이트의 실제 효과는 라이브 arm에서만 주장한다.

---

## 6. 재실험 arm (갱신)

선결 조건은 **공저자 재협상**(D1)이다. 모든 arm에 새 키를 그림자 채점으로 기록한다(GPU만, API 증분 0). 결정 키는 arm별로 실행 전에 고정하고, 세 키 기준 통과율을 모두 보고한다. 헤드라인은 35토픽이다.

| arm | 구성 | 결정 키 | 목적 |
|---|---|---|---|
| L | legacy + global retry, s1 | coverage | drift 확인 |
| L-F | legacy + 축소 컨트롤러, s1-3 | coverage | R1-6·R3 라이브 재현 |
| E | 인용·status 명시 + global retry, s1-3 | coverage_answered | R1-8 working baseline |
| F | E + 축소 컨트롤러, s1-3 | coverage_answered | 본 비교 |
| **G-F** | F + 답변가능성 게이트 | coverage_answered (분모 n_b) | 분해 수정의 라이브 효과 |

**E arm 설계 변경 (H1 반영)**: 기권을 금지하지 않는다. ANSWER 출력에 `status: answered | insufficient`를 받는다. 지금 legacy 프롬프트의 모순("빈 답 허용"과 "2 to 5 sentences")을 이 방식으로 푼다. legacy 프롬프트는 캐시 보존을 위해 그대로 둔다.

**추가 관문**
- G5: 게이트가 뺀 비율이 D18 예측(1 − ρ) ±10pp 안이다.
- G6: G-F의 입력 토큰이 F보다 늘지 않는다.

---

## 7. 코드 변경 (제안 — 아직 아무것도 고치지 않음)

- `lfm/evidence.py`: legacy 함수는 그대로 둔다. 추가할 것은 `coverage_answered`, `coverage_answerable`, `sufficient_answered`, 부속 지표다(점수 dict 주입).
- `lfm/relevance.py` (신규): reranker / NLI 템플릿 / MiniCheck 대조 scorer. 점수 디스크 캐시, 리비전 고정.
- `lfm/verifier.py`: `coverage_key` 분기(기본 legacy).
- `lfm/types.py`: SubAnswer에 `status`, `answerability`(None이면 직렬화 생략 → legacy 바이트 일치).
- `lfm/branch.py`: `decompose_mode == "evidence_gated"`일 때만 게이트. **청크 출처를 질문 id 목록으로 기록**(F7).
- `config.yaml`: `relevance.*`, `verify.coverage_key`, `decompose_mode`. 기본값은 모두 legacy.
- 실험 스크립트: `d18_textdump.py`, `make_h2_sheet.py`, `d18_operator_dev.py`, `d18_score.py`, `d18_retro.py`, `d18_gate.py`
- 테스트 `tests/test_coverage_variants.py`
  - legacy 바이트 일치
  - cov_A ≤ cov_L
  - 전부 기권 → 0
  - status 남발 → 이득 없음
  - 오프토픽 → 0
  - degenerate
  - 기존 `verify_numbers.py`가 무수정 통과

---

## 8. 사람 라벨 (최소화)

| 세트 | 용도 | 건수 | 보는 것 | 시간 [추정] | 누가 |
|---|---|---|---|---|---|
| H1 (완료) | B dev | 86 | — | 0 | — |
| **H2-dev** | A dev | 43 | 질문 + 답 | 1시간 | 본인 |
| **H2-ho-A** | A 검증 | 100 | 질문 + 답 | 2시간 | 본인 |
| **H2-ho-B** | B 검증 | 100 | 질문 + 청크 | 2.5시간 | 본인 |
| R | intra-rater κ | 50 (2주 뒤 재라벨) | 같음 | 1시간 | 본인 |
| **2nd annotator** | inter-rater | **H2-ho 중 50건 이상 + V′** | 같음 | 2~3시간 | **본인 아닌 사람 — 필수(F6)** |

2nd annotator를 끝내 구하지 못하면 응답서의 R1-3을 **"미충족"**으로 명시하고 한계 1순위로 둔다. intra-rater는 inter-rater를 대체하지 못한다.

---

## 9. 리뷰어 대응 (요약)

| 항목 | 이 스펙의 대응 | 남는 것 |
|---|---|---|
| R1-2 | sub-question 단위 국소화는 미측정이라고 명시한다(`EvidenceView(` 생성자 호출 없음 [확인]). G-F가 처음으로 sub-question 단위 status를 실측한다 | EXPAND 효과 |
| R1-3 / R2 | 층 가중, 군집 CI, κ | **2nd annotator** |
| R1-6 | L-F/E/F/G-F × 3 seed | 공저자 승인 |
| R1-7 | off_question, cov_A로 **부분 대응** | 블라인드 품질 평가 + TOST는 별도 트랙(F12) |
| R1-8 | 불통과 사유 귀속, E arm, 절감 귀속은 E 대비 | — |
| R1-9 | 술어 4개를 수식으로 공시, cov_A ≤ cov_L 명제 | minimal / lower-bound 주장은 별도 |
| R1-10 | PREREG 해시, 스크립트, 라벨 지침, MANIFEST, 리비전 공개 | — |
| R3 (215 오염 / 0.80 보정) | 세 방식 민감도, 사람 라벨로 본 기권 정밀도, τ 민감도. 0/150과 6/150 병기 | — |
| R3 (8.5% vs 20%) | 차이의 원인 설명과 seed 간 분산 | — |
| R3 (second-generator, 저장소 공개, 표 번호) | 원고 수정과 공개 작업 | — |

---

## 10. 결정할 것

1. **이 스펙대로 D18(0원 소급 분석)을 시작할지.** 저장소에 코드가 추가된다.
2. **H2 라벨 약 290건(약 6.5시간)을 할지.** 하지 않으면 연산자를 held-out으로 검증할 수 없으므로 coverage_answered는 민감도 지표로만 쓴다.
3. **2nd annotator를 구할 수 있는지.**
4. **재실험(공저자 재협상).**
