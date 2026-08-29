# LFM M1 Test A — 인수인계 (Claude 채팅용 단일 문서)

이 문서 하나로 맥락이 서도록 썼다. 저장소 접근 없이 읽을 수 있다.
작성 2026-08-29 · 저장소 `C:\Users\skyop\Desktop\나의 rag\LFM\m1\` · 커밋 9개

---

# 0. 세 줄 요약

1. **Test A 완료.** τ=0.8 낭비율 **0.5%** [95% CI 0.0–1.5], 재실행 branch 400개 중 2개.
2. **그런데 50개 질의 중 verify 통과가 0개다.** 낭비율이 낮은 건 "재실행이 낭비가
   아니어서"가 아니라 **"충분한 branch 자체가 없어서"** 다. 측정이 유효 영역에
   도달하지 못했다.
3. **다음 결정은 사람 몫이다** — 전제 폐기 / 난이도 조정 후 재측정 / 진단 먼저.
   마스터플랜 §9가 연구 질문 교체를 사람 판단으로 규정한다.

---

# 1. 무엇을 만들었나

논문 *Localize, Freeze, Mutate: Evidence-Driven Partial Re-execution for
Multi-Perspective RAG* 의 M1 마일스톤. **Test A 하나만** 구현했다 —
"global retry가 실패 시점에 이미 충분했던 branch를 얼마나 낭비 재실행하는가".

M2 금지 항목(relation.py, LFM/oracle/random controller, dense retrieval, 웹 검색,
Test B~E)은 손대지 않았다.

## 파이프라인

```
질의 → 관점 분해(4개 고정 role, LLM 1콜)
     → branch 4개 병렬: BM25 검색 → 인용 필수 답안 생성 → claim 분해
     → evidence 4지표(NLI) → 전역 verify
     → 실패면 GlobalRetry: 전 branch 재실행 (최대 3라운드)
```

- **관점 role 고정**: overview_factual / technical_mechanism / impact_application /
  controversy_limitation
- **관점 분해는 라운드 0에서 1회만.** 라운드마다 다시 뽑으면 branch 정체성이 바뀌어
  "같은 branch를 다시 돌렸다"는 낭비율의 전제가 무너진다.
- **라운드 교란 3층**: top_k ×{1,2,3} / sub-question paraphrase / retry directive.
  sampling 파라미터(temperature·seed)에 기대지 않는다 — 모델별 지원이 다르다.
- **캐시 키** = sha256(canonical_json([model, system, prompt, max_tokens,
  reasoning_effort, cache_nonce])). 라운드 r>0은 `cache_nonce="r{r}"`.
  이게 없으면 캐시가 라운드 1 응답을 돌려줘 3라운드가 동일 복제가 되고
  낭비율은 측정이 아니라 항등식이 된다.

## 지표 정의

| 지표 | 정의 |
|---|---|
| coverage | 인용이 붙은 sub-question 비율 (결정론) |
| grounding | 답안 문장 중 인용 청크가 entail하는 비율 (NLI, τ_ground=0.5) |
| source_diversity | `min(1, H(p)/log K)`, K=5, 출처 문서 기준, 단일 소스 → 0 |
| support_strength | claim별 최대 entailment 확률의 평균 |
| **sufficient(b)** | `grounding(b) > τ AND coverage(b) > τ` — **엄격 부등호** |
| **verify 통과** | `mean_b grounding ≥ 0.80 AND min_b coverage ≥ 0.80` (비대칭, 의도된 것) |
| **낭비율** | (재실행된 branch 중 **재실행 시점에** sufficient였던 수) / (재실행 총수) |

---

# 2. 데이터

| 항목 | 값 |
|---|---|
| 출처 | HF `wikimedia/structured-wikipedia`, config `enwiki_namespace_0` (영어판) |
| revision (고정) | `417c267bb457fa645c22eb3b5c77764963194c70` (2026-05-19) |
| 라이선스 | CC BY-SA 4.0 — Wikimedia Enterprise / Wikipedia contributors |
| 상류 규모 | 86 샤드 / 37.2 GB |
| 코퍼스 | 150,000행 / **유니크 142,972** / 293→1,231 MB |
| SHA256 | `7fa93410a5e301b9ca48048c396acff03daa1f7c0ff047c6fa93e4280e308816` |
| 표집 | 86 샤드 전부에 HTTP range read, 시드 고정 층화 (샤드당 row group 3개) |
| 다운로드 | 975.9초, 부족분 0 |

## 필드 판정 (schema_check.py, PASS)

| 필드 | 채움률 |
|---|---|
| identifier / name / url / abstract / sections | 1.0000 |
| references | 0.9281 → `citation_mode = "references"` |
| description / infoboxes / image / tables | 0.8880 / 0.7562 / 0.4643 / 0.3234 |
| `version.scores.referencerisk` | 0.9073 |
| `version.scores.referenceneed` | 0.8838 |
| `version.scores.revertrisk` | 0.9799 |
| **redirect 계열** | **부재** → `heldout_match_mode = "title_normalization"` |

## 발견 3건 (M2 설계에 영향)

**① referencerisk의 샤드 간 이질성.** 샤드 0만 보면 채움률 0.47%, 전 코퍼스로는 90.7%.
row group 1개씩 직접 재본 값 — 샤드 0: 0.0005 / 1: 0.0073 / 2: 0.0044 / 40: 0.9912 /
85: 0.9912. **저번호 샤드에만 점수가 비어 있다.** 단일 샤드로 코퍼스를 만들었다면
M2 Test C-2(referencerisk 상위 문서로 자연 conflict 선별)가 불가능하다고 잘못
결론냈을 것이다. → **Test C-2는 이 스냅샷에서 성립한다.**

**② 코퍼스 중복 4.69%.** 상류 샤드가 서로 겹쳐 150,000행 중 7,028행이 중복이다
(`2026 United Kingdom local elections`는 44행). 인덱스 빌드에서 identifier 기준
첫 행만 남긴다. 안 걸렀으면 `chunk_id`가 충돌하고 같은 본문이 검색 상위를 중복 점유했다.

**③ redirect 필드가 스키마에 아예 없다.** SPEC v1.1 §2.2가 예상한 폴백대로
title 정규화 매칭(NFKD → 발음구별기호 제거 → 괄호 한정어 제거 → 선행 관사 제거)을 쓴다.
**redirect로만 잡히는 별칭 문서는 코퍼스에 남아 있을 수 있다. 규모는 측정하지 않았다.**

---

# 3. 토픽 50개

## 선별을 두 번 짰다 (첫 판은 폐기)

**첫 판.** 용어별 포스팅을 만들고 infobox 유형 수로 정렬 → **실패.**
유형 수는 교차 도메인성이 아니라 **흔함의 대리 지표**였다. 상위 후보가
`eight`(관련 564) · `open` · `point` · `came` · `make` · `days` · `date`로 채워졌고,
target article도 무관했다 — 용어 `date` → 문서 `Hideo Date`. 생성된 질문:

> *"How did an eight-car endurance racing event shape championship outcomes,
> technical strategies, safety protocols, and historical controversies?"*

**둘째 판(채택).** 순서를 뒤집었다.
1. target 후보 = abstract ≥600자, 본문 ≥6,000자, 참고문헌 ≥20개, infobox 보유.
   disambiguation / list of / index of 제외
2. 변별력 용어 = 제목+abstract 용어 중 df ∈ [20, 3000]인 것을 idf 상위 5개
3. 이웃 = 그 용어 중 2개 이상을 담은 문서
4. 이웃 ≥30, 이웃의 infobox 유형 ≥3종
5. **주 infobox 유형당 상한 3개** (축구선수 독식 방지)
6. 정렬 키는 유형 수가 아니라 **관련 문서 수**

결과: 후보 60개, 이웃 **142~241 문서**, infobox 유형 **8~88종**.
첫 판을 무너뜨린 기능어들은 stopword로 막고 회귀 테스트로 고정했다.

## 승인 게이트 — 검사가 게이트다

사용자가 승인을 위임했으므로 `approve_topics.py`로 기계 검증을 수행했다.
HANDOFF §4 경고("대충 승인하면 낭비율이 토픽 난이도의 산물이 된다") 때문이다.

**승인 50 / 탈락 4** — 전부 동일 계열 중복(Eurovision 리투아니아 연도별 3건,
Welsh Labour 3건). 승인 경위·검사 항목·탈락 사유는 `topics_pilot50.yaml`에 있다.

**철회한 기준 1건.** 처음엔 "토픽 문구가 target 제목을 재현하면 held-out 누수"로
10건을 탈락시켜 승인 48 < 50으로 게이트가 FAIL을 냈다. **잘못된 판정이었다.**
held-out의 통제 수단은 target article을 인덱스에서 빼는 것이지 질문에서 이름을
숨기는 것이 아니다. 게다가 제목이 한 단어면(`Bharath (actor)` → `bharath`)
주제를 언급만 해도 비율이 1.0이 되어 지표가 무의미해진다. `title_overlap`으로 기록만 한다.

## 토픽 예시 (승인 50개 중 5개)

| id | target (held-out) | 이웃 | 유형 | 질의 |
|---|---|---|---|---|
| t000 | The Beginning After the End | 241 | 19 | How did a long-running American web novel and illustrated webtoon evolve across serialization platforms, adaptations, audience reach, and criticisms? |
| t001 | Bosnia and Herzegovina national football team | 236 | 88 | How has the national men's football side from Bosnia and Herzegovina developed competitively, administratively, and politically since Yugoslav-era participation? |
| t003 | Wang Chuqin | 199 | 75 | How has the Chinese world-ranked table tennis champion progressed technically, influenced national team dynamics, commercial impact, and faced performance scrutiny? |
| t004 | Abitibi Power and Paper Company | 196 | 73 | How did the Montreal-based forest products firm grow through acquisitions and mergers, influence newsprint markets, and confront industry decline controversies? |
| t008 | Newton N. Minow | 160 | 39 | How did the prominent US communications regulator shape broadcasting policy, legal practice, civic engagement, and provoke debates over media standards? |

---

# 4. 인덱스

| 항목 | 값 |
|---|---|
| 인덱싱 문서 | 142,922 |
| held-out 제외 | 55행 (토픽 50개, 중복 행 포함) |
| 중복 제외 | 7,023행 |
| 청크 | **685,683** |
| 평균 청크 토큰 | 214.15 (총 146,838,655) |
| 인덱스 크기 | 1,304 MB |
| 청킹 / 빌드 | 247.7초 / 85.2초 |
| 백엔드 | `bm25s` (sparse). dense 인덱스 없음 |
| 토크나이저 | `o200k_base` |
| chunk_id | `d{doc_id}s{section_idx}p{part}` — 재빌드해도 동일 |

**커버리지 게이트는 실질적으로 작동하지 않았다 (한계).** 50토픽 전부 "60"으로 통과했는데,
검색 상한 k가 `max(min_chunks_per_topic*3, 60) = 60`이라 **전 토픽이 상한에 붙었다.**
즉 "≥20개"는 통과했지만 토픽 간 변별이 전혀 없었다. 재측정 시 k를 크게 잡아야 한다.

---

# 5. 테스트 — 205개 전부 통과

`C:\Users\skyop\anaconda3\python.exe -m pytest` → **205 passed in 4.10s**, 실패 0.

| 파일 | 개수 | 무엇을 지키는가 |
|---|---|---|
| `test_approve_topics.py` | 10 | 승인 기준(이웃 수·유형·target 실재·계열 중복), title_overlap이 탈락 사유가 아님 |
| `test_chunk_index.py` | 13 | 섹션 순회·참고문헌 절 제외·토큰 분할, **중복 행 1회만 인덱싱**, chunk_id 안정성 |
| `test_config.py` | 10 | SPEC §7 필수 키, 고정값(τ=0.80 등), **한국어 UTF-8 왕복**, int/str 키 |
| `test_download.py` | 7 | 할당량 배분, revision 고정, 샤드 번호 정렬, **시드 재현성** |
| `test_evidence.py` | 20 | **손계산 고정 케이스** — 단일소스=0, 2소스=ln2/ln5=0.4306765581, 5소스=1.0, 엄격 부등호 |
| `test_heldout.py` | 15 | 제목 정규화 7케이스, 발음구별기호, **approved:true 없으면 에러 종료** |
| `test_llm.py` | 19 | 캐시 히트/미스, **문자열 연결 충돌 방지**, nonce 분기, reasoning_effort가 캐시 키에 포함, 재시도, provider 게이트 |
| `test_pipeline.py` | 24 | 관점 4개 강제, **라운드 2 프롬프트 해시 ≠ 라운드 1**, 절단 재시도, verify 비대칭, GlobalRetry 전량 재실행, **병렬=직렬 동일**, 검색 앵커 |
| `test_schema_check.py` | 13 | 존재/채움률 2층, 빈 필드 게이트 FAIL, credibility 희소성 판정, redirect 폴백 |
| `test_test_a.py` | 22 | **재실행 시점 기준 sufficiency**, 낭비율 손계산, **쿼리 단위 bootstrap**, 판정 밴드 8케이스 |
| `test_topics.py` | 11 | **첫 판을 무너뜨린 기능어 회귀 방지**, idf 선별, 유형당 상한, thin target 제외 |
| `test_trace.py` | 12 | 동결 스키마 8키, reward 항상 null, M2 필드 존재, 상태 라벨 |
| `test_trace_validate.py` | 25 | 키 누락 14케이스, 여분 필드 허용, reward 비-null 실패, 잘못된 상태 라벨 |
| `test_waste_supplement.py` | 4 | 파일럿 제외 추정 |

## 통합 검증

| 검증 | 결과 |
|---|---|
| trace 동결 스키마 (본실험) | **150 records, 0 errors** |
| trace 동결 스키마 (파일럿 3종) | 각 15 records, 0 errors |
| **결정성** (동일 config 2회) | **154/154 캐시 히트, 청구 토큰 0, CSV 완전 동일** |
| 모델 검증 (SPEC §1, 실 API) | `gpt-5-mini` 응답 `'ok'`, in/out 13/10 |

---

# 6. 본실험 결과 (핵심)

`results/test_a/20260829-041445/` · 50토픽 · 67.8분

## 6.1 낭비율

| τ | 낭비율 | 95% CI | 낭비 / 재실행 |
|---|---|---|---|
| 0.70 | 1.8% | 0.0–4.3% | 7 / 400 |
| **0.80 (primary)** | **0.5%** | **0.0–1.5%** | **2 / 400** |
| 0.90 | 0.5% | 0.0–1.5% | 2 / 400 |

τ_global = τ_c_global = 0.80. CI는 실패 쿼리 집합에서 **쿼리 단위** 복원추출 1,000회
(seed 42, LLM 재호출 0).

마스터플랜 §6 밴드로는 **"20% 미만 → 전제 오류, 연구 질문 재설계"**.
**그러나 아래 6.2를 읽기 전에 이 판정을 받아들이면 안 된다.**

## 6.2 왜 이걸 전제 반증으로 읽으면 안 되는가

**50개 질의 중 verify 통과 0개.**

| 지표 | 값 |
|---|---|
| branch-round | 600 (50 × 3라운드 × 4관점) |
| **verify 통과 질의** | **0 / 50** |
| coverage 중앙값 / 평균 | 0.40 / 0.427 |
| **grounding 중앙값 / 평균** | **0.00** / 0.173 |
| coverage = 1.0 branch | 128 (21.3%) |
| grounding > 0.8 branch | 30 (5.0%) |
| **둘 다 만족** | **3 (0.5%)** |
| 답안이 빈 branch | 215 (35.8%) |
| source_diversity 평균 | 0.215 |
| support_strength 평균 | 0.231 |

낭비율이 낮은 원인은 둘로 갈린다 — ① 전역 재실행이 실제로 낭비가 아니거나,
② **애초에 충분한 branch가 없어서 낭비할 것이 없거나.** 이번은 명백히 ②다.

Test A의 전제("일부 branch는 이미 충분한데 전역 재실행이 그것까지 다시 돌린다")는
branch 품질에 **편차**가 있어야 성립한다. 이 설정은 그 영역에 도달하지 못했다.

**파이프라인 결함은 아니다.** answer 600회 중 절단 2회, 파싱 실패 1회.
`claims` 호출 385 = 600 − 215로 빈 답안 수와 정확히 일치 → 215개는 **근거가 없어
모델이 정당하게 답을 비운 것**이다(프롬프트가 그렇게 지시).

## 6.3 τ 미세 스윕 — 절벽이 아니라 연속

| τ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| 낭비율 | 36.8% | 29.8% | 20.8% | 14.8% | 5.0% | 4.2% | 1.8% | 0.5% | 0.5% |
| n | 147 | 119 | 83 | 59 | 20 | 17 | 7 | 2 | 2 |

임계 근처 인공적 절벽이 아니라 품질 분포가 낮은 쪽에 쏠린 결과다.

**coverage는 양자화돼 있다.** sub-question이 3~5개이므로 `coverage > 0.8`은
사실상 `coverage = 1.0`을 요구한다(4/5 = 0.8은 초과가 아니다).
τ를 연속량처럼 읽으면 안 된다.

## 6.4 재실행이 품질을 **떨어뜨렸다**

| 라운드 | branch | coverage | grounding | cov=1.0 | grd>0.8 | 빈 답안 |
|---|---|---|---|---|---|---|
| 0 | 200 | 0.461 | 0.200 | 51 | 12 | 67 |
| 1 | 200 | 0.418 | 0.161 | 37 | 9 | 74 |
| 2 | 200 | 0.401 | 0.158 | 40 | 9 | 74 |

**라운드 0 → 2: coverage −0.060, grounding −0.042.** 둘 다 하락.
top_k 5→10→15, sub-question 재표현, retry directive를 전부 적용했는데도 그렇다.
전체 토큰의 약 2/3를 라운드 1~2에 쓰고 통과는 여전히 0건.

낭비율과 **독립적인 사실**이고 같은 방향을 가리킨다 — 이 설정의 GlobalRetry는
**건질 것도 버릴 것도 없었다.** (하락 원인은 측정하지 않았다. paraphrase가 더 나쁜
질의를 만들었을 가능성, top_k 확대가 잡음을 늘렸을 가능성 — **둘 다 추정이다.**)

## 6.5 관점별 지표

| 관점 | n | coverage | grounding | src_div | support | cov=1.0 | grd>0.8 | 빈답안 |
|---|---|---|---|---|---|---|---|---|
| overview_factual | 150 | 0.359 | 0.147 | 0.116 | 0.207 | 27 | 5 | 69 |
| technical_mechanism | 150 | **0.589** | **0.210** | 0.312 | 0.279 | **56** | 9 | **31** |
| impact_application | 150 | 0.428 | 0.185 | 0.227 | 0.248 | 25 | 8 | 47 |
| controversy_limitation | 150 | 0.331 | 0.150 | 0.206 | 0.191 | 20 | 8 | 68 |

`technical_mechanism`이 일관되게 가장 높고 `controversy_limitation`이 가장 낮다.
**관점 간 편차는 존재한다** — 다만 전부 임계 아래에 있다. 이건 6.2의 "편차가 없다"와
모순되지 않는다: 편차가 있어도 **전원이 문턱 아래**면 sufficient는 0이다.
(이 관점 순서가 일반적인지는 검정하지 않았다. n=50 단일 실행이다.)

## 6.6 낭비가 실제로 발생한 쿼리

- τ=0.8: **t025 하나** (재실행 8개 중 2개가 낭비)
- τ=0.7: t021 (2/8), t025 (4/8), t039 (1/8)

50개 중 1~3개 쿼리만 기여한다. CI 하한이 0인 이유다.

## 6.7 파일럿 오염 점검 — 결과를 밀지 않았다

`t000`~`t004`로 프롬프트·검색을 교정했고 그 5개가 본실험에도 있다.

| τ | 전체 50토픽 | 제외 후 45토픽 | 차이 |
|---|---|---|---|
| 0.70 | 1.8% | 1.9% | +0.2%p |
| 0.80 | 0.5% | 0.6% | +0.1%p |
| 0.90 | 0.5% | 0.6% | +0.1%p |

CI 폭 대비 무시할 수준이다.

---

# 7. 실행 중 고친 측정 결함 2건 (중요)

첫 파일럿이 전 τ에서 낭비율 0.0%를 뱉었다. 그대로 "전제 오류"로 보고할 뻔했는데
원자료를 열어보니 **결함**이었다.

## ① 생성기가 세계가 아니라 코퍼스를 서술했다

실제 문장-청크 쌍을 NLI에 직접 넣어본 결과:

```
문장:  CPs acquire web novels and adapt them into various formats and generate
       revenue through serialized publication, indicating CPs and platforms...
청크:  The main participants in the web novel industry are platforms, content
       providers (CPs), and writers. Platforms distribute web novels to readers...
max_entailment = 0.000
```

청크가 문장을 **명백히 뒷받침하는데도 0.000**이다. 답안이
`The provided context identifies...` / `The context does not list...` 같은
**코퍼스에 대한 메타 진술**로 쓰였기 때문 — 가설이 세계가 아니라 코퍼스를 서술하면
그 코퍼스로 entail될 수 없다. 원인은 내 프롬프트("근거가 없으면 있는 만큼만 쓰라")였다.

직접 사실 진술만 쓰고 근거가 없으면 문장을 줄이라고 고쳤다.

## ② BM25가 주제 신호 없이 일반 명사에 끌려갔다

| sub-question | 상위 검색 결과 |
|---|---|
| 웹툰 연재 플랫폼·기간 | Burgess Hill railway station / Crouch Hill railway station |
| 웹툰 파일 형식·레이아웃 | List of web archiving initiatives |
| 축구협회 지배구조 | Euroscepticism |
| 플랫폼별 콘텐츠 관리 도구 | Job safety analysis |

토픽의 변별 용어를 검색 앵커로 질의 앞에 붙였다. 앵커는 라운드·조건 무관 고정이므로
retrieval은 여전히 전 조건 공통 고정 변인이다.

## 효과

| 파일럿 | 변경 | grounding | coverage | cov=1.0 |
|---|---|---|---|---|
| `_pilot5` | (없음) | 0.097 | 0.983 | 59/60 |
| `_pilot5b` | 메타 진술 금지 | 0.292 | 0.468 | 11/60 |
| `_pilot5c` | + 검색 앵커 | 0.307 | 0.561 | 16/60 |

**coverage 0.98 → 0.47은 붕괴가 아니라 거품이 빠진 것이다.** 0.98은
`"맥락에 없다 [c1]"` 같은 문장도 인용으로 세던 값이었다.

**여기서 튜닝을 멈췄다.** 두 수정 모두 들여다본 결함이 근거이지 출력 숫자가 근거가
아니었다. 파일럿 3회는 전부 보존돼 있다(`results/test_a/_pilot5{,b,c}/`).

---

# 8. 비용

| 실행 | 호출 | 캐시 | 청구 입력 | 청구 출력 |
|---|---|---|---|---|
| P2 토픽 후보 (30k 폐기분) | 6 | 0 | 4,864 | 5,460 |
| P2 토픽 후보 (150k 채택분) | 6 | 0 | 15,956 | 4,610 |
| 파일럿 `_pilot5` | 165 | 67 | 133,808 | 67,860 |
| 파일럿 `_pilot5b` | 151 | 45 | 207,926 | 65,668 |
| 파일럿 `_pilot5c` | 154 | 48 | 207,157 | 75,943 |
| 결정성 재실행 | 154 | 154 | 0 | 0 |
| **본실험 50토픽** | **1,439** | **154** | **1,882,035** | **697,798** |
| **합계** | **2,075** | **468** | **2,451,746** | **917,339** |

본실험 태그별: `answer` 600 · `paraphrase` 400 · `claims` 385 · `decompose` 50 ·
`answer_retry` 2 · `answer_repair` 1 · `paraphrase_repair` 1.
쿼리당 평균 56,953 토큰 (최다 t036 84,740 / 최소 t037 43,160).

**달러 환산은 하지 않았다.** 마스터플랜 §6이 "입·출력 단가는 config 작성 시 확정해
기재"라고 했는데 그 항목이 채워진 적이 없다. 역산은 지어내는 것이라 토큰만 보고한다.

**출력 토큰에는 reasoning 토큰이 포함된다.** `gpt-5-mini`는 추론 모델이고,
기본값(medium)에서 출력의 84%가 reasoning이며 가시 답안이 630자로 눌린다.
`reasoning_effort: low`로 고정했다(전 조건 공통).

| effort | 출력 토큰 | reasoning | 가시 답안 |
|---|---|---|---|
| minimal | 425 | 0 | 2,029자 |
| **low** (채택) | 566 | 256 | 1,447자 |
| medium (기본) | 902 | 768 | 630자 |

**비용 귀속 주의**: `OPENAI_API_KEY`를 「샤딩 mcp를 통한 rag」 프로젝트와 공유한다
(2026-08-29부터). **대시보드 청구액은 두 프로젝트가 섞인 값이다.** LFM 비용의 1차
근거는 `usage.json`의 `billed_tokens_*`(캐시 히트 제외)다.

---

# 9. Threats to Validity

1. **임계 종속성.** 낭비율은 τ와 τ_global의 상대 위치에 직접 좌우된다. 두 값을 항상 병기.
2. **coverage 양자화.** sub-question 3~5개 → `coverage > 0.8`은 사실상 `= 1.0`.
3. **코퍼스가 위키백과의 ~2%.** 어떤 주제든 이웃이 성기고, held-out으로 target까지
   빼면 더 그렇다. 30,000건에서는 교차 도메인 후보가 28개뿐이라 실험이 성립하지 않았다.
   150,000은 그 하한을 넘긴 값이지 **충분성이 검증된 값이 아니다.**
4. **redirect 부재.** 별칭 문서가 코퍼스에 남아 있을 수 있다. 규모 미측정.
5. **NLI 미검증.** 통제 쌍은 정상(entail 0.997 / contradiction 1.000, `id2label`을
   모델 config에서 읽음)이지만, **사람 판단과의 일치도는 측정하지 않았다.**
   `tau_ground = 0.5`는 스펙 기본값이며 캘리브레이션하지 않았다.
   전제가 청크 전체(평균 214토큰)라 512토큰에서 절단된다.
6. **커버리지 게이트가 변별하지 못했다.** 전 토픽이 검색 상한 60에 붙어 통과했다.
7. **파일럿 오염** — 6.7에서 정량화(+0.1%p).
8. **다중 비교 미보정.** τ 3개는 같은 데이터의 재라벨링이라 독립 검정이 아니다.
   판정은 `primary_tau = 0.8` 하나로만.
9. **범위.** GlobalRetry 하나만 측정했다. "전역 재실행이 낭비인가"에만 답하고
   "국소 재실행이 더 나은가"에는 답하지 않는다.
10. **사람 평가·LLM-judge 없음.** M1 전 과정에 없다.

---

# 10. 다음 결정 — 사람 몫

마스터플랜 §9는 낭비율 <20%를 연구 질문 교체 신호로 규정하고, §0은 go/no-go를
사람 판단으로 둔다. (§6 밴드 자체는 논문 미노출 항목이다.)

**A. 전제 폐기** — 밴드대로 연구 질문 재설계.
단, verify 통과가 0건이면 "낭비"의 모집단 자체가 비어서 이 결론을 지지하기 어렵다.

**B. 난이도 조정 후 재측정** — 무엇을 바꾸든 **낭비율을 보고 고르면 안 되고**,
바꾸기 전에 정해 한 번만 돌려야 한다.

| 조건 | 현재 | 대안 | 근거 |
|---|---|---|---|
| 코퍼스 | 142,972 (~2%) | 50만~ | 이웃이 성기다 |
| held-out | target 제거 | 유지하되 이웃 밀도 높은 토픽만 | 가장 관련 깊은 문서를 빼면 남는 게 없다 |
| τ_global | 0.80 | 0.5~0.6 | 현재 임계에서 통과 0건 |
| 검색 | BM25 top-5 | top-10~20 | 관점당 최대 25청크뿐 |
| 커버리지 게이트 k | 60 (상한 포화) | 300+ | 변별력이 없었다 |

**C (권장). 진단 먼저** — 낭비율을 다시 재기 전에 "branch 품질에 편차가 생기는 설정"을
소규모로 찾는다. 지표는 낭비율이 아니라 **`coverage=1.0 & grounding>0.8` branch 비율**
(현재 **0.5%**). 이 값이 최소 10~20%는 나와야 낭비율이 의미를 갖는다.

어느 쪽이든 **이번 실행 결과를 덮어쓰지 않는다.** 재측정은 새 결과 디렉토리로.

---

# 11. 저장소 지도

```
LFM/m1/
├ CLAUDE.md, SPEC_M1_TestA_v1.1.md   # 우선 문서 (마스터플랜은 의도적으로 저장소 밖)
├ P1_REPORT.md P2_P3_REPORT.md P4_REPORT.md
├ MODEL_NOTES.md VALIDITY.md REPRODUCE.md README.md
├ config.yaml                        # 모든 파라미터. 코드 하드코딩 없음
├ lfm/    llm trace config types jsonio nli evidence verifier
│         perspectives branch pipeline retriever graph controller/
├ data/   download schema_check topics approve_topics heldout chunk_index
│         MANIFEST.json topics_pilot50.yaml heldout_ids.json index_stats.json
├ experiments/  test_a trace_validate model_check waste_supplement
├ tests/  205개
└ results/test_a/
   ├ 20260829-041445/   # 본실험: waste_ratio.csv, waste_ratio_ci.csv,
   │                    #   waste_ratio_supplement.csv, report.md, trace.jsonl(150),
   │                    #   usage.json, results.json, waste_ratio.png, config.yaml
   └ _pilot5{,b,c}/     # 파일럿 3회 — 측정 결함 수정의 증거
```

**실행 순서는 코드가 강제한다**: `chunk_index.py`는 `heldout_ids.json` 없으면 에러 종료,
`heldout.py`는 `approved: true` 없으면 에러 종료.

**환경**: 파이썬은 반드시 `C:\Users\skyop\anaconda3\python.exe`.
그냥 `python`은 Windows 스토어 껍데기라 아무것도 안 하고 exit 0으로 끝난다.
