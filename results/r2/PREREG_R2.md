# PREREG_R2 — Partial Rewind 재실험 사전등록 (post-hoc, label-held-out registration)

> **상태: 동결 (2026-11-05). 첫 API 호출 전에 커밋한다.**
> 커밋 해시를 원고 §V와 응답서에 적는다.
> 커밋한 뒤 이 문서를 바꾸려면 `DEVIATIONS.md`에 날짜, 사유, 영향받는 실행을 기록해야 한다.

설계 원문: `논문통합\03_LFM\심사_2026-11-01\재실험설계_확정안_R2.md`. 이 문서와 설계 원문이 충돌하면 **이 문서가 우선**한다.

---

## 0. 고정 식별자

| 항목 | 값 |
|---|---|
| 코드 | 저장소 `m1`, 브랜치 `r2-impl`, 커밋 `f6dd344` (실행 코드 `lfm/`, `config.yaml`, `experiments/r2_run.py`는 `ab0f4bb`와 동일) |
| 토픽 분할 `data/split_r2.json` | SHA-256 `6993e8cce363105ad230a186165abf89c22b52e2d3a1f08fe55fbe913e8a4d75` |
| `config.yaml` | SHA-256 `d5f1af71bca241e6f317109b3ac72e31ba612999cba63f886b91468988be63ea` (r2-impl 기준) |
| legacy ANSWER_SYSTEM / ANSWER_PROMPT | `9eed32d5…74cd8` / `7aacf26b…2061a4` — **바꾸지 않는다** |
| E ANSWER_SYSTEM_E / ANSWER_PROMPT_E (v0) | `ecb1a9a0…72854d` / `14665f93…79117b` — dev 수정 시 v1, v2로 갱신하고 동결 |
| 생성 모델 | `gpt-5-mini`, reasoning_effort low, max_tokens 2000, temperature 생략 |
| 결정·보고 연산자 | MiniCheck-Flan-T5-Large 단일 문서, p > 0.5 (`estimator.kind=minicheck_doc`) |
| support_strength NLI | `cross-encoder/nli-deberta-v3-base` (config 그대로) |

---

## 1. 이미 본 결과 — 이 등록이 prospective가 아닌 이유

이 등록 전에 다음을 이미 봤다.
- Test A 집계 전체(0/50 통과, W₀.₈ 0.5% / 8.5%)
- D15~D19 결과
- H1 사람 라벨(86), H2 사람 라벨(283)
- V′ 최종(100)

그래서 "prospective"라는 표현을 쓰지 않는다.

**held-out의 범위.** held-out 35토픽은 연산자 선택(V′ = dev15)과 E 프롬프트 문구 조정에 대해서만 held-out이다.
- held-out 라운드 0의 생성물은 이미 H2·D18·D19에서 봤다.
- **E 프롬프트를 만든 동기**(근거가 없으면 비껴간 답을 쓴다)는 held-out H2 관찰에서 나왔다.

**파일럿 토픽 혼입.** held-out 35토픽에는 원 시스템 개발에 쓴 파일럿 토픽 `t000~t004`가 들어 있다(`pick_topics`가 dev 표집에서 이 다섯 개를 뺀다). 모든 헤드라인 수치에 **30토픽(t000~t004 제외) 민감도**를 함께 낸다.

---

## 2. 토픽

| 집합 | 토픽 | 용도 |
|---|---|---|
| pilot5 ⊂ dev15 | t007, t031, t037, t038, t050 | 파일럿과 관문만. 헤드라인에 섞지 않는다 |
| dev15 | split 파일 그대로 | E 프롬프트 수정 재파일럿(seed 9) |
| **heldout35** | split 파일 그대로 | **모든 헤드라인 수치** |

**축소 규칙**(G4 실패 시에만): held-out을 seed 42로 25토픽까지 줄인다. H2 라벨이 있는 토픽을 먼저 남긴다. 축소는 이탈로 기록한다.

---

## 3. arm

| arm | 프롬프트 | 컨트롤러 | 실행 |
|---|---|---|---|
| L-GR | legacy | GlobalRetry | seed 1, 2, 3 |
| L-BR | legacy | FreezeRetry(branch) | seed 1, 2, 3 |
| E-GR | E (status) | GlobalRetry | seed 1, 2, 3 |
| E-SQ | E | FreezeRetry(subq) | seed 1, 2, 3 |
| E-BR | E | FreezeRetry(branch) | seed 1, 2, 3 |
| E-R1 | E | 재시도 없음 = **E-GR의 라운드 0 출력**(실행하지 않음) | seed 1, 2, 3 |
| N (부록) | E, 생성 모델만 gpt-5.4-nano | GR, SQ | seed 1 |

- **실행 순서**(seed마다 고정): L-BR → L-GR → E-SQ → E-BR → E-GR. 그다음 N.
- **관계 모듈은 끈다.** VERIFY/PRUNE은 평가 시스템에서 제외한다. 나오면 pipeline이 AssertionError로 멈춘다.

---

## 4. seed, nonce, 공통 난수

- **nonce.** `s{seed}|{family}|r{round}`이고 family ∈ {L, E}다. sub-question 재실행은 `…|sq`를 붙인다. paraphrase에도 같은 규칙을 적용한다.
- **분해.** nonce 접두사가 없고 Test A 분해 모델로 Test A 캐시(`.cache/llm`, 읽기 전용)를 그대로 재생한다. 캐시에 없으면 새로 분해하지 않고 멈춘다.
- **공통 난수(CRN), 공시한다.**
  - 같은 family와 seed 안에서 BR이 다시 돌리는 branch는 GR의 같은 branch와 바이트가 같다(BR ⊂ GR).
  - SQ 호출은 독립 표본이다.
  - 응답서 문구: "공유는 짝 비교 설계이며, 첫 실행의 모든 생성은 새 표본이다."
- **seed의 정의.** seed는 **생성 단계 반복**이다. 분해는 고정이고 검색은 결정론이다. 헤드라인에 "3 generation-stage replicates"라고 적는다. provider seed 재현성은 주장하지 않는다.
- **E 프롬프트를 수정하면** dev 재파일럿은 seed 9로 한다.

---

## 5. 판정 규칙

| 규칙 | 정의 | 코드 |
|---|---|---|
| τ | τ_g = τ_c = 0.80. **사후 변경 금지.** 민감도 {0.7, 0.9}는 offline 재집계로만 낸다(컨트롤러를 다시 돌리지 않는다) | config |
| sufficient (FREEZE) | g > τ ∧ c > τ, 엄격 부등호 | `evidence.py` `sufficient` |
| verify | mean_g ≥ 0.80 ∧ min_c ≥ 0.80 | `verifier.py` |
| 동치 조건 | n_sub ∈ {3, 4, 5}에서 c > 0.8 ⇔ c = 1.0. 파일럿에서 n_sub 분포를 assert한다 | — |
| coverage | legacy 정의(유효 인용이 붙은 sub-answer 비율). status는 보지 않는다. insufficient·dropped·format_fail은 미커버다 | `evidence.py` `coverage` |
| EXPAND 대상 | 미커버 ∪ MiniCheck 비지지 문장이 있는 sub-question | `evidence.py` `subq_targets` |
| 병합 | 대상 밖 sub-answer와 그 청크는 그대로 물려받는다. claims는 대상 답이 글자 그대로 같을 때만 재사용한다 | `branch.py` |
| 종료 | pipeline의 `range(max_rounds)`와 컨트롤러 STOP. Graph 라운드 상한 assert는 평가 경로 밖이다 | `pipeline.py` |
| 관련성 연산자 | **쓰지 않는다.** D18(reranker)과 D19(nano judge)가 held-out 관문을 탈락했기 때문이다. coverage_answered·answerable은 주장에 쓰지 않는다 | — |

---

## 6. 파일럿 관문 (pilot5 × 5 arm × seed 1)

**결과 지표(절감, 통과 수)로 진행 여부를 정하지 않는다. 효과 관문은 없다.**

| 관문 | 기준 | 실패 시 |
|---|---|---|
| **G0** | 승인: 범위, 비용 상한 $25, 2nd annotator 여부. **2026-11-05 충족** — 교신저자(저자 본인)가 승인했다. 공저자는 원고 검토만 맡는다는 분담에 따라 실험 범위·비용 결정권은 저자에게 있다. 2nd annotator는 없다(§9 C) | 승인이 없으면 실행하지 않는다 |
| **G1 기술** | ① seed 간 키 교집합 0(설계상 CRN 공유 제외) ② decompose api_calls = 0 ③ answer 호출의 (파싱 실패 + finish=length + status 누락) ≤ 1% ④ `_repair` 호출 ≤ 2% ⑤ 남은 문장의 위조 id 0 ⑥ answered 문장 중 drop된 무인용 문장 ≤ 5% ⑦ 원장 합 = usage 합 ⑧ MiniCheck가 cuda에서 단일 조각으로 돈다 ⑨ 물려받은 sub-answer 바이트 동일 100% | 1회 고치고 1회 다시 돈다(이탈 기록) |
| **G2 E 프롬프트** (P-H 사람 라벨) | (a) E insufficient 30건 중 사람 "답있음" ≤ 4건(≤ 15%) (b) E answered의 "답함" 비율이 L answered보다 10pp 이상 높다 (c) sub-question 단위 insufficient 비율 45~85% (d) E arm 평균 legacy coverage ≥ 0.15 | dev에서만 문구를 고친다(최대 2회, dev15 × seed 9). 그래도 실패하면 E를 헤드라인에서 내리고 본 비교를 L-BR vs L-GR로 옮긴다 |
| **G3 SQ 기전** | 라운드 ≥ 1의 비동결 실행 중 대상 < 전체인 것이 20% 이상. SQ의 실행당 입력 토큰 < BR | 병합이 실패하면 E-SQ를 빼고 "sub-question 단위 미평가"로 쓴다 |
| **G4 예산·시간** | 파일럿 실측으로 다시 외삽한 본실험이 1.5 × 추정 이하(gpt-5-mini 15M in / 5.6M out), GPU ≤ 30시간 | 축소 규칙(§2) |
| **G5 표본 크기** | P-H에서 UA 짝 차이 SD를 추정한다. H4 슬롯 수 n = ⌈(1.645 + 0.842)² · SD² / 0.10² · 1.2⌉. 하한 120, 기본 200 | 공식은 고정이다. SD만 파일럿에서 넣는다 |

---

## 7. 본실험

- **범위**: heldout35 × 5 arm × seed 1, 2, 3. 순서는 s1 → s2 → s3, 그다음 N.
- **점검.** seed마다 원장 청구량, `_repair` 비율, `model_returned` 일관성을 본다.
  - 파일럿 외삽의 1.5배를 넘으면 멈추고 공저자에게 보고한다.
  - 같은 실행 안에서 model_returned가 섞이면 그 실행을 다시 돈다.
- **drift.** L-GR s1과 Test A의 같은 35토픽에서 degraded, verify, coverage 차이를 보고한다.
  - ±10pp를 넘으면 drift로 보고하고 계속 진행한다.
  - 결론은 같은 시기 arm끼리만 비교한다. Test A는 역사 기록이다.

---

## 8. 분석

**부트스트랩.** topic × seed 2단 계층, 10,000회. seed별 점추정, seed별 부호표, 일원 랜덤효과 분산 성분을 함께 낸다.

**절감(1차).**
- 비교 셋: E-SQ vs E-GR(본 비교), L-BR vs L-GR(라이브 재현), E-BR vs E-SQ(세분도)
- 지표: 회계 토큰(in + out), 청구 토큰(원장), branch-round 수, sub-question 실행 수
- **절감 주장은 회계 토큰으로 한다.** 공통 난수 때문에 뒤에 도는 arm(GR)은 앞 arm(BR)의 캐시를 맞혀 청구가 작게 나온다. arm별 청구(executed branch billing의 합)는 보고만 하고, 실험 전체 청구 총량은 원장 기준이다. 파일럿 G1에서 branch billing 합과 usage 합(분해 제외)이 같은지 대조한다.
- **주장 조건**: 세 seed의 부호가 같고 합친 95% CI가 0을 제외할 때만 주장한다. 사전등록 대효과 20%와 그대로 비교한다.
- **family 간(E vs L) 비용 차이는 "절감"이라 부르지 않는다.** "기권 효과"로 따로 보고한다.

**비열등성.** 명칭은 one-sided noninferiority + 양측 equivalence 보고다. 세 검정이 **모두** 통과해야 주장한다(교집합-합집합).
1. **사람 UA 짝 차이**
   - 비교: (E-SQ − E-GR), (E-R1 − E-GR)
   - 슬롯 = (질의, 관점, 번호). 최종 출력 = 슬롯별 마지막 생성본. 라벨은 원래 sub-question 문구로 한다.
   - UA = 사람 "답함" ∧ 모든 문장이 MiniCheck 지지. 기권은 0이다. 텍스트가 같은 짝은 차이 0으로 포함한다.
   - 기준: topic 군집 90% CI 하한 > −0.10. 근거: "10 슬롯당 온전한 답 1개 이하 손실".
   - 공동 보고: 비율 하한 0.75. 민감도: 0.05, 0.20.
2. **자동**: 질의 단위 legacy coverage 차이, δ = 0.05(질의당 sub-answer 1개 이하 손실). 민감도 0.03, 0.10.
3. **자동**: 질의 단위 MiniCheck grounding 차이, δ = 0.05.

**E-R1 규칙.** E-R1도 비열등이면 주결과를 "재시도 자체의 가치가 없다"로 바꾼다. Partial Rewind의 의의는 "재시도를 할 때 비용을 줄인다"로 좁힌다.

**R1-8 귀속 — 상태 전이표.**
- 재실행 단위마다 (r → r+1) 상태를 기록한다. 상태는 {format_fail, insufficient, dropped, cited-unsupported, sufficient}다.
- 형식 회복 몫 = format_fail → sufficient 전이의 토큰. 상한은 `_repair` + `answer_retry` 토큰이다.
- 회복 불가 지출 = 모든 라운드에서 insufficient였던 단위의 토큰.
- **"인과가 아닌 회계 분해"라고 명시한다.**

**게이밍 감시**(모두 보고): 불일치 짝 수, 최소 한쪽이 비지 않은 슬롯의 조건부 UA, arm·라운드별 기권률, insufficient → answered 전이율.

**offline 재채점.** 모든 출력을 MiniCheck와 g1으로 다시 채점한다. 루프 안 판정과 100% 같은지 assert한다. g1 수치는 그림자로만 둔다.

**예측**(방향만, 결과와 무관하게 보고):
- 모든 arm에서 verify 통과는 0 근처다.
- E는 사람 "답함" 비율을 높이고 legacy coverage를 낮춘다.
- 토큰은 SQ < BR < GR 순이다.
- **E 계열에서는 FREEZE가 드물다.** E-BR 절감은 0 근처이고, E-SQ 절감은 대부분 실행당 입력 축소에서 나온다.
- L-BR 절감은 캐시 기반 8.3%와 비슷하거나 그보다 작다.

---

## 9. 사람 라벨 (저자 1인, 블라인드)

**절차.** arm 이름은 해시로 가리고 순서를 섞는다. 빈 답은 "(no answer)"로 통일한다. 키는 따로 봉인하고, **라벨 CSV를 채점 전에 커밋**한다(SHA-256 기록). RNG seed는 세트마다 고정한다.

| 세트 | 대상 | 건수 | 라벨 |
|---|---|---|---|
| P-H | pilot5 s1 라운드 0, E-GR와 L-GR. 관점 층화 비례 추출 | E insufficient 30 (B) + E answered 30 (A) + L answered 20 (A) | A: 답함/일부/답안함, B: 답있음/일부/답없음 |
| H3 | held-out s1 라운드 0. H2-B 100개 sub-question 재사용(B 새 라벨 0) + E answered 약 35 + H2-A 앵커 25 | 약 60 | A |
| H4 | held-out 최종 출력, E-SQ / E-GR / E-R1 | G5 슬롯 수(기본 200) × 3 arm. 비어 있지 않은 답만 라벨 | A |
| H5 | held-out E-SQ 최종 insufficient. seed·관점 층화, 라운드 ≥ 1에서 새로 생긴 answered 층 포함 | 40 | B |
| R | 2026-11-16 재라벨 50(H2-ho, seed 43) + H4에서 14일 뒤 30 | 80 | 원 세트 조건 그대로 |
| C (2nd annotator) | **실시하지 않는다**(2026-11-05 결정: 공저자는 검토만 맡는다). 신뢰도 근거는 R(저자 재라벨)만이다 | 0 | — |

- **표집 세부값** (`experiments/r2_sheets.py`에서 고정)
  - 세트별 RNG seed: P-H 4201, H3 4203, H4 4204, H5 4205
  - arm 가림 해시의 salt: `r2-blind-v1`
  - H4 슬롯 단위는 **(seed, 질의, 관점, 번호)**다. 빈 답은 시트에 넣지 않고 키에 UA = 0 행으로만 남긴다. 한 슬롯 안에서 텍스트가 같은 답은 문항 하나로 묶고 차이 0으로 센다.
  - H5 배분: insufficient 30건 + 라운드 ≥ 1에서 새로 answered가 된 층 10건
  - P-H 짝: 뽑힌 슬롯에서 다른 arm의 답이 비어 있으면 그 arm을 키에 UA = 0 행으로 남긴다(G5 짝 차이용). 짝이 30개 미만이면 SD 추정이 약하다고 보고한다.
  - 봉인 키(`_key_*.csv`, `_meta_*.json`)는 시트 폴더와 다른 `<out>_sealed/` 폴더에 쓴다. 라벨러는 시트 폴더만 받는다.
- **추정**: Hájek 가중(w = N_h / n_h), 토픽 군집 부트스트랩. 사람 UA 비열등성 분석 코드는 라벨이 들어온 뒤 이 등록 규칙대로 작성한다. 작성 시점과 커밋을 `DEVIATIONS.md`에 적는다.
- **κ 해석**: κ ≥ 0.60이면 조정 라벨을 쓴다. 미달이면 라벨러별 결과를 함께 내고 "신뢰도 제한"으로 적는다.
- **공시**: 라벨러는 저자이고 가설을 알고 있다. 2nd annotator가 없으면 R1-3을 "미충족"으로 적는다. judge 일치로 대체하지 않는다.

---

## 10. 예산과 중단

- **청구 원장** `results/r2/<run_tag>/billing_ledger.jsonl`이 기준이다. 실제 API 호출마다 기록하고, arm·seed·재개를 넘어 누적한다.
- **하드 스톱**: 청구 15,000,000 in 또는 5,600,000 out에 도달하면 멈춘다(`BudgetExceeded`).
- **단가** (2026-11-05 확인, 출처 https://developers.openai.com/api/docs/pricing, 표준 요금, 1M 토큰당)
  - gpt-5-mini: 입력 $0.25, 캐시 입력 $0.025, 출력 $2.00
  - gpt-5.4-nano: 입력 $0.20, 캐시 입력 $0.02, 출력 $1.25
- **추정 금액**
  - gpt-5-mini(본실험 + 파일럿 약 9.9M in / 3.7M out): 약 $9.9
  - N arm(nano 1.7M / 0.63M): 약 $1.1
  - 하드 스톱까지 다 쓰면 gpt-5-mini 약 $15
- **금액 상한**: $25 (2026-11-05 승인). 코드는 토큰 하드 스톱으로만 멈춘다. gpt-5-mini 원장이 하드 스톱까지 차도 약 $15이므로, 상한 $25에서 남는 몫은 N arm(nano)과 재실행분이다. 실행마다 원장 금액 환산을 기록하고 누적 $25 전에 멈춘다
- **모델 지속성 위험**: gpt-5-mini가 실험 중 교체되거나 퇴역하면 `model_returned`가 바뀐다. 그 실행은 이탈로 기록하고, 같은 시기 arm끼리만 비교한다.
- **GPU** 30시간 상한.
- **실측이 추정의 1.5배를 넘으면 멈추고 보고**한다.

---

## 11. 제외 항목과 응답서 문구

| 제외 | 사유 | 문구 |
|---|---|---|
| VERIFY / PRUNE | 발화 조건이 이 데이터에서 성립하지 않는다(모순 신호 26.4%가 양립 추정 쌍) | "removed from the evaluated system; Proposition 1 restated for the evaluated two-action system" |
| 답변가능성 게이트(G-F) | 연산자가 held-out 관문을 탈락했다(κ 0.41 / 0.583) | "no gate was deployed" |
| 두 번째 코퍼스 | 범위 밖 | "claims restricted to the evaluated corpus" |
| retrieve-then-decompose | sub-question이 arm마다 달라져 짝 비교가 깨진다 | future work |

---

## 12. 이탈 규칙

- 이탈은 모두 `DEVIATIONS.md`에 날짜, 사유, 영향받는 실행을 적는다.
- 파일럿 dev 수치는 헤드라인에 섞지 않는다.
- 모든 arm × seed 표를 공개한다.

---

## 커밋 전 체크리스트

- [x] 승인(G0) — 2026-11-05, 교신저자. 공저자는 검토만 맡는다
- [x] 2nd annotator — 없음 (§9 C). R1-3은 "미충족"으로 공시
- [x] OpenAI 단가 확인 (§10, 2026-11-05). 금액 상한 $25 승인
- [ ] Author Portal 재투고 마감일 확인 — 미확인. 축소 규칙(§2)은 이미 등록돼 있어 실행을 막지 않는다. 마감 때문에 축소하면 이탈로 기록한다
- [x] 코드 커밋 해시 `f6dd344`, config SHA 확인 (§0, 2026-11-05)
- [ ] 이 파일의 커밋 해시를 원고 §V에 적는다
