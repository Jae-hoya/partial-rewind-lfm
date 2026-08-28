# P2·P3 보고 — 토픽 선별, 승인 게이트, 인덱스, 측정 결함 2건

## 요약

| 단계 | 결과 |
|---|---|
| P1 재실행 | 코퍼스 150,000행 / 유니크 142,972 (중복 4.69%). 스키마 게이트 PASS |
| P2 | 후보 60개 → 승인 50개, 탈락 4개 (전부 동일 계열 중복) |
| P3 | 인덱스 142,922 문서 / 685,683 청크 / 1.3 GB. 50토픽 전부 커버리지 통과 |
| 파일럿 | 5토픽 × 3회. 낭비율이 아니라 **측정 결함 2건**을 찾는 데 썼다 |

---

## 1. 30,000 건으로는 안 됐다 — P1 재실행

SPEC v1.1 §2.1 의 파일럿 기본값은 `max_articles = 30000` 이다. 그 코퍼스로
교차 도메인 조건(관련 문서 ≥30, infobox 유형 ≥3)을 만족하는 후보를 뽑으니
**28개**가 나왔고, 그마저 축구선수·시즌 문서였다. 3만 건은 영어 위키백과의 0.4%라
어떤 주제든 이웃이 없다.

HANDOFF §4 의 "커버리지 미달이면 `max_articles` 를 올려 P1 부터 재실행" 경로를 탔다.
150,000 으로 올려 다시 받았다 (975.9s, 부족분 0).

## 2. 토픽 선별을 두 번 짰다

**첫 판(폐기).** 용어별 포스팅을 만들고 그 문서들의 infobox 유형 수로 정렬했다.
유형 수는 교차 도메인성이 아니라 **흔함의 대리 지표**였다. 상위 후보가
`eight`(관련 564) · `sports` · `open` · `point` · `came` · `make` · `days` · `date` 로
채워졌고, target article 도 주제와 무관했다 — 용어 `date` → 문서 `Hideo Date`.
생성된 질문도 그대로 무너졌다:

> *"How did an eight-car endurance racing event shape championship outcomes, technical
> strategies, safety protocols, and historical controversies?"*

**둘째 판(채택).** 순서를 뒤집었다.

1. target 후보 = abstract ≥600자, 본문 ≥6,000자, 참고문헌 ≥20개, infobox 보유.
   disambiguation / list of / index of 제외.
2. 그 문서의 **변별력 용어** = 제목+abstract 용어 중 df ∈ [20, 3000] 인 것을 idf 순 상위 5개.
3. 이웃 = 그 용어 중 2개 이상을 담은 문서.
4. 이웃 ≥30 이고 이웃의 infobox 유형 ≥3종.
5. **주 infobox 유형당 상한 3개** — 안 걸면 축구선수가 후보를 독식한다.
6. 정렬 키를 유형 수가 아니라 **관련 문서 수**로. 유형 수로 정렬하면 다시 흔한 문서가 올라온다.

결과 후보 60개: 이웃 142~241 문서, infobox 유형 8~88종.

첫 판을 무너뜨린 기능어들(`world`, `war`, `state`, `series`, `list`, `total`, `area`,
`region` …)은 stopword 로 막았고, 그 회귀를 테스트로 고정했다
(`test_stopwords_cover_the_generic_words_that_broke_the_first_attempt`).

## 3. 승인 게이트 — 검사가 게이트다

사용자가 승인 권한을 위임했으므로 `data/approve_topics.py` 로 대신 수행했다.
HANDOFF §4 의 경고("여기서 대충 승인하면 낭비율이 토픽 난이도의 산물이 된다")를
지키려면 통과 도장이 아니라 검사가 게이트여야 한다. 기계 검증 항목:

- 토픽 문구 존재·길이
- 이웃 문서 수 ≥ `topics.min_related_docs`
- 이웃 infobox 유형 ≥ `topics.min_infobox_types`
- target_article 이 코퍼스에 실재 (제목 정규화 일치)
- 변별 용어 자카드 ≤ 0.6 (near-duplicate 배제)
- 연도만 다른 동일 계열 target 배제

**승인 50 / 탈락 4.** 탈락은 전부 계열 중복이었다 — Eurovision 리투아니아 연도별 3건,
Welsh Labour 3건. 승인 경위·검사 항목·탈락 사유는 `topics_pilot50.yaml` 안에 남겼다.

### 철회한 기준 하나

처음엔 "토픽 문구가 target 제목을 재현하면 held-out 누수"로 보고 10건을 탈락시켰다
(승인 48 < 50 으로 게이트가 FAIL 을 냈다). **잘못된 판정이었다.**

held-out 의 통제 수단은 **target article 을 인덱스에서 빼는 것**이지 질문에서 이름을
숨기는 것이 아니다. 주제를 지목하지 않는 질문은 held-out reference 과제로서 오히려
부자연스럽다. 게다가 제목이 한 단어면(`Bharath (actor)` → `bharath`) 주제를 언급만
해도 비율이 1.0 이 되어 지표 자체가 무의미하다. 값은 `title_overlap` 으로 기록만 한다.

## 4. held-out · 인덱스

- **held-out**: 50토픽 전부 target 확정. 이 스냅샷에 redirect 필드가 없어
  title 정규화 매칭(NFKD → 발음구별기호 제거 → 괄호 한정어 제거 → 선행 관사 제거).
  제목 정규화가 2건 이상 잡은 토픽 3개는 전부 코퍼스 중복 행이었다.
- **코퍼스 중복**: 150,000행 중 유니크 142,972 (4.69%). 상류 샤드가 서로 겹친다 —
  `2026 United Kingdom local elections` 는 44행. `chunk_index.py` 가 identifier 기준으로
  첫 행만 남긴다. 안 걸렀으면 `chunk_id` 가 충돌하고 같은 본문이 검색 상위를 중복 점유했다.
- **인덱스**: 문서 142,922 · 청크 685,683 · 평균 214.2 토큰 · 1,304 MB.
  청킹 247.7s, 빌드 85.2s. 50토픽 전부 `min_chunks_per_topic=20` 통과.

## 5. 파일럿 3회 — 낭비율 0.0% 는 결과가 아니라 결함이었다

| 파일럿 | 변경 | grounding | coverage | coverage=1.0 branch |
|---|---|---|---|---|
| `_pilot5` | (없음) | 0.097 | 0.983 | 59/60 |
| `_pilot5b` | 메타 진술 금지 | 0.292 | 0.468 | 11/60 |
| `_pilot5c` | + 검색 앵커 | 0.307 | 0.561 | 16/60 |

### 결함 ① 생성기가 코퍼스를 서술했다

첫 파일럿에서 낭비율이 전 τ 에서 0.0% 였다. 원자료를 열어보니 coverage 는 0.98 인데
grounding 이 0.097 이었다. 실제 문장-청크 쌍을 NLI 에 직접 넣어보니:

```
SENT: CPs acquire web novels and adapt them into various formats and generate revenue
      through serialized publication, indicating CPs and platforms jointly ...
CHUNK: The main participants in the web novel industry are platforms, content
      providers (CPs), and writers. Platforms distribute web novels to readers ...
max_ent = 0.000
```

청크가 문장을 **명백히 뒷받침하는데도** entailment 가 0 이다. 답안이
`The provided context identifies ...` / `The context does not list ...` 처럼
**코퍼스에 대한 메타 진술**로 쓰였기 때문이다. 가설이 세계가 아니라 코퍼스를 서술하면
그 코퍼스로 entail 될 수 없다.

원인은 내 프롬프트였다: "근거가 없으면 있는 만큼만 쓰라" 가 모델을 헤지 문장으로
몰았다. 직접 사실 진술만 쓰고 근거가 없으면 문장을 줄이라고 고쳤다.

NLI 자체는 정상이었다 — 통제 쌍에서 entail 0.997, contradiction 1.000, GPU(cuda) 사용,
`id2label` 을 모델 config 에서 읽는다.

**coverage 0.98 → 0.47 은 붕괴가 아니라 거품이 빠진 것이다.** 0.98 은
`"맥락에 없다 [c1]"` 같은 문장도 인용으로 세던 값이었다.

### 결함 ② BM25 가 주제 신호 없이 일반 명사에 끌려갔다

검색 결과를 직접 찍어보니:

| sub-question | 상위 검색 결과 |
|---|---|
| 웹툰 연재 플랫폼·기간 | Burgess Hill railway station / Crouch Hill railway station |
| 웹툰 파일 형식·레이아웃 | List of web archiving initiatives |
| 축구협회 지배구조 | Euroscepticism |
| 플랫폼별 콘텐츠 관리 도구 | Job safety analysis |

sub-question 의 일반 명사(`station`, `layout`, `systems`, `web`)에 BM25 가 끌려간다.
토픽의 변별 용어를 검색 앵커로 질의 앞에 붙였다. 앵커는 라운드·조건에 무관하게
고정이므로 retrieval 은 여전히 전 조건 공통 고정 변인이다 (SPEC §2.6).

### 여기서 튜닝을 멈춘다

두 수정 모두 **들여다본 결함**이 근거이지 출력 숫자가 근거가 아니었다. 앵커 이후에도
낭비율이 낮으면 그게 결과다. 마스터플랜 §9 는 낭비율 <20% 를 연구 질문 교체 신호로
규정하며, 그건 엔지니어링으로 지울 대상이 아니다.

### 파일럿 오염 고지

`t000`~`t004` 는 본실험 50토픽에 포함된다. 이 5개로 프롬프트를 교정했으므로
본실험 리포트는 전체 50 기준 수치와 **파일럿 제외 45토픽** 보조 수치를 함께 싣는다.
교정 대상은 측정 타당도(메타 진술·검색 잡음)였지 낭비율 값이 아니었다.

## 6. Acceptance Criteria 점검

- [x] `pytest` 전체 통과 (201개). evidence 는 손계산 고정 케이스 포함 — 단일 소스
      diversity=0, 2소스 = ln2/ln5 = 0.4306765581, 5소스 = 1.0
- [x] `schema_check.py` 가 MANIFEST 생성, 필수 필드·redirect 판정 기록
- [x] `topics.py` 가 후보 출력 후 종료, `approved: true` 없이 다음 단계 실행 안 됨
- [x] `chunk_index.py` 가 `heldout_ids.json` 부재 시 에러 종료
- [x] 라운드 2 프롬프트 해시가 라운드 1과 다름 (`test_round_2_prompt_hash_differs_from_round_1`)
- [x] 동일 config 2회 실행 시 `waste_ratio.csv` 동일 — 154/154 캐시 히트, 청구 토큰 0
- [x] trace JSONL 이 동결 스키마 검증 통과, `action.params` 에 교란 파라미터 기록
- [ ] `report.md` τ 3개 + CI + τ_global 병기 + 판정 문장 → 본실행 산출물
- [x] 총 LLM 비용 로그 (`usage.json`, `UsageStats.billed_tokens_*`)
