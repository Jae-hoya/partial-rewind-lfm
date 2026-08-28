# SPEC M1 — Test A (동기 실험) · v1.1

목표: **"global retry가 실패 시점에 이미 충분했던 branch를 얼마나 낭비 재실행하는가"** 를 측정한다. 이 낭비율이 논문 전체의 전제다. LFM controller는 만들지 않는다.

산출물: ① 동작하는 global-retry 파이프라인 ② `results/test_a/`의 낭비율 리포트(CSV + 차트 + 요약 md) ③ 전 실행 trace(JSONL).

---

## v1.1 변경점 (v1.0 대비 — 이 6개만 바뀌었다)

| # | 항목 | v1.0 | v1.1 |
|---|---|---|---|
| 1 | LLM 캐시 ↔ 재실행 | 캐시 키 = model+prompt hash. RETRY_ALL은 "seed/파라미터 교란" | 캐시 키에 `cache_nonce` 추가. 라운드 교란을 **프롬프트·검색 파라미터 수준**으로 명시 (§3.1, §3.6) |
| 2 | 검증 임계 | `cfg.tau_g_global` 기본값 없음 | `tau_g_global = tau_c_global = primary_tau = 0.80` 고정, 집계 방식(mean/min) 명시 (§3.5, §4) |
| 3 | 모듈 순서 | 2.3 chunk_index → 2.4 topics | topics·held-out 확정 **후** chunk_index. held-out 파일 없으면 인덱스 빌드 에러 종료 (§2) |
| 4 | τ 스윕 반복 | "τ별 3~4 반복 계산" (결정론이라 무의미) | 점추정 1회 + **쿼리 단위 bootstrap CI** (LLM 재호출 0) (§4.5) |
| 5 | 인덱스 규모 | `rank_bm25`, 20만~50만 문서 | 파일럿 `max_articles=30000`, 백엔드 `bm25s` 기본. 청크 수 상한 가드 (§2.5) |
| 6 | 수동 승인 게이트 | "수동 승인 후 사용" (자율 실행이 통과함) | `topics.py`는 후보 출력 후 **강제 종료**. `approved: true` 없으면 이후 단계 실행 금지 (§2.4) |

**CLAUDE.md 동반 수정 1줄** — "응답은 디스크 캐시(키: model+prompt hash)" → **"키: model + system + prompt + max_tokens + cache_nonce"**. (변경 #1과 정합)

---

## 1. 환경

```
pip install datasets pyarrow duckdb bm25s rank_bm25 transformers torch sentencepiece pyyaml pytest openai
```

- 생성 LLM: `config.llm.model`로 주입 — **OpenAI API 전용, 기본값 `gpt-5-mini`** (CLAUDE.md 모델 정책). max_tokens 상한 필수.
- NLI: `config.nli.model` — 기본 `cross-encoder/nli-deberta-v3-base` (속도), 옵션 `microsoft/deberta-v3-large` MNLI 계열 (정확도). GPU 없으면 base 강제.
- 착수 시 `config.llm.model`이 실제 API에서 호출 가능한지 1콜로 검증하고 실패 시 즉시 중단·보고.

## 2. 데이터 파이프라인 (`data/`)

**실행 순서 고정 (변경 #3).** 이 순서를 어기면 held-out 누수가 발생한다.

```
download → schema_check → topics(후보) → [사람 승인] → heldout 확정 → chunk_index
```

### 2.1 `download.py`
- HF `wikimedia/structured-wikipedia`에서 **영어판만**, Parquet 스트리밍으로 로컬 저장.
- **파일럿 기본값 `config.data.max_articles = 30000`** (변경 #5). 20만~50만은 M2 본실험 값이며 M1에서 쓰지 않는다.
- 3.4의 토픽 관련 도메인을 커버하는 서브셋이면 충분하다.

### 2.2 `schema_check.py` (다운로드 직후 필수 실행)
- 필수 필드 존재 검사: abstract, sections, infoboxes, **references/citations**, **referencerisk**(또는 credibility signal 상당 필드).
- **redirect 필드 존재 여부도 함께 기록** — 없으면 held-out에서 title 정규화 매칭으로 fallback하고 그 사실을 MANIFEST에 남긴다.
- 결과와 스냅샷 식별자·SHA256을 `data/MANIFEST.json`에 기록.
- references/referencerisk 부재 시: 경고 후 `citation_mode="chunk"`로 fallback 플래그 기록 (실험은 계속 가능, Test C-2만 영향 — M1 무관).
- **MANIFEST.json 생성 후 종료·보고한다** (P1 게이트).

### 2.3 `topics.py` — 후보 생성
- **파일럿 50개 토픽** 후보 생성 (Test A는 실패 사례 수집이 목적 — 200개는 M2에서 확장).
- 토픽 조건: 교차 도메인성 — infobox 카테고리가 서로 다른 3개 이상 하위주제를 갖는 주제.
- **인덱스를 요구하지 않는다.** 후보 선별은 Parquet 메타데이터 스캔(duckdb) + LLM 후보 생성으로 수행한다.
- 각 후보에 대해 코퍼스 내 관련 문서 수를 세어 함께 출력한다.
- 출력: `data/topics_pilot50_candidates.yaml`.

### 2.4 승인 게이트 (변경 #6)
- `topics.py`는 후보 파일을 쓴 뒤 **반드시 종료한다.** 이후 단계를 이어서 실행하지 않는다.
- 사람이 검토·편집하여 `data/topics_pilot50.yaml`로 저장하며, 파일 최상단에 `approved: true`를 넣는다.
- 이후 모든 단계는 시작 시 `topics_pilot50.yaml`의 `approved: true`를 확인한다. 없거나 false면 **에러로 종료**한다. 경고 후 진행 금지.

### 2.5 `heldout.py`
- 승인된 각 토픽의 target article + redirect를 식별해 `data/heldout_ids.json`에 기록하고, reference 정답으로 별도 보관한다.
- redirect 필드가 없는 스냅샷이면 title 정규화 매칭으로 대체하고 그 사실을 기록한다.
- (near-dup audit은 M2 — 여기서 하지 않는다.)

### 2.6 `chunk_index.py`
- **입력으로 `data/heldout_ids.json`을 필수로 받는다. 파일이 없으면 에러로 종료한다** (변경 #3).
- 청킹: **섹션 단위** (구조화 JSON의 section 경계 그대로). 섹션이 1,200 토큰 초과 시 문단 분할.
- 인덱스: sparse lexical BM25. `config.retrieval.backend ∈ {"bm25s", "rank_bm25"}`, **기본 `bm25s`** (변경 #5). 둘 다 sparse이므로 dense 금지 규칙과 무관하다. `rank_bm25`는 소규모 단위 테스트용으로만 둔다.
- M1에서 retrieval은 전 조건 공통 고정 변인 — **dense 인덱스 만들지 말 것.**
- chunk 메타: `{doc_id, title, section_path, source_url}`.
- 빌드 후 통계 로그 필수: 문서 수, 청크 수, 평균 청크 길이, 빌드 시간, 인덱스 메모리. **청크 수가 `config.retrieval.max_chunks`(기본 2,000,000)를 넘으면 중단하고 보고한다.**
- 토픽 커버리지 검증: 승인된 50토픽 각각에 대해 관련 청크가 `config.data.min_chunks_per_topic`(기본 20) 이상 존재하는지 확인. 미달 토픽이 있으면 리스트를 출력하고 **중단·보고**한다.

## 3. 모듈 스펙 (`lfm/`)

### 3.1 `llm.py`
```python
def call(prompt: str, *, system: str = "", max_tokens: int, tag: str,
         cache_nonce: str = "") -> LLMResult
# LLMResult: text, tokens_in, tokens_out, latency_ms
# 디스크 캐시 키 = sha256(model + system + prompt + max_tokens + cache_nonce)
```
- **`cache_nonce`는 라운드 교란의 유일한 캐시 회피 수단이다** (변경 #1). 라운드 r>0의 모든 호출은 `cache_nonce=f"r{r}"`를 전달한다.
- nonce는 라운드 번호의 결정론적 함수이므로, 같은 config를 2회 실행하면 같은 nonce → 캐시 히트 → 동일 결과가 된다. 재현성과 재실행이 양립한다.
- 호출마다 (tokens_in, tokens_out, latency_ms)를 반환하고 trace에 기록한다.

### 3.2 `perspectives.py` — 고정 role set 4개
```python
ROLES = ["overview_factual", "technical_mechanism", "impact_application", "controversy_limitation"]
def decompose(query: str) -> list[Perspective]   # role별 관점 질문 + sub-question 3~5개 생성 (LLM 1회)
```

### 3.3 `branch.py`
```python
def run_branch(p: Perspective, retriever, llm, cfg, *, round: int = 0) -> BranchOutput
# 절차: sub-question별 BM25 top-k 검색 → 관점 답안 생성(인용 필수, [chunk_id] 표기)
#      → BranchOutput(answer, claims, citations, chunks)
# claims: 답안을 atomic claim으로 분해 (LLM 1회, structured output)
# round > 0이면 §3.6의 교란 파라미터를 적용하고 모든 LLM 호출에 cache_nonce=f"r{round}" 전달
```

### 3.4 `evidence.py` — E_i 4지표
```python
def coverage(p, out) -> float          # 근거 인용이 붙은 sub-question 비율 (결정론)
def grounding(out, nli) -> float       # 답안 문장 중 인용 chunk가 entail하는 비율 (NLI, threshold cfg.nli.tau_ground)
def source_diversity(out, K=5) -> float# min(1, H(p)/log K); 단일 소스→0.0
def support_strength(out, nli) -> float# claim별 최대 entailment 확률의 평균 (NLI)
```

### 3.5 `verifier.py`
```python
def global_verify(branch_outputs, query, nli, cfg) -> VerifyResult(passed: bool, agg: dict)
```
**통과 조건 (집계 방식 명시 — 변경 #2):**

```
passed  ⟺  mean_b grounding(b) ≥ cfg.verify.tau_g_global
       AND  min_b  coverage(b)  ≥ cfg.verify.tau_c_global
```

- grounding은 branch 평균, coverage는 branch 최솟값이다. 이 비대칭은 의도된 것이며 report.md에 그대로 기술한다.
- `agg`에 mean_grounding, min_coverage, branch별 원값을 모두 담는다.
- LLM judge 금지 — NLI/규칙 기반만.

### 3.6 `controller/base.py`, `controller/global_retry.py`
```python
class Controller(ABC):
    def step(self, state: State) -> Action        # M2 호환 인터페이스

class GlobalRetry(Controller):
    # verify 실패 시 Action("RETRY_ALL"): 모든 branch 재실행, 최대 cfg.max_rounds=3
```

**라운드 교란 정의 (변경 #1).** 교란은 프롬프트·검색 파라미터 수준에서 일어나며, sampling 파라미터(temperature/seed)에 의존하지 않는다. 모델별 sampling 파라미터 지원 여부가 다르기 때문이다.

라운드 r ∈ {1, 2}에서 각 branch에 적용:

1. **검색 폭**: `top_k_r = cfg.retrieval.top_k * cfg.retrieval.retry_k_multiplier[r]` — 기본 `[1, 2, 3]`.
2. **sub-question paraphrase**: 직전 라운드의 sub-question을 LLM 1콜로 재표현 (`tag="paraphrase"`, `cache_nonce=f"r{r}"`). 관점 role과 개수는 유지한다.
3. **retry directive**: 생성 프롬프트에 `cfg.retry_directives[r]` 고정 문자열을 삽입. 예) r=1 — "이전 시도는 근거가 부족했다. 다른 하위 질문 각도와 다른 출처를 우선하라."
4. 모든 LLM 호출에 `cache_nonce=f"r{r}"` 전달.

적용된 교란 파라미터(top_k, directive id)를 trace의 `action.params`에 기록한다.

> 이 정의가 없으면 캐시가 라운드 1의 응답을 그대로 반환해 3라운드가 동일 복제가 되고, 낭비율은 측정이 아니라 항등식이 된다.

### 3.7 `graph.py`, `trace.py`
- CLAUDE.md의 동결 스키마 그대로. `graph.py`는 라운드별 (V,E,S) 스냅샷 제공 (M1은 S 전부 ACTIVE).

## 4. Test A 프로토콜 (`experiments/test_a.py`)

1. 50개 토픽 × GlobalRetry 실행 (max 3 rounds). 전량 trace 기록.
2. **실패 케이스** = 1라운드 verify 불통과 쿼리 전부.
3. 각 실패 케이스에서, 실패 시점의 각 branch를 **사후 개별 평가**: `sufficient(b) := grounding(b) > τ ∧ coverage(b) > τ`.
4. **낭비율** = (재실행된 branch 중 sufficient였던 수) / (재실행된 branch 총수). 쿼리별·전체 집계.
5. **Threshold 스윕 (변경 #2, #4):**
   - τ ∈ `cfg.sweep.tau` = {0.7, 0.8, 0.9}. 사후 라벨링은 결정론이므로 **각 τ당 점추정 1회**만 계산한다. 반복 계산하지 않는다.
   - 불확실성은 **쿼리 단위 bootstrap**으로 구한다: 실패 쿼리 집합에서 복원추출 `cfg.sweep.bootstrap_n`(기본 1000)회, 낭비율의 95% CI. `cfg.seed`로 고정하며 **LLM 재호출은 0회**다.
   - `cfg.verify.tau_g_global = cfg.verify.tau_c_global = cfg.sweep.primary_tau = 0.80`으로 고정한다. 판정 문장은 `primary_tau` 기준이다.
   - 낭비율은 전역 검증 임계와 τ의 상대 위치에 직접 좌우되므로, report.md는 두 값을 반드시 같은 표에 병기한다.
6. 산출:
   - `waste_ratio.csv`: query_id, round, branch, sufficient@0.7/0.8/0.9, re_executed, tokens
   - `waste_ratio_ci.csv`: tau, point, ci_low, ci_high, n_failed_queries, n_reexec_branches
   - `report.md`: 실패 케이스 수, τ별 낭비율 표(τ_global 병기), 라운드별 토큰 분해, **판정 문장** ("τ=0.8 낭비율 X% [CI a–b] → go/no-go 기준 대비 판정")
   - 막대 차트 1장 (τ별 낭비율 + CI 에러바)

## 5. Acceptance Criteria

- [ ] `pytest` 전체 통과 (모듈별 최소 1 테스트; evidence 지표는 손계산 고정 케이스 포함 — 예: 단일 소스 diversity=0)
- [ ] `schema_check.py`가 MANIFEST.json 생성, 필수 필드·redirect 필드 판정 기록
- [ ] `topics.py` 실행이 후보 파일 출력 후 종료하고, `approved: true` 없이 다음 단계가 실행되지 않음
- [ ] `chunk_index.py`가 `heldout_ids.json` 부재 시 에러 종료
- [ ] 라운드 2의 프롬프트 해시가 라운드 1과 다름을 확인하는 테스트 존재 (캐시 회피 검증)
- [ ] 동일 config 2회 실행 시 waste_ratio.csv 동일 (LLM 캐시 히트 기준)
- [ ] trace JSONL이 동결 스키마 검증 스크립트(`trace_validate.py`) 통과, `action.params`에 라운드 교란 파라미터 기록
- [ ] report.md에 τ 3개 낭비율 + CI + τ_global 병기 + go/no-go 판정 문장 존재
- [ ] 총 LLM 비용 로그 출력 (파일럿 50토픽 기준 상한 config로 통제)

## 6. 금지사항 (재확인)

relation.py, lfm/oracle/random controller, dense retrieval, 웹 검색, Test B~E, 논문용 그림 생성. **Test A report 산출 후 중단·보고.**

## 7. config.yaml 필수 키 (신규·변경분)

```yaml
seed: 42

data:
  max_articles: 30000            # 변경 #5 (M2에서 200000+로 상향)
  min_chunks_per_topic: 20

retrieval:
  backend: bm25s                 # 변경 #5
  top_k: 5
  retry_k_multiplier: [1, 2, 3]  # 변경 #1
  max_chunks: 2000000

llm:
  model: gpt-5-mini
  max_tokens: 1500

nli:
  model: cross-encoder/nli-deberta-v3-base
  tau_ground: 0.5

verify:
  tau_g_global: 0.80             # 변경 #2 — primary_tau와 동일하게 고정
  tau_c_global: 0.80

sweep:
  tau: [0.7, 0.8, 0.9]
  primary_tau: 0.8
  bootstrap_n: 1000

max_rounds: 3

retry_directives:                # 변경 #1
  1: "이전 시도는 근거가 부족했다. 다른 하위 질문 각도와 다른 출처를 우선하라."
  2: "앞선 두 시도가 실패했다. 하위 질문을 더 구체적인 사실 단위로 쪼개고 인용을 늘려라."
```
