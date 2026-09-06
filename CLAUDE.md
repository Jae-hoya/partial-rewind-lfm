# LFM — Claude Code 프로젝트 지침

이 저장소는 논문 **"Localize, Freeze, Mutate: Evidence-Driven Partial Re-execution for Multi-Perspective RAG" (LFM)** 의 실험 코드다. 내부 프로젝트명은 CPE-MGM이나 코드·논문에는 **LFM만 사용**한다.

## 현재 마일스톤: M1′ — Test A 신규 실행 (확정 추정자)

구현 범위는 `specs_from_chat/SPEC_M1prime.md` 에 정의된 것**만**이다.
`SPEC_M1_TestA_v1.1.md`(M1 완료) · `SPEC_M1D_진단스윕.md` v1.1(M1-D 완료) ·
`SPEC_M1D7_추정자판별.md`(D7~D14 완료) · `SPEC_M1Vprime.md`(V′ 완료, 판정 **보류**)
는 전부 기록용이다.

### 🛑 STOP RULE (절대 준수)
1. **시드 1회 완료 후 표 산출·중단·보고 → 승인 후 나머지 2회 → 집계 후 중단·보고.**
   시드 1 에서 FREEZE 가 한 번도 발화하지 않으면 **결과 문장 C 확정**이므로 나머지
   2시드를 돌리지 말고 멈춘다. 비용 때문이 아니라, C 면 시드를 늘려도 답이 같다.
2. **결과 문장은 실행 전에 고정됐다.** 결과를 보고 프레이밍을 고르지 않는다.
   - **A** `W₀.₈ ≥ 20%` — 재실행된 branch 의 x% 가 이미 충분했다. 반사실 하한(7.0%)이
     과소평가였음이 확인된다.
   - **B** FREEZE 발화, 절감이 반사실 이하 — 얼리기가 verifier 입력을 바꿔 이후 라운드의
     충분 판정이 줄어든다. **반사실이 하한이라는 가정은 이 궤적에서 지지되지 않는다.**
   - **C** FREEZE 발화 0회 — 확정 추정자로도 실행 중에는 충분 판정이 서지 않는다.
     재채점에서 본 헤드룸은 궤적 의존이다.
   세 시드의 `W₀.₈` 범위가 판정 구간을 걸치면 **"구간을 걸침"으로 보고**하고 한쪽으로
   몰지 않는다. 시드 간 편차는 어느 경우든 표에 함께 싣는다.
3. 아래는 **금지**다:
   - **프롬프트·검색·토픽·생성 모델 변경.** 전부 동결이다.
   - **추정자를 `estimator.py` 밖에서 부르는 것.** grounding 은 반드시 그 경유다.
   - **원본 Test A 디렉토리(`results/test_a/20260829-041445/`) 쓰기.** 읽기 전용이다.
   - `estimator.kind` 기본값 변경. 기본은 `nli_max` 이고, 바뀌면 D7~D14 의 재현이
     조용히 깨진다. M1′ 는 실행 시에만 `minicheck_doc` 을 켠다.
   - Test B/C/D 하니스 — C-1 은 관계 통제 탈락으로 **취소**, B/D 는 미실행 확정.
     (Test E 는 M1′ trace 에서 호출 0 으로 뽑는 것이라 허용.)
4. 스펙에 없는 기능을 "미리" 만들지 않는다. 확장 포인트는 인터페이스로만 남긴다.

## 최종 확정 스펙 (구버전 무효화 표)

계획 문서가 여러 버전을 거쳤으므로, **다른 어떤 문서·주석과 충돌해도 아래가 우선**한다.

| 항목 | 최종 확정 | 폐기된 구버전 |
|---|---|---|
| Action space | `FREEZE, EXPAND, VERIFY, PRUNE` (4개) | 7개(DEEPEN/BRANCH/SWITCH 포함) |
| Evidence state E_i | `[coverage, grounding, source_diversity, support_strength]` | `confidence`(self-report) 포함판 |
| source_diversity | K-cap entropy: `min(1, H(p)/log K)`, K=5, 단일 소스→0 | unique/citations 비율 |
| Relation 복잡도 표기 | O(n²m²) (n=관점, m=관점당 claim) | O(n²) |
| Contradiction 집계 | 양방향 NLI의 **max** | 평균 |
| 미해소 충돌 상태명 | `UNRESOLVED_CONFLICT` | GENUINE_DISAGREEMENT |
| 주 데이터셋 | HF `wikimedia/structured-wikipedia` (2026 snapshot) | Kaggle판 primary |
| NLI | 전용 NLI 모델 (config로 선택) | LLM-as-judge |
| 목적함수 | quality–cost Pareto 직접 보고 | λ,μ,ν 가중합 |

## 코드 규약

- Python 3.11+, `pyproject.toml`, `pytest`. 모든 모듈에 최소 1개 단위 테스트.
- 모델명·경로·시드·예산은 전부 `config.yaml`에서 주입. **코드에 하드코딩 금지.**
- 모든 무작위성은 `config.seed`로 고정. 같은 config + 같은 데이터 스냅샷 → 같은 결과여야 한다.
- **모델 정책:** 생성·평가용 상용 API는 **OpenAI만** 사용 — 주모델 `gpt-5-mini`(전 시스템·전 baseline 동일 고정), `gpt-5.4-nano`는 LLM-judge(M3, 검증 게이트 통과 조건부)와 부록 모델 일반화 체크 한정. Anthropic·Gemini 등 다른 상용 API 연동 금지. 로컬 오픈소스 모델(NLI, 선택적 보조 judge)은 허용. 인간 평가는 전 과정에 없음 — 평가는 객관 지표 + LLM-judge 프로토콜(마스터플랜 M3)로만.
- LLM 호출은 `llm.py` 단일 래퍼 경유. 호출마다 (tokens_in, tokens_out, latency_ms)를 반환하고 trace에 기록. 응답은 디스크 캐시(키: model + system + prompt + max_tokens + reasoning_effort + cache_nonce) — 재실행 비용 절감 및 결정성.
- 컨트롤러는 전부 `controller/base.py`의 동일 인터페이스를 구현한다. **비교 공정성의 핵심이므로 baseline 전용 우회 경로 금지.**

## Trace 스키마 (동결 — 필드 추가만 허용, 변경·삭제 금지)

라운드마다 1레코드, JSONL:

```json
{
  "run_id": "uuid", "query_id": "q0042", "round": 3,
  "state": {
    "E": {"p1": {"coverage": 0.62, "grounding": 0.81, "source_diversity": 0.50, "support_strength": 0.74}},
    "R": [],
    "budget_left": 0.55,
    "frozen": [],
    "verify_count": {},
    "graph": {"V": ["p1","p2","p3"], "E": [], "S": {"p1": "ACTIVE"}}
  },
  "action": {"type": "RETRY_ALL", "target": [], "params": {}},
  "cost": {"tokens_in": 0, "tokens_out": 0, "llm_calls": 0, "retrievals": 0, "latency_ms": 0},
  "reward": null,
  "next_state": {}
}
```

- `reward`는 항상 `null`로 기록 (종료 후 별도 스크립트가 backfill — M1 범위 아님).
- M1에서 `R`, `verify_count`는 빈 값이지만 **필드는 반드시 존재**해야 한다 (M2 호환).
- 상태 라벨: `ACTIVE | FROZEN | PRUNED | UNRESOLVED_CONFLICT` (M1은 ACTIVE만 사용).

## 재현성 규칙

- 데이터 스냅샷: 다운로드 직후 `data/schema_check.py` 실행, snapshot 식별자·파일 SHA256을 `data/MANIFEST.json`에 기록. 스키마 검사 실패 시 실험 시작 금지.
- 실험 산출물은 `results/{exp_name}/{timestamp}/`에 config 사본과 함께 저장.
