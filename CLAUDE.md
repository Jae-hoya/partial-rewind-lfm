# LFM — Claude Code 프로젝트 지침

이 저장소는 논문 **"Localize, Freeze, Mutate: Evidence-Driven Partial Re-execution for Multi-Perspective RAG" (LFM)** 의 실험 코드다. 내부 프로젝트명은 CPE-MGM이나 코드·논문에는 **LFM만 사용**한다.

## 현재 마일스톤: M1-D7 — 추정자 판별

구현 범위는 `SPEC_M1D7_추정자판별.md`에 정의된 것**만**이다.
`SPEC_M1_TestA_v1.1.md`(M1 완료)와 `SPEC_M1D_진단스윕.md` v1.1(M1-D 완료)은 기록용이다.

### 🛑 STOP RULE (절대 준수)
1. **D7-0 산출 후 중단·보고. D7-1~3 산출 후 다시 중단·보고.**
   (M1 Test A 리포트와 M1-D 진단 스윕은 완료됐다 — `P4_REPORT.md`, `results/diag/D1_D3_FINDINGS.md`.)
2. 아래는 **금지**다. M2 는 D7 판별이 끝나고 사람이 결정한 뒤에만 진행한다:
   - **생성 LLM 호출.** `llm.call` 을 import 하지 않는다. NLI forward pass 만 허용한다.
     D7 은 청구 토큰 0 이 설계 전제다 — 스펙에만 적어두면 우선순위가 밀린다.
   - **낭비율 계산.** `test_a`, `sufficient()` 를 import 하지 않는다 (AST 검사 테스트로 강제 중).
   - `relation.py` (cross-perspective relation)
   - `controller/lfm.py`, `controller/oracle.py`, `controller/random.py`
   - 웹 검색 retriever (ddgs) — 웹 실험은 부록용, 훨씬 뒤
   - Test B/C/D/E 하니스
3. 스펙에 없는 기능을 "미리" 만들지 않는다. 확장 포인트는 인터페이스로만 남긴다.

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
