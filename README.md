# LFM — M1 Test A

논문 *Localize, Freeze, Mutate: Evidence-Driven Partial Re-execution for Multi-Perspective RAG*
의 실험 저장소. 현재 마일스톤은 **M1 — Test A(동기 실험)** 하나뿐이다.

우선 문서: `CLAUDE.md` → `SPEC_M1_TestA_v1.1.md`. 이 둘과 충돌하는 다른 문서는 무효다.
마스터플랜(M2/M3 설계)은 의도적으로 이 저장소 밖(`../계획/`)에 둔다.

## 실행

    C:\Users\skyop\anaconda3\python.exe -m pytest

`python` 을 그냥 쓰면 Windows 스토어 껍데기가 조용히 exit 0 으로 끝난다. 위 절대경로를 쓴다.

    # trace 동결 스키마 검증
    C:\Users\skyop\anaconda3\python.exe experiments/trace_validate.py results/test_a/<ts>/trace.jsonl

## API 키

`.env` 에 `OPENAI_API_KEY=...` 를 넣는다(`.env.example` 참고). `.gitignore` 1번 줄에 있다.
`lfm.llm.load_dotenv_if_present()` 가 올린다. 키 값은 로그·커밋에 남기지 않는다.

## 진행 단계

| 단계 | 범위 | 상태 |
|---|---|---|
| P0 | 스켈레톤 · config · llm.py · trace.py · trace_validate.py · pytest | ✅ |
| P1 | download.py · schema_check.py → `data/MANIFEST.json` | ✅ (150k 로 재실행) |
| P2 | topics.py 후보 60 → approve_topics.py 승인 50 | ✅ |
| P3 | heldout · chunk_index(685,683 청크) · 모듈 · mock 스모크 | ✅ |
| P4 | 50토픽 실행 · 낭비율 리포트 | 진행 |

## 문서 지도

| 파일 | 내용 |
|---|---|
| `P1_REPORT.md` | 코퍼스·스키마 게이트. referencerisk 샤드 간 이질성 |
| `P2_P3_REPORT.md` | 토픽 선별 2회, 승인 게이트, 파일럿 3회 — 측정 결함 2건 |
| `MODEL_NOTES.md` | 모델 검증, reasoning 토큰, 키 사용 개시일·비용 귀속 |
| `VALIDITY.md` | Threats to Validity 재료 |
| `REPRODUCE.md` | 재현 절차 · 스냅샷 고정값 · 산출물 지도 |

## 실험 결과가 나쁘게 나오면

그게 결과다. 마스터플랜 §9 는 낭비율 <20% 를 연구 질문 교체 신호로 규정한다.
숫자를 밀어 올리는 방향으로 파라미터를 만지지 않는다. 측정 결함(들여다본 근거가
있는 것)만 고친다.
