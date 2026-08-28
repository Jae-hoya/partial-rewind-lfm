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
| P0 | 스켈레톤 · config · llm.py · trace.py · trace_validate.py · pytest | ✅ 완료 |
| P1 | download.py · schema_check.py → `data/MANIFEST.json` | 대기 |
| P2 | topics.py 후보 생성 → **사람 승인** | 대기 |
| P3 | heldout · chunk_index · perspectives/branch/evidence/verifier · mock 스모크 | 대기 |
| P4 | 50토픽 실행 · 낭비율 리포트 | 대기 |

각 단계는 완료 조건을 만족하면 중단·보고한다. 이어서 다음 단계로 넘어가지 않는다.
