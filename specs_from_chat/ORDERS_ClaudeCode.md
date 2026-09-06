# ORDERS — Claude Code 실행 명령서 (2026-09-11)

사용자가 논문 완성까지의 판단을 채팅(Claude)에 위임했다. 아래는 확정 결정과 실행 순서다. **질문 없이 진행한다.** 정지점에서만 멈추고, 정지점마다 지정된 산출물을 사용자 → 채팅으로 전달한다. 스펙 문구가 모호할 때만 질문한다. 결과가 마음에 안 들어 규칙을 바꾸고 싶을 때는 질문하지 말고 **실행 전에 써 둔 분기 문장을 따른다.**

---

## A. 확정 결정 (재론 없음)

| 항목 | 결정 |
|---|---|
| 논문 | **Partial Rewind** 1편, IEEE Access. 제목: *Partial Rewind: Localize, Freeze, and Mutate for Evidence-Driven Partial Re-execution in Multi-Perspective RAG*. 본문 약어 없음. 코드 `lfm/` 내부명은 유지하되 본문 등장 금지 |
| 기여 3개 | C1 검증된 충분성 연산자를 제어 상태로 · C2 최소 그래프 변경 + 종료 증명 · C3 품질≈전역 재실행, 비용 절감. 관계 모듈은 구성요소 + ablation |
| 감사 논문 | 폐기(archive_audit/). 감사 결과는 원고 §IV-A 한 절 + 부록 |
| 추정자 | V′(사람 라벨 60+40) 관문. 후보 MiniCheck-Flan-T5-Large 단일문서(1차)·bart-large-mnli(2차). 관문 숫자 고정. 둘 다 통과 시 MiniCheck |
| 관계 상태 | v1.1(관련 pair 제한) + 통제 N1·P1·P2. 1회 한정. 탈락 시 관계 모듈 비활성으로 진행(연구 중단 아님) |
| 낭비율 판정 | M1-C·M1′ 결과가 낮아도 **M2·M3 진행**. 낮으면 논문 결과 문장 B(headroom 작음, 메커니즘 검증 중심) |
| 프롬프트 | 전 실험 동결. 미인용 답안은 파이프라인 행동으로 두고 Partial Rewind의 regen 분기가 처리 |
| 모델 | 생성 `gpt-5-mini`(reasoning_effort=low, 2000), 판정 `gpt-5.4-nano`(게이트), 관계 NLI `bart-large-mnli`, OpenAI 외 상용 API 없음 |
| 토픽·시드 | 50토픽 고정. M1′만 3회 반복. 확장 없음 |
| 사람 시간 | V′ 라벨 60(+40) · C-2 판정 20~30건 · 그 외 없음 |
| 숫자 규칙 | 모든 %p 차이는 원행 비율끼리. 요약 CSV 재반올림 금지. verify_numbers §6b 유지 |

---

## B. 실행 순서와 정지점

```
[지금 병행 가능 — 호출 0]
  1. SPEC_M2 §2.1 relation v1.1 구현 → §2.2 통제 N1·P1·P2 (nano ≤100)          → 정지점 ①
  2. SPEC_M1Vprime: 라벨 시트 60건은 이미 발송. 사용자 라벨 커밋 확인 → 채점    → 정지점 ②
     (보류 판정이면 +40 시트 생성·발송 → 재채점 → 정지점 ②′)
[V′ 확정 후]
  3. SPEC_M1D8 §D8-C: 확정 추정자로 Test A 재채점 (verify 두 절, 호출 0)        → 정지점 ③
  4. SPEC_M1prime: 시드 1회 → 정지점 ④ → 나머지 2회 → 정지점 ④′
[관계 통제 통과 시 / 탈락 시 관계 비활성]
  5. SPEC_M2 §8 Test C-1 (오라클·LFM·Global·Random) + relation P/R/F1            → 정지점 ⑤
  6. SPEC_M2 §8 C-2: 후보 edge 추출 → 사용자 판정 20~30건 대기 → 정밀도          → 정지점 ⑥
[M3]
  7. SPEC_M3 §1: baselines/README.md 고정 + 판정자 게이트                       → 정지점 ⑦
  8. Test B 전 시스템 → summary.csv, pareto.csv                                 → 정지점 ⑧
  9. Test D → Test E                                                            → 정지점 ⑨
[집필]
 10. 원고 manuscript_PartialRewind의 [[ ]] 전수 채움 + verify_numbers 대조       → 정지점 ⑩
```

각 정지점 보고 형식: (a) 산출 파일 경로 (b) 스펙의 판정 기준 대비 결과 한 줄 (c) 실행 전 확정 분기 중 어느 쪽인지 (d) 원고에 들어갈 숫자(원행 값). 그 외 서술은 최소.

병렬: 1과 2는 동시에. 3·4는 2 뒤. 5는 1 뒤(2와 무관). 7~9는 4′와 5 뒤.

---

## C. 탈락 분기 (이미 써 둔 문장을 그대로 따른다)

- V′ 둘 다 탈락 → 정지·보고. 추정자를 더 만들지 않는다. (채팅이 재설계)
- 관계 v1.1 통제 탈락 → Test C-1 미실행, `− relation` 구성이 Full이 된다. SPEC_M2 §12 탈락 문장 사용. 계속 진행.
- M1-C/M1′ 낭비율 <20% → 결과 문장 B. 계속 진행.
- Test C-1 LFM ≈ Random → SPEC_M2 §12 탈락 문장, 관계 비활성으로 M3 진행.
- Test B에서 Partial Rewind가 어느 예산에서도 우위 없음 → SPEC_M3 부정 문장 그대로.
- Test D −relation ≈ Full → 제목·기여에서 cross-perspective 철회(채팅이 원고 수정).

---

## D. CLAUDE.md 3줄 — 마일스톤 전환 시 아래를 그대로 교체
- 지금: `M2 — relation v1.1 + 통제` / `정지점 ① 후 중단·보고. 병행: V′ 채점 → 정지점 ②` / `생성 호출은 §2.2 양성 통제 nano ≤100만. 임계 고정. g₁ import 금지 유지.`
- 이후 각 스펙 §"CLAUDE.md 3줄" 그대로.

---

## E. 원고
- 기준 파일: `manuscript_PartialRewind_v0.1.md`(+docx/pdf). 결과 절은 표 골격과 실행 전 확정 문장이 있다. 숫자 채움만 하고 문장 구조는 바꾸지 않는다. 문장 수정이 필요하면 정지점 보고에 적고 채팅이 고친다.
- 부록 재료: archive_audit/manuscript_IEEEAccess_v0.9(축약), 프롬프트 전문, 토픽 예시 3, 임계 민감도, leakage audit, 재현 절차(`pip install git+https://github.com/Liyan06/MiniCheck.git`; `pip install minicheck`는 다른 패키지).
- 그림 4개: Fig.1 M1′ 낭비율(막대+CI, 시드 3), Fig.2 아키텍처, Fig.3 Pareto, Fig.4 trajectory. 데이터 파일은 `results/figs/`.
