# D9 검증기 후보 조사 — 사전 등록용

2026-09-06 조사. **D8-B 결과를 보고 고른 것이 아니라 문헌 기본 설정만으로 골랐다는
기록을 남기기 위해, 아래는 전부 공식 문서·소스 코드에서 그대로 옮긴 값이다.**
우리 자료로 튜닝한 값은 하나도 없다.

---

## 결론 먼저 — 가장 중요한 발견

**MiniCheck 도 청크 위에서 최댓값 집계를 쓴다.** 소스(`minicheck/inference.py`)에
`label_probs[:, 1].max().item()` 이 그대로 있다. 즉 **검증기를 바꿔도 D8-A 가 잡아낸
최댓값 팽창은 자동으로 사라지지 않는다.**

> **따라서 D9 스펙에 음성 통제를 필수로 박아야 한다.** D8-A 와 동일한 교란 배정
> (i → i+7 mod 15)으로 위양성률을 재고, **순증(정상 − 교란)** 으로만 판정한다.
> grounded 비율만 보고 채택하면 D7 에서 저지른 실수를 그대로 반복한다.

다행히 **팽창 자체를 피하는 설계가 가능하다.** MiniCheck-Flan-T5-Large 의 입력 상한은
**2048 토큰**이고, branch 의 인용 청크는 중앙 3개 × 약 214토큰 ≈ **640토큰**이라
**한 문서로 통째로 들어간다.** 즉 청크 분할도, 최댓값 집계도 일어나지 않는다.
`docs=[인용 청크 전체를 이어붙인 문자열]`, `claims=[문장]` 으로 **1회 호출**하면
집계 단계가 아예 없어진다. 이것이 D9 의 기본 구성이어야 한다고 본다.

이 설계가 성립하는 근거가 바로 우리 D7-0 이다. 현행 DeBERTa-NLI 는 274토큰에서
이미 무너졌지만(0.992 → 0.152), MiniCheck 는 2048토큰 입력으로 **학습된** 모델이다.
"긴 전제를 견디는가"가 두 계열을 가르는 축이고, D7-0 이 그 축을 이미 측정해 뒀다.

---

## 후보 A — MiniCheck (EMNLP 2024, Liyan06/MiniCheck)

| 항목 | 값 (문서/소스 그대로) |
|---|---|
| 체크포인트 | `lytang/MiniCheck-Flan-T5-Large` (770M) · `lytang/MiniCheck-RoBERTa-Large` (0.4B) · `lytang/MiniCheck-DeBERTa-v3-Large` (0.4B) · `bespokelabs/Bespoke-MiniCheck-7B` (7B, SOTA) |
| **입력 상한** | Flan-T5-Large **2048** · DeBERTa-v3-Large **2048** · RoBERTa-Large **512** |
| 긴 문서 처리 | 문장 경계로 청크 분할 후 **최댓값 풀링** (`label_probs[:,1].max()`) |
| **판정 임계** | 소스에 `1 if final_score > 0.5 else 0` — **0.5 초과(엄격 부등호)** |
| API | `scorer.score(docs=[...], claims=[...])` → `(pred_label, raw_prob, _, _)` |
| 입력 단위 | **문서 × 문장**. "MiniCheck(document, sentence) → [0,1] is a sentence-level fact-checking model" |
| **가설 단위 지침** | "In order to fact-check a multi-sentence claim, the claim should first be broken up into **sentences**" — **문장**이지 atomic claim 이 아니다 |
| 벤치마크 | LLM-AggreFact (11개 사람 주석 데이터셋). Bespoke-7B 77.4% |
| 라이선스 | Flan-T5-Large 계열 **MIT**. Bespoke-7B 는 **비상업 한정**(상업은 별도 문의) |

**주의 — 우리 D7 결론과 어긋나는 지점이 있다.** D7 은 가설을 atomic claim 으로 쪼개면
좋아진다고 했는데(15.7% → 31.7%), MiniCheck 공식 지침은 **문장으로 쪼개라**다.
MiniCheck 는 문장 단위 claim 으로 학습됐으므로 atomic claim 을 넣으면 학습 분포를
벗어난다. **D9 에서는 우리 D7 결론이 아니라 MiniCheck 지침(문장)을 따라야 한다.**
문헌 기본 설정을 수정 없이 쓴다는 원칙에도 그쪽이 맞다.

## 후보 B — AlignScore (ACL 2023, yuh-zha/AlignScore)

| 항목 | 값 (문서 그대로) |
|---|---|
| 체크포인트 | `AlignScore-base.ckpt` (125M, RoBERTa-base) · `AlignScore-large.ckpt` (355M, RoBERTa-large), HF `yzha/AlignScore` |
| 모드 | `nli_sp` / `nli` / `bin_sp` / `bin`. **기본은 `nli_sp`** (3-way 헤드 + 청크-문장 분할) |
| 긴 문서 처리 | 문맥을 **약 350토큰 청크**로 자동 분할, claim 은 spaCy 로 문장 분할 후 정렬 |
| **판정 임계** | **문서에 없다.** 연속 점수를 내는 지표이고 이진 기준을 정의하지 않는다 |
| API | `scorer.score(contexts=[...], claims=[...])` → 수치 1개 |
| 학습 | 7개 과제(NLI·QA·패러프레이즈·사실검증·검색·의미유사·요약) 4.7M 예제 통합 |

**임계가 없다는 것이 D9 설계상 문제다.** 사전 등록 원칙상 실행 전에 하나로 고정해야
하는데 문헌 기본값이 존재하지 않는다. 두 가지 길뿐이다:
- (a) **0.5 를 임의 고정**하고 "문헌 기본값이 없어 임의로 정했다"고 논문에 명기한다.
- (b) AlignScore 를 **1차 후보에서 빼고** MiniCheck 만 사전 등록한다. AlignScore 는
  임계 없이 **연속 점수의 순위 상관**으로만 보조 보고한다.

**(b) 를 권한다.** (a) 는 "우리가 임계를 골랐다"는 자유도를 남기고, 그 자유도가 바로
이 논문이 비판하는 대상이다. 자기가 지적한 함정을 자기가 밟는 모양이 된다.

---

## 권고 — D9 사전 등록 초안

1. **1차 검증기: `lytang/MiniCheck-Flan-T5-Large`.** MIT 라이선스, 2048토큰,
   임계 0.5 초과가 소스에 명시. Bespoke-7B 는 성능이 높지만 **비상업 라이선스**라
   논문 재현성에 걸림돌이고, GPU 요구도 크다. 2차로만 둔다.
2. **가설 단위는 문장.** MiniCheck 공식 지침을 따른다(우리 D7 의 claim 결론이 아니라).
3. **전제는 branch 인용 청크를 이어붙인 단일 문서.** 2048토큰 안에 들어가므로
   내부 청크 분할과 최댓값 풀링이 일어나지 않는다. 집계 자유도를 0으로 만든다.
   (2048 초과 시 어떻게 할지도 실행 전에 정해야 한다 — 앞에서부터 절단을 권한다.
   최댓값 풀링을 허용하면 D8-A 를 다시 해야 한다.)
4. **음성 통제 필수.** D8-A 와 같은 교란(i → i+7 mod 15). **판정은 순증으로만.**
   통과 기준은 D8-A 와 동일하게 교란 위양성 ≤ 5% 로 두기를 권한다.
5. **비교 대상은 D8-B 의 N층 40건.** 현행 추정자가 "미지지"라 한 것 중 판정자가
   "지지"라 한 29건을 MiniCheck 가 되찾는지가 핵심 질문이다. 되찾으면
   "표준 추정자가 근거를 못 본다"가 확증되고, 못 되찾으면 결론이 뒤집힌다.
   **양쪽 결과의 문장을 실행 전에 다 써 두는 것이 맞다.**
6. **AlignScore 는 보조.** 임계 없이 연속 점수로만, 순위 상관으로 보고.

## 아직 확인 못 한 것

- MiniCheck 가 2048토큰을 **어느 토크나이저 기준**으로 세는지(Flan-T5 SentencePiece).
  우리 청크 토큰 수는 `o200k_base` 기준이라 직접 비교가 안 된다. 실행 전 실측 필요.
- `Bespoke-MiniCheck-7B` 의 상업 라이선스 조건.
- AlignScore 의 기반 모델 입력 상한(문서에 없음).

## 출처

- https://github.com/Liyan06/MiniCheck
- https://huggingface.co/lytang/MiniCheck-Flan-T5-Large
- https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B
- https://github.com/yuh-zha/AlignScore
- https://aclanthology.org/2023.acl-long.634/
