# 코퍼스2 판정 — `C:\Users\skyop\Desktop\나의 rag\03_LFM\LFM\m1\results\c2\heldout25`

자동 생성(`experiments/c2_analyze.py`). 수치는 ANALYSIS.json 의 `c2` 블록.

- C-G2 E-GR 라운드 0 answered: 0.255 (퇴화 False)
- FREEZE 사건(라운드 수): {'E-BR': 0, 'E-GR': 0, 'L-BR': 3, 'L-GR': 0}
- C-G3: False · C-G6(a): True · grounding 재채점: True
- assay coverage: E-GR 평균 0.097, 90% 하한 0.070 > δ 0.05 → True
- assay grounding: E-GR 평균 0.224, 90% 하한 0.165 > δ 0.05 → True
- **C-H1**: 방향만 일치 {'point': 0.0024228345888355207, 'ci95': [0.0, 0.009862852600450861], 'per_seed': {'1': 0.0, '2': 0.0, '3': 0.007203465514635576}, 'claim': False}
- **C-H2**: 판정 불가 {'reason': 'C-G3 미통과 (SQ 기전)'}
- **C-H3**: 판정 불가 {'reason': 'C-G3 미통과 (SQ 기전)'}
- **C-H4**: 재현 {'counted_legs': ['coverage', 'grounding'], 'legs': {'coverage': {'counted': True, 'reason': None, 'diff': 0.16479377104377105, 'ci90': [0.12348484848484849, 0.20776620370370372], 'noninferior': True}, 'grounding': {'counted': True, 'reason': None, 'diff': 0.09225701805250318, 'ci90': [0.005743048125320002, 0.1768341862509643], 'noninferior': True}}}
- **C-H5_secondary**: 판정 불가 {'reason': '짝 없음', 'definition': '1 − Σ(E-BR)/Σ(E-SQ) (r2_analyze SAVINGS 정의 그대로; 보고만)'}

## 코퍼스1 정합

| 항목 | c2 | c1 | 차이 95% CI | 정합 |
|---|---|---|---|---|
| C-H1 | 0.002 | 0.086 | [-0.124, -0.046] | False |
| C-H4_coverage | 0.165 | 0.083 | [0.023, 0.141] | False |
| C-H4_grounding | 0.092 | 0.137 | [-0.200, 0.106] | True |
