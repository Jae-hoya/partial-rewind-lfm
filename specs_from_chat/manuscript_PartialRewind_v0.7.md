# Partial Rewind: Localize, Freeze, and Mutate for Evidence-Driven Partial Re-execution in Multi-Perspective Retrieval-Augmented Generation

<!--
==================== 공저자 노트 (원고 아님) ====================
버전: v0.7 (2026-09-15). v0.6 + D12 사람 기준 재집계 반영: 5추정자 순증 표(부록 C.7), 유지 칸 교환(12/16), 판정자 값이 사람 구간 안에 있다는 주의, like-for-like(33건 중 30건이 N층) 한 줄.
버전: v0.6 (2026-09-15). 사람 라벨 60건 반영. 잠정 표기 제거, 판정자 기준 수치를 사람 기준으로 교체(0.913→0.80, 0.725→0/33 vs 22/33), V′ 관문 결과(3/4 통과, recall 0.694, 규칙상 미결) 공개, 판정자 65.6% 결과를 §VI-F에 신설. D12 판정자 표는 부록으로.
버전: v0.5 (2026-09-14). v0.4 + §XI 결론 · §VIII-C 절감 분포 문단(Fig.4a) · Fig.1/Fig.4 본문 참조와 캡션 · 부록 A~F 연결 · Fig.3 삭제.
버전: v0.4 (2026-09-13). v0.3 + §VIII-C(궤적 선택 비교, 표 16) 채움 · 순환 지적 반영(선택 기준 지표 vs 독립 지표 분리) · §X 한계 3건(branch 단위 과소평가 / 생성 미발생 / 관점 분해 미측정) · 초록·기여 4 갱신. 전부 [[prov.]].
버전: v0.3 (2026-09-12). v0.2 + §VIII-A(M1-C 재채점, 표 14)·§VIII-B(반사실 절감, 표 15) 채움, §VIII-C M2-min 자리(표 16), §X 반사실 한계 문장, 기여 4·초록 갱신. 전부 [[prov.]].
버전: v0.2 (2026-09-12). 사용자 지시(부정 결과를 논문에 싣는다)에 따라 감사 원고 v0.9의 결과 절을 Part 1(§VI)로 흡수, 관계 결과를 Part 2(§VII), 방법 결과를 Part 3(§VIII)로 재편. 기여 4개. 표 번호: Part 1 = 1–12, Part 2 = 13, Part 3 = 14–17(재정렬 필요 시 조판에서). 결과에 의존하지 않는 절은 전부 완성, 결과 절은 표 골격 + 실행 전 확정 문장.
결정(채팅이 위임받아 확정): ① 시스템명 Partial Rewind(본문 약어 없음; 코드 내부명 lfm/ 유지, 본문 등장 금지) ② 기여 3개를 위험 분산형으로 재편(§I) ③ 관계 모듈은 구성요소+ablation ④ IEEE Access.
채움 규칙: [[ ]]는 에이전트가 results/에서 채움. [[prov.]]는 사람 라벨 전 잠정. 모든 %p 차이는 원행 비율끼리 뺀 값만.
임계 경로: 사용자 라벨 60(+40) → V′ → M1-C → GO → M1′(시드 3) ∥ 관계 v1.1 통제 → Test C-1 → Test B/D/E.
표 번호는 확정. 그림: Fig.1 = M1′ 낭비율(또는 M1-C), Fig.2 = 아키텍처, Fig.3 = Test B Pareto, Fig.4 = trajectory 예시.
==================================================================
-->

**Abstract** — Multi-perspective retrieval-augmented generation decomposes a query into perspectives, answers each from retrieved evidence, and re-executes every perspective when verification of the assembled answer fails. We ask how much of that re-execution is unnecessary. We present Partial Rewind, a control architecture that keeps the evidence state of every perspective and, on failure, freezes the sufficient branches and mutates only the deficient subgraph under a budget with a proved termination bound. Building it produced two negative results, reported as findings. First, the grounding estimator commonly used to judge sufficiency—a sentence-pair inference model—judged three of six hundred branch executions sufficient and enabled no selective re-execution at all; validated against blind human labels it is right four times in five when it reports support and recovers none of the support it misses, because it collapses as unrelated content fills its premise. Its two obvious repairs fail a shuffled-evidence control; a claim-versus-document verifier with no aggregation freedom passes it. Second, cross-perspective contradiction from off-the-shelf inference models detects synthetic contradictions and paraphrases almost perfectly yet labels a quarter of compatible fact pairs from one answer as contradictory, so it cannot trigger verification. With the relation module disabled and a validated operator in place, the sufficient population rises from one branch in 180 to nineteen, and a controller that freezes them cuts generation calls by eight percent with metrics outside the freeze criteria unchanged—a lower bound, since the execution unit measured is coarser than the one the method specifies. We release the protocol, code, and traces.

**Index Terms** — retrieval-augmented generation, multi-agent systems, partial re-execution, execution graphs, evidence sufficiency, groundedness evaluation.

---

## I. INTRODUCTION

Retrieval-augmented generation (RAG) has moved from a single retrieve-then-generate pass toward systems that decide what to execute and when: on-demand retrieval with self-reflection [1], corrective retrieval [2], query-complexity routing [3], and multi-perspective decomposition, in which a complex query is split into perspectives that are researched in parallel and then assembled [4]. Multi-agent variants add orchestration over these branches [5]–[7]. What these designs share is a recovery policy: when verification of the assembled answer fails, the workflow is re-executed. Every perspective is re-run, including those whose evidence was already adequate.

Re-execution is where the cost of these systems concentrates. In the pipeline studied here, 72% of billed tokens were spent in retry rounds, and those rounds left coverage slightly lower and the share of uncited answers unchanged. Whether the re-executed branches needed re-executing is a question we could not find measured. We define the quantity directly—the *branch-level waste ratio*, the fraction of re-executed branches that were already sufficient when verification failed—and set out to measure it. The first measurement returned 0.5%, and it was uninformative: no query passed verification at any round, and only three of six hundred branch executions were judged sufficient, because the operator that decides sufficiency—a grounding score from a sentence-pair natural-language-inference (NLI) model—could not see support that was present in the cited chunks. Validated against blind human labels, that operator was right four times in five when it reported support and recovered none of the 33 sentences it had scored as unsupported that a human marked supported; a replacement recovers 22. A control loop built on the deployed operator would freeze nothing and re-execute everything; it would be global retry with extra steps.

This paper therefore does two things in order. It first establishes a sufficiency operator whose error profile is known, using a compact validation protocol—negative controls against topic-shuffled evidence, human-labelled precision and recall on the system's own outputs—and re-measures the waste ratio with it. It then presents **Partial Rewind**, a control architecture for multi-perspective RAG that treats the execution as a graph whose nodes carry evidence state, and that recovers from failed verification by *localizing* the deficient sub-questions and conflicting perspective pairs, *freezing* the branches that are sufficient, and *mutating* only the affected subgraph: expanding retrieval for uncovered sub-questions, verifying conflicting claim pairs through a targeted verification subgraph, and pruning branches that another branch already covers. The unit of re-execution is the sub-question, not the branch; the recovery is a minimal graph mutation $\Delta G$ rather than a restart; and the loop terminates under a budget by a bound we prove.

Prior work uses semantic conflict to trigger further reasoning or retrieval, attributes failures post hoc, or replays computations invalidated by changed inputs. We instead localize insufficiency and inconsistency at the level of sub-questions and perspective pairs and selectively mutate only the affected execution subgraph while preserving validated branches.

The contributions are as follows. Two of them are negative results, and they are reported as results rather than as caveats, because each decides what the method can do.

1. **The standard sufficiency operator cannot support a freeze decision, and we show why.** In 600 branch executions it judged three sufficient and enabled zero selective re-executions; validated against 60 blind human labels it is precise (0.80) and blind (recall 0.33; of 33 sentences it scored as unsupported that a human marked supported it recovers none, where the accepted replacement recovers 22). The blindness is driven by the fraction of unrelated content in the premise, not by its length; it is a property of the deployed checkpoint, not of the model family; and its two obvious repairs—sentence splitting with maximum aggregation, and a larger checkpoint that doubles the raw score—both fail a shuffled-evidence control. Two repairs survive the control: a claim-versus-document verifier with no aggregation freedom, and single-chunk premise cleaning (Section VI).
2. **Cross-perspective contradiction from off-the-shelf NLI is a precision failure, not an absence of signal.** The relation module detects synthetic contradictions (97.8%) and paraphrases (100%) almost perfectly and labels 26.4% of compatible fact pairs from the same answer as contradictory; at that false-positive rate no threshold or aggregation yields a usable VERIFY trigger, and defined over all pairs the score measures unrelatedness—shuffled pairs score higher than same-topic pairs (Section VII).
3. **Minimal graph mutation with a proved bound.** We formalize the execution as $G_t=(V_t,E_t,S_t)$ with a four-action space whose structural effects are explicit, define the mutation $\Delta G_t$ and its size, localize at sub-question granularity, and prove termination under a call budget (Sections III–IV).
4. **Headroom appears under a validated operator, and freezing converts it into savings.** Re-scoring the same trajectory with the accepted operator raises the sufficient population from 1 to 19 of 180 branches at $\tau=0.8$—a nineteen-fold change produced by the sufficiency judgment alone. Letting a freeze-and-expand controller select its own trajectory over the same artifacts reduces generation calls by 7.8% and tokens by 9.4% while metrics outside the freeze criteria are unchanged, a lower bound because the execution unit measured is the branch rather than the sub-question (Section VIII).

A methodological result runs through the first two: an estimator cannot serve as the reference for its own replacement. The one gate in this study that used the deployed checkpoint's confident positives as ground truth failed every candidate, the deployed checkpoint included; human labels replaced it.

---

## II. RELATED WORK

### A. Adaptive, Iterative, and Self-Reflective RAG

Self-RAG learns when to retrieve and critiques its own output with reflection tokens [1]; CRAG grades retrieved documents and triggers alternative retrieval when they are inadequate [2]; Adaptive-RAG routes queries among no-retrieval, single-step, and iterative strategies by a complexity classifier [3]. Recent iterative systems make the sufficiency judgment explicit, judging structured sufficiency and gaps to drive the next query [8] or maintaining an evidence state across reasoning steps [9]. These adapt *retrieval*, and their unit of adaptation is the query. Partial Rewind adapts the *execution graph* of a multi-branch answer and treats sufficiency as a validated classifier rather than a heuristic.

### B. Multi-Perspective and Multi-Agent RAG

STORM decomposes a topic into perspectives and simulates per-perspective research before synthesis [4]. Conflict- and consensus-driven multi-view systems retrieve from several views and reconcile disagreement among candidate answers by further retrieval and agreement [10], [11]; dynamic orchestration frameworks choose which agents and edges to activate per query [5], [6]. These establish the value of decomposition into branches and inherit the recovery policy we study: on failed verification they re-run the round or re-sample all candidates. In MA-RAG [10], conflict arises among candidate answers to the *same* question and is resolved by retrieval; in Partial Rewind, inconsistency is measured between perspectives with *different* evidence requirements, so that its location points to a specific evidence channel, and it is one of three signals—with coverage gaps and redundancy—that select a local mutation. We are careful to distinguish two works named MA-RAG [10], [12].

### C. Failure Localization and Partial Re-execution

Post-hoc attribution identifies the agent or step responsible for a failed run [13], [14] and stops at diagnosis. Lineage-based execution recomputes only the nodes whose inputs changed [15]; it does not address outputs that are wrong from unchanged inputs. Debate gating decides whether to deliberate, not what to repair [16]. Partial Rewind closes the loop that these leave open: the localization drives the next execution, and the state of the graph—frozen, pruned, unresolved—persists across rounds.

### D. Groundedness Estimation

NLI-based factual-consistency scoring was established for summarization by FactCC [17], TRUE [18], and SummaC [19], which also documented that sentence-pair NLI degrades on long premises and that sentence-level maximum aggregation is noisy. AlignScore [20] and MiniCheck [21] train claim-versus-document verifiers to avoid both. RAG evaluation frameworks such as RAGAS [22] and ARES [23] decompose answers into statements and check each against the context, usually with an LLM judge. Section IV-A applies these results to the operator that controls execution rather than the score that is reported, and adds the controls that distinguish a valid operator from one that merely reports a higher number.

---

## III. PROBLEM FORMULATION

### A. Multi-Perspective Pipeline

Given a query $q$, the system generates $n$ perspectives $P_1,\dots,P_n$ ($n=4$ throughout). For each perspective, an LLM call produces $m_i$ sub-questions $Q_i=\{q_{i1},\dots,q_{im_i}\}$; each sub-question retrieves the top-$k$ chunks from a fixed index; an LLM call writes a sub-answer $a_{ik}$ with inline citations to chunk identifiers $C_{ik}$; the branch answer $A_i$ is the ordered concatenation of its sub-answers; an LLM call decomposes $A_i$ into atomic claims $\mathcal{C}_i$. Retrieval is lexical and free; the billed unit is the LLM call.

### B. Evidence State

For branch $i$, the evidence state is $E_i=(\mathrm{cov}_i,\mathrm{grd}_i,\mathrm{div}_i,\mathrm{sup}_i)$:

- $\mathrm{cov}_i=\frac{1}{m_i}\sum_k \mathbf{1}[C_{ik}\neq\emptyset]$, the fraction of sub-questions with at least one citation (quantized in steps of $1/m_i$);
- $\mathrm{grd}_i=\frac{1}{|\mathcal{S}(A_i)|}\sum_{s\in\mathcal{S}(A_i)}\mathbf{1}[g(s, C(s))\ge\tau_g]$, the fraction of answer sentences that the sufficiency operator $g$ judges supported by the chunks cited in the sub-answer containing $s$, with $\tau_g=0.5$;
- $\mathrm{div}_i=\min(1, H(p_i)/\log K)$ with $K=5$, where $p_i$ is the distribution of citations over source documents; a single-source branch has $\mathrm{div}_i=0$;
- $\mathrm{sup}_i$, the mean over claims of the operator's maximum support score against the branch's cited chunks.

An empty branch—one with no cited sub-answer—has $\mathrm{cov}_i=\mathrm{grd}_i=0$. The operator $g$ is not a fixed choice; Section IV-A is about choosing it.

### C. Sufficiency, Verification, and Global Retry

A branch is *sufficient* at threshold $\tau$ if
$$S_\tau(E_i)=\mathbf{1}[\mathrm{cov}_i=1.0\ \wedge\ \mathrm{grd}_i>\tau],\qquad \tau\in\{0.7,0.8,0.9\},$$
with $\tau=0.8$ primary. The conjunction is deliberate: coverage alone is met by citing anything for every sub-question, grounding alone by a single supported sentence, and the grounding ratio rewards brevity (Section IV-A); a fully covered branch has at least three sentences in our data.

Global verification after round $r$ passes if
$$V^r=\mathbf{1}\Big[\tfrac{1}{n}\textstyle\sum_i \mathrm{grd}_i^r\ge\tau^{\mathrm{glob}}_g\Big]\wedge\mathbf{1}\Big[\min_i \mathrm{cov}_i^r\ge\tau^{\mathrm{glob}}_c\Big],\quad \tau^{\mathrm{glob}}_g=\tau^{\mathrm{glob}}_c=0.80 .$$
Grounding is averaged so that one weak perspective does not veto an otherwise supported answer; coverage is taken as a minimum so that the answer is not accepted while any perspective leaves a sub-question without evidence. Verification uses non-strict comparisons and sufficiency strict ones; the difference binds only when $m_i=5$ and $\mathrm{cov}_i=0.8$. *Global retry*—the baseline recovery policy—re-executes all $n$ branches when $V^r=0$ and $r<R-1$, with $R=3$.

### D. Waste Ratio

A retry event is a pair $(q,r)$ with $V^r=0$ and $r<R-1$. The branch-level waste ratio is
$$W_\tau=\frac{\sum_{(q,r)\in\mathcal{T}}\sum_{i} S_\tau(E_i^r)}{\sum_{(q,r)\in\mathcal{T}} n},$$
the fraction of re-executed branches that were already sufficient at the moment of failure. $W_\tau$ is the headroom available to any policy that preserves sufficient branches; it is a property of the global-retry trajectory and does not require running the alternative. It depends on $g$ through $S_\tau$, which is why $g$ must be validated first.

### E. Execution Graph, Actions, and Mutation

The execution is a graph $G_t=(V_t,E_t,S_t)$ at control step $t$. $V_t$ contains branch nodes $v_i$, sub-question nodes $u_{ik}$, and verification nodes $w^{(\ell)}_{ij}$ created by VERIFY. $E_t$ contains containment edges $(v_i,u_{ik})$ and relation edges $e_{ij}$ carrying the relation state $R_{ij}$ (Section IV-D). $S_t:V_t\to\{\mathrm{ACTIVE},\mathrm{FROZEN},\mathrm{PRUNED},\mathrm{UNRESOLVED}\}$ is the status map. The action space is

| Action | Precondition | Structural effect $\Delta G$ | Cost $c(a)$ |
|---|---|---|---|
| FREEZE$(i)$ | $S_\tau(E_i)=1$, no unresolved conflict edge at $i$ | $\Delta S$: $v_i\to$ FROZEN; artifacts retained; no further calls | 0 |
| EXPAND$(i,Q')$ | $Q'\subseteq Q_i$ uncovered or ungrounded | $\Delta V>0$: sub-question nodes in $Q'$ re-executed with wider retrieval ($k\to k_e$) | $|Q'|$ |
| VERIFY$(i,j)$ | $\mathrm{con}_{ij}\ge\tau_c$, fewer than $K$ prior verifications on $e_{ij}$ | $\Delta V>0$: a verification subgraph $w_{ij}$ (targeted retrieval for the conflicting claim pair, two sub-answers regenerated); $R_{ij}$ recomputed | 2 |
| PRUNE$(i)$ | $\mathrm{red}_{ij}\ge\tau_p$ for some sufficient $j$ | $\Delta V<0$: $v_i\to$ PRUNED; artifact retained, excluded from synthesis | 0 |

$\Delta G_t=(\Delta V,\Delta E,\Delta S)$, $G_{t+1}=G_t\oplus\Delta G_t$, and $|\Delta G_t|=|\Delta V|+|\Delta E|+|\Delta S|$. Cost is counted in generation calls, which is deterministic; a budget $B$ bounds the total. Global retry is the special case in which the only action is "re-execute all active branches" at cost $\sum_i m_i$ per round.

### F. Objective

We do not optimize a weighted sum of quality and cost. For each policy we sweep $B$ and report the quality–cost frontier; policies are compared on the same plane (Section X-C).

---

## IV. METHOD

Fig. 2 shows the loop: perspectives are executed once; their evidence states and pairwise relation states are computed; the controller localizes, freezes what is sufficient, and mutates the affected subgraph; only the affected nodes are re-evaluated; the loop repeats until every branch is frozen, pruned, or unresolved, or the budget is exhausted.

### A. The Sufficiency Operator and Its Validation

Every decision in this section rests on $S_\tau$, and $S_\tau$ rests on $g$. Section VI reports, as results, that the standard choice for $g$ cannot support the freeze decision in this setting and how a valid operator was selected. The method assumes only what Section VI establishes: an operator that passes a shuffled-evidence control (false-positive rate $\le5\%$) and reaches precision $\ge0.85$ and recall $\ge0.70$ against reference labels on the pipeline's own outputs. The accepted operator is MiniCheck-Flan-T5-Large with the cited chunks concatenated as a single premise, used with an assertion on every call that no internal chunking or maximum aggregation occurs; it passed the control and three of the four human-label criteria, missing the recall bar by 0.006 (Section VI-F); the relation module requires three-way NLI output and uses `bart-large-mnli` regardless.

### B. Localization

At control step $t$, with active set $\mathcal{A}_t$, evidence states $E_i$, relation states $R_{ij}$, and remaining budget, localization produces a prioritized list of candidate actions:

1. **FREEZE candidates**: $i\in\mathcal{A}_t$ with $S_\tau(E_i)=1$ and $\max_j \mathrm{con}_{ij}<\tau_c$.
2. **VERIFY candidates**: edges $(i,j)$ with $\mathrm{con}_{ij}\ge\tau_c$ and fewer than $K=2$ prior verifications; the target is the arg-max claim pair and the sub-questions containing those claims; the branch with lower $\mathrm{sup}$ is marked suspect and re-retrieved first.
3. **PRUNE candidates**: $i$ such that some sufficient $j$ covers at least $\tau_p=0.8$ of $i$'s claims ($\mathrm{red}_{ij}\ge\tau_p$).
4. **EXPAND candidates**: $i\in\mathcal{A}_t$ not sufficient and not in conflict; the target $Q'$ is the set of sub-questions without citations plus those containing unsupported sentences. A branch with $\mathrm{cov}_i=0$—an *uncited* answer, which in our data is never a retrieval failure (every such branch had retrieved chunks) but a generation format failure—is regenerated with citation formatting enforced rather than expanded; this is recorded as EXPAND with kind *regen*.

Priority is VERIFY $>$ EXPAND $>$ PRUNE $>$ FREEZE; freezes are applied in batch at the end of each step because they cost nothing. The re-execution unit is the sub-question; a branch is never re-executed whole except in the regen case.

### C. Controller

Algorithm 1 gives the rule-based controller used in all experiments. One action per control step is taken in the localization experiments (so that "the first action" is well defined against an oracle); the number of actions per step is a parameter in the cost–quality experiments.

**Algorithm 1** Partial Rewind (rule-based)

```
Input: query q, perspectives P_1..P_n, budget B, thresholds (τ, τ_c, τ_p, τ_e), K
G_0 ← execute all branches once; compute E_i, R_ij; S ← ACTIVE for all v_i
t ← 0
while A_t ≠ ∅ and B > 0:
    cands ← Localize(E, R, S, B)            # ordered: VERIFY > EXPAND > PRUNE > FREEZE
    if cands = ∅: break
    a ← cands[0]
    ΔG_t ← Apply(a); B ← B − c(a)          # VERIFY creates w_ij; EXPAND re-runs Q'; PRUNE/FREEZE change S
    re-evaluate E_i for affected i; recompute R_ij for affected edges only
    if a = VERIFY(i,j) and con_ij ≥ τ_c after K verifications: S(v_i), S(v_j) ← UNRESOLVED
    for i in A_t with S_τ(E_i)=1 and max_j con_ij < τ_c: S(v_i) ← FROZEN
    G_{t+1} ← G_t ⊕ ΔG_t; t ← t+1
return synthesis over FROZEN ∪ ACTIVE branches; disagreement report for UNRESOLVED pairs
```

Unresolved conflicts are reported, not adjudicated: after $K$ verifications the pair is marked UNRESOLVED and the output states both positions with their evidence, as "not resolved within the evidence budget", because retrieval failure, corpus incompleteness, and estimator error cannot be excluded.

### D. Relation State

The relation state $R_{ij}\in[0,1]^4$ is a feature vector, not a distribution. Its definition was revised once, and we report both versions because the first is the natural one and it fails.

**Version 1 (all pairs).** Bidirectional NLI over all claim pairs $(a,b)\in\mathcal{C}_i\times\mathcal{C}_j$; contradiction as the maximum contradiction probability over pairs; agreement as the mean symmetric entailment over the top quartile of pairs. Under a negative control that assigns each branch the claims of a branch from a *different topic*, contradiction exceeded $\tau_c=0.70$ on 91.2% of same-topic edges and on 93.3% of shuffled edges; at the pair level, with the maximum removed, 16.3% of same-topic pairs and 28.3% of shuffled pairs exceeded it (medians 0.059 and 0.296). Shuffled pairs score *higher*. The score measures unrelatedness: NLI models trained on entailment corpora do not model what a hypothesis is about, and two true facts about different subjects are often scored as contradictory. Agreement exceeded 0.60 on no edge in either assignment (median 0.041), which would have left the redundancy signal, and PRUNE, permanently inactive.

**Version 2 (related pairs).** Let $\mathrm{Rel}_{ij}$ be the set of claim pairs whose sub-questions are mapped as the same or adjacent across the two branches (a lexical mapping computed without model calls, already used for complementarity). NLI is computed only on $\mathrm{Rel}_{ij}$:
- $\mathrm{con}_{ij}=\max_{(a,b)\in\mathrm{Rel}_{ij}}\max\{p_{\mathrm{con}}(a\!\to\!b),p_{\mathrm{con}}(b\!\to\!a)\}$, with the arg-max pair stored as the VERIFY target and the count of pairs above $\tau_c$ recorded;
- $\mathrm{agr}_{ij}$, the mean symmetric entailment over $\mathrm{Rel}_{ij}$;
- $\mathrm{red}_{ij}=|\{a\in\mathcal{C}_i:\exists b\in\mathcal{C}_j,\ p_{\mathrm{ent}}(b\!\to\!a)>\tau_e\}|/|\mathcal{C}_i|$, the fraction of $i$'s claims covered by $j$ (asymmetric; $\tau_e=0.70$);
- $\mathrm{comp}_{ij}=\mathrm{Relatedness}\times(1-\mathrm{red}_{ij})\times\mathrm{CoverageGain}$, where CoverageGain is the coverage of the union of the two sub-question sets minus the larger individual coverage.

An edge with $\mathrm{Rel}_{ij}=\emptyset$ carries no signal; this is recorded and distinguished from "no conflict". Version 2 is used only if it passes three controls fixed before it was run: claim pairs from *within* one branch and one sub-question—related by construction and consistent because they come from one answer—must exceed $\tau_c$ on at most 5% of pairs; claims paired with LLM-generated negations must be detected as contradictory at least 80% of the time; claims paired with paraphrases must be scored as mutually entailing at least 80% of the time. Section IX reports the outcome: the two positive controls pass at 97.8% and 100%, the negative control fails at 26.4%, and the relation edges are therefore removed—the controller of Section VIII runs on evidence states alone.

### E. Termination

**Proposition 1.** Let $c(a)\ge1$ for EXPAND and VERIFY, $c(\text{FREEZE})=c(\text{PRUNE})=0$, and let status changes be irreversible (ACTIVE$\to$FROZEN/PRUNED/UNRESOLVED only). Then Algorithm 1 terminates after at most $B+2n$ control steps, and performs at most $K\binom{n}{2}$ verifications.

*Proof.* Each EXPAND or VERIFY step decreases $B$ by at least one, so there are at most $B$ such steps. Each FREEZE or PRUNE step changes the status of one ACTIVE branch to a terminal status; since statuses are irreversible there are at most $n$ such changes of each kind before $\mathcal{A}_t=\emptyset$, at which point the loop exits. Hence the number of steps is at most $B+2n$. VERIFY on edge $e_{ij}$ increments a per-edge counter bounded by $K$, and there are $\binom{n}{2}$ edges. $\square$

The frozen set is monotone, $F_t\subseteq F_{t+1}$, which is the progress property that distinguishes the loop from a restart: what has been validated is never recomputed. Both bounds and the monotonicity are asserted in code.

### F. Complexity

Evidence-state evaluation is linear in the number of sentences. Version-2 relation extraction costs $\sum_{i<j}|\mathrm{Rel}_{ij}|$ NLI pairs, typically tens per edge against $m_im_j\approx225$ for all pairs, and is recomputed only on edges incident to a mutated branch. Generation cost is the budget $B$ by construction.

---

## V. EXPERIMENTAL SETUP

**Corpus and topics.** Wikimedia Structured Wikipedia snapshot [[version/date]]: 150,000 rows, 142,972 unique articles after de-duplication (4.69%), chunked by section and split at paragraph boundaries above 1,200 tokens (`tiktoken o200k_base`); full held-out index of 142,922 documents and 685,683 chunks (target articles removed). [[N]] cross-domain topics derived from held-out target articles, requiring evidence from more than one domain; five pilot topics used for prompt calibration are excluded. Near-duplicate leakage is audited by embedding similarity and only substantive reproductions are removed; the audit list is released.

**Models.** Generation `gpt-5-mini` (reasoning effort low, 2,000 max tokens), fixed prompts, identical across all systems. Retrieval BM25 (`bm25s`), $k=5$, $k_e=10$ for EXPAND. Sufficiency operator per Section IV-A; relation NLI `facebook/bart-large-mnli`. LLM judge `gpt-5.4-nano`, used only for holistic quality (Section V, metrics) and gated as described. No commercial API other than OpenAI's is used.

**Systems.** Global retry (the baseline recovery; identical code path with the mutation loop disabled), Partial Rewind (Algorithm 1), and, for the localization experiment, an oracle controller that reads the injected fault labels and a random controller that selects uniformly among admissible actions. For the cost–quality frontier we add Naive RAG, Self-RAG, Adaptive-RAG, and a STORM-like multi-perspective baseline without retry, all with the same generator, retriever, and corpus [[reimplementation notes]].

**Metrics.** Cost: generation calls (primary), tokens, retrieval calls, wall-clock. Quality: coverage, grounding under the accepted operator, citation accuracy against the structured references, entity recall against the target article, heading soft recall; holistic quality by pairwise LLM judgment with position swapping, blind system names, temperature 0, three repetitions with self-agreement reported, and a validation gate on synthetic clean-versus-corrupted pairs ($\ge90\%$ accuracy) before use. Graph-specific: waste ratio, localization accuracy, frozen-branch reuse rate, $|\Delta G|$ per step, convergence steps. No human evaluation of answer quality is performed; human labels are used only to validate the sufficiency operator and the natural-conflict candidates.

**Protocol.** Conditions, thresholds ($\tau=0.8$, $\tau_c=0.70$, $\tau_e=0.70$, $\tau_p=0.8$, $K=2$), and pass criteria were fixed before each experiment and are not revised after results; threshold sensitivity is reported separately. The waste ratio was never consulted while the sufficiency operator was chosen: the diagnostic code was prevented at import time from accessing the sufficiency and waste modules, and every re-scoring of an existing trajectory was pre-registered by a committed configuration file whose hash we report. Departures from this protocol—two candidate operators entered after partial results were known, and one gate replaced after it failed every candidate—are disclosed in Section VIII.

---

## VI. RESULTS, PART 1 — THE SUFFICIENCY OPERATOR

### A. Under the Standard Operator, FREEZE Never Fires

Table 1 summarizes the main measurement.

**TABLE 1.** Main measurement (50 topics, 4 perspectives, up to 3 rounds; estimator $g_1$). Verification clauses use $\ge$ (Section III-C); the sufficiency operator uses strict comparisons. In this run the two queries that met the coverage clause at round 0 had minimum coverage exactly 1.000, so the distinction did not bind.

| Quantity | Value |
|---|---|
| Queries passing global verification (any round) | 0 / 50 (Wilson 95% CI [0.0, 7.1]%) |
| — grounding clause ($\overline{\mathrm{grd}}\ge0.80$) passed, round 0 / final round | 0 / 50 · 0 / 50 |
| — coverage clause ($\min\mathrm{cov}$) passed, round 0 / final round | 2 / 50 · 0 / 50 |
| Best query mean grounding, round 0 / final round | 0.750 / 0.512 |
| Best query minimum coverage, round 0 | 1.000 |
| Branch executions | 600 (385 non-empty, 215 empty) |
| Empty branches with zero retrieved chunks / failed decomposition | 0 / 0 (all 215 received chunks; none produced a cited sub-answer) |
| Empty-answer rate by round (0 / 1 / 2) | 33.5% / 37.0% / 37.0% |
| Sufficient branches at $\tau=0.8$ (all rounds) | 3 / 600 |
| Waste ratio $W_{0.8}$ | 0.5% [0.0, 1.5] |
| Waste ratio $W_{0.7}$, $W_{0.9}$ | [[fill]] |
| Mean coverage, round 0 → round 2 | Δ = −0.060 |
| Mean grounding, round 0 → round 2 | Δ = −0.042 |
| Share of billed tokens in rounds 1–2 | 72% |

Four observations. First, $W_{0.8}$ is 0.5% and its upper confidence bound is below every pre-registered band in the study design; under $g_1$ the answer to the title question is "almost none." Second, this value is uninformative, because verification never passed and only 3 of 600 branch executions were sufficient: the numerator of $W$ is drawn from an almost empty set. Third, the failure is entirely the grounding clause. Verification is a conjunction of two clauses (Section III-C), and Table 1 splits them: the coverage clause was met by two queries at round 0 (both with minimum coverage exactly 1.000, and with mean grounding of 0.000 and 0.094: every branch fully covered, almost nothing scored as grounded), whereas the grounding clause was met by none—the best query's mean grounding was 0.750 at round 0 against a threshold of 0.80, and 0.512 by the final round. Every one of the 50 failures is attributable to the single quantity that the rest of this section audits. Fourth—and independent of the estimator—re-execution did not repair the answers it re-executed. Per-query paired comparisons from round 0 to round 2 (bootstrap, 10,000 resamples) show mean coverage lower by 0.060 (95% CI excludes zero [[bounds]]), although the direction was not uniform across queries (sign test $p=0.144$), and mean grounding unchanged within noise ($\Delta=-0.042$, CI $[-0.096, +0.010]$, sign test $p=0.883$; 24 queries lower, 22 higher). The empty-answer rate did not fall (33.5%, 37.0%, 37.0% by round). Rounds 1–2 consumed 72% of billed tokens and moved none of these quantities in the intended direction. An earlier draft of this paragraph described re-execution as harmful; the paired analysis supports only "not helpful," and we report it as such. This narrows the attribution of the 0.5% further: the empty sufficient population is not the product of degraded answers but of the estimator alone.

**Empty answers are uncited answers.** All 215 empty branch executions received retrieved chunks (median 35 per branch, against 33 for non-empty branches) and produced sub-questions; none of their sub-answers carried a citation. Retrieval did not fail and decomposition did not fail; the generator did not attach citations, and re-execution did not change that. This is a generator-side format behavior, and it is why the second-generator check of Section X matters.

The rest of this section asks why the sufficient population is empty.

### B. Audit Stage 1: Length Bias in the Grounding Ratio (D0)

The grounding ratio divides by the number of answer sentences. Table 2 cross-tabulates grounding with answer length on the 385 non-empty branches of the main measurement.

**TABLE 2.** Answer length by grounding band (non-empty branches, $g_1$).

| Grounding band | $n$ | Median sentences | Range |
|---|---|---|---|
| $0<\mathrm{grd}\le 0.5$ | 178 | 7.0 | 2–26 |
| $0.5<\mathrm{grd}\le 0.8$ | 33 | 5.0 | 3–23 |
| $\mathrm{grd}>0.8$ | 30 | 2.0 | 1–4 |
| $\mathrm{cov}=1.0$ (any grounding) | 128 | 10 | 3–26 |

Of the 30 branches with $\mathrm{grd}>0.8$, 14 consist of a single sentence and 8 of two; one entailed sentence is a perfect score. Spearman correlation between sentence count and grounding is −0.160 on non-empty branches (it is +0.467 on all 600 only because the 215 empty answers sit at zero on both axes, a population artifact that must be stated whenever these correlations are reported). Grounding alone therefore measures brevity. The conjunction with $\mathrm{cov}=1.0$ removes the bias: the 128 fully covered branches have a median of 10 sentences and none has fewer than 3, because answering four or five sub-questions with citations requires that many sentences. The three sufficient branches in Table 1 each have 4 sentences, all sub-questions answered, 5–10 citations and 11–17 claims; they are not short-answer artifacts.

Two consequences follow. The intersection $S_\tau$ is the correct sufficiency operator and the grounding ratio must not be used alone as a target, because any sweep that maximizes it selects configurations that write shorter answers. And the empty sufficient population is not a length artifact: the 128 fully covered branches (all rounds) have a median grounding of only 0.200. The evidence, as scored by $g_1$, is absent even where coverage is complete.

### C. Retrieval Is Not the Bottleneck (D1–D3, summary)

Four retrieval conditions on a 15-topic subset (baseline; depth $k=5\to15$; the target article's body left in the index; both) moved coverage and the empty-answer rate but not grounding: with the gold article present, coverage rose 0.474 → 0.620 and empty answers fell 35.0% → 13.3% while grounding stayed at 0.188 → 0.181, and deeper retrieval lowered every axis. The full table is in the supplement. Whatever the operator could not see was not missing from the index.

### D. Audit Stage 3: Anatomy of the Estimator

All results in this subsection are computed on the C3 outputs (294 cited answer sentences; 546 claims), because C3 is the condition in which support is known to be available.

**Score distribution.** The maximum entailment probability per sentence is bimodal: 76.5% of sentences score below 0.05, 11.9% at or above 0.90, and only 8.5% fall in the middle band (0.1–0.9). Lowering $\tau_{\mathrm{g}}$ from 0.5 to 0.1 raises the grounded fraction only from 15.6% to 20.4% (0.3: 16.3%; 0.2: 17.7%); an eleven-point sweep of $\tau_{\mathrm{g}}$ over 0.05–0.95 moves it by between 10 and 19 points in total, whereas changing the estimator's configuration moves it by 41.5 points (Section VI-E). For this checkpoint, threshold calibration cannot matter because there is no mass near the threshold; Section VI-G shows that the absence of a middle band is itself a property of the checkpoint.

**Attribution (rejected).** If sentences were supported by a retrieved chunk other than the one cited, widening the premise pool from cited chunks (median 3 chunks) to all retrieved chunks (median 14) would recover them. It recovers 1.7 points (15.6% → 17.3%): of the 225 sentences scoring below 0.05 against cited chunks, 218 (96.9%) score below 0.05 against every retrieved chunk. The answers do not cite the wrong chunk; no retrieved chunk entails those sentences under $g_1$.

**Parametric leakage (minor).** Because the generator has seen Wikipedia in pretraining, it may write true facts from memory that lie in unretrieved parts of the target article. Scoring against *all* chunks of the target article (median 9) raises 23 of the 218 near-zero sentences (10.6%) above 0.5; 185 remain below 0.05. Leakage exists and is small.

**Hypothesis unit.** Table 4 crosses three premise pools with two hypothesis units. Decomposing sentences into atomic claims doubles the grounded fraction (15.6% → 31.7% against cited chunks): a sentence carrying several facts scores zero if any one is unsupported. The target-only pool is worse than the retrieved pool at claim level (17.9% vs. 31.9%) because retrieved non-target chunks support claims the target article does not; the answers draw on retrieved evidence beyond the target article. In no cell does the middle band exceed 12%.

**TABLE 4.** Grounded fraction at $\tau_{\mathrm{g}}=0.5$ / middle band (0.1–0.9) / below 0.05, by premise pool and hypothesis unit (C3; 294 sentences, 546 claims; 22,064 NLI pairs; zero generation calls). Pools are scoped to the hypothesis unit's own citations: the sub-answer's cited chunks for sentence hypotheses and the branch's cited chunks for claim hypotheses [[confirm scoping for the retrieved and target columns]].

| Hypothesis unit | Cited chunks | All retrieved chunks | All target-article chunks |
|---|---|---|---|
| Sentence | 15.6 / 8.5 / 76.5 | 17.3 / 8.8 / 74.2 | 17.3 / 11.9 / 72.8 |
| Claim | 31.7 / 10.3 / 61.9 | 31.9 / 9.9 / 61.2 | 17.9 / 9.5 / 74.5 |

### E. Audit Stage 4: Dilution and Inflation (D7-0, D8-A)

**Positive control: premise dilution.** We took the 35 (chunk, sentence) pairs with $p_{\mathrm{ent}}>0.90$, kept their content, and extended the premise with unrelated chunks to 2×, 3× and 4× its length, recording tokenizer truncation.

**TABLE 5.** Positive control: known-entailed pairs under premise extension (35 pairs).

| Premise length | Median tokens | Truncated pairs | Median $p_{\mathrm{ent}}$ | Pairs with $p_{\mathrm{ent}}\ge 0.5$ |
|---|---|---|---|---|
| 1× (original) | 142 | 0 / 35 | 0.992 | 100% |
| 2× | 274 | 6 / 35 | 0.152 | 34% |
| 3× | 443 | 15 / 35 | 0.003 | 11% |
| 4× | 626 | 28 / 35 | 0.001 | 3% |

Thirty-four of 35 pairs collapse. At 2× only 6 of 35 pairs are truncated, yet the median falls from 0.992 to 0.152: the loss is dilution, not truncation, and it occurs well inside the model's input window. This is the long-premise weakness documented for summarization [16], observed here at chunk lengths of a few hundred tokens that are ordinary for RAG. An earlier control with short premises (scores 0.997 and 1.000) had failed to expose it, for exactly this reason.

**Sentence-level premises.** Splitting cited chunks into sentences and taking the maximum over sentences—the standard repair [16]—changes Table 6.

**TABLE 6.** Grounded fraction at 0.5 and median $\max p_{\mathrm{ent}}$ by premise granularity (31,926 NLI pairs; same answers, corpus and retrieval). Pools are the unit's own cited chunks: for sentence hypotheses the sub-answer's cited chunks (median 22.0 sentences at sentence granularity, $n=157$ sub-answers, range 3–204); for claim hypotheses the branch's cited chunks (median 19.5, $n=52$ branches, range 2–78).

| Hypothesis | Premise = chunk | Premise = chunk sentence |
|---|---|---|
| Sentence | 15.6% (median 0.002) | 38.4% (median 0.031) |
| Claim | 31.7% (median 0.006) | 57.1% (median 0.971) |

Nothing about the answers, corpus, or retrieval changed between the cells of Table 6; the estimator configuration alone moves the grounded fraction from 15.6% to 57.1%. The distribution remains bimodal (middle band 8.5%, 14.3%, 10.3%, 6.6% for the four cells in reading order); what changed is which mode the mass sits in.

**Negative control: max-aggregation inflation.** The claim × sentence-premise estimator compares each claim with every sentence of a premise pool and takes the maximum. For the control the pool is the topic's four branches' cited chunks combined: a median of 96 sentences per claim (claim-weighted; 83 per topic over the 15 topics). We assigned each claim the premise pool of a *different* topic (seed-fixed cyclic shift, zero same-topic assignments by assertion) and re-scored.

**TABLE 7.** Negative control (topic-shuffled premises) and chance-corrected gain. Pass criterion: shuffled grounded rate $\le 5\%$. E2 and E3 use topic-scoped pools (the four branches' cited chunks combined); the shuffle assigns the pool of a different topic. Median pool sizes are weighted by hypothesis instance ($n$ in the table). $\rho$ is Spearman correlation between pool size and $\max p$ under normal assignment; pool size takes one value per topic (15 distinct values), so no $p$-value is reported.

| Estimator | Hypothesis × premise | Median pool | $n$ | Grounded, normal | Grounded, shuffled | Net gain | $\rho$(pool size, $\max p$) | Pass |
|---|---|---|---|---|---|---|---|---|
| E1 (= $g_1$) | sentence × chunk | 7 | 294 | 15.6% | 0.0% | 15.6 pp | −0.085 | yes |
| E2 | claim × chunk-sentence, topic pool | 96 | 546 | 62.3% | 50.5% | 11.7 pp | −0.045 (shuffled: +0.293)† | no |
| E3 | claim × BM25 top-5 sentences | 5 | 546 | 53.5% | 15.2% | 38.3 pp | 0.000 | no |

† Under shuffled assignment, with each claim paired to the pool it was actually scored against, $\rho=+0.293$: larger foreign pools produce higher spurious maxima, which is the inflation mechanism stated directly.

Half of the claims find an "entailing" sentence in a corpus about a different topic. The arithmetic is that of multiple comparisons: with a per-pair false-positive rate near 0.7%, the probability that at least one of 96 unrelated sentences exceeds 0.5 is $1-(1-0.007)^{96}\approx 0.49$. Narrowing the pool to the five sentences most lexically similar to the claim (E3) reduces the shuffled rate to 15.2%—three times the pre-registered ceiling—and E3 was not adopted despite the largest net gain, because the criterion was fixed before the run and because a 15% floor of spurious grounding would raise every branch's score by that amount. Under the pre-registered rule, exactly one fallback configuration was permitted, and no third configuration of this family was tried. [[Characterize the 15.2% of claims that E3 grounds against a foreign topic: if they are generic or templated claims, say so here as evidence that the cross-encoder responds to surface similarity.]]

The inflation is visible without shuffling. The same claim × chunk-sentence estimator, applied to the same 546 claims, gives 57.1% against the branch's own cited chunks (Table 6; median 19.5 sentences per branch) and 62.3% against the four branches' chunks pooled (Table 7; median 96 per claim, 83 per topic)—a pool roughly four to five times larger. Nothing changed but the number of comparisons; the 5.1-point rise with genuine premises is the same effect that the shuffle measures at 50.5% with irrelevant ones. Chance-corrected, E2's gain (11.7 points) is *below* E1's (15.6): once inflation is removed, the repair is worse than the original. Grounded fractions from pools of different sizes are not comparable, and the paper reports each with its pool scope stated.

The standard estimator $g_1$ is the only configuration that passes the negative control: its shuffled rate is 0.0% and it shows no pool-size dependence. Read together with Table 5, $g_1$ is *precise but strict*: it does not hallucinate support, and it fails to see support that is present whenever the premise exceeds a couple of hundred tokens. The obvious repair trades that strictness for a false-positive rate the raw grounded fraction cannot reveal. Which of the two error directions dominates $g_1$'s near-zero mass—unseen support or genuine absence—is a question the controls cannot answer; Section VI-F answers it. Whether the strictness is a property of the checkpoint or of the family is a separate question; Section VI-G answers that one.

### F. Validity Against Human Labels (V′), and the Judge That Did Not Survive Them

The controls establish that $g_1$ does not hallucinate support and loses it under dilution, but they do not say which effect dominates the 76.5% of sentences it scores below 0.05. Two reference sets were collected. The first, an LLM judge (`gpt-5.4-nano`, fixed prompt, three labels) over 106 sentences stratified by $g_1$'s own score, reported precision 0.913 on sentences $g_1$ called supported and 72.5% support among sentences it called unsupported. The second, collected after the candidate operators were fixed, is 60 sentences labelled blind by one human under a three-value rubric (supported; unsupported, including partial support; unclear), with the full cited chunks shown for every sentence and no scores or strata visible; the sample is stratified by the candidate's and $g_1$'s labels (25 sentences the candidate recovers and $g_1$ rejects, 15 both accept, 15 both reject, 5 the candidate rejects and $g_1$ accepts). The human marked 49 supported, 11 unsupported, none unclear.

**The judge did not pass its gate.** Thirty-two of the 60 human-labelled sentences coincide with the judge's sample. Judge and human agree on 21 of 32 (65.6%, Wilson [48.3, 79.6]); the pre-registered gate was 90%. The disagreements are symmetric—five sentences the judge rejected and the human accepted, five the reverse, one the judge could not decide—which is noise rather than a systematic lean. The 32 sentences are an overlap, not the 25-item gate sample the protocol specified, so we report the agreement as observed and do not adjudicate the gate; the consequence is the same either way. Every figure in this paper that rested on judge labels is replaced below by its human-labelled counterpart, and the judge-referenced tables move to the supplement with this agreement figure attached. The replacement is made for provenance, not because the judge's figures were refuted: on the strata the two samples share, the human rates run in the same direction as the judge's and their intervals contain the judge's values (support among sentences $g_1$ accepted, 16 of 20 = 0.80 [0.58, 0.92] against the judge's 0.913; among sentences $g_1$ rejected, 9 of 10 = 0.90 [0.60, 0.98] against 0.725). What changes is that the reference is no longer a model. That a standard LLM-judge protocol for groundedness agrees with a human two times in three on this material is itself a result, and one that the RAG evaluation frameworks of Section II-D assume away.

**TABLE 8.** Operator validity against 60 blind human labels (49 supported, 11 unsupported). Precision over the sentences each operator scores as supported; recall over the 49 human-supported sentences; recovery over the 33 sentences $g_1$ scored as unsupported that the human marked supported. Wilson 95% intervals. Bars as pre-registered.

| Operator | Precision (bar ≥ 0.85) | Recall (bar ≥ 0.70) | Recovery of $g_1$'s misses (bar > 0.50) | Shuffled rate (bar ≤ 5%) |
|---|---|---|---|---|
| nli-deberta-v3-base ($g_1$) | 16/20 = 0.800 [0.584, 0.919] | 16/49 = 0.327 [0.212, 0.466] | 0/33 = 0.000 [0.000, 0.104] | 0.00% |
| bart-large-mnli | 31/39 = 0.795 [0.645, 0.892] | 31/49 = 0.633 [0.493, 0.753] | 18/33 = 0.545 [0.380, 0.702] | 1.02% |
| MiniCheck-Flan-T5-Large, single document | 34/40 = **0.850** [0.709, 0.929] | 34/49 = **0.694** [0.555, 0.805] | 22/33 = **0.667** [0.496, 0.802] | **0.00%** |

Three things are settled by the human labels. When $g_1$ reports support it is right four times in five (0.800), and the three operators' precision intervals overlap: they are similar when they say yes. They differ when they say no. Of the 33 sentences $g_1$ rejected that the human accepted, $g_1$ recovers none by construction and MiniCheck recovers 22; the intervals [0.00, 0.10] and [0.50, 0.80] do not touch. This is the human-labelled form of the diagnosis the judge had suggested: the deployed operator is precise and blind, and the mass it scores near zero is largely support it cannot see. Within the stratum that both operators reject, 11 of 15 sentences (73%) are nonetheless human-supported, so the accepted operator still misses a good deal; its recall of 0.694 says the same. Recovery comes at a small price on the other side: of the 16 human-supported sentences $g_1$ accepted, MiniCheck accepts 12. Across the five operators tested, MiniCheck is the only one whose recovery of $g_1$'s misses exceeds its false recoveries (net +23.8 points on denominators of 33 and 7; the full table is in the supplement, and the denominator of 7 fixes sign and rough size, not rank). The 33 recovery targets are defined by $g_1<0.5$; 30 of them fall below 0.05, the judge sample's N stratum, so the two analyses measure nearly the same population.

**The selection rule returned an inconclusive verdict, and we report it as such.** MiniCheck passes the control and three of the four human-label criteria; its recall of 0.694 falls 0.006 short of the pre-registered bar—one sentence in 49—with an interval that straddles the bar. Under the pre-registered rule this is neither acceptance nor rejection but a call for a second labelling stage, which we did not collect: the candidate leads every other operator on every criterion and trails on none, so further labels could not change which operator is chosen, and collecting labels until a bar is cleared is the practice this paper argues against. The operator is therefore *selected*, not *certified*, and every downstream figure carries that qualification. The claim-fidelity check bounds the claim-level rows of Tables 4 and 6: of 30 sampled branch answers the claim set was complete for 26, omitted content in 2, and added content in 0; the pipeline decomposes claims from the whole answer, so the check is at answer level, and zero additions is the direction that matters.

### G. Checkpoint Replacement: The Mechanisms Belong to the Checkpoint (D10)

Everything above was measured with one checkpoint, `cross-encoder/nli-deberta-v3-base`. To test whether dilution, binary scoring, and blindness are properties of sentence-pair NLI or of that checkpoint, we replaced the NLI model with three MNLI checkpoints of larger size—`deberta-v3-large-mnli` [[exact id]], `roberta-large-mnli`, and `facebook/bart-large-mnli`—and re-scored the same 294 sentences with the same premises, the same threshold, and the same topic-shuffled negative control as Table 7 (pool = the topic's cited chunks; shift $i\to i+7 \bmod 15$; ceiling 5%). The base checkpoint's row reproduces Table 7's E1 exactly (15.65% / 0.00% / $\rho=-0.085$), which validates the harness. Generation calls: zero.

**TABLE 9.** Checkpoint replacement on the same sentences, premises, and threshold (C3; $n=294$; topic-scoped pools). Grounded fraction under normal and shuffled assignment; chance-corrected gain; $\rho$(pool size, $\max p$) under normal assignment; retention of known-entailed pairs at 4× premise length (Table 5 protocol); middle-band mass (0.1–0.9).

| Checkpoint | Size | Grounded, normal | Grounded, shuffled | Net gain | $\rho$ | Retention at 4× | Middle band | Negative control |
|---|---|---|---|---|---|---|---|---|
| nli-deberta-v3-base (= $g_1$) | base | 15.65% | 0.00% | 15.65 pp | −0.085 | 11% | 8.5% | pass |
| deberta-v3-large-mnli | large | 37.76% | 21.09% | 16.67 pp | [[fill]] | [[fill]] | 19.7% | **fail** |
| roberta-large-mnli | large | 19.73% | 2.72% | 17.01 pp | [[fill]] | [[fill]] | 35.4% | pass |
| bart-large-mnli | large | 60.88% | 1.02% | 59.86 pp | −0.079 | 71% | 51.7% | pass |

Three things change. First, the mechanisms are checkpoint-specific. Retention at 4× premise length is 11% for the base checkpoint and 71% for `bart-large-mnli`; the dilution of Table 5 is a property of the deployed checkpoint, not of the family. The middle band, 8.5% for the base checkpoint, is 51.7% for `bart-large-mnli` and exceeds 20% for all three larger models: binary scoring, and with it the irrelevance of threshold calibration, is likewise a property of the checkpoint. Second, the raw grounded fraction remains a poor guide. `deberta-v3-large-mnli` more than doubles the fraction (37.8%) and fails the negative control at 21.1%: over half of its apparent gain is leniency. Only the shuffled control separates it from `bart-large-mnli`, which reaches 60.9% with a shuffled rate of 1.0% and no pool-size dependence—the same 60% that the sentence-split repair of Table 7 obtained by inflation, obtained here without it. Third, and consequently, the measurement of Section VI-A was decided by a checkpoint choice. On the same 294 sentences, the deployed checkpoint sees support in one in six and `bart-large-mnli` in three of five; neither result required any change to the system.

What Table 9 does not establish is validity. A high grounded fraction with a clean negative control is necessary, not sufficient; the estimator must also agree with reference labels on the system's own outputs. Against the human labels of Table 8, `bart-large-mnli` reaches precision 0.795 and recall 0.633 and recovers 18 of the 33 sentences $g_1$ missed: a real improvement within the same family, and behind the document-level verifier on every criterion. Whether a sentence-pair checkpoint exists that is both precise and sensitive on this material is thus answered in part—one is markedly less blind than the deployed one—and none tested matches a verifier built for claim-versus-document input.

### H. Premise Construction: A Second, Model-Dependent Axis (D13–D14)

Tables 6, 7, and 9 vary the hypothesis unit and the checkpoint. A third choice, how the premise is assembled from the cited chunks, has so far been treated as fixed. It is not innocuous. We measured, on the same 294 sentences and with $g_1$ unchanged, three constructions: the single best-matching chunk (*single*), all cited chunks concatenated (*union*), and each of these truncated to the $L$ chunk sentences most similar to the hypothesis by BM25 (*trunc*, $L\in\{64, 256\}$ tokens) [[confirm the L semantics: tokens vs. sentences]].

**TABLE 10.** Premise construction on the same sentences and checkpoint(s). Grounded fraction at $\tau_{\mathrm{g}}=0.5$; median premise tokens; negative control as in Table 7 [[fill shuffled rates when D14-A completes]].

| Construction | Median premise tokens | $g_1$ (base) grounded | bart-large-mnli grounded | Shuffled ($g_1$) | Net gain ($g_1$) |
|---|---|---|---|---|---|
| single chunk (deployed) | 142 | 15.6% | 60.5% | 0.0% | 15.6 pp |
| single, sentence-split + max | — | 38.4% | 28.2% | (inflates; Table 7) | — |
| single, trunc $L=64$ | 47 | 31.6% | [[fill]] | [[fill]] | [[fill]] |
| union of cited chunks | [[fill]] | [[fill]] | **72.8%** | [[fill]] | [[fill]] |
| union, trunc $L=64$ | 130 | 34.0% | [[fill]] | [[fill]] | [[fill]] |
| union, trunc $L=256$ | 613 | **67.0%** (median $p$ 0.987) | [[fill]] | [[fill]] | [[fill]] |

Three findings, two of which revise statements made earlier in this section.

**Dilution is a matter of irrelevant content, not of length.** Under *union, trunc 256*, the premise is 613 tokens—longer than the 4× padded premises of Table 5 that reduced $g_1$'s median to 0.001—and $g_1$'s median entailment is 0.987. Under padding, a 215-token premise half of which is irrelevant gives 0.108. The natural-length curve of the deployed pipeline (grounded fraction falling with chunk length, with a knee near [[181]] tokens [[fill from D13]]) is consistent with this reading: longer chunks contain proportionally more content unrelated to any one sentence. The correct statement of the mechanism is that the deployed checkpoint's score collapses with the *fraction* of the premise that is unrelated to the hypothesis. Table 5 measured that fraction by padding and reported it as a length effect; we correct the description here.

**The prescription changes with it.** "Keep premises short" is the wrong rule; it discards support. "Remove unrelated content from the premise" is the right one, and it is actionable: keeping only the chunk sentences most similar to the hypothesis raises the deployed checkpoint, unchanged, from 15.6% to 67.0%—a larger movement than replacing the checkpoint (Table 9, 60.9%). The shuffled-premise control separates the constructions. Truncating the single cited chunk to the $L=128$ most similar tokens passes (shuffled 1.36%; net gain 40.82 points, 2.6× the deployed construction's 15.65) without any change of checkpoint; every *union* construction fails (shuffled 6.5–11.2%), because lexical selection over a larger pool finds overlapping sentences in the wrong topic too. The premise-cleaning repair that survives is therefore single-chunk truncation, and the 67.0% of union-256 is reported as the raw fraction it is. Two configuration details are reported for reproducibility: the pre-registered $L=64$ is not the best setting (31.6% single, 34.0% union), and the protocol's requirement that *single* and *union* be truncated to equal token budgets cannot be met—union concatenates a median of three chunks and is roughly three times longer at any $L$ (47 vs. 130 tokens at $L=64$)—so the two are compared at equal $L$ rather than equal length.

**There is no model-independent right way to build a premise.** Sentence splitting raises the base checkpoint (15.6% → 38.4%) and lowers `bart-large-mnli` (60.5% → 28.2%). Union is the best construction for the base checkpoint (67.0% with truncation) and single is the best for `bart-large-mnli` (72.8%). The direction matches a positional asymmetry observed in D13: the base checkpoint weights the end of its input and `bart-large-mnli` the beginning; neither reads the whole premise. Earlier subsections of this paper, and much of the RAG evaluation literature, describe premise construction as a fixed design choice; it is a second axis, interacting with the first, and an estimator is validated only as a (checkpoint, construction) pair.

### I. The Pre-registered Gate, Its Failure, and Its Replacement (D9)

The decision rule fixed before Table 8 was that a false-negative rate at or above 0.5 would send the study to a different estimator class. The observed rate was 0.725 on judge labels; on human labels the deployed operator misses 33 of 49 supported sentences (recall 0.327). The reading that the answers lack evidence is rejected, and the re-execution question remains open. The pre-registered next estimator is a verifier trained on claim-versus-document input rather than sentence pairs [17], [18]: `MiniCheck-Flan-T5-Large`, run with the cited chunks as a single document, with an assertion on every call that no internal chunking occurred—and none did, on any of the calls. It is the only configuration in this paper with zero aggregation freedom. `bart-large-mnli` was evaluated under identical gates as a second candidate, entered after its negative-control result was known and disclosed as such (Section X).

**Under the gate as registered, no candidate passes.** Table 11 applies the positive-control gate—agreement with $g_1$'s 35 confident positives at 1× (bar 32/35) and retention of those positives at 4× premise length (bar 80%)—to every estimator in the paper.

**TABLE 11.** The pre-registered positive-control gate applied to all estimators (35 pairs with $g_1$ $p>0.90$; bars fixed before any candidate was seen). Negative control and human-referenced validity (Table 8) are shown alongside; the judge-referenced recovery columns are retained only for comparison with the judge's own gate result.

| Estimator | 1× agreement (bar 32/35) | 4× retention (bar 80%) | Gate | Shuffled rate | Net gain | N-stratum recovery (of 29) | N-stratum false recovery (of 10) | G retention (of 42) |
|---|---|---|---|---|---|---|---|---|
| nli-deberta-v3-base ($g_1$) | 35/35 | 4/35 = 11.4% | fail | 0.00% | 15.65 pp | — | — | — |
| roberta-large-mnli | 17/35 | 1/35 = 2.9% | fail | 2.72% | 17.01 pp | [[fill]] | [[fill]] | [[fill]] |
| deberta-v3-large-mnli | 30/35 | 15/35 = 42.9% | fail | 21.09% | 16.67 pp | [[fill]] | [[fill]] | [[fill]] |
| bart-large-mnli | 29/35 | 25/35 = 71.4% | fail | 1.02% | 59.86 pp | 18 (62.1%) | 7 (70.0%) | 37 (88.1%) |
| MiniCheck-Flan-T5-Large | 27/35 | 25/35 = 71.4% | fail | **0.00%** | **64.29 pp** | **19 (65.5%)** | **3 (30.0%)** | 35 (83.3%) |

MiniCheck fails the gate as registered, and we report it as failed; the bars were not moved. But the failure is instructive, because MiniCheck's loss under 4× padding is 5.7 points (77.1% → 71.4%) where $g_1$'s is 89 points (100% → 11.4%): the candidate does not have the defect the gate was designed to catch, and fails on the other criterion—disagreement with 8 of $g_1$'s 35 confident positives. That criterion is the problem. The positive set was defined by $g_1$, the estimator under audit, whose precision on such sentences is about 0.91 (Table 8); a perfect estimator would therefore be expected to agree with roughly 32 of 35, which is exactly where the bar was set, leaving no allowance for reference noise. The gate is jointly unsatisfiable in practice rather than in principle—$g_1$ itself passes the agreement bar only by construction and fails retention—and it is also underpowered: at $n=35$ the Wilson interval of 71.4% is [55, 84]% and includes the 80% bar, so the retention criterion could not have distinguished the candidates from the bar at any reasonable confidence. We used the audited estimator as the audit's reference. That is the methodological lesson this section adds to the paper, and it is reported rather than repaired in place.

**Replacement gate, registered before human labels exist.** The reference an estimator must be compared with is not another estimator but blind human judgment on the pipeline's own outputs at their natural premise lengths. The replacement gate therefore consists of (i) the negative control as before (shuffled rate ≤ 5%) and (ii) precision ≥ 0.85, recall ≥ 0.70, and N-stratum recovery > 50% against blind human labels on the 100-item sample described in Section X (N 50, G 30, M 20), with retention under padding reported as a diagnostic without a bar. The three validity criteria are the ones already registered for the judge-referenced sample; no new number is introduced. Both candidates are evaluated under both gates and all results appear in Table 11. Against the provisional judge labels, MiniCheck leads every criterion with power—shuffled rate 0.00%, net gain 64.3 points, N-stratum recovery 19 of 29, and false recovery 3 of 10 against `bart-large-mnli`'s 7 of 10—but the last comparison rests on ten sentences and is not a decision. Re-scoring of the main measurement waits for the human labels. [[D9 outcome under the replacement gate: fill one of the following.]]

[[If a candidate passes the replacement gate: Table 12 reports the re-scoring in two columns, standard versus accepted estimator—verification passes by round (both clauses), sufficient branches at $\tau\in\{0.7,0.8,0.9\}$, $W_\tau$, median grounding of non-empty branches—with the assertion that coverage and the empty-answer rate are unchanged, the caveat that the trajectory was produced by the standard verifier, and the GO/STOP decision (round-0 passes ≥ 5/50 or $W_{0.8}\ge 20\%$) for a fresh run with the accepted verifier in the loop.]]
[[If none passes: Table 12 is removed; Table 11 is the terminal result.]]

**TABLE 12.** [[Re-scoring under the accepted estimator — fill or remove.]]

---


### J. Summary: What Passes (TABLE 12)

**TABLE 12.** Candidate operators on the same 294 sentences (topic-scoped pools; shuffled-premise control ≤ 5%; reference-label validity [[V′]]). Net gain = normal − shuffled, from raw counts.

| Operator (checkpoint × premise) | Normal | Shuffled | Net | Precision (ref.) | Recall (ref.) | Status |
|---|---|---|---|---|---|---|
| nli-deberta-v3-base × cited chunk ($g_1$) | 15.65% | 0.00% | 15.65 pp | 0.800 | 0.327 | passes control; blind |
| nli-deberta-v3-base × single chunk, truncated to $L=128$ | [[fill]] | 1.36% | 40.82 pp | [[fill]] | [[fill]] | passes control |
| nli-deberta-v3-base × union of cited chunks, truncated | [[fill]] | 6.5–11.2% | — | — | — | fails control |
| roberta-large-mnli × cited chunk | 19.73% | 2.72% | 17.01 pp | [[fill]] | [[fill]] | passes control; blind |
| deberta-v3-large-mnli × cited chunk | 37.76% | 21.09% | 16.67 pp | — | — | fails control |
| bart-large-mnli × cited chunk | 60.88% | 1.02% | 59.86 pp | 0.795 | 0.633 | passes control |
| MiniCheck-Flan-T5-Large × single document, no chunking | 64.29% | 0.00% | 64.29 pp | **0.850** | **0.694** | passes control; selected |


## VII. RESULTS, PART 2 — THE RELATION STATE

**TABLE 13.** Relation-state controls (`bart-large-mnli`, $\tau_c=0.70$, $\tau_e=0.70$; criteria fixed before execution). Version 1 uses all claim pairs; Version 2 restricts to related pairs. Generation calls: 100 (synthetic negations and paraphrases).

| Control | Pairs | Criterion | Result | Verdict |
|---|---|---|---|---|
| V1 shuffled-topic edges, contradiction $>\tau_c$ (max over pairs) | 179 edges | ≤ 5% | 93.3% (same-topic: 91.2%) | fail |
| V1 pair level, contradiction $>\tau_c$ | 14,535 shuffled / 5,375 same-topic | — | 28.3% vs. 16.3% (medians 0.296 vs. 0.059) | shuffled higher |
| V1 agreement $>0.60$, same-topic edges | 68 | — | 0.0% (median 0.041) | dead |
| V2 N1: within-branch, within-sub-question pairs, contradiction $>\tau_c$ | 823 | ≤ 5% | **26.4%** (median 0.122; 16.2% above 0.95) | **fail** |
| V2 P1: claim vs. LLM negation, contradiction $>\tau_c$ | 46 | ≥ 80% | 97.8% (median 0.9995) | pass |
| V2 P2: claim vs. paraphrase, symmetric entailment $>\tau_e$ | 50 | ≥ 80% | 100.0% (median 0.9986) | pass |
| V2 N1 pairs, entailment $\ge 0.5$ | 823 | — | 1.2% (median 0.0030) | no agreement signal |

Three facts settle the module's status. Defined over all pairs, contradiction measures unrelatedness: claim pairs from different topics score higher than pairs from different perspectives on the same topic, at the edge level and at the pair level, and the maximum aggregation is not the cause (Section IV-D). Restricting to related pairs removes that confound and exposes a second one: among 823 pairs of facts drawn from the same answer to the same sub-question—related by construction, consistent by construction—26.4% exceed the contradiction threshold and 16.2% exceed 0.95. The model is not blind. It detects explicit negations (97.8%, median 0.9995) and paraphrases (100%, median 0.9986) almost perfectly. What it cannot do is distinguish a *different* fact from an *opposing* one, and among facts that answer the same sub-question, different facts are the common case. Claim attribution to sub-questions was deterministic and failed on none of 458 claims, so the failure cannot be assigned to the mapping.

At a false-positive rate of 26.4% per related pair, an edge between two perspectives—tens of related pairs—is flagged as conflicting almost always; of the six edges among four perspectives, roughly 5.5 would trigger VERIFY on every round, and localization would become exhaustive verification. No threshold rescues this (16.2% of pairs sit above 0.95), and no aggregation does (the pair-level rate is the problem). The relation module is therefore disabled in every system reported in Section VIII, and the pre-registered branch for this outcome—"no localization signal beyond random"—was taken with its mechanism corrected: the signal exists, and its precision is unusable. Entailment offers no substitute for agreement: within the same-sub-question pairs only 1.2% entail each other, so PRUNE by redundancy would never fire either. A relation defined on the same predicate—opposing values of the same attribute—rather than on any claim pair is the natural next definition and is not pursued here.

## VIII. RESULTS, PART 3 — HEADROOM AND SAVINGS UNDER A VALIDATED OPERATOR

### A. Is Re-execution Wasted? Headroom Under a Validated Operator (M1-C)

The main measurement's trajectory (Section VI-A) was re-scored under the accepted operator in a single pre-registered pass, with generation, retrieval, and citations untouched. Because the re-scoring uses cached round-0 artifacts, its population is the 45 non-pilot queries and their 180 round-0 branches; the standard operator's figures are recomputed on the same population so that the two columns share a denominator. The operator was selected under the human-label criteria of Section VI-F, meeting three of four with the fourth missed by 0.006; the re-scoring was executed once, from a pre-registration committed before the labels existed.

**TABLE 14.** Sufficient population at round 0, same 45 queries and 180 branches, standard versus accepted operator. Verification uses both clauses (Section III-C).

| Quantity | Standard $g_1$ | Accepted operator (MiniCheck, single document) | Ratio |
|---|---|---|---|
| Branches with $\mathrm{cov}=1.0$ | [[fill]] | 47 / 180 | — |
| Branches with $\mathrm{grd}>0.8$ | [[fill]] | 56 / 180 | — |
| Sufficient at $\tau=0.7$ | 3 / 180 (1.67%) | **30 / 180 (16.67%)** | 10× |
| Sufficient at $\tau=0.8$ | 1 / 180 (0.56%) | **19 / 180 (10.56%)** | 19× |
| Sufficient at $\tau=0.9$ | 1 / 180 (0.56%) | 15 / 180 (8.33%) | 15× |
| Queries passing verification (grounding clause / coverage clause / both) | 0 / 2 / 0 of 45 | 4 / 2 / **0** of 45 | — |
| Operator calls (all cached generations; internal chunking violations) | — | 719 (0 violations) | — |

The change of operator alone moves the sufficient population from one branch in 180 to nineteen (Fig. 1). Global verification still passes no query—the grounding clause is met by four queries and the coverage clause by two, never the same two—which is the situation Partial Rewind is designed for: a query that will not pass as a whole may contain branches that are done.

### B. Savings From Freezing on the Observed Trajectory (counterfactual lower bound)

With the sufficient population non-empty, we count what global retry would not have had to re-execute had it frozen the round-0 sufficient branches. The count is taken on the *observed* trajectory: 45 queries, 540 branch-rounds, 2,534,331 tokens, of which rounds 1–2 comprise 360 branch-rounds and 1,825,851 tokens. Frozen branches reuse their round-0 artifacts, so coverage and grounding are preserved by construction.

**TABLE 15.** Counterfactual savings from FREEZE at round 0 on the observed global-retry trajectory (45 queries). Calls are branch-rounds; the standard operator's column is the same count under $g_1$.

| $\tau$ | Frozen branches (accepted / $g_1$) | Branch-rounds saved | Calls saved | Tokens saved | Calls saved under $g_1$ |
|---|---|---|---|---|---|
| 0.7 | 30 / 3 | 60 / 540 | **−11.11%** | −13.77% | −1.11% |
| 0.8 | 19 / 1 | 38 / 540 | **−7.04%** | −8.72% | −0.37% |
| 0.9 | 15 / 1 | 30 / 540 | −5.56% | −6.92% | −0.37% |

At the primary threshold the accepted operator freezes 19 branches and saves 7.0% of branch-rounds and 8.7% of tokens; the standard operator freezes one and saves 0.4%—a nineteen-fold difference produced by the sufficiency judgment alone, with the pipeline, answers, retrieval, and citations identical. Two limits bound the reading. This is a counterfactual on a trajectory that global retry produced: freezing changes the verifier's inputs and therefore the later rounds, so a live controller follows a different trajectory, and the figure is a **lower bound on what freezing removes, not an estimate of Partial Rewind's cost**. And the savings counted here are branch-level; the live controller re-executes only deficient sub-questions of unfrozen branches (Section IV-B), which the counterfactual does not credit. The live comparison is Section VIII-C.

### C. Trajectory Selection: Global Retry Versus Freeze-and-Expand

Section VIII-B counted what freezing would have removed. This subsection lets the controller choose. Because the cache key of a branch execution depends on the perspective, the round, and the configuration—not on what happened in earlier rounds—every branch-round the reduced controller would select already exists from the main measurement, and the comparison runs at zero generation cost: the two systems draw different trajectories from the same pool of artifacts. Global retry re-executes all four branches whenever verification fails; the reduced controller freezes the branches that are sufficient at $\tau=0.8$ and re-executes the rest. The execution unit is the branch, not the sub-question, because sub-question-level artifacts do not exist in the cache; the controller of Section IV-B re-executes less than this, so the savings below are a lower bound on its cost advantage.

**TABLE 16.** Trajectory selection on cached artifacts (45 queries, one seed, accepted operator in the verifier). Cost and the two sufficiency components are shown first; the lower block reports metrics that play no part in the freeze decision. Confidence intervals are query-level bootstrap.

| | Global retry ($R=3$) | Freeze-and-expand | Difference (95% CI) |
|---|---|---|---|
| Generation calls (branch-rounds) | 540 | 498 | **−7.78%** |
| Tokens | 2,534,331 | 2,295,403 | **−9.43%** |
| Sub-questions re-executed | 2,268 | 2,104 | −7.23% |
| Branches frozen | 0 | 29 | — |
| Queries passing verification | 0 / 45 | 0 / 45 | — |
| *Selection criteria* | | | |
| Coverage | 0.3811 | 0.4228 | +0.0417 [+0.021, +0.065] |
| Grounding | 0.4390 | 0.4819 | +0.0430 [+0.018, +0.072] |
| Uncited rate | 0.3889 | 0.3722 | −0.0167 |
| *Independent of selection* | | | |
| Entity recall | 0.1693 | 0.1713 | +0.0020 [−0.0102, +0.0144] |
| Heading soft recall | 0.2304 | 0.2157 | −0.0147 [−0.0563, +0.0097] |
| Answer length (characters) | 3,365.0 | 3,499.8 | +134.8 [−102.4, +382.9] |

The cost reduction is unambiguous: 7.8% fewer generation calls and 9.4% fewer tokens for the same set of queries, with 29 branches preserved rather than recomputed.

The saving is not spread evenly, and the distribution matters as much as the mean (Fig. 4a). Twenty-three of the 45 queries have no sufficient branch at round 0 and are executed identically by both systems; the 7.8% comes entirely from the other 22, which freeze a median of one branch each and at most three. This is the behaviour the method should have: where nothing has been validated, freezing neither costs nor saves, and the controller degenerates to global retry rather than to something worse. It also sets the scale honestly—on this corpus, under this operator, about half the queries produce something worth keeping after one round, and the achievable saving is bounded by that fraction rather than by the controller.

The quality comparison must be read in two parts, and only the second part is evidence. Coverage and grounding rise significantly, but these are the two quantities the freeze decision selects on—branches are frozen because their coverage is complete and their grounding is high—so a system that preserves them scores higher on them by construction, and we do not claim the increase as a quality gain. The metrics that take no part in the decision show no difference: entity recall, heading soft recall, and answer length all have intervals containing zero. Answer length in particular rules out the reading that the frozen trajectory simply produces longer answers. The defensible statement is therefore that **freezing reduces cost at unchanged quality**, with the increase confined to the selection criteria. Heading soft recall points in the negative direction without reaching significance, which is consistent with re-execution occasionally widening a branch's section coverage even as it lowers coverage on average (Section VIII-A); we note the direction and do not interpret it further.

Fig. 4b traces a single query through both systems. One mechanism is worth stating because it was not anticipated. Global retry can overwrite a good round-0 branch with a worse retry: paired over queries, coverage falls by 0.060 from round 0 to the final round. Freezing removes that exposure. Whether it converts into a measurable gain on metrics outside the selection criteria is not established here—it did not—but it explains why cost can fall without quality falling with it.

### D. Quality–Cost Frontier (Test B) [[M3 실행 시에만]]

**TABLE 17.** Quality at matched cost. All systems use the same generator, retriever, corpus, and accepted operator. Budget sweep $B\in\{0.5,1,2\}\times B_0$, where $B_0$ is the call count of three-round global retry. Fig. 3 plots the frontiers.

| System | Calls (÷$B_0$) | Coverage | Grounding | Citation acc. | Entity recall | Judge win rate vs. global retry (swap-averaged, 95% CI) |
|---|---|---|---|---|---|---|
| Naive RAG | [[fill]] | | | | | |
| Self-RAG | [[fill]] | | | | | |
| Adaptive-RAG | [[fill]] | | | | | |
| STORM-like (no retry) | [[fill]] | | | | | |
| Global retry ($R=3$) | 1.00 | | | | | — |
| Partial Rewind, $B=0.5B_0$ | [[fill]] | | | | | |
| Partial Rewind, $B=B_0$ | [[fill]] | | | | | |
| Partial Rewind, $B=2B_0$ | [[fill]] | | | | | |

[[결과 문장(고정): "At the global-retry budget, Partial Rewind matches global retry on every quality metric within [[Δq]] and its judge win rate is [[w]]% [CI]; at half that budget it retains [[…]]." / 부정: "Partial Rewind does not dominate global retry at any budget; the mechanism localizes correctly (Section VI-B) but the preserved branches contribute [[…]]."]]

### E. Ablation (Test D)

**TABLE 18.** Components removed one at a time at $B=B_0$.

| Variant | Calls | Coverage | Grounding | Judge win vs. full | Localization acc. |
|---|---|---|---|---|---|
| Full | | | | — | |
| − FREEZE (re-evaluate sufficient branches anyway) | | | | | |
| − relation (no VERIFY, no PRUNE) | | | | | |
| − EXPAND | | | | | |
| − VERIFY | | | | | |
| − PRUNE | | | | | |
| Single perspective | | | | | |

[[결과 문장(고정): "Removing FREEZE returns cost to that of global retry with no quality gain; removing the relation module costs [[…]] in localization and [[…]] in quality, so cross-perspective relations carry signal beyond the evidence states." / "Removing the relation module leaves quality and cost unchanged within noise; the savings of Partial Rewind come from freezing and sub-question-level expansion, and the cross-perspective claim is withdrawn from the title."]]

### F. Graph Behaviour (Test E)

$|\Delta G|$ per step, frozen-branch reuse rate, actions by topic difficulty, convergence steps, and the sweep $n\in\{2,4,8\}\times$ pairing $\in\{$full, blocked$\}$ [[fill]]. Three failure trajectories are shown in the supplement: [[wrong freeze / unresolved conflict / budget exhaustion before convergence]].

---

## IX. DISCUSSION

**Sufficiency is a classifier, and a control loop is only as good as its classifier.** The first measurement in this paper would have been reported, under a less careful protocol, as evidence that re-execution is not wasted. It was evidence that the operator could not see support. Every adaptive RAG system that retries, expands, or stops on an evidence-sufficiency judgment inherits this dependence, and the two controls we use—shuffled-evidence false positives and human-labelled recall on the system's own outputs—are cheap enough to be standard.

**Minimal mutation is a statement about state, not about caching.** Lineage systems reuse computations whose inputs are unchanged; Partial Rewind preserves branches whose *evidence* is validated and mutates where the evidence state, not the input, says the graph is deficient. The monotone frozen set and the explicit $\Delta G$ make the difference measurable: $|\Delta G|$ per step against the size of a restart.

**Cross-perspective contradiction is a narrower signal than the literature suggests.** Defined over all claim pairs, as conflict-driven systems implicitly do, it tracks unrelatedness. Restricted to related pairs it [[passes/fails]] our controls. Where it works, its value is positional: it says which evidence channel to verify; where it does not, coverage gaps and redundancy still localize, and FREEZE still saves the cost. [[Test D 결과에 따라 한 문장]]

**Limitations of the mechanism.** A wrong freeze fixes a wrong answer; the accepted operator's measured false-positive rate bounds how often this happens and we report the bound rather than assume it away. Unresolved conflicts are reported, not adjudicated. The controller is rule-based with fixed thresholds; a learned policy is future work, and the traces released here (with a reward field left null) are its training data.

---

## X. THREATS TO VALIDITY

**Construct.** Quality is measured by automatic metrics and an LLM judge; no human rates the answers. The judge is from the same vendor as the generator; self-preference cancels between systems that share the generator, and the judge is validated on synthetic pairs before use. Grounding is measured by the accepted operator whose error profile is reported; conclusions about quality that depend on it are stated with that profile.

**Internal.** All systems share generator, retriever, corpus, prompts, and operator, so differences are attributable to control. Thresholds were fixed before each experiment. Three departures from pre-registration are disclosed: two candidate operators (`bart-large-mnli`, and the truncated-premise construction that was *not* adopted) were examined after partial results were known; the positive-control gate for operators was replaced after it failed every candidate, on grounds independent of any candidate's result; and the relation state was redefined once after its first definition failed a control. In each case both versions are reported and no further revision was permitted.

**External.** One generator, one corpus, [[N]] topics, one relation checkpoint. The uncited-answer rate (35% at round 0) is a behaviour of one generator under one prompt. The empty sufficient population under the standard operator is specific to a common default checkpoint; larger checkpoints of the same family behave differently, which is why the operator is validated rather than assumed.

**Scope of the cost comparison.** Three limits bound Sections VIII-B and VIII-C. The execution unit is the branch: sub-question-level artifacts were never generated, so the controller specified in Section IV-B—which re-executes only deficient sub-questions of unfrozen branches—is not what was measured, and both figures understate its saving. No generation occurred in either comparison; both select trajectories from the artifacts the main measurement produced, so neither is a live run, and a live run under the accepted verifier—with different verification outcomes and therefore different later rounds—remains outstanding. And the value of perspective decomposition itself is not measured anywhere in this paper: every system reported uses four perspectives, no single-perspective condition was run, and the 33.5% uncited rate at round 0 with mean coverage 0.474 indicates the decomposition is not obviously well-calibrated for this corpus. The paper's claims concern what to do with a multi-perspective execution once it exists, not whether to decompose.

**Reference labels.** Sixty sentences were labelled by a single human, blind to scores and strata; there is no second annotator and no inter-annotator statistic. The LLM judge, checked on the 32 sentences it shares with that sample, agrees with the human on 65.6% and is not used as a reference anywhere in the body. The operator-selection rule returned an inconclusive verdict on recall (0.694 against a bar of 0.70) and the second labelling stage it prescribes was not collected; Section VI-F states why, and the operator is described throughout as selected rather than certified.

**Conclusion.** The savings of Section VIII-B are counterfactual: they are counted on the trajectory that global retry actually produced, and a controller that freezes branches changes the verifier's inputs and hence the trajectory, so the figure bounds from below what freezing removes and is not a cost estimate for Partial Rewind; the live comparison of Section VIII-C is the estimate, and it is run with a single seed. Confidence intervals are bootstrap over queries; localization accuracy is per-topic with [[N]] topics per fault type; the human-labelled validation sample is [[60/100]] sentences. Waste ratios under the re-scored trajectory describe a trajectory generated by the standard verifier; the fresh run is the estimate we report.

---

## XI. CONCLUSION

We asked how much of the re-execution in multi-perspective RAG is unnecessary, and building a controller to avoid it turned the question into one about measurement. The sufficiency operator that decides which branches are done—a sentence-pair NLI model, the standard choice—judged three of six hundred branch executions sufficient and enabled no selective re-execution at all. Validated against blind human labels it is right four times in five when it reports support and recovers none of the 33 sentences it scored as unsupported that a human marked supported. The blindness tracks the fraction of unrelated content in the premise rather than its length: a 613-token premise of relevant sentences scores 0.987 where a 215-token premise half of which is irrelevant scores 0.108. It belongs to a particular checkpoint rather than to sentence-pair inference as such—a larger checkpoint of the same family retains support under four-fold padding where the deployed one loses it—and its two obvious repairs, sentence splitting with maximum aggregation and a checkpoint that doubles the raw score, both fail a shuffled-evidence control that a claim-versus-document verifier passes. The lesson that generalizes beyond this pipeline is procedural: a grounding score used as a control signal must be validated as a classifier in both error directions, and it cannot be validated against itself. The one gate here that used the deployed checkpoint's confident positives as ground truth failed every candidate, that checkpoint included.

The second negative result concerns the signal the architecture was designed around. Contradiction between perspectives, computed with an off-the-shelf inference model, detects synthetic contradictions and paraphrases almost perfectly and labels a quarter of compatible fact pairs drawn from a single answer as contradictory. At that false-positive rate no threshold or aggregation yields a usable trigger, and defined over all claim pairs the score tracks unrelatedness rather than conflict—pairs from different topics score higher than pairs from different perspectives on the same topic. The relation module is therefore disabled in every system we report. The failure is one of precision, not of sensitivity, and a relation defined on opposing values of the same predicate, rather than on arbitrary claim pairs, remains open.

What survives is the part of the architecture that does not depend on that signal, and it works. Under an operator that passes the shuffled-evidence control and three of four human-label criteria—selected, not certified—the sufficient population rises from one branch in 180 to nineteen; a controller that freezes those branches and re-executes the rest reduces generation calls by 7.8% and tokens by 9.4%, with entity recall, heading recall, and answer length unchanged. Coverage and grounding also rise, but they are the criteria the freeze decision selects on and we do not count them as a quality gain. The saving concentrates in the half of the queries that have something worth keeping, and it is a lower bound twice over: the execution unit measured is the branch rather than the sub-question the method specifies, and neither comparison generated new text—both select trajectories from artifacts the main measurement produced. A live run under the validated verifier, and a measurement of whether perspective decomposition earns its cost at all, are the immediate next steps.

The formal contribution stands independently of these figures. Execution as $G_t=(V_t,E_t,S_t)$ with an explicit mutation $\Delta G_t$, a monotone frozen set, and a termination bound of $B+2n$ control steps gives partial re-execution a definition that separates it from caching: what is preserved is a branch whose evidence has been validated, not a computation whose inputs have not changed. That distinction is what makes the sufficiency operator load-bearing, and why a paper about a control architecture spent most of its measurements on the instrument. We release the protocol, the code, the traces, and the scripts that check every figure in this paper against its raw data.

## REFERENCES

[1] A. Asai et al., "Self-RAG: Learning to retrieve, generate, and critique through self-reflection," ICLR 2024. arXiv:2310.11511.
[2] S.-Q. Yan et al., "Corrective retrieval augmented generation," arXiv:2401.15884, 2024. [verify]
[3] S. Jeong et al., "Adaptive-RAG," NAACL 2024. arXiv:2403.14403.
[4] Y. Shao et al., "Assisting in writing Wikipedia-like articles from scratch with large language models," NAACL 2024. arXiv:2402.14207.
[5] S. Li and N. Ramakrishnan, "Experience as a compass: Multi-agent RAG with evolving orchestration and agent prompts (HERA)," arXiv:2604.00901, 2026.
[6] Y. Lu et al., "DyTopo: Dynamic topology routing for multi-agent reasoning via semantic matching," arXiv:2602.06039, 2026.
[7] [[additional orchestration reference — optional]]
[8] M. Li et al., "S2G-RAG: Structured sufficiency and gap judging for iterative retrieval-augmented QA," arXiv:2604.23783, 2026.
[9] Q. Dong et al., "Stateful evidence-driven retrieval-augmented generation with iterative reasoning," arXiv:2604.14170, 2026.
[10] Wu et al., "MA-RAG," ICML 2026. arXiv:2603.03292. [verify]
[11] Y. Zhu et al., "ConRAG: Consensus-driven multi-view retrieval for multi-hop question answering," arXiv:2605.28093, 2026.
[12] Nguyen et al., "MA-RAG: collaborative chain-of-thought," arXiv:2505.20096, 2025. [verify]
[13] AgentLocate, arXiv:2607.07989, 2026. [verify]
[14] VerifyMAS, arXiv:2605.17467, 2026. [verify]
[15] Execution lineage for agent workflows, arXiv:2605.06365, 2026. [verify]
[16] SELENE / debate gating [verify]
[17] W. Kryscinski et al., "Evaluating the factual consistency of abstractive text summarization," EMNLP 2020.
[18] O. Honovich et al., "TRUE: Re-evaluating factual consistency evaluation," NAACL 2022.
[19] P. Laban et al., "SummaC: Re-visiting NLI-based models for inconsistency detection in summarization," TACL 2022.
[20] Y. Zha et al., "AlignScore: Evaluating factual consistency with a unified alignment function," ACL 2023.
[21] L. Tang, P. Laban, G. Durrett, "MiniCheck: Efficient fact-checking of LLMs on grounding documents," EMNLP 2024. [verify]
[22] S. Es et al., "RAGAS: Automated evaluation of retrieval augmented generation," EACL 2024 demo. [verify]
[23] J. Saad-Falcon et al., "ARES," NAACL 2024. [verify]

<!--
그림: FIGURE 1 충분 branch 수(M1-C, g₁ vs 확정 추정자 × τ 3점, 막대 위 개수) — 완성 · FIGURE 2 아키텍처(Query→P_i→E_i→Localize→{FREEZE|EXPAND}→ΔG→partial re-eval→loop; 관계 모듈 비활성) — 채팅 작성 예정 · FIGURE 3 Pareto — Test B 미실행이므로 삭제 · FIGURE 4 (a) 질의당 얼린 branch 분포 (b) t005 궤적 — 완성. 그래픽 초록 = FIGURE 2 축약.
캡션(확정):
 FIG. 1. Sufficient branches at round 0 under the standard and accepted operators (45 queries, 180 branches; counts above bars). Provisional pending human-gated labels.
 FIG. 2. [[아키텍처 — 채팅]]
 FIG. 4. (a) Distribution of branches frozen per query. Twenty-three of 45 queries freeze none; the saving comes from the remaining 22, which freeze a median of one. (b) One query (t005) through both systems: of four branches one is frozen and calls fall from 12 to 10. (b) is an illustration, selected as the median of the queries that freeze at least one branch with ties broken by query id; representativeness is carried by (a).
부록: 별도 파일 appendices.md — A 프롬프트 14(AST 추출) · B 토픽 예시 3(시드 42 무작위) · C 감사 축약(채팅 작성) · D 임계 민감도 · E leakage · F 재현(MiniCheck 동명이인 포함). 본문은 참조만 한다.
-->
