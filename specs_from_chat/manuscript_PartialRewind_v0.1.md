# Partial Rewind: Localize, Freeze, and Mutate for Evidence-Driven Partial Re-execution in Multi-Perspective Retrieval-Augmented Generation

<!--
==================== 공저자 노트 (원고 아님) ====================
버전: v0.1 (2026-09-11). LFM 논문 초안. 결과에 의존하지 않는 절은 전부 완성, 결과 절은 표 골격 + 실행 전 확정 문장.
결정(채팅이 위임받아 확정): ① 시스템명 Partial Rewind(본문 약어 없음; 코드 내부명 lfm/ 유지, 본문 등장 금지) ② 기여 3개를 위험 분산형으로 재편(§I) ③ 관계 모듈은 구성요소+ablation ④ IEEE Access.
채움 규칙: [[ ]]는 에이전트가 results/에서 채움. [[prov.]]는 사람 라벨 전 잠정. 모든 %p 차이는 원행 비율끼리 뺀 값만.
임계 경로: 사용자 라벨 60(+40) → V′ → M1-C → GO → M1′(시드 3) ∥ 관계 v1.1 통제 → Test C-1 → Test B/D/E.
표 번호는 확정. 그림: Fig.1 = M1′ 낭비율(또는 M1-C), Fig.2 = 아키텍처, Fig.3 = Test B Pareto, Fig.4 = trajectory 예시.
==================================================================
-->

**Abstract** — Multi-perspective retrieval-augmented generation decomposes a complex query into perspectives, answers each from retrieved evidence, and, when verification of the assembled answer fails, re-executes every perspective. We ask how much of that re-execution is unnecessary and whether it can be avoided without loss. We define a branch-level waste ratio—the fraction of re-executed branches that were already sufficient at the moment of failure—and find that measuring it requires a sufficiency operator whose errors are known: the sentence-pair inference model commonly used for grounding could not see three quarters of the support present in cited evidence, so no branch was ever judged sufficient. We validate a replacement operator against blind human labels under controls in both error directions, re-measure, and find a waste ratio of [[W]] percent. We then present Partial Rewind, a control architecture that keeps the evidence state of every perspective and, on failure, localizes the deficient sub-questions, freezes the branches that are sufficient, and mutates only the affected subgraph—expanding retrieval where coverage is missing, verifying where perspectives conflict, and pruning what is redundant—under a budget with a proved termination bound. Against global retry and adaptive baselines on [[N]] cross-domain topics over a pinned Wikipedia snapshot, Partial Rewind matches answer quality within [[Δq]] while reducing generation calls by [[Δc]] percent. Ablations show that [[relation outcome sentence]]. We release the protocol, code, and traces.

**Index Terms** — retrieval-augmented generation, multi-agent systems, partial re-execution, execution graphs, evidence sufficiency, groundedness evaluation.

---

## I. INTRODUCTION

Retrieval-augmented generation (RAG) has moved from a single retrieve-then-generate pass toward systems that decide what to execute and when: on-demand retrieval with self-reflection [1], corrective retrieval [2], query-complexity routing [3], and multi-perspective decomposition, in which a complex query is split into perspectives that are researched in parallel and then assembled [4]. Multi-agent variants add orchestration over these branches [5]–[7]. What these designs share is a recovery policy: when verification of the assembled answer fails, the workflow is re-executed. Every perspective is re-run, including those whose evidence was already adequate.

Re-execution is where the cost of these systems concentrates. In the pipeline studied here, 72% of billed tokens were spent in retry rounds, and those rounds left coverage slightly lower and the share of uncited answers unchanged. Whether the re-executed branches needed re-executing is a question we could not find measured. We define the quantity directly—the *branch-level waste ratio*, the fraction of re-executed branches that were already sufficient when verification failed—and set out to measure it. The first measurement returned 0.5%, and it was uninformative: no query passed verification at any round, and only three of six hundred branch executions were judged sufficient, because the operator that decides sufficiency—a grounding score from a sentence-pair natural-language-inference (NLI) model—could not see support that was present in the cited chunks. Validated against blind human labels, that operator was precise when it reported support and wrong three times in four when it reported its absence. A control loop built on it would freeze nothing and re-execute everything; it would be global retry with extra steps.

This paper therefore does two things in order. It first establishes a sufficiency operator whose error profile is known, using a compact validation protocol—negative controls against topic-shuffled evidence, human-labelled precision and recall on the system's own outputs—and re-measures the waste ratio with it. It then presents **Partial Rewind**, a control architecture for multi-perspective RAG that treats the execution as a graph whose nodes carry evidence state, and that recovers from failed verification by *localizing* the deficient sub-questions and conflicting perspective pairs, *freezing* the branches that are sufficient, and *mutating* only the affected subgraph: expanding retrieval for uncovered sub-questions, verifying conflicting claim pairs through a targeted verification subgraph, and pruning branches that another branch already covers. The unit of re-execution is the sub-question, not the branch; the recovery is a minimal graph mutation $\Delta G$ rather than a restart; and the loop terminates under a budget by a bound we prove.

Prior work uses semantic conflict to trigger further reasoning or retrieval, attributes failures post hoc, or replays computations invalidated by changed inputs. We instead localize insufficiency and inconsistency at the level of sub-questions and perspective pairs and selectively mutate only the affected execution subgraph while preserving validated branches.

The contributions are as follows.

1. **A validated sufficiency operator as control state.** We show that the standard grounding estimator cannot support a freeze decision in this setting (zero sufficient branches, zero selective re-executions in 600 branch executions), specify a validation protocol that costs no generation calls beyond a human-labelled sample, and use it to select an operator with measured precision and recall (Section IV-A). [[V′ 결과]]
2. **Minimal graph mutation with a proved bound.** We formalize the execution as $G_t=(V_t,E_t,S_t)$ with a four-action space whose structural effects are explicit, define the mutation $\Delta G_t$ and its size, localize at sub-question granularity, and prove termination under a call budget (Sections III, IV-B–E).
3. **Partial re-execution matches global retry at lower cost.** [[Test B: quality within Δq, calls −Δc%]] with localization accuracy [[Test C-1]] against oracle and random controllers, and an ablation isolating the contribution of cross-perspective relations (Section VI).

A methodological finding accompanies the second contribution: a cross-perspective contradiction score defined over all claim pairs, as in prior conflict-driven systems, measures unrelatedness rather than conflict—topic-shuffled pairs scored *higher* than same-topic pairs—and only a definition restricted to related claim pairs passes a control (Section IV-D). We report both.

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

We do not optimize a weighted sum of quality and cost. For each policy we sweep $B$ and report the quality–cost frontier; policies are compared on the same plane (Section VI-D).

---

## IV. METHOD

Fig. 2 shows the loop: perspectives are executed once; their evidence states and pairwise relation states are computed; the controller localizes, freezes what is sufficient, and mutates the affected subgraph; only the affected nodes are re-evaluated; the loop repeats until every branch is frozen, pruned, or unresolved, or the budget is exhausted.

### A. The Sufficiency Operator and Its Validation

Every decision above rests on $S_\tau$, and $S_\tau$ rests on $g$. We found that the standard choice for $g$—a sentence-pair NLI cross-encoder (`cross-encoder/nli-deberta-v3-base`) with the cited chunk as premise—cannot support the freeze decision in this setting, and we report the evidence briefly because it determines the rest of the design.

**Under the standard operator, FREEZE never fires.** In the first measurement (50 topics, global retry, 600 branch executions; Section VI-A), all 50 queries re-executed all four branches in every round; the grounding clause of verification passed 0 of 50 at every round and the coverage clause 2 of 50 at round 0; three branch executions were sufficient at $\tau=0.8$. The measured waste ratio, 0.5%, reflected the operator, not the system.

**The operator was precise and blind.** A stratified sample of the operator's own outputs, judged by an LLM against the full cited chunks and gated by blind human agreement [[prov.: human gate pending]], gave precision 0.913 (42 of 46) when the operator reported support and a false-negative rate of 0.725 (29 of 40) when it reported its absence. The cause is dilution: on 35 pairs the operator scored above 0.90, extending the premise with unrelated chunks to 2× its length (274 tokens, 6 of 35 truncated) dropped the median score from 0.992 to 0.152. Supplying the gold article to the retriever raised coverage but not grounding; the answers were grounded and the operator could not see it.

**Validation protocol.** A candidate operator is accepted only if it passes (i) a negative control—the shuffled-premise rate, obtained by scoring each sentence against the cited chunks of a *different* topic, must not exceed 5%—and (ii) human-labelled validity on the system's own outputs: precision $\ge0.85$ on sentences the candidate scores as supported, recall $\ge0.70$ on sentences a blind human marks supported, and recovery above 50% of the sentences the standard operator missed. The first control is decisive on its own for several candidates: a larger checkpoint of the same family that doubled the raw grounded fraction (37.8%) accepted 21.1% of shuffled premises, and sentence-splitting with maximum aggregation reached 57–62% grounded with a 50.5% shuffled rate from multiple comparisons. Raw grounded fractions and thresholds say nothing about validity.

**TABLE 1.** Candidate sufficiency operators on the same 294 sentences, premises, and threshold. Negative control with topic-shuffled premises (ceiling 5%); human-labelled validity [[V′ 결과로 채움]].

| Operator | Grounded, normal | Grounded, shuffled | Net | Precision (human) | Recall (human) | Recovery of $g_1$ misses | Accepted |
|---|---|---|---|---|---|---|---|
| nli-deberta-v3-base ($g_1$, standard) | 15.6% | 0.00% | 15.6 pp | [[fill]] | [[fill]] | — | no (recall) |
| roberta-large-mnli | 19.7% | 2.72% | 17.0 pp | [[fill]] | [[fill]] | [[fill]] | [[fill]] |
| deberta-v3-large-mnli | 37.8% | 21.09% | 16.7 pp | — | — | — | no (control) |
| bart-large-mnli | 60.9% | 1.02% | 59.9 pp | [[fill]] | [[fill]] | [[fill]] | [[fill]] |
| MiniCheck-Flan-T5-Large, single document, no internal chunking | 64.3% | 0.00% | 64.3 pp | [[fill]] | [[fill]] | [[fill]] | [[fill]] |

The accepted operator is [[MiniCheck-Flan-T5-Large / bart-large-mnli]], used with the cited chunks concatenated as a single premise so that no internal chunking or maximum aggregation occurs; an assertion on every call enforces this. Two details matter for reproduction: the relation module (Section IV-D) requires three-way NLI output, which a document verifier does not provide, so it uses `bart-large-mnli` regardless of the choice of $g$; and a pre-registered positive control that used $g_1$'s confident positives as reference failed every candidate including $g_1$, and was replaced by the human-labelled criteria above—an estimator cannot be the reference for its own replacement. The full audit is in the supplementary material.

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

An edge with $\mathrm{Rel}_{ij}=\emptyset$ carries no signal; this is recorded and distinguished from "no conflict". Version 2 is used only if it passes three controls fixed before it was run: claim pairs from *within* one branch and one sub-question—related by construction and consistent because they come from one answer—must exceed $\tau_c$ on at most 5% of pairs; claims paired with LLM-generated negations must be detected as contradictory at least 80% of the time; claims paired with paraphrases must be scored as mutually entailing at least 80% of the time. [[v1.1 통제 결과: N1 / P1 / P2]] If Version 2 fails, the relation edges are removed and the controller runs on evidence states alone; Section VI-E ablates the module either way.

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

## VI. RESULTS

### A. Is Re-execution Wasted?

**TABLE 2.** Waste ratio under global retry (50 topics, up to 3 rounds). Round 1: standard operator $g_1$. Re-scored: the same trajectory scored by the accepted operator (pre-registered, one pass). Fresh: global retry re-run with the accepted operator in the verifier, three seeds.

| Quantity | Round 1 ($g_1$) | Re-scored (accepted $g$) | Fresh run (accepted $g$, 3 seeds) |
|---|---|---|---|
| Queries passing verification (round 0 / any) | 0 / 0 of 50 | [[fill]] | [[fill]] |
| Sufficient branch executions at $\tau=0.8$ | 3 / 600 | [[fill]] | [[fill]] |
| $W_{0.7}$ / $W_{0.8}$ / $W_{0.9}$ | — / 0.5% [0.0, 1.5] / — | [[fill]] | [[fill]] |
| Share of calls in retry rounds | 72% (tokens) | — | [[fill]] |
| Coverage, round 0 → final (paired) | −0.060 (CI excl. 0; sign $p=0.144$) | — | [[fill]] |
| Empty (uncited) branches by round | 33.5% / 37.0% / 37.0% | — | [[fill]] |

Under the standard operator the question cannot be answered (Section IV-A). Under the accepted operator, [[결과 문장 A: "W_{0.8} = x% [CI]: y of every ten re-executed branches were already sufficient" / 결과 문장 B: "the sufficient population remains below z%, and Partial Rewind's headroom in this regime is correspondingly small; Sections VI-B–D therefore test the mechanism rather than its savings"]]. Independently of the operator, retry rounds consumed 72% of tokens, left coverage lower on average, and did not reduce the share of uncited answers.

### B. Localization Accuracy (Test C-1)

One fault is injected per topic at round 0—a contradicting claim with a fabricated supporting chunk in one branch, a deleted citation set for one sub-question, or a paraphrased duplicate of one branch into another—so that the correct first action (type and target) is known. Localization accuracy is the fraction of topics on which the controller's first action matches the oracle in type and target (type-only matches score 0.5).

**TABLE 3.** Localization accuracy by fault type ([[N]] topics × 3 faults). Bootstrap 95% CIs.

| Controller | Contradiction → VERIFY$(i,j)$ | Coverage gap → EXPAND$(i,Q')$ | Redundancy → PRUNE | Overall |
|---|---|---|---|---|
| Oracle | 1.00 | 1.00 | 1.00 | 1.00 |
| Partial Rewind | [[fill]] | [[fill]] | [[fill]] | [[fill]] |
| Global verifier (re-execute all) | [[fill]] | [[fill]] | [[fill]] | [[fill]] |
| Random | [[fill]] | [[fill]] | [[fill]] | [[fill]] |

Relation extraction alone: contradiction detection precision/recall/F1 on injected versus within-branch pairs [[fill]]; agreement on paraphrase pairs [[fill]].

[[결과 문장(고정): 통과 — "The location of a cross-perspective contradiction points to the failed evidence channel: the rule-based controller matches the oracle's first action on x% of topics, against y% for random selection and z% for global re-execution." / 탈락 — "Partial Rewind localizes coverage gaps and redundancy above chance but does not localize contradictions better than random; the relation module is disabled in the remaining experiments and the title's claim is reduced accordingly."]]

### C. Natural Conflicts (Test C-2)

On [[20]] topics derived from articles with high reference-risk annotations, [[k]] perspective pairs exceeded $\tau_c$ under Version 2; blind human inspection of [[m]] of them found [[p]]% to be genuine disagreements between perspectives, [[q]]% retrieval or estimator artifacts. [[예시 1개: disagreement report 발췌]]

### D. Quality–Cost Frontier (Test B)

**TABLE 4.** Quality at matched cost. All systems use the same generator, retriever, corpus, and accepted operator. Budget sweep $B\in\{0.5,1,2\}\times B_0$, where $B_0$ is the call count of three-round global retry. Fig. 3 plots the frontiers.

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

**TABLE 5.** Components removed one at a time at $B=B_0$.

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

## VII. DISCUSSION

**Sufficiency is a classifier, and a control loop is only as good as its classifier.** The first measurement in this paper would have been reported, under a less careful protocol, as evidence that re-execution is not wasted. It was evidence that the operator could not see support. Every adaptive RAG system that retries, expands, or stops on an evidence-sufficiency judgment inherits this dependence, and the two controls we use—shuffled-evidence false positives and human-labelled recall on the system's own outputs—are cheap enough to be standard.

**Minimal mutation is a statement about state, not about caching.** Lineage systems reuse computations whose inputs are unchanged; Partial Rewind preserves branches whose *evidence* is validated and mutates where the evidence state, not the input, says the graph is deficient. The monotone frozen set and the explicit $\Delta G$ make the difference measurable: $|\Delta G|$ per step against the size of a restart.

**Cross-perspective contradiction is a narrower signal than the literature suggests.** Defined over all claim pairs, as conflict-driven systems implicitly do, it tracks unrelatedness. Restricted to related pairs it [[passes/fails]] our controls. Where it works, its value is positional: it says which evidence channel to verify; where it does not, coverage gaps and redundancy still localize, and FREEZE still saves the cost. [[Test D 결과에 따라 한 문장]]

**Limitations of the mechanism.** A wrong freeze fixes a wrong answer; the accepted operator's measured false-positive rate bounds how often this happens and we report the bound rather than assume it away. Unresolved conflicts are reported, not adjudicated. The controller is rule-based with fixed thresholds; a learned policy is future work, and the traces released here (with a reward field left null) are its training data.

---

## VIII. THREATS TO VALIDITY

**Construct.** Quality is measured by automatic metrics and an LLM judge; no human rates the answers. The judge is from the same vendor as the generator; self-preference cancels between systems that share the generator, and the judge is validated on synthetic pairs before use. Grounding is measured by the accepted operator whose error profile is reported; conclusions about quality that depend on it are stated with that profile.

**Internal.** All systems share generator, retriever, corpus, prompts, and operator, so differences are attributable to control. Thresholds were fixed before each experiment. Three departures from pre-registration are disclosed: two candidate operators (`bart-large-mnli`, and the truncated-premise construction that was *not* adopted) were examined after partial results were known; the positive-control gate for operators was replaced after it failed every candidate, on grounds independent of any candidate's result; and the relation state was redefined once after its first definition failed a control. In each case both versions are reported and no further revision was permitted.

**External.** One generator, one corpus, [[N]] topics, one relation checkpoint. The uncited-answer rate (35% at round 0) is a behaviour of one generator under one prompt. The empty sufficient population under the standard operator is specific to a common default checkpoint; larger checkpoints of the same family behave differently, which is why the operator is validated rather than assumed.

**Conclusion.** Confidence intervals are bootstrap over queries; localization accuracy is per-topic with [[N]] topics per fault type; the human-labelled validation sample is [[60/100]] sentences. Waste ratios under the re-scored trajectory describe a trajectory generated by the standard verifier; the fresh run is the estimate we report.

---

## IX. CONCLUSION

[[결과 확정 후 작성. 골격: (1) 낭비율 측정은 검증된 충분성 연산자 위에서만 가능하며, 표준 연산자는 그것을 지탱하지 못했다 (2) Partial Rewind는 그래프 상태·최소 변경·종료 보장으로 정의되며 (3) 품질≈전역 재실행, 비용 −x% (4) 관계 신호의 지위(통과/철회) (5) 후속: 학습형 컨트롤러, 세션 메모리로서의 frozen graph.]]

---

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
그림 목록: FIGURE 1 낭비율(M1′; 없으면 M1-C) 막대 + CI · FIGURE 2 아키텍처(Query→P_i→E_i→R_ij→Localize→{FREEZE|VERIFY(새 subgraph)|EXPAND|PRUNE}→ΔG→partial re-eval→loop) · FIGURE 3 Pareto · FIGURE 4 trajectory 예시(|ΔG| per step, 상태 변화). 그래픽 초록 = FIGURE 2 축약.
부록(별도 파일): 감사 전문(archive_audit/manuscript_IEEEAccess_v0.9) 축약본, 프롬프트 전문, 토픽 예시 3개, 임계 민감도, leakage audit, 재현 절차(MiniCheck 설치 함정 포함).
-->
