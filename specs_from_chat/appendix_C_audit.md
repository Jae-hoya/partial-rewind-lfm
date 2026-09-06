## Appendix C. Condensed Estimator Audit

The body reports the findings that bear on the method: that the deployed operator cannot support a freeze decision (Section VI-A), why it fails (Sections VI-B, VI-E), which replacements pass a shuffled-evidence control (Sections VI-G, VI-J), and what the accepted operator changes (Section VIII). This appendix records the parts of the audit that the body omits: the sequence in which the diagnoses were made, the criteria fixed before each stage, the conditions that failed, and two results kept out of the body for space.

### C.1 Sequence and pre-registration

Each stage below fixed its criterion before execution. Where a stage failed, the branch taken was the one written down in advance; where the mechanism turned out to differ from the one predicted, both are stated. No stage was re-run with adjusted criteria.

| Stage | Question | Criterion fixed in advance | Outcome |
|---|---|---|---|
| D0 | Does the grounding ratio reward brevity? | none (descriptive) | Yes: of 30 branches scoring above 0.8, fourteen are one sentence. Conjunction with full coverage adopted as the sufficiency operator |
| D1–D3 | Is retrieval the bottleneck? | intersection rate ≥ 10%, median grounding of fully covered branches ≥ 0.50, fully covered rate ≥ 30%, empty rate ≤ 15% | All four conditions fail every criterion; the gold-article condition raises coverage and leaves grounding flat |
| D7-0 | Is the collapse truncation or dilution? | none (positive control) | Dilution: at 2× premise length only 6 of 35 pairs are truncated, and the median falls 0.992 → 0.152 |
| D8-A | Does the standard repair inflate? | shuffled-premise rate ≤ 5% | Fails at 50.5%; a lexical-prefilter fallback fails at 15.2%; only the deployed configuration passes at 0.00% |
| D8-B | Is the near-zero mass unsupported or unseen? | precision ≥ 0.85 on the supported stratum, false-negative rate ≤ 0.25 on the near-zero stratum | Precision 0.913; false-negative rate 0.725 — the mass is unseen, not absent |
| D9-P | Does a claim-versus-document verifier keep support under padding? | agreement ≥ 32/35 at 1×, retention ≥ 80% at 4× | Every candidate fails, the deployed operator included (see C.2) |
| D10 | Are the mechanisms properties of the family or the checkpoint? | shuffled rate ≤ 5% | Checkpoint: retention at 4× is 11% for the deployed model and 71% for `bart-large-mnli`; one larger checkpoint doubles the raw score and fails the control at 21.1% |
| D13–D14 | Is dilution driven by length or by irrelevant content? | pre-registered $L=64$; premise-length parity within 10% between constructions | Irrelevant content. The parity requirement is unsatisfiable by construction (see C.3); $L=64$ is not the best setting, and only single-chunk truncation passes the control |

### C.2 Why the pre-registered positive control failed every candidate

The positive control asked a candidate to agree with the deployed operator on the 35 pairs it scored above 0.90, and to retain those pairs when the premise was padded four-fold. No estimator passed, including the one under audit: it agrees with itself on 35 of 35 and retains 4.

The two clauses exclude each other. The positive set is defined by the audited operator, whose precision on such sentences is about 0.91, so a perfect estimator would be expected to agree on roughly 32 of 35 — exactly where the bar sat, leaving no allowance for reference noise. The retention clause then penalizes precisely the behaviour that agreement rewards. The control was also underpowered: at $n=35$, the Wilson interval around 71.4% is [55, 84] and contains the 80% bar, so the clause could not have separated any candidate from the bar at reasonable confidence.

| Candidate | Agreement at 1× (bar 32/35) | Retention at 4× (bar 80%) | Shuffled rate | Net gain |
|---|---|---|---|---|
| `nli-deberta-v3-base` (deployed) | 35/35 | 4/35 (11.4%) | 0.00% | 15.65 pp |
| `MiniCheck-Flan-T5-Large` | 27/35 | 25/35 (71.4%) | 0.00% | 64.29 pp |
| `bart-large-mnli` | 29/35 | 25/35 (71.4%) | 1.02% | 59.86 pp |
| `deberta-v3-large-mnli` | 30/35 | 15/35 (42.9%) | 21.09% | 16.67 pp |
| `roberta-large-mnli` | 17/35 | 1/35 (2.9%) | 2.72% | 17.01 pp |

The gate was not adjusted after this result. It is reported as it was applied, and the replacement uses reference labels obtained independently of any estimator, with the validity thresholds that were already registered for the judge sample. A reader who rejects that substitution should read this table as the terminal result of the search.

### C.3 Premise construction: the unsatisfiable parity requirement and the sweep

The protocol required the *single* and *union* constructions to be compared at premise lengths within 10% of each other. They cannot be: *union* concatenates the branch's cited chunks, a median of three, so at any truncation budget $L$ its premise is roughly three times longer (47 versus 130 tokens at $L=64$). The requirement was not met and the budget was not tuned to meet it; the two are compared at equal $L$ and the discrepancy is reported.

The sweep over $L$ was declared secondary before it was run, and the judgment below uses the shuffled-premise control rather than the raw fraction.

| Construction | Median premise tokens | Grounded | Shuffled | Net gain | Control |
|---|---|---|---|---|---|
| whole chunk (deployed) | 368 | 15.65% | 0.00% | 15.65 pp | pass |
| sentence-split, max | 28 | 38.44% | — | — | (see D8-A) |
| single, $L=64$ (pre-registered) | 47 | 31.63% | 8.50% | 23.13 pp | fail |
| single, $L=128$ | 105 | 42.18% | 1.36% | 40.82 pp | pass |
| single, $L=256$ | 219 | 30.27% | 0.68% | 29.59 pp | pass |
| union, $L=64$ (pre-registered) | 130 | 34.01% | 6.46% | 27.55 pp | fail |
| union, $L=128$ | 314 | 57.82% | 9.86% | 47.96 pp | fail |
| union, $L=256$ | 613 | 67.01% | 11.22% | 55.78 pp | fail |

Every *union* construction fails: concatenation gives the lexical selector a larger pool, and it finds overlapping sentences in the wrong topic too. Single-chunk truncation at $L=128$ passes with a shuffled rate of 1.36% and a net gain 2.6 times the deployed configuration's, without changing the model — the cheapest repair found in this study, and one that was not adopted here only because it entered after the candidate set had been closed. The response is not monotone in $L$: 64 cuts into relevant sentences, 256 lets irrelevant content back in. We state that as an observation, not a mechanism.

### C.4 Retrieval conditions (D1–D3), in full

Fifteen topics, round 0 only, one run per condition, criteria fixed in advance; none was met.

| Condition | Sufficient rate | Median grounding, full coverage | Fully covered | Empty answers | Mean coverage | Mean grounding | Mean grounding, non-empty |
|---|---|---|---|---|---|---|---|
| C1 baseline | 0.0% | 0.000 | 28.3% | 35.0% | 0.474 | 0.188 | 0.289 (39) |
| C2 depth $k=15$ | 0.0% | 0.136 | 23.3% | 40.0% | 0.408 | 0.148 | 0.246 (36) |
| C3 gold article body in index | 0.0% | 0.000 | 33.3% | 13.3% | 0.620 | 0.181 | 0.209 (52) |
| C4 both | 1.7% | 0.136 | 33.3% | 20.0% | 0.578 | 0.180 | 0.225 (48) |
| Criterion | ≥ 10% | ≥ 0.50 | ≥ 30% | ≤ 15% | — | — | — |

C3 is the informative row: with the gold article present the pipeline answers more and almost never returns empty, and grounding does not move. C2 lowers every axis at once. Two protocol notes: the 15-topic subset is not distributionally identical to the 50-topic set (median grounding among fully covered branches 0.000 over 17 branches versus 0.231 over the 51 round-0 fully covered branches of the full run), so conditions are compared with C1 rather than with the main measurement; and Wilson intervals on the fully-covered column overlap across conditions, so no conclusion is drawn from it.

### C.5 Two results kept out of the body

**Attribution is not the problem.** If answers cited the wrong chunk, widening the premise pool from the cited chunks (median 3) to all retrieved chunks (median 14) would recover the missing support. It recovers 1.7 points: of 225 sentences below 0.05 against cited chunks, 218 remain below 0.05 against every retrieved chunk. The citations point where the support would be if the operator could see it.

**Claim decomposition does not manufacture support.** Of 30 sampled branch answers, the claim set was judged complete for 26 and omitting content for 2, with zero additions. Because the pipeline decomposes claims from the whole branch answer rather than per sentence, the check is at answer level rather than the sentence level the protocol specified; we report the unit as it exists rather than construct a mapping the data lacks. Zero additions is the direction that matters for the claim-level rows of the body.

### C.6 What this appendix does not contain

The per-stage findings documents, raw score files, and the scripts that regenerate every figure in this paper from them are in the released repository, together with three verification scripts that check the manuscript's numbers against the raw rows and exit non-zero on any mismatch or any figure they cannot locate.
