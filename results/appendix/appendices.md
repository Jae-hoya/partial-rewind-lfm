# Appendices

> Appendix C (condensed estimator audit) is written by the co-author and is not included here.

## Appendix A. Prompts

All prompts are frozen for the study and reproduced here verbatim from the source; they are not transcribed. Placeholders in braces are filled at call time.

### SYSTEM (`lfm/perspectives.py`)

```
You decompose a research question into fixed analytical perspectives for a multi-perspective RAG system. Reply with JSON only.
```

### PROMPT (`lfm/perspectives.py`)

```
Research question:
{query}

Decompose it into exactly {n} perspectives, one for each role below, in this order:
{roles}

For each perspective produce:
- "role": the role name, copied verbatim
- "question": one question (10-25 words) framing that role's angle on the research question
- "sub_questions": {lo} to {hi} concrete, factual sub-questions answerable from an
  encyclopedia. Each must be self-contained and searchable on its own.

Return a JSON array of {n} objects with exactly those keys. No prose, no code fences.{extra}
```

### ANSWER_SYSTEM (`lfm/branch.py`)

```
You answer from one assigned perspective using only the supplied context chunks. Every factual sentence must carry a [chunk_id] citation. Reply with JSON only.
```

### ANSWER_PROMPT (`lfm/branch.py`)

```
Perspective role: {role}
Perspective question: {question}

Context chunks:
{context}

Answer each sub-question below using ONLY the context chunks.

Rules:
- Cite with [chunk_id] inline, using the exact ids shown above. Every factual sentence
  needs at least one citation.
- Write DIRECT factual statements about the subject itself. Never write about the sources:
  no "the context shows", "the provided chunks indicate", "the context does not specify",
  no hedging about what is or is not documented. Such sentences describe the corpus rather
  than the world and cannot be verified against it.
- If the context does not support an answer, simply write fewer sentences — or leave the
  text empty. Do not describe the absence. Do not invent facts. Do not cite ids that are
  not listed.
- Keep each sentence short and self-contained: one fact per sentence, stated plainly.
- 2 to 5 sentences per sub-question.
- Return a JSON array. Each element: {{"index": <int>, "text": "<answer with [citations]>"}}
- Return the array only. No prose, no code fences.
{directive}
Sub-questions:
{subs}
```

### CLAIM_SYSTEM (`lfm/branch.py`)

```
You decompose text into atomic factual claims. Reply with JSON only.
```

### CLAIM_PROMPT (`lfm/branch.py`)

```
Decompose the text below into atomic factual claims.

Rules:
- Each claim is one self-contained sentence asserting a single fact.
- Strip citation markers like [c12] from the claim text.
- Do not add facts that are not in the text.
- At most {max_claims} claims.
- Return a JSON array of strings. No prose, no code fences.

--- TEXT ---
{answer}
--- END ---
```

### PARAPHRASE_SYSTEM (`lfm/branch.py`)

```
You rewrite search sub-questions. Reply with JSON only.
```

### PARAPHRASE_PROMPT (`lfm/branch.py`)

```
The previous attempt at these sub-questions retrieved insufficient
evidence. Rewrite each one to approach the same information need from a different angle:
different wording, different entities to look up, more specific facts.

Keep the same number of items and the same order. Do not change the topic.
Return a JSON array of strings only. No prose, no code fences.

{subs}
```

### REPAIR_SYSTEM (`lfm/jsonio.py`)

```
You fix malformed JSON. Reply with JSON only, no prose, no code fences.
```

### REPAIR_PROMPT (`lfm/jsonio.py`)

```
The text below was supposed to be a JSON {kind} but is malformed or wrapped
in prose. Return the corrected JSON {kind} only.

--- BEGIN ---
{body}
--- END ---
```

### d8b_fidelity (`prompts/d8b_fidelity.txt`)

```
A sentence was decomposed into atomic claims. Judge whether the claim set preserves
the factual content of the original sentence.

- "complete" : the claims together cover every factual element of the sentence
- "omission" : the claims drop at least one factual element
- "addition" : the claims assert something the sentence does not say
  (if both omission and addition apply, answer "addition")

Reply with exactly two lines and nothing else:
LABEL: complete | omission | addition
REASON: <one short sentence>

--- SENTENCE ---
{sentence}
--- CLAIMS ---
{claims}
```

### d8b_judge (`prompts/d8b_judge.txt`)

```
You judge whether a CLAIM is supported by the EVIDENCE below.

Rules:
- "supported" means every factual element of the claim can be verified from the
  evidence text alone. Paraphrase is fine. Common-sense rewording is fine.
- "unsupported" means at least one factual element is absent from, or contradicted
  by, the evidence.
- "unclear" only when the claim is too vague to judge. Use it sparingly.
- Do not use outside knowledge. Judge only against the evidence given.

Reply with exactly two lines and nothing else:
LABEL: supported | unsupported | unclear
REASON: <one short sentence>

--- EVIDENCE ---
{evidence}
--- CLAIM ---
{claim}
```

### m2_negate (`prompts/m2_negate.txt`)

```
Rewrite the sentence below so that it asserts the OPPOSITE of the original fact.

Rules:
- Change only what is needed to reverse the fact. Keep the same subject, entities,
  numbers and wording everywhere else.
- Do not add hedging ("may", "possibly"). State the opposite as a plain fact.
- Do not add or remove information beyond the reversal.
- Output the rewritten sentence only. No quotes, no preface, no explanation.

--- SENTENCE ---
{claim}
--- END ---
```

### m2_paraphrase (`prompts/m2_paraphrase.txt`)

```
Rewrite the sentence below so that it states exactly the same fact in different words.

Rules:
- Preserve every fact, entity, number and date exactly. Add nothing, drop nothing.
- Change the wording and sentence structure substantially.
- Do not hedge and do not comment on the sentence.
- Output the rewritten sentence only. No quotes, no preface, no explanation.

--- SENTENCE ---
{claim}
--- END ---
```

Total: 14 prompts.

## Appendix B. Example topics

Three of the 50 approved topics, drawn with a fixed seed (42); no topic was chosen for its outcome. Each shows the research question, the target article held out of the index, and the search anchor terms.

### t004

**Question.** How did the Montreal‑based forest products firm grow through acquisitions and mergers, influence newsprint markets, and confront industry decline controversies?

| | |
|---|---|
| Target article | Abitibi Power and Paper Company |
| Identifier | 21424895 |
| Anchor terms | mainstay, excellence, price, montreal, quebec |
| Related documents in corpus | 196 |
| Primary infobox type | Infobox company |

### t034

**Question.** How did the Wake Forest star’s collegiate dominance translate into a decade-long professional basketball career, influence, and critiques of his legacy?

| | |
|---|---|
| Target article | Len Chappell |
| Identifier | 4701939 |
| Anchor terms | chappell, first-team, consensus, leonard, all-american |
| Related documents in corpus | 91 |
| Primary infobox type | Infobox basketball biography |

### t039

**Question.** How did the British track and road cyclist achieve sustained team pursuit success, refine training techniques, influence national programs, and confront doping debates?

| | |
|---|---|
| Target article | Ed Clancy |
| Identifier | 9308734 |
| Anchor terms | clancy, pursuit, bicycle, racer, cycling |
| Related documents in corpus | 87 |
| Primary infobox type | Infobox cyclist |

## Appendix C. Condensed estimator audit

[[공저자 작성 — v0.9 에서 본문에 안 들어간 것만]]

## Appendix D. Threshold sensitivity

Grounded fraction as $\tau_g$ is swept over the full range, for the four estimator configurations of Section VI. Counted from the raw per-sentence scores, not from rounded summaries.

| Configuration | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | Range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sentence x chunk (deployed) | 23.5 | 20.4 | 17.7 | 16.3 | 16.0 | 15.6 | 15.3 | 14.6 | 13.6 | 11.9 | 10.5 | 12.9 pp |
| Claim x chunk | 38.1 | 36.1 | 34.4 | 33.0 | 32.1 | 31.7 | 31.1 | 29.7 | 28.2 | 25.8 | 24.5 | 13.6 pp |
| Sentence x chunk-sentence | 46.6 | 44.6 | 41.8 | 40.1 | 38.8 | 38.4 | 36.4 | 35.4 | 34.0 | 30.3 | 27.6 | 19.0 pp |
| Claim x chunk-sentence | 62.6 | 60.4 | 59.0 | 58.8 | 57.9 | 57.1 | 56.8 | 56.6 | 54.9 | 53.8 | 52.4 | 10.3 pp |

Moving the threshold across its entire range changes the grounded fraction by 10 to 19 points. Changing the estimator configuration changes it by 41.5. Threshold calibration is not the lever.

## Appendix E. Parametric leakage audit

The generator has seen Wikipedia in pretraining, so it may state true facts from memory that lie in parts of the target article the retriever never returned. We bound that by scoring the same sentences against **all** chunks of the held-out target article.

| | Sentences |
|---|---|
| Scoring below 0.05 against cited chunks | 225 |
| Still below 0.05 against every retrieved chunk | 218 (96.9%) |
| Of those, reaching 0.5 against the full target article | **23 (10.6%)** |
| Of those, still below 0.05 | 185 (84.9%) |

Leakage exists and is small: about one sentence in ten that no retrieved chunk supports is supported somewhere in the target article. It does not account for the bulk of the unsupported mass.

## Appendix F. Reproduction

### Environment

```
python 3.11 · transformers 4.42.3 · torch 2.3.1+cu118 · numpy 1.26.4
bm25s · tiktoken (o200k_base) · pyarrow
```

### Models

| Role | Model |
|---|---|
| Generation | `gpt-5-mini`, reasoning_effort=low, max_tokens=2000 |
| Judge (gates only) | `gpt-5.4-nano` |
| Standard operator $g_1$ | `cross-encoder/nli-deberta-v3-base`, $\tau_{\mathrm{ground}}=0.5$ |
| Accepted operator | `lytang/MiniCheck-Flan-T5-Large`, single-document premise, threshold $>0.5$ (strict) |
| Relation (3-way) | `facebook/bart-large-mnli` |

### Installing MiniCheck — a name collision to avoid

```bash
pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git"
```

**Do not run `pip install minicheck`.** The package of that name on PyPI is a different project — an explicit-state model checker. It installs cleanly and `import minicheck` succeeds, so the mistake surfaces only later, when `minicheck.minicheck` is not found. The repository install adds `minicheck-0.1.0` alone and does not alter transformers, torch, or numpy; it downloads the nltk `punkt` tokenizer on first import.

### Corpus

| | |
|---|---|
| Source | HuggingFace `wikimedia/structured-wikipedia` |
| Revision | `417c267bb457fa645c22eb3b5c77764963194c70` |
| Rows loaded | 150,000 → 142,972 unique (4.69% duplicates) |
| Full index | 142,922 documents / 685,683 chunks |
| Partial held-out index | 142,972 / 686,221 |
| Chunking | one chunk per section; sections over 1,200 tokens split at paragraph boundaries |

### Numerical conventions

- Every percentage-point difference is computed from raw per-row rates, never by subtracting two rounded values.
- Summary CSVs store four decimals; scripts read the per-row files instead, because re-rounding a rounded value shifts figures (15.6463% became 15.7% this way once).
- Proportions carry Wilson 95% intervals; means over queries carry query-level bootstrap intervals (B=10,000, seed 42).
- Every grounded fraction is reported with its premise pool scope stated, because max-aggregation rises with pool size.

### Verification

```bash
python experiments/verify_numbers.py           # 112 reported figures vs raw data
python experiments/verify_manuscript.py        # manuscript VII/VIII vs raw data
python experiments/verify_manuscript_part1.py  # manuscript VI vs raw data
python -m pytest -q                            # unit tests
```

Each exits non-zero on any mismatch and reports, rather than silently skips, any figure it cannot locate in the manuscript.
