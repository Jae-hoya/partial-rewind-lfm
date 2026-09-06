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
