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
