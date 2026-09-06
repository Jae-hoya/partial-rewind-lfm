## Appendix E. Parametric leakage audit

The generator has seen Wikipedia in pretraining, so it may state true facts from memory that lie in parts of the target article the retriever never returned. We bound that by scoring the same sentences against **all** chunks of the held-out target article.

| | Sentences |
|---|---|
| Scoring below 0.05 against cited chunks | 225 |
| Still below 0.05 against every retrieved chunk | 218 (96.9%) |
| Of those, reaching 0.5 against the full target article | **23 (10.6%)** |
| Of those, still below 0.05 | 185 (84.9%) |

Leakage exists and is small: about one sentence in ten that no retrieved chunk supports is supported somewhere in the target article. It does not account for the bulk of the unsupported mass.
