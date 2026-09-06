## Appendix D. Threshold sensitivity

Grounded fraction as $\tau_g$ is swept over the full range, for the four estimator configurations of Section VI. Counted from the raw per-sentence scores, not from rounded summaries.

| Configuration | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | Range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sentence x chunk (deployed) | 23.5 | 20.4 | 17.7 | 16.3 | 16.0 | 15.6 | 15.3 | 14.6 | 13.6 | 11.9 | 10.5 | 12.9 pp |
| Claim x chunk | 38.1 | 36.1 | 34.4 | 33.0 | 32.1 | 31.7 | 31.1 | 29.7 | 28.2 | 25.8 | 24.5 | 13.6 pp |
| Sentence x chunk-sentence | 46.6 | 44.6 | 41.8 | 40.1 | 38.8 | 38.4 | 36.4 | 35.4 | 34.0 | 30.3 | 27.6 | 19.0 pp |
| Claim x chunk-sentence | 62.6 | 60.4 | 59.0 | 58.8 | 57.9 | 57.1 | 56.8 | 56.6 | 54.9 | 53.8 | 52.4 | 10.3 pp |

Moving the threshold across its entire range changes the grounded fraction by 10 to 19 points. Changing the estimator configuration changes it by 41.5. Threshold calibration is not the lever.
