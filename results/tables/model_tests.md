**Corrected resampled t-test with Holm's correction, and bootstrap 95% interval of the MCC difference**

| model_a | model_b | diff_mean | folds_a_better | t | p_holm | ci_low | ci_high |
|---|---|---|---|---|---|---|---|
| lightgbm | random_forest | 0.0081 | 5 | 17.5924 | 6.1e-05 | 0.0067 | 0.0094 |
| lightgbm | mlp | 0.0314 | 5 | 44.303 | 6.2e-06 | 0.0295 | 0.0335 |
| lightgbm | decision_tree | 0.0394 | 5 | 41.8481 | 6.2e-06 | 0.0375 | 0.0414 |
| random_forest | decision_tree | 0.0313 | 5 | 32.825 | 1e-05 | 0.0295 | 0.0333 |
