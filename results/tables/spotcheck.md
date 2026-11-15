**Spot-checking on sample E: default hyperparameters, 5 folds x 2 repetitions**

| algorithm | inductive_bias | mcc_mean | mcc_std | mcc_train_mean | mcc_train_minus_validation | sensitivity_mean | specificity_mean |
|---|---|---|---|---|---|---|---|
| lightgbm | ensemble of trees (boosting) | 0.8067 | 0.0058 | 0.8411 | 0.0344 | 0.8594 | 0.9444 |
| random_forest | ensemble of trees (bagging) | 0.8035 | 0.0058 | 0.9997 | 0.1962 | 0.8634 | 0.9378 |
| mlp | neural network | 0.7756 | 0.0076 | 0.789 | 0.0133 | 0.8466 | 0.9265 |
| svm | kernel (maximum margin) | 0.7659 | 0.0042 | 0.7729 | 0.007 | 0.8066 | 0.9512 |
| knn | instance-based | 0.7522 | 0.0046 | 0.8198 | 0.0676 | 0.8169 | 0.9304 |
| logreg | linear | 0.7256 | 0.0077 | 0.7266 | 0.001 | 0.7855 | 0.9322 |
| decision_tree | single tree | 0.723 | 0.0042 | 0.9997 | 0.2767 | 0.8624 | 0.8606 |
| gaussian_nb | probabilistic | 0.6935 | 0.0041 | 0.6939 | 0.0004 | 0.6878 | 0.9762 |
| dummy | baseline | 0.0 | 0.0 | 0.0 | 0.0 | 0.4 | 0.6 |
