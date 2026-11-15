**Hyperparameters chosen in each outer fold, and those of the final model (outer_fold empty)**

| model | outer_fold | best_params |
|---|---|---|
| lightgbm | 0 | {"colsample_bytree": 0.6783931449676581, "learning_rate": 0.011662890273931383, "min_child_samples": 49, "n_estimators": 824, "num_leaves": 222, "reg_lambda": 0.9756194291174574, "subsample": 0.815876852955632} |
| lightgbm | 1 | {"colsample_bytree": 0.6783931449676581, "learning_rate": 0.011662890273931383, "min_child_samples": 49, "n_estimators": 824, "num_leaves": 222, "reg_lambda": 0.9756194291174574, "subsample": 0.815876852955632} |
| lightgbm | 2 | {"colsample_bytree": 0.9341209982356952, "learning_rate": 0.029773579612263783, "min_child_samples": 79, "n_estimators": 563, "num_leaves": 107, "reg_lambda": 2.7129089558470034, "subsample": 0.730383562080754} |
| lightgbm | 3 | {"colsample_bytree": 0.6783931449676581, "learning_rate": 0.011662890273931383, "min_child_samples": 49, "n_estimators": 824, "num_leaves": 222, "reg_lambda": 0.9756194291174574, "subsample": 0.815876852955632} |
| lightgbm | 4 | {"colsample_bytree": 0.6783931449676581, "learning_rate": 0.011662890273931383, "min_child_samples": 49, "n_estimators": 824, "num_leaves": 222, "reg_lambda": 0.9756194291174574, "subsample": 0.815876852955632} |
| random_forest | 0 | {"max_depth": null, "max_features": 0.5, "min_samples_leaf": 3, "n_estimators": 406} |
| random_forest | 1 | {"max_depth": null, "max_features": 0.5, "min_samples_leaf": 3, "n_estimators": 406} |
| random_forest | 2 | {"max_depth": null, "max_features": 0.5, "min_samples_leaf": 3, "n_estimators": 406} |
| random_forest | 3 | {"max_depth": null, "max_features": 0.5, "min_samples_leaf": 3, "n_estimators": 406} |
| random_forest | 4 | {"max_depth": null, "max_features": 0.5, "min_samples_leaf": 3, "n_estimators": 406} |
| mlp | 0 | {"alpha": 2.1618942406574445e-05, "hidden_layer_sizes": [128, 64], "learning_rate_init": 0.0026070247583707684} |
| mlp | 1 | {"alpha": 0.0048696409415209, "hidden_layer_sizes": [128, 64], "learning_rate_init": 0.0013832094546570486} |
| mlp | 2 | {"alpha": 2.1618942406574445e-05, "hidden_layer_sizes": [128, 64], "learning_rate_init": 0.0026070247583707684} |
| mlp | 3 | {"alpha": 5.415244119402541e-06, "hidden_layer_sizes": [128, 64], "learning_rate_init": 0.0016722697006183683} |
| mlp | 4 | {"alpha": 5.415244119402541e-06, "hidden_layer_sizes": [128, 64], "learning_rate_init": 0.0016722697006183683} |
| decision_tree | 0 | {"criterion": "gini", "max_depth": 20, "min_samples_leaf": 41} |
| decision_tree | 1 | {"criterion": "gini", "max_depth": 18, "min_samples_leaf": 55} |
| decision_tree | 2 | {"criterion": "entropy", "max_depth": 14, "min_samples_leaf": 4} |
| decision_tree | 3 | {"criterion": "entropy", "max_depth": 14, "min_samples_leaf": 4} |
| decision_tree | 4 | {"criterion": "gini", "max_depth": 20, "min_samples_leaf": 41} |
| lightgbm |  | {"colsample_bytree": 0.6783931449676581, "learning_rate": 0.011662890273931383, "min_child_samples": 49, "n_estimators": 824, "num_leaves": 222, "reg_lambda": 0.9756194291174574, "subsample": 0.815876852955632} |
