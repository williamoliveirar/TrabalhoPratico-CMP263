"""Algorithms of D7 and the estimator each one starts from (PLANO §7, Etapas 6 and 7).

Spot-checking (Etapa 6) uses each algorithm's default hyperparameters, plus the `fixed` entries of
configs/search_spaces.yaml (what an algorithm needs to converge or to be deterministic) and the seed.
Hyperparameter tuning (Etapa 7) only changes the `params` entries of that file.

Pre-processing group of each algorithm (D13; Aula 13-14, "Transformação de dados"):
- TRANSFORMED: sensitive to the scale or the distribution of the attributes (distances, gradients, Gaussian
  likelihoods), so their pipelines get log1p, standardisation and one-hot encoding (src/models/pipelines.py);
- every other algorithm (trees, ensembles of trees and the dummy baseline) gets the attributes as they are.
"""
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

BASELINE = "dummy"

FACTORIES = {
    "logreg": lambda seed, n_jobs: LogisticRegression(random_state=seed),
    "gaussian_nb": lambda seed, n_jobs: GaussianNB(),
    "knn": lambda seed, n_jobs: KNeighborsClassifier(n_jobs=n_jobs),
    "decision_tree": lambda seed, n_jobs: DecisionTreeClassifier(random_state=seed),
    "mlp": lambda seed, n_jobs: MLPClassifier(random_state=seed),
    "random_forest": lambda seed, n_jobs: RandomForestClassifier(random_state=seed, n_jobs=n_jobs),
    "lightgbm": lambda seed, n_jobs: LGBMClassifier(random_state=seed, n_jobs=n_jobs),
    BASELINE: lambda seed, n_jobs: DummyClassifier(),
}

TRANSFORMED = frozenset({"logreg", "gaussian_nb", "knn", "mlp"})


def make_estimator(name: str, seed: int, n_jobs: int, search_cfg: dict) -> ClassifierMixin:
    """Default estimator of `name` with the fixed parameters of configs/search_spaces.yaml."""
    if name not in FACTORIES:
        raise KeyError(f"unknown algorithm {name!r}; known: {sorted(FACTORIES)}")
    estimator = FACTORIES[name](seed, n_jobs)
    return estimator.set_params(**search_cfg.get(name, {}).get("fixed", {}))
