"""Single source of randomness (docs/PLANO.md §5.2): every random_state and generator derives from configs/base.yaml.

Global seeding is only a safety net. Code passes the seed explicitly to every `random_state` and draws from
`make_rng`, never from the global generators. Polars `group_by` returns groups in no fixed order, so every
output built from it is sorted before being written.
"""
import random

import numpy as np


def seed_everything(cfg: dict) -> int:
    """Seed Python's and NumPy's global generators with cfg["seed"] and return the seed for explicit use."""
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    return seed


def make_rng(cfg: dict) -> np.random.Generator:
    """A fresh generator; a script that needs one consumes it in a fixed order (e.g. Etapa 2, step 3)."""
    return np.random.default_rng(int(cfg["seed"]))
