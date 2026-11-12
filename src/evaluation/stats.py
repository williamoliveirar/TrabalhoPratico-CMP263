"""Statistics of Etapa 8 (D10, P5, P1): hypothesis tests, confidence intervals and projected precision.

- corrected_resampled_ttest: paired t-test over the outer folds whose variance accounts for the overlap of the training
  sets (Nadeau & Bengio, 2003); the plain paired t-test has an inflated type I error here (Dietterich, 1998).
- holm: step-down adjustment of a family of p-values.
- wilson: score interval for a proportion (per-type sensitivity, false positive rate).
- bootstrap_counts / percentile_interval: percentile bootstrap over the out-of-fold predictions, optionally stratified,
  paired across models (the same resampled instances for every model). It measures only the variability of the test
  instances, not that of training on another training set, and treats instances as independent.
- projected_ppv: precision at another attack prevalence, from sensitivity and specificity.
"""
import numpy as np
from scipy import stats


def corrected_resampled_ttest(scores_a, scores_b, n_train: int, n_test: int) -> tuple[float, float]:
    """t statistic and two-sided p-value of mean(scores_a - scores_b) over k folds; (nan, nan) if the
    differences have no variance."""
    d = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    k = d.size
    var = np.var(d, ddof=1)
    if var == 0:
        return float("nan"), float("nan")
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    adjusted, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, (p.size - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def wilson(successes, n, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval of successes / n at the given confidence level (vectorised)."""
    successes, n = np.asarray(successes, dtype=float), np.asarray(n, dtype=float)
    z = stats.norm.ppf(0.5 + level / 2)
    phat = successes / n
    denom = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    # exact ends: rounding would otherwise put the bound a hair inside 0 or 1 when phat is 0 or 1
    return np.where(successes == 0, 0.0, centre - half), np.where(successes == n, 1.0, centre + half)


def mcc_from_counts(tn, fp, fn, tp) -> np.ndarray:
    """MCC from confusion-matrix cells (vectorised); 0 when a margin is empty, as in scikit-learn."""
    tn, fp, fn, tp = (np.asarray(v, dtype=float) for v in (tn, fp, fn, tp))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        mcc = (tp * tn - fp * fn) / denom
    return np.where(denom == 0, 0.0, mcc)


def bootstrap_counts(y: np.ndarray, preds: np.ndarray, strata: np.ndarray | None, n_resamples: int,
                     seed: int) -> np.ndarray:
    """Confusion-matrix cells of every model in every bootstrap resample, shape (n_resamples, n_models, 4) with the
    cells in the order tn, fp, fn, tp.

    preds has shape (n_models, n_instances), aligned with y. Each resample draws, with replacement, as many instances
    as there are in each stratum (or in the whole set when strata is None), and the same draw is applied to every
    model, so differences between models are paired.
    """
    y = np.asarray(y).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    if preds.ndim != 2 or preds.shape[1] != y.size:
        raise ValueError("preds must have shape (n_models, n_instances) aligned with y")
    groups = [np.arange(y.size)] if strata is None else \
        [np.flatnonzero(strata == s) for s in np.unique(np.asarray(strata))]
    rng = np.random.default_rng(seed)
    out = np.zeros((n_resamples, preds.shape[0], 4), dtype=np.int64)
    for b in range(n_resamples):
        for g in groups:
            draw = g[rng.integers(0, g.size, g.size)]
            yd, pd = y[draw], preds[:, draw]
            out[b, :, 0] += ((1 - pd) * (1 - yd)).sum(axis=1)
            out[b, :, 1] += (pd * (1 - yd)).sum(axis=1)
            out[b, :, 2] += ((1 - pd) * yd).sum(axis=1)
            out[b, :, 3] += (pd * yd).sum(axis=1)
    return out


def percentile_interval(values: np.ndarray, level: float) -> tuple[float, float]:
    """Percentile interval of a bootstrap distribution (e.g. 2.5 and 97.5 percentiles for level 0.95)."""
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(np.asarray(values, dtype=float), [tail, 100 - tail])
    return float(low), float(high)


def projected_ppv(sensitivity, specificity, prevalence):
    """Precision when the attack prevalence is `prevalence`: sens·π / (sens·π + (1 − spec)·(1 − π))."""
    sensitivity, specificity, prevalence = (np.asarray(v, dtype=float) for v in (sensitivity, specificity, prevalence))
    detected = sensitivity * prevalence
    return detected / (detected + (1 - specificity) * (1 - prevalence))
