"""Binary metrics of D5 and D6 (PLANO §4; Aula 09), from labels, predicted classes and ranking scores.

Attack is the positive class (y = 1). Threshold metrics (MCC, F1, balanced accuracy, sensitivity, specificity,
false positive rate) use the predicted class. Ranking metrics (AUC-ROC, added by P3, and AUPRC) use a score that
orders attack above benign: P(attack) when the model has predict_proba, otherwise decision_function (the SVM).
The four cells of the confusion matrix are kept as well. An optional `sample_weight` re-weights every metric, for
instance to the original class and attack-type distribution of the dataset (D16, P1); without it the result is
exactly the unweighted one, and the confusion-matrix cells stay integer counts.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)

POSITIVE = 1


def predictions_and_scores(model, X) -> tuple[np.ndarray, np.ndarray]:
    """Predicted class and ranking score of each instance, from a single pass over X.

    The class follows each estimator's own `predict` rule: the most probable class, or the sign of the decision
    function (tests/test_spotcheck.py checks that both give `model.predict`).
    """
    classes = np.asarray(model.classes_)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return classes[proba.argmax(axis=1)], proba[:, int(np.flatnonzero(classes == POSITIVE)[0])]
    score = model.decision_function(X)
    if classes[1] != POSITIVE:
        raise ValueError("decision_function must score the positive class")
    return classes[(score > 0).astype(int)], score


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray, sample_weight=None) -> dict:
    """MCC (main metric) and the secondary metrics of D6, plus AUC-ROC, specificity and the confusion matrix."""
    w = {"sample_weight": sample_weight}
    cells = confusion_matrix(y_true, y_pred, labels=[0, POSITIVE], **w).ravel()
    tn, fp, fn, tp = (int(v) if sample_weight is None else float(v) for v in cells)
    return {
        "mcc": matthews_corrcoef(y_true, y_pred, **w),
        "f1_attack": f1_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0.0, **w),
        "f1_benign": f1_score(y_true, y_pred, pos_label=0, zero_division=0.0, **w),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred, **w),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "fpr": fp / (tn + fp),
        "auprc": average_precision_score(y_true, score, pos_label=POSITIVE, **w),
        "roc_auc": roc_auc_score(y_true, score, **w),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }
