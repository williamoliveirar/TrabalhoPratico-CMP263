"""Binary metrics of D5 and D6 (PLANO §4; Aula 09), from labels, predicted classes and ranking scores.

Attack is the positive class (y = 1). Threshold metrics (MCC, F1, balanced accuracy, sensitivity, specificity,
false positive rate) use the predicted class. Ranking metrics (AUC-ROC, added by P3, and AUPRC) use a score that
orders attack above benign: P(attack) when the model has predict_proba, otherwise decision_function (the SVM).
The four cells of the confusion matrix are kept as well, so metrics can be re-weighted later (proposal D16).
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


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict:
    """MCC (main metric) and the secondary metrics of D6, plus AUC-ROC, specificity and the confusion matrix."""
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, POSITIVE]).ravel())
    return {
        "mcc": matthews_corrcoef(y_true, y_pred),
        "f1_attack": f1_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0.0),
        "f1_benign": f1_score(y_true, y_pred, pos_label=0, zero_division=0.0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "fpr": fp / (tn + fp),
        "auprc": average_precision_score(y_true, score, pos_label=POSITIVE),
        "roc_auc": roc_auc_score(y_true, score),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }
