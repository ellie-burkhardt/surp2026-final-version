import numpy as np
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)

@dataclass
class BinaryMetrics:
    target: str
    threshold: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    specificity: float
    mcc: float
    tn: int
    fp: int
    fn: int
    tp: int

def evaluate_binary(target, y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities have different lengths.")
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return BinaryMetrics(
        target, float(threshold),
        float(accuracy_score(y_true, predictions)),
        float(balanced_accuracy_score(y_true, predictions)),
        float(precision_score(y_true, predictions, zero_division=0)),
        float(recall_score(y_true, predictions, zero_division=0)),
        float(f1_score(y_true, predictions, zero_division=0)),
        float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2 else float("nan"),
        float(average_precision_score(y_true, probabilities)),
        float(specificity),
        float(matthews_corrcoef(y_true, predictions)),
        int(tn), int(fp), int(fn), int(tp),
    )

def metrics_to_dict(m):
    return {
        "target": m.target, "threshold": m.threshold,
        "accuracy": m.accuracy, "balanced_accuracy": m.balanced_accuracy,
        "precision": m.precision, "recall": m.recall, "f1": m.f1,
        "roc_auc": m.roc_auc, "pr_auc": m.pr_auc,
        "specificity": m.specificity, "mcc": m.mcc,
        "tn": m.tn, "fp": m.fp, "fn": m.fn, "tp": m.tp,
    }