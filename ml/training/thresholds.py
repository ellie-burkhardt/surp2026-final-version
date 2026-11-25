import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.metrics import precision_recall_curve

logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    threshold: float
    f1: float
    precision: float
    recall: float


class ThresholdOptimizer:

    def __init__(
        self,
        lower: float = 0.01,
        upper: float = 0.90,
        steps: int = 90,
        min_recall: Optional[float] = None,
        min_precision: Optional[float] = None,
        default_fallback: float = 0.25,
    ):
        self.lower = lower
        self.upper = upper
        self.steps = steps
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.default_fallback = default_fallback

    def optimize(self, y_true: np.ndarray, probabilities: np.ndarray) -> ThresholdResult:
        y_true = np.asarray(y_true, dtype=int)
        probabilities = np.asarray(probabilities, dtype=float)

        # Handle edge case where no positive samples exist in fold
        if y_true.sum() == 0:
            logger.warning("No positive samples present in validation set. Using fallback threshold.")
            return ThresholdResult(self.default_fallback, f1=0.0, precision=0.0, recall=0.0)

        thresholds = np.linspace(self.lower, self.upper, self.steps)

        # Vectorized generation of predictions matrix (N_thresholds, N_samples)
        preds = (probabilities[:, None] >= thresholds[None, :]).astype(int)

        # Vectorized calculation of TP, FP, FN
        y_true_col = y_true[:, None]
        tp = (preds & y_true_col).sum(axis=0)
        fp = (preds & (1 - y_true_col)).sum(axis=0)
        fn = ((1 - preds) & y_true_col).sum(axis=0)

        # Calculate metrics vector
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(tp, dtype=float),
            where=(precision + recall) != 0,
        )

        best_candidate = None
        unconstrained_best = None

        for idx, thresh in enumerate(thresholds):
            candidate = ThresholdResult(
                threshold=float(thresh),
                f1=float(f1[idx]),
                precision=float(precision[idx]),
                recall=float(recall[idx]),
            )

            # Track global max F1 regardless of constraints
            if unconstrained_best is None or (candidate.f1, candidate.recall) > (
                unconstrained_best.f1,
                unconstrained_best.recall,
            ):
                unconstrained_best = candidate

            # Apply constraints check
            if self.min_recall is not None and candidate.recall < self.min_recall:
                continue
            if self.min_precision is not None and candidate.precision < self.min_precision:
                continue

            if best_candidate is None or (candidate.f1, candidate.recall) > (
                best_candidate.f1,
                best_candidate.recall,
            ):
                best_candidate = candidate

        # Fallback handling if constraints fail to yield a result
        if best_candidate is None:
            logger.warning(
                "No threshold satisfied constraints (min_recall=%s, min_precision=%s). "
                "Falling back to unconstrained F1 optimal threshold.",
                self.min_recall,
                self.min_precision,
            )
            return unconstrained_best or ThresholdResult(self.default_fallback, 0.0, 0.0, 0.0)

        return best_candidate