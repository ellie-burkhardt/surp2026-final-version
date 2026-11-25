"""
CatBoost hyperparameter optimization for one binary avalanche target.

Model selection uses validation PR-AUC. The test set is never used here.

The prevalence baseline is reported for context:
    baseline PR-AUC ~= positive prevalence

This allows us to distinguish:
    - raw PR-AUC
    - improvement over the naive prevalence baseline

Seasonal feature permutation importance is reported as a diagnostic only.
It is NOT used for model selection.
"""

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from ..config import (
    CATBOOST_BASE_PARAMS,
    CATBOOST_SEARCH_SPACE,
    EARLY_STOPPING_ROUNDS,
    MAX_SEARCH_TRIALS,
    MIN_TRAIN_POSITIVES,
    MIN_VALIDATION_POSITIVES,
    RANDOM_STATE,
    RESTRICTED_FEATURE_WEIGHTS,
)


@dataclass
class CandidateResult:
    params: dict[str, Any]

    validation_pr_auc: float
    validation_pr_auc_baseline: float
    validation_pr_auc_lift: float

    validation_roc_auc: float

    seasonal_importance: float
    seasonal_importances: dict[str, float]

    best_iteration: int | None


@dataclass
class OptimizationResult:
    target: str

    best_params: dict[str, Any]

    best_validation_pr_auc: float
    best_validation_pr_auc_baseline: float
    best_validation_pr_auc_lift: float

    best_validation_roc_auc: float

    seasonal_importance: float
    seasonal_importances: dict[str, float]

    best_iteration: int | None

    trials: list[CandidateResult]


class CatBoostOptimizer:
    def __init__(
        self,
        random_state=RANDOM_STATE,
        max_trials=MAX_SEARCH_TRIALS,
    ):
        self.random_state = random_state
        self.max_trials = max_trials

    def optimize(
        self,
        target,
        X_train,
        y_train,
        X_validation,
        y_validation,
    ):
        y_train = pd.to_numeric(y_train, errors="raise").astype(int)
        y_validation = pd.to_numeric(
            y_validation,
            errors="raise",
        ).astype(int)

        self._validate_target(
            target,
            y_train,
            y_validation,
        )

        candidates = self._generate_candidates()
        results = []

        validation_prevalence = float(y_validation.mean())

        print("\n" + "=" * 70)
        print(f"OPTIMIZING TARGET: {target}")
        print("=" * 70)

        print(
            f"Validation positives: {int(y_validation.sum())} "
            f"/ {len(y_validation)} "
            f"({validation_prevalence:.3%})"
        )

        print(
            f"PR-AUC baseline: {validation_prevalence:.5f}"
        )

        print(f"Candidate configurations: {len(candidates)}")

        for i, candidate in enumerate(candidates, 1):
            result = self._evaluate(
                target,
                candidate,
                X_train,
                y_train,
                X_validation,
                y_validation,
                i,
                len(candidates),
            )

            results.append(result)

        # Primary selection criterion:
        # validation PR-AUC.
        #
        # ROC-AUC is only used as a deterministic tie-breaker.
        best = max(
            results,
            key=lambda r: (
                r.validation_pr_auc,
                r.validation_roc_auc
                if np.isfinite(r.validation_roc_auc)
                else -np.inf,
            ),
        )

        print("-" * 70)
        print(f"BEST VALID MODEL: {target}")

        print(
            f"PR-AUC:            "
            f"{best.validation_pr_auc:.5f}"
        )

        print(
            f"PR-AUC baseline:   "
            f"{best.validation_pr_auc_baseline:.5f}"
        )

        print(
            f"PR-AUC lift:       "
            f"{best.validation_pr_auc_lift:.2f}x"
        )

        if np.isfinite(best.validation_roc_auc):
            print(
                f"ROC-AUC:           "
                f"{best.validation_roc_auc:.5f}"
            )
        else:
            print("ROC-AUC:           undefined")

        print(
            f"Seasonal importance: "
            f"{best.seasonal_importance:.2%}"
        )

        print(
            f"Best iteration: "
            f"{best.best_iteration}"
        )

        return OptimizationResult(
            target=target,
            best_params=best.params,

            best_validation_pr_auc=best.validation_pr_auc,

            best_validation_pr_auc_baseline=(
                best.validation_pr_auc_baseline
            ),

            best_validation_pr_auc_lift=(
                best.validation_pr_auc_lift
            ),

            best_validation_roc_auc=(
                best.validation_roc_auc
            ),

            seasonal_importance=(
                best.seasonal_importance
            ),

            seasonal_importances=(
                best.seasonal_importances
            ),

            best_iteration=best.best_iteration,

            trials=results,
        )

    def _generate_candidates(self):
        keys = list(CATBOOST_SEARCH_SPACE)

        candidates = [
            dict(zip(keys, values))
            for values in product(
                *(CATBOOST_SEARCH_SPACE[k] for k in keys)
            )
        ]

        rng = np.random.default_rng(
            self.random_state
        )

        if len(candidates) > self.max_trials:
            indices = rng.choice(
                len(candidates),
                size=self.max_trials,
                replace=False,
            )

            candidates = [
                candidates[i]
                for i in indices
            ]

        return candidates

    def _evaluate(
        self,
        target,
        candidate,
        X_train,
        y_train,
        X_validation,
        y_validation,
        number,
        total,
    ):
        params = dict(CATBOOST_BASE_PARAMS)
        params.update(candidate)

        params["random_seed"] = self.random_state

        # Restrict only the explicitly designated features.
        params["feature_weights"] = [
            RESTRICTED_FEATURE_WEIGHTS.get(
                c,
                1.0,
            )
            for c in X_train.columns
        ]

        model = CatBoostClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=(
                X_validation,
                y_validation,
            ),
            early_stopping_rounds=(
                EARLY_STOPPING_ROUNDS
            ),
            verbose=False,
        )

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        # ------------------------------------------------------------
        # PRIMARY METRIC: AVERAGE PRECISION / PR-AUC
        # ------------------------------------------------------------

        pr_auc = average_precision_score(
            y_validation,
            probabilities,
        )

        # ------------------------------------------------------------
        # PREVALENCE BASELINE
        #
        # For a random classifier, average precision is approximately
        # the positive prevalence.
        # ------------------------------------------------------------

        prevalence = float(
            y_validation.mean()
        )

        if prevalence > 0:
            pr_auc_lift = (
                float(pr_auc) / prevalence
            )
        else:
            pr_auc_lift = np.nan

        # ------------------------------------------------------------
        # ROC-AUC
        #
        # ROC-AUC is secondary. It can be undefined when the validation
        # set contains only one class.
        # ------------------------------------------------------------

        if y_validation.nunique() >= 2:
            roc_auc = roc_auc_score(
                y_validation,
                probabilities,
            )
        else:
            roc_auc = np.nan

        # ------------------------------------------------------------
        # SEASONAL PERMUTATION IMPORTANCE
        #
        # Diagnostic only -- never used for model selection.
        # ------------------------------------------------------------

        seasonal = self._seasonal_importance(
            model,
            X_validation,
            y_validation,
        )

        seasonal_total = sum(
            seasonal.values()
        )

        print(
            f"[{number:>2}/{total}] "
            f"PR-AUC={pr_auc:.5f} | "
            f"Baseline={prevalence:.5f} | "
            f"Lift={pr_auc_lift:.2f}x | "
            f"ROC-AUC="
            f"{roc_auc:.5f} | "
            f"weight="
            f"{params['scale_pos_weight']:.2f} | "
            f"seasonal="
            f"{seasonal_total:.2%}"
        )

        return CandidateResult(
            params=params.copy(),

            validation_pr_auc=float(
                pr_auc
            ),

            validation_pr_auc_baseline=float(
                prevalence
            ),

            validation_pr_auc_lift=float(
                pr_auc_lift
            ),

            validation_roc_auc=float(
                roc_auc
            ),

            seasonal_importance=float(
                seasonal_total
            ),

            seasonal_importances=seasonal,

            best_iteration=self._best_iteration(
                model
            ),
        )

    @staticmethod
    def _seasonal_importance(
        model,
        X,
        y,
    ):
        features = [
            f
            for f in RESTRICTED_FEATURE_WEIGHTS
            if f in X.columns
        ]

        if not features:
            return {}

        # If validation contains only one class, permutation importance
        # using average precision is not informative.
        if y.nunique() < 2:
            return {
                f: 0.0
                for f in features
            }

        result = permutation_importance(
            model,
            X,
            y,
            scoring="average_precision",
            n_repeats=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        importance = pd.Series(
            result.importances_mean,
            index=X.columns,
        ).clip(lower=0)

        total = float(
            importance.sum()
        )

        if total <= 0:
            return {
                f: 0.0
                for f in features
            }

        return {
            f: float(
                importance.get(f, 0.0)
                / total
            )
            for f in features
        }

    @staticmethod
    def _validate_target(
        target,
        y_train,
        y_validation,
    ):
        train_positive = int(
            y_train.sum()
        )

        validation_positive = int(
            y_validation.sum()
        )

        if train_positive < MIN_TRAIN_POSITIVES:
            raise ValueError(
                f"{target}: too few positive "
                f"training examples "
                f"({train_positive})."
            )

        if validation_positive < MIN_VALIDATION_POSITIVES:
            raise ValueError(
                f"{target}: too few positive "
                f"validation examples "
                f"({validation_positive})."
            )

        if y_train.nunique() < 2:
            raise ValueError(
                f"{target}: training set "
                f"contains only one class."
            )

        if y_validation.nunique() < 2:
            raise ValueError(
                f"{target}: validation set "
                f"contains only one class."
            )

    @staticmethod
    def _best_iteration(model):
        value = model.get_best_iteration()

        return (
            None
            if value is None or value < 0
            else int(value)
        )