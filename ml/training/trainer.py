"""
Independent CatBoost trainer.

Each avalanche target receives its own optimized model rather than being
wrapped in sklearn.MultiOutputClassifier. This allows each target to have
its own depth, regularization, class weighting, stopping point, and
seasonal-feature behavior.
"""

from dataclasses import dataclass

import pandas as pd
from catboost import CatBoostClassifier

from ..config import (
    CATBOOST_BASE_PARAMS,
    EARLY_STOPPING_ROUNDS,
    MAX_SEASONAL_IMPORTANCE,
    RANDOM_STATE,
    RESTRICTED_FEATURE_WEIGHTS,
)
from .optimizer import CatBoostOptimizer, OptimizationResult
from .splitter import TemporalSplit, split_target


@dataclass
class TrainedTarget:
    target: str
    model: CatBoostClassifier
    optimization: OptimizationResult
    threshold: float | None
    feature_names: list[str]
    train_rows: int
    validation_rows: int
    test_rows: int
    train_positive_rate: float
    validation_positive_rate: float
    test_positive_rate: float

    # Out-of-fold validation results
    validation_targets: pd.Series | None = None
    validation_probabilities: pd.Series | None = None


class AvalancheTrainer:
    def __init__(
        self,
        optimizer=None,
        random_state=RANDOM_STATE,
    ):
        self.random_state = random_state
        self.optimizer = optimizer or CatBoostOptimizer(
            random_state=random_state,
        )

    def train_target(
        self,
        target,
        feature_matrix,
        dataframe,
        temporal_splits,
    ):
        if target not in dataframe.columns:
            raise ValueError(
                f"Target '{target}' is not present."
            )

        fold_results = []

        # =========================================================
        # ROLLING-ORIGIN VALIDATION
        # =========================================================
        validation_targets = []
        validation_probabilities = []
        
        for fold_number, temporal_split in enumerate(
            temporal_splits,
            start=1,
        ):
            (
                X_train, y_train,
                X_valid, y_valid,
                _, _,
            ) = split_target(
                temporal_split,
                feature_matrix,
                dataframe[target],
            )

            self._validate_train_validation_alignment(
                X_train,
                y_train,
                X_valid,
                y_valid,
            )

            print(
                f"\n{'=' * 70}\n"
                f"{target} - ROLLING FOLD {fold_number}\n"
                f"{'=' * 70}"
            )

            print(
                f"Train:      {temporal_split.train_start} "
                f"-> {temporal_split.train_end}"
            )
            print(
                f"Validation: {temporal_split.validation_start} "
                f"-> {temporal_split.validation_end}"
            )

            # Optimize this fold independently
            optimization = self.optimizer.optimize(
                target=target,
                X_train=X_train,
                y_train=y_train,
                X_validation=X_valid,
                y_validation=y_valid,
            )

            # Train fold model
            fold_model = self._fit_final_model(
                X_train,
                y_train,
                X_valid,
                y_valid,
                optimization,
            )

            # Validation predictions from a model that NEVER
            # saw this validation period during training.
            valid_probabilities = fold_model.predict_proba(
                X_valid
            )[:, 1]

            validation_targets.append(y_valid)
            validation_probabilities.append(
                pd.Series(
                    valid_probabilities,
                    index=y_valid.index,
                )
            )

            fold_results.append({
                "fold": fold_number,
                "model": fold_model,
                "optimization": optimization,
                "X_train": X_train,
                "y_train": y_train,
                "X_valid": X_valid,
                "y_valid": y_valid,
                "valid_probabilities": valid_probabilities,
            })

            print(
                f"Fold {fold_number}: "
                f"PR-AUC={optimization.best_validation_pr_auc:.5f}"
            )

        if not fold_results:
            raise ValueError(
                f"No valid rolling folds were created for '{target}'."
            )
        
        all_validation_targets = pd.concat(
            validation_targets
        )

        all_validation_probabilities = pd.concat(
            validation_probabilities
        )



        # =========================================================
        # SELECT BEST OPTIMIZATION RESULT
        # =========================================================
        best_fold = max(
            fold_results,
            key=lambda result:
                result["optimization"].best_validation_pr_auc,
        )

        optimization = best_fold["optimization"]

        # =========================================================
        # FINAL MODEL
        # =========================================================
        #
        # The final model is trained on ALL data that was available
        # before the completely untouched test period.
        #
        # We use the original chronological split for this.
        # =========================================================

        final_temporal_split = temporal_splits[-1]

        (
            X_train,
            y_train,
            X_valid,
            y_valid,
            _,
            _,
        ) = split_target(
            final_temporal_split,
            feature_matrix,
            dataframe[target],
        )

        # Combine the final rolling validation period into training.
        X_final_train = pd.concat(
            [X_train, X_valid],
            axis=0,
        )

        y_final_train = pd.concat(
            [y_train, y_valid],
            axis=0,
        )

        # Train final model using the optimized parameters.
        final_model = self._fit_model_without_validation(
            X_final_train,
            y_final_train,
            optimization,
        )

        # =========================================================
        # PRINT ROLLING VALIDATION SUMMARY
        # =========================================================

        fold_scores = [
            result["optimization"].best_validation_pr_auc
            for result in fold_results
        ]

        mean_pr_auc = sum(fold_scores) / len(fold_scores)

        print(
            f"\n{'=' * 70}\n"
            f"ROLLING VALIDATION SUMMARY: {target}\n"
            f"{'=' * 70}"
        )

        for i, score in enumerate(fold_scores, start=1):
            print(
                f"Fold {i} PR-AUC: {score:.5f}"
            )

        print(
            f"Mean PR-AUC:    {mean_pr_auc:.5f}"
        )

        # Use the best fold's validation information for the
        # optimization metadata.
        result = TrainedTarget(
            target=target,
            model=final_model,
            optimization=optimization,
            threshold=None,
            feature_names=list(X_final_train.columns),
            train_rows=len(X_final_train),
            validation_rows=sum(
                len(result["X_valid"])
                for result in fold_results
            ),
            test_rows=0,
            train_positive_rate=float(
                y_final_train.mean()
            ),
            validation_positive_rate=float(
                sum(
                    result["y_valid"].sum()
                    for result in fold_results
                )
                / sum(
                    len(result["y_valid"])
                    for result in fold_results
                )
            ),
            test_positive_rate=0.0,

            # Out-of-fold validation results
            validation_targets=all_validation_targets,
            validation_probabilities=all_validation_probabilities,

        )

        print(
            f"\nTRAINED {target}: "
            f"Mean rolling PR-AUC={mean_pr_auc:.5f}, "
            f"seasonal={optimization.seasonal_importance:.2%}"
        )

        return result

    def train_all(
        self,
        targets,
        feature_matrix,
        dataframe,
        temporal_split,
    ):
        return {
            target: self.train_target(
                target,
                feature_matrix,
                dataframe,
                temporal_split,
            )
            for target in targets
        }

    def _fit_final_model(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
        optimization,
    ):
        params = dict(CATBOOST_BASE_PARAMS)
        params.update(optimization.best_params)
        params["random_seed"] = self.random_state
        params["feature_weights"] = [
            RESTRICTED_FEATURE_WEIGHTS.get(c, 1.0)
            for c in X_train.columns
        ]
        
        model = CatBoostClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
        return model
    
    def _fit_model_without_validation(
        self,
        X_train,
        y_train,
        optimization,
    ):
        params = dict(CATBOOST_BASE_PARAMS)
        params.update(optimization.best_params)

        # Use the stopping point discovered during validation.
        #
        # CatBoost's best_iteration is zero-based, so:
        # best_iteration=842 means 843 trees were used.
        if optimization.best_iteration is not None:
            params["iterations"] = (
                int(optimization.best_iteration) + 1
            )

        params["random_seed"] = self.random_state
        params["feature_weights"] = [
            RESTRICTED_FEATURE_WEIGHTS.get(c, 1.0)
            for c in X_train.columns
        ]

        

        model = CatBoostClassifier(**params)

        model.fit(
            X_train,
            y_train,
            verbose=False,
        )

        return model

    @staticmethod
    def _validate_alignment(
        X_train, y_train,
        X_valid, y_valid,
        X_test, y_test,
    ):
        if len(X_train) != len(y_train):
            raise RuntimeError("Training X/y length mismatch.")
        if len(X_valid) != len(y_valid):
            raise RuntimeError("Validation X/y length mismatch.")
        if len(X_test) != len(y_test):
            raise RuntimeError("Test X/y length mismatch.")

        if not X_train.columns.equals(X_valid.columns):
            raise RuntimeError(
                "Training and validation features differ."
            )
        if not X_train.columns.equals(X_test.columns):
            raise RuntimeError(
                "Training and test features differ."
            )

        if X_train.index.intersection(X_valid.index).size:
            raise RuntimeError("Train/validation indices overlap.")
        if X_train.index.intersection(X_test.index).size:
            raise RuntimeError("Train/test indices overlap.")
        if X_valid.index.intersection(X_test.index).size:
            raise RuntimeError("Something bad happened")

    @staticmethod
    def _validate_train_validation_alignment(
        X_train,
        y_train,
        X_valid,
        y_valid,
    ):
        if len(X_train) != len(y_train):
            raise RuntimeError(
                "Training X/y length mismatch."
            )

        if len(X_valid) != len(y_valid):
            raise RuntimeError(
                "Validation X/y length mismatch."
            )

        if not X_train.columns.equals(X_valid.columns):
            raise RuntimeError(
                "Training and validation features differ."
            )

        if X_train.index.intersection(X_valid.index).size:
            raise RuntimeError(
                "Train/validation indices overlap."
            )