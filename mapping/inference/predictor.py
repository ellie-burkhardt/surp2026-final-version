"""
predictor.py

Run avalanche model inference.
"""

from typing import Dict, List

import pandas as pd


class AvalanchePredictor:

    PROBLEM_TO_MODEL = {
        "storm_slab":
            "storm_slab_present_future",

        "wind_slab":
            "wind_slab_present_future",

        "wet_loose":
            "wet_loose_present_future",

        "persistent_slab":
            "persistent_slab_present_future",
    }

    def __init__(
        self,
        models: Dict[str, object],
        metadata: Dict[str, dict],
        feature_names: List[str],
    ):

        self.models = models
        self.metadata = metadata
        self.feature_names = feature_names

    def predict(
        self,
        X: pd.DataFrame,
        problem: str = "all",
    ) -> Dict[str, Dict[str, float]]:

        X = self._validate_features(X)

        predictions = {}

        #
        # User-facing problem names:
        #
        # wind_slab
        # storm_slab
        # wet_loose
        # persistent_slab
        #
        # are mapped internally to:
        #
        # wind_slab_present_future
        # storm_slab_present_future
        # etc.
        #

        if problem == "all":

            selected_problems = list(
                self.PROBLEM_TO_MODEL.keys()
            )

        else:

            if problem not in self.PROBLEM_TO_MODEL:

                raise ValueError(
                    f"Unknown avalanche problem "
                    f"'{problem}'. "
                    f"Available problems: "
                    f"{list(self.PROBLEM_TO_MODEL.keys())}"
                )

            selected_problems = [
                problem
            ]

        for problem_name in selected_problems:

            model_name = (
                self.PROBLEM_TO_MODEL[
                    problem_name
                ]
            )

            if model_name not in self.models:

                raise ValueError(
                    f"Model not loaded for avalanche "
                    f"problem '{problem_name}'. "
                    f"Expected model '{model_name}'. "
                    f"Available models: "
                    f"{list(self.models.keys())}"
                )

            if model_name not in self.metadata:

                raise ValueError(
                    f"Metadata not loaded for model "
                    f"'{model_name}'. "
                    f"Available metadata: "
                    f"{list(self.metadata.keys())}"
                )

            model = self.models[
                model_name
            ]

            metadata = self.metadata[
                model_name
            ]

            predictions[
                problem_name
            ] = self._predict_problem(
                model=model,
                metadata=metadata,
                X=X.copy(),
                problem_name=problem_name,
                model_name=model_name,
            )

        return predictions

    def _validate_features(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:

        if not isinstance(
            X,
            pd.DataFrame,
        ):

            raise TypeError(
                "X must be a pandas DataFrame."
            )

        if X.empty:

            raise ValueError(
                "Prediction dataframe is empty."
            )

        #
        # Normalize known capitalization differences.
        #
        rename_map = {}

        for column in X.columns:

            normalized = column

            if column.startswith(
                "wind_load_"
            ):

                normalized = (
                    column.replace(
                        "wind_load_E",
                        "wind_load_e",
                    )
                    .replace(
                        "wind_load_N",
                        "wind_load_n",
                    )
                    .replace(
                        "wind_load_NE",
                        "wind_load_ne",
                    )
                    .replace(
                        "wind_load_NW",
                        "wind_load_nw",
                    )
                    .replace(
                        "wind_load_S",
                        "wind_load_s",
                    )
                    .replace(
                        "wind_load_SE",
                        "wind_load_se",
                    )
                    .replace(
                        "wind_load_SW",
                        "wind_load_sw",
                    )
                    .replace(
                        "wind_load_W",
                        "wind_load_w",
                    )
                )

            elif column == "PREC":

                normalized = "prec"

            if normalized != column:
                rename_map[column] = normalized

        if rename_map:

            X = X.rename(
                columns=rename_map
            )

        return X

    def _get_expected_features(
        self,
        model,
    ):

        if hasattr(
            model,
            "feature_names_",
        ):

            return list(
                model.feature_names_
            )

        raise ValueError(
            "CatBoost model does not contain "
            "feature_names_."
        )

    def _get_targets(
        self,
        metadata,
        model_name,
    ):

        #
        # Your metadata apparently does not necessarily
        # use the exact key "Targets".
        #
        possible_keys = [
            "Targets",
            "targets",
            "Target",
            "target",
        ]

        for key in possible_keys:

            if key in metadata:

                targets = metadata[key]

                if isinstance(
                    targets,
                    str,
                ):

                    return [targets]

                return list(targets)

        raise KeyError(
            f"Could not find target information "
            f"in metadata for '{model_name}'. "
            f"Available metadata keys: "
            f"{list(metadata.keys())}"
        )

    def _predict_problem(
        self,
        model,
        metadata,
        X: pd.DataFrame,
        problem_name: str,
        model_name: str,
    ) -> Dict[str, float]:

        targets = self._get_targets(
            metadata,
            model_name,
        )

        expected = (
            self._get_expected_features(
                model
            )
        )

        #
        # Determine missing features.
        #
        missing = sorted(
            set(expected)
            - set(X.columns)
        )

        #
        # Determine extra features.
        #
        extra = sorted(
            set(X.columns)
            - set(expected)
        )

        #
        # If the schemas don't match, provide a useful
        # diagnostic rather than allowing CatBoost to
        # fail with an obscure error.
        #
        if missing:

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"MODEL FEATURE MISMATCH: "
                f"{problem_name}"
            )

            print(
                "=" * 70
            )

            print(
                f"\nModel: {model_name}"
            )

            print(
                f"Model expects "
                f"{len(expected)} features."
            )

            print(
                f"Dataframe contains "
                f"{len(X.columns)} features."
            )

            print(
                "\n===== MISSING FEATURES ====="
            )

            for feature in missing:

                print(
                    f"  {feature}"
                )

            print(
                "\n===== EXTRA DATAFRAME FEATURES ====="
            )

            for feature in extra:

                print(
                    f"  {feature}"
                )

            print(
                "\nThis usually means the model was "
                "trained with a different feature-engineering "
                "version than the inference dataframe."
            )

            raise ValueError(
                f"Prediction dataframe is missing "
                f"{len(missing)} features for "
                f"'{model_name}'."
            )

        #
        # Reorder exactly to CatBoost's training order.
        #
        X = X.reindex(
            columns=expected
        )

        #
        # Check that the order is exact.
        #
        if list(X.columns) != expected:

            raise ValueError(
                f"Feature order does not match "
                f"the model '{model_name}'."
            )

        #
        # CatBoost prediction.
        #
        probability_output = (
            model.predict_proba(X)
        )

        probabilities = {}

        #
        # Standard CatBoost binary classifier:
        #
        # shape = (n_samples, 2)
        #
        if hasattr(
            probability_output,
            "shape",
        ):

            if len(targets) != 1:

                raise ValueError(
                    f"Model '{model_name}' returned "
                    f"a standard binary probability array, "
                    f"but metadata contains "
                    f"{len(targets)} targets: "
                    f"{targets}"
                )

            if (
                probability_output.ndim != 2
                or probability_output.shape[1] < 2
            ):

                raise ValueError(
                    f"Model '{model_name}' does not "
                    "contain a positive-class probability."
                )

            probabilities[
                targets[0]
            ] = float(
                probability_output[
                    0,
                    1,
                ]
            )

        else:

            for target, probs in zip(
                targets,
                probability_output,
            ):

                probabilities[
                    target
                ] = float(
                    probs[0, 1]
                )

        return probabilities
