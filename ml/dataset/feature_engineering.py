"""
Feature engineering and feature-policy layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import re
import pandas as pd
from typing import List, Tuple

from ..config import (
    AVALANCHE_PROBLEMS,
    AVALANCHE_TARGET_COLUMNS,
    CURRENT_PROBLEM_METADATA_COLUMNS,
    DATE_COLUMNS,
    EXPLICIT_LEAKAGE_COLUMNS,
    FUTURE_TARGET_SUFFIX,
    IDENTIFIER_COLUMNS,
    RESTRICTED_FEATURE_WEIGHTS,
    SEASONAL_FEATURES,
)

@dataclass
class FeatureBundle:
    """Final feature matrix plus metadata needed by later training."""

    X: pd.DataFrame
    feature_names: list[str]
    restricted_feature_weights: dict[str, float]
    removed_columns: list[str]


class FeatureEngineer:
    """Construct the numerical predictor matrix."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

        self.feature_names: list[str] = []
        self.removed_columns: list[str] = []

    def build(self) -> FeatureBundle:
        """Run the complete feature-engineering pipeline."""
        self._validate_input()
        self._add_date_features()
        self._add_cyclic_wind_features()
        self._add_avalanche_features()
        self._encode_categorical_features()

        # We deliberately remove leakage/targets before the final
        # feature matrix is constructed.
        X = self._build_feature_matrix()

        print("\n" + "=" * 70)
        print("FEATURE MATRIX AUDIT")
        print("=" * 70)

        print(f"Rows:    {len(X):,}")
        print(f"Features: {len(X.columns):,}")

        print("\nFeatures:")
        for feature in X.columns:
            print(f"  {feature}")

        print("\nDtypes:")
        print(X.dtypes.value_counts())

        print("\nMissing values:")
        missing = X.isna().sum()
        missing = missing[missing > 0].sort_values(
            ascending=False
        )

        if missing.empty:
            print("  ✓ No missing values")
        else:
            print(missing.to_string())

        print("\nPotential leakage names:")
        suspicious = [
            c for c in X.columns
            if any(
                token in c.lower()
                for token in (
                    "present",
                    "likelihood",
                    "future",
                    "elevations",
                    "aspects",
                )
            )
        ]

        if suspicious:
            print("  ⚠️ SUSPICIOUS:")
            for c in suspicious:
                print(f"    {c}")
        else:
            print("  ✓ No obvious avalanche-target names found")

        self.feature_names = list(X.columns)

        return FeatureBundle(
            X=X,
            feature_names=self.feature_names,
            restricted_feature_weights={
                name: weight
                for name, weight in RESTRICTED_FEATURE_WEIGHTS.items()
                if name in X.columns
            },
            removed_columns=self.removed_columns,
        )

    # Date features
    def _add_date_features(self) -> None:
        """
        Add cyclic annual-season features.

        We use day-of-year rather than raw month/day so the model gets a
        smooth representation of seasonal progression.

        day_sin/day_cos are deliberately retained and restricted later.
        """
        if "date" not in self.df.columns:
            raise ValueError(
                "Feature engineering requires a 'date' column."
            )

        day_of_year = self.df["date"].dt.dayofyear.astype(float)

        # 365.25 prevents a hard discontinuity around leap years.
        angle = 2.0 * np.pi * (
            (day_of_year - 1.0) / 365.25
        )

        self.df["day_sin"] = np.sin(angle)
        self.df["day_cos"] = np.cos(angle)

    # Wind direction
    def _add_cyclic_wind_features(self) -> None:
        """Encode wind direction continuously instead of as a raw angle."""
        if "wind_dir" not in self.df.columns:
            return

        radians = np.deg2rad(
            pd.to_numeric(
                self.df["wind_dir"],
                errors="coerce",
            )
        )

        self.df["wind_dir_sin"] = np.sin(radians)
        self.df["wind_dir_cos"] = np.cos(radians)

        self.df.drop(
            columns=["wind_dir"],
            inplace=True,
        )
    
    # Avalanche-specific engineered features
    def _add_avalanche_features(self) -> None:
        """Add the engineered features from the original pipeline."""
        self._wind_features()
        self._storm_features()
        self._warming_features()
        self._persistent_slab_features()
        self._heavy_loading_features()
        #self._dry_loose_features()

    def _dry_loose_features(self) -> None:
        """Engineered indicators specifically for  sluffs."""
        # Un-consolidated cold snow indicator (recent snow + sustained cold + low wind)
        if {"snowfall_3d", "temp_mean_3d", "wind_speed"}.issubset(self.df.columns):
            self.df["dry_loose_powder_index"] = (
                (self.df["snowfall_3d"] > 5.0)
                & (self.df["temp_mean_3d"] < -2.0)
                & (self.df["wind_speed"] < 15.0)
            ).astype("int8")

    def _wind_features(self) -> None:
        if {
            "wind_speed",
            "snowfall_3d",
        }.issubset(self.df.columns):
            self.df["wind_loading_index"] = (
                self.df["wind_speed"]
                * self.df["snowfall_3d"]
            )

        if {
            "wind_speed",
            "wind_7d_mean",
        }.issubset(self.df.columns):
            self.df["wind_persistence_index"] = (
                self.df["wind_speed"]
                * self.df["wind_7d_mean"]
            )

    def _storm_features(self) -> None:
        if {
            "snowfall_3d",
            "swe_3d",
        }.issubset(self.df.columns):
            # Preserved from the original feature definition.
            self.df["storm_loading_index"] = (
                self.df["snowfall_3d"]
                + 10.0 * self.df["swe_3d"]
            )

        if {
            "snowfall_7d",
            "snow_depth_30d",
        }.issubset(self.df.columns):
            self.df["new_snow_ratio"] = (
                self.df["snowfall_7d"]
                / (self.df["snow_depth_30d"] + 1.0)
            )

    def _warming_features(self) -> None:
        if "temp_change_3d" in self.df.columns:
            temp_change = pd.to_numeric(
                self.df["temp_change_3d"],
                errors="coerce",
            )

            self.df["rapid_warming_event"] = (
                temp_change > 5.0
            ).astype("int8")

        if "temp_change_7d" in self.df.columns:
            self.df["warming_week_index"] = (
                pd.to_numeric(
                    self.df["temp_change_7d"],
                    errors="coerce",
                )
                .clip(lower=0)
            )

        if {
            "temp_mean_3d",
            "temp_mean_14d",
        }.issubset(self.df.columns):
            self.df["freeze_thaw_instability"] = (
                self.df["temp_mean_3d"]
                - self.df["temp_mean_14d"]
            )

    def _persistent_slab_features(self) -> None:
        if {
            "temp_mean_14d",
            "snowfall_14d",
        }.issubset(self.df.columns):
            cold_period = (
                self.df["temp_mean_14d"] < -5.0
            )

            loading_after_cold = (
                self.df["snowfall_14d"] > 10.0
            )

            self.df["persistent_slab_setup"] = (
                cold_period
                & loading_after_cold
            ).astype("int8")

    def _heavy_loading_features(self) -> None:
        if "swe_7d" in self.df.columns:
            self.df["heavy_snow_loading_event"] = (
                self.df["swe_7d"] > 25.0
            ).astype("int8")

    def _encode_categorical_features(self) -> None:
        """Explicitly encode approved categorical predictor columns."""
        if "mountain" not in self.df.columns:
            return

        mountain = self.df["mountain"].astype("string")

        expected_mountains = {"hood", "rainier", "shasta"}
        observed_mountains = set(mountain.dropna().unique())

        unexpected = observed_mountains - expected_mountains

        if unexpected:
            raise ValueError(
                "Unexpected values found in 'mountain': "
                f"{sorted(unexpected)}. "
                f"Expected only: {sorted(expected_mountains)}."
            )

        encoded = pd.get_dummies(
            mountain,
            prefix="mountain",
            dtype=float,
        )

        self.df = pd.concat(
            [
                self.df.drop(columns=["mountain"]),
                encoded,
            ],
            axis=1,
        )

    # Feature matrix
    def _build_feature_matrix(self) -> pd.DataFrame:
        """
        Construct the final numeric matrix.

        All current/future avalanche labels and explicit leakage columns
        are removed. Raw object/string columns are not silently coerced.
        """
        remove = set()

        # Date representations are not included directly. day_sin and
        # day_cos are the intended seasonal representation.
        remove.update(DATE_COLUMNS)

        # Explicit leakage.
        remove.update(EXPLICIT_LEAKAGE_COLUMNS)

        # Current target labels.
        remove.update(AVALANCHE_TARGET_COLUMNS)

        # Current problem descriptors/lists.
        remove.update(CURRENT_PROBLEM_METADATA_COLUMNS)

        # Generic identifiers.
        remove.update(IDENTIFIER_COLUMNS)

        # All future labels are targets, never predictors.
        future_columns = {
            column
            for column in self.df.columns
            if column.endswith(FUTURE_TARGET_SUFFIX)
        }
        remove.update(future_columns)

                # Conservative avalanche-target safety filter.
        #
        # Anything belonging to a supported avalanche problem is removed
        # unless it is explicitly recognized as a legitimate engineered
        # predictor.
        for column in self.df.columns:

            for problem in AVALANCHE_PROBLEMS:

                if column == problem or column.startswith(
                    f"{problem}_"
                ):

                    # Explicitly allowed engineered predictors can remain.
                    # Explicitly allowed engineered predictors can remain.
                    if column in {
                        "wind_loading_index",
                        "wind_persistence_index",
                        "storm_loading_index",
                        "dry_loose_powder_index",
                    }:
                        continue

                    # Everything else belonging to an avalanche problem
                    # is considered target/metadata and excluded.
                    remove.add(column)

                    break

        existing_remove = [
            column
            for column in self.df.columns
            if column in remove
        ]

        self.removed_columns = sorted(set(existing_remove))

        X = self.df.drop(
            columns=self.removed_columns,
            errors="ignore",
        ).copy()

        # We currently train numeric CatBoost models. Rather than silently
        # one-hot encode arbitrary object columns here, fail loudly so a
        # future feature policy can explicitly decide how to handle them.
        invalid_columns = X.select_dtypes(
            exclude=[np.number, "bool"]
        ).columns.tolist()

        if invalid_columns:
            raise ValueError(
                "Non-numeric predictor columns remain after feature "
                "policy was applied:\n"
                f"{invalid_columns}\n\n"
                "Add them explicitly to features.py rather than "
                "silently coercing them."
            )

        # Remove constants.
        constant_columns = [
            column
            for column in X.columns
            if X[column].nunique(dropna=False) <= 1
        ]

        if constant_columns:
            self.removed_columns.extend(constant_columns)
            X = X.drop(
                columns=constant_columns
            )

        # Remove duplicate column names.
        duplicate_columns = X.columns[
            X.columns.duplicated()
        ].tolist()

        if duplicate_columns:
            self.removed_columns.extend(duplicate_columns)
            X = X.loc[
                :,
                ~X.columns.duplicated(),
            ]

        # Do NOT fill NaNs with zero. CatBoost can handle missing numerical
        # values natively, and zero can be a physically meaningful value.
        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Ensure deterministic feature order.
        X = X.reindex(
            sorted(X.columns),
            axis=1,
        )

        # Final sanity checks.
        if X.empty:
            raise ValueError(
                "Feature matrix is empty after feature filtering."
            )

        if X.columns.duplicated().any():
            raise ValueError(
                "Duplicate feature names remain."
            )

        return X
    
    # Diagnostics
    def diagnostics(self, X: pd.DataFrame) -> dict:
        """Return useful feature diagnostics without training a model."""
        missing = (
            X.isna()
            .sum()
            .sort_values(ascending=False)
        )

        seasonal = {
            feature: {
                "present": feature in X.columns,
                "missing": int(
                    X[feature].isna().sum()
                ) if feature in X.columns else None,
                "dtype": str(
                    X[feature].dtype
                ) if feature in X.columns else None,
            }
            for feature in SEASONAL_FEATURES
        }

        return {
            "rows": len(X),
            "features": len(X.columns),
            "feature_names": list(X.columns),
            "seasonal_features": seasonal,
            "features_with_missing_values": (
                missing[missing > 0].to_dict()
            ),
            "removed_columns": sorted(
                set(self.removed_columns)
            ),
        }

    def _validate_input(self) -> None:
        required = {"date", "mountain"}

        missing = sorted(
            required - set(self.df.columns)
        )

        if missing:
            raise ValueError(
                "Feature engineering is missing required columns: "
                f"{missing}"
            )

# Funny function
def prune_avalanche_features(
        df: pd.DataFrame, verbose: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Systematically prunes redundant lag, long-horizon, and high-cardinality

        directional features from the avalanche feature matrix.
        """
        initial_cols = df.columns.tolist()
        drop_patterns = [
            # 1. Drop extreme long-horizon aggregates (keep <= 7d/14d)
            r".*_(30d|60d)$",
            # 2. Drop intermediate/granular lag variants (keep lag1 or core aggregations)
            r".*_lag[23]$",
            # 3. Drop granular directional wind load change metrics
            r"^wind_load_(n|s|e|w|ne|nw|se|sw)_change_(1d|3d)$",
            # 4. Drop redundant directional max metrics (keep total_wind_loading instead)
            r"^wind_load_(n|s|e|w|ne|nw|se|sw)_max7d$",
            # 5. Drop duplicate/overlapping feature names ending in _new
            r".*_new$",
            # 6. Drop redundant intermediate cumulative windows
            r"^precip_(14d_sum|3d_mean|7d_mean|5d)$",
            r"^snow_depth_(5d|14d)$",
            r"^snowfall_5d$",
            r"^swe_5d$",
        ]

        # Combine into a single regex pattern
        combined_pattern = "|".join(drop_patterns)

        # Identify features to drop
        to_drop = [
            col for col in initial_cols if re.match(combined_pattern, col, re.IGNORECASE)
        ]

        # Clean dataframe
        df_pruned = df.drop(columns=to_drop)

        if verbose:
            print(f"Initial Features : {len(initial_cols)}")
            print(f"Features Dropped : {len(to_drop)}")
            print(f"Remaining        : {len(df_pruned.columns)}")
            print("\nDropped Feature Categories:")
            for pattern in drop_patterns:
                matches = [
                    c for c in to_drop if re.match(pattern, c, re.IGNORECASE)
                ]
                if matches:
                    print(f"  • Pattern '{pattern}': {len(matches)} dropped")

        return df_pruned, to_drop

def add_regional_risk_features(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
        """Computes historical rolling mean risk per mountain location."""
        df = df.sort_values("date").copy()
        
        for target in target_cols:
            if target in df.columns:
                df[f"{target}_mountain_hist_rate"] = (
                    df.groupby("mountain")[target]
                    .transform(lambda x: x.shift(1).expanding().mean())
                    .fillna(df[target].mean())
                )
                
        return df