"""
Dataset loading and target construction for the avalanche ML pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np
import pandas as pd

import re
from typing import List, Tuple


from ..config import (
    ASPECTS,
    AVALANCHE_PROBLEMS,
    FORECAST_DAYS,
    FUTURE_TARGET_SUFFIX,
    PRESENT_TARGET_SUFFIX,
)

# Extra funny function 
# ml/features.py
import re
import pandas as pd
from typing import List, Tuple


def prune_avalanche_features(
    df: pd.DataFrame, verbose: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """Systematically prunes redundant lag, long-horizon, and high-cardinality

    directional features from the avalanche dataset.
    """
    initial_cols = df.columns.tolist()

    # Regex patterns for feature pruning
    drop_patterns = [
        r".*_(30d|60d)$",  # Long-horizon aggregates
        r".*_lag[23]$",  # Secondary/tertiary daily lags
        r"^wind_load_(n|s|e|w|ne|nw|se|sw)_change_(1d|3d)$",  # Directional changes
        r"^wind_load_(n|s|e|w|ne|nw|se|sw)_max7d$",  # Directional maxes
        r".*_new$",  # Duplicate ending variants
        r"^precip_(14d_sum|3d_mean|7d_mean|5d)$",  # Redundant intermediate precip
        r"^snow_depth_(5d|14d)$",  # Redundant intermediate snow
        r"^snowfall_5d$",
        r"^swe_5d$",
    ]

    combined_pattern = "|".join(drop_patterns)

    # Protected columns to NEVER drop regardless of pattern matches
    protected_cols = {
        "date",
        "year",
        "month",
        "day",
        "mountain",
        "season",
    }  # Add any other metadata keys here

    # Identify candidate columns to drop
    to_drop = []
    for col in initial_cols:
        # Don't drop protected metadata or target columns (e.g. anything containing '_present' or '_future')
        if (
            col in protected_cols
            or "present" in col
            or "future" in col
            or "target" in col
        ):
            continue

        if re.match(combined_pattern, col, re.IGNORECASE):
            to_drop.append(col)

    df_pruned = df.drop(columns=to_drop)

    if verbose:
        print(f"\n==================================================")
        print(f"FEATURE PRUNING SUMMARY")
        print(f"==================================================")
        print(f"Initial columns : {len(initial_cols)}")
        print(f"Columns dropped : {len(to_drop)}")
        print(f"Remaining cols  : {len(df_pruned.columns)}")

    return df_pruned, to_drop

@dataclass
class DatasetBundle:
    """Immutable-style container returned by the dataset preparation stage."""

    df: pd.DataFrame
    current_targets: list[str]
    forecast_targets: list[str]
    problem_targets: list[str]
    aspect_targets: list[str]

    @property
    def dates(self) -> pd.Series:
        """Return the parsed date series."""
        return pd.to_datetime(self.df["date"])

class AvalancheDataset:
    """
    Load and validate the avalanche dataset and create future targets.

    The original pipeline constructs future labels by sorting by
    mountain/date and shifting each target by forecast_days within
    mountain. That behavior is preserved here.
    """

    def __init__(
        self,
        filename: str | Path,
        forecast_days: int = FORECAST_DAYS,
    ):
        self.filename = Path(filename)
        self.forecast_days = forecast_days

        self.df: pd.DataFrame | None = None
        self.raw_df: pd.DataFrame | None = None

        self.problem_targets: list[str] = []
        self.aspect_targets: list[str] = []
        self.current_targets: list[str] = []
        self.forecast_targets: list[str] = []
        self.problem_groups: dict[str, list[str]] = {}

    def load(self) -> "AvalancheDataset":
        """Load the parquet dataset."""
        if not self.filename.exists():
            raise FileNotFoundError(
                f"Could not find dataset:\n{self.filename.resolve()}"
            )

        self.df = pd.read_parquet(self.filename)
        self.raw_df = self.df.copy()

        print("=" * 70)
        print("DATASET LOADED")
        print("=" * 70)
        print(f"Path: {self.filename}")
        print(f"Rows: {len(self.df):,}")
        print(f"Columns: {len(self.df.columns)}")

        return self
    
    def validate(self) -> "AvalancheDataset":
        """Validate the minimum structure required by the pipeline."""
        self._require_loaded()

        required = {"year", "month", "day", "mountain"}

        missing = sorted(required - set(self.df.columns))

        if missing:
            raise ValueError(
                "Dataset is missing required columns: "
                f"{missing}"
            )

        if self.df.empty:
            raise ValueError("Dataset contains zero rows.")

        return self
    
    def build_dates(self) -> "AvalancheDataset":
        """Construct validated date and season columns."""
        self._require_loaded()

        for column in ("year", "month", "day"):
            self.df[column] = pd.to_numeric(
                self.df[column],
                errors="coerce",
            )

        date_parts = self.df[["year", "month", "day"]]

        if date_parts.isna().any().any():
            bad = self.df.loc[
                date_parts.isna().any(axis=1),
                ["year", "month", "day"],
            ]

            raise ValueError(
                "Missing/non-numeric date components:\n"
                f"{bad.head()}"
            )

        invalid_year = (
            (self.df["year"] < 1900)
            | (self.df["year"] > 2100)
        )

        if invalid_year.any():
            raise ValueError(
                "Invalid year values found:\n"
                f"{self.df.loc[invalid_year, ['year', 'month', 'day']].head()}"
            )

        self.df["date"] = pd.to_datetime(
            {
                "year": self.df["year"].astype(int),
                "month": self.df["month"].astype(int),
                "day": self.df["day"].astype(int),
            },
            errors="raise",
        )

        self.df["day_of_year"] = self.df["date"].dt.dayofyear

        # Avalanche season:
        # October–December belong to the following year's season.
        # January–September belong to the current year's season.
        self.df["season"] = (
            self.df["date"].dt.year
            + (self.df["date"].dt.month >= 10).astype(int)
        )

        return self
    
    # Targets
    def identify_targets(self) -> "AvalancheDataset":
        """
        Identify current avalanche-problem targets.

        Supported problems:
        - wind_slab
        - storm_slab
        - wet_loose
        - persistent_slab

        For each problem we recognize:
        - <problem>_present
        - <problem>_<aspect>
        """
        self._require_loaded()

        self.problem_targets = []
        self.aspect_targets = []

        for problem in AVALANCHE_PROBLEMS:
            present = f"{problem}{PRESENT_TARGET_SUFFIX}"

            if present in self.df.columns:
                self.problem_targets.append(present)

            for aspect in ASPECTS:
                target = f"{problem}_{aspect}"

                if target in self.df.columns:
                    self.aspect_targets.append(target)

        self.current_targets = (
            self.problem_targets
            + self.aspect_targets
        )

        if not self.current_targets:
            raise ValueError(
                "No supported avalanche targets were found."
            )

        print("\n===== TARGETS =====")
        print(
            f"Problem targets: {len(self.problem_targets)}"
        )
        print(self.problem_targets)

        print(
            f"Aspect targets: {len(self.aspect_targets)}"
        )
        print(self.aspect_targets)

        return self

    def group_avalanche_problems(self) -> "AvalancheDataset":
        """Group aspect targets by avalanche problem."""
        self.problem_groups = {
            problem: []
            for problem in AVALANCHE_PROBLEMS
        }

        for target in self.aspect_targets:
            for problem in self.problem_groups:
                if target.startswith(f"{problem}_"):
                    self.problem_groups[problem].append(target)
                    break

        return self
    
    # Future targets
    def create_forecast_targets(self) -> "AvalancheDataset":
        """
        Create future targets using an explicit mountain-specific temporal shift.

        For each row at time t, the future target is the target observed
        exactly FORECAST_DAYS rows later within the same mountain.

        Because the dataset has been verified to contain one row per mountain/day,
        shift(-forecast_days) corresponds to t + forecast_days.
        """
        self._require_loaded()

        if "date" not in self.df.columns:
            raise ValueError(
                "Dates must be built before forecast targets."
            )

        if "mountain" not in self.df.columns:
            raise ValueError(
                "A 'mountain' column is required for future targets."
            )

        # Critical: guarantee chronological order before shifting.
        self.df = (
            self.df
            .sort_values(["mountain", "date"])
            .reset_index(drop=True)
        )

        self.forecast_targets = []

        for target in self.current_targets:

            if target not in self.df.columns:
                raise ValueError(
                    f"Target column '{target}' not found in dataset."
                )

            future_target = (
                f"{target}{FUTURE_TARGET_SUFFIX}"
            )

            # Explicitly take the target from t + forecast_days.
            self.df[future_target] = (
                self.df
                .groupby("mountain", sort=False)[target]
                .shift(-self.forecast_days)
            )

            self.forecast_targets.append(
                future_target
            )

        # ------------------------------------------------------------
        # Remove rows for which future labels do not exist.
        # ------------------------------------------------------------

        before = len(self.df)

        self.df = self.df.dropna(
            subset=self.forecast_targets
        ).copy()

        after = len(self.df)

        # ------------------------------------------------------------
        # Validate future labels.
        # ------------------------------------------------------------

        for target in self.forecast_targets:

            values = pd.to_numeric(
                self.df[target],
                errors="raise",
            )

            invalid = ~values.isin([0, 1])

            if invalid.any():
                raise ValueError(
                    f"Future target '{target}' contains values "
                    f"other than 0/1:\n"
                    f"{values[invalid].value_counts()}"
                )

            self.df[target] = values.astype(np.int8)

        # ------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------

        print("\n===== FORECAST TARGETS =====")
        print(
            f"Forecast horizon: {self.forecast_days} days"
        )
        print(
            f"Created: {len(self.forecast_targets)}"
        )

        print(
            f"Rows removed without future labels: "
            f"{before - after:,}"
        )

        print(
            f"Rows remaining: {after:,}"
        )

        print("\nFuture target positive counts:")

        for target in self.forecast_targets:

            positives = int(
                self.df[target].sum()
            )

            total = len(self.df)

            rate = positives / total

            print(
                f"{target:40s} "
                f"positive={positives:4d} "
                f"rate={rate:.3%}"
            )

        return self
    
    # Complete dataset build

    def prepare(self) -> DatasetBundle:
        """Run dataset-level preparation, target creation, and feature

        pruning.
        """
        self.load()
        self.validate()
        self.build_dates()
        self.identify_targets()
        self.group_avalanche_problems()
        self.create_forecast_targets()

        # --- NEW: Prune redundant lag/directional features ---
        # We prune self.df before returning it in the DatasetBundle
        self.df, dropped_cols = prune_avalanche_features(
            self.df, verbose=True
        )

        return DatasetBundle(
            df=self.df.copy(),
            current_targets=list(self.current_targets),
            forecast_targets=list(self.forecast_targets),
            problem_targets=list(self.problem_targets),
            aspect_targets=list(self.aspect_targets),
        )
    
    # Helpers
    def _require_loaded(self) -> None:
        if self.df is None:
            raise RuntimeError(
                "Dataset has not been loaded. "
                "Call .load() first."
            )
        
    # Bonus Function
    def engineer_features(self):
        """
        Create the exact engineered features expected by the trained models.

        IMPORTANT:
        This method must remain synchronized with the feature-engineering
        pipeline used during model training.
        """

        self._require_loaded()

        df = self.df

        # ============================================================
        # Helper
        # ============================================================

        def require(columns, feature_name):
            missing = [
                c for c in columns
                if c not in df.columns
            ]

            if missing:
                raise ValueError(
                    f"Cannot create '{feature_name}'. "
                    f"Missing source features: {missing}"
                )

        # ============================================================
        # WIND DIRECTION
        # ============================================================

        if "wind_dir" in df.columns:

            radians = np.deg2rad(
                pd.to_numeric(
                    df["wind_dir"],
                    errors="coerce",
                )
            )

            df["wind_dir_sin"] = np.sin(radians)
            df["wind_dir_cos"] = np.cos(radians)

        elif (
            "wind_dir_sin" not in df.columns
            or "wind_dir_cos" not in df.columns
        ):

            raise ValueError(
                "Cannot create wind direction features. "
                "Neither 'wind_dir' nor both "
                "'wind_dir_sin'/'wind_dir_cos' exist."
            )

        # ============================================================
        # WIND LOADING INDEX
        # ============================================================

        require(
            ["wind_speed", "snowfall_3d"],
            "wind_loading_index",
        )

        df["wind_loading_index"] = (
            df["wind_speed"] *
            df["snowfall_3d"]
        )

        # ============================================================
        # WIND PERSISTENCE INDEX
        # ============================================================

        require(
            ["wind_speed", "wind_7d_mean"],
            "wind_persistence_index",
        )

        df["wind_persistence_index"] = (
            df["wind_speed"] *
            df["wind_7d_mean"]
        )

        # ============================================================
        # STORM LOADING INDEX
        # ============================================================

        require(
            ["snowfall_3d", "swe_3d"],
            "storm_loading_index",
        )

        df["storm_loading_index"] = (
            df["snowfall_3d"] +
            10.0 * df["swe_3d"]
        )

        # ============================================================
        # NEW SNOW RATIO
        # ============================================================

        require(
            ["snowfall_7d", "snow_depth_30d"],
            "new_snow_ratio",
        )

        df["new_snow_ratio"] = (
            df["snowfall_7d"] /
            (df["snow_depth_30d"] + 1.0)
        )

        # ============================================================
        # RAPID WARMING
        # ============================================================

        require(
            ["temp_change_3d"],
            "rapid_warming_event",
        )

        df["rapid_warming_event"] = (
            df["temp_change_3d"] > 5
        ).astype("int8")

        # ============================================================
        # STRONG WARMING
        # ============================================================

        # Not explicitly listed in the ENGINEERED_FEATURES tuple,
        # but required by your trained model.

        require(
            ["temp_change_3d"],
            "strong_warming_event",
        )

        df["strong_warming_event"] = (
            df["temp_change_3d"] >= 8
        ).astype("int8")

        # ============================================================
        # WARMING WEEK INDEX
        # ============================================================

        require(
            ["temp_change_7d"],
            "warming_week_index",
        )

        df["warming_week_index"] = (
            df["temp_change_7d"]
            .clip(lower=0)
        )

        # ============================================================
        # FREEZE / THAW INSTABILITY
        # ============================================================

        require(
            ["temp_mean_3d", "temp_mean_14d"],
            "freeze_thaw_instability",
        )

        df["freeze_thaw_instability"] = (
            df["temp_mean_3d"] -
            df["temp_mean_14d"]
        )

        # ============================================================
        # PERSISTENT SLAB SETUP
        # ============================================================

        require(
            ["temp_mean_14d", "snowfall_14d"],
            "persistent_slab_setup",
        )

        cold_period = (
            df["temp_mean_14d"] < -5
        )

        loading_after_cold = (
            df["snowfall_14d"] > 10
        )

        df["persistent_slab_setup"] = (
            cold_period &
            loading_after_cold
        ).astype("int8")

        # ============================================================
        # HEAVY SNOW LOADING
        # ============================================================

        require(
            ["swe_7d"],
            "heavy_snow_loading_event",
        )

        df["heavy_snow_loading_event"] = (
            df["swe_7d"] > 25
        ).astype("int8")

        # ============================================================
        # DAY CYCLIC FEATURES
        # ============================================================

        if "date" in df.columns:

            date = pd.to_datetime(
                df["date"],
                errors="coerce",
            )

            day_of_year = (
                date.dt.dayofyear
            )

            df["day_sin"] = np.sin(
                2 * np.pi * day_of_year / 365.25
            )

            df["day_cos"] = np.cos(
                2 * np.pi * day_of_year / 365.25
            )

        elif (
            "day_sin" not in df.columns
            or "day_cos" not in df.columns
        ):

            raise ValueError(
                "Cannot create day_sin/day_cos. "
                "Missing 'date'."
            )

        # ============================================================
        # FINAL CLEANUP
        # ============================================================

        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns

        df[numeric_cols] = (
            df[numeric_cols]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        self.df = df

        return self

        
        def encode_categoricals(self):
            """
            One-hot encode categorical avalanche problem metadata.
            """

            categorical_cols = [

                # Storm slab
                "storm_slab_size",
                "storm_slab_aspects",
                "storm_slab_elevations",

                # Wind slab
                "wind_slab_size",
                "wind_slab_aspects",
                "wind_slab_elevations",

                # Wet loose
                "wet_loose_size",
                "wet_loose_aspects",
                "wet_loose_elevations",

                # Dry loose
                "dry_loose_size",
                "dry_loose_aspects",
                "dry_loose_elevations",

                # Other avalanche problems
                "wet_slab_size",
                "wet_slab_aspects",
                "wet_slab_elevations",

                "persistent_slab_size",
                "persistent_slab_aspects",
                "persistent_slab_elevations",

                "mountain",
            ]


            categorical_cols = [
                c for c in categorical_cols
                if c in self.df.columns
            ]


            self.df = pd.get_dummies(
                self.df,
                columns=categorical_cols,
                drop_first=False
            )

            return self

        def encode_lists(self):
            """
            Convert list-based aspect columns into multi-hot encoded features.
            """

            list_cols = [
                "wind_slab_aspects_list",
                "storm_slab_aspects_list",
                "wet_loose_aspects_list",
                "dry_loose_aspects_list",
                "wet_slab_aspects_list",
                "persistent_slab_aspects_list",
            ]

            for col in list_cols:
                if col not in self.df.columns:
                    continue

                # ensure lists (sometimes stored as strings)
                self.df[col] = self.df[col].apply(
                    lambda x: x if isinstance(x, list) else []
                )

                mlb = MultiLabelBinarizer()

                encoded = mlb.fit_transform(self.df[col])
                
                encoded_df = pd.DataFrame(
                    encoded,
                    index=self.df.index,
                    columns=[
                        f"{col}__{c}"
                        for c in mlb.classes_
                    ]
                )

                self.df = pd.concat([self.df, encoded_df], axis=1)

            return self
        
    def prepare_dataset(self):
        """
        Run the complete preprocessing pipeline in the correct order.
        """

        self.df = self.raw_df.copy()

        self.build_dates()

        print("===== Date Mins and Maxes =====")
        print(self.df["date"].min())
        print(self.df["date"].max())

        self.identify_targets()

        self.create_forecast_targets()

        self.engineer_features()

        self.encode_categoricals()

        self.group_avalanche_problems()

        self.encode_lists()

        print("===== Leakage =====")
        self.remove_leakage()

        self.build_feature_matrix()

        self.processed_df = self.df.copy()

        return self