"""
Configuration for the avalanche ML training pipeline.
"""

from pathlib import Path 

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_avalanche_dataset.parquet"
)

MODEL_OUTPUT_DIR = PROJECT_ROOT / "trained_models"
RESEARCH_OUTPUT_DIR = PROJECT_ROOT / "research_outputs"

# Reproducibility
RANDOM_STATE = 47

# Avalanche Problems
AVALANCHE_PROBLEMS = (
    "wind_slab",
    "storm_slab",
    "wet_loose",
    "persistent_slab",
)

ASPECTS = (
    "N",
    "NE",
    "E",
    "SE",
    "S",
    "SW",
    "W",
    "NW",
)

# Forecast Target Construction
FORECAST_DAYS = 3

PRESENT_TARGET_SUFFIX = "_present"
FUTURE_TARGET_SUFFIX = "_future"

# Temporal split
TRAIN_DATE_RATIO = 0.70
VALIDATION_DATE_RATIO = 0.85
TEST_DATE_RATIO = 1.00

# Feature Policy
DATE_COLUMNS = {
    "date",
    "year",
    "month",
    "day",
    "day_of_year",
}

# ============================================================================
# AVALANCHE TARGETS / LEAKAGE
# ============================================================================
#
# Goal:
#   Predict avalanche problems at prediction time:
#     1. Whether each problem is present
#     2. Likelihood of each problem
#     3. Size of each problem
#
# IMPORTANT:
#   Any column describing the CURRENT avalanche problem/outcome must NOT be
#   available to the model as a predictor.
#
# This includes:
#   *_present
#   *_likelihood
#   *_size
#   *_aspects
#   *_elevations
#   *_aspects_list
#
# ============================================================================
EXPLICIT_LEAKAGE_COLUMNS = {
    "winter_start",
    "season",
    "year",
    "month",
    "day",
    "day_of_year",
}

# ----------------------------------------------------------------------------
# Avalanche problem types
# ----------------------------------------------------------------------------
#
# Include every problem type that may appear in the dataset. Keeping this
# centralized prevents leakage when new target columns are added.
#
AVALANCHE_PROBLEM_TYPES = {
    "storm_slab",
    "wind_slab",
    "wet_loose",
    "dry_loose",
    "wet_slab",
    "persistent_slab",
}


# ----------------------------------------------------------------------------
# Target columns
# ----------------------------------------------------------------------------
#
# These are the direct prediction targets.
#
AVALANCHE_TARGET_SUFFIXES = {
    "present",
    "likelihood",
    "size",
    "aspects",
    "elevations",
}


AVALANCHE_TARGET_COLUMNS = {
    f"{problem}_{suffix}"
    for problem in AVALANCHE_PROBLEM_TYPES
    for suffix in AVALANCHE_TARGET_SUFFIXES
}


# ----------------------------------------------------------------------------
# Current-problem metadata
# ----------------------------------------------------------------------------
#
# These columns contain structured representations of CURRENT avalanche
# problem characteristics. They are leakage even when they are not themselves
# prediction targets.
#
CURRENT_PROBLEM_METADATA_SUFFIXES = {
    "size",
    "aspects",
    "elevations",
    "aspects_list",
}


CURRENT_PROBLEM_METADATA_COLUMNS = {
    f"{problem}_{suffix}"
    for problem in AVALANCHE_PROBLEM_TYPES
    for suffix in CURRENT_PROBLEM_METADATA_SUFFIXES
}


# ----------------------------------------------------------------------------
# Explicit leakage columns
# ----------------------------------------------------------------------------
#
# Columns that should never be used as predictors because they directly encode
# seasonal grouping, target construction, or other information unavailable at
# the intended prediction point.
#
EXPLICIT_LEAKAGE_COLUMNS = {
    "winter_start",
}


# ----------------------------------------------------------------------------
# Combine all columns that must be excluded from X
# ----------------------------------------------------------------------------

EXCLUDED_FROM_FEATURES = (
    AVALANCHE_TARGET_COLUMNS
    | CURRENT_PROBLEM_METADATA_COLUMNS
    | EXPLICIT_LEAKAGE_COLUMNS
)

IDENTIFIER_COLUMNS = {
    "id",
    "index",
}

# Seasonal feature policy
SEASONAL_FEATURES = (
    "day_sin",
    "day_cos",
)

RESTRICTED_FEATURE_WEIGHTS = {
    "day_sin": 0.10,
    "day_cos": 0.10,
}

MAX_SEASONAL_IMPORTANCE = 0.10

# Feature Engineering

ENGINEERED_FEATURES = (
    "wind_dir_sin",
    "wind_dir_cos",
    "wind_loading_index",
    "wind_persistence_index",
    "storm_loading_index",
    "new_snow_ratio",
    "rapid_warming_event",
    "warming_week_index",
    "freeze_thaw_instability",
    "persistent_slab_setup",
    "heavy_snow_loading_event",
    "day_sin",
    "day_cos",
)

# Missing value policy
FILL_NUMERIC_NANS = False

# Optimization defaults for later batches
OPTIMIZATION_METRIC = "PR_AUC"
THRESHOLD_METRIC = "F1"

EARLY_STOPPING_ROUNDS = 100

CATBOOST_BASE_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "verbose": False,
    "allow_writing_files": False,
}

CATBOOST_SEARCH_SPACE = {
    "depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "l2_leaf_reg": [1, 3, 5, 10, 20],
    "random_strength": [0.0, 0.25, 0.5, 1.0, 2.0],
    "bagging_temperature": [0.0, 0.25, 0.5, 1.0],
    "iterations": [1000, 2000, 3000, 5000],
    "scale_pos_weight": [
        1.0, 
        1.5,
        2.0,
        3.0,
        5.0,
        10.0,
        15.0,
        20.0,
    ],
}

# Maximum number of candidate configurations per target in the first
# optimization pass. Later batches can use randomized/Bayesian search.
MAX_SEARCH_TRIALS = 30

# Validation thresholds
MIN_TRAIN_SAMPLES = 100
MIN_VALIDATION_SAMPLES = 50
MIN_TEST_SAMPLES = 50

MIN_TRAIN_POSITIVES = 10
MIN_VALIDATION_POSITIVES = 5
MIN_TEST_POSITIVES = 5


def future_target_name(target: str) -> str:
    """Return the forecast target name for a current target."""
    return f"{target}{FUTURE_TARGET_SUFFIX}"


def all_current_targets() -> list[str]:
    """Return present + aspect targets for the four supported problems."""
    targets = []

    for problem in AVALANCHE_PROBLEMS:
        targets.append(f"{problem}{PRESENT_TARGET_SUFFIX}")

        for aspect in ASPECTS:
            targets.append(f"{problem}_{aspect}")

    return targets


def all_forecast_targets() -> list[str]:
    """Return future versions of all supported targets."""
    return [future_target_name(target) for target in all_current_targets()]

def is_avalanche_metadata_column(column: str) -> bool:
    """
    Return True when a column describes a current avalanche problem
    rather than weather/terrain available at prediction time.
    """

    suffixes = (
        "_present",
        "_likelihood",
        "_size",
        "_aspects",
        "_elevations",
        "_aspects_list",
    )

    return (
        any(
            column.startswith(f"{problem}_")
            for problem in AVALANCHE_PROBLEM_TYPES
        )
        and any(
            column.endswith(suffix)
            for suffix in suffixes
        )
    )

def get_target_param_grid(target_name: str, positive_rate: float) -> dict:
    """Dynamically generates XGBoost/LightGBM hyperparameter search spaces

    tailored to the target's imbalance ratio and vulnerability to overfitting.
    """
    # Calculate natural baseline ratio
    base_pos_weight = (1.0 - positive_rate) / max(positive_rate, 0.001)

    # Base grid with strong regularization for small dataset sizes
    grid = {
        "max_depth": [3, 4, 5],  # Keep trees shallow to prevent 2026 drift
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.6, 0.7, 0.8],  # Force feature sampling
        "learning_rate": [0.01, 0.03, 0.05],
    }

    # Custom scale_pos_weight bounds per target class
    if "persistent_slab" in target_name:
        # Extreme rare event (~1%)
        grid["scale_pos_weight"] = [base_pos_weight * factor for factor in [0.3, 0.5, 0.8]]
    elif "storm_slab" in target_name:
        # Prone to drift (~12%)
        grid["scale_pos_weight"] = [1.5, 2.5, 3.5]
    else:
        # Moderate targets (wind_slab, wet_loose)
        grid["scale_pos_weight"] = [1.0, 1.5, 2.0]

    return grid