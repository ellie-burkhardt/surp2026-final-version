import pandas as pd
import numpy as np

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)

DATASET = "data/processed/final_avalanche_dataset.parquet"

TARGETS = [
    "wind_slab_present_future",
    "storm_slab_present_future",
    "wet_loose_present_future",
    "persistent_slab_present_future",
]

MOUNTAIN_COL = "mountain"
DATE_COL = "date"


def load_data():

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df = pd.read_parquet(DATASET)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


def target_distribution(df):

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    for target in TARGETS:

        print(f"\n--- {target} ---")

        if target not in df.columns:
            print("MISSING")
            continue

        y = df[target].dropna()

        print(f"Total:     {len(y):,}")
        print(f"Positive:  {int(y.sum()):,}")
        print(f"Negative:  {int((y == 0).sum()):,}")
        print(f"Rate:      {y.mean():.4f}")

        print("\nBy year:")

        year_stats = (
            df.groupby(df[DATE_COL].dt.year)[target]
            .agg(["count", "sum", "mean"])
        )

        print(year_stats.to_string())

        print("\nBy month:")

        month_stats = (
            df.groupby(df[DATE_COL].dt.month)[target]
            .agg(["count", "sum", "mean"])
        )

        print(month_stats.to_string())

        print("\nBy mountain:")

        mountain_stats = (
            df.groupby(MOUNTAIN_COL)[target]
            .agg(["count", "sum", "mean"])
        )

        print(mountain_stats.to_string())


def test_season_distribution(df):

    print("\n" + "=" * 70)
    print("2026 TEST SEASON")
    print("=" * 70)

    test = df[df[DATE_COL].dt.year == 2026].copy()

    print(f"2026 rows: {len(test):,}")

    for target in TARGETS:

        y = test[target].dropna()

        print(
            f"{target:35s} "
            f"positive={int(y.sum()):4d} "
            f"rate={y.mean():.4f}"
        )


def compare_years(df):

    print("\n" + "=" * 70)
    print("YEAR-TO-YEAR TARGET DISTRIBUTION")
    print("=" * 70)

    years = sorted(df[DATE_COL].dt.year.unique())

    for target in TARGETS:

        print(f"\n--- {target} ---")

        table = (
            df.groupby(df[DATE_COL].dt.year)[target]
            .mean()
            .to_frame("positive_rate")
        )

        table["positive_rate"] *= 100

        print(table.to_string())


def feature_distribution(df):

    print("\n" + "=" * 70)
    print("FEATURE DISTRIBUTION SHIFT")
    print("=" * 70)

    exclude = set(
        TARGETS
        + [
            "wind_slab_present",
            "storm_slab_present",
            "wet_loose_present",
            "persistent_slab_present",
            DATE_COL,
            MOUNTAIN_COL,
        ]
    )

    feature_cols = [
        c for c in df.columns
        if c not in exclude
        and not c.endswith("_future")
    ]

    print(f"Candidate features: {len(feature_cols)}")

    train = df[df[DATE_COL].dt.year < 2026]
    test = df[df[DATE_COL].dt.year == 2026]

    results = []

    for feature in feature_cols:

        if not pd.api.types.is_numeric_dtype(df[feature]):
            continue

        train_values = train[feature].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        test_values = test[feature].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        if len(train_values) == 0 or len(test_values) == 0:
            continue

        train_mean = train_values.mean()
        test_mean = test_values.mean()

        train_std = train_values.std()

        if train_std > 0:
            standardized_shift = (
                (test_mean - train_mean) / train_std
            )
        else:
            standardized_shift = 0.0

        results.append({
            "feature": feature,
            "train_mean": train_mean,
            "test_mean": test_mean,
            "train_std": train_std,
            "standardized_shift": standardized_shift,
            "abs_shift": abs(standardized_shift),
        })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        "abs_shift",
        ascending=False,
    )

    print("\nLargest feature shifts in 2026:")

    print(
        result_df.head(30).to_string(index=False)
    )

    result_df.to_csv(
        "feature_distribution_shift.csv",
        index=False,
    )

    print(
        "\nSaved: feature_distribution_shift.csv"
    )


def leakage_scan(df):

    print("\n" + "=" * 70)
    print("POSSIBLE FUTURE-LEAKAGE SCAN")
    print("=" * 70)

    suspicious = []

    for col in df.columns:

        lower = col.lower()

        if (
            "future" in lower
            or "next" in lower
            or "lead" in lower
            or "tomorrow" in lower
            or "forecast" in lower
        ):
            suspicious.append(col)

    print("\nColumns with potentially future-looking names:")

    for col in suspicious:
        print("  ", col)

    print(
        f"\nFound {len(suspicious)} potentially suspicious columns."
    )


def main():

    df = load_data()

    target_distribution(df)

    test_season_distribution(df)

    compare_years(df)

    feature_distribution(df)

    leakage_scan(df)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()