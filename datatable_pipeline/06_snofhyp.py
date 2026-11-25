import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

INPUT_FILE = "data/interim/master_dataset.parquet"
OUTPUT_FILE = "data/processed/final_avalanche_dataset.parquet"


def grouped_rolling(
    df: pd.DataFrame,
    column: str,
    days: int,
    agg: str,
    min_periods: int = 1,
) -> pd.Series:
    """
    Calendar-day rolling statistic calculated independently
    for each mountain.

    The current observation is included.
    No future observations are used.
    """

    result = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    for mountain, group in df.groupby(
        "mountain",
        sort=False,
    ):
        group = group.sort_values("date")

        indexed = group.set_index("date")[column]

        rolling = indexed.rolling(
            f"{days}D",
            min_periods=min_periods,
        )

        if agg == "sum":
            values = rolling.sum()

        elif agg == "mean":
            values = rolling.mean()

        elif agg == "max":
            values = rolling.max()

        else:
            raise ValueError(
                f"Unsupported rolling aggregation: {agg}"
            )

        result.loc[group.index] = values.to_numpy()

    return result


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(INPUT_FILE)

print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ============================================================
# STANDARDIZE AND VALIDATE
# ============================================================

df.columns = df.columns.str.lower()

required = [
    "date",
    "mountain",
    "temp",
    "snow_depth",
    "swe",
    "precip",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns:\n{df.columns.tolist()}"
    )

df["date"] = pd.to_datetime(df["date"])

# Sort is critical for all rolling features
df = (
    df
    .sort_values(["mountain", "date"])
    .reset_index(drop=True)
)


# ============================================================
# DATE INTEGRITY CHECK
# ============================================================

# Calculate the number of days since the previous observation
# within each mountain.
df["date_gap_days"] = (
    df.groupby("mountain")["date"]
      .diff()
      .dt.days
)

# Ignore the first observation for each mountain because it
# naturally has no previous observation.
unexpected_gaps = df[
    df["date_gap_days"].notna()
    & (df["date_gap_days"] != 1)
    & ~df["date_gap_days"].between(150, 160)
]

if not unexpected_gaps.empty:

    print("\nWARNING: DATE GAPS DETECTED")

    print(
        unexpected_gaps[
            ["mountain", "date", "date_gap_days"]
        ]
        .head(25)
        .to_string(index=False)
    )

    print(
        f"\nTotal unexpected gaps: "
        f"{len(unexpected_gaps):,}"
    )

    raise ValueError(
        f"Found {len(unexpected_gaps):,} unexpected date gaps."
    )

else:
    print("✓ All mountain time series are daily with no gaps.")


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicates = df.duplicated(
    subset=["mountain", "date"],
    keep=False,
)

n_duplicates = duplicates.sum()

if n_duplicates > 0:

    duplicate_rows = df.loc[
        duplicates,
        ["mountain", "date"]
    ].sort_values(
        ["mountain", "date"]
    )

    raise ValueError(
        "Duplicate (mountain, date) observations detected.\n"
        "Feature engineering cannot safely continue.\n\n"
        f"{duplicate_rows.head(20).to_string(index=False)}"
    )

else:
    print("✓ No duplicate mountain/date combinations.")


# ============================================================
# STORM HISTORY FEATURES
# ============================================================

# Maximum precipitation event over recent windows
for window in [7, 30]:

    col = f"max_storm_{window}d"

    df[col] = grouped_rolling(
        df,
        "precip",
        window,
        "max",
    )


# ============================================================
# PRECIPITATION ACCUMULATION
# ============================================================

for window in [1, 2, 3]:

    col = f"precip_{window * 24}h"

    df[col] = grouped_rolling(
        df,
        "precip",
        window,
        "sum",
    )


# Maximum recent loading events
df["max_24h_precip"] = df["precip_24h"]

df["max_48h_precip"] = grouped_rolling(
    df,
    "precip_48h",
    30,
    "max",
)

df["max_72h_precip"] = grouped_rolling(
    df,
    "precip_72h",
    30,
    "max",
)


# ============================================================
# TEMPERATURE / WEAK-LAYER FEATURES
# ============================================================

print("Creating temperature / weak-layer features...")

df["temp_gradient"] = (
    -df["temp"]
    / (df["snow_depth"] + 1.0)
)


# ============================================================
# DAYS SINCE MAJOR LOADING
# ============================================================

def days_since_loading(series, threshold):

    result = []
    count = 999

    for value in series:

        if pd.notna(value) and value >= threshold:
            count = 0
        else:
            count += 1

        result.append(count)

    return pd.Series(
        result,
        index=series.index
    )


df["days_since_major_loading"] = (
    df.groupby("mountain")["precip"]
    .transform(
        lambda x: days_since_loading(x, 25)
    )
)


# ============================================================
# COLD SPELL DURATION
# ============================================================

def cold_spell_duration(series):

    result = []
    count = 0

    for value in series:

        if pd.notna(value) and value < -10:
            count += 1
        else:
            count = 0

        result.append(count)

    return pd.Series(
        result,
        index=series.index,
    )


df["cold_spell_duration"] = (
    df.groupby("mountain")["temp"]
    .transform(cold_spell_duration)
)


# ============================================================
# SNOW DEPTH AND SWE CHANGES
# ============================================================

df["snow_depth_change"] = (
    df.groupby("mountain")["snow_depth"]
    .diff()
)

df["swe_change"] = (
    df.groupby("mountain")["swe"]
    .diff()
)


# ============================================================
# LOW-DENSITY SNOWFALL PROXY
# ============================================================

snow_depth_change = (
    df.groupby("mountain")["snow_depth"]
    .diff()
)

swe_change = (
    df.groupby("mountain")["swe"]
    .diff()
)

df["snow_depth_change"] = snow_depth_change
df["swe_change"] = swe_change

# Only evaluate low-density snow when there is actual new snow.
new_snow_mask = (
    (snow_depth_change > 0) &
    (swe_change > 0)
)

density_proxy = (
    snow_depth_change /
    swe_change.replace(0, np.nan)
)

df["low_density_snow_day"] = (
    new_snow_mask &
    (density_proxy > 10)
).astype("int8")


df["low_density_snow_days_30d"] = grouped_rolling(
    df,
    "low_density_snow_day",
    30,
    "sum",
)


# ============================================================
# ADDITIONAL PRECIPITATION FEATURES
# ============================================================

df["precip_change_1d"] = (
    df.groupby("mountain")["precip"]
    .diff()
)

df["precip_7d_mean"] = grouped_rolling(
    df,
    "precip",
    7,
    "mean",
)

df["precip_14d_sum_new"] = grouped_rolling(
    df,
    "precip",
    14,
    "sum",
)


# Precipitation intensification
df["precip_3d_mean"] = grouped_rolling(
    df,
    "precip",
    3,
    "mean",
)

df["precip_intensification"] = (
    df["precip_3d_mean"]
    / (
        df["precip_14d_sum_new"] / 14.0
        + 0.01
    )
)

# ============================================================
# Short-term lag features
# ============================================================

lag_variables = [
    "temp",
    "precip",
    "wind_speed",
    "snowfall",
    "snow_depth",
    "swe",
]

for variable in lag_variables:

    if variable not in df.columns:
        continue

    for lag in [1, 2, 3]:

        df[f"{variable}_lag{lag}"] = (
            df.groupby("mountain")[variable]
            .shift(lag)
        )

# ============================================================
# Wind-loading change
# ============================================================

wind_aspects = [
    "n", "ne", "e", "se",
    "s", "sw", "w", "nw",
]

for aspect in wind_aspects:

    column = f"wind_load_{aspect}"

    if column not in df.columns:
        continue

    df[f"{column}_change_1d"] = (
        df.groupby("mountain")[column]
        .diff(1)
    )

    df[f"{column}_change_3d"] = (
        df.groupby("mountain")[column]
        .diff(3)
    )

# ============================================================
# Avalanche-mechanism interaction features
# ============================================================

df["wind_snow_interaction"] = (
    df["wind_speed"] *
    df["snowfall_3d"]
)

df["wind_loading_snow_interaction"] = (
    df["total_wind_loading"] *
    df["snowfall_3d"]
)

df["warming_snow_interaction"] = (
    df["temp_change_3d"] *
    df["snow_depth"]
)

df["rain_snow_interaction"] = (
    df["precip_24h"] *
    df["snow_depth"]
)

df["recent_loading_temperature"] = (
    df["precip_72h"] *
    df["temp_mean_3d"]
)

# ============================================================
# Rapid warming / wet-loose features
# ============================================================

df["temp_change_1d"] = (
    df.groupby("mountain")["temp"]
    .diff(1)
)

df["temp_change_2d"] = (
    df.groupby("mountain")["temp"]
    .diff(2)
)

df["temp_change_3d"] = (
    df.groupby("mountain")["temp"]
    .diff(3)
)

df["temp_change_7d"] = (
    df.groupby("mountain")["temp"]
    .diff(7)
)

df["rapid_warming_event"] = (
    (df["temp_change_1d"] >= 5)
).astype("int8")

df["strong_warming_event"] = (
    (df["temp_change_3d"] >= 8)
).astype("int8")

# ============================================================
# Snowfall intensity
# ============================================================

df["snowfall_intensity_3d"] = (
    df["snowfall_3d"] / 3.0
)

df["snowfall_intensity_7d"] = (
    df["snowfall_7d"] / 7.0
)

df["snowfall_acceleration"] = (
    df["snowfall_intensity_3d"] -
    (df["snowfall_14d"] / 14.0)
)

# ============================================================
# STORM RECENCY INDICATORS
# ============================================================

def days_since_threshold(series, threshold):

    """
    Number of days since the most recent day where
    precipitation was >= threshold.

    Example for threshold=10:

        precip   result
        0        NaN/initial
        0        ...
        12       0
        0        1
        0        2
        15       0

    The result is calculated independently for each mountain.
    """

    result = []
    days_since = np.nan

    for value in series:

        if pd.notna(value) and value >= threshold:
            days_since = 0.0

        elif pd.notna(days_since):
            days_since += 1.0

        result.append(days_since)

    return pd.Series(
        result,
        index=series.index,
        dtype=float,
    )


for threshold in [10, 25, 50]:

    col = f"days_since_{threshold}mm_new"

    df[col] = (
        df.groupby("mountain")["precip"]
        .transform(
            lambda x: days_since_threshold(
                x,
                threshold,
            )
        )
    )


# ============================================================
# CHECK FOR NON-SCALAR / OBJECT COLUMNS
# ============================================================

print("\nNon-scalar/object columns:")

for col in df.columns:

    if df[col].dtype == "object":

        sample = (
            df[col]
            .dropna()
            .head(1)
        )

        if len(sample):

            print(
                col,
                type(sample.iloc[0]),
                repr(sample.iloc[0])[:150],
            )


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

# Replace +/- infinity with NaN only in numeric columns
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


# ============================================================
# SUMMARY OF NEW FEATURES
# ============================================================

new_features = [
    "max_storm_7d",
    "max_storm_30d",
    "precip_24h",
    "precip_48h",
    "precip_72h",
    "max_24h_precip",
    "max_48h_precip",
    "max_72h_precip",
    "temp_gradient",
    "days_since_major_loading",
    "cold_spell_duration",
    "snow_depth_change",
    "swe_change",
    "low_density_snow_day",
    "low_density_snow_days_30d",
    "precip_change_1d",
    "precip_7d_mean",
    "precip_14d_sum_new",
    "precip_3d_mean",
    "precip_intensification",
    "days_since_10mm_new",
    "days_since_25mm_new",
    "days_since_50mm_new",
]


# ============================================================
# FINAL DATE GAP CHECK
# ============================================================

# date_gap_days is still present here, so this no longer
# produces KeyError.
unexpected_gaps = df[
    df["date_gap_days"].notna()
    & (df["date_gap_days"] != 1)
    & ~df["date_gap_days"].between(150, 160)
]

if not unexpected_gaps.empty:

    raise ValueError(
        f"Found {len(unexpected_gaps):,} unexpected date gaps."
    )

print("✓ No unexpected date gaps detected.")


# ============================================================
# NEW FEATURES
# ============================================================

print("\n" + "=" * 70)
print("NEW FEATURES")
print("=" * 70)

for feature in new_features:

    print(
        f"{feature:35s} "
        f"missing={df[feature].isna().sum():,}"
    )


# ============================================================
# REMOVE INTERNAL VALIDATION COLUMN
# ============================================================

# date_gap_days has served its purpose. It is not intended
# to be a model feature, so remove it before saving.
df.drop(
    columns=["date_gap_days"],
    inplace=True,
)


# ============================================================
# FINAL DATASET
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns):,}")

print("\nMountain count:")
print(df["mountain"].nunique())

print("\nDate range:")
print(df["date"].min())
print(df["date"].max())


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True,
)

df.to_parquet(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print("Saved final ML dataset:")
print(OUTPUT_FILE)

print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ============================================================
# VERIFY
# ============================================================

check = pd.read_parquet(
    OUTPUT_FILE
)

print("\nVerification:")
print("✓ File successfully reloaded")
print(f"✓ Rows:    {len(check):,}")
print(f"✓ Columns: {len(check.columns):,}")

missing_new = [
    f
    for f in new_features
    if f not in check.columns
]

if missing_new:

    raise RuntimeError(
        f"Feature verification failed: "
        f"{missing_new}"
    )

print("✓ All new features present")

# ============================================================
# TARGET DIAGNOSTICS
# ============================================================

print("\n" + "=" * 70)
print("AVALANCHE TARGET DISTRIBUTIONS")
print("=" * 70)

for problem in [
    "wind_slab",
    "storm_slab",
    "wet_loose",
    "persistent_slab",
]:

    column = f"{problem}_present"

    if column not in df.columns:
        continue

    positives = int(df[column].sum())
    negatives = int((df[column] == 0).sum())
    total = positives + negatives

    print(
        f"\n{problem}"
    )
    print(
        f"  Positive: {positives:,}"
    )
    print(
        f"  Negative: {negatives:,}"
    )
    print(
        f"  Positive rate: {positives / total:.3%}"
    )
    print(
        f"  Negative/positive ratio: "
        f"{negatives / max(positives, 1):.2f}"
    )