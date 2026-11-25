import numpy as np 
import pandas as pd

df = pd.read_parquet("data/interim/weather_standardized.parquet")

# Snowfall
df["snowfall"] = (
    df.groupby("mountain")["snow_depth"]
      .diff()
      .clip(lower=0)
)

# Rolling Windows
windows = [1,3,5,7,14,30,60]

variables = [
    "precip",
    "snowfall",
    "swe",
    "snow_depth"
]

for var in variables:

    for w in windows:

        df[f"{var}_{w}d"] = (
            df.groupby("mountain")[var]
              .transform(lambda x: x.rolling(w,min_periods=1).sum())
        )

# Temp Changes
for lag in [1,3,7,14]:

    df[f"temp_change_{lag}d"] = (
        df.groupby("mountain")["temp"]
        .diff(lag)
    )

# Rolling Temp Means
for w in [3,7,14]:

    df[f"temp_mean_{w}d"] = (
        df.groupby("mountain")["temp"]
        .transform(lambda x:x.rolling(w).mean())
    )

# Strom History
thresholds = [10,25,50]

for threshold in thresholds:

    days_since = []

    for m, group in df.groupby("mountain"):

        count = 999

        vals = []

        for precip in group["precip"]:

            if precip >= threshold:
                count = 0
            else:
                count += 1

            vals.append(count)

        days_since.extend(vals)

    df[f"days_since_{threshold}mm"] = days_since

# Seasonal Features
df["day_of_year"] = df.date.dt.dayofyear

df["day_sin"] = np.sin(
    2*np.pi*df.day_of_year/365
)

df["day_cos"] = np.cos(
    2*np.pi*df.day_of_year/365
)

# Save
df.to_parquet("data/interim/weather_features.parquet")
