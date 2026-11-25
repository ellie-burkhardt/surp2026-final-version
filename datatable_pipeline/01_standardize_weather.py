import pandas as pd
import numpy as np

def clean_weather(df):

    for col in ["snow_depth", "swe", "temp", "precip", "wind_speed", "wind_dir"]:
        if col in df.columns:
            df[col] = df[col].replace(["---", "NaN", "null", ""], np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def remove_old_winter_data(
    input_file="data/interim/mount_hood_10winter_FAST.parquet",
    output_file="data/interim/mount_hood_10winter_FAST.parquet"
):
    # Load the parquet file
    df = pd.read_parquet(input_file)

    # Convert winter_start from string to numeric
    df["winter_start"] = pd.to_numeric(df["winter_start"], errors="coerce")

    # Remove entries where winter_start is before 2020
    df = df[df["winter_start"] >= 2020].copy()

    # Save the cleaned data
    df.to_parquet(output_file, index=False)

    return df

# Mount Hood
remove_old_winter_data()
hood = pd.read_parquet("data/interim/mount_hood_10winter_FAST.parquet")

hood = hood.rename(columns={
    "SNWD":"snow_depth",
    "WTEQ":"swe",
    "t2m_corr_c":"temp",
    "precip_mm":"precip",
    "wind_speed_ms":"wind_speed",
    "wind_dir_deg":"wind_dir"
})

hood["mountain"] = "hood"

# Mount Ranier
rainier = pd.read_parquet("data/interim/mount_rainier_10winter_FAST.parquet")

rainier = rainier.rename(columns={
    "SNWD":"snow_depth",
    "WTEQ":"swe",
    "t2m_corr_c":"temp",
    "precip_mm":"precip",
    "wind_speed_ms":"wind_speed",
    "wind_dir_deg":"wind_dir"
})

rainier["mountain"] = "rainier"

# Mount Shasta
shasta = pd.read_parquet("data/interim/mount_shasta_10winter_FAST.parquet")

shasta = shasta.rename(columns={
    "SnowDepth":"snow_depth",
    "SWE":"swe",
    "Temperature":"temp",
    "precip_mm":"precip",
    "wind_speed_ms":"wind_speed",
    "wind_dir_deg":"wind_dir"
})

shasta["mountain"] = "shasta"

# Clean Files
hood = clean_weather(hood)
rainier = clean_weather(rainier)
shasta = clean_weather(shasta)

weather = pd.concat([hood, rainier, shasta])

weather["date"] = pd.to_datetime(weather["date"])

# Smash Together
weather.sort_values(["mountain", "date"], inplace=True)

weather["winter_start"] = pd.to_numeric(
    weather["winter_start"], errors="coerce"
).astype("Int64")

weather.to_parquet("data/interim/weather_standardized.parquet")

# Remove t2m_corr_c
df = pd.read_parquet("data/interim/weather_standardized.parquet")
print(df.columns.tolist())

df = df.drop(columns=["t2m_corr_c"])

df.to_parquet("data/interim/weather_standardized.parquet", index=False)