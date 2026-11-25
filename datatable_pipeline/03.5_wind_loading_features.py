import numpy as np
import pandas as pd

df = pd.read_parquet("data/interim/weather_features.parquet")

df["date"] = pd.to_datetime(df["date"])

aspect_angles = {
    "N":0,
    "NE":45,
    "E":90,
    "SE":135,
    "S":180,
    "SW":225,
    "W":270,
    "NW":315
}

def compute_aspect_loading(
        wind_speed,
        wind_dir,
        precip,
        aspect_angle):

    # loading occurs on lee side
    lee_angle = (aspect_angle + 180) % 360

    angle_diff = np.deg2rad(wind_dir - lee_angle)

    projection = np.cos(angle_diff)

    # only positive deposition
    projection = np.maximum(projection,0)

    loading = wind_speed * precip * projection

    return loading

for aspect, angle in aspect_angles.items():

    df[f"wind_load_{aspect}"] = compute_aspect_loading(
        df["wind_speed"],
        df["wind_dir"],
        df["precip"],
        angle
    )

windows = {
    "24h":1,
    "72h":3,
    "7d":7
}

for aspect in aspect_angles.keys():

    for label, window in windows.items():

        df[f"wind_load_{aspect}_{label}"] = (
            df.groupby("mountain")[f"wind_load_{aspect}"]
            .transform(
                lambda x:
                x.rolling(
                    window,
                    min_periods=1
                ).sum()
            )
        )

for aspect in aspect_angles.keys():

    df[f"wind_load_{aspect}_max7d"] = (
        df.groupby("mountain")[f"wind_load_{aspect}"]
        .transform(
            lambda x:
            x.rolling(
                7,
                min_periods=1
            ).max()
        )
    )

df["total_wind_loading"] = (
    df["wind_speed"] *
    df["precip"]
)

df["total_wind_loading_3d"] = (
    df.groupby("mountain")["total_wind_loading"]
    .transform(
        lambda x:
        x.rolling(3,min_periods=1).sum()
    )
)

df["total_wind_loading_7d"] = (
    df.groupby("mountain")["total_wind_loading"]
    .transform(
        lambda x:
        x.rolling(7,min_periods=1).sum()
    )
)

df.to_parquet(
    "data/interim/weather_features_wind.parquet",
    index=False
)

print(df.shape)
print("Wind features added.")