import pandas as pd

weather = pd.read_parquet("data/interim/weather_features_wind.parquet")
targets = pd.read_parquet("data/interim/avalanche_targets.parquet")

weather["date"] = pd.to_datetime(weather["date"]).dt.date
targets["date"] = pd.to_datetime(targets["date"]).dt.date

master = weather.merge(
    targets,
    on=["date", "mountain"],
    how="left"
)

mapping = {
    "unlikely": 0,
    "possible": 1,
    "likely": 2,
    "": None,
    " ": None
}

master["wind_slab_likelihood"] = master["wind_slab_likelihood"].replace(mapping)

# convert everything safely to numeric
master["wind_slab_likelihood"] = pd.to_numeric(
    master["wind_slab_likelihood"],
    errors="coerce"
)

# Fill missing numeric values with 0
numeric_cols = master.select_dtypes(include="number").columns
master[numeric_cols] = master[numeric_cols].fillna(0)

# Fill missing string/object values with empty strings
string_cols = master.select_dtypes(include=["object", "string"]).columns
master[string_cols] = master[string_cols].fillna("")

# force all object columns to string unless already numeric
for col in master.columns:
    if master[col].dtype == "object":
        master[col] = master[col].astype("string")

# safety: replace empty strings
master = master.replace(["", " "], None)

# final fallback fill only numeric columns
num_cols = master.select_dtypes(include="number").columns
master[num_cols] = master[num_cols].fillna(0)

master.to_parquet("data/interim/master_dataset.parquet")

print(master.shape)
print(master.head())