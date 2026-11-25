import pandas as pd

df = pd.read_parquet("data/interim/master_dataset.parquet")

# Likelihood
likelihood_map = {
    "Unlikely":0,
    "Possible":1,
    "Likely":2,
    "Very Likely":3,
    "Almost Certain":4
}

# Size
size_map = {
    "D1":1,
    "D1.5":2,
    "D2":3,
    "D2.5":4,
    "D3":5,
    "D4":6
}

problems = [
    "storm_slab",
    "wind_slab",
    "persistent_slab",
    "wet_loose",
    "wet_slab",
    "cornice"
]

for p in problems:

    if f"{p}_likelihood" in df.columns:
        df[f"{p}_likelihood"] = (
            df[f"{p}_likelihood"]
            .replace(likelihood_map)
        )

    if f"{p}_size" in df.columns:
        df[f"{p}_size"] = (
            df[f"{p}_size"]
            .replace(size_map)
        )

df.to_parquet("data/interim/master_dataset.parquet")