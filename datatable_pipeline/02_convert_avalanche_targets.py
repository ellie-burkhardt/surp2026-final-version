import pandas as pd

problem_names = {
    "Storm Slab": "storm_slab",
    "Wind Slab": "wind_slab",
    "Wet Loose": "wet_loose",
    "Persistent Slab": "persistent_slab",
    "Deep Persistent Slab": "persistent_slab",
}

def convert_problem_file(df, mountain):

    df["date"] = pd.to_datetime(df["published_time"]).dt.date

    rows = []

    for date, group in df.groupby("date"):

        row = {
            "date": date,
            "mountain": mountain
        }

        # initialize all avalanche problem targets
        for problem in problem_names.values():
            row[f"{problem}_present"] = 0
            row[f"{problem}_likelihood"] = ""
            row[f"{problem}_size"] = ""
            row[f"{problem}_aspects"] = ""
            row[f"{problem}_elevations"] = ""

        for _, g in group.iterrows():

            p = problem_names.get(g["type"])

            if p is None:
                continue

            # mark avalanche problem as present
            row[f"{p}_present"] = 1

            # store attributes
            row[f"{p}_likelihood"] = g.get("likelihood", "")
            row[f"{p}_size"] = g.get("size", "")
            row[f"{p}_aspects"] = g.get("aspects", "")
            row[f"{p}_elevations"] = g.get("elevations", "")

        rows.append(row)

    return pd.DataFrame(rows)


hood = pd.read_csv(
    "data/raw/avalanche/avalanche_problems/mount_hood_avalanche_problems.csv"
)

rainier = pd.read_csv(
    "data/raw/avalanche/avalanche_problems/mount_rainier_avalanche_problems.csv"
)

shasta = pd.read_csv(
    "data/raw/avalanche/avalanche_problems/mount_shasta_avalanche_problems.csv"
)


targets = pd.concat([
    convert_problem_file(hood, "hood"),
    convert_problem_file(rainier, "rainier"),
    convert_problem_file(shasta, "shasta")
], ignore_index=True)


# clean types
for col in targets.columns:

    if col.endswith("_present"):
        targets[col] = targets[col].fillna(0).astype(int)

    elif col in ["date", "mountain"]:
        continue

    else:
        targets[col] = targets[col].fillna("").astype(str)


targets.to_parquet(
    "data/interim/avalanche_targets.parquet"
)