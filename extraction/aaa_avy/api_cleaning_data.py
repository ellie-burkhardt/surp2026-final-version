import os
import pandas as pd
import re

# Get directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)

# INPUT_FILE and OUTPUT_FILE must be changed for every forecast location 
INPUT_FILE = os.path.join(script_dir, "nwac_ranier_api.csv")
OUTPUT_FILE = os.path.join(script_dir, "nwac_ranier_api_clean.csv")

records = []

with open(INPUT_FILE, "r") as f:
    for line in f:
        line = line.strip()

        match = re.search(
            r"New forecast:\s*(\d+)\s*\((.*?)\)",
            line
        )

        if match:
            records.append({
                "api_id": int(match.group(1)),
                "published_time": match.group(2)
            })

df = pd.DataFrame(records)

df["published_time"] = pd.to_datetime(df["published_time"])
df = df.sort_values("published_time").reset_index(drop=True)

df["forecast_num"] = range(1, len(df)+1)
df["date"] = df["published_time"].dt.date

df = df[["forecast_num", "date", "api_id", "published_time"]]

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(df)} forecasts to {OUTPUT_FILE}")