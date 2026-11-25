import os
import requests
import pandas as pd

API_KEY = "YOUR_API_KEY"
BASE_URL = "https://api.avalanche.org/v2/public/product/{}"


# Manually specify CSV path
csv_path = r"/Users/ellie/SURP_2026_2.0/data/raw/avalanche/api_codes/clean/msac_api_clean.csv"

# Read csv
df = pd.read_csv(csv_path)

total_rows = len(df)
records = []

for idx, row in enumerate(df.itertuples(index=False), start=1):

    api_id = row.api_id

    print(f"[{idx}/{total_rows}] Reading api_id: {api_id}")

    try:
        response = requests.get(
            BASE_URL.format(api_id),
            headers={"x-api-key": API_KEY},
            timeout=30
        )

        if response.status_code != 200:
            print(
                f"Failed {api_id} "
                f"(status {response.status_code})"
            )
            continue

        data = response.json()

        problems = data.get("forecast_avalanche_problems", [])

        print(
            f"    ✓ Success {api_id} "
            f"({len(problems)} avalanche problems)"
        )

        for i, p in enumerate(problems):

            locations = p.get("location", [])

            aspects = []
            elevations = []

            for loc in locations:
                pieces = loc.split()

                if len(pieces) >= 2:
                    aspects.append(pieces[0])
                    elevations.append(pieces[1])

            records.append({
                "api_id": api_id,
                "published_time": row.published_time,
                "problem_num": i + 1,
                "type": p.get("name"),
                "likelihood": p.get("likelihood"),
                "size": ",".join(p.get("size", [])),
                "aspects": ",".join(sorted(set(aspects))),
                "elevations": ",".join(sorted(set(elevations)))
            })

    except Exception as e:
        print(f"Error processing {api_id}: {e}")

output_df = pd.DataFrame(records)

output_file = r"/Users/ellie/SURP_2026_2.0/data/raw/avalanche/avalanche_problems/mount_shasta_avalanche_problems.csv"

output_df.to_csv(output_file, index=False)

print("\n====================")
print("Processing complete")
print(f"Rows processed: {total_rows}")
print(f"Records created: {len(output_df)}")
print(f"Output saved to: {output_file}")
print("====================")