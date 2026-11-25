import csv
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

START_ID = 90000
END_ID = 190000

API_URL = "https://api.avalanche.org/v2/public/product/{}"

MAX_WORKERS = 50
# OUTPUT_CSV name must be changed for every forecast location 
OUTPUT_CSV = "msac_unique_forecasts.csv"


session = requests.Session()


def fetch_product(product_id):
    try:
        r = session.get(
            API_URL.format(product_id),
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        if r.status_code != 200:
            return None

        data = r.json()

        if not isinstance(data, dict):
            return None

        if "id" not in data:
            return None

        return data

    except Exception:
        return None


def make_fingerprint(product):

    text = "||".join([
        str(product.get("bottom_line", "")),
        str(product.get("hazard_discussion", "")),
        str(product.get("travel_advice", "")),
    ])

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def main():

    seen = set()
    unique_rows = []

    total = END_ID - START_ID + 1
    processed = 0

    with ThreadPoolExecutor(MAX_WORKERS) as executor:

        futures = {
            executor.submit(fetch_product, pid): pid
            for pid in range(START_ID, END_ID + 1)
        }

        for future in as_completed(futures):

            processed += 1

            if processed % 1000 == 0:
                print(f"{processed:,}/{total:,}")

            product = future.result()

            if not product:
                continue

            fp = make_fingerprint(product)

            if fp in seen:
                continue

            seen.add(fp)

            unique_rows.append({
                "product_id": product.get("id"),
                "published_time": product.get("published_time"),
                "expires_time": product.get("expires_time"),
                "author": product.get("author"),
                "product_type": product.get("product_type"),
                "bottom_line": product.get("bottom_line"),
                "hazard_discussion": product.get("hazard_discussion"),
            })

    unique_rows.sort(key=lambda r: r["product_id"])

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "product_id",
                "published_time",
                "expires_time",
                "author",
                "product_type",
                "bottom_line",
                "hazard_discussion",
            ]
        )

        writer.writeheader()
        writer.writerows(unique_rows)

    print()
    print(f"Unique forecasts: {len(unique_rows):,}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
