import requests
import cdsapi
import os
import zipfile
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from io import StringIO

def get_snotel(start_date, end_date):

    # Mount Hood
    #STATION = "651:OR:SNTL"
    # Mount Ranier
    #STATION = "679:WA:SNTL"
    # Gray Butte (aka Mount Shasta)
    STATION = "487:CA:SNTL" 

    ELEMENTS = ["WTEQ", "SNWD", "PREC"]

    cache_file = f"ranier_snotel_{start_date[:4]}_{end_date[:4]}.csv"

    # Checks whether the file exists
    if os.path.exists(cache_file):

        print(f"[CACHE HIT] {cache_file}")
        return pd.read_csv(cache_file)
    
    url = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"
    
    params = {
        "stationTriplets": STATION,
        "elements": ",".join(ELEMENTS),
        "duration": "DAILY",
        "beginDate": start_date,
        "endDate": end_date,
    }

    # Maximum weight time is 60s
    r = requests.get(url, params = params, timeout = 60)
    r.raise_for_status()
    data = r.json()

    dfs = [] 

    for station in data: 
        for series in station["data"]:

            element = series["stationElement"]["elementCode"]
            values = pd.DataFrame(series["values"])

            if values.empty: 
                continue 

            values["date"] = pd.to_datetime(values["date"])
            values = values.rename(columns = {"value": element})

            dfs.append(values[["date", element]])

    df = dfs[0]

    for d in dfs[1:]: 
        df = df.merge(d, on = "date", how = "outer")
    
    df = df.sort_values("date")

    df.to_csv(cache_file, index = False)

    return df

def get_diff_snotel(start_date, end_date):

    STATION = "SDF"  # Sand Flat (Mount Shasta)

    SENSOR_MAP = {
        "3": "SWE",
        "18": "SnowDepth",
        "30": "Temperature"
    }

    cache_file = f"shasta_cdec_{start_date[:4]}_{end_date[:4]}.csv"

    if os.path.exists(cache_file):
        print(f"[CACHE HIT] {cache_file}")
        return pd.read_csv(cache_file, parse_dates=["date"])

    url = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"

    params = {
        "Stations": STATION,
        "SensorNums": ",".join(SENSOR_MAP.keys()),
        "dur_code": "D",
        "Start": start_date,
        "End": end_date,
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    # Debug: inspect response if needed
    print("Response content type:", r.headers.get("Content-Type"))

    # Read CSV response
    raw = pd.read_csv(StringIO(r.text))

    if raw.empty:
        print("No data returned from CDEC.")
        return pd.DataFrame()

    print("Columns returned:")
    print(raw.columns.tolist())

    # Standardize column names
    raw.columns = [c.strip() for c in raw.columns]

    # Typical CDEC CSV columns include:
    # STATION_ID, SENSOR_NUMBER, SENSOR_TYPE, DATE TIME, VALUE, DATA_FLAG

    sensor_col = None
    date_col = None
    value_col = None

    for c in raw.columns:
        uc = c.upper()

        if "SENSOR" in uc and "NUMBER" in uc:
            sensor_col = c

        elif "DATE" in uc:
            date_col = c

        elif uc == "VALUE":
            value_col = c

    if sensor_col is None:
        raise ValueError(
            f"Could not find SENSOR_NUMBER column. Columns={raw.columns.tolist()}"
        )

    if date_col is None:
        raise ValueError(
            f"Could not find date column. Columns={raw.columns.tolist()}"
        )

    if value_col is None:
        raise ValueError(
            f"Could not find VALUE column. Columns={raw.columns.tolist()}"
        )

    raw[date_col] = pd.to_datetime(raw[date_col])

    # Create wide dataframe:
    # one row per date, one column per sensor
    df = raw.pivot_table(
        index=date_col,
        columns=sensor_col,
        values=value_col,
        aggfunc="first"
    )

    df = df.reset_index()

    # Rename columns
    rename_dict = {}
    for sensor_num, name in SENSOR_MAP.items():
        try:
            rename_dict[int(sensor_num)] = name
        except ValueError:
            rename_dict[sensor_num] = name

    df = df.rename(columns=rename_dict)

    df = df.rename(columns={date_col: "date"})
    df = df.sort_values("date")

    df.to_csv(cache_file, index=False)

    return df

def safe_open(file_path): 

    if not os.path.exists(file_path): 
        raise FileNotFoundError(file_path)
    
    # Extracts .nc file from .zip
    if file_path.endswith(".zip"): 

        extract_dir = file_path.replace(".zip", "_extracted")

        with zipfile.ZipFile(file_path, "r") as z: 
            z.extractall(extract_dir)

        nc_files = [f for f in os.listdir(extract_dir) if f.endswith(".nc")]

        if not nc_files:
            raise ValueError("No .nc inside zip")
        
        return xr.open_dataset(
            os.path.join(extract_dir, nc_files[0]),
            engine = "netcdf4"
        )
    
    # Opens a NetCDF file and return it as an xarray Dataset
    return xr.open_dataset(file_path, engine = "netcdf4")

def get_era5(lat, lon, station_elev_m, start_date, end_date): 

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    c = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key="e920a217-18ee-4c89-8c32-d59085065a6c"
    )

    # Mount Hood
    '''
    area = [
        45.414738,
        -121.7472905,
        45.314738,
        -121.6472905
    ]
    '''

    # Mount Ranier
    '''
    area = [
        46.976040,
        -121.894367,
        46.766752,
        -121.556632
    ]
    '''

    # Mount Shasta
    area = [
        41.468399,
        -122.299319,
        41.300585,
        -122.081384
    ]

    variables = [
        "2m_temperature",
        "total_precipitation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]

    months = pd.date_range(start, end, freq = "MS")

    era5_chunks = []

    for i in range(len(months)): 

        # Each request is one month
        chunk = months[i:i+1]

        years = sorted(list(set(chunk.year)))
        months_list = sorted(list(set(chunk.month)))

        cache_file = f"era5_{start.year}_{i}.zip"

        if os.path.exists(cache_file):
            print(f"[CACHE HIT] {cache_file}")

        else: 
            print(f"[DOWNLOAD] {cache_file}")

            c.retrieve(
                "reanalysis-era5-land",
                {
                    "variable": variables,
                    "year": [str(y) for y in years],
                    "month": [f"{m:02d}" for m in months_list],
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": area,
                    "format": "netcdf",
                },
                cache_file
            )

        ds = safe_open(cache_file)

        if isinstance(ds, xr.Dataset):
            era5_chunks.append(ds)
        
        else:
            print(f"[SKIP] bad dataset: {cache_file}")
    
    if len(era5_chunks) == 0: 
        raise RuntimeError("No valid ERA5 data loaded")
    
    ds = xr.combine_by_coords(era5_chunks, combine_attrs="override")

    ds = ds.sel(latitude=lat, longitude=lon, method="nearest")

    df = ds.to_dataframe().reset_index()

    # Changes "valid_time" to "time"
    if "valid_time" in df.columns:
        df = df.rename(columns={"valid_time": "time"})

    # Converts Kelvin to Celsius
    df["t2m_corr_c"] = df["t2m"] - 273.15
    # Converts meters to millimeters 
    df["precip_mm"] = df["tp"] * 1000

    # Extracts wind components
    u, v = df["u10"], df["v10"]
    # Calculate wind speed
    df["wind_speed_ms"] = np.sqrt(u**2 + v**2)
    # Calculate wind direction
    df["wind_dir_deg"] = (270 - np.degrees(np.arctan2(v, u))) % 360

    return df[
        [
            "time",
            "t2m_corr_c",
            "precip_mm",
            "wind_speed_ms",
            "wind_dir_deg"
        ]
    ]

def generate_winters(start_year, n=6):
    return [
        (f"{y}-10-01", f"{y+1}-04-30")
        for y in range(start_year, start_year + n)
    ]

def combine_snotel_era5(snotel_df, era5_df, output_file = "mount_shasta_xgboost_dataset.parquet"):

    snotel_df["date"] = pd.to_datetime(snotel_df["date"])
    era5_df["time"] = pd.to_datetime(era5_df["time"])

    era5_daily = (
        era5_df
        .set_index("time")
        .resample("D")
        .agg({
            "t2m_corr_c": "mean",
            "precip_mm": "sum",
            "wind_speed_ms": "mean",
            "wind_dir_deg": "mean",
        })
        .reset_index()
        .rename(columns={"time": "date"})
    )

    combined = pd.merge(
        snotel_df,
        era5_daily,
        on="date",
        how="inner"
    )

    combined["year"] = combined["date"].dt.year
    combined["month"] = combined["date"].dt.month
    combined["day"] = combined["date"].dt.day
    combined["day_of_year"] = combined["date"].dt.dayofyear

    combined["temp_7d_mean"] = (
        combined["t2m_corr_c"]
        .rolling(7, min_periods=1)
        .mean()
    )

    combined["precip_7d_sum"] = (
        combined["precip_mm"]
        .rolling(7, min_periods=1)
        .sum()
    )

    combined["wind_7d_mean"] = (
        combined["wind_speed_ms"]
        .rolling(7, min_periods=1)
        .mean()
    )

    combined.to_parquet(output_file, index=False)

    csv_file = output_file.replace(".parquet", ".csv")
    combined.to_csv(csv_file, index=False)

    print(f"Saved parquet: {output_file}")
    print(f"Saved csv: {csv_file}")
    print(f"Rows: {len(combined)}")
    print(f"Columns: {len(combined.columns)}")

    return combined

def run_pipeline():

    seasons = generate_winters(2020, n=6)

    all_data = []

    for start_date, end_date in seasons:

        print(f"\n=== {start_date} → {end_date} ===")

        snotel = get_diff_snotel(start_date, end_date)

        era5 = get_era5(
            lat= 41.40989,
            lon=-122.19483,
            station_elev_m=1080,
            start_date=start_date,
            end_date=end_date
        )

        df = combine_snotel_era5(snotel, era5)

        df["winter_start"] = start_date[:4]

        all_data.append(df)

    master = pd.concat(all_data, ignore_index=True)

    master.to_parquet("mount_shasta_10winter_FAST.parquet", index=False)

    print("\nDONE:", master.shape)

def rebuild_hood_parquet(
    snotel_dir="data/raw/weather/hood_snotel_csv",
    era5_dir="data/raw/weather/hood_era5_extracted",
    output_file="data/interim/mount_hood_10winter_FAST.parquet"
):

    snotel_dir = Path(snotel_dir)
    era5_dir = Path(era5_dir)
    output_file = Path(output_file)

    print("\nSNOTEL directory:")
    print(snotel_dir.resolve())

    print("\nERA5 directory:")
    print(era5_dir.resolve())

    # Find files
    all_snotel_files = [
        f for f in snotel_dir.rglob("*")
        if f.is_file()
    ]

    all_era5_files = [
        f for f in era5_dir.rglob("*")
        if f.is_file()
    ]

    print("\nALL SNOTEL FILES:")
    for f in all_snotel_files:
        print("  ", f)

    print("\nALL ERA5 FILES:")
    for f in all_era5_files:
        print("  ", f)

    # Only select the files we need
    snotel_files = [
        f for f in all_snotel_files
        if f.suffix.lower() == ".csv"
    ]

    era5_files = [
        f for f in all_era5_files
        if f.suffix.lower() == ".nc"
    ]

    print("\nCSV files:", len(snotel_files))
    print("NC files:", len(era5_files))

    if not snotel_files:
        raise FileNotFoundError(
            f"No CSV files found in {snotel_dir.resolve()}"
        )

    if not era5_files:
        raise FileNotFoundError(
            f"No NetCDF files found in {era5_dir.resolve()}"
        )

    # Load ALL ERA5 files
    era5_chunks = []

    for era5_file in sorted(era5_files):

        print(f"\nLoading ERA5: {era5_file.name}")

        ds = xr.open_dataset(
            era5_file,
            engine="netcdf4"
        )

        ds = ds.sel(
            latitude=45.364738,
            longitude=-121.6972905,
            method="nearest"
        )

        era5_chunks.append(ds)

    ds = xr.combine_by_coords(
        era5_chunks,
        combine_attrs="override"
    )

    era5 = ds.to_dataframe().reset_index()

    if "valid_time" in era5.columns:
        era5 = era5.rename(
            columns={"valid_time": "time"}
        )

    era5["t2m_corr_c"] = era5["t2m"] - 273.15
    era5["precip_mm"] = era5["tp"] * 1000

    u = era5["u10"]
    v = era5["v10"]

    era5["wind_speed_ms"] = np.sqrt(u**2 + v**2)

    era5["wind_dir_deg"] = (
        270 - np.degrees(np.arctan2(v, u))
    ) % 360

    era5 = era5[
        [
            "time",
            "t2m_corr_c",
            "precip_mm",
            "wind_speed_ms",
            "wind_dir_deg"
        ]
    ]

    ds.close()

    # Process each SNOTEL file
    all_data = []

    for snotel_file in sorted(snotel_files):

        print(f"\n=== {snotel_file.name} ===")

        snotel = pd.read_csv(
            snotel_file,
            parse_dates=["date"]
        )

        if snotel.empty:
            print("[SKIP] Empty SNOTEL file")
            continue

        start_year = snotel["date"].dt.year.min()

        print(
            f"Winter: {start_year}-{start_year + 1}"
        )

        winter_era5 = era5[
            (era5["time"] >= snotel["date"].min()) &
            (era5["time"] <= snotel["date"].max())
        ].copy()

        if winter_era5.empty:
            print("[SKIP] No ERA5 data for this winter")
            continue

        # Re-run the original combine function
        df = combine_snotel_era5(
            snotel,
            winter_era5
        )

        df["winter_start"] = start_year

        all_data.append(df)

        print("Added:", len(df), "rows")

    # Rebuild master parquet
    if not all_data:
        raise RuntimeError(
            "No datasets were successfully combined."
        )

    master = pd.concat(
        all_data,
        ignore_index=True
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    master.to_parquet(
        output_file,
        index=False
    )

    print("\nDONE:", master.shape)
    print("Saved:", output_file)

    return master