"""
Single-problem avalanche inference + terrain mapping pipeline.

Data source:
    data/processed/final_avalanche_dataset.parquet

Workflow:
    mountain -> date -> ONE avalanche problem -> ONE model -> ONE map

The pipeline intentionally does not use WeatherProvider and does not
create a combined four-problem map.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
print(PROJECT_ROOT)

from src.mapping.inference.model_loader import ModelLoader


AVALANCHE_PROBLEMS = (
    "wind_slab",
    "storm_slab",
    "wet_loose",
    "persistent_slab",
)

PROBLEM_LABELS = {
    "wind_slab": "Wind Slab",
    "storm_slab": "Storm Slab",
    "wet_loose": "Wet Loose",
    "persistent_slab": "Persistent Slab",
}

MODEL_NAMES = {
    "wind_slab": "wind_slab_present_future",
    "storm_slab": "storm_slab_present_future",
    "wet_loose": "wet_loose_present_future",
    "persistent_slab": "persistent_slab_present_future",
}

# These are the terrain filters already used by the project's mapper.
SLOPE_RULES = {
    "wind_slab": (30.0, 60.0),
    "storm_slab": (30.0, 60.0),
    "wet_loose": (35.0, 80.0),
    "persistent_slab": (40.0, 80.0),
}

ELEVATION_RULES = {
    "wind_slab": (1800.0, None),
    "storm_slab": (1500.0, None),
    "wet_loose": (None, 2500.0),
    "persistent_slab": (None, None),
}

MAP_CMAPS = {
    "wind_slab": "Reds",
    "storm_slab": "Greens",
    "wet_loose": "Blues",
    "persistent_slab": "YlOrBr",
}


# ----------------------------------------------------------------------
# Causal feature engineering
# ----------------------------------------------------------------------

def grouped_rolling(df, column, days, agg, min_periods=1):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, group in df.groupby("mountain", sort=False):
        group = group.sort_values("date")
        s = group.set_index("date")[column]
        r = s.rolling(f"{days}D", min_periods=min_periods)
        if agg == "sum":
            values = r.sum()
        elif agg == "mean":
            values = r.mean()
        elif agg == "max":
            values = r.max()
        else:
            raise ValueError(f"Unsupported rolling aggregation: {agg}")
        out.loc[group.index] = values.to_numpy()
    return out


def days_since_threshold(series, threshold):
    out = []
    days = np.nan
    for value in series:
        if pd.notna(value) and value >= threshold:
            days = 0.0
        elif pd.notna(days):
            days += 1.0
        out.append(days)
    return pd.Series(out, index=series.index, dtype=float)


def cold_spell_duration(series):
    out = []
    count = 0
    for value in series:
        if pd.notna(value) and value < -10:
            count += 1
        else:
            count = 0
        out.append(count)
    return pd.Series(out, index=series.index, dtype=float)


def engineer_model_features(df):
    """Recreate the inference features used by the current pipeline."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Dataset contains invalid dates.")
    df = df.sort_values(["mountain", "date"]).reset_index(drop=True)

    numeric = [
        "temp", "snow_depth", "swe", "precip", "snowfall", "wind_speed", "wind_dir",
        "total_wind_loading", "total_wind_loading_3d", "total_wind_loading_7d",
        "wind_7d_mean", "temp_mean_3d", "temp_mean_7d", "temp_mean_14d", "temp_7d_mean",
        "snowfall_3d", "snowfall_7d", "snowfall_14d", "snow_depth_3d", "snow_depth_7d",
        "snow_depth_14d", "snow_depth_30d", "swe_3d", "swe_7d", "swe_14d",
    ]
    for col in numeric:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "precip" in df:
        df["prec"] = df["precip"]

    df["season"] = df["date"].dt.year + (df["date"].dt.month >= 10).astype(int)

    doy = df["date"].dt.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * doy / 365.25)

    if "wind_dir" in df:
        radians = np.deg2rad(df["wind_dir"])
        df["wind_dir_sin"] = np.sin(radians)
        df["wind_dir_cos"] = np.cos(radians)

    if "precip" in df:
        for w in (1, 2, 3):
            df[f"precip_{w * 24}h"] = grouped_rolling(df, "precip", w, "sum")
        df["max_storm_7d"] = grouped_rolling(df, "precip", 7, "max")
        df["max_24h_precip"] = df["precip_24h"]
        df["max_48h_precip"] = grouped_rolling(df, "precip_48h", 30, "max")
        df["max_72h_precip"] = grouped_rolling(df, "precip_72h", 30, "max")
        df["precip_change_1d"] = df.groupby("mountain")["precip"].diff(1)
        df["precip_7d_mean"] = grouped_rolling(df, "precip", 7, "mean")
        df["precip_14d_sum_new"] = grouped_rolling(df, "precip", 14, "sum")
        df["precip_3d_mean"] = grouped_rolling(df, "precip", 3, "mean")
        df["precip_intensification"] = df["precip_3d_mean"] / (df["precip_14d_sum_new"] / 14.0 + 0.01)
        df["days_since_major_loading"] = df.groupby("mountain")["precip"].transform(lambda x: days_since_threshold(x, 25))
        for threshold in (10, 25, 50):
            df[f"days_since_{threshold}mm_new"] = df.groupby("mountain")["precip"].transform(lambda x, t=threshold: days_since_threshold(x, t))

    if "temp" in df:
        for lag in (1, 2, 3, 7):
            df[f"temp_change_{lag}d"] = df.groupby("mountain")["temp"].diff(lag)
        if "snow_depth" in df:
            df["temp_gradient"] = -df["temp"] / (df["snow_depth"] + 1.0)
        df["cold_spell_duration"] = df.groupby("mountain")["temp"].transform(cold_spell_duration)
        df["rapid_warming_event"] = (df["temp_change_1d"] >= 5).astype("int8")
        df["strong_warming_event"] = (df["temp_change_3d"] >= 8).astype("int8")

    if "snow_depth" in df:
        df["snow_depth_change"] = df.groupby("mountain")["snow_depth"].diff()
    if "swe" in df:
        df["swe_change"] = df.groupby("mountain")["swe"].diff()

    if "snow_depth_change" in df and "swe_change" in df:
        density = df["snow_depth_change"] / df["swe_change"].replace(0, np.nan)
        df["low_density_snow_day"] = (
            (df["snow_depth_change"] > 0) & (df["swe_change"] > 0) & (density > 10)
        ).astype("int8")

    for variable in ("temp", "precip", "wind_speed", "snowfall", "snow_depth", "swe"):
        if variable not in df:
            continue
        for lag in (1, 2, 3):
            df[f"{variable}_lag{lag}"] = df.groupby("mountain")[variable].shift(lag)

    if "snowfall_3d" in df:
        df["snowfall_intensity_3d"] = df["snowfall_3d"] / 3.0
    if "snowfall_7d" in df:
        df["snowfall_intensity_7d"] = df["snowfall_7d"] / 7.0
    if "snowfall_intensity_3d" in df and "snowfall_14d" in df:
        df["snowfall_acceleration"] = df["snowfall_intensity_3d"] - df["snowfall_14d"] / 14.0

    if "wind_speed" in df and "snowfall_3d" in df:
        df["wind_snow_interaction"] = df["wind_speed"] * df["snowfall_3d"]
    if "total_wind_loading" in df and "snowfall_3d" in df:
        df["wind_loading_snow_interaction"] = df["total_wind_loading"] * df["snowfall_3d"]
    if "temp_change_3d" in df and "snow_depth" in df:
        df["warming_snow_interaction"] = df["temp_change_3d"] * df["snow_depth"]
    if "precip_24h" in df and "snow_depth" in df:
        df["rain_snow_interaction"] = df["precip_24h"] * df["snow_depth"]
    if "precip_72h" in df and "temp_mean_3d" in df:
        df["recent_loading_temperature"] = df["precip_72h"] * df["temp_mean_3d"]

    if "temp_mean_14d" in df and "snowfall_14d" in df:
        df["persistent_slab_setup"] = ((df["temp_mean_14d"] < -5) & (df["snowfall_14d"] > 10)).astype("int8")
    if "swe_7d" in df:
        df["heavy_snow_loading_event"] = (df["swe_7d"] > 25).astype("int8")

    # --- Added Domain Feature Calculations ---
    if "temp_change_1d" in df and "temp" in df:
        df["freeze_thaw_instability"] = ((df["temp_change_1d"] > 3) & (df["temp"] > 0)).astype("int8")
    if "precip_72h" in df and "wind_speed" in df:
        df["storm_loading_index"] = df["precip_72h"] * df["wind_speed"]
    if "temp_mean_7d" in df:
        df["warming_week_index"] = (df["temp_mean_7d"] > 2).astype("int8")
    if "wind_speed" in df and "snowfall_3d" in df:
        df["wind_loading_index"] = df["wind_speed"] * df["snowfall_3d"]
    if "wind_7d_mean" in df:
        df["wind_persistence_index"] = df["wind_7d_mean"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Return a defragmented copy to silence PerformanceWarning
    return df.copy()


class AvalancheForecastPipeline:
    """Load Parquet data, run one model, and map only that problem."""

    def __init__(
        self,
        dataset_file = PROJECT_ROOT / "data" / "processed" / "final_avalanche_dataset.parquet",
        model_directory="trained_models",
        dem_file=None,
        output_directory="research_outputs/maps",
    ):
        self.dataset_file = self._resolve(dataset_file)
        self.model_directory = self._resolve(model_directory)
        self.dem_file = self._resolve(dem_file) if dem_file else None
        self.output_directory = self._resolve(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.dataset = self._load_dataset()

        # Load artifacts once for compatibility with the existing model
        # directory structure. Only the selected model is ever predicted.
        artifacts = ModelLoader(str(self.model_directory)).load_all()
        self.models = artifacts["models"]
        self.metadata = artifacts["metadata"]

        print(f"✓ Loaded Parquet dataset: {len(self.dataset):,} rows")
        print(f"✓ Loaded model artifacts: {len(self.models)}")

    @staticmethod
    def _resolve(path):
        path = Path(path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _load_dataset(self):
        if not self.dataset_file.exists():
            raise FileNotFoundError(f"Parquet dataset not found:\n{self.dataset_file}")
        df = pd.read_parquet(self.dataset_file)
        df.columns = [str(c).strip().lower() for c in df.columns]
        required = {"date", "mountain"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise ValueError("Dataset contains invalid dates.")
        return df.sort_values(["mountain", "date"]).reset_index(drop=True)

    def available_mountains(self):
        return sorted(self.dataset["mountain"].dropna().astype(str).unique())

    def available_dates(self, mountain):
        return sorted(
            self.dataset.loc[self.dataset["mountain"].astype(str) == str(mountain), "date"]
            .dt.strftime("%Y-%m-%d").unique().tolist()
        )

    def available_problems(self):
        return list(AVALANCHE_PROBLEMS)

    def get_input_history(self, mountain, date):
        date = pd.Timestamp(date)
        mountain_mask = self.dataset["mountain"].astype(str) == str(mountain)
        history = self.dataset.loc[mountain_mask & (self.dataset["date"] <= date)].copy()
        if history.empty:
            raise ValueError(f"No data for {mountain} on or before {date:%Y-%m-%d}.")
        if not (history["date"] == date).any():
            raise ValueError(f"No Parquet observation exists for {mountain} on {date:%Y-%m-%d}.")
        return history

    def prepare_features(self, history, date, model_name):
        date = pd.Timestamp(date)
        engineered = engineer_model_features(history)
        row = engineered.loc[engineered["date"] == date].tail(1).copy()
        if row.empty:
            raise ValueError(f"Could not construct inference row for {date:%Y-%m-%d}.")

        row = pd.get_dummies(row, columns=["mountain"], drop_first=False)
        for col in row.select_dtypes(include=["bool"]).columns:
            row[col] = row[col].astype("int8")
        for col in row.columns:
            if row[col].dtype == "object":
                converted = pd.to_numeric(row[col], errors="coerce")
                if converted.notna().any():
                    row[col] = converted

        row = row.drop(columns=["date", "year", "month", "day", "day_of_year"], errors="ignore")
        model = self.models[model_name]
        expected = list(model.feature_names_)

        # Dynamically populate missing categorical dummy columns (e.g., mountain_rainier) or indices with 0
        for col in expected:
            if col not in row.columns:
                row[col] = 0.0

        row = row.reindex(columns=expected)
        if row.isna().any().any():
            bad = row.columns[row.isna().any()].tolist()
            raise ValueError(f"Inference row contains NaN values in: {bad}")
        return row

    # ------------------------------------------------------------------
    # Terrain mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _terrain_derivatives(elevation, transform):
        dx = abs(transform.a)
        dy = abs(transform.e)
        dz_dy, dz_dx = np.gradient(elevation.astype(float), dy, dx)
        slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
        aspect = (np.degrees(np.arctan2(-dz_dx, dz_dy)) + 360.0) % 360.0
        return slope, aspect

    @staticmethod
    def classify_aspect(aspect):
        classes = np.empty(aspect.shape, dtype=object)
        ranges = [
            ((aspect >= 337.5) | (aspect < 22.5), "N"),
            ((aspect >= 22.5) & (aspect < 67.5), "NE"),
            ((aspect >= 67.5) & (aspect < 112.5), "E"),
            ((aspect >= 112.5) & (aspect < 157.5), "SE"),
            ((aspect >= 157.5) & (aspect < 202.5), "S"),
            ((aspect >= 202.5) & (aspect < 247.5), "SW"),
            ((aspect >= 247.5) & (aspect < 292.5), "W"),
            ((aspect >= 292.5) & (aspect < 337.5), "NW"),
        ]
        for mask, label in ranges:
            classes[mask] = label
        return classes

    def build_problem_raster(self, problem, probability):
        if rasterio is None:
            raise ImportError("rasterio is required to generate a terrain map.")
        if self.dem_file is None:
            raise ValueError("A DEM GeoTIFF is required for mapping. Pass dem_file='...tif'.")
        if not self.dem_file.exists():
            raise FileNotFoundError(f"DEM not found: {self.dem_file}")

        with rasterio.open(self.dem_file) as src:
            elevation = src.read(1).astype(float)
            transform = src.transform
            nodata = src.nodata
            valid = np.isfinite(elevation)
            if nodata is not None:
                valid &= elevation != nodata

            slope, aspect = self._terrain_derivatives(elevation, transform)
            aspect_classes = self.classify_aspect(aspect)
            min_slope, max_slope = SLOPE_RULES[problem]
            mask = valid & (slope >= min_slope) & (slope <= max_slope)
            min_elev, max_elev = ELEVATION_RULES[problem]
            if min_elev is not None:
                mask &= elevation >= min_elev
            if max_elev is not None:
                mask &= elevation <= max_elev

            raster = np.zeros(elevation.shape, dtype=float)
            raster[mask] = probability
            return raster, elevation, slope, aspect, aspect_classes, mask, transform

    def save_problem_map(self, mountain, date, problem, probability, raster, elevation, slope, mask):
        date = pd.Timestamp(date)
        label = PROBLEM_LABELS[problem]
        outfile = self.output_directory / f"{mountain}_{date:%Y%m%d}_{problem}_map.png"

        fig, ax = plt.subplots(figsize=(11, 9))
        background = np.where(np.isfinite(elevation), elevation, np.nan)
        ax.imshow(background, cmap="gray", alpha=0.60)
        mapped = np.ma.masked_where(~mask, raster)
        im = ax.imshow(mapped, cmap=MAP_CMAPS[problem], vmin=0, vmax=1, alpha=0.75)
        try:
            ax.contour(np.where(mask, slope, np.nan), levels=[30], linewidths=0.5, alpha=0.35)
        except Exception:
            pass
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{label} probability")
        ax.set_title(f"{label} Probability Map\n{str(mountain).title()} — {date:%Y-%m-%d} — {probability:.1%}")
        ax.set_xlabel("Map X")
        ax.set_ylabel("Map Y")
        ax.text(
            0.01, 0.01,
            f"Problem: {label}\nModel probability: {probability:.1%}\nTerrain: slope/elevation filtered",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        fig.tight_layout()
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return outfile

    def run(self, mountain, date, problem, make_map=True):
        print("\n" + "=" * 70)
        print("SINGLE-PROBLEM AVALANCHE INFERENCE")
        print("=" * 70)
        print(f"Mountain : {mountain}")
        print(f"Date     : {pd.Timestamp(date):%Y-%m-%d}")
        print(f"Problem  : {PROBLEM_LABELS[problem]}")
        print("\n🤖 Using Parquet history through requested date...")
        probability, X = self.predict_problem(mountain, date, problem)
        print(f"✓ {PROBLEM_LABELS[problem]} probability: {probability:.2%}")

        result = {
            "mountain": mountain,
            "date": pd.Timestamp(date),
            "problem": problem,
            "problem_label": PROBLEM_LABELS[problem],
            "probability": probability,
            "features": X,
            "map_file": None,
        }

        if make_map:
            print(f"🗺️ Mapping {PROBLEM_LABELS[problem]} only...")
            raster, elevation, slope, aspect, aspect_classes, mask, transform = self.build_problem_raster(problem, probability)
            outfile = self.save_problem_map(mountain, date, problem, probability, raster, elevation, slope, mask)
            result["map_file"] = outfile
            print(f"✓ Map saved: {outfile}")
        return result
    
    def predict_problem(self, mountain, date, problem):
        if problem not in AVALANCHE_PROBLEMS:
            raise ValueError(f"Unknown problem '{problem}'. Choose from {AVALANCHE_PROBLEMS}.")
        model_name = MODEL_NAMES[problem]
        if model_name not in self.models:
            raise FileNotFoundError(f"No trained model loaded for {problem}: {model_name}")
        history = self.get_input_history(mountain, date)
        X = self.prepare_features(history, date, model_name)
        model = self.models[model_name]
        proba = model.predict_proba(X)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError(f"{model_name} did not return binary probabilities.")
        return float(np.clip(proba[0, 1], 0.0, 1.0)), X

def select_mountain(pipeline):
    mountains = pipeline.available_mountains()
    print("\nAvailable mountains:")
    for i, m in enumerate(mountains, 1):
        print(f"  {i}. {m}")
    while True:
        value = input("\nSelect mountain: ").strip().lower()
        for m in mountains:
            if value == str(m).lower() or value == str(m).lower().replace("mount ", ""):
                return m
        try:
            idx = int(value) - 1
            if 0 <= idx < len(mountains):
                return mountains[idx]
        except ValueError:
            pass
        print("Invalid mountain.")


def select_date(pipeline, mountain):
    dates = pipeline.available_dates(mountain)
    print(f"\nAvailable date range: {dates[0]} → {dates[-1]}")
    while True:
        value = input("Forecast date (YYYY-MM-DD): ").strip()
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            print("Invalid date format.")
            continue
        if value in dates:
            return value
        print("That date is not available for this mountain.")


def select_problem():
    print("\nAvalanche Problems — select ONE:")
    for i, p in enumerate(AVALANCHE_PROBLEMS, 1):
        print(f"  {i}. {PROBLEM_LABELS[p]}")
    aliases = {
        "1": "wind_slab", "wind": "wind_slab", "wind slab": "wind_slab", "wind_slab": "wind_slab",
        "2": "storm_slab", "storm": "storm_slab", "storm slab": "storm_slab", "storm_slab": "storm_slab",
        "3": "wet_loose", "wet": "wet_loose", "wet loose": "wet_loose", "wet_loose": "wet_loose",
        "4": "persistent_slab", "persistent": "persistent_slab", "persistent slab": "persistent_slab", "persistent_slab": "persistent_slab",
    }
    while True:
        value = input("\nSelect ONE avalanche problem: ").strip().lower()
        if value in aliases:
            return aliases[value]
        print("Invalid avalanche problem.")


def main():
    print("\n" + "═" * 62)
    print("🏔️  ELLIES AVALANCHE TOOL")
    print("═" * 62)
    print("\n⚠️  Experimental ML Model")
    print("⚠️  This tool maps ONE selected avalanche problem at a time.")

    dem = input("\nDEM GeoTIFF path (press Enter to use MountHoodGEOTiff.tiff): ").strip()
    dem = dem or "MountHoodGEOTiff.tiff"
    pipeline = AvalancheForecastPipeline(dem_file=dem)

    while True:
        mountain = select_mountain(pipeline)
        date = select_date(pipeline, mountain)
        problem = select_problem()
        try:
            result = pipeline.run(mountain, date, problem, make_map=True)
            print("\n" + "═" * 62)
            print("🏔️ RESULT")
            print("═" * 62)
            print(f"Location:    {mountain}")
            print(f"Date:        {date}")
            print(f"Problem:     {result['problem_label']}")
            print(f"Probability: {result['probability']:.2%}")
            print(f"Map:         {result['map_file']}")
        except Exception as exc:
            print(f"\n❌ Prediction failed: {exc}")
        again = input("\nPredict another avalanche problem? (yes/no): ").strip().lower()
        if again not in ("y", "yes"):
            break


if __name__ == "__main__":
    main()
