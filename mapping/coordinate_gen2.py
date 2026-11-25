from pathlib import Path
import sys
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
TRAINED_MODELS_ROOT = REPO_ROOT / "trained_models"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(TRAINED_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINED_MODELS_ROOT))

from mapping.terrain.shading import (
    shadeSlopes,
    shadeCliffs,
    shadeFlats,
    shadeCurvature,
    shadeAspect,
)

from mapping.terrain.terrainFeatures import (
    slope,
    roughness,
    tpi_small,
    tpi_large,
    distance_to_ridge,
)

from mapping.terrain.terrainScores import calculate_terrain_score

from src.mapping.download_protocol import download

from src.mapping.inference.terrain_mapping import TerrainMapper
from src.mapping.inference.model_loader import ModelLoader
from mapping.inference.predictor import AvalanchePredictor
from src.mapping.inference.inference_pipeline import AvalancheForecastPipeline
from src.mapping.inference.avalanche_problem_mapper import AvalancheProblemMapper

from ml.dataset.dataset_loader import AvalancheDataset


class UI:

    WIDTH = 62

    @staticmethod
    def header():
        print("\n" + "─" * UI.WIDTH)
        print("🏔️  Ellie's Avalanche Tool")
        print("─" * UI.WIDTH)

    @staticmethod
    def section(title):
        print(f"\n{title}")
        print("─" * UI.WIDTH)

    @staticmethod
    def forecast_header():
        print("\n" + "═" * UI.WIDTH)
        print("🏔️ AVALANCHE FORECAST")
        print("═" * UI.WIDTH)

    @staticmethod
    def success(msg):
        print(f"✓ {msg}")

    @staticmethod
    def warning(msg):
        print(f"⚠️  {msg}")

    @staticmethod
    def info(msg):
        print(f"ℹ️  {msg}")

    @staticmethod
    def map(msg):
        print(f"🗺️  {msg}")

    @staticmethod
    def download(msg):
        print(f"📥 {msg}")

    @staticmethod
    def model(msg):
        print(f"🤖 {msg}")

    @staticmethod
    def spacer():
        print()

FEATURE_NAMES = [
    'PREC',
    'snow_depth',
    'swe',
    'temp',
    'precip',
    'wind_speed',
    'year',
    'temp_7d_mean',
    'precip_7d_sum',
    'wind_7d_mean',
    'snowfall',
    'precip_1d',
    'precip_3d',
    'precip_5d',
    'precip_7d',
    'precip_14d',
    'precip_30d',
    'precip_60d',
    'snowfall_1d',
    'snowfall_3d',
    'snowfall_5d',
    'snowfall_7d',
    'snowfall_14d',
    'snowfall_30d',
    'snowfall_60d',
    'swe_1d',
    'swe_3d',
    'swe_5d',
    'swe_7d',
    'swe_14d',
    'swe_30d',
    'swe_60d',
    'snow_depth_1d',
    'snow_depth_3d',
    'snow_depth_5d',
    'snow_depth_7d',
    'snow_depth_14d',
    'snow_depth_30d',
    'snow_depth_60d',
    'temp_change_1d',
    'temp_change_3d',
    'temp_change_7d',
    'temp_change_14d',
    'temp_mean_3d',
    'temp_mean_7d',
    'temp_mean_14d',
    'days_since_10mm',
    'days_since_25mm',
    'days_since_50mm',
    'day_sin',
    'day_cos',
    'wind_dir_sin',
    'wind_dir_cos',
]

def execute():

    #
    # Elevation Data
    #

    UI.section("📥 Elevation Data")

    location, elevData, xmin, ymin, xmax, ymax = download()

    if elevData != None: 

        # Load USGS DEM Elevation Data
        with rasterio.open(elevData) as src:

            transform = src.transform

            window = from_bounds(
                xmin,
                ymin,
                xmax,
                ymax,
                transform,
            )

            Z = src.read(
                1,
                window=window,
            )

            transform = src.window_transform(
                window
            )

            rows, cols = Z.shape

        x = (
            transform.c
            + np.arange(cols) * transform.a
        )

        y = (
            transform.f
            + np.arange(rows) * transform.e
        )

        Z = np.where(
            Z <= -9999,
            np.nan,
            Z,
        )

        slope_raster = slope(
            Z,
            ymin,
            ymax,
            x,
            y,
        )

        roughness_raster = roughness(
            Z,
        )

        mapper = TerrainMapper()

        problem_mapper = AvalancheProblemMapper()

        aspect_classes = (
            mapper.build_aspect_classes(
                Z,
                ymin,
                ymax,
                x,
                y,
            )
        )

        risk_raster = None
        rgb_raster = None

        def make_plot():

            fig, ax = plt.subplots(
                figsize=(12, 8)
            )

            ax.set_facecolor(
                "white"
            )

            contour = input(
                "\nShow contour lines? (yes/no): "
            ).strip().lower()

            if contour == "yes":

                ax.contour(
                    x,
                    y,
                    Z,
                    levels=np.arange(
                        500,
                        3500,
                        40,
                    ),
                    colors="black",
                    linewidths=0.25,
                )

                major = ax.contour(
                    x,
                    y,
                    Z,
                    levels=np.arange(
                        500,
                        3500,
                        200,
                    ),
                    colors="black",
                    linewidths=1.0,
                )

                ax.clabel(
                    major,
                    inline=True,
                    fontsize=8,
                    fmt="%d m",
                )

            t_feat = input("\nShow terrain features? (yes/no): ").strip().lower()

            if t_feat == "yes":

                UI.section(
                    "Terrain Features"
                )

                print("1 - Cliffs")
                print("2 - Flats")
                print("3 - Curvature")
                print("4 - Aspect")
                print("5 - Slope")

                selection = input(
                    "\nEnter feature numbers separated by commas (e.g. 1,3,4): "
                )

                selected = {
                    s.strip()
                    for s in selection.split(",")
                }

                if "1" in selected:
                    shadeCliffs(
                        ax,
                        Z,
                        x,
                        y,
                        ymin,
                        ymax,
                    )

                if "2" in selected:
                    shadeFlats(
                        ax,
                        Z,
                        x,
                        y,
                        ymin,
                        ymax,
                    )

                if "3" in selected:
                    shadeCurvature(
                        ax,
                        Z,
                        x,
                        y,
                        ymin,
                        ymax,
                    )

                if "4" in selected:
                    shadeAspect(
                        ax,
                        Z,
                        x,
                        y,
                        ymin,
                        ymax,
                    )

                if "5" in selected:
                    shadeSlopes(
                        ax,
                        Z,
                        x,
                        y,
                        ymin,
                        ymax,
                    )
        
            if rgb_raster is not None:

                ax.imshow(
                    rgb_raster,
                    extent=[
                        xmin,
                        xmax,
                        ymin,
                        ymax,
                    ],
                    origin="upper",
                    alpha=0.75,
                )

                legend_elements = []

                for problem, color in (
                    problem_mapper.problem_colors.items()
                ):

                    legend_elements.append(

                        Patch(
                            facecolor=color,
                            label=problem.replace(
                                "_",
                                " "
                            ).title(),
                        )
                    )

                ax.legend(
                    handles=legend_elements,
                    loc="upper right",
                )

            mean_lat = (
                ymin + ymax
            ) / 2

            ax.set_aspect(
                1 / np.cos(
                    np.radians(
                        mean_lat
                    )
                )
            )

            ax.set_xlim(
                xmin,
                xmax,
            )

            ax.set_ylim(
                ymin,
                ymax,
            )

            if rgb_raster is None:

                ax.set_title(
                    "Terrain Map"
                )

            else:

                ax.set_title(
                    "Combined Avalanche Problem Map"
                )

            ax.set_xlabel(
                "Longitude"
            )

            ax.set_ylabel(
                "Latitude"
            )

            plt.tight_layout()
            plt.show()

        if location in {"yes", "y"}:

            forecast = (
                AvalancheForecastPipeline.run_main(True)
            )

        else:

            forecast = (
                AvalancheForecastPipeline.run_main(False)
            )

        mountain, date, forecast_problem = forecast

        pipeline = AvalancheForecastPipeline()

        predictions = (
            pipeline.get_predictions(
                mountain=mountain,
                date=date,
            )
        )

        UI.section(
            "Forecast Summary"
        )

        for problem in predictions:

            max_prob = max(
                predictions[problem].values()
            )

            print(
                f"{problem:<25}"
                f"{max_prob:.1%}"
            )

        UI.section(
            "🗺️ Terrain Mapping"
        )

        if mountain == "hood":

            print("Automatic raster mapping is available for this location.")

        else:

            print("Automatic raster mapping is not available for this location.")

            print("\nA terrain map can still be generated.")

            print("Avalanche probabilities will not be mapped automatically.")

        show = input("\nGenerate terrain map? (yes/no): ").strip().lower()

        if show == "yes":

            while True:

                print(
                    "\n🚀 Opening interactive terrain map..."
                )

                overlay = input(
                    "\nShow forecasted avalanche hazards? (yes/no): "
                ).strip().lower()

                if overlay == "yes":

                    rgb_raster = (
                        problem_mapper.build_problem_color_map(
                            aspect_classes,
                            predictions,
                            elevation_raster=Z,
                            slope_raster=slope_raster,
                        )
                    )

                    terrain_weight = (
                        mapper.apply_terrain_weight(
                            np.ones_like(
                                slope_raster,
                                dtype=float,
                            ),
                            slope_raster,
                            roughness_raster,
                        )
                    )

                    background_mask = np.all(
                        rgb_raster == [1.0, 1.0, 1.0],
                        axis=-1,
                    )

                    rgb_raster[~background_mask] *= (
                        terrain_weight[~background_mask, np.newaxis]
                    )

                    max_val = np.nanmax(rgb_raster)

                    if max_val > 0:
                        rgb_raster /= max_val

                else:
                    rgb_raster = None

                make_plot()

                reopen = input(
                    "\nReopen map? (yes/no): "
                ).strip().lower()

                if reopen != "yes":
                    print("\n👋 Program exited.")
                    break
    else: 

        forecast = (
                AvalancheForecastPipeline.run_main(False)
            )

        mountain, date, forecast_problem = forecast

        pipeline = AvalancheForecastPipeline()

        predictions = (
            pipeline.get_predictions(
                mountain=mountain,
                date=date,
            )
        )

        UI.section(
            "Forecast Summary"
        )

        for problem in predictions:

            max_prob = max(
                predictions[problem].values()
            )

            print(
                f"{problem:<25}"
                f"{max_prob:.1%}"
            )
        
        print("\n👋 Program exited.")
            
def main():

    UI.header()

    response = input(
        "\nRun program? (yes/no): "
    ).strip().lower()

    if response in {"yes", "y"}:
        execute()
    else:
        print("\n👋 Program exited.")


if __name__ == "__main__":
    main()

