'''
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import rasterio
from rasterio.windows import from_bounds

# Ensure path availability
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import interactive mapping components
try:
    from mapping.terrain.shading import (
        shadeSlopes,
        shadeCliffs,
        shadeFlats,
        shadeCurvature,
        shadeAspect,
    )
    from mapping.terrain.terrainFeatures import slope, roughness
    from src.mapping.inference.terrain_mapping import TerrainMapper
    from src.mapping.inference.avalanche_problem_mapper import AvalancheProblemMapper
except ImportError:
    # Fallbacks if imports are under src/
    from src.mapping.terrain.shading import (
        shadeSlopes,
        shadeCliffs,
        shadeFlats,
        shadeCurvature,
        shadeAspect,
    )
    from src.mapping.terrain.terrainFeatures import slope, roughness
    from src.mapping.inference.terrain_mapping import TerrainMapper
    from src.mapping.inference.avalanche_problem_mapper import AvalancheProblemMapper

# Import model inference and generation pipeline
from src.mapping.inference.inference_pipeline import AvalancheForecastPipeline, select_mountain, select_date
from src.mapping.inference.forecast_generator import ForecastGenerator

# Download module (uses provided custom download script logic)
try:
    from download_protocol import download
except ImportError:
    from src.mapping.download_protocol import download


class UI:
    WIDTH = 62

    @staticmethod
    def header():
        print("\n" + "─" * UI.WIDTH)
        print("🏔️  Ellie's Avalanche Tool (Wind Slab Focus Engine)")
        print("─" * UI.WIDTH)

    @staticmethod
    def section(title):
        print(f"\n{title}")
        print("─" * UI.WIDTH)

    @staticmethod
    def forecast_header():
        print("\n" + "═" * UI.WIDTH)
        print("🏔️ AVALANCHE FORECAST: WIND SLAB")
        print("═" * UI.WIDTH)

    @staticmethod
    def success(msg):
        print(f"✓ {msg}")

    @staticmethod
    def warning(msg):
        print(f"⚠️  {msg}")


ASPECTS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def execute():
    UI.section("📥 Location & Elevation Data Protocol")

    # Download or reference DEM and return geographic bounds
    location, elevData, xmin, ymin, xmax, ymax = download()

    UI.section("🤖 Machine Learning Inference")
    
    # Initialize pipeline with DEM reference
    pipeline = AvalancheForecastPipeline(dem_file=elevData if elevData else "MountHoodGEOTiff.tiff")
    generator = ForecastGenerator()

    # Determine targeted mountain and forecast date
    mountain = select_mountain(pipeline)
    date = select_date(pipeline, mountain)

    # Isolated inference for wind slab only
    predictions = {}
    target_problem = "wind_slab"
    
    try:
        prob, _ = pipeline.predict_problem(mountain, date, target_problem)
        # Populate wind_slab predictions across all aspect angles for aspect matching
        predictions["wind_slab"] = {f"wind_slab_{asp}": prob for asp in ASPECTS}
        UI.success(f"Wind Slab risk probability calculated: {prob:.2%}")
    except Exception as e:
        UI.warning(f"Could not calculate risk for wind_slab: {e}")

    # Generate text forecast discussion for wind slab
    UI.forecast_header()
    if predictions:
        forecast_output = generator.generate(predictions)
        print(forecast_output["discussion"])

    # If DEM data is missing, terminate cleanly without plotting
    if elevData is None or not Path(elevData).exists():
        UI.warning("Elevation raster missing. Interactive terrain map skipped.")
        print("\n👋 Program exited.")
        return

    UI.section("🗺️ Interactive Terrain Map Generator")

    # Open DEM dataset window corresponding to bounding box
    with rasterio.open(elevData) as src:
        transform = src.transform
        window = from_bounds(xmin, ymin, xmax, ymax, transform)
        Z = src.read(1, window=window)
        transform = src.window_transform(window)
        rows, cols = Z.shape

    x = transform.c + np.arange(cols) * transform.a
    y = transform.f + np.arange(rows) * transform.e
    Z = np.where(Z <= -9999, np.nan, Z)

    # Calculate terrain matrices
    slope_raster = slope(Z, ymin, ymax, x, y)
    roughness_raster = roughness(Z)

    mapper = TerrainMapper()
    problem_mapper = AvalancheProblemMapper()
    aspect_classes = mapper.build_aspect_classes(Z, ymin, ymax, x, y)

    show_map = input("\nGenerate interactive terrain map? (yes/no): ").strip().lower()
    if show_map not in {"yes", "y"}:
        print("\n👋 Program exited.")
        return

    # Interactive map configuration loop
    while True:
        rgb_raster = None
        
        overlay = input("\nShow forecasted Wind Slab hazard color overlay? (yes/no): ").strip().lower()
        if overlay in {"yes", "y"} and predictions:
            # Build risk array exclusively for wind_slab
            rgb_raster = problem_mapper.build_problem_color_map(
                aspect_classes,
                predictions,
                elevation_raster=Z,
                slope_raster=slope_raster,
            )
            terrain_weight = mapper.apply_terrain_weight(
                np.ones_like(slope_raster, dtype=float),
                slope_raster,
                roughness_raster,
            )
            background_mask = np.all(rgb_raster == [1.0, 1.0, 1.0], axis=-1)
            rgb_raster[~background_mask] *= terrain_weight[~background_mask, np.newaxis]
            
            max_val = np.nanmax(rgb_raster)
            if max_val > 0:
                rgb_raster /= max_val

        # Matplotlib Rendering Logic
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor("white")

        # Optional Topographic Contour Lines
        contour = input("\nShow contour lines? (yes/no): ").strip().lower()
        if contour in {"yes", "y"}:
            ax.contour(x, y, Z, levels=np.arange(500, 3500, 40), colors="black", linewidths=0.25)
            major = ax.contour(x, y, Z, levels=np.arange(500, 3500, 200), colors="black", linewidths=1.0)
            ax.clabel(major, inline=True, fontsize=8, fmt="%d m")

        # Optional Terrain Feature Shading Options
        t_feat = input("\nShow terrain feature shading? (yes/no): ").strip().lower()
        if t_feat in {"yes", "y"}:
            print("\nSelect features to shade:")
            print("1 - Cliffs\n2 - Flats\n3 - Curvature\n4 - Aspect\n5 - Slope")
            selection = input("Enter feature numbers (e.g. 1,3,5): ")
            selected = {s.strip() for s in selection.split(",")}

            if "1" in selected:
                shadeCliffs(ax, Z, x, y, ymin, ymax)
            if "2" in selected:
                shadeFlats(ax, Z, x, y, ymin, ymax)
            if "3" in selected:
                shadeCurvature(ax, Z, x, y, ymin, ymax)
            if "4" in selected:
                shadeAspect(ax, Z, x, y, ymin, ymax)
            if "5" in selected:
                shadeSlopes(ax, Z, x, y, ymin, ymax)

        # Plot Wind Slab Overlay with Alpha Transparency
        if rgb_raster is not None:
            rgba_raster = np.dstack([rgb_raster, np.ones((rgb_raster.shape[0], rgb_raster.shape[1]))])
            
            # Mask out background so terrain features and contours remain visible
            background_mask = np.all(rgb_raster >= 0.98, axis=-1)
            rgba_raster[background_mask, 3] = 0.0    # Transparent background
            rgba_raster[~background_mask, 3] = 0.75  # 75% opacity for Wind Slab hazard zones

            ax.imshow(
                rgba_raster,
                extent=[xmin, xmax, ymin, ymax],
                origin="upper",
            )
            
            # Isolate legend element to display only Wind Slab
            wind_color = problem_mapper.problem_colors.get("wind_slab", "#3366CC")
            legend_elements = [Patch(facecolor=wind_color, label="Wind Slab")]
            ax.legend(handles=legend_elements, loc="upper right")

        # Scale aspect ratio based on average latitude
        mean_lat = (ymin + ymax) / 2
        ax.set_aspect(1 / np.cos(np.radians(mean_lat)))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Wind Slab Avalanche Hazard Map" if rgb_raster is not None else "Terrain Map")

        plt.tight_layout()
        plt.show()

        reopen = input("\nReopen and adjust map display? (yes/no): ").strip().lower()
        if reopen not in {"yes", "y"}:
            print("\n👋 Program exited.")
            break


def main():
    UI.header()
    response = input("\nRun program? (yes/no): ").strip().lower()
    if response in {"yes", "y"}:
        execute()
    else:
        print("\n👋 Program exited.")


if __name__ == "__main__":
    main()

'''

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import rasterio
from rasterio.windows import from_bounds

# Ensure path availability
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import interactive mapping components
try:
    from mapping.terrain.shading import (
        shadeSlopes,
        shadeCliffs,
        shadeFlats,
        shadeCurvature,
        shadeAspect,
    )
    from mapping.terrain.terrainFeatures import slope, roughness
    from src.mapping.inference.terrain_mapping import TerrainMapper
    from src.mapping.inference.avalanche_problem_mapper import AvalancheProblemMapper
except ImportError:
    # Fallbacks if imports are under src/
    from src.mapping.terrain.shading import (
        shadeSlopes,
        shadeCliffs,
        shadeFlats,
        shadeCurvature,
        shadeAspect,
    )
    from src.mapping.terrain.terrainFeatures import slope, roughness
    from src.mapping.inference.terrain_mapping import TerrainMapper
    from src.mapping.inference.avalanche_problem_mapper import AvalancheProblemMapper

# Import model inference and generation pipeline
from src.mapping.inference.inference_pipeline import AvalancheForecastPipeline, select_mountain, select_date
from src.mapping.inference.forecast_generator import ForecastGenerator

# Download module (uses provided custom download script logic)
try:
    from download_protocol import download
except ImportError:
    from src.mapping.download_protocol import download


class UI:
    WIDTH = 62

    @staticmethod
    def header():
        print("\n" + "─" * UI.WIDTH)
        print("🏔️  Ellie's Avalanche Tool (Interactive GIS Engine)")
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


KEY_MAP = {
    "dry_loose": "persistent_slab",
    "wind_slab": "wind_slab",
    "storm_slab": "storm_slab",
    "wet_loose": "wet_loose",
}

ASPECTS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def execute():
    UI.section("📥 Location & Elevation Data Protocol")

    # Download or reference DEM and return geographic bounds
    location, elevData, xmin, ymin, xmax, ymax = download()

    UI.section("🤖 Machine Learning Inference")
    
    # Initialize pipeline with DEM reference
    pipeline = AvalancheForecastPipeline(dem_file=elevData if elevData else "MountHoodGEOTiff.tiff")
    generator = ForecastGenerator()

    # Determine targeted mountain and forecast date
    mountain = select_mountain(pipeline)
    date = select_date(pipeline, mountain)

    # Predict probabilities for all active problem types across all cardinal aspects
    predictions = {}
    for problem in pipeline.available_problems():
        try:
            prob, _ = pipeline.predict_problem(mountain, date, problem)
            fg_key = KEY_MAP.get(problem, problem)
            # Expand probability across all 8 aspect directions for AvalancheProblemMapper matching
            predictions[fg_key] = {f"{fg_key}_{asp}": prob for asp in ASPECTS}
        except Exception as e:
            UI.warning(f"Could not calculate risk for {problem}: {e}")

    # Generate text forecast discussion
    UI.forecast_header()
    if predictions:
        forecast_output = generator.generate(predictions)
        print(forecast_output["discussion"])

    # If DEM data is missing, terminate cleanly without plotting
    if elevData is None or not Path(elevData).exists():
        UI.warning("Elevation raster missing. Interactive terrain map skipped.")
        print("\n👋 Program exited.")
        return

    UI.section("🗺️ Interactive Terrain Map Generator")

    # Open DEM dataset window corresponding to bounding box
    with rasterio.open(elevData) as src:
        transform = src.transform
        window = from_bounds(xmin, ymin, xmax, ymax, transform)
        Z = src.read(1, window=window)
        transform = src.window_transform(window)
        rows, cols = Z.shape

    x = transform.c + np.arange(cols) * transform.a
    y = transform.f + np.arange(rows) * transform.e
    Z = np.where(Z <= -9999, np.nan, Z)

    # Calculate terrain matrices
    slope_raster = slope(Z, ymin, ymax, x, y)
    roughness_raster = roughness(Z)

    mapper = TerrainMapper()
    problem_mapper = AvalancheProblemMapper()
    aspect_classes = mapper.build_aspect_classes(Z, ymin, ymax, x, y)

    show_map = input("\nGenerate interactive terrain map? (yes/no): ").strip().lower()
    if show_map not in {"yes", "y"}:
        print("\n👋 Program exited.")
        return

    # Interactive map configuration loop
    while True:
        rgb_raster = None
        
        overlay = input("\nShow forecasted avalanche hazard color overlay? (yes/no): ").strip().lower()
        if overlay in {"yes", "y"} and predictions:
            # Build problem risk array
            rgb_raster = problem_mapper.build_problem_color_map(
                aspect_classes,
                predictions,
                elevation_raster=Z,
                slope_raster=slope_raster,
            )
            terrain_weight = mapper.apply_terrain_weight(
                np.ones_like(slope_raster, dtype=float),
                slope_raster,
                roughness_raster,
            )
            background_mask = np.all(rgb_raster == [1.0, 1.0, 1.0], axis=-1)
            rgb_raster[~background_mask] *= terrain_weight[~background_mask, np.newaxis]
            
            max_val = np.nanmax(rgb_raster)
            if max_val > 0:
                rgb_raster /= max_val

        # Matplotlib Rendering Logic
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor("white")

        # Optional Topographic Contour Lines
        contour = input("\nShow contour lines? (yes/no): ").strip().lower()
        if contour in {"yes", "y"}:
            ax.contour(x, y, Z, levels=np.arange(500, 3500, 40), colors="black", linewidths=0.25)
            major = ax.contour(x, y, Z, levels=np.arange(500, 3500, 200), colors="black", linewidths=1.0)
            ax.clabel(major, inline=True, fontsize=8, fmt="%d m")

        # Optional Terrain Feature Shading Options
        t_feat = input("\nShow terrain feature shading? (yes/no): ").strip().lower()
        if t_feat in {"yes", "y"}:
            print("\nSelect features to shade:")
            print("1 - Cliffs\n2 - Flats\n3 - Curvature\n4 - Aspect\n5 - Slope")
            selection = input("Enter feature numbers (e.g. 1,3,5): ")
            selected = {s.strip() for s in selection.split(",")}

            if "1" in selected:
                shadeCliffs(ax, Z, x, y, ymin, ymax)
            if "2" in selected:
                shadeFlats(ax, Z, x, y, ymin, ymax)
            if "3" in selected:
                shadeCurvature(ax, Z, x, y, ymin, ymax)
            if "4" in selected:
                shadeAspect(ax, Z, x, y, ymin, ymax)
            if "5" in selected:
                shadeSlopes(ax, Z, x, y, ymin, ymax)

        # Plot Avalanche Hazard Overlay with Alpha Channel Transparency
        if rgb_raster is not None:
            # Convert RGB array (H, W, 3) to RGBA array (H, W, 4)
            rgba_raster = np.dstack([rgb_raster, np.ones((rgb_raster.shape[0], rgb_raster.shape[1]))])
            
            # Mask out pure white background pixels so contours/terrain show through
            background_mask = np.all(rgb_raster >= 0.98, axis=-1)
            rgba_raster[background_mask, 3] = 0.0    # Fully transparent background
            rgba_raster[~background_mask, 3] = 0.75  # 75% opacity for active hazards

            ax.imshow(
                rgba_raster,
                extent=[xmin, xmax, ymin, ymax],
                origin="upper",
            )
            
            legend_elements = [
                Patch(facecolor=color, label=p.replace("_", " ").title())
                for p, color in problem_mapper.problem_colors.items()
            ]
            ax.legend(handles=legend_elements, loc="upper right")

        # Scale aspect ratio based on average latitude
        mean_lat = (ymin + ymax) / 2
        ax.set_aspect(1 / np.cos(np.radians(mean_lat)))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Combined Avalanche Problem Map" if rgb_raster is not None else "Terrain Map")

        plt.tight_layout()
        plt.show()

        reopen = input("\nReopen and adjust map display? (yes/no): ").strip().lower()
        if reopen not in {"yes", "y"}:
            print("\n👋 Program exited.")
            break


def main():
    UI.header()
    response = input("\nRun program? (yes/no): ").strip().lower()
    if response in {"yes", "y"}:
        execute()
    else:
        print("\n👋 Program exited.")


if __name__ == "__main__":
    main()