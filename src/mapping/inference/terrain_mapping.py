"""
terrain_mapping.py

Map avalanche probabilities onto terrain.
"""

import numpy as np

from src.mapping.terrain.terrainFeatures import aspect

class TerrainMapper:

    def classify_aspect(
        self,
        aspect_degrees,
    ):

        classes = np.empty(
            aspect_degrees.shape,
            dtype=object,
        )

        classes[
            (aspect_degrees >= 337.5)
            | (aspect_degrees < 22.5)
        ] = "N"

        classes[
            (aspect_degrees >= 22.5)
            & (aspect_degrees < 67.5)
        ] = "NE"

        classes[
            (aspect_degrees >= 67.5)
            & (aspect_degrees < 112.5)
        ] = "E"

        classes[
            (aspect_degrees >= 112.5)
            & (aspect_degrees < 157.5)
        ] = "SE"

        classes[
            (aspect_degrees >= 157.5)
            & (aspect_degrees < 202.5)
        ] = "S"

        classes[
            (aspect_degrees >= 202.5)
            & (aspect_degrees < 247.5)
        ] = "SW"

        classes[
            (aspect_degrees >= 247.5)
            & (aspect_degrees < 292.5)
        ] = "W"

        classes[
            (aspect_degrees >= 292.5)
            & (aspect_degrees < 337.5)
        ] = "NW"

        return classes
    
    def build_aspect_classes(
        self,
        Z,
        ymin,
        ymax,
        x,
        y,
    ):

        aspect_raster = aspect(
            Z,
            ymin,
            ymax,
            x,
            y,
        )

        return self.classify_aspect(
            aspect_raster
        )

    def build_problem_raster(
        self,
        aspect_classes,
        predictions,
        problem,
    ):

        risk = np.zeros(
            aspect_classes.shape,
            dtype=float,
        )

        directions = [
            "N",
            "NE",
            "E",
            "SE",
            "S",
            "SW",
            "W",
            "NW",
        ]

        for direction in directions:

            target = (
                f"{problem}_{direction}"
            )

            if target not in predictions[problem]:
                continue

            probability = (
                predictions[problem][target]
            )

            mask = (
                aspect_classes
                == direction
            )

            risk[mask] = probability

        return risk

    def build_all_rasters(
        self,
        aspect_classes,
        predictions,
    ):

        rasters = {}

        for problem in predictions.keys():

            rasters[problem] = (
                self.build_problem_raster(
                    aspect_classes,
                    predictions,
                    problem,
                )
            )

        return rasters

    def build_combined_raster(
        self,
        aspect_classes,
        predictions,
        method="max",
    ):
        """
        Combine all avalanche problems into a single raster.

        method:
            max  -> worst-case hazard
            mean -> average hazard
            sum  -> cumulative hazard
        """

        rasters = self.build_all_rasters(
            aspect_classes,
            predictions,
        )

        stack = np.stack(
            list(rasters.values()),
            axis=0,
        )

        if method == "max":
            return np.nanmax(
                stack,
                axis=0,
            )

        elif method == "mean":
            return np.nanmean(
                stack,
                axis=0,
            )

        elif method == "sum":
            return np.nansum(
                stack,
                axis=0,
            )

        raise ValueError(
            f"Unknown method: {method}"
        )

    def apply_terrain_weight(
        self,
        risk_raster,
        slope_raster,
        roughness_raster=None,
    ):
        """
        Boost avalanche terrain and suppress flat terrain.

        Typical avalanche terrain:
            30-45 degrees.
        """

        slope_weight = np.clip(
            (slope_raster - 25.0) / 15.0,
            0.0,
            1.0,
        )

        if roughness_raster is None:

            return (
                risk_raster
                * slope_weight
            )

        roughness_norm = (
            roughness_raster
            / np.nanpercentile(
                roughness_raster,
                90,
            )
        )

        roughness_norm = np.clip(
            roughness_norm,
            0.0,
            1.0,
        )

        terrain_weight = (
            0.8 * slope_weight
            + 0.2 * roughness_norm
        )

        return (
            risk_raster
            * terrain_weight
        )