import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT / "src")
)

from src.mapping.inference.terrain_mapping import TerrainMapper


class AvalancheProblemMapper:

    def __init__(self):

        self.terrain_mapper = TerrainMapper()

        self.problem_colors = {

            # red
            "wind_slab":
                np.array([1.0, 0.0, 0.0]),

            # green
            "storm_slab":
                np.array([0.0, 1.0, 0.0]),

            # blue
            "wet_loose":
                np.array([0.0, 0.0, 1.0]),

            # yellow
            "persistent_slab":
                np.array([1.0, 1.0, 0.0]),
        }

        self.elevation_rules = {

            # Wind slabs:
            # More likely to be relevant in upper/elevated terrain
            # where wind loading is common.
            "wind_slab": {
                "min": 1800,
                "max": None,
            },

            # Storm slabs:
            # Can occur across a broad elevation range,
            # but start with mid/high terrain.
            "storm_slab": {
                "min": 1500,
                "max": None,
            },

            # Persistent slabs:
            # Often important in colder/high-elevation terrain.
            "persistent_slab": {
                "min": 1800,
                "max": None,
            },

            # Wet loose:
            # More relevant at lower elevations where warming
            # and/or rain can affect the snowpack.
            "wet_loose": {
                "min": None,
                "max": 2500,
            },

        }


    def build_problem_raster(
        self,
        aspect_classes,
        predictions,
        problem,
        elevation_raster=None,
        slope_raster=None,
    ):

        raster = (
            self.terrain_mapper
            .build_problem_raster(
                aspect_classes,
                predictions,
                problem,
            )
        )

        #
        # Elevation filtering
        #

        if elevation_raster is not None:

            elev_mask = (
                self.build_elevation_mask(
                    elevation_raster,
                    problem,
                )
            )

            raster = np.where(
                elev_mask,
                raster,
                0.0,
            )

        #
        # Slope filtering
        #

        if slope_raster is not None:

            slope_rules = {

                "wind_slab": (30, 60),
                "storm_slab": (30, 60),
                "persistent_slab": (30, 55),
                "deep_persistent_slab": (30, 55),
                "wet_loose": (35, 80),
                "dry_loose": (40, 80),
                "wet_slab": (25, 60),
                "cornice": (20, 90),
                "glide": (15, 45),
            }

            min_slope, max_slope = (
                slope_rules.get(
                    problem,
                    (30, 60),
                )
            )

            slope_mask = (
                (slope_raster >= min_slope)
                &
                (slope_raster <= max_slope)
            )

            raster = np.where(
                slope_mask,
                raster,
                0.0,
            )

        return raster

    def build_all_rasters(
        self,
        aspect_classes,
        predictions,
        elevation_raster=None,
        slope_raster=None,
    ):

        rasters = {}

        for problem in predictions:

            rasters[problem] = (
                self.build_problem_raster(
                    aspect_classes,
                    predictions,
                    problem,
                    elevation_raster=elevation_raster,
                    slope_raster=slope_raster,
                )
            )

        return rasters

    def build_problem_color_map(
        self,
        aspect_classes,
        predictions,
        elevation_raster=None,
        slope_raster=None,
        normalize=True,
        threshold=0.25,
        alpha=0.35,
    ):
        """
        Build an RGB avalanche-problem map.

        Each avalanche problem is treated as a translucent
        color layer. Where multiple problems overlap, their
        colors are alpha-blended together.

        Parameters
        ----------
        threshold : float
            Minimum prediction probability required for a
            problem to contribute.

        alpha : float
            Maximum opacity of each avalanche problem.

        normalize : bool
            Normalize prediction rasters before applying them
            as opacity.

        Returns
        -------
        rgb : ndarray
            RGB image with values in [0, 1].
        """

        shape = aspect_classes.shape

        # Start with white background.
        rgb = np.ones(
            (*shape, 3),
            dtype=float,
        )

        rasters = self.build_all_rasters(
            aspect_classes,
            predictions,
            elevation_raster=elevation_raster,
            slope_raster=slope_raster,
        )

        ALLOWED_PROBLEMS = {
            "wind_slab",
            "storm_slab",
            "wet_loose",
            "persistent_slab",
        }

        #
        # Alpha-composite each avalanche problem
        # onto the previous result.
        #

        for problem, raster in rasters.items():

            if problem not in ALLOWED_PROBLEMS:
                continue

            raster = np.asarray(
                raster,
                dtype=float,
            )

            #
            # Ignore weak predictions.
            #

            raster = np.where(
                raster >= threshold,
                raster,
                0.0,
            )

            #
            # Replace invalid values.
            #

            raster = np.nan_to_num(
                raster,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            #
            # Optional normalization.
            #

            if normalize:

                max_val = np.max(raster)

                if max_val > 0:
                    raster = raster / max_val

            #
            # Convert prediction strength into alpha.
            #
            # A prediction of 1.0 gives the requested
            # maximum alpha.
            #

            layer_alpha = (
                np.clip(raster, 0.0, 1.0)
                * alpha
            )

            color = self.problem_colors[
                problem
            ]

            #
            # Alpha compositing:
            #
            # new = old * (1 - alpha) + color * alpha
            #

            layer_alpha = (
                layer_alpha[..., np.newaxis]
            )

            rgb = (
                rgb * (1.0 - layer_alpha)
                + color * layer_alpha
            )

        rgb = np.clip(
            rgb,
            0.0,
            1.0,
        )

        return rgb

    def build_dominant_problem_map(
        self,
        aspect_classes,
        predictions,
    ):
        """
        Returns:

        dominant_problem_index
        dominant_probability
        """

        rasters = self.build_all_rasters(
            aspect_classes,
            predictions,
        )

        names = list(
            rasters.keys()
        )

        stack = np.stack(
            [
                rasters[p]
                for p in names
            ],
            axis=0,
        )

        dominant_idx = np.argmax(
            stack,
            axis=0,
        )

        dominant_prob = np.max(
            stack,
            axis=0,
        )

        return (
            dominant_idx,
            dominant_prob,
            names,
        )

    def build_elevation_mask(
        self,
        elevation_raster,
        problem,
    ):
        """
        Return a boolean mask indicating where the elevation
        is appropriate for the avalanche problem.
        """

        rules = self.elevation_rules.get(
            problem,
            {
                "min": None,
                "max": None,
            },
        )

        elevation_raster = np.asarray(
            elevation_raster,
            dtype=float,
        )

        mask = np.isfinite(
            elevation_raster
        )

        min_elevation = rules.get("min")
        max_elevation = rules.get("max")

        if min_elevation is not None:

            mask &= (
                elevation_raster
                >= min_elevation
            )

        if max_elevation is not None:

            mask &= (
                elevation_raster
                <= max_elevation
            )

        return mask
