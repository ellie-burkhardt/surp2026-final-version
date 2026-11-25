import numpy as np
from  src.mapping.terrain.terrainFeatures import (
    slope,
    aspect,
    curvature,
)

def shadeSlopes(ax, Z, x, y, ymin, ymax, alpha = 0.35):
    slope_degrees = slope(Z, ymin, ymax, x, y)

    # Gentle terrain (<30°)
    ax.contourf(
        x,
        y,
        slope_degrees,
        levels=[0, 30],
        colors=["green"],
        alpha=alpha
    )

    # Moderate terrain (30–45°)
    ax.contourf(
        x,
        y,
        slope_degrees,
        levels=[30, 45],
        colors=["red"],
        alpha=alpha
    )

    # Steep terrain (>45°)
    ax.contourf(
        x,
        y,
        slope_degrees,
        levels=[45, np.nanmax(slope_degrees)],
        colors=["yellow"],
        alpha=alpha
    )

    return slope_degrees

def shadeCliffs(ax, Z, x, y, ymin, ymax, alpha = 0.35):
    slope_degrees = slope(Z, ymin, ymax, x, y)

    ax.contourf(
        x,
        y,
        slope_degrees,
        levels=[
            70,
            np.nanmax(slope_degrees)
        ],
        colors=["darkred"],
        alpha=alpha
    )

def shadeFlats(ax, Z, x, y, ymin, ymax, alpha = 0.35):
    slope_degrees = slope(Z, ymin, ymax, x, y)

    ax.contourf(
        x,
        y,
        slope_degrees,
        levels=[0, 5],
        colors=["chartreuse"],
        alpha=alpha
    )

def shadeCurvature(ax, Z, x, y, ymin, ymax, alpha = 0.35):
    curvatureV = curvature(
        Z,
        ymin,
        ymax,
        x,
        y
    )

    # Valleys (concave)
    ax.contourf(
        x,
        y,
        curvatureV,
        levels=[np.nanmin(curvatureV), -1e-5],
        colors=["blue"],
        alpha=alpha
    )

    # Ridges (convex)
    ax.contourf(
        x,
        y,
        curvatureV,
        levels=[1e-5, np.nanmax(curvatureV)],
        colors=["orange"],
        alpha=alpha
    )

    return curvatureV

def shadeAspect(ax, Z, x, y, ymin, ymax, alpha=0.35):
    aspectV = aspect(
        Z,
        ymin,
        ymax,
        x,
        y
    )

    ax.contourf(
        x,
        y,
        aspectV,
        levels = np.arange(0, 361, 45),
        cmap = "hsv",
        alpha = alpha
    )

    return aspectV
