import numpy as np

def normalize(x):
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)

    if x_max == x_min:
        return np.zeros_like(x)

    return (x - x_min) / (x_max - x_min)

def calculate_terrain_score(slope, roughness, tpi, distance_to_ridge, plan_curvature = None, profile_curvature = None):

    # Normalize features to [0, 1]
    roughness_norm = normalize(roughness)
    ridge_norm = normalize(distance_to_ridge)

    # Avalanche likelihood peaks around 37.5° slopes
    slope_score = np.exp(-((slope - 37.5) / 8.0) ** 2)

    # Negative TPI corresponds to gullies/depressions
    # Positive TPI corresponds to convex terrain features
    convexity_score = normalize(tpi)

    curvature_score = 0.0

    if plan_curvature is not None and profile_curvature is not None: 
        curvature_score = (
            normalize(plan_curvature) + 
            normalize(profile_curvature)
        ) / 2.0

    terrain_score = (
        0.45 * slope_score +
        0.20 * convexity_score +
        0.15 * (1 - roughness_norm) +
        0.10 * (1 - ridge_norm) +
        0.10 * curvature_score
    )

    return np.clip(terrain_score, 0.0, 1.0)