import numpy as np
from scipy.ndimage import generic_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage import distance_transform_edt

def grad(Z, ymin, ymax, x, y):
    mean_lat = (ymin + ymax) / 2

    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * np.cos(np.radians(mean_lat))

    dx_m = abs(x[1] - x[0]) * meters_per_degree_lon
    dy_m = abs(y[1] - y[0]) * meters_per_degree_lat

    Zy, Zx = np.gradient(
        Z,
        dy_m,
        dx_m
    )

    return (Zx, Zy, dx_m, dy_m)

def elevation(Z): 
    return Z.copy()

def slope(Z, ymin, ymax, x, y):
    Zx, Zy, *rest= grad(Z, ymin, ymax, x, y)
    
    gradient_magnitude = np.sqrt(Zx**2 + Zy**2)

    slope_degrees = np.degrees(np.arctan(gradient_magnitude))

    return slope_degrees

def aspect(Z, ymin, ymax, x, y):
    Zx, Zy, *rest = grad(Z, ymin, ymax, x, y)

    aspect = np.degrees(np.arctan2(-Zx, -Zy))
    aspect = (aspect + 360) % 360
    return aspect

# Laplacian Curvature
def curvature(Z, ymin, ymax, x, y):
    Zx, Zy, dx_m, dy_m = grad(Z, ymin, ymax, x, y)

    _, Zxx = np.gradient(
        Zx,
        dy_m,
        dx_m
    )

    Zyy, _ = np.gradient(
        Zy,
        dy_m,
        dx_m
    )

    curvature = Zxx + Zyy
    return curvature

def roughness(Z, window_size = 11): 
    roughness = generic_filter(
        Z,
        np.std,
        size=window_size,
        mode="nearest"
    )

    return roughness

# Positive values indicate ridges.
# Negative values indicate valleys.
def tpi_small(Z, window_size = 21): 

    neighborhood_mean = uniform_filter(
        Z,
        size = window_size
    )

    tpi = Z - neighborhood_mean

    return tpi

def tpi_large(Z, window_size = 101): 
    
    neighborhood_mean = uniform_filter(
        Z,
        size = window_size
    )

    tpi = Z - neighborhood_mean

    return tpi

def distance_to_ridge(tpi, ridge_threshold = 20):

    ridge_mask = tpi > ridge_threshold

    distance = distance_transform_edt(
        ~ridge_mask
    )

    return distance

'''
northness:
+1 = north
0  = east/west
-1 = south
'''
def northness(aspect): 
    northness = np.cos(np.radians(aspect))
    return northness

'''
eastness:
+1 = east
0  = north/south
-1 = west
'''
def eastness(aspect): 
    eastness  = np.sin(np.radians(aspect))
    return eastness

def tri(Z): 

    def f(w): 
        center = w[4]
        neighbors = np.delete(w, 4)
        return np.sqrt(np.sum((neighbors - center)**2))
    
    return generic_filter(Z, f, size = 3)

def winter_radiation(Z, ymin, ymax, x, y): 
    slope_degrees = slope (Z, ymin, ymax, x, y)
    aspect_degrees = aspect(Z, ymin, ymax, x, y)

    latitude = np.radians((ymin + ymax) / 2)

    solar_azimuth = np.radains(180.0)
    solar_elevation = np.radians(22.0)

    slope_radians = np.radians(slope_degrees) # south
    aspect_radians = np.radians(aspect_degrees) # representative winter elevation

    radiation = (
        np.cos(slope_radians) * np.sin(solar_elevation)
        + np.sin(slope_radians)
        * np.cos(solar_elevation)
        * np.cos(solar_azimuth - aspect_radians)
    )

    radiation = np.maximum(radiation, 0)

    radiation /= radiation.max() + 1e-8

    return radiation
