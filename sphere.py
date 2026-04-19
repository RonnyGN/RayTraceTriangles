import numpy as np
from numba import njit
from utils import *

# Rays should be of format: [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, r, g, b]
# Sphere should be of format: [material_id, position_x, position_y, position_z, radius]
@njit
def calc_sphere_intersection(ray: np.ndarray, sphere: np.ndarray):
    # Offset
    Lx, Ly, Lz = ray[0] - sphere[1], ray[1] - sphere[2], ray[2] - sphere[3]
    # Directions
    dx, dy, dz = ray[3], ray[4], ray[5]
    # Radius
    R = sphere[4]
    # Determinant
    D = 4 * (dot3(dx, dy, dz, Lx, Ly, Lz)**2 - magn(Lx, Ly, Lz)**2 + R**2)

    # Check determinant's sign
    if D < 0:
        # Return -1.0 if nothing is hit
        return -1.0
    elif D == 0:
        return -dot3(dx, dy, dz, Lx, Ly, Lz)
    elif D > 0:
        return (-2 * dot3(dx, dy, dz, Lx, Ly, Lz) - np.sqrt(D))/2
    
# It is to be ensured that point lies on the sphere
@njit
def sphere_normal(point: np.ndarray, sphere: np.ndarray, out_buffer: np.ndarray):
    # Centre
    Px, Py, Pz = sphere[1], sphere[2], sphere[3]
    # Offstes
    ox, oy, oz = point[0] - Px, point[1] - Py, point[2] - Pz
    # Normalized
    r = sphere[4]
    ox, oy, oz = ox/r, oy/r, oz/r
    
    out_buffer[0], out_buffer[1], out_buffer[2] = ox, oy, oz