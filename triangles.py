import numpy as np
from numba import njit
from utils import *

# Rays should be of format: [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, r, g, b]
@njit
def calc_triangle_intersection(ray: np.ndarray, triangle: np.ndarray, ray_buffer: np.ndarray):
    # Coords
    ax, ay, az, bx, by, bz, cx, cy, cz = triangle[1], triangle[2], triangle[3], triangle[4], triangle[5], triangle[6], triangle[7], triangle[8], triangle[9]
    # Ray info
    ox, oy, oz, dx, dy, dz = ray[0], ray[1], ray[2], ray[3], ray[4], ray[5]

    # Triangle normal
    triangle_normal(triangle, ray_buffer)
    nx, ny, nz = ray_buffer[0], ray_buffer[1], ray_buffer[2]

    # Check if parallel
    dot_prod = dot3(dx, dy, dz, nx, ny, nz)
    if abs(dot_prod) <= 10e-4:
        return -1.0
    
    # Check plane intersection
    a_minus_o_x, a_minus_o_y, a_minus_o_z = ax - ox, ay - oy, az - oz
    numerator = dot3(a_minus_o_x, a_minus_o_y, a_minus_o_z, nx, ny, nz)
    denominator = dot3(dx, dy, dz, nx, ny, nz)
    t = numerator/(denominator+1e-7)

    if t <= 0:
        return -1.0
    
    # Check if o + dt lies inside the triangle
    px, py, pz = ox + dx*t, oy + dy*t, oz + dz*t
    e1x, e1y, e1z = bx - ax, by - ay, bz - az
    e2x, e2y, e2z = cx - bx, cy - by, cz - bz
    e3x, e3y, e3z = ax - cx, ay - cy, az - cz

    p1x, p1y, p1z = px - ax, py - ay, pz - az
    p2x, p2y, p2z = px - bx, py - by, pz - bz
    p3x, p3y, p3z = px - cx, py - cy, pz - cz

    c1x, c1y, c1z = cross3(e1x, e1y, e1z, p1x, p1y, p1z)
    c2x, c2y, c2z = cross3(e2x, e2y, e2z, p2x, p2y, p2z)
    c3x, c3y, c3z = cross3(e3x, e3y, e3z, p3x, p3y, p3z)

    dot_prod_1 = dot3(c1x, c1y, c1z, nx, ny, nz)
    dot_prod_2 = dot3(c2x, c2y, c2z, nx, ny, nz)
    dot_prod_3 = dot3(c3x, c3y, c3z, nx, ny, nz)

    if (dot_prod_1 >= -1e-9 and dot_prod_2 >= -1e-9 and dot_prod_3 >= -1e-9) or (dot_prod_1 <= 1e-9 and dot_prod_2 <= 1e-9 and dot_prod_3 <= 1e-9):
        return t
    else:
        return -1.0

# It is to be ensured that point lies on the triangle
@njit
def triangle_normal(triangle: np.ndarray, out_buffer: np.ndarray):
    out_buffer[0], out_buffer[1], out_buffer[2] = triangle[10], triangle[11], triangle[12]