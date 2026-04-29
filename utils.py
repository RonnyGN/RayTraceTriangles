import numpy as np
from numba import njit

@njit
def dot3(a1x, a1y, a1z, a2x, a2y, a2z):
    return a1x * a2x + a1y * a2y + a1z * a2z

@njit 
def magn(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

@njit 
def cross3(ax, ay, az, bx, by, bz):
    x = ay * bz - az * by
    y = az * bx - ax * bz
    z = ax * by - ay * bx
    return x, y, z

@njit
def det2(a00, a01, a10, a11):
    return a00 * a11 - a10 * a01

@njit
def det3(matrix: np.ndarray):
    i_det2 = matrix[0, 0] * det2(matrix[1, 1], matrix[1, 2], matrix[2, 1], matrix[2, 2])
    j_det2 = matrix[0, 1] * det2(matrix[1, 0], matrix[1, 2], matrix[2, 0], matrix[2, 2])
    k_det2 = matrix[0, 2] * det2(matrix[1, 0], matrix[1, 1], matrix[2, 0], matrix[2, 1])
    return i_det2 - j_det2 + k_det2