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