import numpy as np
from numba import njit

# Rays should be of format: [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, r, g, b]
@njit
def init_camera_ray(i: int, j: int, fov: float, width: int, height: int, output_buffer: np.ndarray):

    # Convert degrees to radians
    fov = np.pi/180 * fov
    x = np.tan(fov/2) * (2.0*(i + np.random.random())/width - 1)
    y = np.tan(fov/2) * (height/width - 2.0*(j + np.random.random())/width)
    z = 1

    # Calculate magnitude
    magn = np.sqrt(x**2 + y**2 + z**2)

    # Origin is (0, 0, 0)  and colour is white
    output_buffer[0] = 0.0
    output_buffer[1] = 0.0
    output_buffer[2] = 0.0

    # Directions
    output_buffer[3] = x/magn
    output_buffer[4] = y/magn
    output_buffer[5] = z/magn

    # Colours
    output_buffer[6] = 1.0
    output_buffer[7] = 1.0
    output_buffer[8] = 1.0