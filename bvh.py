import numpy as np
from numba import njit
from utils import *

# Bounding boxes to be of format [min_x, min_y, min_z, max_x, max_y, max_z, start_index, count,
#                                 childA_index, childB_index, parent_index]
# Size: 11

MIN_X = 0
MIN_Y = 1
MIN_Z = 2
MAX_X = 3
MAX_Y = 4
MAX_Z = 5
START_INDEX = 6
COUNT = 7
CHILDA = 8
CHILDB = 9
PARENT = 10

@njit
def init_box(buffer):
    buffer[MIN_X], buffer[MIN_Y], buffer[MIN_Z] = 1e9, 1e9, 1e9
    buffer[MAX_X], buffer[MAX_Y], buffer[MAX_Z] = -1e9, -1e9, -1e9
    buffer[START_INDEX] = 0.0
    buffer[COUNT] = 0.0
    buffer[CHILDA] = 0.0
    buffer[CHILDB] = 0.0
    buffer[PARENT] = 0.0

@njit
def add_triangle(box, triangle):
    ax, ay, az, bx, by, bz, cx, cy, cz = triangle[1], triangle[2], triangle[3], triangle[4], triangle[5], triangle[6], triangle[7], triangle[8], triangle[9]
    min_x, min_y, min_z, max_x, max_y, max_z = box[MIN_X], box[MIN_Y], box[MIN_Z], box[MAX_X], box[MAX_Y], box[MAX_Z]

    min_x_ = min(ax, bx, cx, min_x)
    min_y_ = min(ay, by, cy, min_y)
    min_z_ = min(az, bz, cz, min_z)

    max_x_ = max(ax, bx, cx, max_x)
    max_y_ = max(ay, by, cy, max_y)
    max_z_ = max(az, bz, cz, max_z)

    box[MIN_X], box[MIN_Y], box[MIN_Z], box[MAX_X], box[MAX_Y], box[MAX_Z] = min_x_, min_y_, min_z_, max_x_, max_y_, max_z_

@njit
def add_triangles(triangles, bvh_buffer):
    init_box(bvh_buffer)
    for triangle in triangles:
        add_triangle(bvh_buffer, triangle)

@njit
def box_intersection(box, ray):
    ox, oy, oz, dx, dy, dz = ray[0], ray[1], ray[2], ray[3], ray[4], ray[5]
    
    one_by_dx, one_by_dy, one_by_dz = 1/(dx+1e-9), 1/(dy+1e-9), 1/(dz+1e-9)

    t0x, t0y, t0z = (box[MIN_X] - ox)*one_by_dx, (box[MIN_Y] - oy)*one_by_dy, (box[MIN_Z] - oz)*one_by_dz
    t1x, t1y, t1z = (box[MAX_X] - ox)*one_by_dx, (box[MAX_Y] - oy)*one_by_dy, (box[MAX_Z] - oz)*one_by_dz

    t_near = max(min(t0x, t1x), 
                 min(t0y, t1y), 
                 min(t0z, t1z))
    t_far = min(max(t0x, t1x), 
                max(t0y, t1y), 
                max(t0z, t1z))

    if (t_near <= t_far) and (t_far >= 1e-9):
        return True
    else:
        return False 
    
