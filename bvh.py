import numpy as np
from numba import njit
from utils import *
from tqdm import tqdm

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
        return True, t_near
    else:
        return False, -1.0
    
@njit
def axis_len(bvh):
    return abs(bvh[MAX_X] - bvh[MIN_X]), abs(bvh[MAX_Y] - bvh[MIN_Y]), abs(bvh[MAX_Z] - bvh[MIN_Z])
    
# Main work on BVH starts from here, God please wish me luck
@njit
def init_parent(parent_bvh, triangles: np.ndarray):
    init_box(parent_bvh)
    add_triangles(triangles, parent_bvh)

    # Checks would be done to ensure this is the parent node based on this -1.0 later in code (if neccassiry)
    parent_bvh[PARENT] = -1.0
    parent_bvh[COUNT] = triangles.shape[0]
    parent_bvh[START_INDEX] = 0.0

@njit 
def split(bvh, index, triangles, triangle_indices, childA_buffer, childB_buffer, buffer, arranged_indices):
    # Check longest bvh axis, and split from there
    split_x, split_y, split_z = 0.0, 0.0, 0.0
    len_x, len_y, len_z = axis_len(bvh)
    parent_index = index
    triangle_len = triangle_indices.shape[0]
    offset = bvh[START_INDEX]

    init_box(childA_buffer)
    init_box(childB_buffer)

    childA_buffer[PARENT] = parent_index
    childB_buffer[PARENT] = parent_index
    
    if (len_x >= len_y) and (len_x >= len_z):
        split_x = (bvh[MIN_X] + bvh[MAX_X])/2
        split_axis = 0
    elif (len_y >= len_x) and (len_y >= len_z):
        split_y = (bvh[MIN_Y] + bvh[MAX_Y])/2
        split_axis = 1
    else:
        split_z = (bvh[MIN_Z] + bvh[MAX_Z])/2
        split_axis = 2

    left_count = 0
    right_count = 0
    for i in triangle_indices:
        ax, ay, az, bx, by, bz, cx, cy, cz = triangles[i, 1], triangles[i, 2], triangles[i, 3], \
                                            triangles[i, 4], triangles[i, 5], triangles[i, 6], \
                                            triangles[i, 7], triangles[i, 8], triangles[i, 9]
        centre_x = (ax + bx + cx)/3
        centre_y = (ay + by + cy)/3
        centre_z = (az + bz + cz)/3
        is_left = False
        is_right = False
        if split_axis == 0:
            if centre_x >= split_x:
                add_triangle(childA_buffer, triangles[i])
                childA_buffer[COUNT] += 1.0
                is_left = True
                left_count += 1
            if centre_x < split_x:
                add_triangle(childB_buffer, triangles[i])
                childB_buffer[COUNT] += 1.0
                is_right = True
                right_count += 1

            if is_left:
                arranged_indices[left_count - 1] = i
            elif is_right:
                arranged_indices[triangle_len - right_count] = i
        
        elif split_axis == 1:
            if centre_y >= split_y:
                add_triangle(childA_buffer, triangles[i])
                childA_buffer[COUNT] += 1.0
                is_left = True
                left_count += 1
            if centre_y < split_y:
                add_triangle(childB_buffer, triangles[i])
                childB_buffer[COUNT] += 1.0
                is_right = True
                right_count += 1

            if is_left:
                arranged_indices[left_count - 1] = i
            elif is_right:
                arranged_indices[triangle_len - right_count] = i
        
        elif split_axis == 2:
            if centre_z >= split_z:
                add_triangle(childA_buffer, triangles[i])
                childA_buffer[COUNT] += 1.0
                is_left = True
                left_count += 1
            if centre_z < split_z:
                add_triangle(childB_buffer, triangles[i])
                childB_buffer[COUNT] += 1.0
                is_right = True
                right_count += 1

            if is_left:
                arranged_indices[left_count - 1] = i
            elif is_right:
                arranged_indices[triangle_len - right_count] = i

    childA_buffer[START_INDEX] = offset
    childB_buffer[START_INDEX] = offset + childA_buffer[COUNT]   

def run_split(triangles, max_iter, max_leaf_size=6):
    triangle_indices = np.array(range(triangles.shape[0]))
    bvh_buffer = np.zeros((11))

    bvhs = []

    # Init the parent
    init_parent(bvh_buffer, triangles)
    bvhs.append(bvh_buffer)
    i = 0
    for _ in tqdm(range(max_iter), desc="Initializing BVHs..."):
        bvh = bvhs[i]
        start_index = int(bvh[START_INDEX])
        end_index = int(start_index + bvh[COUNT])
        tri_indices = triangle_indices[start_index:end_index]
        arranged_indices = np.zeros_like(tri_indices)

        if bvh[COUNT] <= max_leaf_size:
            i += 1
            continue

        bvh_buffer_children = np.zeros((2, 11))
        split(bvh,
              index=i,
              triangles=triangles,
              triangle_indices=tri_indices,
              childA_buffer=bvh_buffer_children[0],
              childB_buffer=bvh_buffer_children[1],
              buffer=np.zeros_like(tri_indices),
              arranged_indices=arranged_indices)
        
        bvhs.append(bvh_buffer_children[0])
        bvhs.append(bvh_buffer_children[1])

        # Update the parent bvh children indices
        bvhs[i][CHILDA] = len(bvhs) - 2
        bvhs[i][CHILDB] = len(bvhs) - 1

        # Update the triangle indices
        triangle_indices[start_index:end_index] = arranged_indices
        i += 1

    return np.stack(bvhs), triangle_indices
