from numba import njit, int64
from numba.typed import List
import numpy as np
from camera import *
from triangles import *
from bvh import *
from tqdm import tqdm

# Material should be of format: [r, g, b, luminance, reflectivity]
# Rays should be of format: [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, r, g, b]
# World would be a 2D array where each row represents a shape
# ID of 0 corresponds to environment color. that is, sky
VIP_INDICES = np.array([0, 1, 2])
@njit
def trace_ray_path(ray: np.ndarray,
                   world: np.ndarray,
                   point_hit: np.ndarray,
                   ray_buffer: np.ndarray,
                   bvh: np.ndarray,
                   triangle_indices: np.ndarray,
                   stack_ptr,
                   debug_num=10):

    closest_shape_idx = 0
    stack_ptr.append(0)
    min_dist = 1e9
    # Check box inetrsection
    while True:
        if len(stack_ptr) == 0:
            break
        box_id = stack_ptr.pop()
        hit, dist = box_intersection(bvh[box_id], ray=ray)
        idA = int(bvh[box_id, CHILDA])
        idB = int(bvh[box_id, CHILDB])

        if hit and ((idA != 0.0) or (idB != 0.0)):
            if idA != 0.0: stack_ptr.append(idA)
            if idB != 0.0: stack_ptr.append(idB)

        elif hit and ((idA == 0.0) and (idB == 0.0)):
            start_index = int(bvh[box_id, START_INDEX])
            end_index = int(bvh[box_id, COUNT] + start_index)
            if (bvh[box_id, COUNT] != 0.0) and (box_id != -1): 
                for idx in triangle_indices[start_index:end_index]:
                    intersection = calc_triangle_intersection(ray, world[idx+3], ray_buffer)
                    if (min_dist > intersection) and (intersection > 0.0):
                        min_dist = intersection
                        closest_shape_idx = idx + 3                            

    ox, oy, oz = ray[0], ray[1], ray[2]
    dx, dy, dz = ray[3], ray[4], ray[5]
    point_hit[0], point_hit[1], point_hit[2] = ox + dx*min_dist, oy + dy*min_dist, oz + dz*min_dist
    if closest_shape_idx != 0: return closest_shape_idx 

    closest_shape_idx = 0 
    min_dist = 1e9       
    for idx in VIP_INDICES:
        # Index 0 corresponds to sky
        if idx != 0:
            intersection = calc_triangle_intersection(ray, world[idx], ray_buffer)
            if intersection < 0.0:
                # The ray misses
                pass
            else:
                if min_dist > intersection:
                    min_dist = intersection
                    closest_shape_idx = idx
                else:
                    pass
    ox, oy, oz = ray[0], ray[1], ray[2]
    dx, dy, dz = ray[3], ray[4], ray[5]
    point_hit[0], point_hit[1], point_hit[2] = ox + dx*min_dist, oy + dy*min_dist, oz + dz*min_dist
    return closest_shape_idx
    
@njit
def get_random_dir(ray: np.ndarray, reflectivity: float, point: np.ndarray, triangle: np.ndarray, buffer: np.ndarray, ray_buffer: np.ndarray):
    NORMAL_BUFFER = 2
    
    # Get triangle normal
    triangle_normal(triangle=triangle, out_buffer=buffer[NORMAL_BUFFER])
    Nx, Ny, Nz = buffer[NORMAL_BUFFER, 0], buffer[NORMAL_BUFFER, 1], buffer[NORMAL_BUFFER, 2]
    point_x, point_y, point_z = point[0], point[1], point[2]
    dx, dy, dz = ray[3], ray[4], ray[5]

    # Sample random point on the hemitriangle
    u_1 = np.random.random()
    u_2 = np.random.random()

    Px, Py, Pz = np.cos(2*np.pi*u_1) * np.sqrt(1 - u_2**2), np.sin(2*np.pi*u_1) * np.sqrt(1 - u_2**2), u_2

    if abs(Nx) > 0.9: 
        Tx, Ty, Tz = 0.0, 1.0, 0.0
    else:
        Tx, Ty, Tz = 1.0, 0.0, 0.0

    Bx, By, Bz = cross3(Nx, Ny, Nz, Tx, Ty, Tz)
    P_world_x = Px * Tx + Py * Bx + Pz * Nx
    P_world_y = Px * Ty + Py * By + Pz * Ny
    P_world_z = Px * Tz + Py * Bz + Pz * Nz

    # Now, get the reflected direction
    dot_prod = dot3(dx, dy, dz, Nx, Ny, Nz)
    Rx = dx - 2 * dot_prod * Nx
    Ry = dy - 2 * dot_prod * Ny
    Rz = dz - 2 * dot_prod * Nz

    # Translate the sampled direction to the reflected direction
    sampled_x, sampled_y, sampled_z = P_world_x + (Rx - P_world_x)*reflectivity, P_world_y + (Ry - P_world_y)*reflectivity, P_world_z + (Rz - P_world_z)*reflectivity
    magnitude = magn(sampled_x, sampled_y, sampled_z) + 1e-6
    sampled_x, sampled_y, sampled_z = sampled_x/magnitude, sampled_y/magnitude, sampled_z/magnitude
    
    # Elevate the origin a bit in the direction of the normal
    Ox, Oy, Oz = point_x + Nx * 1e-5, point_y + Ny * 1e-5, point_z + Nz * 1e-5
    ray_buffer[0], ray_buffer[1], ray_buffer[2] = Ox, Oy, Oz
    ray_buffer[3], ray_buffer[4], ray_buffer[5] = sampled_x, sampled_y, sampled_z
    return dot3(sampled_x, sampled_y, sampled_z, Nx, Ny, Nz)

@njit
def is_directly_illuminated(point: np.ndarray, world: np.ndarray, shape_idx: int, source_idx: int, buffer: np.ndarray, shadow_ray: np.ndarray, bvh: np.ndarray, triangle_indices: np.ndarray, stack_ptr):
    NORMAL_BUFFER = 2
    RAY_BUFFER = 3
    POINT_BUFFER = 4
    RAY_BUFFER_2 = 8

    if shape_idx == source_idx:
        return True

    # First create a shadow ray from point to source body centre
    ax, ay, az, bx, by, bz, cx, cy, cz = world[source_idx, 1], world[source_idx, 2], world[source_idx, 3], world[source_idx, 4], world[source_idx, 5], world[source_idx, 6], world[source_idx, 7], world[source_idx, 8], world[source_idx, 9]
    Cx, Cy, Cz = (ax+bx+cx)/3, (ay+by+cy)/3, (az+bz+cz)/3
    # Add a small vector parallel to normal to the point to avoid double intersection
    triangle_normal(triangle=world[shape_idx], out_buffer=buffer[NORMAL_BUFFER])
    Ox, Oy, Oz = point[0] + buffer[NORMAL_BUFFER, 0]*1e-5, point[1] + buffer[NORMAL_BUFFER, 1]*1e-5, point[2] + buffer[NORMAL_BUFFER, 2]*1e-5
    
    # Direction to source
    Dx, Dy, Dz = Cx - Ox, Cy - Oy, Cz - Oz
    magnitude = magn(Dx, Dy, Dz)
    Dx, Dy, Dz = Dx/(magnitude+1e-6), Dy/(magnitude+1e-6), Dz/(magnitude+1e-6)

    # Make the ray
    buffer[RAY_BUFFER, 0], buffer[RAY_BUFFER, 1], buffer[RAY_BUFFER, 2] = Ox, Oy, Oz
    buffer[RAY_BUFFER, 3], buffer[RAY_BUFFER, 4], buffer[RAY_BUFFER, 5] = Dx, Dy, Dz

    # Get nearest intersection
    closest_idx = trace_ray_path(buffer[RAY_BUFFER], world=world, point_hit=buffer[POINT_BUFFER], ray_buffer=buffer[RAY_BUFFER_2], bvh=bvh, triangle_indices=triangle_indices, stack_ptr=stack_ptr)

    if closest_idx == source_idx:
        shadow_ray[0], shadow_ray[1], shadow_ray[2], shadow_ray[3], shadow_ray[4], shadow_ray[5] = buffer[RAY_BUFFER, 0], buffer[RAY_BUFFER, 1], buffer[RAY_BUFFER, 2], buffer[RAY_BUFFER, 3], buffer[RAY_BUFFER, 4], buffer[RAY_BUFFER, 5]
        return True
    else:
        return False
    
@njit
def get_illuminance(point: np.ndarray, shape_idx: int, source_idx: int, world: np.ndarray, materials: np.ndarray, buffer: np.ndarray, bvh: np.ndarray, triangle_indices: np.ndarray, stack_ptr) -> float:
    SHADOW_BUFFER = 5
    NORMAL_BUFFER = 6

    # First check if directly illuminated
    illuminated = is_directly_illuminated(point, world=world, shape_idx=shape_idx, source_idx=source_idx, buffer=buffer, shadow_ray=buffer[SHADOW_BUFFER], bvh=bvh, triangle_indices=triangle_indices, stack_ptr=stack_ptr)
    if illuminated:
        # Return illuminance directly if hit the source
        if source_idx == shape_idx:
            material_idx = int(world[source_idx, 0])
            return materials[material_idx, 3]

        # If illuminated, calculate the dot product between the shadow ray direction and normal
        Dx, Dy, Dz = buffer[SHADOW_BUFFER, 3], buffer[SHADOW_BUFFER, 4], buffer[SHADOW_BUFFER, 5]
        triangle_normal(triangle=world[shape_idx], out_buffer=buffer[NORMAL_BUFFER])
        Nx, Ny, Nz = buffer[NORMAL_BUFFER, 0], buffer[NORMAL_BUFFER, 1], buffer[NORMAL_BUFFER, 2]
        dot_prod = dot3(Dx, Dy, Dz, Nx, Ny, Nz)

        # If dot product is negative, it means the ray passes through the body, and no illuminance
        if dot_prod < 0:
            return 0.0
        else:
            # If dot product is positive, multiply the dot product with the source's illuminance
            material_idx = int(world[source_idx, 0])
            return dot_prod * materials[material_idx, 3]
    else:
        return 0.0

@njit
def trace(i: int, 
          j: int, 
          height: int, 
          width: int,
          fov: float, 
          num_rays: int,
          num_bounces: int,
          world: np.ndarray,
          source_idx: int, 
          materials: np.ndarray,
          buffer: np.ndarray, 
          pixel: np.ndarray,
          bvh: np.ndarray,
          triangle_indices: np.ndarray,
          stack_ptr,
          debug_num=10):
        RAY_BUFFER = 0
        POINT_BUFFER = 1
        RAY_BUFFER_2 = 8

        pixel[0] = 0.0 
        pixel[1] = 0.0
        pixel[2] = 0.0
        for _ in range(num_rays):
            init_camera_ray(i, j, fov=fov, width=width, height=height, output_buffer=buffer[RAY_BUFFER])
            bounce = num_bounces
            luminance = 0
            while bounce > 0:
                closest_shape_idx = trace_ray_path(ray=buffer[RAY_BUFFER], world=world, point_hit=buffer[POINT_BUFFER], ray_buffer=buffer[RAY_BUFFER_2], bvh=bvh, triangle_indices=triangle_indices, debug_num=debug_num, stack_ptr=stack_ptr)
                if closest_shape_idx == 0:
                    material_idx = int(world[0, 0])
                    luminance += materials[material_idx, 3]
                else:
                    luminance += get_illuminance(buffer[POINT_BUFFER], shape_idx=closest_shape_idx, source_idx=source_idx, world=world, materials=materials, buffer=buffer, bvh=bvh, triangle_indices=triangle_indices, stack_ptr=stack_ptr)

                material_idx = int(world[closest_shape_idx, 0])
                pixel[0] += buffer[RAY_BUFFER, 6] * luminance * materials[material_idx, 0]  
                pixel[1] += buffer[RAY_BUFFER, 7] * luminance * materials[material_idx, 1]
                pixel[2] += buffer[RAY_BUFFER, 8] * luminance * materials[material_idx, 2]

                # Break from this loop if hit the sky or the source
                if (closest_shape_idx == source_idx) or (closest_shape_idx == 0):
                    break

                # Get a new ray for bounce
                scaling_factor = get_random_dir(buffer[RAY_BUFFER], reflectivity=materials[material_idx, 4], point=buffer[POINT_BUFFER], triangle=world[closest_shape_idx], buffer=buffer, ray_buffer=buffer[RAY_BUFFER])

                buffer[RAY_BUFFER, 6] *= materials[material_idx, 0] * scaling_factor
                buffer[RAY_BUFFER, 7] *= materials[material_idx, 1] * scaling_factor
                buffer[RAY_BUFFER, 8] *= materials[material_idx, 2] * scaling_factor
                bounce -= 1
            #print("Luminance: ", luminance, "    \r")
        pixel[0] /= num_rays
        pixel[1] /= num_rays
        pixel[2] /= num_rays

def trace_rays(height: int,
               width: int,
               num_rays_per_pixel: int,
               num_bounces: int,
               world: np.ndarray,
               source_idx: int,
               materials: np.ndarray,
               fov: float, 
               buffer: np.ndarray,
               img_buffer: np.ndarray,
               bvh: np.ndarray,
               triangle_indices: np.ndarray,
               stack_ptr,
               debug_num=10):
    PIXEL_BUFFER = 7
    for i in tqdm(range(width), desc="Rendering..."):
        for j in range(height):
            debug_num -= 1
            trace(i=i,
                  j=j,
                  height=height,
                  width=width,
                  fov=fov,
                  num_rays=num_rays_per_pixel,
                  num_bounces=num_bounces,
                  world=world,
                  source_idx=source_idx,
                  materials=materials,
                  buffer=buffer,
                  pixel=buffer[PIXEL_BUFFER],
                  bvh=bvh,
                  triangle_indices=triangle_indices,
                  stack_ptr=stack_ptr,
                  debug_num=debug_num)
            img_buffer[i, j, 0], img_buffer[i, j, 1], img_buffer[i, j, 2] = buffer[PIXEL_BUFFER, 0], buffer[PIXEL_BUFFER, 1], buffer[PIXEL_BUFFER, 2]