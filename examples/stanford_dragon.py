from tracer import *
from renderer import *
import numpy as np
from world import *
from bvh import *

materials = np.array([
    [0.53, 0.81, 0.92, 10.0, 0.0],
    [1.0, 1.0, 1.0, 1000.0, 0.0],
    [0.95, 0.45, 0.45, 0.1, 0.75],
    [0.99, 0.99, 0.99, 100.0, 1.0]
])

# Thanks to hackmans for this low poly stanford dragon model
# You can find the model at https://sketchfab.com/3d-models/stanford-dragon-pbr-5d610f842a4542ccb21613d41bbd7ea1
# Model is licensed under CC Attribution
dragon = Object("resources\stanford_dragon_pbr.obj", name="Dragon")
dragon.scale(2/100)
dragon.translate(0, -0.5, 5)
dragon.rotate(theta=-np.pi/6, phi=0)
world = dragon.load_world(2)

dragon_world = world

world = np.concatenate([np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                        np.array([[1.0, 10.0, 10.0, -10.0, 11.0, 10.0, -11.0, 12.0, 12.0, -12.0, 1.0, 0.0, 0.0]]),
                        np.array([[3.0, 0.0, -0.5, -100.0, 100.0, -0.5, 100.0, -100.0, -0.5, 100.0, 0.0, 1.0, 0.0]]),
                       world], 
                        axis=0)

bvhs, triangle_indices = run_split(dragon_world, 1000)

img = trace_rays(height=1080, 
           width=1920, 
           num_rays_per_pixel=32, 
           num_bounces=5,
           world=world,
           source_idx=1,
           materials=materials, 
           fov=60,
           bvh=bvhs,
           triangle_indices=triangle_indices,
           )
render(img, filepath="bin/stanford_dragon.png", use_reducer=True)