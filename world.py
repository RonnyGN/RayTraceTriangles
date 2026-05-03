import numpy as np
from tqdm import tqdm

class Object:

    def __init__(self, filepath, name="Object"):
        self.path = filepath
        self.translate_coords = np.zeros((3))
        self.rotation_matrix = np.eye(3)
        self.scale_factor = 1.0
        self.name = name

    def load_world(self, material_id):
        vertices = []
        triangles = []
        with open(self.path, 'r') as f:
            for line in tqdm(f.readlines(), desc=f"Loading object {self.name}..."):
                line = line.split(' ')

                # Check if first letter is v, else discard
                if line[0] == 'v':
                    vertex = np.array([float(i) for i in line[1:]])

                    vertex = vertex @ self.rotation_matrix.T
                    vertex *= self.scale_factor
                    vertex += self.translate_coords
                    
                    vertices.append(vertex)
                elif line[0] == 'f':
                    faces = np.array([int(i.split('/')[0])-1 for i in line[1:]])
                    fx, fy, fz = faces

                    triangle = np.zeros((10))
                    triangle[0] = material_id
                    triangle[1:] = np.hstack((vertices[fx], vertices[fy], vertices[fz]))
                    triangles.append(triangle)

        return np.stack(triangles)
    
    def translate(self, x, y, z):
        self.translate_coords = self.translate_coords + np.array([x, y, z])

    def rotate(self, theta, phi):
        rot_matrix = np.array([
            [np.cos(phi)*np.cos(theta), -np.sin(phi), np.cos(phi)*np.sin(theta)],
            [np.sin(phi)*np.cos(theta), np.cos(phi), np.sin(phi)*np.sin(theta)],
            [-np.sin(theta), 0.0, np.cos(theta)]
        ])
        self.rotation_matrix = rot_matrix @ self.rotation_matrix
    
    def scale(self, factor):
        self.scale_factor *= factor
                    
                    