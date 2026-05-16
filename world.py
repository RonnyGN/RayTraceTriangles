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
        normals = []
        with open(self.path, 'r') as f:
            for line in tqdm(f.readlines(), desc=f"Loading object {self.name}..."):
                line = line.split(' ')

                # Check if first letter is v/vn/f, else discard
                if line[0] == 'v':
                    vertex = np.array([float(i) for i in line[1:]])

                    vertex = vertex @ self.rotation_matrix.T
                    vertex *= self.scale_factor
                    vertex += self.translate_coords
                    
                    vertices.append(vertex)

                elif line[0] == 'f':
                    face_parts = [p.split('/') for p in line[1:]]
                    v_idxs = [int(p[0]) - 1 for p in face_parts]
                    
                    v0 = vertices[v_idxs[0]]
                    v1 = vertices[v_idxs[1]]
                    v2 = vertices[v_idxs[2]]

                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    
                    nx = edge1[1] * edge2[2] - edge1[2] * edge2[1]
                    ny = edge1[2] * edge2[0] - edge1[0] * edge2[2]
                    nz = edge1[0] * edge2[1] - edge1[1] * edge2[0]
                    
                    mag = (nx**2 + ny**2 + nz**2)**0.5
                    if mag > 0:
                        nx /= mag
                        ny /= mag
                        nz /= mag

                    triangle = np.zeros(13)
                    triangle[0] = material_id

                    triangle[1:4] = v0
                    triangle[4:7] = v1
                    triangle[7:10] = v2

                    triangle[10], triangle[11], triangle[12] = nx, ny, nz
                    
                    triangles.append(triangle)

        return np.stack(triangles)
    
    def translate(self, x, y, z):
        self.translate_coords = self.translate_coords + np.array([x, y, z])

    # Theta rotates along the y axis, and phi along the z axis. Phi would generally be not useful, similarly
    # Very niche applications require rotation along the x axis
    def rotate(self, theta, phi):
        rot_matrix = np.array([
            [np.cos(phi)*np.cos(theta), -np.sin(phi), np.cos(phi)*np.sin(theta)],
            [np.sin(phi)*np.cos(theta), np.cos(phi), np.sin(phi)*np.sin(theta)],
            [-np.sin(theta), 0.0, np.cos(theta)]
        ])
        self.rotation_matrix = rot_matrix @ self.rotation_matrix
    
    def scale(self, factor):
        self.scale_factor *= factor                   