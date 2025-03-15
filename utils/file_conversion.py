import open3d as o3d
import numpy as np
import os

# Function to convert .ply to .xyz
def convert_ply_to_xyz(ply_file, output_xyz):
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)
    np.savetxt(output_xyz, points, fmt='%f %f %f')
    print(f"XYZ file saved at: {output_xyz}")

# Function to convert .ply to .obj
def convert_ply_to_obj(ply_file, output_obj):
    mesh = o3d.io.read_triangle_mesh(ply_file)
    o3d.io.write_triangle_mesh(output_obj, mesh)
    print(f"OBJ file saved at: {output_obj}")

# Function to convert .ply to .stl
def convert_ply_to_stl(ply_file, output_stl):
    mesh = o3d.io.read_triangle_mesh(ply_file)
    o3d.io.write_triangle_mesh(output_stl, mesh)
    print(f"STL file saved at: {output_stl}")
