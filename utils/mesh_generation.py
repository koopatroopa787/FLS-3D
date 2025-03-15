import open3d as o3d

def set_mesh_color(mesh, color):
    if not mesh.has_vertices():
        print("Mesh has no vertices to color.")
        return
    # Assign the specified color to all vertices
    colors = [color for _ in range(len(mesh.vertices))]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

def process_point_cloud(ply_file, output_mesh_file, color=(0.1, 0.1, 0.7)):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_file)

    # Downsample the point cloud (optional)
    pcd = pcd.voxel_down_sample(voxel_size=0.02)

    # Noise removal (optional)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Estimate normals for the point cloud
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(100)

    # Surface reconstruction using Poisson method
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    # Check if the mesh has vertices
    if not mesh.has_vertices():
        print(f"No vertices found in the generated mesh for {ply_file}. Skipping.")
        return

    mesh.compute_vertex_normals()

    # Set the mesh color (optional)
    set_mesh_color(mesh, color)

    # Save the mesh
    o3d.io.write_triangle_mesh(output_mesh_file, mesh, write_vertex_normals=True, write_vertex_colors=True)

    print(f"Processed and saved: {output_mesh_file}")
