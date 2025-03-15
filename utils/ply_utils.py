import cv2
import numpy as np
from pyntcloud import PyntCloud
import pandas as pd
from scipy.ndimage import gaussian_filter

def save_point_cloud_to_ply(depth_image_path, ply_filename):
    """
    Generate an optimized point cloud with controlled size and detail.
    """
    # Read depth image
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
    if depth_image is None:
        raise FileNotFoundError(f"Depth image {depth_image_path} not found")

    # Convert to float32 for better precision
    depth_image = depth_image.astype(np.float32)
    
    # Downsample the image to reduce points
    scale_factor = 0.5  # Adjust this value to control output size (0.5 = half size)
    new_size = (int(depth_image.shape[1] * scale_factor), 
                int(depth_image.shape[0] * scale_factor))
    depth_image = cv2.resize(depth_image, new_size, interpolation=cv2.INTER_AREA)
    
    # Apply depth thresholds
    min_depth = 20
    max_depth = 200
    depth_image = np.clip(depth_image, min_depth, max_depth)
    
    # Smooth depth values to reduce noise
    depth_image = gaussian_filter(depth_image, sigma=1.0)
    
    # Create coordinate grid
    height, width = depth_image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Apply conservative depth scaling
    depth_scale = 0.05  # Reduced from 0.1 for less distortion
    z = depth_image * depth_scale
    
    # Normalize coordinates to center
    x = (x - width/2) * depth_scale
    y = (y - height/2) * depth_scale
    
    # Combine coordinates
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    
    # Remove invalid points
    valid_mask = ~np.isnan(z) & (z != 0)
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]
    
    # Randomly sample points if too many
    max_points = 100000  # Adjust this value to control file size
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        x = x[indices]
        y = y[indices]
        z = z[indices]
    
    # Create DataFrame with minimal attributes
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z
    })
    
    # Add simple colors based on height (normalized z values)
    normalized_z = ((z - z.min()) / (z.max() - z.min()) * 255).astype(np.uint8)
    df['red'] = normalized_z
    df['green'] = normalized_z
    df['blue'] = normalized_z
    
    # Create and save point cloud
    cloud = PyntCloud(df)
    cloud.to_file(ply_filename)
    
    return cloud

def apply_poisson_surface_reconstruction(ply_file_path, output_mesh_file):
    """
    Simplified Poisson reconstruction for web-friendly meshes.
    """
    import open3d as o3d
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_file_path)
    
    # Estimate normals with conservative parameters
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Apply Poisson reconstruction with conservative parameters
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=7,  # Reduced depth for smaller file size
        scale=1.1,
        linear_fit=True
    )
    
    # Clean up the mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    
    # Save the mesh
    o3d.io.write_triangle_mesh(output_mesh_file, mesh)