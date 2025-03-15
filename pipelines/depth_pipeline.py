"""
Depth-based reconstruction pipeline for creating 3D models from images and videos.
"""
import os
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
import torch
from PIL import Image
import numpy as np
import diffusers
import open3d as o3d

from utils.depth_processing import process_depth_image, process_disparity_image
from utils.ply_utils import save_point_cloud_to_ply, apply_poisson_surface_reconstruction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DepthPipeline:
    """
    Pipeline for creating 3D models from depth maps generated from images or video frames.
    """
    
    def __init__(
        self, 
        depth_model_path: str = "prs-eth/marigold-depth-lcm-v1-0", 
        device: Optional[str] = None,
        use_fp16: bool = False
    ):
        """
        Initialize the depth pipeline.
        
        Args:
            depth_model_path (str): Path to the depth model
            device (str, optional): Device to use ('cuda' or 'cpu'). If None, will use CUDA if available.
            use_fp16 (bool): Whether to use half-precision (FP16)
        """
        self.depth_model_path = depth_model_path
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.use_fp16 = use_fp16 and self.device == "cuda"
        
        # The models will be loaded on demand
        self.depth_pipe = None
        self.disparity_pipe = None
        
    def load_models(self):
        """
        Load the depth and disparity models.
        
        Returns:
            bool: Whether models were loaded successfully
        """
        try:
            # Determine torch data type
            torch_dtype = torch.float16 if self.use_fp16 else torch.float32
            
            # Load the depth model
            logger.info(f"Loading depth model from {self.depth_model_path} on {self.device}")
            self.depth_pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
                self.depth_model_path, 
                prediction_type='depth', 
                torch_dtype=torch_dtype
            ).to(self.device)
            
            # Load the disparity model
            logger.info(f"Loading disparity model from {self.depth_model_path} on {self.device}")
            self.disparity_pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
                self.depth_model_path, 
                prediction_type='disparity', 
                torch_dtype=torch_dtype
            ).to(self.device)
            
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
            
    def process_images(
        self, 
        input_dir: str, 
        output_dir: Dict[str, str], 
        progress_callback=None
    ) -> bool:
        """
        Process images to generate depth and disparity maps.
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (dict): Dictionary of output directories
            progress_callback (callable, optional): Callback for progress updates
            
        Returns:
            bool: Success status
        """
        if self.depth_pipe is None or self.disparity_pipe is None:
            if not self.load_models():
                return False
                
        depth_output_dir = output_dir["depth_output_dir"]
        disparity_output_dir = output_dir["disparity_output_dir"]
        
        # Ensure output directories exist
        os.makedirs(depth_output_dir, exist_ok=True)
        os.makedirs(disparity_output_dir, exist_ok=True)
        
        # Get list of images to process
        image_files = [
            f for f in os.listdir(input_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ]
        
        if not image_files:
            logger.error(f"No image files found in {input_dir}")
            return False
            
        # Process each image
        total_files = len(image_files)
        start_time = time.time()
        
        for idx, filename in enumerate(image_files):
            input_path = os.path.join(input_dir, filename)
            depth_output_path = os.path.join(depth_output_dir, f"depth_{filename}")
            disparity_output_path = os.path.join(disparity_output_dir, f"disparity_{filename}")
            
            try:
                # Process depth image
                if not os.path.exists(depth_output_path):
                    process_depth_image(input_path, depth_output_path, self.depth_pipe)
                    
                # Process disparity image
                if not os.path.exists(disparity_output_path):
                    process_disparity_image(input_path, disparity_output_path, self.disparity_pipe)
                    
                # Calculate progress
                progress = (idx + 1) / total_files
                elapsed_time = time.time() - start_time
                avg_time_per_file = elapsed_time / (idx + 1)
                remaining_files = total_files - (idx + 1)
                est_remaining_time = remaining_files * avg_time_per_file
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(
                        progress=progress,
                        current=idx+1,
                        total=total_files,
                        remaining_time=est_remaining_time
                    )
                    
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                # Continue with next image instead of failing completely
                
        return True
        
    def generate_point_clouds(
        self, 
        image_dir: str, 
        ply_output_dir: str,
        progress_callback=None
    ) -> bool:
        """
        Generate point clouds from depth or disparity images.
        
        Args:
            image_dir (str): Directory containing depth/disparity images
            ply_output_dir (str): Directory to save point clouds
            progress_callback (callable, optional): Callback for progress updates
            
        Returns:
            bool: Success status
        """
        # Ensure output directory exists
        os.makedirs(ply_output_dir, exist_ok=True)
        
        # Get list of images to process
        image_files = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ]
        
        if not image_files:
            logger.error(f"No image files found in {image_dir}")
            return False
            
        # Process each image
        total_files = len(image_files)
        start_time = time.time()
        
        for idx, filename in enumerate(image_files):
            image_path = os.path.join(image_dir, filename)
            ply_output_path = os.path.join(ply_output_dir, f"{os.path.splitext(filename)[0]}.ply")
            
            try:
                # Generate point cloud
                save_point_cloud_to_ply(image_path, ply_output_path)
                
                # Calculate progress
                progress = (idx + 1) / total_files
                elapsed_time = time.time() - start_time
                avg_time_per_file = elapsed_time / (idx + 1)
                remaining_files = total_files - (idx + 1)
                est_remaining_time = remaining_files * avg_time_per_file
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(
                        progress=progress,
                        current=idx+1,
                        total=total_files,
                        remaining_time=est_remaining_time
                    )
                    
            except Exception as e:
                logger.error(f"Error generating point cloud for {filename}: {str(e)}")
                # Continue with next image instead of failing completely
                
        return True
        
    def generate_meshes(
        self, 
        ply_dir: str, 
        mesh_output_dir: str,
        progress_callback=None
    ) -> bool:
        """
        Generate meshes from point clouds using Poisson reconstruction.
        
        Args:
            ply_dir (str): Directory containing .ply point clouds
            mesh_output_dir (str): Directory to save meshes
            progress_callback (callable, optional): Callback for progress updates
            
        Returns:
            bool: Success status
        """
        # Ensure output directory exists
        os.makedirs(mesh_output_dir, exist_ok=True)
        
        # Get list of point clouds to process
        ply_files = [
            f for f in os.listdir(ply_dir) 
            if f.lower().endswith('.ply')
        ]
        
        if not ply_files:
            logger.error(f"No .ply files found in {ply_dir}")
            return False
            
        # Process each point cloud
        total_files = len(ply_files)
        start_time = time.time()
        
        for idx, filename in enumerate(ply_files):
            ply_path = os.path.join(ply_dir, filename)
            mesh_output_path = os.path.join(mesh_output_dir, f"{os.path.splitext(filename)[0]}_mesh.ply")
            
            try:
                # Generate mesh
                apply_poisson_surface_reconstruction(ply_path, mesh_output_path)
                
                # Calculate progress
                progress = (idx + 1) / total_files
                elapsed_time = time.time() - start_time
                avg_time_per_file = elapsed_time / (idx + 1)
                remaining_files = total_files - (idx + 1)
                est_remaining_time = remaining_files * avg_time_per_file
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(
                        progress=progress,
                        current=idx+1,
                        total=total_files,
                        remaining_time=est_remaining_time
                    )
                    
            except Exception as e:
                logger.error(f"Error generating mesh for {filename}: {str(e)}")
                # Continue with next point cloud instead of failing completely
                
        return True
        
    def run_pipeline(
        self, 
        input_dir: str, 
        output_dirs: Dict[str, str],
        mesh_from: str = "depth",  # Use "depth" or "disparity"
        progress_callback=None
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Run the full depth-based reconstruction pipeline.
        
        Args:
            input_dir (str): Directory containing input images
            output_dirs (dict): Dictionary of output directories
            mesh_from (str): Whether to generate meshes from depth or disparity images
            progress_callback (callable, optional): Callback for progress updates
            
        Returns:
            tuple: (success_status, generated_ply_files, generated_mesh_files)
        """
        if self.depth_pipe is None or self.disparity_pipe is None:
            logger.info("Loading models...")
            if not self.load_models():
                return False, [], []
        
        # Process images to generate depth and disparity maps
        logger.info("Generating depth and disparity maps...")
        if not self.process_images(input_dir, output_dirs, progress_callback):
            return False, [], []
            
        # Determine which images to use for point clouds
        if mesh_from.lower() == "depth":
            image_dir = output_dirs["depth_output_dir"]
        else:
            image_dir = output_dirs["disparity_output_dir"]
            
        # Generate point clouds from depth/disparity images
        logger.info(f"Generating point clouds from {mesh_from} images...")
        ply_dir = output_dirs["ply_output_dir"]
        if not self.generate_point_clouds(image_dir, ply_dir, progress_callback):
            return False, [], []
            
        # Generate meshes from point clouds
        logger.info("Generating meshes from point clouds...")
        mesh_dir = output_dirs["mesh_output_dir"]
        if not self.generate_meshes(ply_dir, mesh_dir, progress_callback):
            return False, [], []
            
        # Get list of generated files
        generated_ply_files = [os.path.join(ply_dir, f) for f in os.listdir(ply_dir) if f.endswith('.ply')]
        generated_mesh_files = [os.path.join(mesh_dir, f) for f in os.listdir(mesh_dir) if f.endswith('.ply')]
        
        return True, generated_ply_files, generated_mesh_files