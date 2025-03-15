"""
Predefined COLMAP command templates for different reconstruction scenarios.
"""
import os
from typing import Dict, List, Any

def get_frame_extraction_command(video_path: str, fps: float = 10) -> str:
    """
    Generate a command to extract frames from a video.
    
    Args:
        video_path (str): Path to the video file
        fps (float): Frames per second to extract
        
    Returns:
        str: FFMPEG command
    """
    frames_dir = "frames"
    return f'ffmpeg -i "{video_path}" -vf "fps={fps}" {frames_dir}/frame_%04d.jpg'

def get_colmap_commands(
    video_metadata: Dict[str, Any], 
    user_choices: Dict[str, Any],
    working_dir: str
) -> List[str]:
    """
    Generate optimized COLMAP commands based on video metadata and user choices.
    
    Args:
        video_metadata (dict): Video metadata from FFMPEG
        user_choices (dict): User-selected options
        working_dir (str): Working directory
        
    Returns:
        list: COLMAP commands
    """
    # Setup directory structure
    frames_dir = os.path.join(working_dir, "frames")
    sparse_dir = os.path.join(working_dir, "sparse")
    database_path = os.path.join(working_dir, "database.db")
    
    # Get motion type and scene type
    motion_type = user_choices.get("motion_type", "linear")
    scene_type = user_choices.get("scene_type", "object")
    
    # Determine optimal parameters based on inputs
    commands = []
    
    # 1. Create necessary directories
    commands.append(f"mkdir -p {frames_dir}")
    commands.append(f"mkdir -p {sparse_dir}")
    
    # 2. Extract frames if needed (already handled in our pipeline)
    fps = determine_optimal_fps(motion_type, video_metadata)
    if user_choices.get("extract_frames", True):
        video_path = user_choices.get("video_path", "input.mp4")
        commands.append(get_frame_extraction_command(video_path, fps))
    
    # 3. Feature extraction with optimized parameters
    sift_params = get_sift_extraction_params(scene_type, motion_type)
    commands.append(
        f"colmap feature_extractor "
        f"--database_path {database_path} "
        f"--image_path {frames_dir} "
        f"{sift_params}"
    )
    
    # 4. Feature matching with optimized parameters
    matcher_type, matcher_params = get_matcher_params(motion_type, scene_type)
    commands.append(
        f"colmap {matcher_type}_matcher "
        f"--database_path {database_path} "
        f"{matcher_params}"
    )
    
    # 5. Mapping (SfM) with optimized parameters
    mapper_params = get_mapper_params(scene_type, motion_type)
    commands.append(
        f"colmap mapper "
        f"--database_path {database_path} "
        f"--image_path {frames_dir} "
        f"--output_path {sparse_dir} "
        f"{mapper_params}"
    )
    
    # 6. Convert model to PLY format
    commands.append(
        f"colmap model_converter "
        f"--input_path {os.path.join(sparse_dir, '0')} "
        f"--output_path {os.path.join(working_dir, 'model.ply')} "
        f"--output_type PLY"
    )
    
    # 7. Optional: Dense reconstruction (if requested)
    if user_choices.get("dense_reconstruction", False):
        dense_dir = os.path.join(working_dir, "dense")
        commands.append(f"mkdir -p {dense_dir}")
        
        # Image undistorter
        commands.append(
            f"colmap image_undistorter "
            f"--image_path {frames_dir} "
            f"--input_path {os.path.join(sparse_dir, '0')} "
            f"--output_path {dense_dir} "
            f"--output_type COLMAP"
        )
        
        # Patch match stereo
        commands.append(
            f"colmap patch_match_stereo "
            f"--workspace_path {dense_dir}"
        )
        
        # Stereo fusion
        commands.append(
            f"colmap stereo_fusion "
            f"--workspace_path {dense_dir} "
            f"--output_path {os.path.join(dense_dir, 'fused.ply')}"
        )
        
        # Meshing (optional)
        if user_choices.get("generate_mesh", False):
            commands.append(
                f"colmap poisson_mesher "
                f"--input_path {os.path.join(dense_dir, 'fused.ply')} "
                f"--output_path {os.path.join(dense_dir, 'mesh.ply')}"
            )
    
    return commands

def determine_optimal_fps(motion_type: str, video_metadata: Dict[str, Any]) -> float:
    """
    Determine the optimal frame extraction rate based on motion type.
    
    Args:
        motion_type (str): Type of camera motion
        video_metadata (dict): Video metadata
        
    Returns:
        float: Optimal frames per second
    """
    original_fps = video_metadata.get('fps', 30)
    duration = video_metadata.get('duration', 10)
    
    # For short videos (< 5 seconds), use higher sampling rate
    if duration < 5:
        return min(original_fps, 15)
    
    # For different motion types
    if motion_type == "stationary":
        # Stationary needs fewer frames
        return min(original_fps / 6, 5)
    elif motion_type == "linear":
        # Linear motion can use moderate frame rate
        return min(original_fps / 3, 10)
    elif motion_type == "circular":
        # Circular motion needs more frames for complete coverage
        return min(original_fps / 2, 15)
    elif motion_type == "complex" or motion_type == "handheld":
        # Complex motions need more frames
        return min(original_fps / 1.5, 20)
    else:
        # Default case
        return min(original_fps / 3, 10)

def get_sift_extraction_params(scene_type: str, motion_type: str) -> str:
    """
    Get optimized SIFT extraction parameters based on scene type.
    
    Args:
        scene_type (str): Type of scene
        motion_type (str): Type of camera motion
        
    Returns:
        str: SIFT extraction parameters
    """
    # Common parameters
    params = "--SiftExtraction.use_gpu 1 "
    
    # Determine camera model
    camera_model = "SIMPLE_PINHOLE"
    if motion_type in ["complex", "handheld"]:
        camera_model = "RADIAL"
    
    params += f"--ImageReader.camera_model {camera_model} "
    
    # Adjust parameters based on scene type
    if scene_type == "indoor" or scene_type == "object":
        params += "--SiftExtraction.max_image_size 2048 "
    else:
        params += "--SiftExtraction.max_image_size 3200 "
    
    # Add number of threads
    params += "--SiftExtraction.num_threads 8 "
    
    return params

def get_matcher_params(motion_type: str, scene_type: str) -> tuple:
    """
    Get optimized matcher type and parameters based on motion type.
    
    Args:
        motion_type (str): Type of camera motion
        scene_type (str): Type of scene
        
    Returns:
        tuple: (matcher_type, matcher_parameters)
    """
    # For sequential motion (linear, handheld with forward movement)
    if motion_type in ["linear", "handheld"]:
        return "sequential", "--Sequential.overlap 20 --SiftMatching.use_gpu 1"
    
    # For circular motion (orbiting an object, 360 captures)
    elif motion_type == "circular":
        return "sequential", "--Sequential.overlap 30 --Sequential.loop_detection 1 --SiftMatching.use_gpu 1"
    
    # For stationary with small movement or complex scenes
    elif motion_type == "stationary" or scene_type in ["indoor", "architecture"]:
        return "exhaustive", "--SiftMatching.use_gpu 1"
    
    # Default to exhaustive (most comprehensive but slowest)
    else:
        return "exhaustive", "--SiftMatching.use_gpu 1"

def get_mapper_params(scene_type: str, motion_type: str) -> str:
    """
    Get optimized mapper parameters based on scene type.
    
    Args:
        scene_type (str): Type of scene
        motion_type (str): Type of camera motion
        
    Returns:
        str: Mapper parameters
    """
    params = ""
    
    # Basic parameters
    params += "--Mapper.num_threads 8 "
    
    # For architecture, try to adjust for scale
    if scene_type == "architecture":
        params += "--Mapper.ba_refine_focal_length 1 "
        params += "--Mapper.ba_refine_principal_point 0 "
    else:
        params += "--Mapper.ba_refine_focal_length 0 "
        params += "--Mapper.ba_refine_principal_point 0 "
    
    # For handheld or complex motion, enable more refinement
    if motion_type in ["handheld", "complex"]:
        params += "--Mapper.ba_local_max_refinements 5 "
        params += "--Mapper.ba_local_max_num_iterations 50 "
    else:
        params += "--Mapper.ba_local_max_refinements 3 "
        params += "--Mapper.ba_local_max_num_iterations 30 "
    
    return params