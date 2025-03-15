"""
Utilities for working with COLMAP for Structure from Motion reconstruction.
"""
import os
import subprocess
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, working_dir=None, verbose=True):
    """
    Execute a shell command and optionally display output.
    
    Args:
        command (str): Command to execute
        working_dir (str, optional): Directory to run command in
        verbose (bool): Whether to print output
        
    Returns:
        tuple: (return_code, stdout, stderr)
    """
    try:
        # Skip empty commands and comments
        command = command.strip()
        if not command or command.startswith("#"):
            return 0, "", ""
        
        # If command starts with 'colmap', use the COLMAP_EXECUTABLE from config
        if command.strip().startswith('colmap'):
            # Import here to avoid circular imports
            import sys
            # Add the project root to sys.path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
                
            from config import COLMAP_EXECUTABLE
            
            # Replace just the beginning 'colmap' with the full path
            colmap_cmd = COLMAP_EXECUTABLE.replace("\\", "/")  # Normalize path
            command = command.replace('colmap', f'"{colmap_cmd}"', 1)
            
            logger.info(f"Using COLMAP at: {COLMAP_EXECUTABLE}")
            logger.info(f"Modified command: {command}")
            
        # For mkdir commands, use Python's os.makedirs instead
        if command.startswith("mkdir"):
            parts = command.split()
            if len(parts) >= 2:
                dir_path = parts[-1]
                # Create directory with Python
                if working_dir:
                    dir_path = os.path.join(working_dir, dir_path)
                os.makedirs(dir_path, exist_ok=True)
                if verbose:
                    logger.info(f"Created directory: {dir_path}")
                return 0, f"Created directory: {dir_path}", ""
        
        if verbose:
            logger.info(f"Executing command: {command}")
        
        # Set up environment variables for subprocess
        env = os.environ.copy()
        
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=working_dir,
            env=env
        )
        
        # Capture output in real-time
        stdout_lines = []
        stderr_lines = []
        
        # Read stdout and stderr simultaneously
        while True:
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()
            
            if stdout_line == '' and stderr_line == '' and process.poll() is not None:
                break
                
            if stdout_line:
                if verbose:
                    logger.info(stdout_line.rstrip())
                stdout_lines.append(stdout_line)
                
            if stderr_line:
                if process.poll() != 0 and verbose:
                    logger.error(stderr_line.rstrip())
                elif verbose:
                    logger.info(stderr_line.rstrip())
                stderr_lines.append(stderr_line)
        
        # Get return code and combine output
        returncode = process.poll()
        stdout = ''.join(stdout_lines)
        stderr = ''.join(stderr_lines)
        
        return returncode, stdout, stderr
        
    except Exception as e:
        logger.error(f"Error running command: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return -1, "", str(e)

def setup_colmap_workspace(output_dir, image_dir):
    """
    Setup a workspace for COLMAP processing.
    
    Args:
        output_dir (str): The main output directory
        image_dir (str): Directory containing input images
        
    Returns:
        dict: Paths to COLMAP working directories
    """
    # Create necessary subdirectories
    colmap_dir = os.path.join(output_dir, "colmap")
    sparse_dir = os.path.join(colmap_dir, "sparse")
    dense_dir = os.path.join(colmap_dir, "dense")
    database_path = os.path.join(colmap_dir, "database.db")
    
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(dense_dir, exist_ok=True)
    
    return {
        "colmap_dir": colmap_dir,
        "sparse_dir": sparse_dir,
        "dense_dir": dense_dir,
        "database_path": database_path,
        "image_dir": image_dir
    }

def extract_colmap_video_metadata(video_path):
    """
    Extract metadata from a video file using FFMPEG to help with COLMAP parameters.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Video metadata
    """
    command = f'ffprobe -v quiet -print_format json -show_format -show_streams "{video_path}"'
    returncode, stdout, stderr = run_command(command, verbose=False)
    
    if returncode != 0:
        logger.error(f"Failed to extract video metadata: {stderr}")
        return {}
    
    try:
        metadata = json.loads(stdout)
        
        # Extract useful information
        result = {}
        
        # Get video properties
        for stream in metadata.get('streams', []):
            if stream.get('codec_type') == 'video':
                result['width'] = stream.get('width', 0)
                result['height'] = stream.get('height', 0)
                result['fps'] = eval(stream.get('r_frame_rate', '0/1'))
                result['duration'] = float(stream.get('duration', 0))
                result['total_frames'] = int(float(stream.get('nb_frames', 0)))
                break
        
        return result
    except json.JSONDecodeError:
        logger.error("Failed to parse video metadata JSON")
        return {}
    except Exception as e:
        logger.error(f"Error processing video metadata: {str(e)}")
        return {}

def run_colmap_feature_extraction(colmap_paths, camera_model="SIMPLE_PINHOLE"):
    """
    Run COLMAP feature extraction.
    
    Args:
        colmap_paths (dict): Paths to COLMAP directories
        camera_model (str): Camera model to use
        
    Returns:
        bool: Success status
    """
    command = (
        f"colmap feature_extractor "
        f"--database_path {colmap_paths['database_path']} "
        f"--image_path {colmap_paths['image_dir']} "
        f"--ImageReader.camera_model {camera_model} "
        f"--SiftExtraction.use_gpu 1"
    )
    
    returncode, _, _ = run_command(command)
    return returncode == 0

def run_colmap_matcher(colmap_paths, matcher_type="exhaustive"):
    """
    Run COLMAP matching.
    
    Args:
        colmap_paths (dict): Paths to COLMAP directories
        matcher_type (str): Type of matcher to use
        
    Returns:
        bool: Success status
    """
    if matcher_type not in ["exhaustive", "sequential", "vocabulary_tree"]:
        logger.warning(f"Unknown matcher type {matcher_type}, falling back to exhaustive")
        matcher_type = "exhaustive"
    
    command = (
        f"colmap {matcher_type}_matcher "
        f"--database_path {colmap_paths['database_path']} "
        f"--SiftMatching.use_gpu 1"
    )
    
    returncode, _, _ = run_command(command)
    return returncode == 0

def run_colmap_mapper(colmap_paths):
    """
    Run COLMAP mapping to create sparse reconstruction.
    
    Args:
        colmap_paths (dict): Paths to COLMAP directories
        
    Returns:
        bool: Success status
    """
    command = (
        f"colmap mapper "
        f"--database_path {colmap_paths['database_path']} "
        f"--image_path {colmap_paths['image_dir']} "
        f"--output_path {colmap_paths['sparse_dir']}"
    )
    
    returncode, _, _ = run_command(command)
    return returncode == 0

def run_colmap_automatic_reconstructor(output_dir, image_dir, quality="high"):
    """
    Run COLMAP automatic reconstruction pipeline.
    
    Args:
        output_dir (str): Output directory for COLMAP results
        image_dir (str): Directory containing input images
        quality (str): Quality level (high, medium, low)
        
    Returns:
        bool: Success status
    """
    quality_options = {
        "high": "--quality high",
        "medium": "--quality medium",
        "low": "--quality low"
    }
    
    quality_str = quality_options.get(quality, "--quality medium")
    
    command = (
        f"colmap automatic_reconstructor "
        f"--workspace_path {output_dir} "
        f"--image_path {image_dir} "
        f"{quality_str} "
        f"--use_gpu 1"
    )
    
    returncode, _, _ = run_command(command)
    return returncode == 0

def run_colmap_pipeline_from_commands(commands, working_dir, video_path=None):
    """
    Run a series of COLMAP commands from a command list.
    
    Args:
        commands (list): List of COLMAP command strings to execute
        working_dir (str): Working directory for command execution
        video_path (str, optional): Path to the actual video file
        
    Returns:
        bool: Overall success status
    """
    success = True
    
    # Preprocess commands to use correct file paths
    processed_commands = []
    for command in commands:
        # If a command references a generic input video, replace with the actual path
        if "input.mp4" in command and video_path:
            command = command.replace("input.mp4", f'"{video_path}"')
            
        # If a command references generic frame output, ensure it goes to the frames directory
        if "frames_" in command and "frames/" not in command:
            frames_dir = os.path.join(working_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            command = command.replace("frames_", f"{frames_dir}/frames_")
            
        processed_commands.append(command)
    
    for i, command in enumerate(processed_commands):
        logger.info(f"Running command {i+1}/{len(processed_commands)}: {command}")
        returncode, stdout, stderr = run_command(command, working_dir=working_dir)
        
        if returncode != 0:
            logger.error(f"Command failed: {command}")
            success = False
            break
    
    return success

def check_colmap_installation():
    """
    Check if COLMAP is installed and available in the system path.
    
    Returns:
        bool: Whether COLMAP is available
    """
    try:
        # Import here to avoid circular imports
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import COLMAP_EXECUTABLE
        
        result = subprocess.run(
            [COLMAP_EXECUTABLE, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error checking COLMAP installation: {e}")
        return False