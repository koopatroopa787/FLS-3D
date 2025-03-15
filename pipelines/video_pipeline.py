"""
Video pipeline for 3D reconstruction using COLMAP and Mistral API.
"""
import os
import time
import logging
from typing import Dict, Any, Optional, Tuple

from utils.video_processing import extract_frames
from utils.colmap_utils import (
    extract_colmap_video_metadata,
    check_colmap_installation,
    setup_colmap_workspace,
    run_colmap_pipeline_from_commands
)
from pipelines.mistral_pipeline import MistralPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoPipeline:
    """
    Pipeline for processing videos and generating 3D reconstructions using COLMAP.
    """
    
    def __init__(
        self, 
        mistral_api_key: str,
        mistral_api_url: str,
        mistral_model: str = "mistral-medium",
        colmap_executable: str = "colmap"
    ):
        """
        Initialize the video pipeline.
        
        Args:
            mistral_api_key (str): Mistral API key
            mistral_api_url (str): Mistral API endpoint URL
            mistral_model (str): Mistral model to use
            colmap_executable (str): Path to COLMAP executable
        """
        self.colmap_executable = colmap_executable
        self.mistral_pipeline = MistralPipeline(
            api_key=mistral_api_key,
            api_url=mistral_api_url,
            model=mistral_model
        )
        
        # Check COLMAP installation
        if not check_colmap_installation():
            logger.warning("COLMAP not found in system path. Some features may not work.")
    
    def process_video(
        self,
        video_path: str,
        output_dir: str,
        user_analysis: Dict[str, Any],
        downsample_rate: int = 8,
        template_path: Optional[str] = None,
        extract_frames_only: bool = False
    ) -> Tuple[bool, str]:
        """
        Process a video file to generate a 3D reconstruction.
        
        Args:
            video_path (str): Path to the video file
            output_dir (str): Output directory
            user_analysis (dict): User inputs about the video
            downsample_rate (int): Rate for downsampling frames
            template_path (str, optional): Path to command template
            extract_frames_only (bool): If True, only extract frames without running COLMAP
            
        Returns:
            tuple: (success_status, message)
        """
        start_time = time.time()
        
        try:
            # Create output directories
            frames_folder = os.path.join(output_dir, "frames")
            os.makedirs(frames_folder, exist_ok=True)
            
            # Step 1: Extract frames from the video
            logger.info(f"Extracting frames from video: {video_path}")
            extract_frames(
                video_file=video_path,
                downsample_rate=downsample_rate,
                output_folder=frames_folder
            )
            
            # If only frame extraction is requested, stop here
            if extract_frames_only:
                elapsed_time = time.time() - start_time
                return True, f"Frames extracted successfully in {elapsed_time:.2f} seconds."
            
            # Step 2: Extract video metadata using FFMPEG
            logger.info("Extracting video metadata")
            video_metadata = extract_colmap_video_metadata(video_path)
            
            if not video_metadata:
                return False, "Failed to extract video metadata. Please check the video file."
            
            # Step 3: Generate COLMAP commands using templates
            from utils.colmap_templates import get_colmap_commands
            
            # Add video path to user analysis
            user_analysis["video_path"] = video_path
            
            logger.info("Generating COLMAP commands from templates")
            commands = get_colmap_commands(
                video_metadata=video_metadata,
                user_choices=user_analysis,
                working_dir=output_dir
            )
            
            if not commands:
                return False, "Failed to generate COLMAP commands. Please try again."
            
            # Log the generated commands
            logger.info(f"Generated {len(commands)} COLMAP commands:")
            for i, cmd in enumerate(commands):
                logger.info(f"Command {i+1}: {cmd}")
            
            # Step 4: Setup COLMAP workspace
            colmap_paths = setup_colmap_workspace(output_dir, frames_folder)
            
            # Step 5: Run COLMAP pipeline with template commands
            logger.info("Running COLMAP pipeline")
            success = run_colmap_pipeline_from_commands(
                commands=commands,
                working_dir=output_dir
            )
            
            elapsed_time = time.time() - start_time
            
            if success:
                message = (
                    f"Video processing completed successfully in {elapsed_time:.2f} seconds.\n"
                    f"COLMAP reconstruction is available in: {colmap_paths['sparse_dir']}"
                )
                return True, message
            else:
                return False, f"COLMAP pipeline failed after {elapsed_time:.2f} seconds."
                
        except Exception as e:
            logger.error(f"Error in video pipeline: {str(e)}")
            return False, f"Error processing video: {str(e)}"
    
    def get_user_analysis_form(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the form fields for user analysis of the video.
        
        Returns:
            dict: Form field definitions
        """
        return {
            "motion_type": {
                "type": "select",
                "label": "Camera Motion Type",
                "options": ["linear", "circular", "stationary", "complex", "handheld"],
                "default": "linear",
                "help": "The primary type of camera movement in the video"
            },
            "scene_type": {
                "type": "select",
                "label": "Scene Type",
                "options": ["indoor", "outdoor", "object", "landscape", "architecture", "mixed"],
                "default": "object",
                "help": "The type of scene being captured"
            },
            "camera_movement": {
                "type": "select",
                "label": "Camera Movement Speed",
                "options": ["slow", "medium", "fast", "variable"],
                "default": "medium",
                "help": "How quickly the camera moves relative to the subject"
            },
            "lighting_conditions": {
                "type": "select",
                "label": "Lighting Conditions",
                "options": ["consistent", "variable", "low_light", "bright", "harsh_shadows"],
                "default": "consistent",
                "help": "The lighting conditions in the video"
            },
            "subject_characteristics": {
                "type": "text",
                "label": "Subject Characteristics",
                "default": "static object with good texture",
                "help": "Brief description of the main subject (texture, reflectivity, etc.)"
            },
            "additional_notes": {
                "type": "text_area",
                "label": "Additional Notes",
                "default": "",
                "help": "Any other details that might be relevant for reconstruction"
            }
        }