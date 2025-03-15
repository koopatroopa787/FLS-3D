"""
Configuration settings for the 3D reconstruction pipeline.
"""
import os
import json
import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_BASE_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Create necessary directories if they don't exist
os.makedirs(OUTPUTS_BASE_DIR, exist_ok=True)

# Load environment configuration if available
ENV_CONFIG_PATH = os.path.join(PROJECT_ROOT, "env_config.json")
env_config = {}
if os.path.exists(ENV_CONFIG_PATH):
    try:
        with open(ENV_CONFIG_PATH, 'r') as f:
            env_config = json.load(f)
        logger.info(f"Loaded environment configuration from {ENV_CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Error loading environment configuration: {e}")

# Mistral API settings
MISTRAL_API_KEY = "RzVjPRq5NWai8PDEBPWT2o1B1fMjmIE2"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-medium"  # or another appropriate model

# COLMAP settings
# First try the symlink created by the script, then environment variable, then default
COLMAP_EXECUTABLE = (
    env_config.get("colmap_path") or 
    os.environ.get("COLMAP_PATH") or
    "colmap"
)

# Validate COLMAP path
if not os.path.exists(COLMAP_EXECUTABLE):
    logger.warning(f"COLMAP executable not found at: {COLMAP_EXECUTABLE}")
    
    # Try which command as a fallback
    try:
        colmap_path = subprocess.check_output(["which", "colmap"], 
                                             universal_newlines=True).strip()
        if os.path.exists(colmap_path):
            COLMAP_EXECUTABLE = colmap_path
            logger.info(f"Found COLMAP using 'which' at: {COLMAP_EXECUTABLE}")
    except subprocess.SubprocessError:
        logger.warning("Failed to find COLMAP using 'which' command")

logger.info(f"Using COLMAP executable: {COLMAP_EXECUTABLE}")

COLMAP_TEMP_DIR = os.path.join(OUTPUTS_BASE_DIR, "colmap_temp")
os.makedirs(COLMAP_TEMP_DIR, exist_ok=True)

# FFMPEG settings
FFMPEG_EXECUTABLE = "ffmpeg"  # Path to FFMPEG executable, adjust if needed

# Video processing settings
DEFAULT_DOWNSAMPLE_RATE = 8  # Default frame extraction rate

# Depth processing settings
DEPTH_MODEL_PATH = "prs-eth/marigold-depth-lcm-v1-0"  # Path to the depth model

# Point cloud processing settings
MAX_POINTS = 100000  # Maximum number of points in the point cloud

# Mesh generation settings
MESH_POISSON_DEPTH = 9  # Depth for Poisson surface reconstruction

# Check if COLMAP is available
def check_colmap_availability():
    """Check if COLMAP is available and executable"""
    try:
        logger.info(f"Testing COLMAP availability at: {COLMAP_EXECUTABLE}")
        result = subprocess.run(
            [COLMAP_EXECUTABLE, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        success = result.returncode == 0
        if success:
            logger.info("COLMAP test successful")
        else:
            logger.warning(f"COLMAP test failed with return code: {result.returncode}")
            if result.stderr:
                logger.warning(f"COLMAP error output: {result.stderr}")
        return success
    except Exception as e:
        logger.error(f"Error testing COLMAP: {str(e)}")
        return False

# Print information about COLMAP availability
COLMAP_AVAILABLE = check_colmap_availability()
if COLMAP_AVAILABLE:
    logger.info("✓ COLMAP is available and working correctly")
else:
    logger.warning("⚠ COLMAP is not available or not working correctly. COLMAP features will be disabled.")