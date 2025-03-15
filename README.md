3D Reconstruction Pipeline
A toolkit for creating 3D models from images and videos using depth-based reconstruction and COLMAP-based structure-from-motion.
📋 Overview
This project provides an easy-to-use framework for 3D reconstruction with two main approaches:

Depth-based Reconstruction: Uses the Marigold depth model to generate depth maps and create 3D models from single images or videos
COLMAP-based Reconstruction: Integrates COLMAP for traditional structure-from-motion reconstruction with an AI-assisted pipeline

The project includes a Streamlit web interface for easy interaction with both pipelines.
🌟 Key Features

Streamlit Web Interface: User-friendly interface for all reconstruction methods
Multiple Input Types: Support for single images or videos
Depth Map Generation: Uses state-of-the-art Marigold depth models for high-quality depth estimation
Point Cloud & Mesh Generation: Automated pipeline from depth maps to 3D meshes
COLMAP Integration: AI-assisted COLMAP pipeline with Mistral LLM for generating optimal COLMAP commands
Format Conversion: Tools for converting between different 3D formats (PLY, OBJ, STL, XYZ)
3D Visualization: Interactive 3D model visualization directly in the web interface using Plotly

⚙️ Installation
Prerequisites

Python 3.8+
CUDA-capable GPU (recommended for faster processing)
COLMAP (for structure-from-motion reconstruction)

Step 1: Clone the Repository
bashCopygit clone https://github.com/yourusername/3d-reconstruction-pipeline.git
cd 3d-reconstruction-pipeline
Step 2: Install COLMAP
COLMAP is required for the structure-from-motion reconstruction pipeline.
Ubuntu/Debian
bashCopysudo apt install libsparsehash-dev
sudo apt install colmap
Windows
Download the latest release from COLMAP's GitHub page and follow the installation instructions.
Step 3: Set Up Python Environment
The project uses two separate environments:
Main Environment for Web Interface
This environment runs the Streamlit interface and depth-based reconstruction pipeline:
bashCopy# Create a virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Gaussian Splatting Environment for COLMAP
This environment contains COLMAP and other 3D reconstruction dependencies:
bashCopy# Create conda environment
conda env create -f environment.yml
conda activate gaussian_splatting
The run_app.sh script handles switching between these environments, using:

gaussian_splatting environment to access COLMAP
myenv environment to run the web interface

For GPU acceleration, ensure that you have a compatible version of PyTorch with CUDA support.
Step 4: Configure Environment
Create an env_config.json file in the project root:
jsonCopy{
    "colmap_path": "/path/to/colmap",
    "script_dir": "/path/to/project"
}
Replace /path/to/colmap with the path to your COLMAP executable from the gaussian_splatting environment.
The run_app.sh script creates this configuration automatically by:

Activating the gaussian_splatting environment
Locating COLMAP using which colmap
Creating a symlink to COLMAP in colmap_bin/
Generating the env_config.json file
Switching to myenv to run the application

🚀 Running the Application
bashCopy# Using the provided script (recommended)
./run_app.sh

# Or manually
streamlit run main.py
The Streamlit interface will guide you through:

Selecting a reconstruction method (Depth-based or COLMAP-based)
Uploading images or videos
Setting processing parameters
Generating depth maps, point clouds, and 3D meshes
Visualizing and downloading results

📊 Pipeline Details
Depth-based Reconstruction Pipeline
The depth-based pipeline (pipelines/depth_pipeline.py) provides a complete workflow:

Depth Estimation: Generates depth maps from images using Marigold depth models
Point Cloud Generation: Converts depth maps to 3D point clouds
Mesh Reconstruction: Applies Poisson surface reconstruction to create 3D meshes
Model Optimization: Cleans and optimizes meshes for better visual quality

Example command:
bashCopypython -m pipelines.depth_pipeline --input_dir /path/to/images --output_dir /path/to/output
COLMAP-based Reconstruction Pipeline
The COLMAP pipeline (pipelines/video_pipeline.py) integrates with Mistral LLM:

Frame Extraction: Extracts frames from input videos at optimal intervals
Analysis: Uses video metadata and user input to determine optimal parameters
Command Generation: Generates optimized COLMAP commands via Mistral API
Structure from Motion: Runs COLMAP pipeline for camera pose estimation and 3D reconstruction
Point Cloud & Mesh Generation: Creates point clouds and meshes from COLMAP results

Example command:
bashCopypython -m pipelines.video_pipeline --video_path /path/to/video.mp4 --output_dir /path/to/output
📁 Project Structure
Copy3d-reconstruction-pipeline/
├── config.py                   # Configuration settings
├── main.py                     # Streamlit web application
├── requirements.txt            # Python dependencies
├── run_app.sh                  # Script to run the application
│
├── pipelines/                  # Reconstruction pipelines
│   ├── __init__.py
│   ├── depth_pipeline.py       # Depth-based reconstruction
│   ├── video_pipeline.py       # Video/COLMAP pipeline
│   └── mistral_pipeline.py     # Integration with Mistral API
│
└── utils/                      # Utility modules
    ├── __init__.py
    ├── colmap_utils.py         # COLMAP utilities
    ├── colmap_templates.py     # Templates for COLMAP commands
    ├── depth_processing.py     # Depth map processing functions
    ├── file_conversion.py      # 3D format conversion
    ├── mesh_generation.py      # Mesh creation utilities
    ├── ply_utils.py            # Point cloud utilities
    └── video_processing.py     # Video frame extraction
