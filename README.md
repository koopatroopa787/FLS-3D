# 3D Reconstruction Pipeline

A toolkit for creating 3D models from images and videos using depth-based reconstruction and COLMAP-based structure-from-motion.

## 📋 Overview

This project provides an easy-to-use framework for 3D reconstruction with two main approaches:

- **Depth-based Reconstruction**: Uses the Marigold depth model to generate depth maps and create 3D models from single images or videos
- **COLMAP-based Reconstruction**: Integrates COLMAP for traditional structure-from-motion reconstruction with an AI-assisted pipeline

The project includes a Streamlit web interface for easy interaction with both pipelines.

## 🌟 Key Features

- **Streamlit Web Interface**: User-friendly interface for all reconstruction methods
- **Multiple Input Types**: Support for single images or videos
- **Depth Map Generation**: Uses state-of-the-art Marigold depth models for high-quality depth estimation
- **Point Cloud & Mesh Generation**: Automated pipeline from depth maps to 3D meshes
- **COLMAP Integration**: AI-assisted COLMAP pipeline with Mistral LLM for generating optimal COLMAP commands
- **Format Conversion**: Tools for converting between different 3D formats (PLY, OBJ, STL, XYZ)
- **3D Visualization**: Interactive 3D model visualization directly in the web interface using Plotly

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- COLMAP (for structure-from-motion reconstruction)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/3d-reconstruction-pipeline.git
cd 3d-reconstruction-pipeline
```

### Step 2: Install COLMAP

COLMAP is required for the structure-from-motion reconstruction pipeline.

#### Ubuntu/Debian

```bash
sudo apt install libsparsehash-dev
sudo apt install colmap
```

#### Windows

Download the latest release from [COLMAP's GitHub page](https://github.com/colmap/colmap/releases) and follow the installation instructions.

### Step 3: Set Up Python Environment

The project uses two separate environments:

#### Main Environment for Web Interface
This environment runs the Streamlit interface and depth-based reconstruction pipeline:

```bash
# Create a virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Gaussian Splatting Environment for COLMAP
This environment contains COLMAP and other 3D reconstruction dependencies:

```bash
# Create conda environment
conda env create -f environment.yml
conda activate gaussian_splatting
```

The `run_app.sh` script handles switching between these environments, using:
1. `gaussian_splatting` environment to access COLMAP
2. `myenv` environment to run the web interface

For GPU acceleration, ensure that you have a compatible version of PyTorch with CUDA support.

### Step 4: Configure Environment

Create an `env_config.json` file in the project root:

```json
{
    "colmap_path": "/path/to/colmap",
    "script_dir": "/path/to/project"
}
```

Replace `/path/to/colmap` with the path to your COLMAP executable from the `gaussian_splatting` environment.

The `run_app.sh` script creates this configuration automatically by:
1. Activating the `gaussian_splatting` environment
2. Locating COLMAP using `which colmap`
3. Creating a symlink to COLMAP in `colmap_bin/`
4. Generating the `env_config.json` file
5. Switching to `myenv` to run the application

## 🚀 Running the Application

```bash
# Using the provided script (recommended)
./run_app.sh

# Or manually
streamlit run main.py
```

The Streamlit interface will guide you through:
1. Selecting a reconstruction method (Depth-based or COLMAP-based)
2. Uploading images or videos
3. Setting processing parameters
4. Generating depth maps, point clouds, and 3D meshes
5. Visualizing and downloading results

## 📊 Pipeline Details

### Depth-based Reconstruction Pipeline

The depth-based pipeline (`pipelines/depth_pipeline.py`) provides a complete workflow:

1. **Depth Estimation**: Generates depth maps from images using Marigold depth models
2. **Point Cloud Generation**: Converts depth maps to 3D point clouds
3. **Mesh Reconstruction**: Applies Poisson surface reconstruction to create 3D meshes
4. **Model Optimization**: Cleans and optimizes meshes for better visual quality

Example command:
```bash
python -m pipelines.depth_pipeline --input_dir /path/to/images --output_dir /path/to/output
```

### COLMAP-based Reconstruction Pipeline

The COLMAP pipeline (`pipelines/video_pipeline.py`) integrates with Mistral LLM:

1. **Frame Extraction**: Extracts frames from input videos at optimal intervals
2. **Analysis**: Uses video metadata and user input to determine optimal parameters
3. **Command Generation**: Generates optimized COLMAP commands via Mistral API
4. **Structure from Motion**: Runs COLMAP pipeline for camera pose estimation and 3D reconstruction
5. **Point Cloud & Mesh Generation**: Creates point clouds and meshes from COLMAP results

Example command:
```bash
python -m pipelines.video_pipeline --video_path /path/to/video.mp4 --output_dir /path/to/output
```

## 📁 Project Structure

```
3d-reconstruction-pipeline/
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
```

## 💻 Technical Implementation

### Depth Estimation

The project uses Marigold depth models through the Diffusers library:

```python
# From utils/depth_processing.py
def process_depth_image(image_path, output_path, pipe):
    image = Image.open(image_path)
    depth = pipe(image, num_inference_steps=2, match_input_resolution=True, ensemble_size=15)
    vis = pipe.image_processor.visualize_depth(depth.prediction)
    vis[0].save(output_path)
```

### Point Cloud Generation

Point clouds are generated from depth maps:

```python
# From utils/ply_utils.py
def save_point_cloud_to_ply(depth_image_path, ply_filename):
    # Read depth image and create 3D points
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
    # Convert 2D coordinates and depth values to 3D points
    # ...
    cloud = PyntCloud(df)
    cloud.to_file(ply_filename)
```

### Mesh Reconstruction

Meshes are created using Poisson surface reconstruction with Open3D:

```python
# From utils/ply_utils.py
def apply_poisson_surface_reconstruction(ply_file_path, output_mesh_file):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_file_path)
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # Apply Poisson reconstruction
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7, scale=1.1, linear_fit=True)
    # Save the mesh
    o3d.io.write_triangle_mesh(output_mesh_file, mesh)
```

### COLMAP Integration

The COLMAP pipeline is enhanced with intelligent command generation:

```python
# From pipelines/video_pipeline.py
def process_video(self, video_path, output_dir, user_analysis, downsample_rate=8):
    # Extract frames from video
    extract_frames(video_file=video_path, downsample_rate=downsample_rate, output_folder=frames_folder)
    
    # Generate optimal COLMAP commands based on video metadata and user input
    commands = get_colmap_commands(video_metadata=video_metadata, user_choices=user_analysis, working_dir=output_dir)
    
    # Run COLMAP pipeline
    success = run_colmap_pipeline_from_commands(commands=commands, working_dir=output_dir)
```

