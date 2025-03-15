import os
import shutil
import datetime
import time
import streamlit as st
import torch
import numpy as np
import diffusers
import plotly.graph_objs as go
from PIL import Image

# Import configuration
from config import (
    OUTPUTS_BASE_DIR, 
    MISTRAL_API_KEY, 
    MISTRAL_API_URL, 
    MISTRAL_MODEL,
    COLMAP_EXECUTABLE
)

# Import utilities
from utils.depth_processing import process_depth_image, process_disparity_image
from utils.video_processing import extract_frames
from utils.ply_utils import save_point_cloud_to_ply, apply_poisson_surface_reconstruction
from utils.file_conversion import convert_ply_to_xyz, convert_ply_to_obj, convert_ply_to_stl
from utils.colmap_utils import extract_colmap_video_metadata, check_colmap_installation

# Import pipelines
from pipelines.video_pipeline import VideoPipeline

# Function to generate a unique output directory
def get_new_output_directory(base_dir):
    # Path to the file storing the last used number
    number_file = os.path.join(base_dir, 'last_dir_number.txt')
    
    # Read the last used numbers
    if os.path.exists(number_file):
        with open(number_file, 'r') as f:
            last_number = int(f.read())
    else:
        last_number = 0  # If the file doesn't exist, start from 0

    next_number = last_number + 1

    # Get current date and time with microseconds to ensure uniqueness
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    # Create the directory name
    new_dir_name = f"{next_number}_{current_time}"
    # Full path
    new_output_dir = os.path.join(base_dir, new_dir_name)

    # Update the last used number in the file
    with open(number_file, 'w') as f:
        f.write(str(next_number))

    return new_output_dir

# Function to create output directories
def create_output_directories(base_dir):
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Get new output directory
    new_output_dir = get_new_output_directory(base_dir)
    os.makedirs(new_output_dir, exist_ok=True)
    
    # Create subdirectories
    frames_folder = os.path.join(new_output_dir, 'frames')
    depth_output_dir = os.path.join(new_output_dir, 'depth_output')
    disparity_output_dir = os.path.join(new_output_dir, 'disparity_output')
    ply_output_dir = os.path.join(new_output_dir, 'ply_output')
    mesh_output_dir = os.path.join(new_output_dir, 'mesh_output')
    colmap_output_dir = os.path.join(new_output_dir, 'colmap')
    
    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(depth_output_dir, exist_ok=True)
    os.makedirs(disparity_output_dir, exist_ok=True)
    os.makedirs(ply_output_dir, exist_ok=True)
    os.makedirs(mesh_output_dir, exist_ok=True)
    os.makedirs(colmap_output_dir, exist_ok=True)
    
    # Return the paths
    return {
        "new_output_dir": new_output_dir,
        "frames_folder": frames_folder,
        "depth_output_dir": depth_output_dir,
        "disparity_output_dir": disparity_output_dir,
        "ply_output_dir": ply_output_dir,
        "mesh_output_dir": mesh_output_dir,
        "colmap_output_dir": colmap_output_dir
    }

# Add a reset function to clear data and reset the app
def reset_app():
    # Clear the output directories if they exist
    if "output_dirs" in st.session_state and st.session_state["output_dirs"] is not None:
        output_dirs = st.session_state["output_dirs"]
        # Remove the entire output directory
        shutil.rmtree(output_dirs["new_output_dir"], ignore_errors=True)
    # Clear session state
    st.session_state.clear()
    # Increment reset counter
    st.session_state["reset_counter"] = st.session_state.get("reset_counter", 0) + 1
    # Rerun the app to refresh
    st.rerun()

# Visualization Function for the Mesh
def visualize_mesh_with_plotly(mesh_file_path):
    """Visualize a mesh file using Plotly"""
    import open3d as o3d
    
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # No need to flatten Z values here as flattening is already done in point cloud generation
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    fig = go.Figure(data=[go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='violet',
        opacity=1.0,  # Solid appearance
    )])

    return fig

def render_form_field(field_key, field_config):
    """Render form fields based on their configuration"""
    field_type = field_config.get("type", "text")
    label = field_config.get("label", field_key.replace("_", " ").title())
    help_text = field_config.get("help", "")
    
    if field_type == "select":
        options = field_config.get("options", [])
        default = field_config.get("default", options[0] if options else "")
        return st.selectbox(label, options, index=options.index(default) if default in options else 0, help=help_text)
    elif field_type == "text":
        default = field_config.get("default", "")
        return st.text_input(label, value=default, help=help_text)
    elif field_type == "text_area":
        default = field_config.get("default", "")
        return st.text_area(label, value=default, help=help_text)
    elif field_type == "number":
        default = field_config.get("default", 0)
        min_val = field_config.get("min", 0)
        max_val = field_config.get("max", 100)
        step = field_config.get("step", 1)
        return st.number_input(label, min_value=min_val, max_value=max_val, value=default, step=step, help=help_text)
    elif field_type == "checkbox":
        default = field_config.get("default", False)
        return st.checkbox(label, value=default, help=help_text)
    else:
        return st.text_input(label, help=help_text)

# Initialize session state variables
if "reset_counter" not in st.session_state:
    st.session_state["reset_counter"] = 0
if "models_loaded" not in st.session_state:
    st.session_state["models_loaded"] = False
if "input_uploaded" not in st.session_state:
    st.session_state["input_uploaded"] = False
if "depth_disparity_processed" not in st.session_state:
    st.session_state["depth_disparity_processed"] = False
if "mesh_generated" not in st.session_state:
    st.session_state["mesh_generated"] = False
if "colmap_processed" not in st.session_state:
    st.session_state["colmap_processed"] = False
# Add session state variable for output directories
if "output_dirs" not in st.session_state:
    st.session_state["output_dirs"] = None
# Add session state for pipeline mode
if "pipeline_mode" not in st.session_state:
    st.session_state["pipeline_mode"] = "depth"  # Default to depth pipeline

# Streamlit UI components
st.title("3D Object Reconstruction Pipeline")

# Select pipeline mode
pipeline_mode = st.sidebar.radio(
    "Select Pipeline Mode",
    ["Depth-based Reconstruction", "COLMAP-based Reconstruction"],
    index=0
)

# Update session state based on selection
st.session_state["pipeline_mode"] = "depth" if pipeline_mode == "Depth-based Reconstruction" else "colmap"

# Input type selection
input_type = st.selectbox("Choose input type", ("Video", "Image"))

# Frame extraction rate - relevant for both pipelines
downsample_rate = st.number_input(
    "Select frame extraction rate (higher means fewer frames)", 
    min_value=1, 
    max_value=30, 
    value=8, 
    step=1
)

# FP16 option - relevant for depth pipeline
use_fp16 = st.checkbox(
    "Use half-precision (FP16) for faster processing (requires supported GPU)"
)

# Check COLMAP installation if COLMAP pipeline selected
if st.session_state["pipeline_mode"] == "colmap":
    if not check_colmap_installation():
        st.warning("COLMAP is not installed or not found in the system path. Please install COLMAP to use this pipeline.")

# Load models based on selected pipeline
if input_type:
    if not st.session_state["models_loaded"]:
        if st.button("Load Models"):
            with st.spinner('Loading models...'):
                # Determine the device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                if st.session_state["pipeline_mode"] == "depth":
                    # Load Marigold Depth Models for depth pipeline
                    torch_dtype = torch.float16 if use_fp16 and device == "cuda" else torch.float32

                    st.session_state["depth_pipe"] = diffusers.MarigoldDepthPipeline.from_pretrained(
                        "prs-eth/marigold-depth-lcm-v1-0", prediction_type='depth', torch_dtype=torch_dtype).to(device)
                    st.session_state["disparity_pipe"] = diffusers.MarigoldDepthPipeline.from_pretrained(
                        "prs-eth/marigold-depth-lcm-v1-0", prediction_type='disparity', torch_dtype=torch_dtype).to(device)
                elif st.session_state["pipeline_mode"] == "colmap":
                    # Initialize the video pipeline for COLMAP
                    st.session_state["video_pipeline"] = VideoPipeline(
                        mistral_api_key=MISTRAL_API_KEY,
                        mistral_api_url=MISTRAL_API_URL,
                        mistral_model=MISTRAL_MODEL,
                        colmap_executable=COLMAP_EXECUTABLE
                    )
                    
                st.session_state["models_loaded"] = True
            st.success('Models loaded successfully!')

# COLMAP Pipeline Processing Logic
if st.session_state["pipeline_mode"] == "colmap" and st.session_state["models_loaded"]:
    if input_type == "Video":
        video_file = st.file_uploader(
            "Upload a video file", type=["mp4"], key=f"colmap_video_file_{st.session_state['reset_counter']}"
        )
        
        if video_file is not None and st.session_state["output_dirs"] is None:
            # Create new output directories
            output_dirs = create_output_directories(OUTPUTS_BASE_DIR)
            st.session_state["output_dirs"] = output_dirs
            
            # Save the uploaded video
            video_path = os.path.join(output_dirs["new_output_dir"], video_file.name)
            with open(video_path, "wb") as f:
                f.write(video_file.getbuffer())
            
            # Extract video metadata
            metadata = extract_colmap_video_metadata(video_path)
            if metadata:
                st.session_state["video_metadata"] = metadata
                st.session_state["video_path"] = video_path
                st.session_state["input_uploaded"] = True
            else:
                st.error("Failed to extract video metadata. Please try a different video.")
        
        # Once video is uploaded, show the analysis form
        if st.session_state.get("input_uploaded", False) and not st.session_state.get("colmap_processed", False):
            st.write("### Video Analysis")
            st.write("Please provide information about your video to optimize the reconstruction process:")
            
            # Get the form configuration from the video pipeline
            form_config = st.session_state["video_pipeline"].get_user_analysis_form()
            
            # Initialize a dictionary to store form values if not already in session state
            if "user_analysis" not in st.session_state:
                st.session_state["user_analysis"] = {}
            
            # Render form fields and collect values
            for field_key, field_config in form_config.items():
                value = render_form_field(field_key, field_config)
                st.session_state["user_analysis"][field_key] = value
            
            # Process video after form is submitted
            if st.button("Process Video with COLMAP"):
                with st.spinner("Processing video with COLMAP. This may take a while..."):
                    success, message = st.session_state["video_pipeline"].process_video(
                        video_path=st.session_state["video_path"],
                        output_dir=st.session_state["output_dirs"]["new_output_dir"],
                        user_analysis=st.session_state["user_analysis"],
                        downsample_rate=downsample_rate
                    )
                    
                    if success:
                        st.success(message)
                        st.session_state["colmap_processed"] = True
                    else:
                        st.error(message)
        
        # Display results if COLMAP processing is complete
        if st.session_state.get("colmap_processed", False):
            st.write("### COLMAP Results")
            
            # Display sparse point cloud if available
            sparse_dir = os.path.join(st.session_state["output_dirs"]["colmap_output_dir"], "sparse", "0")
            if os.path.exists(sparse_dir):
                st.write("Sparse reconstruction completed successfully.")
                
                # TODO: Visualization of COLMAP results could be added here
                # This would require additional functionality to convert COLMAP output to viewable format
                
                # Provide download buttons for COLMAP output
                st.write("### Download COLMAP Results")
                
                # Create a zip file of COLMAP results
                import zipfile
                zip_path = os.path.join(st.session_state["output_dirs"]["new_output_dir"], "colmap_results.zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for root, dirs, files in os.walk(st.session_state["output_dirs"]["colmap_output_dir"]):
                        for file in files:
                            zipf.write(
                                os.path.join(root, file),
                                os.path.relpath(os.path.join(root, file), st.session_state["output_dirs"]["new_output_dir"])
                            )
                
                # Provide download button
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="Download COLMAP Results",
                        data=f.read(),
                        file_name="colmap_results.zip",
                        mime="application/zip"
                    )
            else:
                st.warning("COLMAP reconstruction data not found. There might have been an issue with the process.")

# Depth Pipeline Processing Logic
if st.session_state["pipeline_mode"] == "depth" and st.session_state["models_loaded"]:
    # Process video input
    if input_type == "Video":
        video_file = st.file_uploader(
            "Upload a video file", type=["mp4"], key=f"depth_video_file_{st.session_state['reset_counter']}"
        )
        if video_file is not None and st.session_state["output_dirs"] is None:
            # Create new output directories
            output_dirs = create_output_directories(OUTPUTS_BASE_DIR)
            st.session_state["output_dirs"] = output_dirs
            
            video_path = os.path.join(output_dirs["new_output_dir"], video_file.name)
            with open(video_path, "wb") as f:
                f.write(video_file.getbuffer())
            extract_frames(video_path, downsample_rate=downsample_rate, output_folder=output_dirs["frames_folder"])
            st.session_state["input_uploaded"] = True

    # Process image input
    elif input_type == "Image":
        image_file = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"], key=f"depth_image_file_{st.session_state['reset_counter']}"
        )
        if image_file is not None and st.session_state["output_dirs"] is None:
            # Create new output directories
            output_dirs = create_output_directories(OUTPUTS_BASE_DIR)
            st.session_state["output_dirs"] = output_dirs
            
            image = Image.open(image_file)
            image.save(os.path.join(output_dirs["frames_folder"], image_file.name))
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.session_state["input_uploaded"] = True

    # Process frames for depth and disparity after models are loaded and input is uploaded
    if st.session_state["models_loaded"] and st.session_state["input_uploaded"]:
        if not st.session_state["depth_disparity_processed"]:
            st.write("Processing frames for depth and disparity...")

            # Get directories from session state
            frames_folder = st.session_state["output_dirs"]["frames_folder"]
            depth_output_dir = st.session_state["output_dirs"]["depth_output_dir"]
            disparity_output_dir = st.session_state["output_dirs"]["disparity_output_dir"]

            files_to_process = os.listdir(frames_folder)
            total_files = len(files_to_process)

            progress_bar = st.progress(0)
            status_text = st.empty()

            start_time = time.time()

            for idx, filename in enumerate(files_to_process):
                input_path = os.path.join(frames_folder, filename)

                depth_output_path = os.path.join(depth_output_dir, f"depth_{filename}")
                disparity_output_path = os.path.join(disparity_output_dir, f"disparity_{filename}")

                # Generate depth and disparity images only if they don't already exist
                if not os.path.exists(depth_output_path):
                    try:
                        process_depth_image(input_path, depth_output_path, st.session_state["depth_pipe"])
                    except Exception as e:
                        st.error(f"Failed to process depth image for {filename}: {e}")
                if not os.path.exists(disparity_output_path):
                    try:
                        process_disparity_image(input_path, disparity_output_path, st.session_state["disparity_pipe"])
                    except Exception as e:
                        st.error(f"Failed to process disparity image for {filename}: {e}")

                # Update progress bar and status
                elapsed_time = time.time() - start_time
                avg_time_per_file = elapsed_time / (idx + 1)
                remaining_files = total_files - (idx + 1)
                est_remaining_time = remaining_files * avg_time_per_file

                progress_bar.progress((idx + 1) / total_files)
                status_text.text(f"Processed {idx + 1}/{total_files} frames. Estimated time remaining: {int(est_remaining_time)} seconds.")

            st.write("Depth and disparity processing completed!")
            st.session_state["depth_disparity_processed"] = True
        else:
            st.write("Depth and disparity images are already processed!")

        # Display Depth and Disparity Images side by side
        st.write("### Depth and Disparity Images")
        cols = st.columns(2)
        with cols[0]:
            depth_output_dir = st.session_state["output_dirs"]["depth_output_dir"]
            # Get list of depth images
            depth_images = [
                os.path.join(depth_output_dir, f)
                for f in os.listdir(depth_output_dir)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ]
            if depth_images:
                # Create captions matching the number of images
                captions = [f"Depth Image {i+1}" for i in range(len(depth_images))]
                st.image(depth_images, caption=captions, width=300)
            else:
                st.warning("No depth images found to display.")
        with cols[1]:
            disparity_output_dir = st.session_state["output_dirs"]["disparity_output_dir"]
            # Get list of disparity images
            disparity_images = [
                os.path.join(disparity_output_dir, f)
                for f in os.listdir(disparity_output_dir)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ]
            if disparity_images:
                captions = [f"Disparity Image {i+1}" for i in range(len(disparity_images))]
                st.image(disparity_images, caption=captions, width=300)
            else:
                st.warning("No disparity images found to display.")

        # Dropdown to choose between Depth or Disparity for mesh construction
        mesh_choice = st.selectbox("Choose which image to use for mesh construction", ["Depth", "Disparity"])

        if st.button("Generate Mesh") and st.session_state["depth_disparity_processed"]:
            # Generate point cloud and mesh
            st.write(f"Using {mesh_choice} image for mesh generation...")
            if mesh_choice == "Depth":
                image_dir = st.session_state["output_dirs"]["depth_output_dir"]
            else:
                image_dir = st.session_state["output_dirs"]["disparity_output_dir"]

            # Get directories from session state
            ply_output_dir = st.session_state["output_dirs"]["ply_output_dir"]
            mesh_output_dir = st.session_state["output_dirs"]["mesh_output_dir"]

            files_to_process = os.listdir(image_dir)
            total_files = len(files_to_process)

            progress_bar = st.progress(0)
            status_text = st.empty()

            start_time = time.time()

            for idx, filename in enumerate(files_to_process):
                image_path = os.path.join(image_dir, filename)
                ply_output_path = os.path.join(ply_output_dir, filename.rsplit('.', 1)[0] + ".ply")
                save_point_cloud_to_ply(image_path, ply_output_path)

                # Update progress bar and status
                progress_bar.progress((idx + 1) / total_files)
                status_text.text(f"Generated point cloud for {idx + 1}/{total_files} images.")

            st.write("Point cloud generation completed!")

            # Apply Poisson surface reconstruction and display the mesh
            st.write("Applying Poisson surface reconstruction...")
            for ply_file in os.listdir(ply_output_dir):
                ply_file_path = os.path.join(ply_output_dir, ply_file)
                output_mesh_file = os.path.join(mesh_output_dir, ply_file.replace(".ply", "_mesh.ply"))

                # Apply Poisson reconstruction
                apply_poisson_surface_reconstruction(ply_file_path, output_mesh_file)

                # Visualize the generated mesh
                st.write("### 3D Mesh Visualization")
                fig = visualize_mesh_with_plotly(output_mesh_file)

                # Display the Plotly chart without altering dimensions
                st.plotly_chart(fig)

            st.write("3D mesh reconstruction and visualization completed!")

            # Mark the mesh as generated in session state
            st.session_state["mesh_generated"] = True

    # If mesh is already generated, allow downloading without resetting
    if st.session_state.get("mesh_generated", False):
        st.write("### Download Converted Files")

        # Provide a dropdown to select the format
        formats = ['.ply', '.xyz', '.obj', '.stl']
        selected_format = st.selectbox('Select format to download', formats)

        # A unique key to avoid DuplicateWidgetID errors
        download_key_counter = 0

        # Get directories from session state
        ply_output_dir = st.session_state["output_dirs"]["ply_output_dir"]

        for ply_file in os.listdir(ply_output_dir):
            ply_file_path = os.path.join(ply_output_dir, ply_file)
            # Determine the file to download based on the selected format
            if selected_format == '.ply':
                file_to_download = ply_file
                file_path = ply_file_path
            else:
                file_to_download = ply_file.replace('.ply', selected_format)
                file_path = os.path.join(ply_output_dir, file_to_download)
                # Convert if not already done
                if not os.path.exists(file_path):
                    if selected_format == '.xyz':
                        convert_ply_to_xyz(ply_file_path, file_path)
                    elif selected_format == '.obj':
                        convert_ply_to_obj(ply_file_path, file_path)
                    elif selected_format == '.stl':
                        convert_ply_to_stl(ply_file_path, file_path)

            with open(file_path, "rb") as f:
                # Increment key counter for uniqueness
                download_key_counter += 1
                st.download_button(
                    label=f"Download {file_to_download}",
                    data=f.read(),
                    file_name=file_to_download,
                    key=f"download_{file_to_download}_{download_key_counter}"
                )

# Add the reset button at the end of the app
if st.button("Reset"):
    reset_app()

# Footer
st.sidebar.markdown("---")
st.sidebar.write("3D Object Reconstruction Pipeline")
st.sidebar.write("© 2024")