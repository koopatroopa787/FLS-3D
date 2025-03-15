#!/bin/bash
# Improved run_app.sh - combines best practices from both scripts

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Initialize conda
eval "$(conda shell.bash hook)"

# Check for gaussian_splatting environment
if ! conda env list | grep -q "gaussian_splatting"; then
    echo "Error: 'gaussian_splatting' conda environment not found."
    exit 1
fi

# Activate gaussian_splatting to access COLMAP
echo "Activating gaussian_splatting environment..."
conda activate gaussian_splatting

# Get COLMAP path
export COLMAP_PATH=$(which colmap)
if [ -z "$COLMAP_PATH" ]; then
    echo "Error: COLMAP not found in the gaussian_splatting environment."
    conda deactivate
    exit 1
fi
echo "Found COLMAP at: $COLMAP_PATH"

# Save the entire PATH that includes COLMAP
COLMAP_INCLUDED_PATH="$PATH"

# Create symlink if needed (optional - this isn't causing the problem)
SYMLINKS_DIR="${SCRIPT_DIR}/colmap_bin"
mkdir -p "${SYMLINKS_DIR}"
COLMAP_LINK="${SYMLINKS_DIR}/colmap"
ln -sf "${COLMAP_PATH}" "${COLMAP_LINK}"

# Create config file (optional - this part is fine)
CONFIG_FILE="${SCRIPT_DIR}/env_config.json"
cat > "${CONFIG_FILE}" << EOL
{
    "colmap_path": "${COLMAP_LINK}",
    "script_dir": "${SCRIPT_DIR}"
}
EOL

# Deactivate conda
echo "Deactivating gaussian_splatting environment..."
conda deactivate

# Check for myenv
if [ ! -d "$HOME/myenv" ]; then
    echo "Error: 'myenv' virtual environment not found at $HOME/myenv."
    exit 1
fi

# Activate myenv
echo "Activating myenv environment..."
source "$HOME/myenv/bin/activate"

# IMPORTANT: Add COLMAP to PATH but DON'T modify LD_LIBRARY_PATH
export PATH="${COLMAP_INCLUDED_PATH}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Run Streamlit
echo "Starting 3D Reconstruction Pipeline..."
COLMAP_PATH="${COLMAP_PATH}" streamlit run main.py "$@"

# Cleanup
deactivate
echo "Application closed."