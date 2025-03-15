#!/usr/bin/env python3
"""
PyTorch Environment Test Script

This script performs a comprehensive test of your PyTorch installation,
checking for proper functionality, CUDA availability, and compatibility.
"""

import os
import sys
import platform
import subprocess
from datetime import datetime

def print_section(title):
    """Print a formatted section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def run_command(command):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e.stderr}")
        return f"Error: {e.returncode}"

def main():
    """Main function to test PyTorch installation."""
    print_section("SYSTEM INFORMATION")
    print(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    
    print_section("ENVIRONMENT VARIABLES")
    cuda_related_vars = [
        "CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", 
        "LD_LIBRARY_PATH", "PATH", "PYTHONPATH"
    ]
    for var in cuda_related_vars:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")
    
    print_section("NVIDIA GPU INFORMATION")
    print("NVIDIA-SMI Output:")
    print(run_command("nvidia-smi"))
    
    print_section("CUDA TOOLKIT INFORMATION")
    print("NVCC Version:")
    print(run_command("nvcc --version"))
    
    print_section("INSTALLED PYTHON PACKAGES")
    print(run_command(f"{sys.executable} -m pip list | grep -E 'torch|cuda'"))
    
    print_section("PYTORCH IMPORT TEST")
    try:
        import torch
        print(f"PyTorch import successful! Version: {torch.__version__}")
        
        # Print PyTorch build information
        print("\nPyTorch Build Information:")
        print(f"Debug Build: {torch.version.debug}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"CUDNN Version: {torch.version.cudnn}")
        print(f"Git Version: {torch.version.git_version}")
        print(f"HIP Version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
        
        # Check CUDA availability
        print("\nCUDA Information:")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"Device {i} Name: {torch.cuda.get_device_name(i)}")
                print(f"Device {i} Capability: {torch.cuda.get_device_capability(i)}")
            print(f"Current CUDA Device: {torch.cuda.current_device()}")
            
            # Try a simple CUDA operation
            print("\nTesting CUDA Tensor Operations:")
            x = torch.cuda.FloatTensor([1.0, 2.0, 3.0])
            print(f"CUDA Tensor: {x}")
            print(f"CUDA Tensor Device: {x.device}")
            result = x * 2
            print(f"CUDA Operation Result: {result}")
            
        else:
            print("CUDA is not available. Using CPU only.")
            
            # Try a simple CPU operation
            print("\nTesting CPU Tensor Operations:")
            x = torch.FloatTensor([1.0, 2.0, 3.0])
            print(f"CPU Tensor: {x}")
            result = x * 2
            print(f"CPU Operation Result: {result}")
        
    except ImportError as e:
        print(f"PyTorch import failed with error: {e}")
        
    except Exception as e:
        print(f"PyTorch test failed with error: {e}")
        import traceback
        print(traceback.format_exc())
    
    print_section("LIBRARY DEPENDENCY CHECK")
    print("Checking PyTorch library dependencies:")
    if sys.platform == "linux":
        torch_lib_path = run_command(f"find {sys.prefix} -name '*torch*' -type d | grep lib")
        if torch_lib_path:
            print(f"PyTorch library path: {torch_lib_path}")
            print("\nShared library dependencies:")
            print(run_command(f"ldd {torch_lib_path}/lib/libtorch.so"))
            
            print("\nCUSPARSE library dependencies:")
            cusparse_path = run_command(f"find {sys.prefix} -name 'libcusparse.so*'")
            if cusparse_path:
                print(f"CUSPARSE path: {cusparse_path}")
                print(run_command(f"ldd {cusparse_path}"))
            else:
                print("CUSPARSE library not found.")
        else:
            print("PyTorch library path not found.")
    
    print_section("CONCLUSION")
    try:
        import torch
        if torch.cuda.is_available():
            print("PyTorch with CUDA support is working correctly!")
        else:
            print("PyTorch is working with CPU only.")
    except:
        print("PyTorch is not working correctly in this environment.")
    
    print("\nIf you're experiencing issues, consider:")
    print("1. Installing a PyTorch version that matches your CUDA version")
    print("2. Using a CPU-only version of PyTorch")
    print("3. Setting the correct environment variables for CUDA")

if __name__ == "__main__":
    main()