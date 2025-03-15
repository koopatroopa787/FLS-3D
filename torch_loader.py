"""
torch_loader.py - Robust PyTorch and Marigold model loader

This module provides reliable loading of PyTorch and Marigold depth models
across different environments, handling CUDA compatibility issues gracefully.
"""

import os
import sys
import logging
import subprocess
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("torch_loader")

# Set environment variables to control PyTorch behavior
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class TorchEnvironment:
    """Manages PyTorch environment setup and model loading."""
    
    def __init__(self, prefer_cpu: bool = False):
        """
        Initialize the PyTorch environment manager.
        
        Args:
            prefer_cpu: If True, will use CPU even if CUDA is available
        """
        self.prefer_cpu = prefer_cpu
        self.device = "cpu"
        self.torch = None
        self.cuda_available = False
        self.cuda_version = None
        self.torch_version = None
        self.diffusers = None
        
        # Try to clean CUDA paths to avoid conflicts
        self._clean_cuda_paths()
        
    def _clean_cuda_paths(self) -> None:
        """Clean up CUDA paths in environment variables to avoid conflicts."""
        # Only modify PATH and LD_LIBRARY_PATH if they exist
        if 'LD_LIBRARY_PATH' in os.environ:
            paths = os.environ['LD_LIBRARY_PATH'].split(':')
            # Keep only one CUDA path to avoid duplicates
            cuda_paths = [p for p in paths if 'cuda' in p.lower()]
            if cuda_paths:
                # Keep only the first CUDA path found
                first_cuda = cuda_paths[0]
                non_cuda_paths = [p for p in paths if 'cuda' not in p.lower()]
                os.environ['LD_LIBRARY_PATH'] = ':'.join([first_cuda] + non_cuda_paths)
                logger.info(f"Cleaned LD_LIBRARY_PATH, using CUDA path: {first_cuda}")
    
    def _import_torch(self) -> bool:
        """
        Safely import PyTorch.
        
        Returns:
            bool: True if import successful, False otherwise
        """
        try:
            # Try to import torch
            import torch
            self.torch = torch
            
            # Get PyTorch version info
            self.torch_version = torch.__version__
            
            # Check CUDA availability
            if not self.prefer_cpu:
                self.cuda_available = torch.cuda.is_available()
                if self.cuda_available:
                    self.cuda_version = torch.version.cuda
                    self.device = "cuda"
                    logger.info(f"PyTorch {self.torch_version} with CUDA {self.cuda_version} available")
                else:
                    logger.info(f"PyTorch {self.torch_version} (CPU only)")
            else:
                logger.info(f"PyTorch {self.torch_version} (CPU mode preferred)")
                
            # Try to import diffusers
            try:
                import diffusers
                self.diffusers = diffusers
            except ImportError:
                logger.warning("Failed to import diffusers library")
                return False
                
            return True
        except ImportError as e:
            logger.error(f"Failed to import PyTorch: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error importing PyTorch: {e}")
            return False
    
    def _test_cuda_compatibility(self) -> bool:
        """
        Test CUDA compatibility by running a small tensor operation.
        
        Returns:
            bool: True if CUDA works, False otherwise
        """
        if not self.cuda_available or self.device != "cuda":
            return False
            
        try:
            # Create a small tensor and move it to GPU
            x = self.torch.ones(1, 1)
            x = x.to(self.device)
            # Try a simple operation
            y = x + x
            # If we get here, CUDA is working
            return True
        except Exception as e:
            logger.warning(f"CUDA compatibility test failed: {e}")
            # Fall back to CPU
            self.device = "cpu"
            return False
            
    def initialize(self) -> bool:
        """
        Initialize PyTorch and test CUDA compatibility.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        # Import PyTorch
        if not self._import_torch():
            return False
            
        # If CUDA is available, test compatibility
        if self.cuda_available and not self.prefer_cpu:
            if not self._test_cuda_compatibility():
                logger.warning("CUDA compatibility test failed, falling back to CPU")
                self.device = "cpu"
        
        return True
    
    def get_device(self) -> str:
        """Get the PyTorch device to use."""
        return self.device
    
    def get_torch_dtype(self, use_fp16: bool = False) -> Any:
        """
        Get the appropriate torch dtype based on settings.
        
        Args:
            use_fp16: Whether to use half precision when possible
            
        Returns:
            torch.dtype: The appropriate dtype
        """
        if self.torch is None:
            raise RuntimeError("PyTorch not initialized. Call initialize() first.")
            
        if use_fp16 and self.device == "cuda":
            return self.torch.float16
        else:
            return self.torch.float32
    
    def load_marigold_model(
        self, 
        model_path: str = "prs-eth/marigold-depth-lcm-v1-0", 
        use_fp16: bool = False
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load Marigold depth and disparity models.
        
        Args:
            model_path: Path or HuggingFace model ID for Marigold model
            use_fp16: Whether to use half precision when possible
            
        Returns:
            tuple: (depth_pipe, disparity_pipe) or (None, None) if loading failed
        """
        if self.torch is None or self.diffusers is None:
            logger.error("PyTorch or diffusers not initialized. Call initialize() first.")
            return None, None
            
        try:
            # Get appropriate dtype
            torch_dtype = self.get_torch_dtype(use_fp16)
            
            # Load depth model
            logger.info(f"Loading depth model from {model_path} on {self.device}")
            depth_pipe = self.diffusers.MarigoldDepthPipeline.from_pretrained(
                model_path, 
                prediction_type='depth', 
                torch_dtype=torch_dtype
            ).to(self.device)
            
            # Load disparity model
            logger.info(f"Loading disparity model from {model_path} on {self.device}")
            disparity_pipe = self.diffusers.MarigoldDepthPipeline.from_pretrained(
                model_path, 
                prediction_type='disparity', 
                torch_dtype=torch_dtype
            ).to(self.device)
            
            return depth_pipe, disparity_pipe
        except Exception as e:
            logger.error(f"Error loading Marigold models: {e}")
            if self.device == "cuda":
                # Try falling back to CPU if CUDA loading fails
                logger.info("Attempting to fall back to CPU for model loading...")
                self.device = "cpu"
                try:
                    # Load depth model on CPU
                    depth_pipe = self.diffusers.MarigoldDepthPipeline.from_pretrained(
                        model_path, 
                        prediction_type='depth', 
                        torch_dtype=self.torch.float32
                    ).to("cpu")
                    
                    # Load disparity model on CPU
                    disparity_pipe = self.diffusers.MarigoldDepthPipeline.from_pretrained(
                        model_path, 
                        prediction_type='disparity', 
                        torch_dtype=self.torch.float32
                    ).to("cpu")
                    
                    return depth_pipe, disparity_pipe
                except Exception as cpu_e:
                    logger.error(f"CPU fallback also failed: {cpu_e}")
            return None, None


# Singleton instance for application-wide use
torch_env = None


def initialize_torch(prefer_cpu: bool = False) -> TorchEnvironment:
    """
    Initialize the PyTorch environment.
    
    Args:
        prefer_cpu: If True, will use CPU even if CUDA is available
        
    Returns:
        TorchEnvironment: Initialized PyTorch environment
    """
    global torch_env
    torch_env = TorchEnvironment(prefer_cpu=prefer_cpu)
    success = torch_env.initialize()
    
    if not success:
        logger.error("Failed to initialize PyTorch environment")
    
    return torch_env


def get_torch_env() -> Optional[TorchEnvironment]:
    """
    Get the PyTorch environment singleton.
    
    Returns:
        TorchEnvironment or None: The initialized environment or None if not initialized
    """
    global torch_env
    if torch_env is None:
        logger.warning("PyTorch environment not initialized. Call initialize_torch() first.")
    return torch_env


def load_marigold_models(
    model_path: str = "prs-eth/marigold-depth-lcm-v1-0",
    use_fp16: bool = False,
    prefer_cpu: bool = False
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    One-step function to initialize PyTorch and load Marigold models.
    
    Args:
        model_path: Path or HuggingFace model ID for Marigold model
        use_fp16: Whether to use half precision when possible
        prefer_cpu: If True, will use CPU even if CUDA is available
        
    Returns:
        tuple: (depth_pipe, disparity_pipe) or (None, None) if loading failed
    """
    # Initialize PyTorch
    env = initialize_torch(prefer_cpu=prefer_cpu)
    if env.torch is None:
        return None, None
        
    # Load models
    return env.load_marigold_model(model_path=model_path, use_fp16=use_fp16)


if __name__ == "__main__":
    # Simple test/demo of the module
    env = initialize_torch()
    if env.torch is not None:
        print(f"PyTorch {env.torch_version} initialized on {env.device}")
        if env.cuda_available:
            print(f"CUDA version: {env.cuda_version}")
    
    depth_pipe, disparity_pipe = env.load_marigold_model()
    if depth_pipe is not None and disparity_pipe is not None:
        print("Marigold models loaded successfully")
    else:
        print("Failed to load Marigold models")