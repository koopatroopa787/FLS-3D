import os
import torch
from PIL import Image

def process_depth_image(image_path, output_path, pipe):
    image = Image.open(image_path)
    depth = pipe(image, num_inference_steps=2, match_input_resolution=True, ensemble_size=15)
    vis = pipe.image_processor.visualize_depth(depth.prediction)
    vis[0].save(output_path)
    print(f"Depth image saved as '{output_path}'")

def process_disparity_image(image_path, output_path, pipe):
    image = Image.open(image_path)
    disparity = pipe(image, num_inference_steps=2, match_input_resolution=True, ensemble_size=15)
    vis = pipe.image_processor.visualize_depth(disparity.prediction)
    vis[0].save(output_path)
    print(f"Disparity image saved as '{output_path}'")
