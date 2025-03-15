"""
Integration with Mistral API for generating COLMAP commands based on video analysis.
"""
import os
import json
import requests
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MistralPipeline:
    """
    Pipeline for interacting with Mistral API to generate COLMAP commands
    based on video analysis and user input.
    """
    
    def __init__(self, api_key: str, api_url: str, model: str = "mistral-medium"):
        """
        Initialize the Mistral pipeline.
        
        Args:
            api_key (str): Mistral API key
            api_url (str): Mistral API endpoint URL
            model (str): Model name to use
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def generate_colmap_commands(
        self, 
        video_metadata: Dict[str, Any], 
        user_analysis: Dict[str, Any],
        template_path: Optional[str] = None
    ) -> List[str]:
        """
        Generate COLMAP commands based on video metadata and user analysis.
        
        Args:
            video_metadata (dict): Video metadata from FFMPEG
            user_analysis (dict): User inputs about the video
            template_path (str, optional): Path to prompt template file
            
        Returns:
            list: Generated COLMAP commands
        """
        # Load prompt template if provided, otherwise use default
        if template_path and os.path.exists(template_path):
            with open(template_path, 'r') as f:
                prompt_template = f.read()
        else:
            prompt_template = self._get_default_prompt_template()
        
        # Format the prompt with video metadata and user analysis
        prompt = self._format_prompt(prompt_template, video_metadata, user_analysis)
        
        # Call the Mistral API
        try:
            response = self._call_mistral_api(prompt)
            
            # Extract and parse commands from the response
            commands = self._extract_commands(response)
            return commands
            
        except Exception as e:
            logger.error(f"Error generating COLMAP commands: {str(e)}")
            return []
    
    def _format_prompt(
        self, 
        template: str, 
        video_metadata: Dict[str, Any], 
        user_analysis: Dict[str, Any]
    ) -> str:
        """
        Format the prompt template with video metadata and user input.
        
        Args:
            template (str): Prompt template
            video_metadata (dict): Video metadata
            user_analysis (dict): User inputs about the video
            
        Returns:
            str: Formatted prompt
        """
        # Combine metadata and user analysis for formatting
        format_data = {**video_metadata, **user_analysis}
        
        # Add useful derived information
        if 'fps' in video_metadata and 'duration' in video_metadata:
            format_data['total_frames'] = int(video_metadata['fps'] * video_metadata['duration'])
        
        # Special handling for motion_type to provide more context
        if 'motion_type' in user_analysis:
            motion_type = user_analysis['motion_type']
            if motion_type == 'linear':
                format_data['motion_description'] = "camera moving in a straight line"
            elif motion_type == 'circular':
                format_data['motion_description'] = "camera moving in a circular path around the subject"
            elif motion_type == 'stationary':
                format_data['motion_description'] = "camera is mostly stationary with minimal movement"
            else:
                format_data['motion_description'] = f"camera moving with {motion_type} motion"
        
        # Format the template with all data
        try:
            return template.format(**format_data)
        except KeyError as e:
            logger.warning(f"Missing key in prompt template: {e}")
            # Attempt more forgiving formatting
            for key, value in format_data.items():
                placeholder = f"{{{key}}}"
                if placeholder in template:
                    template = template.replace(placeholder, str(value))
            return template
    
    def _call_mistral_api(self, prompt: str) -> str:
        """
        Call Mistral API with the formatted prompt.
        
        Args:
            prompt (str): Formatted prompt
            
        Returns:
            str: API response text
            
        Raises:
            Exception: If API call fails
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,  # Low temperature for more deterministic results
            "max_tokens": 2000   # Allow for detailed command generation
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"API call failed with status {response.status_code}: {response.text}")
                raise Exception(f"API call failed: {response.text}")
            
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise Exception(f"Request error: {str(e)}")
        except (KeyError, IndexError) as e:
            logger.error(f"Response parsing error: {str(e)}, Response: {response.text}")
            raise Exception(f"Response parsing error: {str(e)}")
    
    def _extract_commands(self, response: str) -> List[str]:
        """
        Extract COLMAP commands from the Mistral API response.
        
        Args:
            response (str): Text response from Mistral API
            
        Returns:
            list: Extracted COLMAP commands
        """
        commands = []
        
        # Look for commands between triple backticks or in a structured format
        import re
        
        # Try to find commands in code blocks
        code_blocks = re.findall(r'```(?:bash|shell)?\s*(.*?)```', response, re.DOTALL)
        
        if code_blocks:
            for block in code_blocks:
                # Split by lines and filter out empty lines and comments
                block_commands = [
                    line.strip() for line in block.split('\n') 
                    if line.strip() and not line.strip().startswith('#')
                ]
                commands.extend(block_commands)
        else:
            # If no code blocks, look for command-like lines
            lines = response.split('\n')
            for line in lines:
                if 'colmap' in line.lower() and not line.strip().startswith('#'):
                    commands.append(line.strip())
        
        return commands
    
    def _get_default_prompt_template(self) -> str:
        """
        Get the default prompt template if no external template is provided.
        
        Returns:
            str: Default prompt template
        """
        return """
You are an expert in photogrammetry using COLMAP software. I need your help generating the appropriate COLMAP commands for the following video:

## Video Metadata:
- Resolution: {width}x{height}
- Duration: {duration} seconds
- Frame Rate: {fps} fps
- Total Frames: {total_frames}

## Video Analysis:
- Motion Type: {motion_type} ({motion_description})
- Scene Type: {scene_type}
- Camera Movement: {camera_movement}
- Lighting Conditions: {lighting_conditions}
- Subject Characteristics: {subject_characteristics}

## Task:
Please provide the exact COLMAP command line commands I should run to get the best possible 3D reconstruction from this video. Consider the video characteristics and optimize for:
1. Appropriate frame extraction rate based on camera movement
2. Suitable feature extraction parameters
3. The best matching strategy (exhaustive, sequential, etc.)
4. Proper camera model selection
5. Appropriate dense reconstruction parameters

Format your answer as a sequence of shell commands that I can run. Include each command on a new line. Make sure the commands are complete and don't require any user intervention. Place all commands between triple backticks.

Here's an example of what your answer might look like:
```
# Command 1 explanation
command_1 --param1 value1 --param2 value2

# Command 2 explanation
command_2 --param1 value1 --param2 value2
```

Please provide commands optimized specifically for this video based on the characteristics I've provided.
"""

# Example usage:
# pipeline = MistralPipeline(api_key="your_api_key", api_url="https://api.mistral.ai/v1/chat/completions")
# commands = pipeline.generate_colmap_commands(video_metadata, user_analysis)