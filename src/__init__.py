"""
Qwen-Image Generation Application Package.

A comprehensive text-to-image generation system using the Qwen-Image model
with advanced features including metadata embedding, session organization,
and multi-GPU support.
"""

__version__ = "1.0.0"
__author__ = "Brian"

# Public API
from .analysis import get_analyzer
from .config import init_config, get_config
from .gallery import get_session_manager, get_gallery_manager
from .metadata import embed_metadata, extract_metadata, format_metadata_display
from .models import get_model_manager, get_pipe
from .process import generate_image
from .prompt import enhance_prompt_local, detect_language, count_tokens

__all__ = [
    # Config
    'init_config',
    'get_config',
    
    # Models
    'get_model_manager',
    'get_pipe',
    
    # Generation
    'generate_image',
    
    # Session/Gallery
    'get_session_manager',
    'get_gallery_manager',
    
    
    # Metadata
    'embed_metadata',
    'extract_metadata',
    'format_metadata_display',
    
    # Prompt
    'enhance_prompt_local',
    'detect_language',
    'count_tokens',
    
    # Analysis
    'get_analyzer',
]
