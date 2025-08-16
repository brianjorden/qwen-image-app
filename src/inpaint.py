"""
Inpainting utilities for the qwen-image-app.

This module provides utilities for processing input images and masks for inpainting generation,
including validation, preprocessing, and encoding operations.
"""

import torch
from PIL import Image
from typing import Tuple, Optional, Union
from pathlib import Path

from .config import get_config
from .models import get_model_manager, get_inpaint_pipe


def validate_mask_image(mask: Union[str, Path, Image.Image]) -> Image.Image:
    """Validate and load a mask image for inpainting generation.
    
    Args:
        mask: Mask image path or PIL Image object
        
    Returns:
        Validated PIL Image (grayscale)
        
    Raises:
        ValueError: If mask is invalid or unsupported format
    """
    if isinstance(mask, (str, Path)):
        mask_path = Path(mask)
        if not mask_path.exists():
            raise ValueError(f"Mask file not found: {mask_path}")
        
        try:
            mask = Image.open(mask_path)
        except Exception as e:
            raise ValueError(f"Failed to load mask {mask_path}: {e}")
    
    if not isinstance(mask, Image.Image):
        raise ValueError("Mask must be a PIL Image or valid image path")
    
    # Convert to grayscale if necessary
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    return mask


def validate_input_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    """Validate and load an input image for inpainting generation.
    
    Args:
        image: Image path or PIL Image object
        
    Returns:
        Validated PIL Image
        
    Raises:
        ValueError: If image is invalid or unsupported format
    """
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists():
            raise ValueError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
    
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image or valid image path")
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def resize_images_for_inpainting(
    image: Image.Image, 
    mask: Image.Image, 
    target_width: int, 
    target_height: int
) -> Tuple[Image.Image, Image.Image]:
    """Resize image and mask to match target generation dimensions.
    
    Args:
        image: Input PIL Image
        mask: Mask PIL Image (grayscale)
        target_width: Target width for generation
        target_height: Target height for generation
        
    Returns:
        Tuple of (resized_image, resized_mask)
    """
    config = get_config()
    
    # Ensure dimensions are valid for the model
    from .process import validate_dimensions
    target_width, target_height = validate_dimensions(target_width, target_height)
    
    # If images are already the right size, return as-is
    if image.size == (target_width, target_height) and mask.size == (target_width, target_height):
        return image, mask
    
    # Resize with high-quality resampling
    resized_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    resized_mask = mask.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    print(f"Resized images from {image.size} to {resized_image.size}")
    return resized_image, resized_mask


def get_optimal_inpaint_strength(transformation_level: str = "medium") -> float:
    """Get optimal inpainting strength based on desired transformation level.
    
    Args:
        transformation_level: Desired transformation level ("minimal", "low", "medium", "high", "maximum")
        
    Returns:
        Recommended strength value for inpainting
    """
    strength_presets = {
        "minimal": 0.2,   # Very subtle changes, preserve most structure
        "low": 0.4,       # Light modifications, keep main elements
        "medium": 0.6,    # Balanced transformation
        "high": 0.8,      # Significant changes, creative interpretation
        "maximum": 0.95   # Nearly complete regeneration
    }
    
    return strength_presets.get(transformation_level, 0.6)


def validate_inpaint_parameters(
    image: Optional[Image.Image],
    mask: Optional[Image.Image],
    strength: float,
    width: int,
    height: int
) -> Tuple[bool, str]:
    """Validate parameters for inpainting generation.
    
    Args:
        image: Input image (can be None)
        mask: Mask image (can be None)
        strength: Inpainting strength value
        width: Target width
        height: Target height
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "No input image provided"
    
    if mask is None:
        return False, "No mask image provided"
    
    if not isinstance(image, Image.Image):
        return False, "Invalid image format"
    
    if not isinstance(mask, Image.Image):
        return False, "Invalid mask format"
    
    if not (0.0 <= strength <= 1.0):
        return False, f"Strength must be between 0.0 and 1.0, got {strength}"
    
    if width <= 0 or height <= 0:
        return False, f"Invalid dimensions: {width}x{height}"
    
    # Check if dimensions are reasonable
    if width > 4096 or height > 4096:
        return False, f"Dimensions too large: {width}x{height} (max 4096x4096)"
    
    config = get_config()
    if width % config.resolution_multiple != 0 or height % config.resolution_multiple != 0:
        return False, f"Dimensions must be multiples of {config.resolution_multiple}"
    
    return True, ""


def preprocess_for_inpainting(
    image: Union[str, Path, Image.Image],
    mask: Union[str, Path, Image.Image],
    width: int,
    height: int,
    strength: float = 0.6
) -> Tuple[Image.Image, Image.Image, str]:
    """Preprocess image and mask for inpainting generation.
    
    Args:
        image: Input image (path or PIL Image)
        mask: Mask image (path or PIL Image)
        width: Target generation width
        height: Target generation height  
        strength: Inpainting strength
        
    Returns:
        Tuple of (processed_image, processed_mask, status_message)
        
    Raises:
        ValueError: If preprocessing fails
    """
    # Validate and load images
    processed_image = validate_input_image(image)
    processed_mask = validate_mask_image(mask)
    original_size = processed_image.size
    
    # Validate parameters
    is_valid, error_msg = validate_inpaint_parameters(processed_image, processed_mask, strength, width, height)
    if not is_valid:
        raise ValueError(error_msg)
    
    # Resize if necessary
    processed_image, processed_mask = resize_images_for_inpainting(
        processed_image, processed_mask, width, height
    )
    
    # Create status message
    if original_size != processed_image.size:
        status = f"Images resized from {original_size} to {processed_image.size} for inpainting generation"
    else:
        status = f"Images ready for inpainting generation at {processed_image.size}"
    
    return processed_image, processed_mask, status


def validate_mask_coverage(mask: Image.Image, min_coverage: float = 0.01, max_coverage: float = 0.8) -> Tuple[bool, str, float]:
    """Validate that the mask has reasonable coverage.
    
    Args:
        mask: Mask PIL Image (grayscale)
        min_coverage: Minimum percentage of mask coverage (0.0-1.0)
        max_coverage: Maximum percentage of mask coverage (0.0-1.0)
        
    Returns:
        Tuple of (is_valid, message, coverage_percentage)
    """
    # Convert mask to numpy-like format to calculate coverage
    import numpy as np
    mask_array = np.array(mask)
    
    # Calculate percentage of white pixels (areas to inpaint)
    white_pixels = np.sum(mask_array > 128)  # Threshold for "white"
    total_pixels = mask_array.size
    coverage = white_pixels / total_pixels
    
    if coverage < min_coverage:
        return False, f"Mask coverage too small ({coverage:.1%}). Minimum: {min_coverage:.1%}", coverage
    
    if coverage > max_coverage:
        return False, f"Mask coverage too large ({coverage:.1%}). Maximum: {max_coverage:.1%}", coverage
    
    return True, f"Mask coverage: {coverage:.1%}", coverage


def create_mask_from_bbox(width: int, height: int, x: int, y: int, bbox_width: int, bbox_height: int) -> Image.Image:
    """Create a rectangular mask from bounding box coordinates.
    
    Args:
        width: Image width
        height: Image height
        x: Top-left x coordinate
        y: Top-left y coordinate
        bbox_width: Bounding box width
        bbox_height: Bounding box height
        
    Returns:
        PIL Image mask (black background, white region to inpaint)
    """
    # Create black image
    mask = Image.new('L', (width, height), 0)
    
    # Create white rectangle for the region to inpaint
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x, y, x + bbox_width, y + bbox_height], fill=255)
    
    return mask


def prepare_inpaint_inputs(
    image: Image.Image,
    mask: Image.Image,
    strength: float,
    width: int,
    height: int
) -> Tuple[Image.Image, Image.Image, str]:
    """Prepare final inputs for inpainting pipeline.
    
    Args:
        image: Input PIL Image
        mask: Mask PIL Image
        strength: Inpainting strength
        width: Target width
        height: Target height
        
    Returns:
        Tuple of (final_image, final_mask, status_message)
    """
    # Validate mask coverage
    is_valid, coverage_msg, coverage = validate_mask_coverage(mask)
    if not is_valid:
        print(f"Warning: {coverage_msg}")
    
    # Resize images to exact dimensions
    final_image, final_mask = resize_images_for_inpainting(image, mask, width, height)
    
    status = f"Inpainting ready: {coverage_msg}, strength: {strength}"
    return final_image, final_mask, status