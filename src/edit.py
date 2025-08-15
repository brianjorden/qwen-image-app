"""
Image editing and noise interpolation utilities for the qwen-image-app.

This module provides utilities for processing input images for noise interpolation generation,
including validation, preprocessing, and encoding operations.
"""

import torch
from PIL import Image
from typing import Tuple, Optional, Union
from pathlib import Path

from .config import get_config
from .models import get_model_manager, get_pipe, get_img2img_pipe


def validate_input_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    """Validate and load an input image for noise interpolation generation.
    
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


def resize_image_for_generation(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Resize image to match target generation dimensions.
    
    Args:
        image: Input PIL Image
        target_width: Target width for generation
        target_height: Target height for generation
        
    Returns:
        Resized PIL Image
    """
    config = get_config()
    
    # Ensure dimensions are valid for the model
    from .process import validate_dimensions
    target_width, target_height = validate_dimensions(target_width, target_height)
    
    # If image is already the right size, return as-is
    if image.size == (target_width, target_height):
        return image
    
    # Resize with high-quality resampling
    resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    print(f"Resized input image from {image.size} to {resized.size}")
    return resized


def encode_image_to_latents(image: Image.Image, target_width: int, target_height: int) -> torch.Tensor:
    """Encode a PIL image to latents for noise interpolation generation.
    
    Args:
        image: Input PIL Image
        target_width: Target width for generation
        target_height: Target height for generation
        
    Returns:
        Encoded latents tensor
        
    Raises:
        RuntimeError: If VAE is not loaded or encoding fails
    """
    manager = get_model_manager()
    pipe = get_pipe()
    
    if not manager.vae:
        raise RuntimeError("VAE not loaded - load it in the Models tab first")
    
    # Resize image to match generation dimensions
    processed_image = resize_image_for_generation(image, target_width, target_height)
    
    # Preprocess using pipeline's image processor
    image_processor = pipe.image_processor
    processed_tensor = image_processor.preprocess(processed_image)
    
    with torch.no_grad():
        # Convert PIL to tensor format expected by VAE
        if processed_tensor.dim() == 3:
            processed_tensor = processed_tensor.unsqueeze(0)  # Add batch dimension
        if processed_tensor.dim() == 4 and processed_tensor.shape[1] == 3:
            # Need to add temporal dimension for QwenImage VAE
            processed_tensor = processed_tensor.unsqueeze(2)
        
        processed_tensor = processed_tensor.to(manager.vae.device, manager.vae.dtype)
        
        # Encode to latents
        posterior = manager.vae.encode(processed_tensor, return_dict=True)
        latents = posterior.latent_dist.sample()
        
        # Pack latents for QwenImage format
        batch_size = 1
        latent_height = processed_tensor.shape[-2] // pipe.vae_scale_factor // 2
        latent_width = processed_tensor.shape[-1] // pipe.vae_scale_factor // 2
        packed_latents = pipe._pack_latents(latents, batch_size, latents.shape[1], 
                                          latent_height * 2, latent_width * 2)
        
        print(f"Encoded image to latents, shape: {packed_latents.shape}")
        return packed_latents


def create_noise_interpolation_latents(
    input_image: Image.Image, 
    strength: float, 
    seed: int,
    target_width: int,
    target_height: int
) -> torch.Tensor:
    """Create latents for noise interpolation generation using noise interpolation.
    
    Args:
        input_image: Input PIL Image
        strength: Img2img strength (0.0 = minimal change, 1.0 = maximum change)
        seed: Random seed for noise generation
        target_width: Target width for generation
        target_height: Target height for generation
        
    Returns:
        Mixed latents tensor ready for generation
    """
    pipe = get_pipe()
    
    # Encode input image to latents
    image_latents = encode_image_to_latents(input_image, target_width, target_height)
    
    # Generate original noise from same seed (the starting point)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    original_noise = torch.randn(image_latents.shape, generator=generator, 
                                device=image_latents.device, dtype=image_latents.dtype)
    
    # Mix image latents with original noise based on strength
    # strength=0.0 → pure image latents (minimal change)  
    # strength=1.0 → pure original noise (maximum change)
    print(f"Mixing: {1.0 - strength:.2f} image + {strength:.2f} noise (strength={strength})")
    mixed_latents = (1.0 - strength) * image_latents + strength * original_noise
    
    return mixed_latents


def get_optimal_noise_interpolation_strength(image_similarity: str = "medium") -> float:
    """Get optimal noise interpolation strength based on desired transformation level.
    
    Args:
        image_similarity: Desired similarity level ("minimal", "low", "medium", "high", "maximum")
        
    Returns:
        Recommended strength value
    """
    strength_presets = {
        "minimal": 0.1,   # Very subtle changes
        "low": 0.3,       # Light modifications
        "medium": 0.5,    # Balanced transformation
        "high": 0.7,      # Significant changes
        "maximum": 0.9    # Almost complete regeneration
    }
    
    return strength_presets.get(image_similarity, 0.5)


def validate_noise_interpolation_parameters(
    image: Optional[Image.Image],
    strength: float,
    width: int,
    height: int
) -> Tuple[bool, str]:
    """Validate parameters for noise interpolation generation.
    
    Args:
        image: Input image (can be None)
        strength: Img2img strength value
        width: Target width
        height: Target height
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "No input image provided"
    
    if not isinstance(image, Image.Image):
        return False, "Invalid image format"
    
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


def preprocess_for_noise_interpolation(
    image: Union[str, Path, Image.Image],
    width: int,
    height: int,
    strength: float = 0.5
) -> Tuple[Image.Image, str]:
    """Preprocess an image for noise interpolation generation.
    
    Args:
        image: Input image (path or PIL Image)
        width: Target generation width
        height: Target generation height  
        strength: Img2img strength
        
    Returns:
        Tuple of (processed_image, status_message)
        
    Raises:
        ValueError: If preprocessing fails
    """
    # Validate and load image
    processed_image = validate_input_image(image)
    original_size = processed_image.size
    
    # Validate parameters
    is_valid, error_msg = validate_noise_interpolation_parameters(processed_image, strength, width, height)
    if not is_valid:
        raise ValueError(error_msg)
    
    # Resize if necessary
    processed_image = resize_image_for_generation(processed_image, width, height)
    
    # Create status message
    if original_size != processed_image.size:
        status = f"Image resized from {original_size} to {processed_image.size} for generation"
    else:
        status = f"Image ready for noise interpolation generation at {processed_image.size}"
    
    return processed_image, status


def get_optimal_img2img_strength(transformation_level: str = "medium") -> float:
    """Get optimal img2img strength based on desired transformation level.
    
    Args:
        transformation_level: Desired transformation level ("minimal", "low", "medium", "high", "maximum")
        
    Returns:
        Recommended strength value for true img2img
    """
    strength_presets = {
        "minimal": 0.2,   # Very subtle changes, preserve most structure
        "low": 0.4,       # Light modifications, keep main elements
        "medium": 0.6,    # Balanced transformation
        "high": 0.8,      # Significant changes, creative interpretation
        "maximum": 0.95   # Nearly complete regeneration
    }
    
    return strength_presets.get(transformation_level, 0.6)


def validate_img2img_parameters(
    image: Optional[Image.Image],
    strength: float,
    width: int,
    height: int
) -> Tuple[bool, str]:
    """Validate parameters for true img2img generation.
    
    Args:
        image: Input image (can be None)
        strength: Img2img strength value  
        width: Target width
        height: Target height
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "No input image provided"
    
    if not isinstance(image, Image.Image):
        return False, "Invalid image format"
    
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


def preprocess_for_img2img(
    image: Union[str, Path, Image.Image],
    width: int,
    height: int,
    strength: float = 0.6
) -> Tuple[Image.Image, str]:
    """Preprocess an image for true img2img generation.
    
    Args:
        image: Input image (path or PIL Image)
        width: Target generation width
        height: Target generation height  
        strength: Img2img strength
        
    Returns:
        Tuple of (processed_image, status_message)
        
    Raises:
        ValueError: If preprocessing fails
    """
    # Validate and load image
    processed_image = validate_input_image(image)
    original_size = processed_image.size
    
    # Validate parameters  
    is_valid, error_msg = validate_img2img_parameters(processed_image, strength, width, height)
    if not is_valid:
        raise ValueError(error_msg)
    
    # Resize if necessary
    processed_image = resize_image_for_generation(processed_image, width, height)
    
    # Create status message
    if original_size != processed_image.size:
        status = f"Image resized from {original_size} to {processed_image.size} for img2img generation"
    else:
        status = f"Image ready for img2img generation at {processed_image.size}"
    
    return processed_image, status


def compare_img2img_modes(input_image: Image.Image, strength: float = 0.6) -> str:
    """Compare noise interpolation vs true img2img modes.
    
    Args:
        input_image: Input image for comparison
        strength: Strength value to compare
        
    Returns:
        Comparison description
    """
    noise_desc = f"Noise Interpolation (strength={strength}):\n"
    noise_desc += f"- Mixes image latents with random noise\n"
    noise_desc += f"- Creative interpretation with {strength:.0%} randomness\n"
    noise_desc += f"- Good for artistic variations\n\n"
    
    img2img_desc = f"True Img2img (strength={strength}):\n"
    img2img_desc += f"- Starts denoising from partially noised image\n"
    img2img_desc += f"- Follows proper diffusion img2img process\n"
    img2img_desc += f"- Better structure preservation and coherence\n"
    img2img_desc += f"- More predictable transformations"
    
    return noise_desc + img2img_desc