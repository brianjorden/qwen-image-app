"""
PNG metadata embedding and extraction for generation parameters.
"""

import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def embed_metadata(image: Image.Image, metadata: Dict[str, Any]) -> Image.Image:
    """Embed generation metadata into PNG image.
    
    Args:
        image: PIL Image to embed metadata into
        metadata: Dictionary of generation parameters
        
    Returns:
        Image with embedded metadata
    """
    # Create PNG metadata
    png_info = PngInfo()
    
    # Add timestamp
    metadata['timestamp'] = datetime.now().isoformat()
    
    # Serialize metadata as JSON
    metadata_json = json.dumps(metadata, indent=2)
    
    # Add to PNG info
    png_info.add_text("qwen-image-metadata", metadata_json)
    
    # Also add key fields as separate entries for compatibility
    for key in ['prompt', 'negative_prompt', 'seed', 'steps', 'cfg_scale', 
                'width', 'height', 'model_info']:
        if key in metadata:
            value = metadata[key]
            if isinstance(value, dict):
                value = json.dumps(value)
            else:
                value = str(value)
            png_info.add_text(f"qwen-{key}", value)
    
    # Return image with metadata (will be saved with it)
    image.info["pnginfo"] = png_info
    return image


def extract_metadata(image_path: str) -> Optional[Dict[str, Any]]:
    """Extract generation metadata from PNG image file.
    
    Args:
        image_path: Path to PNG image
        
    Returns:
        Dictionary of metadata or None if not found
    """
    try:
        image = Image.open(image_path)
        return extract_metadata_from_pil_image(image)
    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")
    
    return None


def extract_metadata_from_pil_image(image: Image.Image) -> Optional[Dict[str, Any]]:
    """Extract generation metadata from PIL Image object.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary of metadata or None if not found
    """
    try:
        # Try to get our custom metadata first
        if hasattr(image, 'text'):
            text_data = image.text
            
            # Look for our main metadata
            if 'qwen-image-metadata' in text_data:
                return json.loads(text_data['qwen-image-metadata'])
            
            # Fall back to reconstructing from individual fields
            metadata = {}
            for key, value in text_data.items():
                if key.startswith('qwen-'):
                    field = key[5:]  # Remove 'qwen-' prefix
                    try:
                        # Try to parse as JSON first
                        metadata[field] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        # Keep as string
                        metadata[field] = value
            
            if metadata:
                return metadata
    
    except Exception as e:
        print(f"Error extracting metadata from PIL image: {e}")
    
    return None


def format_metadata_display(metadata: Dict[str, Any]) -> str:
    """Format metadata for display in UI.
    
    Args:
        metadata: Dictionary of generation parameters
        
    Returns:
        Formatted string for display
    """
    if not metadata:
        return "No metadata found"
    
    lines = []
    
    # Timestamp
    if 'timestamp' in metadata:
        lines.append(f"Generated: {metadata['timestamp']}")
        lines.append("")
    
    # Main parameters - one per line for better readability
    if 'prompt' in metadata:
        lines.append(f"Prompt: {metadata['prompt']}")
    
    # Show enhanced/processed prompt if different
    if 'enhanced_prompt' in metadata and metadata['enhanced_prompt']:
        lines.append(f"Enhanced: {metadata['enhanced_prompt']}")
    if 'final_processed_prompt' in metadata and metadata['final_processed_prompt'] != metadata.get('prompt', ''):
        lines.append(f"Final Processed: {metadata['final_processed_prompt']}")
    
    if 'negative_prompt' in metadata and metadata['negative_prompt']:
        lines.append(f"Negative: {metadata['negative_prompt']}")
    
    lines.append("")
    
    # Generation settings - one per line
    if 'width' in metadata and 'height' in metadata:
        lines.append(f"Dimensions: {metadata['width']}x{metadata['height']}")
    if 'steps' in metadata:
        lines.append(f"Steps: {metadata['steps']}")
    if 'cfg_scale' in metadata:
        lines.append(f"CFG Scale: {metadata['cfg_scale']}")
    if 'seed' in metadata:
        lines.append(f"Seed: {metadata['seed']}")
    
    # Show applied template/magic text (but fix the truncation issue)
    if 'applied_template_text' in metadata and metadata['applied_template_text']:
        # Don't show the full template text, just indicate it was applied
        lines.append(f"Template Applied: Yes")
    if 'applied_magic_text' in metadata and metadata['applied_magic_text']:
        lines.append(f"Magic Prompt: {metadata['applied_magic_text']}")
    
    # Generation mode info
    lines.append("")
    
    # Show img2img info
    if 'is_img2img' in metadata and metadata['is_img2img']:
        lines.append("Img2Img Generation: Yes")
        if 'noise_interpolation_strength' in metadata and metadata['noise_interpolation_strength'] is not None:
            lines.append(f"Img2Img Strength: {metadata['noise_interpolation_strength']}")
    
    # Show two-stage info
    if 'is_two_stage' in metadata and metadata['is_two_stage']:
        lines.append("Two-Stage Generation: Yes")
        if 'two_stage_mode' in metadata and metadata['two_stage_mode']:
            lines.append(f"Two-Stage Mode: {metadata['two_stage_mode']}")
        if 'first_stage_steps' in metadata:
            lines.append(f"Stage 1 Steps: {metadata['first_stage_steps']}")
        if 'second_stage_steps' in metadata:
            lines.append(f"Stage 2 Steps: {metadata['second_stage_steps']}")
        if 'first_stage_image_path' in metadata and metadata['first_stage_image_path']:
            lines.append(f"Stage 1 Image: {Path(metadata['first_stage_image_path']).name}")
    
    # Show if this is a stage 1 of two-stage
    if 'is_stage1_of_two_stage' in metadata and metadata['is_stage1_of_two_stage']:
        lines.append("Stage: 1 of 2")
    
    # Model info
    if 'model_info' in metadata:
        lines.append("")
        lines.append("Models:")
        model_info = metadata['model_info']
        if isinstance(model_info, dict):
            for component, info in model_info.items():
                lines.append(f"  {component}: {info}")
    
    # LoRA info
    if 'loras' in metadata and metadata['loras']:
        lines.append("")
        lines.append("LoRAs:")
        for lora in metadata['loras']:
            if isinstance(lora, dict):
                lines.append(f"  {lora.get('name', 'unknown')}: {lora.get('strength', 1.0)}")
            else:
                lines.append(f"  {lora}")
    
    return "\n".join(lines)


def save_image_with_metadata(
    image: Image.Image,
    save_path: str,
    generation_params: Dict[str, Any],
    model_info: Optional[Dict[str, str]] = None
) -> Path:
    """Save image with embedded metadata.
    
    Args:
        image: PIL Image to save
        save_path: Path to save image to
        generation_params: Dictionary of generation parameters
        model_info: Optional dictionary of model component info
        
    Returns:
        Path where image was saved
    """
    save_path = Path(save_path)
    
    # Prepare metadata
    metadata = generation_params.copy()
    
    if model_info:
        metadata['model_info'] = model_info
    
    # Embed metadata
    if save_path.suffix.lower() == '.png':
        png_info = PngInfo()
        metadata['timestamp'] = datetime.now().isoformat()
        metadata_json = json.dumps(metadata, indent=2)
        png_info.add_text("qwen-image-metadata", metadata_json)
        
        # Save with metadata
        image.save(save_path, format='PNG', pnginfo=png_info)
    else:
        # For non-PNG formats, just save normally
        image.save(save_path)
        
        # Save metadata as sidecar JSON file
        json_path = save_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return save_path
