"""
Enhanced image generation pipeline with metadata, queue, and template support.
"""

import torch
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

from .config import get_config
from .models import get_model_manager, get_pipe, get_pipeline
from .metadata import save_image_with_metadata
from .gallery import get_session_manager
from .prompt import add_magic_prompt, detect_language
from .step import setup_step_callback, clear_cancellation, GenerationCancelledException
from .inpaint import preprocess_for_inpainting, get_optimal_inpaint_strength

# Global cancellation flag and lock
_generation_cancelled = threading.Event()
_generation_lock = threading.Lock()


def validate_dimensions(width: int, height: int) -> Tuple[int, int]:
    """Ensure dimensions are valid for the model.
    
    Args:
        width: Requested width
        height: Requested height
        
    Returns:
        Adjusted (width, height) tuple
    """
    config = get_config()
    multiple = config.resolution_multiple
    
    # Round to nearest multiple
    width = (width // multiple) * multiple
    height = (height // multiple) * multiple
    
    # Ensure minimum size
    width = max(width, multiple * 4)
    height = max(height, multiple * 4)
    
    return width, height


def calculate_shift(image_seq_len: int) -> float:
    """Calculate shift value for timestep scheduling.
    
    Args:
        image_seq_len: Image sequence length
        
    Returns:
        Shift value for scheduler
    """
    # From the original pipeline
    base_seq_len = 256
    max_seq_len = 4096
    base_shift = 0.5
    max_shift = 1.15
    
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    
    return mu


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    session: Optional[str] = None,
    name: Optional[str] = None,
    true_cfg_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    seed: Optional[int] = None,
    loras: Optional[List[Dict[str, Any]]] = None,
    enhance_prompt: bool = False,
    add_magic: bool = True,
    second_stage_steps: Optional[int] = None,
    two_stage_mode: str = "Img2Img Mode",
    input_image: Optional[Image.Image] = None,
    mask_image: Optional[Image.Image] = None,  # For inpainting mode
    noise_interpolation_strength: Optional[float] = None,
    img2img_mode: str = "true_img2img",  # "true_img2img", "noise_interpolation", or "inpaint"
    img2img_strength: Optional[float] = None,  # For true img2img mode
    inpaint_strength: Optional[float] = None,  # For inpainting mode
    **kwargs
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Generate a single image with full metadata support.
    
    Args:
        prompt: Positive prompt
        negative_prompt: Negative prompt
        session: Session directory name
        name: Image name
        true_cfg_scale: CFG scale value
        num_inference_steps: Number of denoising steps
        height: Image height
        width: Image width
        seed: Random seed
        loras: List of LoRA configs
        enhance_prompt: Whether to enhance the prompt
        apply_template: Whether to apply the training template
        add_magic: Whether to add magic quality prompts
        second_stage_steps: Additional steps for two-stage generation (0 = disabled)
        two_stage_mode: "Img2Img Mode" for noise interpolation
        input_image: Optional input image for img2img generation
        mask_image: Optional mask image for inpainting generation (white areas will be inpainted)
        noise_interpolation_strength: Strength for img2img transformation (0.0-1.0)
        img2img_mode: Generation mode ("true_img2img", "noise_interpolation", or "inpaint")
        img2img_strength: Strength for true img2img mode (0.0-1.0)
        inpaint_strength: Strength for inpainting mode (0.0-1.0)
        **kwargs: Additional pipeline arguments
        
    Returns:
        Tuple of (generated_image, save_path)
    """
    # Clear any previous cancellation flag at the start
    clear_cancellation()
    
    config = get_config()
    manager = get_model_manager()
    
    # Get defaults
    true_cfg_scale = true_cfg_scale if true_cfg_scale is not None else config.default_cfg
    num_inference_steps = num_inference_steps if num_inference_steps is not None else config.default_steps
    height = height if height is not None else config.default_height
    width = width if width is not None else config.default_width
    seed = seed if seed is not None else config.default_seed
    second_stage_steps = second_stage_steps if second_stage_steps is not None else 0
    noise_interpolation_strength = noise_interpolation_strength if noise_interpolation_strength is not None else config.default_noise_interpolation_strength
    
    # Determine generation mode
    is_two_stage = second_stage_steps > 0
    is_img2img = input_image is not None
    is_inpaint = mask_image is not None and input_image is not None
    is_true_img2img = is_img2img and img2img_mode == "true_img2img" and not is_inpaint
    
    # Set default img2img strength if not provided
    if img2img_strength is None:
        img2img_strength = config.default_img2img_strength
    
    # Set default inpaint strength if not provided
    if inpaint_strength is None:
        inpaint_strength = get_optimal_inpaint_strength("medium")
    
    # Import edit functions based on mode
    if is_img2img and not is_inpaint:
        if is_true_img2img:
            from .edit import preprocess_for_img2img
        else:
            from .edit import preprocess_for_noise_interpolation, create_noise_interpolation_latents
    
    # Import metadata function (used in multiple places)
    from .metadata import save_image_with_metadata
    
    # Validate dimensions
    width, height = validate_dimensions(width, height)
    
    # Set up session
    session_mgr = get_session_manager()
    if session is None:
        session = session_mgr.current_session or session_mgr.get_default_session()
    session_path = session_mgr.set_session(session)
    
    # Generate name if not provided
    if not name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"img_{timestamp}"
    
    # Capture original prompt and track transformations
    original_prompt = prompt
    applied_magic_text = None
    
    # Enhance prompt if requested
    enhanced_prompt = None
    if enhance_prompt and manager.text_encoder:
        from .prompt import enhance_prompt_local
        enhanced_prompt = enhance_prompt_local(prompt, manager.text_encoder, manager.tokenizer)
        prompt = enhanced_prompt
        print(f"Enhanced prompt: {prompt}")
    
    # Capture magic prompt text before applying
    if add_magic:
        # Load magic text from template file
        if hasattr(config, 'template_magic') and config.template_magic:
            try:
                from pathlib import Path
                path = Path(config.template_magic)
                if path.exists():
                    applied_magic_text = path.read_text(encoding='utf-8').strip()
                else:
                    applied_magic_text = ""
            except Exception:
                applied_magic_text = ""
        else:
            applied_magic_text = ""
        prompt = add_magic_prompt(prompt)
    
    
    # Handle negative prompt default
    if not negative_prompt and true_cfg_scale > 1.0:
        negative_prompt = config.default_negative
    
    # Load LoRAs if specified
    if loras and config.enable_lora:
        for lora in loras:
            if isinstance(lora, dict):
                manager.load_lora(lora['path'], lora.get('strength', 1.0))
    
    # Get appropriate pipeline based on mode
    if is_inpaint:
        pipe = get_pipeline("inpaint")
        print("Using inpainting pipeline")
    elif is_true_img2img:
        pipe = get_pipeline("img2img")
        print("Using true img2img pipeline")
    else:
        pipe = get_pipeline("txt2img")
        if is_img2img:
            print("Using txt2img pipeline with noise interpolation")
    
    # Setup step callback - always enabled for cancellation support
    save_steps = kwargs.get('save_steps', False)
    
    # Get VAE for step image saving
    vae = manager.vae if save_steps else None
    callback = setup_step_callback(
        enabled=True,  # Always enable for cancellation
        save_steps=save_steps,
        session_path=session_path,
        name=name,
        vae=vae
    )
    
    # Prepare generation arguments
    gen_args = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'true_cfg_scale': true_cfg_scale,  # QwenImagePipeline expects true_cfg_scale
        'num_inference_steps': num_inference_steps,
        'height': height,
        'width': width,
        'generator': torch.Generator(device=pipe.device).manual_seed(seed),
    }
    
    if callback:
        gen_args['callback_on_step_end'] = callback
    
    # Add any additional arguments (excluding internal parameters)
    pipeline_kwargs = kwargs.copy()
    # Remove internal parameters that aren't for the pipeline
    pipeline_kwargs.pop('save_steps', None)
    gen_args.update(pipeline_kwargs)
    
    # Handle img2img and inpainting preprocessing
    processed_input_image = None
    processed_mask_image = None
    if is_inpaint:
        try:
            processed_input_image, processed_mask_image, status = preprocess_for_inpainting(
                input_image, mask_image, width, height, inpaint_strength
            )
            print(f"Inpainting: {status}")
        except ValueError as e:
            print(f"Inpainting preprocessing failed: {e}")
            return None, None
    elif is_img2img:
        try:
            if is_true_img2img:
                processed_input_image, status = preprocess_for_img2img(input_image, width, height, img2img_strength)
                print(f"True img2img: {status}")
            else:
                processed_input_image, status = preprocess_for_noise_interpolation(input_image, width, height, noise_interpolation_strength)
                print(f"Noise interpolation: {status}")
        except ValueError as e:
            print(f"Img2img preprocessing failed: {e}")
            return None, None
    
    generation_type = ""
    if is_inpaint and is_two_stage:
        generation_type = "(Inpainting + Two-stage)"
    elif is_inpaint:
        generation_type = "(Inpainting)"
    elif is_img2img and is_two_stage:
        img_mode = "True Img2img" if is_true_img2img else "Noise Interpolation"
        generation_type = f"({img_mode} + Two-stage)"
    elif is_img2img:
        img_mode = "True Img2img" if is_true_img2img else "Noise Interpolation"
        generation_type = f"({img_mode})"
    elif is_two_stage:
        generation_type = "(Two-stage)"
    
    print(f"Generating '{name}' in session '{session}' {generation_type}")
    print(f"Settings: {width}x{height}, steps={num_inference_steps}{f'+{second_stage_steps}' if is_two_stage else ''}, cfg={true_cfg_scale}, seed={seed}")
    if is_inpaint:
        print(f"Inpainting strength: {inpaint_strength}")
    elif is_img2img:
        if is_true_img2img:
            print(f"True img2img strength: {img2img_strength}")
        else:
            print(f"Noise interpolation strength: {noise_interpolation_strength}")
    
    try:
        # Variables to track results
        first_stage_image = None
        first_stage_path = None
        final_image = None
        
        # Prepare generation arguments
        stage1_gen_args = gen_args.copy()
        
        # Handle input mode
        if is_inpaint:
            # Use inpainting pipeline with image, mask, and strength
            stage1_gen_args['image'] = processed_input_image
            stage1_gen_args['mask_image'] = processed_mask_image
            stage1_gen_args['strength'] = inpaint_strength
            print(f"Using inpainting with strength {inpaint_strength}")
        elif is_img2img:
            if is_true_img2img:
                # Use true img2img pipeline with image and strength
                stage1_gen_args['image'] = processed_input_image
                stage1_gen_args['strength'] = img2img_strength
                print(f"Using true img2img with strength {img2img_strength}")
            else:
                # Create img2img latents using noise interpolation
                img2img_latents = create_noise_interpolation_latents(
                    processed_input_image, noise_interpolation_strength, seed, width, height
                )
                stage1_gen_args['latents'] = img2img_latents
                print(f"Starting from noise interpolation latents, shape: {img2img_latents.shape}")
        
        # Generate first stage (or only stage)
        stage1_steps = num_inference_steps
        print(f"Stage 1: Generating with {stage1_steps} steps...")
        
        try:
            with torch.no_grad():
                stage1_result = pipe(**stage1_gen_args)
        except GenerationCancelledException as e:
            print(f"Generation cancelled: {e}")
            return None, None
        first_stage_image = stage1_result.images[0]
        
        # Handle two-stage generation
        if is_two_stage:
            # Save stage 1 result
            first_stage_name = f"{name}_stage1"
            first_stage_filename = f"{first_stage_name}_{width}x{height}_{seed}"
            first_stage_path = session_path / f"{first_stage_filename}.{config.output_format}"
            
            # Save stage 1 with metadata
            stage1_metadata = {
                'prompt': original_prompt,
                'enhanced_prompt': enhanced_prompt,
                'final_processed_prompt': prompt,
                'negative_prompt': negative_prompt,
                'applied_magic_text': applied_magic_text,
                'width': width,
                'height': height,
                'steps': stage1_steps,
                'cfg_scale': true_cfg_scale,
                'seed': seed,
                'session': session,
                'name': first_stage_name,
                'loras': loras,
                'model_info': manager.model_info.copy(),
                'is_stage1_of_two_stage': True,
                'two_stage_mode': two_stage_mode,
                'img2img_mode': img2img_mode if (is_img2img or is_inpaint) else None,
                'is_true_img2img': is_true_img2img,
                'is_inpaint': is_inpaint,
                'noise_interpolation_strength': noise_interpolation_strength if (two_stage_mode == "Img2Img Mode" or (is_img2img and not is_true_img2img)) else None,
                'img2img_strength': img2img_strength if is_true_img2img else None,
                'inpaint_strength': inpaint_strength if is_inpaint else None,
                'input_image_used': is_img2img or is_inpaint,
                'mask_image_used': is_inpaint
            }
            
            if config.enable_metadata_embed:
                save_image_with_metadata(first_stage_image, str(first_stage_path), stage1_metadata, manager.model_info)
            else:
                first_stage_image.save(first_stage_path)
            
            print(f"Stage 1 saved: {first_stage_path}")
            
            # Stage 2: Use img2img noise interpolation on the stage 1 result
            print(f"Stage 2: Noise interpolation with strength {noise_interpolation_strength} and {second_stage_steps} steps...")
            
            # Create stage 2 latents using the first stage image
            stage2_latents = create_img2img_latents(
                first_stage_image, noise_interpolation_strength, seed, width, height
            )
            
            # Prepare stage 2 generation arguments
            stage2_gen_args = gen_args.copy()
            stage2_gen_args['latents'] = stage2_latents
            stage2_gen_args['num_inference_steps'] = second_stage_steps
            
            try:
                with torch.no_grad():
                    stage2_result = pipe(**stage2_gen_args)
            except GenerationCancelledException as e:
                print(f"Stage 2 generation cancelled: {e}")
                return None, None
            
            final_image = stage2_result.images[0]
            print("Stage 2: Noise interpolation complete!")
        else:
            # Single-stage generation
            final_image = first_stage_image
        
        # Prepare enhanced metadata with two-stage info and template details
        metadata = {
            'prompt': original_prompt,
            'enhanced_prompt': enhanced_prompt,
            'final_processed_prompt': prompt,
            'negative_prompt': negative_prompt,
            'applied_magic_text': applied_magic_text,
            'width': width,
            'height': height,
            'steps': num_inference_steps,
            'cfg_scale': true_cfg_scale,
            'seed': seed,
            'session': session,
            'name': name,
            'loras': loras,
            'model_info': manager.model_info.copy(),
            # Generation mode metadata
            'is_two_stage': is_two_stage,
            'is_img2img': is_img2img,
            'is_inpaint': is_inpaint,
            'img2img_mode': img2img_mode if (is_img2img or is_inpaint) else None,
            'is_true_img2img': is_true_img2img,
            'two_stage_mode': two_stage_mode if is_two_stage else None,
            'noise_interpolation_strength': noise_interpolation_strength if (is_two_stage or (is_img2img and not is_true_img2img)) else None,
            'img2img_strength': img2img_strength if is_true_img2img else None,
            'inpaint_strength': inpaint_strength if is_inpaint else None,
            'first_stage_steps': stage1_steps if is_two_stage else None,
            'second_stage_steps': second_stage_steps if is_two_stage else None,
            'first_stage_image_path': str(first_stage_path) if first_stage_path else None,
            'input_image_used': is_img2img or is_inpaint,
            'mask_image_used': is_inpaint
        }
        
        # Save with metadata
        if is_two_stage:
            filename = f"{name}_stage2_{width}x{height}_{seed}"
        else:
            filename = f"{name}_{width}x{height}_{seed}"
        save_path = session_path / f"{filename}.{config.output_format}"
        
        if config.enable_metadata_embed:
            save_path = save_image_with_metadata(
                final_image, str(save_path), metadata, manager.model_info
            )
        else:
            final_image.save(save_path)
        
        print(f"Final image saved: {save_path}")
        return final_image, str(save_path)
        
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None



