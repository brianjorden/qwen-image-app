"""
Per-step callback for generation pipeline.
This is a placeholder that can be expanded with actual functionality.
"""

from typing import Dict, Any, Optional
import torch

# Global cancellation flag accessible from UI
_should_cancel_generation = False


def request_cancellation():
    """Request cancellation of current generation."""
    global _should_cancel_generation
    _should_cancel_generation = True


def clear_cancellation():
    """Clear cancellation flag."""
    global _should_cancel_generation
    _should_cancel_generation = False


def should_cancel():
    """Check if generation should be cancelled."""
    return _should_cancel_generation


class GenerationCancelledException(Exception):
    """Exception raised when generation is cancelled."""
    pass


def step_callback(
    pipe,
    step: int,
    timestep: int,
    callback_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Callback function called at each denoising step.
    
    This is called by the diffusion pipeline at each step if enabled.
    Can be used for logging, visualization, or dynamic adjustments.
    
    Args:
        pipe: The diffusion pipeline instance
        step: Current step number
        timestep: Current timestep value
        callback_kwargs: Dictionary containing:
            - latents: Current latent tensor
            - prompt_embeds: Prompt embeddings
            - negative_prompt_embeds: Negative prompt embeddings (if using CFG)
            
    Returns:
        Modified callback_kwargs (can alter latents, etc.)
        
    Raises:
        GenerationCancelledException: If generation should be cancelled
    """
    # Check for cancellation request at each step
    if should_cancel():
        print(f"Generation cancelled at step {step}")
        raise GenerationCancelledException(f"Generation cancelled at step {step}")
    
    # Optional: Log progress for debugging
    # print(f"Step {step}/{timestep}")
    
    return callback_kwargs


def setup_step_callback(enabled: bool = False, save_steps: bool = False, 
                        session_path=None, name=None, vae=None) -> Optional[callable]:
    """Get the step callback if enabled.
    
    Args:
        enabled: Whether to use the callback
        save_steps: Whether to save intermediate images
        session_path: Path to session directory for saving
        name: Base name for saved images
        vae: VAE model for decoding latents to images
        
    Returns:
        Callback function or None
    """
    # Always return a callback for cancellation checking, even if other features are disabled
    if save_steps:
        return create_step_saving_callback(session_path, name, vae)
    else:
        return step_callback  # Always return the basic callback for cancellation


def create_step_saving_callback(session_path, name, vae):
    """Create a callback that saves intermediate images.
    
    Args:
        session_path: Path to session directory
        name: Base name for images
        vae: VAE for decoding latents
        
    Returns:
        Callback function
    """
    def step_saving_callback(*args, **kwargs):
        try:
            # Extract step number first for cancellation check
            if len(args) >= 2:
                step = args[1]
            else:
                step = kwargs.get('step', 0)
                
            # Check for cancellation request at each step
            if should_cancel():
                print(f"Generation cancelled at step {step}")
                raise GenerationCancelledException(f"Generation cancelled at step {step}")
            
            # Basic callback debug info
            print(f"Step callback called for step {step}")
            
            # Try to extract parameters based on common diffusers callback patterns
            if len(args) >= 4:
                # Standard diffusers callback: (pipe, step, timestep, callback_kwargs)
                pipe, step, timestep, callback_kwargs = args[0], args[1], args[2], args[3]
            elif len(args) >= 3:
                # Missing callback_kwargs - use empty dict
                pipe, step, timestep = args[0], args[1], args[2]
                callback_kwargs = kwargs
            else:
                # Alternative format - extract from kwargs
                step = kwargs.get('step', 0)
                timestep = kwargs.get('timestep', 0)
                callback_kwargs = kwargs.get('callback_kwargs', kwargs)
            
            if isinstance(callback_kwargs, dict) and 'latents' in callback_kwargs and vae is not None:
                # Decode latents to image
                latents = callback_kwargs['latents']
                # For QwenImage, we'll apply the proper normalization later
                # Standard VAE scaling doesn't apply here
                
                with torch.no_grad():
                    # QwenImage pipeline conversion: transformer format -> VAE format
                    if latents.dim() == 3:
                        # Need to unpack from transformer format (B, num_patches, channels) to VAE format
                        batch_size, num_patches, channels = latents.shape
                        
                        # Infer spatial dimensions - for 512x512 with 8x scale factor, we get 64x64 spatial
                        # But also need to account for packing that requires divisibility by 2
                        # So 64x64 becomes 32x32 patches = 1024 patches, but we have 64, so it's 8x8 patches
                        spatial_size = int(num_patches ** 0.5)  # Should be 8 for 64 patches
                        if spatial_size * spatial_size != num_patches:
                            print(f"Cannot decode step latents - num_patches {num_patches} is not a perfect square")
                            return callback_kwargs
                        
                        # Calculate actual dimensions based on VAE scale factor and packing
                        height = spatial_size * 2  # Unpack requires *2
                        width = spatial_size * 2
                        
                        # Apply the same unpacking logic as the pipeline
                        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
                        latents = latents.permute(0, 3, 1, 4, 2, 5)
                        latents = latents.reshape(batch_size, channels // 4, 1, height, width)
                        
                        print(f"Unpacked latents from {(batch_size, num_patches, channels)} to {latents.shape}")
                    
                    # Apply VAE mean/std normalization (from pipeline lines 717-725)
                    if hasattr(vae.config, 'latents_mean') and hasattr(vae.config, 'latents_std'):
                        latents_mean = (
                            torch.tensor(vae.config.latents_mean)
                            .view(1, vae.config.z_dim, 1, 1, 1)
                            .to(latents.device, latents.dtype)
                        )
                        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                            latents.device, latents.dtype
                        )
                        latents = latents / latents_std + latents_mean
                        print(f"Applied VAE normalization")
                    
                    # Scale latents (skip the scaling factor division since we already have proper normalization)
                    # latents = latents / scaling_factor  # Skip this for QwenImage
                    
                    # Decode with VAE
                    image = vae.decode(latents, return_dict=False)[0]
                    
                    # QwenImage VAE returns 5D, extract frame 0 (like pipeline line 726)
                    if image.dim() == 5:
                        image = image[:, :, 0]  # Remove frame dimension
                
                # Convert to PIL image
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                
                from PIL import Image
                import numpy as np
                
                # Convert first image in batch
                image = (image[0] * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
                
                # Save step image
                step_path = session_path / f"{name}_step_{step:03d}.png"
                pil_image.save(step_path)
                print(f"Saved step {step} image: {step_path}")
                
        except Exception as e:
            print(f"Step saving failed at step {step if 'step' in locals() else 'unknown'}: {e}")
            import traceback
            traceback.print_exc()
        
        # Return the callback_kwargs if it exists, otherwise return kwargs
        return callback_kwargs if 'callback_kwargs' in locals() else kwargs
    
    return step_saving_callback
