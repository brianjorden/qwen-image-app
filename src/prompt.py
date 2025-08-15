"""
Prompt processing, template wrapping, and enhancement functionality.
"""

import torch
from typing import Tuple, Optional, List, Dict, Any
from transformers import Qwen2Tokenizer

from .config import get_config
from pathlib import Path




def get_prompt_template() -> str:
    """Get image generation template from file.
    
    This is used by the analysis module to understand prompt processing.
    """
    config = get_config()
    if hasattr(config, 'template_image') and config.template_image:
        try:
            path = Path(config.template_image)
            if path.exists():
                return path.read_text(encoding='utf-8').strip()
        except Exception as e:
            print(f"Failed to load image template: {e}")
    
    # Fallback template if file not found
    return "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n"


def get_enhancement_template(language: str = 'en') -> str:
    """Get enhancement template from file."""
    config = get_config()
    template_key = 'template_enhancement_zh' if language == 'zh' else 'template_enhancement'
    
    if hasattr(config, template_key) and getattr(config, template_key):
        try:
            path = Path(getattr(config, template_key))
            if path.exists():
                return path.read_text(encoding='utf-8').strip()
        except Exception as e:
            print(f"Failed to load enhancement template ({language}): {e}")
    
    raise FileNotFoundError(f"Enhancement template file not found or not configured for language: {language}")


def detect_language(text: str) -> str:
    """Detect if text is Chinese or English.
    
    Args:
        text: Text to analyze
        
    Returns:
        'zh' for Chinese, 'en' for English
    """
    chinese_ranges = [
        ('\u4e00', '\u9fff'),  # CJK Unified Ideographs
    ]
    
    for char in text:
        if any(start <= char <= end for start, end in chinese_ranges):
            return 'zh'
    return 'en'






def add_magic_prompt(prompt: str, language: Optional[str] = None) -> str:
    """Add quality enhancement suffix based on language.
    
    Args:
        prompt: Original prompt
        language: Optional language override ('en' or 'zh')
        
    Returns:
        Prompt with magic suffix added
    """
    config = get_config()
    
    if language is None:
        language = detect_language(prompt)
    
    # Load magic from template file
    if hasattr(config, 'template_magic') and config.template_magic:
        try:
            path = Path(config.template_magic)
            if path.exists():
                magic = path.read_text(encoding='utf-8').strip()
            else:
                return prompt  # No magic template found, return unchanged
        except Exception as e:
            print(f"Failed to load magic template: {e}")
            return prompt  # Failed to load, return unchanged
    else:
        return prompt  # No magic template configured, return unchanged
    
    # Don't add if already present
    if magic not in prompt:
        return f"{prompt}{magic}"
    return prompt


def count_tokens(text: str, tokenizer: Optional[Qwen2Tokenizer] = None) -> Tuple[int, bool]:
    """Count tokens in text and check if truncation would occur.
    
    Args:
        text: Text to count tokens for
        tokenizer: Optional tokenizer instance
        
    Returns:
        Tuple of (token_count, would_truncate)
    """
    config = get_config()
    
    if tokenizer is None:
        tokenizer = Qwen2Tokenizer.from_pretrained(
            config.model_tokenizer,
            local_files_only=config.local_files_only
        )
    
    tokens = tokenizer(
        text,
        max_length=config.prompt_max_tokens,
        padding=False,
        truncation=True,
        return_tensors="pt"
    )
    
    actual_tokens = tokenizer(
        text,
        padding=False,
        truncation=False,
        return_tensors="pt"
    )
    
    count = tokens.input_ids.shape[1]
    would_truncate = actual_tokens.input_ids.shape[1] > config.prompt_max_tokens
    
    return count, would_truncate


def enhance_prompt_local(
    prompt: str,
    text_encoder: Any,
    tokenizer: Optional[Qwen2Tokenizer] = None,
    max_new_tokens: Optional[int] = None
) -> str:
    """Enhance a prompt using the local VL model.
    
    Args:
        prompt: Original prompt to enhance
        text_encoder: Loaded text encoder model
        tokenizer: Optional tokenizer instance
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Enhanced prompt
    """
    config = get_config()
    
    if tokenizer is None:
        tokenizer = Qwen2Tokenizer.from_pretrained(
            config.model_tokenizer,
            local_files_only=config.local_files_only
        )
    
    if max_new_tokens is None:
        max_new_tokens = config.prompt_enhance_max_tokens
    
    # Use the model manager's chat functionality for proper VL model interaction
    from .models import get_model_manager
    manager = get_model_manager()
    
    # Detect language and build enhancement prompt
    lang = detect_language(prompt)
    system_prompt = get_enhancement_template(lang).format(prompt)
    
    # Create messages for VL model chat
    messages = [{
        "role": "user",
        "content": system_prompt
    }]
    
    # Use the chat method for proper VL model interaction
    try:
        enhanced = manager.chat(messages, max_new_tokens=max_new_tokens)
        return enhanced.strip()
    except Exception as e:
        print(f"VL chat enhancement failed: {e}, falling back to direct generation")
        # Continue with direct generation as fallback
    
    # Fallback: Tokenize  
    inputs = tokenizer(
        system_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Move to encoder device
    device = next(text_encoder.parameters()).device
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # Generate enhancement
    with torch.no_grad():
        outputs = text_encoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    enhanced = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Add magic prompt
    enhanced = add_magic_prompt(enhanced.strip(), lang)
    
    return enhanced


def blend_negative_prompts(
    negative_prompts: List[str],
    weights: Optional[List[float]] = None
) -> str:
    """Blend multiple negative prompts together.
    
    Args:
        negative_prompts: List of negative prompts
        weights: Optional weights for each prompt
        
    Returns:
        Blended negative prompt
    """
    if not negative_prompts:
        return ""
    
    if len(negative_prompts) == 1:
        return negative_prompts[0]
    
    # For now, simple concatenation with commas
    # Could implement more sophisticated blending with embeddings later
    return ", ".join(p.strip() for p in negative_prompts if p.strip())


