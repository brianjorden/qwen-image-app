"""
Prompt processing, template wrapping, and enhancement functionality.
"""

import torch
from typing import Tuple, Optional, List, Dict, Any
from transformers import Qwen2Tokenizer

from .config import get_config
from pathlib import Path


def get_image_template() -> str:
    """Get image generation template from file."""
    config = get_config()
    if hasattr(config, 'template_image') and config.template_image:
        try:
            path = Path(config.template_image)
            if path.exists():
                return path.read_text(encoding='utf-8').strip()
        except Exception as e:
            print(f"Failed to load image template: {e}")
    
    raise FileNotFoundError("Image template file not found or not configured")

# Load template dynamically
def get_prompt_template() -> str:
    """Get the current prompt template."""
    return get_image_template()

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


def apply_template(prompt: str, tokenizer: Optional[Qwen2Tokenizer] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Apply the training template to a prompt and tokenize.
    
    Args:
        prompt: Raw prompt text
        tokenizer: Optional tokenizer instance (will load if not provided)
        
    Returns:
        Tuple of (input_ids, attention_mask, content_tokens)
    """
    config = get_config()
    
    if tokenizer is None:
        tokenizer = Qwen2Tokenizer.from_pretrained(
            config.model_tokenizer,
            local_files_only=config.local_files_only
        )
    
    # Apply template
    templated = get_prompt_template().format(prompt)
    
    # Tokenize with proper length limits
    tokens = tokenizer(
        templated,
        max_length=config.prompt_max_tokens + config.prompt_template_drop_idx,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Calculate content tokens (after dropping template)
    total_tokens = tokens.attention_mask.sum().item()
    content_tokens = max(0, total_tokens - config.prompt_template_drop_idx)
    
    return tokens.input_ids, tokens.attention_mask, content_tokens


def extract_content_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
    """Extract content hidden states after dropping template tokens.
    
    Args:
        hidden_states: Full hidden states from encoder [B, T, H]
        attention_mask: Attention mask [B, T]
        
    Returns:
        List of content hidden states per batch item
    """
    config = get_config()
    drop_idx = config.prompt_template_drop_idx
    
    # Get valid lengths for each batch item
    bool_mask = attention_mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    
    # Extract hidden states
    batch_size = hidden_states.shape[0]
    content_hidden = []
    
    for i in range(batch_size):
        valid_len = valid_lengths[i].item()
        start_idx = min(drop_idx, valid_len)
        end_idx = valid_len
        
        if end_idx > start_idx:
            content_hidden.append(hidden_states[i, start_idx:end_idx])
        else:
            # No content tokens
            content_hidden.append(hidden_states[i, :0])  # Empty tensor
    
    return content_hidden


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


def prepare_prompts_for_pipeline(
    prompt: str,
    negative_prompt: str = "",
    apply_template_flag: bool = True,
    add_magic: bool = True
) -> Tuple[str, str]:
    """Prepare prompts for the generation pipeline.
    
    Args:
        prompt: Positive prompt
        negative_prompt: Negative prompt
        apply_template_flag: Whether to apply the training template
        add_magic: Whether to add magic quality prompts
        
    Returns:
        Tuple of (processed_prompt, processed_negative)
    """
    # Add magic prompt if requested
    if add_magic:
        prompt = add_magic_prompt(prompt)
    
    # Template is applied in the pipeline, we just return the prompts
    # The pipeline will handle the template wrapping and token dropping
    
    # Ensure negative prompt is at least a space if CFG is being used
    config = get_config()
    if not negative_prompt and config.default_cfg > 1.0:
        negative_prompt = " "
    
    return prompt, negative_prompt
