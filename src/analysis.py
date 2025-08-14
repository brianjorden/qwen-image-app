"""
Text encoder analysis tool for understanding prompt processing.
"""

import torch
from typing import Tuple, Dict, Any, Optional
from transformers import Qwen2Tokenizer

from .config import get_config
from .models import get_model_manager
from .prompt import get_prompt_template


class EncoderAnalyzer:
    """Analyze text encoder behavior and cross-encoder differences."""
    
    def __init__(self):
        self.manager = get_model_manager()
        self.config = get_config()
        self.tokenizer = None
    
    def _ensure_tokenizer(self) -> Qwen2Tokenizer:
        """Ensure tokenizer is loaded."""
        if not self.tokenizer:
            self.tokenizer = self.manager.load_tokenizer()
        return self.tokenizer
    
    def token_count(self, prompt: str, use_template: bool) -> Tuple[int, int, bool]:
        """Count tokens in prompt with optional template.
        
        Args:
            prompt: Input prompt
            use_template: Whether to apply template
            
        Returns:
            Tuple of (total_tokens, content_tokens, was_truncated)
        """
        tokenizer = self._ensure_tokenizer()
        
        text = get_prompt_template().format(prompt) if use_template else prompt
        drop_idx = self.config.prompt_template_drop_idx if use_template else 0
        
        tokens = tokenizer(
            text,
            max_length=self.config.prompt_max_tokens + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        total = int(tokens.attention_mask.sum().item())
        content = max(0, total - drop_idx)
        
        # Check if truncation occurred
        full_tokens = tokenizer(
            text,
            truncation=False,
            return_tensors="pt"
        )
        truncated = full_tokens.input_ids.shape[1] > (self.config.prompt_max_tokens + drop_idx)
        
        return total, content, truncated
    
    @torch.inference_mode()
    def argmax_roundtrip(
        self, 
        encoder: Any,
        prompt: str, 
        use_template: bool
    ) -> Tuple[str, Dict[str, Any]]:
        """Perform argmax roundtrip (encode then decode each position).
        
        Args:
            encoder: Text encoder to use
            prompt: Input prompt
            use_template: Whether to apply template
            
        Returns:
            Tuple of (decoded_text, metadata)
        """
        tokenizer = self._ensure_tokenizer()
        
        text = get_prompt_template().format(prompt) if use_template else prompt
        drop_idx = self.config.prompt_template_drop_idx if use_template else 0
        
        tokens = tokenizer(
            text,
            max_length=self.config.prompt_max_tokens + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Get encoder device
        device = next(encoder.parameters()).device
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        
        # Forward pass
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[-1]
        
        # Project through language model head
        logits = encoder.lm_head(hidden_states)
        pred_ids = logits.argmax(dim=-1)[0]
        
        # Extract content portion
        total_len = int(attention_mask.sum().item())
        start = min(drop_idx, total_len)
        end = total_len
        
        if end <= start:
            return "", {"templated_tokens": total_len, "content_tokens": 0}
        
        content_pred_ids = pred_ids[start:end]
        text_out = tokenizer.decode(content_pred_ids, skip_special_tokens=True)
        
        return text_out, {
            "templated_tokens": total_len,
            "content_tokens": end - start
        }
    
    @torch.inference_mode()
    def argmax_roundtrip_cross(
        self,
        encoder_from: Any,
        encoder_to: Any,
        prompt: str,
        use_template: bool
    ) -> Tuple[str, Dict[str, Any]]:
        """Cross-encoder argmax roundtrip (encode with one, decode with another).
        
        Args:
            encoder_from: Encoder for encoding
            encoder_to: Encoder for decoding (uses its LM head)
            prompt: Input prompt
            use_template: Whether to apply template
            
        Returns:
            Tuple of (decoded_text, metadata)
        """
        tokenizer = self._ensure_tokenizer()
        
        text = get_prompt_template().format(prompt) if use_template else prompt
        drop_idx = self.config.prompt_template_drop_idx if use_template else 0
        
        tokens = tokenizer(
            text,
            max_length=self.config.prompt_max_tokens + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode with first encoder
        device_from = next(encoder_from.parameters()).device
        input_ids = tokens.input_ids.to(device_from)
        attention_mask = tokens.attention_mask.to(device_from)
        
        outputs = encoder_from(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[-1]
        
        # Decode with second encoder's head
        device_to = next(encoder_to.lm_head.parameters()).device
        hidden_states = hidden_states.to(device_to)
        
        logits = encoder_to.lm_head(hidden_states)
        pred_ids = logits.argmax(dim=-1)[0]
        
        # Extract content
        total_len = int(attention_mask.sum().item())
        start = min(drop_idx, total_len)
        end = total_len
        
        if end <= start:
            return "", {"templated_tokens": total_len, "content_tokens": 0}
        
        content_pred_ids = pred_ids[start:end]
        text_out = tokenizer.decode(content_pred_ids, skip_special_tokens=True)
        
        return text_out, {
            "templated_tokens": total_len,
            "content_tokens": end - start
        }
    
    @torch.inference_mode()
    def greedy_continue(
        self,
        encoder: Any,
        prompt: str,
        use_template: bool,
        max_new_tokens: int = 96
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate continuation from assistant turn.
        
        Args:
            encoder: Text encoder to use
            prompt: Input prompt
            use_template: Whether to apply template
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (generated_text, metadata)
        """
        tokenizer = self._ensure_tokenizer()
        
        text = get_prompt_template().format(prompt) if use_template else prompt
        drop_idx = self.config.prompt_template_drop_idx if use_template else 0
        
        tokens = tokenizer(
            text,
            max_length=self.config.prompt_max_tokens + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        device = next(encoder.parameters()).device
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        
        # Generate
        gen_ids = encoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Extract new tokens
        prefix_len = input_ids.shape[1]
        new_ids = gen_ids[0, prefix_len:]
        text_out = tokenizer.decode(new_ids, skip_special_tokens=True)
        
        return text_out, {
            "templated_tokens": int(attention_mask.sum().item()),
            "generated_tokens": int(new_ids.numel())
        }
    
    @torch.inference_mode()
    def greedy_continue_cross(
        self,
        encoder_from: Any,
        encoder_to: Any,
        prompt: str,
        use_template: bool,
        max_new_tokens: int = 96
    ) -> Tuple[str, Dict[str, Any]]:
        """Cross-encoder greedy continuation.
        
        Args:
            encoder_from: Encoder for encoding
            encoder_to: Encoder for decoding head
            prompt: Input prompt
            use_template: Whether to apply template
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (generated_text, metadata)
        """
        tokenizer = self._ensure_tokenizer()
        
        text = get_prompt_template().format(prompt) if use_template else prompt
        drop_idx = self.config.prompt_template_drop_idx if use_template else 0
        
        tokens = tokenizer(
            text,
            max_length=self.config.prompt_max_tokens + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        device_from = next(encoder_from.parameters()).device
        input_ids = tokens.input_ids.to(device_from)
        attention_mask = tokens.attention_mask.to(device_from)
        
        generated = []
        eos_id = tokenizer.eos_token_id or 151643
        
        for _ in range(max_new_tokens):
            # Encode with first encoder
            outputs = encoder_from(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            hidden_states = outputs.hidden_states[-1]
            last_hidden = hidden_states[:, -1:, :]
            
            # Decode with second encoder's head
            device_to = next(encoder_to.lm_head.parameters()).device
            last_hidden = last_hidden.to(device_to)
            
            logits = encoder_to.lm_head(last_hidden)
            next_id = torch.argmax(logits[0, -1, :], dim=-1).item()
            
            generated.append(next_id)
            
            if next_id == eos_id:
                break
            
            # Append to sequence
            next_tensor = torch.tensor([[next_id]], device=device_from, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            
            ones = torch.ones((1, 1), device=device_from, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, ones], dim=1)
        
        if not generated:
            return "", {"templated_tokens": tokens.input_ids.shape[1], "generated_tokens": 0}
        
        text_out = tokenizer.decode(generated, skip_special_tokens=True)
        
        return text_out, {
            "templated_tokens": tokens.input_ids.shape[1],
            "generated_tokens": len(generated)
        }
    
    def analyze_prompt(
        self,
        prompt: str,
        mode: str = "greedy",
        use_template: bool = True,
        max_new_tokens: int = 96,
        use_cross: bool = False
    ) -> Dict[str, Any]:
        """Analyze prompt with available encoders.
        
        Args:
            prompt: Input prompt
            mode: Analysis mode ('greedy' or 'argmax')
            use_template: Whether to apply template
            max_new_tokens: Max tokens for greedy mode
            use_cross: Whether to do cross-encoder analysis
            
        Returns:
            Dictionary of analysis results
        """
        if not prompt.strip():
            return {"error": "Empty prompt"}
        
        # Ensure primary encoder is loaded
        if not self.manager.text_encoder:
            self.manager.load_text_encoder()
        
        results = {}
        
        # Token counting
        total, content, truncated = self.token_count(prompt, use_template)
        results['token_info'] = {
            'total': total,
            'content': content,
            'truncated': truncated,
            'template_used': use_template
        }
        
        # Primary encoder analysis
        if mode == "greedy":
            text, meta = self.greedy_continue(
                self.manager.text_encoder, prompt, use_template, max_new_tokens
            )
            results['primary'] = {
                'mode': 'greedy_continuation',
                'output': text,
                'metadata': meta
            }
        else:
            text, meta = self.argmax_roundtrip(
                self.manager.text_encoder, prompt, use_template
            )
            results['primary'] = {
                'mode': 'argmax_roundtrip',
                'output': text,
                'metadata': meta
            }
        
        # Alternative encoder if available
        if self.manager.text_encoder_alt:
            if mode == "greedy":
                text, meta = self.greedy_continue(
                    self.manager.text_encoder_alt, prompt, use_template, max_new_tokens
                )
                results['alternative'] = {
                    'mode': 'greedy_continuation',
                    'output': text,
                    'metadata': meta
                }
            else:
                text, meta = self.argmax_roundtrip(
                    self.manager.text_encoder_alt, prompt, use_template
                )
                results['alternative'] = {
                    'mode': 'argmax_roundtrip',
                    'output': text,
                    'metadata': meta
                }
            
            # Cross-encoder analysis if requested
            if use_cross:
                if mode == "greedy":
                    # Primary encode -> Alt decode
                    text, meta = self.greedy_continue_cross(
                        self.manager.text_encoder,
                        self.manager.text_encoder_alt,
                        prompt, use_template, max_new_tokens
                    )
                    results['cross_primary_alt'] = {
                        'mode': 'greedy_cross',
                        'direction': 'primary_encode_alt_decode',
                        'output': text,
                        'metadata': meta
                    }
                    
                    # Alt encode -> Primary decode
                    text, meta = self.greedy_continue_cross(
                        self.manager.text_encoder_alt,
                        self.manager.text_encoder,
                        prompt, use_template, max_new_tokens
                    )
                    results['cross_alt_primary'] = {
                        'mode': 'greedy_cross',
                        'direction': 'alt_encode_primary_decode',
                        'output': text,
                        'metadata': meta
                    }
                else:
                    # Primary encode -> Alt decode
                    text, meta = self.argmax_roundtrip_cross(
                        self.manager.text_encoder,
                        self.manager.text_encoder_alt,
                        prompt, use_template
                    )
                    results['cross_primary_alt'] = {
                        'mode': 'argmax_cross',
                        'direction': 'primary_encode_alt_decode',
                        'output': text,
                        'metadata': meta
                    }
                    
                    # Alt encode -> Primary decode
                    text, meta = self.argmax_roundtrip_cross(
                        self.manager.text_encoder_alt,
                        self.manager.text_encoder,
                        prompt, use_template
                    )
                    results['cross_alt_primary'] = {
                        'mode': 'argmax_cross',
                        'direction': 'alt_encode_primary_decode',
                        'output': text,
                        'metadata': meta
                    }
        
        return results


# Global analyzer
_analyzer: Optional[EncoderAnalyzer] = None


def get_analyzer() -> EncoderAnalyzer:
    """Get or create the global analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = EncoderAnalyzer()
    return _analyzer
