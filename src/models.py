"""
Enhanced model management with individual component control and template handling.
"""

import torch
import gc
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    AutoProcessor
)
from diffusers import DiffusionPipeline, AutoencoderKL, AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, QwenImagePipeline
from peft import PeftModel

from .config import get_config
from .prompt import apply_template, extract_content_hidden_states


class ModelManager:
    """Enhanced model manager with individual component control."""
    
    def __init__(self):
        # Individual components
        self.text_encoder = None
        self.text_encoder_alt = None  # Secondary encoder
        self.tokenizer = None
        self.processor = None  # For VL chat functionality
        self.transformer = None
        self.vae = None
        self.scheduler = None
        self.pipe = None
        
        # Component status
        self.component_status = {
            'text_encoder': False,
            'text_encoder_alt': False,
            'tokenizer': False,
            'processor': False,
            'transformer': False,
            'vae': False,
            'scheduler': False,
            'pipeline': False
        }
        
        # LoRA models
        self.loras = []
        
        # Model info for metadata
        self.model_info = {}
    
    def load_tokenizer(self, force_reload: bool = False) -> Qwen2Tokenizer:
        """Load tokenizer component.
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded tokenizer
        """
        if self.tokenizer and not force_reload:
            return self.tokenizer
        
        config = get_config()
        print(f"Loading tokenizer from {config.model_tokenizer}")
        
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            config.model_tokenizer,
            local_files_only=config.local_files_only
        )
        
        self.component_status['tokenizer'] = True
        self.model_info['tokenizer'] = Path(config.model_tokenizer).name
        
        return self.tokenizer
    
    def load_processor(self, force_reload: bool = False) -> AutoProcessor:
        """Load processor for VL model chat.
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded processor
        """
        if self.processor and not force_reload:
            return self.processor
        
        config = get_config()
        print(f"Loading processor from {config.model_text_encoder}")
        
        self.processor = AutoProcessor.from_pretrained(
            config.model_text_encoder,
            local_files_only=config.local_files_only
        )
        
        self.component_status['processor'] = True
        self.model_info['processor'] = Path(config.model_text_encoder).name
        
        return self.processor
    
    def load_text_encoder(self, use_alt: bool = False, force_reload: bool = False) -> Any:
        """Load text encoder component.
        
        Args:
            use_alt: Load alternative encoder
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded text encoder
        """
        encoder_attr = 'text_encoder_alt' if use_alt else 'text_encoder'
        encoder = getattr(self, encoder_attr)
        
        if encoder and not force_reload:
            return encoder
        
        config = get_config()
        model_path = config.model_text_encoder_alt if use_alt else config.model_text_encoder
        
        if use_alt and not model_path:
            print("No alternative text encoder configured")
            return None
        
        print(f"Loading {'alternative' if use_alt else 'primary'} text encoder from {model_path}")
        
        # Get GPU allocation
        gpu_key = 'gpu_text_encoder_alt' if use_alt else 'gpu_text_encoder'
        gpu_map = config.get_gpu_map(gpu_key[4:])  # Remove 'gpu_' prefix
        
        # Load config
        enc_cfg = Qwen2_5_VLConfig.from_pretrained(
            model_path,
            local_files_only=config.local_files_only
        )
        
        # Create empty model to get structure
        with init_empty_weights():
            empty_enc = Qwen2_5_VLForConditionalGeneration(enc_cfg)
        
        # Find layer container for no-split
        layer_container = None
        for cand in ("model.language_model.layers", "model.layers"):
            obj = empty_enc
            ok = True
            for part in cand.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if ok and hasattr(obj, "__len__") and len(obj):
                layer_container = obj
                break
        
        if layer_container:
            enc_layer_cls = type(layer_container[0]).__name__
        else:
            enc_layer_cls = None
        
        # Infer device map
        device_map = infer_auto_device_map(
            empty_enc,
            max_memory=gpu_map,
            dtype=getattr(torch, config.torch_dtype),
            no_split_module_classes=[enc_layer_cls] if enc_layer_cls else None,
        )
        
        # Load model
        encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, config.torch_dtype),
            device_map=device_map,
            low_cpu_mem_usage=config.low_cpu_mem_usage,
            local_files_only=config.local_files_only,
        )
        
        setattr(self, encoder_attr, encoder)
        self.component_status[encoder_attr] = True
        self.model_info[encoder_attr] = Path(model_path).name
        
        print(f"{'Alternative' if use_alt else 'Primary'} text encoder loaded")
        return encoder
    
    def load_vae(self, force_reload: bool = False) -> AutoencoderKLQwenImage:
        """Load VAE component.
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded VAE
        """
        if self.vae and not force_reload:
            return self.vae
        
        config = get_config()
        print(f"Loading VAE from {config.model_vae}")
        
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            config.model_vae,
            torch_dtype=getattr(torch, config.torch_dtype),
            local_files_only=config.local_files_only
        )
        
        # Setup VAE device
        vae_device = config.gpu_vae
        if vae_device.startswith("cuda:"):
            self.vae = self.vae.to(vae_device)
        else:
            # Use device map
            vae_map = {"": vae_device}
            self.vae = dispatch_model(self.vae, device_map=vae_map)
        
        # Enable tiling if configured
        if config.enable_vae_tiling:
            self.vae.enable_tiling()
        
        self.component_status['vae'] = True
        self.model_info['vae'] = Path(config.model_vae).name
        
        print("VAE loaded")
        return self.vae
    
    def load_scheduler(self, force_reload: bool = False) -> FlowMatchEulerDiscreteScheduler:
        """Load scheduler component.
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded scheduler
        """
        if self.scheduler and not force_reload:
            return self.scheduler
        
        config = get_config()
        print(f"Loading scheduler from {config.model_scheduler}")
        
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.model_scheduler,
            local_files_only=config.local_files_only
        )
        
        self.component_status['scheduler'] = True
        self.model_info['scheduler'] = Path(config.model_scheduler).name
        
        print("Scheduler loaded")
        return self.scheduler
    
    def load_transformer(self, force_reload: bool = False) -> Any:
        """Load transformer component from diffusion model.
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded transformer
        """
        if self.transformer and not force_reload:
            return self.transformer
        
        config = get_config()
        print(f"Loading transformer from {config.model_diffusion}")
        
        # Load just the transformer component
        from diffusers import QwenImageTransformer2DModel
        
        transformer_path = Path(config.model_diffusion) / "transformer"
        self.transformer = QwenImageTransformer2DModel.from_pretrained(
            transformer_path,
            torch_dtype=getattr(torch, config.torch_dtype),
            local_files_only=config.local_files_only
        )
        
        # Distribute across GPUs
        gpu_map = config.get_gpu_map('transformer')
        
        # Find transformer block class
        block_cls = None
        for attr in ("transformer_blocks", "blocks", "layers"):
            if hasattr(self.transformer, attr):
                seq = getattr(self.transformer, attr)
                if len(seq):
                    block_cls = type(seq[0]).__name__
                    break
        
        # Infer device map
        device_map = infer_auto_device_map(
            self.transformer,
            max_memory=gpu_map,
            dtype=getattr(torch, config.torch_dtype),
            no_split_module_classes=[block_cls] if block_cls else None,
        )
        
        self.transformer = dispatch_model(self.transformer, device_map=device_map)
        
        self.component_status['transformer'] = True
        self.model_info['transformer'] = Path(config.model_diffusion).name
        
        print("Transformer loaded and distributed")
        return self.transformer
    
    def build_pipeline(self, force_rebuild: bool = False) -> DiffusionPipeline:
        """Build the complete pipeline from components.
        
        Args:
            force_rebuild: Force rebuild even if already built
            
        Returns:
            Assembled pipeline
        """
        if self.pipe and not force_rebuild:
            return self.pipe
        
        # Ensure all components are loaded
        if not self.text_encoder:
            self.load_text_encoder()
        if not self.tokenizer:
            self.load_tokenizer()
        if not self.transformer:
            self.load_transformer()
        if not self.vae:
            self.load_vae()
        if not self.scheduler:
            self.load_scheduler()
        
        config = get_config()
        print("Building diffusion pipeline from components")
        
        # Use the actual QwenImagePipeline from diffusers
        self.pipe = QwenImagePipeline(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler
        )
        
        self.component_status['pipeline'] = True
        print("Pipeline assembled")
        
        return self.pipe
    
    def load_lora(self, lora_path: str, strength: float = 1.0) -> bool:
        """Load a LoRA model.
        
        Args:
            lora_path: Path to LoRA model
            strength: LoRA strength (0-1)
            
        Returns:
            True if loaded successfully
        """
        if not self.transformer:
            print("Transformer must be loaded before LoRAs")
            return False
        
        try:
            print(f"Loading LoRA from {lora_path} with strength {strength}")
            
            # Apply LoRA to transformer
            self.transformer = PeftModel.from_pretrained(
                self.transformer,
                lora_path,
                adapter_name=Path(lora_path).name
            )
            
            self.loras.append({
                'name': Path(lora_path).name,
                'path': lora_path,
                'strength': strength
            })
            
            print(f"LoRA loaded: {Path(lora_path).name}")
            return True
            
        except Exception as e:
            print(f"Failed to load LoRA: {e}")
            return False
    
    def unload_component(self, component: str):
        """Unload a specific component.
        
        Args:
            component: Component name to unload
        """
        if component == 'text_encoder' and self.text_encoder:
            self.text_encoder = None
            self.component_status['text_encoder'] = False
        elif component == 'text_encoder_alt' and self.text_encoder_alt:
            self.text_encoder_alt = None
            self.component_status['text_encoder_alt'] = False
        elif component == 'tokenizer' and self.tokenizer:
            self.tokenizer = None
            self.component_status['tokenizer'] = False
        elif component == 'processor' and self.processor:
            self.processor = None
            self.component_status['processor'] = False
        elif component == 'transformer' and self.transformer:
            self.transformer = None
            self.component_status['transformer'] = False
        elif component == 'vae' and self.vae:
            self.vae = None
            self.component_status['vae'] = False
        elif component == 'scheduler' and self.scheduler:
            self.scheduler = None
            self.component_status['scheduler'] = False
        elif component == 'pipeline' and self.pipe:
            self.pipe = None
            self.component_status['pipeline'] = False
        
        # Clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Unloaded {component}")
    
    def unload_all(self):
        """Unload all components."""
        for component in ['pipeline', 'transformer', 'vae', 'scheduler', 
                         'text_encoder', 'text_encoder_alt', 'tokenizer', 'processor']:
            self.unload_component(component)
        
        self.loras = []
        self.model_info = {}
        print("All models unloaded")
    
    def chat(self, messages: List[Dict], max_new_tokens: int = 512) -> str:
        """Generate chat response using the VL model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Generated response text
        """
        if not self.text_encoder:
            raise RuntimeError("Text encoder not loaded")
        
        if not self.processor:
            self.load_processor()
        
        config = get_config()
        
        # Import here to avoid import at startup
        from qwen_vl_utils import process_vision_info
        
        try:
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process any vision inputs
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            device = next(self.text_encoder.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.text_encoder.generate(
                    **inputs,
                    max_new_tokens=min(max_new_tokens, config.max_context_length - inputs['input_ids'].shape[1]),
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Extract only the new tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            
            # Decode response
            response = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return response.strip()
            
        except Exception as e:
            print(f"Chat generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current model status.
        
        Returns:
            Dictionary of component statuses and info
        """
        return {
            'components': self.component_status.copy(),
            'model_info': self.model_info.copy(),
            'loras': self.loras.copy(),
            'memory_allocated': {
                f"cuda:{i}": torch.cuda.memory_allocated(i) / 1024**3
                for i in range(torch.cuda.device_count())
            } if torch.cuda.is_available() else {}
        }


# The custom pipeline wrapper has been removed since we now use the actual QwenImagePipeline


# Global model manager
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create the global model manager."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def get_pipe() -> Any:
    """Get the pipeline, loading if necessary."""
    manager = get_model_manager()
    return manager.build_pipeline()


def unload_models():
    """Unload all models."""
    manager = get_model_manager()
    manager.unload_all()


def reload_models():
    """Reload all models."""
    manager = get_model_manager()
    manager.unload_all()
    return manager.build_pipeline(force_rebuild=True)


def is_loaded() -> bool:
    """Check if pipeline is loaded."""
    manager = get_model_manager()
    return manager.component_status.get('pipeline', False)
