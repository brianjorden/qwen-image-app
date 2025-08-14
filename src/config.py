"""
Simplified, flat configuration system with full validation.
All config fields must be explicitly defined - no hidden defaults.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import os


class Config:
    """Flat configuration with required field validation."""
    
    # Define all required fields
    REQUIRED_FIELDS = [
        # Model paths
        'model_diffusion', 'model_text_encoder', 'model_vae', 
        'model_tokenizer', 'model_scheduler',
        # GPU allocation
        'gpu_text_encoder', 'gpu_transformer', 'gpu_vae',
        # Model loading
        'torch_dtype', 'low_cpu_mem_usage', 'local_files_only', 'enable_vae_tiling',
        # Generation defaults
        'default_steps', 'default_cfg', 'default_width', 'default_height', 
        'default_seed', 'default_negative', 'default_noise_interpolation_strength',
        # Resolution presets
        'resolution_1_1', 'resolution_16_9', 'resolution_9_16',
        'resolution_4_3', 'resolution_3_4',
        # Features
        'enable_step_callback', 'enable_lora', 'enable_prompt_enhancement',
        'enable_metadata_embed',
        # Prompt processing
        'prompt_template_drop_idx', 'template_magic',
        'prompt_max_tokens', 'max_context_length', 'default_max_new_tokens',
        'prompt_enhance_max_tokens', 'enable_per_step_saving',
        # Output
        'output_directory', 'output_format', 'output_quality',
        # LoRA
        'lora_directory', 'lora_max_count',
        # Advanced
        'resolution_multiple', 'enable_attention_slicing', 
        'enable_cpu_offload', 'num_workers',
        # UI
        'ui_theme'
    ]
    
    # Optional fields that can be null
    OPTIONAL_FIELDS = [
        'auto_load_pipeline', 'template_system', 'template_chat', 
        'template_image', 'template_enhancement', 'template_describe'
    ]
    
    def __init__(self, config_path: str):
        """Load and validate configuration from YAML file."""
        self.config_path = Path(config_path)
        self._config = self._load_and_validate()
        self._expand_paths()
        self._parse_gpu_allocations()
        self._parse_resolutions()
    
    def _load_and_validate(self) -> Dict[str, Any]:
        """Load YAML and validate all required fields exist."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check all required fields
        missing = []
        for field in self.REQUIRED_FIELDS:
            if field not in config:
                missing.append(field)
        
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")
        
        # Add optional fields if not present
        for field in self.OPTIONAL_FIELDS:
            if field not in config:
                config[field] = None
        
        return config
    
    def _expand_paths(self):
        """Expand all path fields to absolute paths."""
        path_fields = [
            'model_diffusion', 'model_text_encoder',
            'model_vae', 'model_tokenizer', 'model_scheduler',
            'output_directory', 'lora_directory'
        ]
        
        for field in path_fields:
            value = self._config.get(field)
            if value and isinstance(value, str):
                self._config[field] = str(Path(value).expanduser().resolve())
        
        # Create directories if they don't exist
        for dir_field in ['output_directory', 'lora_directory']:
            dir_path = self._config.get(dir_field)
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _parse_gpu_allocations(self):
        """Parse GPU allocation strings into dictionaries."""
        self.gpu_maps = {}
        
        for field in ['gpu_text_encoder', 'gpu_transformer']:
            value = self._config.get(field)
            if not value or value == "null":
                self.gpu_maps[field] = {}
                continue
            
            # Handle simple device format (e.g., "cuda:0")
            if value.startswith("cuda:") and ":" not in value[5:]:
                self.gpu_maps[field] = {"device": value}
                continue
            
            # Parse "gpu:mem,gpu:mem" format
            gpu_map = {}
            for part in value.split(","):
                if ":" in part:
                    gpu_id, mem = part.split(":", 1)
                    gpu_map[int(gpu_id)] = mem
            self.gpu_maps[field] = gpu_map
    
    def _parse_resolutions(self):
        """Parse resolution preset strings into tuples."""
        self.resolutions = {}
        
        for key in ['resolution_1_1', 'resolution_16_9', 'resolution_9_16',
                    'resolution_4_3', 'resolution_3_4']:
            value = self._config.get(key)
            if value:
                w, h = value.split(",")
                aspect = key.replace("resolution_", "").replace("_", ":")
                self.resolutions[aspect] = (int(w), int(h))
    
    def get(self, key: str, default=None) -> Any:
        """Get config value with optional default."""
        return self._config.get(key, default)
    
    def __getattr__(self, key: str) -> Any:
        """Allow attribute-style access to config values."""
        if key.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        
        if key in self._config:
            return self._config[key]
        elif key == 'gpu_maps':
            return self.gpu_maps
        elif key == 'resolutions':
            return self.resolutions
        else:
            raise AttributeError(f"Config has no field '{key}'")
    
    def get_gpu_map(self, component: str) -> Dict:
        """Get parsed GPU allocation for a component."""
        return self.gpu_maps.get(f"gpu_{component}", {})
    
    def get_resolution(self, aspect_ratio: str) -> Tuple[int, int]:
        """Get resolution tuple for aspect ratio."""
        return self.resolutions.get(aspect_ratio, (self.default_width, self.default_height))
    
    def validate_models_exist(self):
        """Validate that model paths actually exist."""
        model_fields = ['model_diffusion', 'model_text_encoder', 'model_vae',
                       'model_tokenizer', 'model_scheduler']
        
        missing = []
        for field in model_fields:
            path = self._config.get(field)
            if path and not Path(path).exists():
                missing.append(f"{field}: {path}")
        
        if missing:
            raise FileNotFoundError(f"Model paths not found:\n" + "\n".join(missing))
    
    def save(self, path: Optional[str] = None) -> Path:
        """Save current config to file."""
        save_path = Path(path) if path else self.config_path
        
        # Create a clean copy for saving (unexpand paths for readability)
        save_config = self._config.copy()
        
        # Convert absolute paths back to ~ notation where applicable
        home = str(Path.home())
        for field, value in save_config.items():
            if isinstance(value, str) and value.startswith(home):
                save_config[field] = value.replace(home, "~", 1)
        
        with open(save_path, 'w') as f:
            yaml.dump(save_config, f, default_flow_style=False, 
                     indent=2, sort_keys=False)
        
        return save_path
    
    def reload(self):
        """Reload configuration from file."""
        self._config = self._load_and_validate()
        self._expand_paths()
        self._parse_gpu_allocations()
        self._parse_resolutions()
    
    def to_dict(self) -> Dict[str, Any]:
        """Get config as dictionary."""
        return self._config.copy()


# Global config instance
_config: Optional[Config] = None


def init_config(config_path: str = "config.yaml") -> Config:
    """Initialize global config from file."""
    global _config
    
    # Set environment variables if needed
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        print("Note: HF_HUB_OFFLINE not set, models may download if not found locally")
    
    _config = Config(config_path)
    _config.validate_models_exist()
    
    return _config


def get_config() -> Config:
    """Get the global config instance."""
    if _config is None:
        raise RuntimeError("Config not initialized. Call init_config() first.")
    return _config


def config_exists() -> bool:
    """Check if config has been initialized."""
    return _config is not None
