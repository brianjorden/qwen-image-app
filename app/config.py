"""
Config tab UI for the Qwen-Image application.
"""

import gradio as gr
import yaml
from pathlib import Path
from typing import Tuple

from src.config import init_config, get_config
from .shared import get_config_yaml


def create_config_tab() -> None:
    """Create the configuration management tab."""
    with gr.Column():
        gr.Markdown("### Configuration Management")
        
        with gr.Row():
            config_file = gr.File(
                label="Load Configuration",
                file_types=[".yaml", ".yml"],
                file_count="single"
            )
            load_config_btn = gr.Button("Load Config")
        
        config_editor = gr.Textbox(
            label="Configuration (YAML)",
            lines=20,
            value=get_config_yaml(),
            interactive=True
        )
        
        with gr.Row():
            save_config_btn = gr.Button("Save Config", variant="primary")
            reload_config_btn = gr.Button("Reload from File")
            validate_config_btn = gr.Button("Validate")
        
        config_status = gr.Textbox(
            label="Status",
            lines=3,
            interactive=False
        )
    
    # Setup event handlers
    _setup_config_handlers(
        config_file, load_config_btn, config_editor,
        save_config_btn, reload_config_btn, validate_config_btn,
        config_status
    )


def _setup_config_handlers(
    config_file, load_config_btn, config_editor,
    save_config_btn, reload_config_btn, validate_config_btn,
    config_status
):
    """Setup all event handlers for the config tab."""
    
    def load_config_file(file_obj) -> Tuple[str, str]:
        """Load config from file."""
        if not file_obj:
            return get_config_yaml(), "No file provided"
        
        try:
            # Reload config from new file
            init_config(file_obj.name)
            config = get_config()
            
            return get_config_yaml(), f"Loaded config from {file_obj.name}"
            
        except Exception as e:
            return get_config_yaml(), f"Failed to load: {str(e)}"
    
    def save_config(yaml_text: str) -> str:
        """Save edited config."""
        try:
            # Parse YAML
            config_dict = yaml.safe_load(yaml_text)
            
            # Save to temp file then reload
            temp_path = Path("temp_config.yaml")
            temp_path.parent.mkdir(exist_ok=True)
            
            with open(temp_path, 'w') as f:
                yaml.dump(config_dict, f)
            
            # Try to reload
            init_config(str(temp_path))
            config = get_config()
            
            # Save to default location (config.yaml)
            config.save("config.yaml")
            
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
            
            return "Configuration saved and reloaded"
            
        except Exception as e:
            return f"Failed to save: {str(e)}"
    
    def reload_config() -> Tuple[str, str]:
        """Reload config from file."""
        try:
            config = get_config()
            config.reload()
            return get_config_yaml(), "Configuration reloaded"
        except Exception as e:
            return get_config_yaml(), f"Failed to reload: {str(e)}"
    
    def validate_config(yaml_text: str) -> str:
        """Validate config without saving."""
        try:
            config_dict = yaml.safe_load(yaml_text)
            
            # Check required fields
            from src.config import Config
            missing = []
            for field in Config.REQUIRED_FIELDS:
                if field not in config_dict:
                    missing.append(field)
            
            if missing:
                return f"Missing required fields: {missing}"
            
            return "Configuration is valid"
            
        except Exception as e:
            return f"Invalid YAML: {str(e)}"
    
    # Connect handlers
    load_config_btn.click(
        fn=load_config_file,
        inputs=[config_file],
        outputs=[config_editor, config_status]
    )
    
    save_config_btn.click(
        fn=save_config,
        inputs=[config_editor],
        outputs=[config_status]
    )
    
    reload_config_btn.click(
        fn=reload_config,
        outputs=[config_editor, config_status]
    )
    
    validate_config_btn.click(
        fn=validate_config,
        inputs=[config_editor],
        outputs=[config_status]
    )