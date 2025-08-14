"""
Shared UI utilities and common components for the Qwen-Image app.
"""

import gradio as gr
from typing import Dict, Any, List, Tuple, Optional
from src.config import get_config
from src.prompt import count_tokens
from src.models import get_model_manager


# Common CSS for the application
CUSTOM_CSS = """
.inactive-control { opacity: 0.5; }
.token-counter { font-size: 0.9em; color: #666; }
.metadata-display { font-family: monospace; font-size: 0.9em; }
"""


def update_token_count(text: str, is_negative: bool = False) -> str:
    """Update token counter display.
    
    Args:
        text: Text to count tokens for
        is_negative: Whether this is a negative prompt (affects default handling)
        
    Returns:
        Formatted token count string
    """
    config = get_config()
    model_manager = get_model_manager()
    
    if not text:
        text = " " if is_negative and config.default_negative == " " else ""
    
    count, truncated = count_tokens(text, model_manager.tokenizer)
    label = f"Tokens: {count}/1024"
    if truncated:
        label += " ⚠️ Truncated"
    return label


def update_resolution(aspect_ratio: str) -> Tuple[Optional[int], Optional[int]]:
    """Update resolution based on aspect ratio selection.
    
    Args:
        aspect_ratio: Selected aspect ratio string
        
    Returns:
        Tuple of (width, height) or (None, None) for custom
    """
    config = get_config()
    if aspect_ratio != "Custom":
        w, h = config.resolutions[aspect_ratio]
        return w, h
    return None, None


def check_cfg_interaction(cfg_value: float, neg_prompt: str) -> Tuple[gr.update, gr.update]:
    """Update UI to show when CFG/negative prompt won't be used.
    
    Args:
        cfg_value: CFG scale value
        neg_prompt: Negative prompt text
        
    Returns:
        Tuple of CSS class updates for CFG and negative prompt controls
    """
    cfg_inactive = cfg_value <= 1.0
    neg_inactive = not neg_prompt.strip()
    
    # Return CSS classes or visual indicators
    cfg_class = "inactive-control" if neg_inactive else ""
    neg_class = "inactive-control" if cfg_inactive else ""
    
    return gr.update(elem_classes=[cfg_class]), gr.update(elem_classes=[neg_class])


def create_session_controls() -> Tuple[gr.Dropdown, gr.Button]:
    """Create standard session management controls.
    
    Returns:
        Tuple of (session_dropdown, refresh_button)
    """
    from src.gallery import get_session_manager
    
    session_manager = get_session_manager()
    
    session_dropdown = gr.Dropdown(
        label="Session",
        choices=session_manager.get_sessions(),
        value=session_manager.get_default_session(),
        allow_custom_value=True,
        interactive=True
    )
    refresh_btn = gr.Button("🔄", scale=0)
    
    return session_dropdown, refresh_btn


def setup_session_refresh(session_dropdown: gr.Dropdown, refresh_btn: gr.Button):
    """Setup session refresh functionality.
    
    Args:
        session_dropdown: Session dropdown component
        refresh_btn: Refresh button component
    """
    from src.gallery import get_session_manager
    
    def refresh_sessions():
        session_manager = get_session_manager()
        return gr.update(choices=session_manager.get_sessions())
    
    refresh_btn.click(
        fn=refresh_sessions,
        outputs=[session_dropdown]
    )


def create_model_status_display() -> str:
    """Get formatted model status.
    
    Returns:
        Formatted model status string
    """
    model_manager = get_model_manager()
    status = model_manager.get_status()
    
    lines = ["Component Status:"]
    for component, loaded in status['components'].items():
        emoji = "✅" if loaded else "❌"
        lines.append(f"  {emoji} {component}: {'Loaded' if loaded else 'Not loaded'}")
    
    if status['model_info']:
        lines.append("\nLoaded Models:")
        for component, info in status['model_info'].items():
            lines.append(f"  {component}: {info}")
    
    if status['loras']:
        lines.append("\nLoaded LoRAs:")
        for lora in status['loras']:
            lines.append(f"  {lora['name']}: strength={lora['strength']}")
    
    if status['memory_allocated']:
        lines.append("\nGPU Memory (GB):")
        for device, mem in status['memory_allocated'].items():
            lines.append(f"  {device}: {mem:.2f} GB")
    
    return "\n".join(lines)


def get_config_yaml() -> str:
    """Get current config as YAML.
    
    Returns:
        YAML representation of current config
    """
    import yaml
    config = get_config()
    return yaml.dump(config.to_dict(), default_flow_style=False, indent=2)


def handle_model_component_load(component_name: str, alt_path: Optional[str] = None) -> Tuple[str, str]:
    """Load a model component.
    
    Args:
        component_name: Name of component to load
        alt_path: Alternative path for alt encoder
        
    Returns:
        Tuple of (status_message, model_status)
    """
    model_manager = get_model_manager()
    config = get_config()
    
    try:
        if component_name == "tokenizer":
            model_manager.load_tokenizer(force_reload=True)
        elif component_name == "processor":
            model_manager.load_processor(force_reload=True)
        elif component_name == "text_encoder":
            model_manager.load_text_encoder(force_reload=True)
        elif component_name == "text_encoder_alt":
            if alt_path:
                config._config['model_text_encoder_alt'] = alt_path
            model_manager.load_text_encoder(use_alt=True, force_reload=True)
        elif component_name == "transformer":
            model_manager.load_transformer(force_reload=True)
        elif component_name == "vae":
            model_manager.load_vae(force_reload=True)
        elif component_name == "scheduler":
            model_manager.load_scheduler(force_reload=True)
        elif component_name == "pipeline":
            model_manager.build_pipeline(force_rebuild=True)
        
        return f"{component_name} loaded", create_model_status_display()
        
    except Exception as e:
        return f"Failed to load {component_name}: {str(e)}", create_model_status_display()


def handle_model_component_unload(component_name: str) -> Tuple[str, str]:
    """Unload a model component.
    
    Args:
        component_name: Name of component to unload
        
    Returns:
        Tuple of (status_message, model_status)
    """
    model_manager = get_model_manager()
    
    try:
        if component_name == "all":
            model_manager.unload_all()
        else:
            model_manager.unload_component(component_name)
        
        return f"{component_name} unloaded", create_model_status_display()
        
    except Exception as e:
        return f"Failed to unload {component_name}: {str(e)}", create_model_status_display()