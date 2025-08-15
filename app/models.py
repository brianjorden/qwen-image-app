"""
Models tab UI for the Qwen-Image application.
"""

import gradio as gr
from pathlib import Path
from typing import Tuple

from src.config import get_config
from src.models import get_model_manager
from .shared import (
    create_model_status_display, handle_model_component_load, 
    handle_model_component_unload
)


def create_models_tab() -> None:
    """Create the model management tab."""
    config = get_config()
    model_manager = get_model_manager()
    
    with gr.Column():
        gr.Markdown("### Model Management")
        
        model_status = gr.Textbox(
            label="Model Status",
            lines=10,
            interactive=False,
            value=create_model_status_display()
        )
        
        with gr.Row():
            refresh_status_btn = gr.Button("🔄 Refresh Status")
        
        gr.Markdown("### Component Control")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Primary Components")
                load_tokenizer_btn = gr.Button("Load Tokenizer")
                load_processor_btn = gr.Button("Load Processor")
                load_text_encoder_btn = gr.Button("Load Text Encoder")
                load_transformer_btn = gr.Button("Load Transformer")
                load_vae_btn = gr.Button("Load VAE")
                load_scheduler_btn = gr.Button("Load Scheduler")
                build_pipeline_btn = gr.Button("Build Pipeline", variant="primary")
            
            with gr.Column():
                gr.Markdown("#### Unload Components")
                unload_tokenizer_btn = gr.Button("Unload Tokenizer")
                unload_processor_btn = gr.Button("Unload Processor")
                unload_text_encoder_btn = gr.Button("Unload Text Encoder")
                unload_transformer_btn = gr.Button("Unload Transformer")
                unload_vae_btn = gr.Button("Unload VAE")
                unload_all_btn = gr.Button("Unload All", variant="stop")
        
        # Alternative Encoder Section (Experimental)
        with gr.Accordion("🧪 Alternative Text Encoder (Experimental)", open=False):
            gr.Markdown("""
            **⚠️ Experimental Feature**: Load a second text encoder for comparison analysis.
            This is primarily for testing and analysis purposes, not regular generation workflow.
            """)
            
            with gr.Row():
                alt_encoder_path = gr.Textbox(
                    label="Alternative Encoder Path",
                    placeholder="Path to alternative text encoder model (optional - uses config default if empty)",
                    scale=3
                )
                alt_encoder_gpu = gr.Textbox(
                    label="GPU Allocation",
                    placeholder="e.g., cuda:1 or 2:8GiB (optional - auto-allocates if empty)",
                    scale=1
                )
            
            with gr.Row():
                load_text_encoder_alt_btn = gr.Button("Load Alternative Text Encoder", variant="secondary")
                unload_text_encoder_alt_btn = gr.Button("Unload Alternative Text Encoder")
        
        # LoRA section if enabled
        lora_components = None
        if config.enable_lora:
            gr.Markdown("### LoRA Management")
            with gr.Row():
                lora_path_input = gr.Textbox(
                    label="LoRA Path",
                    placeholder="Path to LoRA model"
                )
                lora_strength_input = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Strength"
                )
                load_lora_btn = gr.Button("Load LoRA")
            
            loaded_loras = gr.Textbox(
                label="Loaded LoRAs",
                lines=3,
                interactive=False
            )
            
            lora_components = (lora_path_input, lora_strength_input, load_lora_btn, loaded_loras)
    
    # Setup event handlers
    _setup_models_handlers(
        model_status, refresh_status_btn,
        load_tokenizer_btn, load_processor_btn, load_text_encoder_btn, load_text_encoder_alt_btn,
        load_transformer_btn, load_vae_btn, load_scheduler_btn, build_pipeline_btn,
        unload_tokenizer_btn, unload_processor_btn, unload_text_encoder_btn, unload_text_encoder_alt_btn,
        unload_transformer_btn, unload_vae_btn, unload_all_btn,
        alt_encoder_path, alt_encoder_gpu, lora_components
    )


def _setup_models_handlers(
    model_status, refresh_status_btn,
    load_tokenizer_btn, load_processor_btn, load_text_encoder_btn, load_text_encoder_alt_btn,
    load_transformer_btn, load_vae_btn, load_scheduler_btn, build_pipeline_btn,
    unload_tokenizer_btn, unload_processor_btn, unload_text_encoder_btn, unload_text_encoder_alt_btn,
    unload_transformer_btn, unload_vae_btn, unload_all_btn,
    alt_encoder_path, alt_encoder_gpu, lora_components
):
    """Setup all event handlers for the models tab."""
    
    def refresh_status():
        """Refresh model status display."""
        return create_model_status_display()
    
    def load_alt_encoder_with_custom_settings(alt_path: str, alt_gpu: str) -> Tuple[str, str]:
        """Load alternative text encoder with custom path and GPU settings."""
        try:
            from src.models import get_model_manager
            model_manager = get_model_manager()
            
            # Use custom path if provided, otherwise use config default
            custom_config = {}
            if alt_path.strip():
                custom_config['model_path'] = alt_path.strip()
            if alt_gpu.strip():
                custom_config['gpu_allocation'] = alt_gpu.strip()
            
            # Load the alternative encoder
            success = model_manager.load_component("text_encoder_alt", custom_config)
            
            if success:
                status_msg = f"Alternative text encoder loaded successfully"
                if alt_path.strip():
                    status_msg += f" from {alt_path}"
                if alt_gpu.strip():
                    status_msg += f" on {alt_gpu}"
                return status_msg, create_model_status_display()
            else:
                return "Failed to load alternative text encoder", create_model_status_display()
        except Exception as e:
            return f"Error loading alternative encoder: {str(e)}", create_model_status_display()
    
    def load_lora(path: str, strength: float) -> Tuple[str, str]:
        """Load a LoRA model."""
        model_manager = get_model_manager()
        
        if not path:
            return "No path provided", ""
        
        success = model_manager.load_lora(path, strength)
        
        if success:
            loras_text = "\n".join([
                f"{l['name']}: {l['strength']}"
                for l in model_manager.loras
            ])
            return f"LoRA loaded: {Path(path).name}", loras_text
        else:
            return "Failed to load LoRA", ""
    
    # Connect status refresh
    refresh_status_btn.click(
        fn=refresh_status,
        outputs=[model_status]
    )
    
    # Connect load handlers
    load_tokenizer_btn.click(
        fn=lambda: handle_model_component_load("tokenizer"),
        outputs=[gr.Textbox(visible=False), model_status]
    )
    
    load_processor_btn.click(
        fn=lambda: handle_model_component_load("processor"),
        outputs=[gr.Textbox(visible=False), model_status]
    )
    
    load_text_encoder_btn.click(
        fn=lambda: handle_model_component_load("text_encoder"),
        outputs=[gr.Textbox(visible=False), model_status]
    )
    
    load_text_encoder_alt_btn.click(
        fn=load_alt_encoder_with_custom_settings,
        inputs=[alt_encoder_path, alt_encoder_gpu],
        outputs=[gr.Textbox(visible=False), model_status]
    )
    
    load_transformer_btn.click(
        fn=lambda: handle_model_component_load("transformer"),
        outputs=[gr.Textbox(visible=False), model_status]
    )
    
    load_vae_btn.click(
        fn=lambda: handle_model_component_load("vae"),
        outputs=[gr.Textbox(visible=False), model_status]
    )
    
    load_scheduler_btn.click(
        fn=lambda: handle_model_component_load("scheduler"),
        outputs=[gr.Textbox(visible=False), model_status]
    )
    
    build_pipeline_btn.click(
        fn=lambda: handle_model_component_load("pipeline"),
        outputs=[gr.Textbox(visible=False), model_status]
    )
    
    # Connect unload handlers
    unload_tokenizer_btn.click(
        fn=lambda: handle_model_component_unload("tokenizer"),
        outputs=[gr.Textbox(visible=False), model_status]
    )
    
    unload_processor_btn.click(
        fn=lambda: handle_model_component_unload("processor"),
        outputs=[gr.Textbox(visible=False), model_status]
    )
    
    unload_text_encoder_btn.click(
        fn=lambda: handle_model_component_unload("text_encoder"),
        outputs=[gr.Textbox(visible=False), model_status]
    )
    
    unload_text_encoder_alt_btn.click(
        fn=lambda: handle_model_component_unload("text_encoder_alt"),
        outputs=[gr.Textbox(visible=False), model_status]
    )
    
    unload_transformer_btn.click(
        fn=lambda: handle_model_component_unload("transformer"),
        outputs=[gr.Textbox(visible=False), model_status]
    )
    
    unload_vae_btn.click(
        fn=lambda: handle_model_component_unload("vae"),
        outputs=[gr.Textbox(visible=False), model_status]
    )
    
    unload_all_btn.click(
        fn=lambda: handle_model_component_unload("all"),
        outputs=[gr.Textbox(visible=False), model_status]
    )
    
    # Connect LoRA handlers if enabled
    if lora_components:
        lora_path_input, lora_strength_input, load_lora_btn, loaded_loras = lora_components
        load_lora_btn.click(
            fn=load_lora,
            inputs=[lora_path_input, lora_strength_input],
            outputs=[gr.Textbox(visible=False), loaded_loras]
        )