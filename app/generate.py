"""
Generation tab UI for the Qwen-Image application.
"""

import gradio as gr
import random
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.config import get_config
from src.models import get_model_manager
from src.process import generate_image
from src.metadata import extract_metadata, format_metadata_display
from src.prompt import enhance_prompt_local
from .shared import (
    update_token_count, update_resolution, check_cfg_interaction,
    create_session_controls, setup_session_refresh
)


def create_generation_tab(session_state: gr.State) -> None:
    """Create the main generation tab."""
    config = get_config()
    
    with gr.Row():
        with gr.Column(scale=2):
            # Session management
            with gr.Row():
                session_dropdown, refresh_sessions_btn = create_session_controls()
            
            # Prompts
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=3
            )
            
            with gr.Row():
                prompt_tokens = gr.Markdown("Tokens: 0/1024", elem_classes=["token-counter"])
                enhance_prompt_btn = gr.Button("Enhance", scale=0)
            
            negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                placeholder="What to avoid in the image...",
                lines=2,
                value=config.default_negative
            )
            
            with gr.Row():
                negative_tokens = gr.Markdown("Tokens: 1/1024", elem_classes=["token-counter"])
                enhance_negative_btn = gr.Button("Enhance", scale=0)
            
            # Image input for noise interpolation
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image (Optional - Upload for noise interpolation)",
                    type="pil",
                    sources=["upload", "clipboard"]
                )
                with gr.Column():
                    noise_interpolation_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=config.default_noise_interpolation_strength,
                        step=0.05,
                        label="Transformation Strength (0.0 = minimal, 1.0 = maximum weirdness)",
                        visible=False  # Initially hidden, shown when image uploaded
                    )
                    
                    clear_input_btn = gr.Button("Clear Input Image", visible=False)
                    use_as_input_btn = gr.Button("Use Generated as Input", visible=False)
            
            # Generation settings
            with gr.Row():
                with gr.Column():
                    # Resolution presets
                    aspect_ratio = gr.Radio(
                        label="Aspect Ratio",
                        choices=["1:1", "16:9", "9:16", "4:3", "3:4", "Custom"],
                        value="1:1"
                    )
                    
                    with gr.Row():
                        width_input = gr.Slider(
                            minimum=256,
                            maximum=2048,
                            value=config.default_width,
                            step=16,
                            label="Width"
                        )
                        height_input = gr.Slider(
                            minimum=256,
                            maximum=2048,
                            value=config.default_height,
                            step=16,
                            label="Height"
                        )
                
                with gr.Column():
                    steps_input = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=config.default_steps,
                        step=1,
                        label="Steps"
                    )
                    
                    cfg_scale_input = gr.Slider(
                        minimum=0.0,
                        maximum=20.0,
                        value=config.default_cfg,
                        step=0.1,
                        label="CFG Scale"
                    )
                    
                    seed_input = gr.Number(
                        label="Seed",
                        value=config.default_seed,
                        precision=0
                    )
                    
                    randomize_seed = gr.Checkbox(
                        label="Random Seed",
                        value=False
                    )
                    
                    continuous_generation = gr.Checkbox(
                        label="Continuous Generation",
                        value=False,
                    )
            
            # Advanced options
            with gr.Accordion("Advanced Options", open=False):
                name_input = gr.Textbox(
                    label="Image Name",
                    placeholder="Optional custom name"
                )
                
                apply_template = gr.Checkbox(
                    label="Apply Training Template",
                    value=True
                )
                
                add_magic = gr.Checkbox(
                    label="Add Quality Enhancement",
                    value=True
                )
                
                save_steps = gr.Checkbox(
                    label="Save Per-Step Images (intermediate steps)",
                    value=config.enable_per_step_saving
                )
                
                second_stage_steps = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Second Stage Steps (0 = disabled)"
                )
                
                two_stage_mode = gr.Dropdown(
                    choices=["Noise Interpolation Mode"],
                    value="Noise Interpolation Mode",
                    label="Two-Stage Mode (noise interpolation)",
                    visible=False  # Hidden since only one option
                )
                
                # LoRA selection (if enabled)
                lora_inputs = []
                if config.enable_lora:
                    lora_container = gr.Column()
                    with lora_container:
                        gr.Markdown("### LoRA Models")
                        for i in range(config.lora_max_count):
                            with gr.Row():
                                lora_path = gr.Textbox(
                                    label=f"LoRA {i+1} Path",
                                    placeholder="Path to LoRA model"
                                )
                                lora_strength = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1,
                                    label="Strength"
                                )
                                lora_inputs.append((lora_path, lora_strength))
            
            # Action buttons
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop", visible=False)
        
        with gr.Column(scale=3):
            # Output
            output_image = gr.Image(label="Generated Image", type="pil")
            
            with gr.Row():
                metadata_display = gr.Textbox(
                    label="Image Metadata",
                    lines=8,
                    interactive=False,
                    elem_classes=["metadata-display"]
                )
            
            with gr.Row():
                apply_settings_btn = gr.Button("Apply Settings from Metadata")
                save_metadata_btn = gr.Button("Save Metadata")
    
    # Prompt Enhancement Popup Panel
    with gr.Column(visible=False, elem_classes=["enhancement-popup"]) as enhance_panel:
        with gr.Row():
            gr.Markdown("## 🎨 Enhanced Prompt Preview")
            gr.Button("✕", elem_id="close-enhancement", scale=0)
        enhanced_prompt_display = gr.Textbox(
            label="Enhanced Prompt (review and edit before applying)",
            lines=5,
            interactive=True
        )
        with gr.Row():
            apply_enhancement_btn = gr.Button("✅ Apply Enhancement", variant="primary")
            cancel_enhancement_btn = gr.Button("❌ Cancel", variant="secondary")
    
    # Variables to track enhancement state  
    current_prompt_input = gr.State()
    is_negative_enhancement = gr.State(False)
    current_metadata = gr.State({})
    
    # Setup event handlers
    _setup_generation_handlers(
        session_dropdown, refresh_sessions_btn, session_state,
        prompt_input, prompt_tokens, enhance_prompt_btn,
        negative_prompt_input, negative_tokens, enhance_negative_btn,
        input_image, noise_interpolation_strength, clear_input_btn, use_as_input_btn,
        aspect_ratio, width_input, height_input,
        steps_input, cfg_scale_input, seed_input, randomize_seed, continuous_generation,
        name_input, apply_template, add_magic, save_steps, second_stage_steps,
        two_stage_mode,
        lora_inputs, generate_btn, stop_btn,
        output_image, metadata_display, apply_settings_btn, save_metadata_btn,
        enhance_panel, enhanced_prompt_display, current_prompt_input,
        is_negative_enhancement, apply_enhancement_btn, cancel_enhancement_btn,
        current_metadata
    )


def _setup_generation_handlers(
    session_dropdown, refresh_sessions_btn, session_state,
    prompt_input, prompt_tokens, enhance_prompt_btn,
    negative_prompt_input, negative_tokens, enhance_negative_btn,
    input_image, noise_interpolation_strength, clear_input_btn, use_as_input_btn,
    aspect_ratio, width_input, height_input,
    steps_input, cfg_scale_input, seed_input, randomize_seed, continuous_generation,
    name_input, apply_template, add_magic, save_steps, second_stage_steps,
    two_stage_mode,
    lora_inputs, generate_btn, stop_btn,
    output_image, metadata_display, apply_settings_btn, save_metadata_btn,
    enhance_panel, enhanced_prompt_display, current_prompt_input,
    is_negative_enhancement, apply_enhancement_btn, cancel_enhancement_btn,
    current_metadata
):
    """Setup all event handlers for the generation tab."""
    
    # Setup session refresh functionality
    setup_session_refresh(session_dropdown, refresh_sessions_btn)
    
    # Aspect ratio changes
    aspect_ratio.change(
        fn=update_resolution,
        inputs=[aspect_ratio],
        outputs=[width_input, height_input]
    )
    
    # Token counting
    prompt_input.change(
        fn=update_token_count,
        inputs=[prompt_input],
        outputs=[prompt_tokens]
    )
    
    negative_prompt_input.change(
        fn=lambda x: update_token_count(x, True),
        inputs=[negative_prompt_input],
        outputs=[negative_tokens]
    )
    
    # CFG interaction checking
    cfg_scale_input.change(
        fn=check_cfg_interaction,
        inputs=[cfg_scale_input, negative_prompt_input],
        outputs=[cfg_scale_input, negative_prompt_input]
    )
    
    # Prompt enhancement
    enhance_prompt_btn.click(
        fn=lambda x: _enhance_prompt(x, False),
        inputs=[prompt_input],
        outputs=[enhance_panel, enhanced_prompt_display, current_prompt_input, is_negative_enhancement]
    )
    
    enhance_negative_btn.click(
        fn=lambda x: _enhance_prompt(x, True),
        inputs=[negative_prompt_input],
        outputs=[enhance_panel, enhanced_prompt_display, current_prompt_input, is_negative_enhancement]
    )
    
    apply_enhancement_btn.click(
        fn=_apply_enhancement,
        inputs=[enhanced_prompt_display, current_prompt_input, is_negative_enhancement],
        outputs=[enhance_panel, prompt_input, negative_prompt_input]
    )
    
    cancel_enhancement_btn.click(
        fn=_cancel_enhancement,
        outputs=[enhance_panel]
    )
    
    # Image input handlers
    input_image.change(
        fn=_handle_image_upload,
        inputs=[input_image],
        outputs=[noise_interpolation_strength, clear_input_btn, use_as_input_btn]
    )
    
    clear_input_btn.click(
        fn=lambda: (None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
        outputs=[input_image, noise_interpolation_strength, clear_input_btn, use_as_input_btn]
    )
    
    use_as_input_btn.click(
        fn=_use_generated_as_input,
        inputs=[output_image],
        outputs=[input_image, noise_interpolation_strength, clear_input_btn, use_as_input_btn]
    )
    
    # Session state updates
    session_dropdown.change(
        fn=lambda x: x,
        inputs=[session_dropdown],
        outputs=[session_state]
    )
    
    # Generate image
    config = get_config()
    gen_inputs = [
        session_dropdown, prompt_input, negative_prompt_input, name_input,
        width_input, height_input, steps_input, cfg_scale_input,
        seed_input, randomize_seed, apply_template, add_magic, save_steps, second_stage_steps,
        two_stage_mode, input_image, noise_interpolation_strength
    ]
    
    if config.enable_lora:
        for lora_path, lora_strength in lora_inputs:
            gen_inputs.extend([lora_path, lora_strength])
    
    def generate_wrapper(*args):
        return _generate_image_handler(*args, current_metadata_state=current_metadata)
    
    generate_btn.click(
        fn=generate_wrapper,
        inputs=gen_inputs,
        outputs=[output_image, metadata_display, seed_input]
    )
    
    # Show "Use as Input" button when image is generated
    output_image.change(
        fn=lambda img: gr.update(visible=(img is not None)),
        inputs=[output_image],
        outputs=[use_as_input_btn]
    )
    
    # Apply metadata settings
    apply_settings_btn.click(
        fn=lambda: _apply_metadata_settings(current_metadata.value),
        outputs=[
            prompt_input, negative_prompt_input,
            width_input, height_input, steps_input,
            cfg_scale_input, seed_input, aspect_ratio
        ]
    )


def _enhance_prompt(prompt: str, is_negative: bool = False) -> Tuple:
    """Route prompt enhancement to chat tab."""
    print(f"Enhancement requested for: '{prompt}' (negative: {is_negative})")
    
    if not prompt or not prompt.strip():
        gr.Warning("Please enter a prompt to enhance")
        return
    
    # Create enhancement message for chat
    enhancement_msg = f"Please enhance this {'negative ' if is_negative else ''}prompt for image generation: {prompt}"
    
    # This would switch to chat tab and start a new thread with enhancement template
    gr.Info(f"Starting prompt enhancement in chat tab...")
    
    # For now, just show a message - actual routing would require app-level state management
    return


def _apply_enhancement(enhanced_prompt: str, current_input: str, is_negative: bool) -> Tuple:
    """Apply the enhanced prompt to the appropriate field."""
    return (
        gr.update(visible=False),  # Hide modal
        enhanced_prompt if not is_negative else gr.update(),  # Update prompt if not negative
        enhanced_prompt if is_negative else gr.update(),  # Update negative if negative
    )


def _cancel_enhancement() -> gr.update:
    """Cancel enhancement and hide modal."""
    return gr.update(visible=False)


def _generate_image_handler(*args, current_metadata_state) -> Tuple:
    """Handle image generation."""
    config = get_config()
    
    # Unpack arguments
    session, prompt, negative_prompt, name = args[:4]
    width, height, steps, cfg_scale = args[4:8]
    seed, randomize, apply_template, add_magic, save_steps, second_stage_steps, two_stage_mode, input_image, noise_interpolation_strength = args[8:17]
    
    lora_args = args[17:] if config.enable_lora else []
    
    try:
        # Handle random seed
        if randomize:
            seed = random.randint(0, 2**32 - 1)
        
        # Collect LoRA configs if enabled
        loras = []
        if config.enable_lora and lora_args:
            for i in range(0, len(lora_args), 2):
                lora_path = lora_args[i]
                lora_strength = lora_args[i + 1] if i + 1 < len(lora_args) else 1.0
                if lora_path:
                    loras.append({
                        'path': lora_path,
                        'strength': lora_strength
                    })
        
        # Generate image
        image, save_path = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            session=session,
            name=name,
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=cfg_scale,
            seed=seed,
            loras=loras,
            apply_template=apply_template,
            add_magic=add_magic,
            save_steps=save_steps,
            second_stage_steps=second_stage_steps,
            two_stage_mode=two_stage_mode,
            input_image=input_image,
            noise_interpolation_strength=noise_interpolation_strength
        )
        
        if image:
            # Extract and format metadata
            metadata = extract_metadata(save_path) if save_path else {}
            metadata_text = format_metadata_display(metadata)
            current_metadata_state.value = metadata
            
            return image, metadata_text, seed
        else:
            return None, "Generation failed", seed
            
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg, seed


def _apply_metadata_settings(metadata: Dict[str, Any]) -> List:
    """Apply settings from current metadata to controls."""
    config = get_config()
    
    if not metadata:
        return [gr.update()] * 8
    
    updates = []
    updates.append(gr.update(value=metadata.get('prompt', '')))
    updates.append(gr.update(value=metadata.get('negative_prompt', '')))
    updates.append(gr.update(value=metadata.get('width', config.default_width)))
    updates.append(gr.update(value=metadata.get('height', config.default_height)))
    updates.append(gr.update(value=metadata.get('steps', config.default_steps)))
    updates.append(gr.update(value=metadata.get('cfg_scale', config.default_cfg)))
    updates.append(gr.update(value=metadata.get('seed', config.default_seed)))
    updates.append("Custom")  # Set aspect ratio to custom
    
    return updates


def _handle_image_upload(image):
    """Handle image upload and show/hide relevant controls."""
    if image is not None:
        return (
            gr.update(visible=True),   # noise_interpolation_strength slider
            gr.update(visible=True),   # clear_input_btn
            gr.update(visible=False)   # use_as_input_btn (hidden when image uploaded)
        )
    else:
        return (
            gr.update(visible=False),  # noise_interpolation_strength slider
            gr.update(visible=False),  # clear_input_btn
            gr.update(visible=False)   # use_as_input_btn
        )


def _use_generated_as_input(generated_image):
    """Use the generated image as input for next generation."""
    if generated_image is not None:
        return (
            generated_image,           # Set as input_image
            gr.update(visible=True),   # Show noise_interpolation_strength slider
            gr.update(visible=True),   # Show clear_input_btn
            gr.update(visible=False)   # Hide use_as_input_btn
        )
    else:
        return (
            None,                      # No input_image
            gr.update(visible=False),  # Hide noise_interpolation_strength slider
            gr.update(visible=False),  # Hide clear_input_btn
            gr.update(visible=False)   # Hide use_as_input_btn
        )