"""
Generation tab UI for the qwen-image-app.
"""

import gradio as gr
import gradio_modal
import random
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.config import get_config
from src.models import get_model_manager
from src.process import generate_image
from src.metadata import extract_metadata, extract_metadata_from_pil_image, format_metadata_display
from src.prompt import enhance_prompt_local
from .shared import (
    update_token_count, update_resolution, check_cfg_interaction,
    create_session_controls, setup_session_refresh
)


def create_generation_tab(session_state: gr.State, shared_image_state: gr.State, shared_prompt_state: gr.State, shared_metadata_state: gr.State, tab_communication_state: gr.State) -> None:
    """Create the main generation tab."""
    config = get_config()
    
    # Generate buttons at top for easy access
    with gr.Row():
        generate_btn = gr.Button("Generate", variant="primary", scale=3)
        stop_btn = gr.Button("Stop", variant="stop", visible=True, scale=1)
    
    with gr.Row():
        with gr.Column(scale=2):
            
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
                    label="Input Image (Optional - Upload for img2img generation)",
                    type="pil",
                    sources=["upload", "clipboard"]
                )
                with gr.Column():
                    # Generation mode selector
                    img2img_mode = gr.Radio(
                        choices=["true_img2img", "noise_interpolation", "inpaint"],
                        value="true_img2img",
                        label="Generation Mode",
                        info="True Img2img: Proper diffusion | Noise Interpolation: Creative mixing | Inpaint: Fill masked areas",
                        visible=True,
                        interactive=False  # Initially grayed out, enabled when image uploaded
                    )
                    
                    # Mode-specific strength controls  
                    noise_interpolation_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=config.default_noise_interpolation_strength,
                        step=0.05,
                        label="Noise Interpolation Strength (0.0 = minimal, 1.0 = maximum weirdness)",
                        visible=True,
                        interactive=False  # Initially grayed out, enabled when mode is selected
                    )
                    
                    img2img_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=config.default_img2img_strength,
                        step=0.05,
                        label="True Img2img Strength (0.0 = minimal change, 1.0 = maximum change)",
                        visible=True,
                        interactive=False  # Initially grayed out, enabled when mode is selected
                    )
                    
                    inpaint_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.6,
                        step=0.05,
                        label="Inpainting Strength (0.0 = minimal change, 1.0 = maximum change)",
                        visible=False,
                        interactive=False  # Initially hidden and grayed out
                    )
                    
            # Mask creation for inpainting (placed after input image row)
            with gr.Row(visible=False) as mask_row:
                with gr.Column():
                    mask_method = gr.Radio(
                        choices=["Draw Mask", "Upload Mask"],
                        value="Draw Mask",
                        label="Mask Method",
                        info="Draw directly on image or upload a pre-made mask"
                    )
                    
                    # ImageEditor for drawing masks directly on the image
                    mask_editor = gr.ImageEditor(
                        label="Draw Mask (Paint white areas to inpaint)",
                        type="pil",
                        brush=gr.Brush(
                            colors=["#FFFFFF", "#000000"],  # White for inpaint areas, black for eraser
                            default_color="#FFFFFF",
                            default_size=20
                        ),
                        layers=True,
                        transforms=["crop"],  # Only allow crop to keep focus on mask drawing
                        visible=True,
                        interactive=True
                    )
                    
                    # Traditional mask upload (hidden by default)
                    mask_upload = gr.Image(
                        label="Upload Mask (White areas will be inpainted)",
                        type="pil",
                        sources=["upload", "clipboard"],
                        visible=False,
                        interactive=True
                    )
                    
                with gr.Column():
                    with gr.Row():
                        clear_mask_btn = gr.Button("Clear Mask", size="sm")
                        auto_mask_btn = gr.Button("Auto Mask Center", size="sm")
                    
                    # Brush size control for drawing
                    brush_size = gr.Slider(
                        minimum=5,
                        maximum=100,
                        value=20,
                        step=5,
                        label="Brush Size",
                        visible=True
                    )
                    
                    # Instructions
                    gr.Markdown(
                        """
                        **Drawing Instructions:**
                        - White: Areas to inpaint
                        - Black: Areas to preserve/erase mask
                        - Use layers for complex masks
                        """,
                        visible=True
                    )
                    
            with gr.Row():                    
                clear_input_btn = gr.Button("Clear Input Image", visible=False)
                use_as_input_btn = gr.Button("Use Generated as Input", visible=True)
            
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
            
            # LoRA selection (if enabled)
            lora_inputs = []
            if config.enable_lora:
                with gr.Accordion("LoRA Models", open=False):
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
            
            # Advanced options
            with gr.Accordion("Advanced Options", open=True):
                name_input = gr.Textbox(
                    label="Image Name",
                    placeholder="Optional custom name"
                )
                
                
                add_magic = gr.Checkbox(
                    label="Add Positive Prompt Magic",
                    value=config.add_positive_prompt_magic
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
            
            # Session management at bottom of left column
            with gr.Row():
                session_dropdown, refresh_sessions_btn = create_session_controls()
        
        with gr.Column(scale=3):
            # Output
            output_image = gr.Image(
                label="Generated Image", 
                type="pil", 
                interactive=False,
                show_download_button=False,
                show_share_button=False
            )
            
            # Action buttons for generated image
            with gr.Row():
                copy_image_btn = gr.Button("Copy", scale=1)
                download_image_btn = gr.DownloadButton(
                    "Download PNG", 
                    scale=1
                )
                apply_metadata_btn = gr.Button("Apply Metadata", scale=1)
                clear_image_btn = gr.Button("Clear", scale=1)
                send_to_chat_btn = gr.Button("Send to Chat", scale=1)
            
            # Input image metadata display (collapsible, auto-collapsed)
            with gr.Accordion("Input Image Metadata", open=False, visible=False) as input_metadata_accordion:
                input_metadata_display = gr.Textbox(
                    label="Metadata from Input Image",
                    lines=6,
                    interactive=False,
                    elem_classes=["metadata-display"]
                )
                use_input_metadata_btn = gr.Button("Use Input Metadata")
            
            # Generated image metadata display
            with gr.Row():
                metadata_display = gr.Textbox(
                    label="Generated Image Metadata",
                    lines=8,
                    interactive=False,
                    elem_classes=["metadata-display"]
                )
            
            with gr.Row():
                save_metadata_btn = gr.Button("Save Metadata")
    
    
    # Input image fullscreen modal
    with gradio_modal.Modal(visible=False, allow_user_close=True) as input_image_modal:
        with gr.Column():
            gr.Markdown("### Input Image Viewer")
            input_modal_close_btn = gr.Button("✕ Close", scale=1)
            
            # Large image display for input image
            input_modal_image = gr.Image(
                label="",
                show_label=False,
                interactive=False,
                height=600,
                show_download_button=True,
                show_share_button=False
            )
            
            # Input image metadata in modal
            input_modal_metadata = gr.Textbox(
                label="Input Image Metadata",
                lines=4,
                interactive=False,
                show_label=True
            )
    
    # Variables to track enhancement state  
    current_prompt_input = gr.State()
    is_negative_enhancement = gr.State(False)
    current_metadata = gr.State({})
    input_image_metadata = gr.State({})
    
    # Setup event handlers
    _setup_generation_handlers(
        session_dropdown, refresh_sessions_btn, session_state,
        prompt_input, prompt_tokens, enhance_prompt_btn,
        negative_prompt_input, negative_tokens, enhance_negative_btn,
        input_image, mask_method, mask_editor, mask_upload, auto_mask_btn, brush_size, img2img_mode, noise_interpolation_strength, img2img_strength, inpaint_strength, mask_row, clear_input_btn, clear_mask_btn, use_as_input_btn,
        aspect_ratio, width_input, height_input,
        steps_input, cfg_scale_input, seed_input, randomize_seed, continuous_generation,
        name_input, add_magic, save_steps, second_stage_steps,
        two_stage_mode,
        lora_inputs, generate_btn, stop_btn,
        output_image, copy_image_btn, download_image_btn, apply_metadata_btn, clear_image_btn, send_to_chat_btn,
        input_metadata_accordion, input_metadata_display, use_input_metadata_btn,
        metadata_display, save_metadata_btn,
        current_metadata, input_image_metadata,
        # Input image modal components
        input_image_modal, input_modal_image, input_modal_metadata, input_modal_close_btn,
        # Cross-tab communication states
        shared_image_state, shared_prompt_state, shared_metadata_state, tab_communication_state
    )


def _setup_generation_handlers(
    session_dropdown, refresh_sessions_btn, session_state,
    prompt_input, prompt_tokens, enhance_prompt_btn,
    negative_prompt_input, negative_tokens, enhance_negative_btn,
    input_image, mask_method, mask_editor, mask_upload, auto_mask_btn, brush_size, img2img_mode, noise_interpolation_strength, img2img_strength, inpaint_strength, mask_row, clear_input_btn, clear_mask_btn, use_as_input_btn,
    aspect_ratio, width_input, height_input,
    steps_input, cfg_scale_input, seed_input, randomize_seed, continuous_generation,
    name_input, add_magic, save_steps, second_stage_steps,
    two_stage_mode,
    lora_inputs, generate_btn, stop_btn,
    output_image, copy_image_btn, download_image_btn, apply_metadata_btn, clear_image_btn, send_to_chat_btn,
    input_metadata_accordion, input_metadata_display, use_input_metadata_btn,
    metadata_display, save_metadata_btn,
    current_metadata, input_image_metadata,
    # Input image modal components
    input_image_modal, input_modal_image, input_modal_metadata, input_modal_close_btn,
    # Cross-tab communication states
    shared_image_state, shared_prompt_state, shared_metadata_state, tab_communication_state
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
    # Enhancement will be implemented as chat conversation starter
    enhance_prompt_btn.click(
        fn=lambda x: _enhance_prompt(x, False, shared_image_state, tab_communication_state),
        inputs=[prompt_input],
        outputs=[shared_image_state, tab_communication_state]
    )
    
    enhance_negative_btn.click(
        fn=lambda x: _enhance_prompt(x, True, shared_image_state, tab_communication_state),
        inputs=[negative_prompt_input],
        outputs=[shared_image_state, tab_communication_state]
    )
    
    # Image input handlers
    input_image.change(
        fn=_handle_image_upload,
        inputs=[input_image],
        outputs=[img2img_mode, noise_interpolation_strength, img2img_strength, clear_input_btn, use_as_input_btn, 
                input_metadata_accordion, input_metadata_display, input_image_metadata]
    )
    
    # Generation mode change handler
    img2img_mode.change(
        fn=_handle_img2img_mode_change,
        inputs=[img2img_mode],
        outputs=[noise_interpolation_strength, img2img_strength, inpaint_strength, mask_row]
    )
    
    clear_input_btn.click(
        fn=lambda: (None, None, None, gr.update(visible=True, interactive=False), gr.update(visible=True, interactive=False), gr.update(visible=True, interactive=False), gr.update(visible=False, interactive=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), "", {}),
        outputs=[input_image, mask_editor, mask_upload, img2img_mode, noise_interpolation_strength, img2img_strength, inpaint_strength, mask_row, clear_input_btn, use_as_input_btn, input_metadata_accordion, input_metadata_display, input_image_metadata]
    )
    
    # Mask method change handler
    mask_method.change(
        fn=_handle_mask_method_change,
        inputs=[mask_method],
        outputs=[mask_editor, mask_upload]
    )
    
    # Auto mask button handler
    auto_mask_btn.click(
        fn=_create_auto_mask,
        inputs=[input_image],
        outputs=[mask_upload]  # Put auto mask in upload component
    )
    
    # Clear mask handlers  
    clear_mask_btn.click(
        fn=lambda: (None, None),
        outputs=[mask_editor, mask_upload]
    )
    
    # Brush size change handler
    brush_size.change(
        fn=lambda size: gr.update(brush=gr.Brush(
            colors=["#FFFFFF", "#000000"],
            default_color="#FFFFFF", 
            default_size=size
        )),
        inputs=[brush_size],
        outputs=[mask_editor]
    )
    
    # Image editor change handler to populate with input image
    input_image.change(
        fn=lambda img: img if img else None,
        inputs=[input_image],
        outputs=[mask_editor]
    )
    
    use_as_input_btn.click(
        fn=_use_generated_as_input,
        inputs=[output_image],
        outputs=[input_image, img2img_mode, noise_interpolation_strength, img2img_strength, clear_input_btn, use_as_input_btn]
    )
    
    # Input image modal handlers
    input_image.select(
        fn=_open_input_image_modal,
        inputs=[input_image],
        outputs=[input_image_modal, input_modal_image, input_modal_metadata]
    )
    
    input_modal_close_btn.click(
        fn=_close_input_image_modal,
        outputs=[input_image_modal]
    )
    
    # New image action buttons
    copy_image_btn.click(
        fn=_copy_image_to_clipboard,
        inputs=[output_image]
    )
    
    # Set up DownloadButton to be reactive to output image changes
    output_image.change(
        fn=_prepare_download_file,
        inputs=[output_image, current_metadata],
        outputs=[download_image_btn]
    )
    
    clear_image_btn.click(
        fn=lambda: None,
        outputs=[output_image]
    )
    
    send_to_chat_btn.click(
        fn=_send_image_to_chat,
        inputs=[output_image, shared_image_state, tab_communication_state],
        outputs=[shared_image_state, tab_communication_state]
    )
    
    # Use input metadata button
    use_input_metadata_btn.click(
        fn=_apply_input_metadata,
        inputs=[input_image_metadata],
        outputs=[
            prompt_input, negative_prompt_input,
            width_input, height_input, steps_input,
            cfg_scale_input, seed_input, aspect_ratio
        ]
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
        seed_input, randomize_seed, add_magic, save_steps, second_stage_steps,
        two_stage_mode, input_image, mask_method, mask_editor, mask_upload, img2img_mode, noise_interpolation_strength, img2img_strength, inpaint_strength
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
    
    # Stop button functionality - uses step callback cancellation
    def stop_generation():
        try:
            from src.step import request_cancellation
            request_cancellation()
            gr.Info("Stop requested - generation will halt at next step")
        except Exception as e:
            gr.Warning(f"Could not request stop: {e}")
    
    # Stop button handler
    stop_btn.click(
        fn=stop_generation,
        show_progress=False
    )
    
    # Note: "Use as Input" button is now always visible
    
    # Apply metadata settings
    apply_metadata_btn.click(
        fn=lambda: _apply_metadata_settings(current_metadata.value),
        outputs=[
            prompt_input, negative_prompt_input,
            width_input, height_input, steps_input,
            cfg_scale_input, seed_input, aspect_ratio
        ]
    )
    
    # Save metadata to file
    save_metadata_btn.click(
        fn=lambda metadata: _save_metadata_to_file(metadata),
        inputs=[current_metadata],
        show_progress=False
    )


def _enhance_prompt(prompt: str, is_negative: bool = False, shared_state=None, tab_comm_state=None):
    """Enhance prompt by starting a chat conversation."""
    if not prompt or not prompt.strip():
        gr.Warning("Please enter a prompt to enhance")
        return shared_state.value if shared_state else None, tab_comm_state.value if tab_comm_state else None
    
    try:
        from src.chat import start_enhancement_chat
        new_history = start_enhancement_chat(prompt, is_negative)
        
        # Communicate to chat tab if states are available
        if shared_state and tab_comm_state:
            shared_state.value = new_history
            tab_comm_state.value = {
                "action": "start_new_chat",
                "source": "enhance",
                "chat_history": new_history,
                "enhancement_type": "negative" if is_negative else "positive",
                "original_prompt": prompt,
                "timestamp": __import__("time").time()
            }
            gr.Info(f"Enhancement conversation started for {'negative ' if is_negative else ''}prompt - switch to Chat tab to continue")
        else:
            gr.Info(f"Enhancement conversation started for {'negative ' if is_negative else ''}prompt. Switch to Chat tab to see the enhanced version.")
        
        return shared_state.value if shared_state else None, tab_comm_state.value if tab_comm_state else None
        
    except Exception as e:
        gr.Warning(f"Failed to start enhancement chat: {e}")
        return shared_state.value if shared_state else None, tab_comm_state.value if tab_comm_state else None




def _generate_image_handler(*args, current_metadata_state) -> Tuple:
    """Handle image generation."""
    config = get_config()
    
    # Debug: print argument count
    print(f"Total args received: {len(args)}")
    print(f"Args 8-22 count: {len(args[8:22])}")
    
    # Unpack arguments
    session, prompt, negative_prompt, name = args[:4]
    width, height, steps, cfg_scale = args[4:8]
    
    # More defensive unpacking for the variable part
    remaining_args = args[8:]
    if len(remaining_args) < 14:
        print(f"Warning: Expected 14 args from position 8, got {len(remaining_args)}")
        # Pad with None values if missing
        remaining_args = list(remaining_args) + [None] * (14 - len(remaining_args))
    
    seed, randomize, add_magic, save_steps, second_stage_steps, two_stage_mode, input_image, mask_method, mask_editor, mask_upload, img2img_mode, noise_interpolation_strength, img2img_strength, inpaint_strength = remaining_args[:14]
    
    # Extract the actual mask from the components
    mask_image = None
    if mask_method == "Draw Mask":
        mask_image = _extract_mask_from_editor(mask_editor)
    elif mask_method == "Upload Mask":
        mask_image = mask_upload
    
    lora_args = args[22:] if config.enable_lora else []
    
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
            add_magic=add_magic,
            save_steps=save_steps,
            second_stage_steps=second_stage_steps,
            two_stage_mode=two_stage_mode,
            input_image=input_image,
            mask_image=mask_image,
            img2img_mode=img2img_mode,
            noise_interpolation_strength=noise_interpolation_strength,
            img2img_strength=img2img_strength,
            inpaint_strength=inpaint_strength
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


def _extract_mask_from_editor(editor_data):
    """Extract mask from ImageEditor data structure."""
    if editor_data is None:
        return None
    
    try:
        # ImageEditor returns a dict with 'background', 'layers', 'composite'
        if isinstance(editor_data, dict):
            # If there are layers (drawn content), use the composite
            if 'layers' in editor_data and editor_data['layers']:
                # Use the composite image which combines all layers
                mask = editor_data.get('composite')
                if mask:
                    # Convert to grayscale mask
                    mask = mask.convert('L')
                    return mask
            
            # If no layers but has background, return None (no mask drawn)
            return None
        
        # If it's a direct PIL image (shouldn't happen with ImageEditor but fallback)
        elif hasattr(editor_data, 'convert'):
            return editor_data.convert('L')
            
    except Exception as e:
        print(f"Error extracting mask from editor: {e}")
        return None
    
    return None


def _create_auto_mask(image, mask_size_percent=0.3):
    """Create a rectangular mask in the center of the image."""
    if image is None:
        return None
    
    try:
        from PIL import Image, ImageDraw
        width, height = image.size
        
        # Calculate center rectangle
        mask_width = int(width * mask_size_percent)
        mask_height = int(height * mask_size_percent)
        
        x1 = (width - mask_width) // 2
        y1 = (height - mask_height) // 2
        x2 = x1 + mask_width
        y2 = y1 + mask_height
        
        # Create mask (black background, white rectangle)
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x1, y1, x2, y2], fill=255)
        
        return mask
        
    except Exception as e:
        print(f"Error creating auto mask: {e}")
        return None


def _handle_mask_method_change(mask_method):
    """Handle switching between draw and upload mask methods."""
    if mask_method == "Draw Mask":
        return (
            gr.update(visible=True),   # Show mask_editor
            gr.update(visible=False)   # Hide mask_upload
        )
    else:  # Upload Mask
        return (
            gr.update(visible=False),  # Hide mask_editor
            gr.update(visible=True)    # Show mask_upload
        )


def _handle_image_upload(image):
    """Handle image upload and show/hide relevant controls."""
    if image is not None:
        # Try to extract metadata from uploaded image
        metadata_text = ""
        metadata_visible = False
        raw_metadata = {}
        try:
            metadata = extract_metadata_from_pil_image(image)
            if metadata:
                metadata_text = format_metadata_display(metadata)
                metadata_visible = True
                raw_metadata = metadata
        except Exception as e:
            print(f"Could not extract metadata from uploaded image: {e}")
        
        return (
            gr.update(visible=True, interactive=True),   # img2img_mode selector - enabled
            gr.update(visible=True, interactive=False),  # noise_interpolation_strength slider - grayed out (default mode is true_img2img)
            gr.update(visible=True, interactive=True),   # img2img_strength slider - active (default mode)
            gr.update(visible=True),   # clear_input_btn
            gr.update(visible=True),   # use_as_input_btn (always visible)
            gr.update(visible=metadata_visible),  # input_metadata_accordion
            metadata_text,             # input_metadata_display
            raw_metadata               # input_image_metadata state
        )
    else:
        return (
            gr.update(visible=True, interactive=False),  # img2img_mode selector - grayed out
            gr.update(visible=True, interactive=False),  # noise_interpolation_strength slider - grayed out
            gr.update(visible=True, interactive=False),  # img2img_strength slider - grayed out
            gr.update(visible=False),  # clear_input_btn
            gr.update(visible=True),   # use_as_input_btn (always visible)
            gr.update(visible=False),  # input_metadata_accordion
            "",                        # input_metadata_display
            {}                         # input_image_metadata state (empty)
        )


def _use_generated_as_input(generated_image):
    """Use the generated image as input for next generation."""
    if generated_image is not None:
        return (
            generated_image,           # Set as input_image
            gr.update(visible=True, interactive=True),   # Enable img2img_mode selector
            gr.update(visible=True, interactive=False),  # Disable noise_interpolation_strength slider initially
            gr.update(visible=True, interactive=True),   # Enable img2img_strength slider (default mode is true_img2img)
            gr.update(visible=True),   # Show clear_input_btn
            gr.update(visible=True)    # Keep use_as_input_btn visible (always visible)
        )
    else:
        return (
            None,                      # No input_image
            gr.update(visible=True, interactive=False),  # Gray out img2img_mode selector
            gr.update(visible=True, interactive=False),  # Gray out noise_interpolation_strength slider
            gr.update(visible=True, interactive=False),  # Gray out img2img_strength slider
            gr.update(visible=False),  # Hide clear_input_btn
            gr.update(visible=True)    # Keep use_as_input_btn visible (always visible)
        )


def _copy_image_to_clipboard(image):
    """Copy generated image to clipboard."""
    if image is not None:
        try:
            import io
            import base64
            
            # Convert PIL image to base64 for JavaScript clipboard API
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Use Gradio's built-in clipboard functionality
            gr.Info("Image copied to clipboard!")
            return
        except Exception as e:
            gr.Warning(f"Failed to copy image: {e}")
    else:
        gr.Warning("No image to copy")


def _prepare_download_file(image, metadata=None):
    """Prepare PNG file with metadata for download."""
    if image is not None:
        try:
            from datetime import datetime
            from src.config import get_config
            from src.metadata import save_image_with_metadata
            from src.models import get_model_manager
            
            config = get_config()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_path = config.get_temp_path_with_name(f"generated_{timestamp}", ".png")
            
            # Get model info for metadata
            model_manager = get_model_manager()
            model_info = model_manager.model_info if hasattr(model_manager, 'model_info') else {}
            
            # Use provided metadata or empty dict
            metadata = metadata or {}
            
            # Save as PNG with embedded metadata
            save_image_with_metadata(image, temp_path, metadata, model_info)
            
            return temp_path
        except Exception as e:
            print(f"Failed to prepare download file: {e}")
            # Fallback: just save as regular PNG
            try:
                from datetime import datetime
                from src.config import get_config
                config = get_config()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_path = config.get_temp_path_with_name(f"generated_{timestamp}", ".png")
                image.save(temp_path, 'PNG')
                return temp_path
            except:
                return None
    else:
        return None


def _open_input_image_modal(image):
    """Open modal with input image in fullscreen."""
    if image is not None:
        try:
            # Extract metadata from the input image
            from src.metadata import extract_metadata_from_pil_image
            metadata = extract_metadata_from_pil_image(image)
            metadata_text = format_metadata_display(metadata) if metadata else "No metadata found"
            
            return (
                gr.update(visible=True),  # Show modal
                image,                    # Set modal image
                metadata_text            # Set modal metadata
            )
        except Exception as e:
            print(f"Error opening input image modal: {e}")
            return (
                gr.update(visible=False), # Keep modal hidden
                None,                     # No image
                f"Error loading metadata: {e}"
            )
    else:
        return (
            gr.update(visible=False), # Keep modal hidden
            None,                     # No image
            "No input image available"
        )

def _close_input_image_modal():
    """Close the input image modal."""
    return gr.update(visible=False)

def _send_image_to_chat(image, shared_image_state, tab_communication_state):
    """Send generated image to chat tab by starting a new conversation."""
    if image is not None:
        try:
            # Check if shared states are available
            if shared_image_state is None or tab_communication_state is None:
                gr.Warning("Cross-tab communication not available")
                return None, None
            
            # Start a new chat conversation with the image
            from src.chat import start_image_chat
            new_history = start_image_chat(image)
            
            # Store the new chat history in shared state
            shared_image_state.value = new_history
            
            # Set communication flag for chat tab
            tab_communication_state.value = {
                "action": "start_new_chat",
                "source": "generate",
                "chat_history": new_history,
                "timestamp": __import__("time").time()
            }
            
            gr.Info("Started new chat conversation with image - switch to Chat tab to continue")
            return shared_image_state.value, tab_communication_state.value
        except Exception as e:
            gr.Warning(f"Failed to start chat: {e}")
            # Return safe defaults
            return (shared_image_state.value if shared_image_state else None, 
                    tab_communication_state.value if tab_communication_state else None)
    else:
        gr.Warning("No image to send")
        # Return safe defaults
        return (shared_image_state.value if shared_image_state else None, 
                tab_communication_state.value if tab_communication_state else None)


def _save_metadata_to_file(metadata):
    """Save current metadata to a JSON file."""
    if not metadata:
        gr.Warning("No metadata to save")
        return
    
    try:
        import json
        from datetime import datetime
        from src.config import get_config
        
        config = get_config()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = config.get_temp_path_with_name(f"metadata_{timestamp}", ".json")
        
        # Save metadata as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        gr.Info(f"Metadata saved to: {filepath}")
        
    except Exception as e:
        gr.Warning(f"Failed to save metadata: {e}")

def _apply_input_metadata(metadata):
    """Apply metadata from input image to generation settings."""
    if not metadata:
        gr.Warning("No metadata to apply")
        return [gr.update() for _ in range(8)]  # Return no updates for all 8 outputs
    
    try:
        from src.config import get_config
        config = get_config()
        
        # Apply metadata values to UI components
        updates = []
        updates.append(gr.update(value=metadata.get('prompt', '')))  # prompt_input
        updates.append(gr.update(value=metadata.get('negative_prompt', '')))  # negative_prompt_input
        updates.append(gr.update(value=metadata.get('width', config.default_width)))  # width_input
        updates.append(gr.update(value=metadata.get('height', config.default_height)))  # height_input
        updates.append(gr.update(value=metadata.get('steps', config.default_steps)))  # steps_input
        updates.append(gr.update(value=metadata.get('cfg_scale', config.default_cfg)))  # cfg_scale_input
        updates.append(gr.update(value=metadata.get('seed', config.default_seed)))  # seed_input
        updates.append("Custom")  # aspect_ratio - set to custom when using specific dimensions
        
        gr.Info("Applied metadata from input image to generation settings")
        return updates
        
    except Exception as e:
        gr.Warning(f"Failed to apply metadata: {e}")
        return [gr.update() for _ in range(8)]  # Return no updates for all 8 outputs


def _handle_img2img_mode_change(img2img_mode):
    """Handle mode change to enable/disable appropriate strength sliders and mask controls."""
    if img2img_mode == "true_img2img":
        return (
            gr.update(visible=True, interactive=False),  # Disable noise_interpolation_strength
            gr.update(visible=True, interactive=True),   # Enable img2img_strength
            gr.update(visible=False, interactive=False), # Hide inpaint_strength
            gr.update(visible=False)                     # Hide mask_row
        )
    elif img2img_mode == "noise_interpolation":
        return (
            gr.update(visible=True, interactive=True),   # Enable noise_interpolation_strength
            gr.update(visible=True, interactive=False),  # Disable img2img_strength
            gr.update(visible=False, interactive=False), # Hide inpaint_strength
            gr.update(visible=False)                     # Hide mask_row
        )
    else:  # inpaint mode
        return (
            gr.update(visible=True, interactive=False),  # Disable noise_interpolation_strength
            gr.update(visible=True, interactive=False),  # Disable img2img_strength
            gr.update(visible=True, interactive=True),   # Enable inpaint_strength
            gr.update(visible=True)                      # Show mask_row
        )