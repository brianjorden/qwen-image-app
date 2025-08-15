"""
Gallery tab UI for the qwen-image-app.
"""

import gradio as gr
import gradio_modal
from typing import Any
from pathlib import Path

from src.gallery import get_gallery_manager, get_session_manager
from src.metadata import format_metadata_display
from .shared import create_session_controls, setup_session_refresh


def create_gallery_tab(session_state: gr.State, shared_image_state: gr.State = None, shared_prompt_state: gr.State = None, shared_metadata_state: gr.State = None, tab_communication_state: gr.State = None) -> None:
    """Create the gallery tab."""
    gallery_manager = get_gallery_manager()
    session_manager = get_session_manager()
    
    with gr.Column():
        with gr.Row():
            gallery_session, refresh_gallery_btn = create_session_controls()
        
        gallery = gr.Gallery(
            label="Session Images",
            show_label=True,
            elem_id="gallery",
            columns=4,
            rows=3,
            object_fit="contain",
            height="auto",
            elem_classes=["fullscreen-gallery"]
        )
        
        # Fullscreen modal for images
        with gradio_modal.Modal(visible=False, allow_user_close=True) as fullscreen_modal:
            with gr.Column():
                gr.Markdown("### Image Viewer")
                with gr.Row():
                    prev_btn = gr.Button("◀ Previous", scale=1)
                    close_modal_btn = gr.Button("✕ Close", scale=1)
                    next_btn = gr.Button("Next ▶", scale=1)
                
                # Large image display
                modal_image = gr.Image(
                    label="",
                    show_label=False,
                    interactive=False,
                    height=600,
                    show_download_button=True,
                    show_share_button=False
                )
                
                # Image info in modal
                modal_metadata = gr.Textbox(
                    label="Image Metadata",
                    lines=4,
                    interactive=False,
                    show_label=True
                )
        
        selected_metadata = gr.Textbox(
            label="Selected Image Metadata",
            lines=10,
            interactive=False
        )
        
        # Auto-refresh timer (hidden)
        refresh_timer = gr.Timer(5)  # Refresh every 5 seconds
        
        with gr.Row():
            apply_from_gallery_btn = gr.Button("Apply Settings")
            use_as_input_btn = gr.Button("Use as Input")
            chat_about_btn = gr.Button("Chat About Image")
            describe_btn = gr.Button("Describe Image")
        
        with gr.Row():
            delete_btn = gr.Button("Delete Selected", variant="stop")
    
    # State to track current metadata and modal state
    current_metadata = gr.State({})
    current_images = gr.State([])  # List of all images in gallery
    current_index = gr.State(0)    # Current image index in modal
    
    # Setup event handlers
    _setup_gallery_handlers(
        gallery_session, refresh_gallery_btn, session_state,
        gallery, selected_metadata, current_metadata,
        delete_btn, apply_from_gallery_btn, use_as_input_btn,
        chat_about_btn, describe_btn, refresh_timer,
        # Modal components
        fullscreen_modal, modal_image, modal_metadata,
        prev_btn, next_btn, close_modal_btn,
        current_images, current_index,
        # Cross-tab communication states
        shared_image_state, shared_prompt_state, shared_metadata_state, tab_communication_state
    )


def _setup_gallery_handlers(
    gallery_session, refresh_gallery_btn, session_state,
    gallery, selected_metadata, current_metadata,
    delete_btn, apply_from_gallery_btn, use_as_input_btn,
    chat_about_btn, describe_btn, refresh_timer,
    # Modal components
    fullscreen_modal, modal_image, modal_metadata,
    prev_btn, next_btn, close_modal_btn,
    current_images, current_index,
    # Cross-tab communication states
    shared_image_state, shared_prompt_state, shared_metadata_state, tab_communication_state
):
    """Setup all event handlers for the gallery tab."""
    
    gallery_manager = get_gallery_manager()
    
    # Setup session refresh functionality
    setup_session_refresh(gallery_session, refresh_gallery_btn)
    
    def load_gallery(session: str):
        """Load images for gallery."""
        images = gallery_manager.get_gallery_images(session)
        # Also update the current_images state for modal navigation
        current_images.value = images if images else []
        return images, images if images else []
    
    def show_metadata(evt: gr.SelectData, session: str):
        """Show metadata for selected image."""
        images_data = gallery_manager.get_images(session)
        if evt.index < len(images_data):
            _, metadata = images_data[evt.index]
            current_metadata.value = metadata
            return format_metadata_display(metadata)
        return "No metadata found"
    
    def open_fullscreen_modal(evt: gr.SelectData, session: str, images_list):
        """Open modal with selected image in fullscreen."""
        images_data = gallery_manager.get_images(session)
        if evt.index < len(images_data) and evt.index < len(images_list):
            image_path, metadata = images_data[evt.index]
            
            # Load the image
            try:
                from PIL import Image
                pil_image = Image.open(image_path)
                metadata_text = format_metadata_display(metadata)
                
                # Update states
                current_images.value = images_list
                current_index.value = evt.index
                
                return (
                    gr.update(visible=True),    # Show modal
                    pil_image,                  # Set modal image
                    metadata_text,              # Set modal metadata
                    images_list,                # Update images state
                    evt.index                   # Update index state
                )
            except Exception as e:
                print(f"Error opening image: {e}")
                return (
                    gr.update(visible=False),   # Keep modal hidden
                    None,                       # No image
                    f"Error loading image: {e}",
                    images_list,                # Keep current images
                    0                           # Reset index
                )
        
        return (
            gr.update(visible=False), None, "No image selected", images_list, 0
        )
    
    def close_modal():
        """Close the fullscreen modal."""
        return gr.update(visible=False)
    
    def navigate_image(direction: int, images_list, current_idx):
        """Navigate to previous/next image in modal."""
        if not images_list or len(images_list) == 0:
            return None, "No images available", current_idx
        
        new_index = current_idx + direction
        if new_index < 0:
            new_index = len(images_list) - 1
        elif new_index >= len(images_list):
            new_index = 0
        
        # Get the image data for the new index
        try:
            session = gallery_session.value if hasattr(gallery_session, 'value') else "main"
            images_data = gallery_manager.get_images(session)
            
            if new_index < len(images_data):
                image_path, metadata = images_data[new_index]
                from PIL import Image
                pil_image = Image.open(image_path)
                metadata_text = format_metadata_display(metadata)
                
                return pil_image, metadata_text, new_index
        except Exception as e:
            print(f"Error navigating to image: {e}")
        
        return None, "Error loading image", current_idx
    
    def delete_selected_image(session: str):
        """Delete the currently selected image."""
        try:
            if not current_metadata.value:
                return gr.update(), "No image selected"
            
            # Get the image path from metadata
            image_path = current_metadata.value.get('image_path')
            if not image_path:
                return gr.update(), "Cannot determine image path"
            
            # Actually delete the image using gallery manager
            if gallery_manager.delete_image(image_path):
                # Clear current metadata since image is deleted
                current_metadata.value = {}
                # Refresh the gallery after successful deletion
                updated_gallery = load_gallery(session)
                return updated_gallery, ""  # Clear metadata display
            else:
                return gr.update(), f"Failed to delete: {Path(image_path).name}"
        except Exception as e:
            return gr.update(), f"Error deleting image: {str(e)}"
    
    def apply_gallery_settings():
        """Apply settings from selected image metadata."""
        if not current_metadata.value:
            gr.Warning("No image selected")
            return
        
        # Store metadata in shared state for generation tab
        if shared_metadata_state is not None:
            shared_metadata_state.value = current_metadata.value
            
            # Set communication flag for generation tab
            if tab_communication_state is not None:
                tab_communication_state.value = {
                    "action": "apply_metadata",
                    "source": "gallery",
                    "timestamp": __import__("time").time()
                }
        
        gr.Info("Settings applied to generation tab")
        return shared_metadata_state.value if shared_metadata_state else None, tab_communication_state.value if tab_communication_state else None
    
    def use_image_as_input(evt: gr.SelectData, session: str):
        """Use selected image as input for generation."""
        try:
            images_data = gallery_manager.get_images(session)
            if evt.index < len(images_data):
                image_path, _ = images_data[evt.index]
                
                # Load the image
                from PIL import Image
                pil_image = Image.open(image_path)
                
                # Store image in shared state for generation tab
                if shared_image_state is not None:
                    shared_image_state.value = pil_image
                    
                    # Set communication flag for generation tab
                    if tab_communication_state is not None:
                        tab_communication_state.value = {
                            "action": "use_as_input",
                            "source": "gallery",
                            "timestamp": __import__("time").time()
                        }
                
                gr.Info(f"Image sent to generation tab as input: {Path(image_path).name}")
                return shared_image_state.value if shared_image_state else None, tab_communication_state.value if tab_communication_state else None
            gr.Warning("No image selected")
            return shared_image_state.value if shared_image_state else None, tab_communication_state.value if tab_communication_state else None
        except Exception as e:
            gr.Error(f"Error using image as input: {str(e)}")
            return shared_image_state.value if shared_image_state else None, tab_communication_state.value if tab_communication_state else None
    
    def chat_about_image(session: str):
        """Start chat conversation about selected image."""
        try:
            # Check if we have a currently selected image
            if not current_metadata.value:
                gr.Warning("No image selected. Please click on an image first.")
                return (shared_image_state.value if shared_image_state else None, 
                        tab_communication_state.value if tab_communication_state else None)
            
            # Get image path from current metadata
            image_path = current_metadata.value.get('image_path')
            if not image_path:
                gr.Warning("Cannot determine image path from selected image.")
                return (shared_image_state.value if shared_image_state else None, 
                        tab_communication_state.value if tab_communication_state else None)
                
            # Load the image
            from PIL import Image
            pil_image = Image.open(image_path)
            
            # Store image in shared state for chat tab
            if shared_image_state is not None:
                shared_image_state.value = pil_image
                
                # Set communication flag for chat tab
                if tab_communication_state is not None:
                    tab_communication_state.value = {
                        "action": "start_chat",
                        "source": "gallery",
                        "timestamp": __import__("time").time()
                    }
            
            gr.Info(f"Starting chat about image: {Path(image_path).name}")
            return (shared_image_state.value if shared_image_state else None, 
                    tab_communication_state.value if tab_communication_state else None)
        except Exception as e:
            gr.Error(f"Error starting chat: {str(e)}")
            return (shared_image_state.value if shared_image_state else None, 
                    tab_communication_state.value if tab_communication_state else None)
    
    def describe_image_for_generation(session: str):
        """Generate description prompt from selected image."""
        try:
            # Check if we have a currently selected image
            if not current_metadata.value:
                gr.Warning("No image selected. Please click on an image first.")
                return (shared_prompt_state.value if shared_prompt_state else None, 
                        tab_communication_state.value if tab_communication_state else None)
            
            # Get image path from current metadata
            image_path = current_metadata.value.get('image_path')
            if not image_path:
                gr.Warning("Cannot determine image path from selected image.")
                return (shared_prompt_state.value if shared_prompt_state else None, 
                        tab_communication_state.value if tab_communication_state else None)
                
            # Load the image and generate description
            from PIL import Image
            from src.chat import get_chat_manager
            pil_image = Image.open(image_path)
            chat_manager = get_chat_manager()
            
            # Generate description using the chat manager
            description = chat_manager.describe_image(pil_image)
            
            # Store the description in shared prompt state
            if shared_prompt_state is not None:
                shared_prompt_state.value = description
                
                # Set communication flag for generation tab
                if tab_communication_state is not None:
                    tab_communication_state.value = {
                        "action": "apply_prompt",
                        "source": "gallery",
                        "timestamp": __import__("time").time()
                    }
            
            gr.Info(f"Generated description for: {Path(image_path).name}")
            return (shared_prompt_state.value if shared_prompt_state else None, 
                    tab_communication_state.value if tab_communication_state else None)
        except Exception as e:
            gr.Error(f"Error describing image: {str(e)}")
            return (shared_prompt_state.value if shared_prompt_state else None, 
                    tab_communication_state.value if tab_communication_state else None)
    
    # Connect handlers
    refresh_gallery_btn.click(
        fn=load_gallery,
        inputs=[gallery_session],
        outputs=[gallery, current_images]
    )
    
    gallery_session.change(
        fn=load_gallery,
        inputs=[gallery_session],
        outputs=[gallery, current_images]
    )
    
    # Gallery select opens fullscreen modal
    gallery.select(
        fn=open_fullscreen_modal,
        inputs=[gallery_session, current_images],
        outputs=[fullscreen_modal, modal_image, modal_metadata, current_images, current_index]
    )
    
    # Also update the metadata display in the main area
    gallery.select(
        fn=show_metadata,
        inputs=[gallery_session],
        outputs=[selected_metadata]
    )
    
    delete_btn.click(
        fn=delete_selected_image,
        inputs=[gallery_session],
        outputs=[gallery, selected_metadata]
    )
    
    apply_from_gallery_btn.click(
        fn=apply_gallery_settings,
        outputs=[shared_metadata_state, tab_communication_state] if shared_metadata_state and tab_communication_state else []
    )
    
    use_as_input_btn.click(
        fn=use_image_as_input,
        inputs=[gallery_session],
        outputs=[shared_image_state, tab_communication_state] if shared_image_state and tab_communication_state else []
    )
    
    chat_about_btn.click(
        fn=chat_about_image,
        inputs=[gallery_session],
        outputs=[shared_image_state, tab_communication_state] if shared_image_state and tab_communication_state else []
    )
    
    describe_btn.click(
        fn=describe_image_for_generation,
        inputs=[gallery_session],
        outputs=[shared_prompt_state, tab_communication_state] if shared_prompt_state and tab_communication_state else []
    )
    
    # Modal navigation handlers
    close_modal_btn.click(
        fn=close_modal,
        outputs=[fullscreen_modal]
    )
    
    prev_btn.click(
        fn=lambda imgs, idx: navigate_image(-1, imgs, idx),
        inputs=[current_images, current_index],
        outputs=[modal_image, modal_metadata, current_index]
    )
    
    next_btn.click(
        fn=lambda imgs, idx: navigate_image(1, imgs, idx),
        inputs=[current_images, current_index],
        outputs=[modal_image, modal_metadata, current_index]
    )
    
    # Auto-refresh timer
    refresh_timer.tick(
        fn=load_gallery,
        inputs=[gallery_session],
        outputs=[gallery, current_images]
    )
    
    # Sync with main session state
    def sync_session_and_load(session):
        gallery_data, images_list = load_gallery(session)
        return session, gallery_data, images_list
    
    session_state.change(
        fn=sync_session_and_load,
        inputs=[session_state],
        outputs=[gallery_session, gallery, current_images]
    )
    
    # Add keyboard navigation for modal (using Gradio's built-in key handlers)
    gr.HTML(
        value="""
        <script>
        // Add keyboard navigation for fullscreen modal
        document.addEventListener('keydown', function(e) {
            const modal = document.querySelector('[data-testid*="modal"]');
            if (modal && modal.style.display !== 'none') {
                if (e.key === 'ArrowLeft') {
                    e.preventDefault();
                    const prevBtn = document.querySelector('button:contains("◀ Previous")');
                    if (prevBtn) prevBtn.click();
                } else if (e.key === 'ArrowRight') {
                    e.preventDefault();
                    const nextBtn = document.querySelector('button:contains("Next ▶")');
                    if (nextBtn) nextBtn.click();
                } else if (e.key === 'Escape') {
                    e.preventDefault();
                    const closeBtn = document.querySelector('button:contains("✕ Close")');
                    if (closeBtn) closeBtn.click();
                }
            }
        });
        </script>
        """,
        visible=False
    )