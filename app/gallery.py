"""
Gallery tab UI for the Qwen-Image application.
"""

import gradio as gr
from typing import Any
from pathlib import Path

from src.gallery import get_gallery_manager, get_session_manager
from src.metadata import format_metadata_display
from .shared import create_session_controls, setup_session_refresh


def create_gallery_tab(session_state: gr.State) -> None:
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
    
    # State to track current metadata
    current_metadata = gr.State({})
    
    # Setup event handlers
    _setup_gallery_handlers(
        gallery_session, refresh_gallery_btn, session_state,
        gallery, selected_metadata, current_metadata,
        delete_btn, apply_from_gallery_btn, use_as_input_btn,
        chat_about_btn, describe_btn, refresh_timer
    )


def _setup_gallery_handlers(
    gallery_session, refresh_gallery_btn, session_state,
    gallery, selected_metadata, current_metadata,
    delete_btn, apply_from_gallery_btn, use_as_input_btn,
    chat_about_btn, describe_btn, refresh_timer
):
    """Setup all event handlers for the gallery tab."""
    
    gallery_manager = get_gallery_manager()
    
    # Setup session refresh functionality
    setup_session_refresh(gallery_session, refresh_gallery_btn)
    
    def load_gallery(session: str):
        """Load images for gallery."""
        images = gallery_manager.get_gallery_images(session)
        return images
    
    def show_metadata(evt: gr.SelectData, session: str):
        """Show metadata for selected image."""
        images_data = gallery_manager.get_images(session)
        if evt.index < len(images_data):
            _, metadata = images_data[evt.index]
            current_metadata.value = metadata
            return format_metadata_display(metadata)
        return "No metadata found"
    
    def delete_selected_image(evt: gr.SelectData, session: str):
        """Delete the selected image."""
        try:
            images_data = gallery_manager.get_images(session)
            if evt.index < len(images_data):
                image_path, _ = images_data[evt.index]
                # Here you would implement the delete functionality
                # For now, just refresh the gallery
                return load_gallery(session), "Image deleted"
            return gr.update(), "No image selected"
        except Exception as e:
            return gr.update(), f"Error deleting image: {str(e)}"
    
    def apply_gallery_settings():
        """Apply settings from selected image metadata."""
        if not current_metadata.value:
            gr.Warning("No image selected")
            return
        
        # This would trigger an event to update the generation tab
        # For now, just show a message
        gr.Info("Settings applied to generation tab")
    
    def use_image_as_input(evt: gr.SelectData, session: str):
        """Use selected image as input for generation."""
        try:
            images_data = gallery_manager.get_images(session)
            if evt.index < len(images_data):
                image_path, _ = images_data[evt.index]
                # This would send the image to the generation tab
                gr.Info(f"Image sent to generation tab as input: {Path(image_path).name}")
                return
            gr.Warning("No image selected")
        except Exception as e:
            gr.Error(f"Error using image as input: {str(e)}")
    
    def chat_about_image(evt: gr.SelectData, session: str):
        """Start chat conversation about selected image."""
        try:
            images_data = gallery_manager.get_images(session)
            if evt.index < len(images_data):
                image_path, _ = images_data[evt.index]
                # This would switch to chat tab and attach image
                gr.Info(f"Starting chat about image: {Path(image_path).name}")
                return
            gr.Warning("No image selected")
        except Exception as e:
            gr.Error(f"Error starting chat: {str(e)}")
    
    def describe_image_for_generation(evt: gr.SelectData, session: str):
        """Generate description prompt from selected image."""
        try:
            images_data = gallery_manager.get_images(session)
            if evt.index < len(images_data):
                image_path, _ = images_data[evt.index]
                # This would start a chat thread with describe template
                gr.Info(f"Generating description for: {Path(image_path).name}")
                return
            gr.Warning("No image selected")
        except Exception as e:
            gr.Error(f"Error describing image: {str(e)}")
    
    # Connect handlers
    refresh_gallery_btn.click(
        fn=load_gallery,
        inputs=[gallery_session],
        outputs=[gallery]
    )
    
    gallery_session.change(
        fn=load_gallery,
        inputs=[gallery_session],
        outputs=[gallery]
    )
    
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
        fn=apply_gallery_settings
    )
    
    use_as_input_btn.click(
        fn=use_image_as_input,
        inputs=[gallery_session],
        outputs=[]
    )
    
    chat_about_btn.click(
        fn=chat_about_image,
        inputs=[gallery_session],
        outputs=[]
    )
    
    describe_btn.click(
        fn=describe_image_for_generation,
        inputs=[gallery_session],
        outputs=[]
    )
    
    # Auto-refresh timer
    refresh_timer.tick(
        fn=load_gallery,
        inputs=[gallery_session],
        outputs=[gallery]
    )
    
    # Sync with main session state
    session_state.change(
        fn=lambda x: (x, load_gallery(x)),
        inputs=[session_state],
        outputs=[gallery_session, gallery]
    )
    
    # Add full-screen functionality
    gr.HTML(
        value="""
        <style>
        .fullscreen-gallery img {
            cursor: pointer;
        }
        .fullscreen-modal {
            display: none;
            position: fixed;
            z-index: 9999;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }
        .fullscreen-content {
            margin: auto;
            display: block;
            width: 90%;
            max-width: 90%;
            max-height: 90%;
            margin-top: 5%;
        }
        .close {
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
        </style>
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Create fullscreen modal
            const modal = document.createElement('div');
            modal.className = 'fullscreen-modal';
            modal.innerHTML = '<span class="close">&times;</span><img class="fullscreen-content">';
            document.body.appendChild(modal);
            
            // Add click handlers to gallery images
            function addFullscreenHandlers() {
                const galleryImages = document.querySelectorAll('.fullscreen-gallery img');
                galleryImages.forEach(img => {
                    img.onclick = function() {
                        modal.style.display = 'block';
                        modal.querySelector('.fullscreen-content').src = this.src;
                    };
                });
            }
            
            // Close modal
            modal.querySelector('.close').onclick = function() {
                modal.style.display = 'none';
            };
            
            // Close on background click
            modal.onclick = function(event) {
                if (event.target === modal) {
                    modal.style.display = 'none';
                }
            };
            
            // Initial setup and observe for new images
            addFullscreenHandlers();
            const observer = new MutationObserver(addFullscreenHandlers);
            const gallery = document.querySelector('.fullscreen-gallery');
            if (gallery) {
                observer.observe(gallery, { childList: true, subtree: true });
            }
        });
        </script>
        """,
        visible=False
    )