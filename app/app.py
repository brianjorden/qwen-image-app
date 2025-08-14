"""
Main Gradio application entry point for Qwen-Image generation.

This module assembles all the tab components into a cohesive application interface.
"""

import gradio as gr

from src.config import init_config, get_config, config_exists
from src.models import get_model_manager
from src.gallery import get_session_manager

from .shared import CUSTOM_CSS
from .generate import create_generation_tab
from .gallery import create_gallery_tab
from .chat import create_chat_tab
from .analysis import create_analysis_tab
from .models import create_models_tab
from .config import create_config_tab


class QwenImageApp:
    """Main application class for Qwen-Image generation."""
    
    def __init__(self):
        self.config = get_config()
        self.model_manager = get_model_manager()
        self.session_manager = get_session_manager()
        
        # Auto-load pipeline if configured
        if self.config.auto_load_pipeline:
            try:
                print("Auto-loading pipeline and processor...")
                self.model_manager.load_processor()
                self.model_manager.build_pipeline()
                print("Auto-loading completed")
            except Exception as e:
                print(f"Auto-loading failed: {e}")
    
    def create_interface(self) -> gr.Blocks:
        """Create the complete Gradio interface."""
        
        # Set theme based on config
        theme = gr.themes.Default()
        if self.config.ui_theme == "dark":
            theme = gr.themes.Base()
        elif self.config.ui_theme == "soft":
            theme = gr.themes.Soft()
        
        with gr.Blocks(title="Qwen-Image Generator", css=CUSTOM_CSS, theme=theme) as demo:
            gr.Markdown("# Qwen-Image Text-to-Image Generator")
            
            # URL parameter handling for session
            session_state = gr.State(value=self.session_manager.get_default_session())
            
            with gr.Tabs():
                with gr.Tab("Generate"):
                    create_generation_tab(session_state)
                
                with gr.Tab("Gallery"):
                    create_gallery_tab(session_state)
                
                with gr.Tab("Chat"):
                    create_chat_tab()
                
                with gr.Tab("Analysis"):
                    create_analysis_tab()
                
                with gr.Tab("Models"):
                    create_models_tab()
                
                with gr.Tab("Config"):
                    create_config_tab()
            
            # Load URL parameters on start
            demo.load(fn=self._get_initial_session, outputs=[session_state])
        
        return demo
    
    def _get_initial_session(self):
        """Get initial session from URL or default."""
        # In production, would parse URL parameters
        return self.session_manager.get_default_session()


def create_app():
    """Create and return the Gradio app."""
    # Initialize config if needed
    if not config_exists():
        init_config()
    
    app = QwenImageApp()
    return app.create_interface()


def main():
    """Launch the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen-Image Generation App")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--share", action="store_true", help="Share via Gradio")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    
    args = parser.parse_args()
    
    # Initialize config
    init_config(args.config)
    
    # Create and launch app
    demo = create_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=False
    )


if __name__ == "__main__":
    main()