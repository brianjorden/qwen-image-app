"""
Analysis tab UI for the Qwen-Image application.
"""

import gradio as gr
from typing import Dict, Any, Tuple

from src.analysis import get_analyzer


def create_analysis_tab() -> None:
    """Create the encoder analysis tab."""
    analyzer = get_analyzer()
    
    with gr.Column():
        gr.Markdown("### Text Encoder Analysis")
        
        with gr.Row():
            analysis_prompt = gr.Textbox(
                label="Prompt to Analyze",
                placeholder="Enter prompt to analyze...",
                lines=4,
                scale=3
            )
        
        with gr.Row():
            use_template_check = gr.Checkbox(
                label="Use Template",
                value=True
            )
            max_tokens_slider = gr.Slider(
                minimum=16,
                maximum=2048,
                value=1024,
                step=16,
                label="Max New Tokens"
            )
            use_cross_check = gr.Checkbox(
                label="Cross-Encoder Analysis",
                value=True
            )
        
        analyze_btn = gr.Button("Analyze", variant="primary")
        
        token_info_display = gr.Textbox(
            label="Token Information",
            lines=2,
            interactive=False
        )
        
        primary_output = gr.Textbox(
            label="Primary Encoder Output",
            lines=8,
            interactive=False
        )
        
        # Secondary encoder analysis (collapsible)
        with gr.Accordion("Secondary Encoder Analysis", open=False) as secondary_accordion:
            gr.Markdown("Load a second encoder in the Models tab for comparison analysis.")
            
            load_secondary_btn = gr.Button("Go to Models Tab")
            
            with gr.Column(visible=False) as comparison_section:
                gr.Markdown("### Encoder Comparison")
                
                with gr.Row():
                    alt_output = gr.Textbox(
                        label="Secondary Encoder Output",
                        lines=8,
                        interactive=False
                    )
                    similarity_output = gr.Textbox(
                        label="Vector Similarity",
                        lines=8,
                        interactive=False
                    )
                
                with gr.Row():
                    cross_primary_alt = gr.Textbox(
                        label="Cross: Primary→Secondary",
                        lines=6,
                        interactive=False
                    )
                    cross_alt_primary = gr.Textbox(
                        label="Cross: Secondary→Primary",
                        lines=6,
                        interactive=False
                    )
    
    # Setup event handlers
    _setup_analysis_handlers(
        analyzer, analysis_prompt, use_template_check,
        max_tokens_slider, use_cross_check, analyze_btn,
        token_info_display, primary_output, alt_output, similarity_output,
        cross_primary_alt, cross_alt_primary, load_secondary_btn, comparison_section
    )


def _setup_analysis_handlers(
    analyzer, analysis_prompt, use_template_check,
    max_tokens_slider, use_cross_check, analyze_btn,
    token_info_display, primary_output, alt_output, similarity_output,
    cross_primary_alt, cross_alt_primary, load_secondary_btn, comparison_section
):
    """Setup all event handlers for the analysis tab."""
    
    def run_analysis(
        prompt: str, use_template: bool, 
        max_tokens: int, use_cross: bool
    ) -> Tuple[str, str, str, str, str, str]:
        """Run encoder analysis."""
        if not prompt.strip():
            return "Empty prompt", "", "", "", "", ""
        
        results = analyzer.analyze_prompt(
            prompt, "greedy", use_template, max_tokens, use_cross
        )
        
        # Format token info
        token_info = results.get('token_info', {})
        token_text = f"Total: {token_info.get('total', 0)} | Content: {token_info.get('content', 0)}"
        if token_info.get('truncated'):
            token_text += " | ⚠️ Truncated"
        
        # Get outputs
        primary = results.get('primary', {})
        primary_text = f"{primary.get('output', '')}\n\n---\nTokens: {primary.get('metadata', {})}"
        
        alt = results.get('alternative', {})
        alt_text = f"{alt.get('output', '')}\n\n---\nTokens: {alt.get('metadata', {})}" if alt else "No alternative encoder loaded"
        
        cross_pa = results.get('cross_primary_alt', {})
        cross_pa_text = f"{cross_pa.get('output', '')}\n\n---\nTokens: {cross_pa.get('metadata', {})}" if cross_pa else ""
        
        cross_ap = results.get('cross_alt_primary', {})
        cross_ap_text = f"{cross_ap.get('output', '')}\n\n---\nTokens: {cross_ap.get('metadata', {})}" if cross_ap else ""
        
        # Calculate vector similarity if both encoders available
        similarity_text = ""
        if alt:
            try:
                # This would calculate cosine similarity between embeddings
                similarity_text = "Similarity analysis available when both encoders loaded"
            except Exception:
                similarity_text = "Similarity calculation failed"
        
        return token_text, primary_text, alt_text, similarity_text, cross_pa_text, cross_ap_text
    
    # Connect handlers (moved below with enhanced functionality)
    
    # Handler for going to models tab
    def go_to_models_tab():
        """Provide guidance for loading secondary encoder."""
        from src.models import get_model_manager
        
        model_manager = get_model_manager()
        if model_manager.text_encoder_alt is not None:
            gr.Info("Secondary encoder already loaded! You can now run comparison analysis.")
            return gr.update(visible=True)  # Show comparison section
        else:
            gr.Info("To load a secondary encoder: 1) Switch to the Models tab, 2) Click 'Load Alternative Text Encoder', 3) Return here for comparison analysis")
            return gr.update(visible=False)  # Keep comparison section hidden
    
    load_secondary_btn.click(
        fn=go_to_models_tab,
        outputs=[comparison_section]
    )
    
    # Check for secondary encoder on page load/refresh
    def check_secondary_encoder():
        """Check if secondary encoder is loaded and update UI accordingly."""
        from src.models import get_model_manager
        
        model_manager = get_model_manager()
        has_secondary = model_manager.text_encoder_alt is not None
        return gr.update(visible=has_secondary)
    
    # Update comparison section visibility when analyze button is clicked
    def enhanced_run_analysis(prompt: str, use_template: bool, max_tokens: int, use_cross: bool):
        """Run analysis and update comparison section visibility."""
        results = run_analysis(prompt, use_template, max_tokens, use_cross)
        
        # Check if secondary encoder is available
        from src.models import get_model_manager
        model_manager = get_model_manager()
        has_secondary = model_manager.text_encoder_alt is not None
        comparison_visibility = gr.update(visible=has_secondary)
        
        return results + (comparison_visibility,)
    
    analyze_btn.click(
        fn=enhanced_run_analysis,
        inputs=[
            analysis_prompt, use_template_check,
            max_tokens_slider, use_cross_check
        ],
        outputs=[
            token_info_display, primary_output, alt_output, similarity_output,
            cross_primary_alt, cross_alt_primary, comparison_section
        ]
    )