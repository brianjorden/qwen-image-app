"""
Chat tab UI for the Qwen-Image application.
"""

import gradio as gr
import gradio_modal
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from src.chat import get_chat_manager
from src.metadata import extract_metadata_from_pil_image, format_metadata_display


def create_chat_tab(shared_image_state: gr.State = None, shared_prompt_state: gr.State = None, shared_metadata_state: gr.State = None, tab_communication_state: gr.State = None) -> None:
    """Create the VL model chat tab."""
    chat_manager = get_chat_manager()
    
    # CSS for chat improvements
    css = """
    .chat-container {
        max-width: 900px !important;
        margin: 0 auto !important;
    }
    """
    
    with gr.Column(elem_classes=["chat-container"]):
        gr.Markdown("### Chat with Qwen2.5-VL Model")
        
        # Chat thread management
        with gr.Row():
            chat_threads = gr.Dropdown(
                label="Chat Thread",
                choices=["Main Conversation"],
                value="Main Conversation",
                scale=4,
            )
            new_thread_btn = gr.Button("New Thread", scale=1)
        
        chatbot = gr.Chatbot(
            height=600, 
            type='messages',
            show_copy_button=True,
            show_share_button=False,
            editable=True
        )
        
        with gr.Row():
            msg_input = gr.Textbox(
                label="Message",
                placeholder="Type your message...",
                lines=2,
                scale=3
            )
            image_input = gr.Image(
                label="Attach Image",
                type="pil",
                scale=1
            )
        
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary", scale=2)
            clear_btn = gr.Button("Clear", scale=1)
            describe_btn = gr.Button("Describe Image for Generation", scale=2)
        
        with gr.Row():
            from src.config import get_config
            config = get_config()
            max_new_tokens_input = gr.Slider(
                label="Max New Tokens",
                minimum=0,
                maximum=4096,
                value=getattr(config, 'default_max_new_tokens', 2048),
                step=16
            )
    
    # Chat image fullscreen modal
    with gradio_modal.Modal(visible=False, allow_user_close=True) as chat_image_modal:
        with gr.Column():
            gr.Markdown("### Chat Image Viewer")
            chat_modal_close_btn = gr.Button("✕ Close", scale=1)
            
            # Large image display for chat image
            chat_modal_image = gr.Image(
                label="",
                show_label=False,
                interactive=False,
                height=600,
                show_download_button=True,
                show_share_button=False
            )
            
            # Chat image metadata in modal
            chat_modal_metadata = gr.Textbox(
                label="Image Metadata",
                lines=4,
                interactive=False,
                show_label=True
            )
    
    # State for managing multiple chat threads
    chat_threads_state = gr.State({"Main Conversation": []})
    current_thread = gr.State("Main Conversation")
    
    # Setup all chat handlers
    _setup_chat_handlers(
        chat_manager, chatbot, msg_input, image_input, 
        send_btn, clear_btn, describe_btn, max_new_tokens_input,
        chat_threads, new_thread_btn,
        chat_threads_state, current_thread,
        # Chat image modal components
        chat_image_modal, chat_modal_image, chat_modal_metadata, chat_modal_close_btn,
        # Cross-tab communication states
        shared_image_state, shared_prompt_state, shared_metadata_state, tab_communication_state
    )
    
    # Add keyboard shortcuts
    gr.HTML(
        value="""
        <script>
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                const sendBtn = document.querySelector('[data-testid*="Send"] button') ||
                               document.querySelector('button:contains("Send")');
                if (sendBtn && !sendBtn.disabled) {
                    sendBtn.click();
                }
            }
        });
        </script>
        """,
        visible=False
    )


def _setup_chat_handlers(
    chat_manager, chatbot, msg_input, image_input, 
    send_btn, clear_btn, describe_btn, max_new_tokens_input,
    chat_threads, new_thread_btn,
    chat_threads_state, current_thread,
    # Chat image modal components
    chat_image_modal, chat_modal_image, chat_modal_metadata, chat_modal_close_btn,
    # Cross-tab communication states
    shared_image_state, shared_prompt_state, shared_metadata_state, tab_communication_state
):
    """Setup all chat event handlers."""
    
    def chat_and_clear(message: str, image: Optional[Image.Image], history: List[Dict], max_new_tokens: int) -> Tuple:
        """Handle chat response and clear inputs."""
        # Get business logic response
        new_history = chat_manager.chat_response(message, image, history, max_new_tokens)
        
        # Convert PIL image objects to proper Gradio 5.42.0 format
        from src.config import get_config
        config = get_config()
        
        processed_history = []
        for msg in new_history:
            if "image" in msg and msg["image"] is not None:
                # Save PIL image to temp directory with metadata preservation
                pil_image = msg["image"]
                temp_path = config.save_temp_image_with_metadata(pil_image, "chat_image")
                
                text_content = msg.get("content", "")
                role = msg.get("role", "user")
                
                # If there's text content, add it as a separate message first
                if text_content:
                    processed_history.append({"role": role, "content": text_content})
                
                # Add the image as a separate message
                processed_history.append({"role": role, "content": {"path": temp_path}})
            else:
                # Regular text message, keep as-is
                processed_history.append(msg)
        
        return processed_history, "", None  # Clear message and image inputs
    
    def clear_chat() -> None:
        """Clear chat history."""
        return None
    
    def describe_image_handler(image: Optional[Image.Image]) -> str:
        """Handle image description by placing description in message input."""
        if image:
            return chat_manager.describe_image(image)
        return ""
    
    def describe_for_generation_handler(image: Optional[Image.Image], history: List[Dict], max_new_tokens: int) -> Tuple:
        """Start a new conversation thread with describe template and image."""
        if not image:
            gr.Warning("Please upload an image first")
            return history, "", None
        
        try:
            # Get the describe template
            describe_template = chat_manager.get_describe_template()
            
            # Create new conversation with image and describe template
            new_history = chat_manager.chat_response(describe_template, image, [], max_new_tokens)
            
            gr.Info("Started new conversation with image description request")
            return new_history, "", None  # Clear inputs
        except Exception as e:
            gr.Warning(f"Failed to start description conversation: {str(e)}")
            return history, "", None
    
    def handle_retry_wrapper(history: List[Dict], retry_data: gr.RetryData) -> Tuple:
        """Handle message retry."""
        # Remove everything from the retry point onwards
        new_history = history[:retry_data.index]
        
        # Get the original message to retry
        if retry_data.index > 0:
            retry_message = history[retry_data.index - 1]
            # If it was a user message, we could regenerate the response
            if retry_message.get("role") == "user":
                if isinstance(retry_message["content"], str):
                    # For now, just return the trimmed history
                    # Could implement actual retry logic here
                    pass
        
        return new_history, "", None
    
    def handle_undo_wrapper(history: List[Dict], undo_data: gr.UndoData) -> Tuple:
        """Handle message undo."""
        print(f"Undo triggered - Index: {undo_data.index}")
        
        if undo_data.index < len(history):
            # Remove the message at the specified index and all subsequent messages
            new_history = history[:undo_data.index]
            undone_message = history[undo_data.index]["content"]
            
            # If undone message is a dict (like an image), convert to string
            if isinstance(undone_message, dict):
                undone_message = undone_message.get("alt_text", "")
            
            return new_history, undone_message
        
        return history, ""
    
    def handle_edit_wrapper(history: List[Dict], edit_data: gr.EditData) -> List[Dict]:
        """Handle message editing."""
        print(f"Edit triggered - Index: {edit_data.index}, Value: {edit_data.value}")
        
        if edit_data.index < len(history):
            # Update the message content at the specified index
            history[edit_data.index]["content"] = edit_data.value
            # Remove any subsequent messages (conversation continues from edited point)
            new_history = history[:edit_data.index + 1]
            return new_history
        
        return history
    
    def create_new_thread_auto(threads_state: Dict, current_thread_name: str):
        """Create a new chat thread with auto-generated name."""
        import datetime
        
        # Generate thread name with timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M")
        thread_count = len(threads_state) + 1
        thread_name = f"Chat {thread_count} ({timestamp})"
        
        # Ensure unique name
        while thread_name in threads_state:
            thread_count += 1
            thread_name = f"Chat {thread_count} ({timestamp})"
        
        # Add new thread
        threads_state[thread_name] = []
        thread_choices = list(threads_state.keys())
        
        gr.Info(f"Started new thread: {thread_name}")
        
        return (
            threads_state,
            thread_name,  # Switch to new thread
            gr.update(choices=thread_choices, value=thread_name),  # Update dropdown
            []  # Clear chatbot for new thread
        )
    
    def switch_thread(thread_name: str, threads_state: Dict):
        """Switch to a different chat thread."""
        if thread_name in threads_state:
            return threads_state[thread_name], thread_name
        return [], thread_name
    
    def handle_cross_tab_communication(comm_data, threads_state: Dict, current_thread_name: str):
        """Handle messages from other tabs (enhance, send to chat, etc.)."""
        if not comm_data:
            return threads_state, current_thread_name, gr.update(), gr.update()
        
        try:
            action = comm_data.get("action")
            source = comm_data.get("source")
            
            if action == "start_new_chat":
                chat_history = comm_data.get("chat_history", [])
                
                # Create a new thread for this conversation
                import datetime
                timestamp = datetime.datetime.now().strftime("%H:%M")
                
                if source == "enhance":
                    enhancement_type = comm_data.get("enhancement_type", "")
                    thread_name = f"Enhanced {enhancement_type} prompt ({timestamp})"
                elif source == "generate":
                    thread_name = f"Image chat ({timestamp})"
                else:
                    thread_name = f"New chat from {source} ({timestamp})"
                
                # Ensure unique name
                counter = 1
                original_name = thread_name
                while thread_name in threads_state:
                    thread_name = f"{original_name} #{counter}"
                    counter += 1
                
                # Add new thread with the chat history
                threads_state[thread_name] = chat_history
                thread_choices = list(threads_state.keys())
                
                gr.Info(f"Started new conversation: {thread_name}")
                
                return (
                    threads_state,
                    thread_name,  # Switch to new thread
                    gr.update(choices=thread_choices, value=thread_name),  # Update dropdown
                    chat_history  # Update chatbot with new conversation
                )
            
        except Exception as e:
            gr.Warning(f"Failed to handle cross-tab communication: {e}")
        
        # Return unchanged state if nothing to do
        return threads_state, current_thread_name, gr.update(), gr.update()
    
    # Main chat handlers
    send_btn.click(
        fn=chat_and_clear,
        inputs=[msg_input, image_input, chatbot, max_new_tokens_input],
        outputs=[chatbot, msg_input, image_input]
    )
    
    msg_input.submit(
        fn=chat_and_clear,
        inputs=[msg_input, image_input, chatbot, max_new_tokens_input],
        outputs=[chatbot, msg_input, image_input]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot]
    )
    
    describe_btn.click(
        fn=describe_for_generation_handler,
        inputs=[image_input, chatbot, max_new_tokens_input],
        outputs=[chatbot, msg_input, image_input]
    )
    
    # Thread management handlers
    new_thread_btn.click(
        fn=create_new_thread_auto,
        inputs=[chat_threads_state, current_thread],
        outputs=[chat_threads_state, current_thread, chat_threads, chatbot]
    )
    
    chat_threads.change(
        fn=switch_thread,
        inputs=[chat_threads, chat_threads_state],
        outputs=[chatbot, current_thread]
    )
    
    # Monitor for cross-tab communication
    if tab_communication_state:
        tab_communication_state.change(
            fn=handle_cross_tab_communication,
            inputs=[tab_communication_state, chat_threads_state, current_thread],
            outputs=[chat_threads_state, current_thread, chat_threads, chatbot]
        )
    
    # Chat interaction event handlers
    chatbot.retry(handle_retry_wrapper, [chatbot], [chatbot, msg_input, image_input])
    chatbot.undo(handle_undo_wrapper, [chatbot], [chatbot, msg_input])
    chatbot.edit(handle_edit_wrapper, [chatbot], [chatbot])
    
    # Chat image modal handlers
    def open_chat_image_modal(evt: gr.SelectData, history: List[Dict]):
        """Open modal with selected chat image in fullscreen."""
        try:
            if evt.index >= len(history):
                return gr.update(visible=False), None, "No image found"
            
            message = history[evt.index]
            image_content = message.get("content")
            
            # Check if the clicked content is an image (path format)
            if isinstance(image_content, dict) and "path" in image_content:
                image_path = image_content["path"]
                try:
                    from PIL import Image
                    pil_image = Image.open(image_path)
                    
                    # Try to extract metadata
                    metadata = extract_metadata_from_pil_image(pil_image)
                    metadata_text = format_metadata_display(metadata) if metadata else "No metadata found"
                    
                    return (
                        gr.update(visible=True),  # Show modal
                        pil_image,                # Set modal image
                        metadata_text            # Set modal metadata
                    )
                except Exception as e:
                    return gr.update(visible=False), None, f"Error loading image: {e}"
            
            return gr.update(visible=False), None, "Selected item is not an image"
            
        except Exception as e:
            return gr.update(visible=False), None, f"Error: {e}"
    
    def close_chat_image_modal():
        """Close the chat image modal."""
        return gr.update(visible=False)
    
    # Set up chat image modal events
    chatbot.select(
        fn=open_chat_image_modal,
        inputs=[chatbot],
        outputs=[chat_image_modal, chat_modal_image, chat_modal_metadata]
    )
    
    chat_modal_close_btn.click(
        fn=close_chat_image_modal,
        outputs=[chat_image_modal]
    )
    
