"""
Chat tab UI for the Qwen-Image application.
"""

import gradio as gr
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from src.chat import get_chat_manager


def create_chat_tab() -> None:
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
                scale=3,
            )
            new_thread_btn = gr.Button("New Thread", scale=1)
            thread_name_input = gr.Textbox(
                label="Thread Name",
                placeholder="Enter thread name...",
                scale=2,
                visible=False
            )
            create_thread_btn = gr.Button("Create", scale=1, visible=False)
            cancel_thread_btn = gr.Button("Cancel", scale=1, visible=False)
        
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
            max_new_tokens_input = gr.Slider(
                label="Max New Tokens",
                minimum=50,
                maximum=4000,
                value=512,
                step=50
            )
    
    # State for managing multiple chat threads
    chat_threads_state = gr.State({"Main Conversation": []})
    current_thread = gr.State("Main Conversation")
    
    # Setup all chat handlers
    _setup_chat_handlers(
        chat_manager, chatbot, msg_input, image_input, 
        send_btn, clear_btn, describe_btn, max_new_tokens_input,
        chat_threads, new_thread_btn, thread_name_input, create_thread_btn, cancel_thread_btn,
        chat_threads_state, current_thread
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
    chat_threads, new_thread_btn, thread_name_input, create_thread_btn, cancel_thread_btn,
    chat_threads_state, current_thread
):
    """Setup all chat event handlers."""
    
    def chat_and_clear(message: str, image: Optional[Image.Image], history: List[Dict], max_new_tokens: int) -> Tuple:
        """Handle chat response and clear inputs."""
        import tempfile
        
        # Get business logic response
        new_history = chat_manager.chat_response(message, image, history, max_new_tokens)
        
        # Convert any image objects to Gradio-compatible format
        for msg in new_history:
            if "image" in msg and msg["image"] is not None:
                # Convert PIL image to temporary file for Gradio display
                temp_path = tempfile.mktemp(suffix='.png')
                msg["image"].save(temp_path)
                # Replace image object with Gradio-compatible format
                msg["content"] = [
                    {"type": "text", "text": msg["content"]},
                    {"path": temp_path, "alt_text": "Uploaded image"}
                ]
                del msg["image"]  # Remove the PIL object
        
        return new_history, "", None  # Clear message and image inputs
    
    def clear_chat() -> None:
        """Clear chat history."""
        return None
    
    def describe_image_handler(image: Optional[Image.Image]) -> str:
        """Handle image description."""
        return chat_manager.describe_image(image)
    
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
    
    def show_thread_creation_ui():
        """Show thread creation input fields."""
        return (
            gr.update(visible=True),  # thread_name_input
            gr.update(visible=True),  # create_thread_btn
            gr.update(visible=True)   # cancel_thread_btn
        )
    
    def hide_thread_creation_ui():
        """Hide thread creation input fields."""
        return (
            gr.update(visible=False, value=""),  # thread_name_input
            gr.update(visible=False),           # create_thread_btn
            gr.update(visible=False)            # cancel_thread_btn
        )
    
    def create_new_thread(thread_name: str, threads_state: Dict, current_thread_name: str):
        """Create a new chat thread."""
        if not thread_name.strip():
            gr.Warning("Please enter a thread name")
            return threads_state, current_thread_name, gr.update(), *hide_thread_creation_ui(), gr.update()
        
        # Add new thread
        threads_state[thread_name] = []
        thread_choices = list(threads_state.keys())
        
        return (
            threads_state,
            thread_name,  # Switch to new thread
            gr.update(choices=thread_choices, value=thread_name),  # Update dropdown
            *hide_thread_creation_ui(),  # Hide creation UI
            []  # Clear chatbot for new thread
        )
    
    def switch_thread(thread_name: str, threads_state: Dict):
        """Switch to a different chat thread."""
        if thread_name in threads_state:
            return threads_state[thread_name], thread_name
        return [], thread_name
    
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
        fn=describe_image_handler,
        inputs=[image_input],
        outputs=[msg_input]
    )
    
    # Thread management handlers
    new_thread_btn.click(
        fn=show_thread_creation_ui,
        outputs=[thread_name_input, create_thread_btn, cancel_thread_btn]
    )
    
    cancel_thread_btn.click(
        fn=hide_thread_creation_ui,
        outputs=[thread_name_input, create_thread_btn, cancel_thread_btn]
    )
    
    create_thread_btn.click(
        fn=create_new_thread,
        inputs=[thread_name_input, chat_threads_state, current_thread],
        outputs=[chat_threads_state, current_thread, chat_threads, thread_name_input, create_thread_btn, cancel_thread_btn, chatbot]
    )
    
    chat_threads.change(
        fn=switch_thread,
        inputs=[chat_threads, chat_threads_state],
        outputs=[chatbot, current_thread]
    )
    
    # Chat interaction event handlers
    chatbot.retry(handle_retry_wrapper, [chatbot], [chatbot, msg_input, image_input])
    chatbot.undo(handle_undo_wrapper, [chatbot], [chatbot, msg_input])
    chatbot.edit(handle_edit_wrapper, [chatbot], [chatbot])
    
