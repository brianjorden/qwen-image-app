"""
Chat business logic for the Qwen-Image application.
"""

import tempfile
import os
import traceback
from typing import List, Dict, Any, Optional
from PIL import Image

from .models import get_model_manager
from .config import get_config
from pathlib import Path


class ChatManager:
    """Manages chat functionality and message handling."""
    
    def __init__(self):
        self.model_manager = get_model_manager()
        self.config = get_config()
    
    def get_describe_template(self) -> str:
        """Get image description template from file."""
        if hasattr(self.config, 'template_describe') and self.config.template_describe:
            try:
                path = Path(self.config.template_describe)
                if path.exists():
                    return path.read_text(encoding='utf-8').strip()
            except Exception as e:
                print(f"Failed to load describe template: {e}")
        
        # Fallback template
        return """Describe this image in detail for use as an image generation prompt. Focus on visual elements, style, composition, and mood."""
    
    
    def chat_response(self, message: str, image: Optional[Image.Image], history: List[Dict], max_new_tokens: int) -> List[Dict]:
        """Generate chat response using VL model.
        
        Args:
            message: User's text message
            image: Optional PIL image
            history: Chat history
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Updated chat history
        """
        if not self.model_manager.text_encoder:
            # Just add the error message to history - UI layer handles image display
            user_message = {"role": "user", "content": message}
            if image:
                user_message["image"] = image  # Store the actual image object
            
            new_history = history + [
                user_message,
                {"role": "assistant", "content": "Text encoder not loaded"}
            ]
            return new_history
        
        try:
            # Prepare messages for VL model
            messages = []
            
            # Add full conversation history - let the model handle 32k context properly
            for msg in history:
                if isinstance(msg, dict):
                    messages.append(msg)
            
            # Prepare current message content
            user_content = []
            
            # Add image if provided
            if image:
                # Convert PIL image to format expected by processor
                temp_path = tempfile.mktemp(suffix='.png')
                image.save(temp_path)
                user_content.append({
                    "type": "image", 
                    "image": f"file://{temp_path}"
                })
            
            # Add text
            user_content.append({"type": "text", "text": message})
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            # Generate response using VL model
            response = self.model_manager.chat(messages, max_new_tokens=int(max_new_tokens))
            
            # Clean up temp file if created
            if image and 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            # Create user message - UI layer handles image display
            user_message = {"role": "user", "content": message}
            if image:
                user_message["image"] = image  # Store the actual image object
            
            new_history = history + [
                user_message,
                {"role": "assistant", "content": response}
            ]
            return new_history
            
        except Exception as e:
            error_msg = f"Chat error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            # Create user message - UI layer handles image display
            user_message = {"role": "user", "content": message}
            if image:
                user_message["image"] = image  # Store the actual image object
            
            new_history = history + [
                user_message,
                {"role": "assistant", "content": f"Error: {str(e)}"}
            ]
            return new_history
    
    def describe_image(self, image: Optional[Image.Image]) -> str:
        """Describe image using VL model for prompt generation.
        
        Args:
            image: PIL image to describe
            
        Returns:
            Image description text
        """
        if not image:
            return ""
        
        if not self.model_manager.text_encoder:
            return "Text encoder not loaded"
        
        try:
            # Get the describe template
            describe_prompt = self.get_describe_template()
            
            # Create description request
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": f"file://{tempfile.mktemp(suffix='.png')}"
                    },
                    {
                        "type": "text", 
                        "text": describe_prompt
                    }
                ]
            }]
            
            # Save temp image
            temp_path = tempfile.mktemp(suffix='.png')
            image.save(temp_path)
            messages[0]["content"][0]["image"] = f"file://{temp_path}"
            
            # Generate description
            description = self.model_manager.chat(messages, max_new_tokens=200)
            
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return description.strip()
            
        except Exception as e:
            print(f"Image description failed: {e}")
            traceback.print_exc()
            return f"Error describing image: {str(e)}"
    


# Global chat manager instance
_chat_manager: Optional[ChatManager] = None


def get_chat_manager() -> ChatManager:
    """Get or create the global chat manager."""
    global _chat_manager
    if _chat_manager is None:
        _chat_manager = ChatManager()
    return _chat_manager