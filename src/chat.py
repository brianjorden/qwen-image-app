"""
Chat business logic for the qwen-image-app.
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
                # Convert PIL image to format expected by processor with metadata preservation
                temp_path = self.config.save_temp_image_with_metadata(image, "vl_input")
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
            
            # Save temp image with metadata preservation
            temp_path = self.config.save_temp_image_with_metadata(image, "describe_input")
            
            # Create description request
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": f"file://{temp_path}"
                    },
                    {
                        "type": "text", 
                        "text": describe_prompt
                    }
                ]
            }]
            
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
    
    def get_enhancement_template(self) -> str:
        """Get prompt enhancement template from file."""
        if hasattr(self.config, 'template_enhancement') and self.config.template_enhancement:
            try:
                path = Path(self.config.template_enhancement)
                if path.exists():
                    return path.read_text(encoding='utf-8').strip()
            except Exception as e:
                print(f"Failed to load enhancement template: {e}")
        
        # Fallback template
        return """Please enhance this prompt for image generation by adding more detailed and expressive descriptions while preserving the original meaning.

User Input: {}

Enhanced Prompt:"""
    
    def enhance_prompt(self, prompt: str, is_negative: bool = False) -> str:
        """Enhance a prompt using the VL model.
        
        Args:
            prompt: The prompt to enhance
            is_negative: Whether this is a negative prompt
            
        Returns:
            Enhanced prompt text
        """
        if not prompt or not prompt.strip():
            return ""
        
        if not self.model_manager.text_encoder:
            return "Text encoder not loaded"
        
        try:
            # Get the enhancement template
            enhancement_template = self.get_enhancement_template()
            
            # Format the template with the user's prompt
            enhancement_prompt = enhancement_template.format(prompt)
            
            # Add negative prompt context if needed
            if is_negative:
                enhancement_prompt = f"This is a negative prompt (things to avoid). {enhancement_prompt}"
            
            # Create enhancement request
            messages = [{
                "role": "user",
                "content": enhancement_prompt
            }]
            
            # Generate enhancement
            enhanced = self.model_manager.chat(messages, max_new_tokens=300)
            
            return enhanced.strip()
            
        except Exception as e:
            print(f"Prompt enhancement failed: {e}")
            traceback.print_exc()
            return f"Error enhancing prompt: {str(e)}"
    


# Global chat manager instance
_chat_manager: Optional[ChatManager] = None


def get_chat_manager() -> ChatManager:
    """Get or create the global chat manager."""
    global _chat_manager
    if _chat_manager is None:
        _chat_manager = ChatManager()
    return _chat_manager


def start_enhancement_chat(prompt: str, is_negative: bool = False) -> List[Dict]:
    """Start a new chat conversation for prompt enhancement.
    
    Args:
        prompt: The prompt to enhance
        is_negative: Whether this is a negative prompt
        
    Returns:
        New chat history with enhancement request
    """
    chat_manager = get_chat_manager()
    
    # Get the enhancement template
    enhancement_template = chat_manager.get_enhancement_template()
    
    # Format the enhancement request
    if is_negative:
        enhancement_message = f"This is a negative prompt (things to avoid in the image). {enhancement_template.format(prompt)}"
    else:
        enhancement_message = enhancement_template.format(prompt)
    
    # Start new conversation with enhancement request
    new_history = chat_manager.chat_response(enhancement_message, None, [], 300)
    
    return new_history


def start_image_chat(image, starter_message: str = None) -> List[Dict]:
    """Start a new chat conversation with an image.
    
    Args:
        image: PIL Image to include in conversation
        starter_message: Optional starter message, defaults to generic prompt
        
    Returns:
        New chat history with image and starter message
    """
    chat_manager = get_chat_manager()
    
    if starter_message is None:
        starter_message = "Here's an image I just generated. What would you like to know about it?"
    
    # Start new conversation with image and starter message
    new_history = chat_manager.chat_response(starter_message, image, [], 300)
    
    return new_history