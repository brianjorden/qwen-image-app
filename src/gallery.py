"""
Gallery and session directory management.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from PIL import Image

from .config import get_config
from .metadata import extract_metadata


class SessionManager:
    """Manages session directories for organized output."""
    
    def __init__(self):
        self.config = get_config()
        self.output_dir = Path(self.config.output_directory)
        self.current_session = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_sessions(self) -> List[str]:
        """Get list of existing session directories.
        
        Returns:
            List of session directory names
        """
        sessions = []
        
        for item in self.output_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                sessions.append(item.name)
        
        # Sort with most recent date-like sessions first
        def sort_key(s):
            # Try to parse as date
            try:
                datetime.strptime(s[:10], '%Y-%m-%d')
                return (0, s)  # Date sessions first
            except:
                return (1, s)  # Then alphabetical
        
        sessions.sort(key=sort_key)
        return sessions
    
    def create_session(self, name: str) -> Path:
        """Create a new session directory.
        
        Args:
            name: Session name (will be sanitized)
            
        Returns:
            Path to created session directory
        """
        # Sanitize name
        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)
        
        session_path = self.output_dir / safe_name
        session_path.mkdir(parents=True, exist_ok=True)
        
        self.current_session = safe_name
        return session_path
    
    def set_session(self, name: str) -> Path:
        """Set current session, creating if necessary.
        
        Args:
            name: Session name
            
        Returns:
            Path to session directory
        """
        session_path = self.output_dir / name
        
        if not session_path.exists():
            session_path = self.create_session(name)
        
        self.current_session = name
        return session_path
    
    def get_session_path(self, session: Optional[str] = None) -> Path:
        """Get path for a session.
        
        Args:
            session: Session name (uses current if not provided)
            
        Returns:
            Path to session directory
        """
        if session is None:
            session = self.current_session or datetime.now().strftime('%Y-%m-%d')
        
        return self.output_dir / session
    
    def get_default_session(self) -> str:
        """Get default session name (current date).
        
        Returns:
            Default session name
        """
        default_name = datetime.now().strftime('%Y-%m-%d')
        # Ensure the default session directory exists
        self.create_session(default_name)
        return default_name


class GalleryManager:
    """Manages image gallery for sessions."""
    
    def __init__(self):
        self.config = get_config()
        self.session_manager = SessionManager()
        self._cache = {}  # Cache loaded images
    
    def get_images(self, session: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all images in a session.
        
        Args:
            session: Session name (uses current if not provided)
            
        Returns:
            List of (image_path, metadata) tuples
        """
        session_path = self.session_manager.get_session_path(session)
        
        if not session_path.exists():
            return []
        
        images = []
        
        # Get all image files
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img_path in session_path.glob(ext):
                # Skip thumbnails or temp files
                if img_path.name.startswith('.') or 'thumb' in img_path.name:
                    continue
                
                # Extract metadata
                metadata = extract_metadata(str(img_path)) or {}
                
                # Add file info to metadata
                metadata['filename'] = img_path.name
                metadata['file_size'] = img_path.stat().st_size
                metadata['modified'] = datetime.fromtimestamp(
                    img_path.stat().st_mtime
                ).isoformat()
                
                images.append((str(img_path), metadata))
        
        # Sort by modification time (newest first)
        images.sort(key=lambda x: x[1].get('modified', ''), reverse=True)
        
        return images
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load an image from path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image or None if failed
        """
        try:
            # Check cache
            if image_path in self._cache:
                return self._cache[image_path]
            
            # Load image
            image = Image.open(image_path)
            
            # Cache it (limit cache size)
            if len(self._cache) > 100:
                # Remove oldest cached items
                self._cache = dict(list(self._cache.items())[-50:])
            
            self._cache[image_path] = image
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def get_gallery_images(self, session: Optional[str] = None) -> List[Image.Image]:
        """Get images for gallery display.
        
        Args:
            session: Session name
            
        Returns:
            List of PIL Images
        """
        image_data = self.get_images(session)
        images = []
        
        for path, _ in image_data:
            img = self.load_image(path)
            if img:
                images.append(img)
        
        return images
    
    def get_latest_image(self, session: Optional[str] = None) -> Optional[Tuple[Image.Image, Dict[str, Any]]]:
        """Get the most recent image from a session.
        
        Args:
            session: Session name
            
        Returns:
            Tuple of (image, metadata) or None
        """
        image_data = self.get_images(session)
        
        if not image_data:
            return None
        
        path, metadata = image_data[0]  # Already sorted newest first
        image = self.load_image(path)
        
        if image:
            return image, metadata
        return None
    
    def delete_image(self, image_path: str) -> bool:
        """Delete an image file.
        
        Args:
            image_path: Path to image to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            path = Path(image_path)
            
            if path.exists():
                # Remove from cache
                if image_path in self._cache:
                    del self._cache[image_path]
                
                # Delete file
                path.unlink()
                
                # Also delete metadata sidecar if it exists
                json_path = path.with_suffix('.json')
                if json_path.exists():
                    json_path.unlink()
                
                return True
                
        except Exception as e:
            print(f"Error deleting image {image_path}: {e}")
        
        return False
    
    def clear_cache(self):
        """Clear the image cache."""
        self._cache.clear()


# Global managers
_session_manager: Optional[SessionManager] = None
_gallery_manager: Optional[GalleryManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_gallery_manager() -> GalleryManager:
    """Get or create the global gallery manager."""
    global _gallery_manager
    if _gallery_manager is None:
        _gallery_manager = GalleryManager()
    return _gallery_manager
