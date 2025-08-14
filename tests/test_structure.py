"""
Test that the refactored project structure is correct.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_directory_structure():
    """Test that all expected directories and files exist."""
    print("Testing directory structure...")
    
    expected_files = [
        # App directory structure
        "app/__init__.py",
        "app/app.py",
        "app/shared.py",
        "app/generate.py",
        "app/gallery.py",
        "app/chat.py",
        "app/analysis.py",
        "app/models.py",
        "app/config.py",
        
        # Src directory (should still exist)
        "src/__init__.py",
        "src/config.py",
        "src/models.py",
        "src/process.py",
        "src/metadata.py",
        "src/gallery.py",
        "src/prompt.py",
        "src/analysis.py",
        "src/chat.py",  # Modified to be business logic only
        "src/step.py",
        
        # Entry points
        "go.sh",
    ]
    
    all_exist = True
    
    for file_path in expected_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist


def test_clean_separation():
    """Test that UI and business logic are properly separated."""
    print("\nTesting clean separation of concerns...")
    
    try:
        # Check that app modules don't import business logic internals directly
        # (they should go through src interfaces)
        
        # Check that src/chat.py no longer has UI components
        chat_content = (project_root / "src/chat.py").read_text()
        
        has_gradio_ui = (
            "gr.Column" in chat_content or
            "gr.Textbox" in chat_content or
            "gr.Button" in chat_content or
            "create_chat_interface" in chat_content
        )
        
        if has_gradio_ui:
            print("❌ src/chat.py still contains UI components")
            return False
        else:
            print("✅ src/chat.py contains only business logic")
        
        # Check that app/app.py imports from app modules
        app_content = (project_root / "app/app.py").read_text()
        
        has_proper_imports = (
            "from .generate import create_generation_tab" in app_content and
            "from .gallery import create_gallery_tab" in app_content and
            "from .chat import create_chat_tab" in app_content
        )
        
        if not has_proper_imports:
            print("❌ app/app.py doesn't have proper modular imports")
            return False
        else:
            print("✅ app/app.py has proper modular structure")
        
        print("✅ Clean separation maintained")
        return True
        
    except Exception as e:
        print(f"❌ Separation test failed: {e}")
        return False


def test_entry_points():
    """Test that entry points are updated correctly."""
    print("\nTesting entry points...")
    
    try:
        # Check go.sh
        go_content = (project_root / "go.sh").read_text()
        if "python -m app.app" in go_content:
            print("✅ go.sh updated to use app.app")
        else:
            print("❌ go.sh not updated correctly")
            return False
        
        # app.sh was removed, so we don't need to check it
        
        return True
        
    except Exception as e:
        print(f"❌ Entry point test failed: {e}")
        return False


def main():
    """Run all structure tests."""
    print("=" * 50)
    print("QWEN-IMAGE REFACTORING STRUCTURE TESTS")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_directory_structure()
    all_passed &= test_clean_separation()
    all_passed &= test_entry_points()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL STRUCTURE TESTS PASSED!")
    else:
        print("❌ SOME STRUCTURE TESTS FAILED")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())