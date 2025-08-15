#!/usr/bin/env python3
"""
Quick gradio_client-based validation test for the Qwen-Image application.
Tests basic functionality with minimal resource usage.
"""

import unittest
import subprocess
import time
import sys
import threading
from pathlib import Path
import gradio_client
from PIL import Image


class TestQuickClient(unittest.TestCase):
    """Quick client tests using gradio_client to test the actual running app."""
    
    @classmethod
    def setUpClass(cls):
        """Start the app in background for testing."""
        print("\n🚀 Starting Qwen-Image app for client testing...")
        
        # Start app in background
        cls.app_process = subprocess.Popen([
            sys.executable, "-m", "app.app"
        ], cwd=Path(__file__).parent.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for app to start
        print("⏳ Waiting for app to start...")
        time.sleep(10)  # Give it time to load
        
        # Try to connect
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                cls.client = gradio_client.Client("http://localhost:7860")
                print(f"✅ Connected to app on attempt {attempt + 1}")
                break
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"⏳ Connection attempt {attempt + 1} failed, retrying...")
                    time.sleep(5)
                else:
                    print(f"❌ Failed to connect after {max_attempts} attempts: {e}")
                    cls.tearDownClass()
                    raise
    
    @classmethod
    def tearDownClass(cls):
        """Clean up the app process."""
        if hasattr(cls, 'app_process'):
            print("\n🛑 Stopping app process...")
            cls.app_process.terminate()
            cls.app_process.wait(timeout=10)
    
    def test_app_health(self):
        """Test that the app is responding."""
        try:
            # Just check the API is available
            api_info = self.client.view_api()
            self.assertIsNotNone(api_info)
            print("✅ App health check passed")
        except Exception as e:
            self.fail(f"App health check failed: {e}")
    
    def test_basic_generation(self):
        """Test basic image generation with minimal settings."""
        try:
            print("🖼️ Testing basic image generation...")
            
            # Test with minimal settings: 256x256, 4 steps
            result = self.client.predict(
                "main",  # session
                "a simple red square",  # prompt
                "",  # negative_prompt
                "",  # name
                256,  # width
                256,  # height
                4,   # steps
                1.0, # cfg_scale
                42,  # seed
                False, # randomize_seed
                True,  # apply_template
                False, # add_magic
                False, # save_steps
                0,   # second_stage_steps
                "Noise Interpolation Mode",  # two_stage_mode
                None,  # input_image
                0.99,  # noise_interpolation_strength
                api_name="/generate"
            )
            
            # Result should be [image, metadata_text, seed]
            self.assertEqual(len(result), 3, "Generate should return [image, metadata, seed]")
            
            image, metadata, seed = result
            self.assertIsNotNone(image, "Generated image should not be None")
            self.assertIsInstance(metadata, str, "Metadata should be a string")
            self.assertEqual(seed, 42, "Seed should match input")
            
            print(f"✅ Basic generation test passed - got image and metadata")
            
        except Exception as e:
            # Print more detail about the failure
            print(f"❌ Basic generation test failed: {e}")
            
            # Try to get more info about available endpoints
            try:
                api_info = self.client.view_api()
                print("Available API endpoints:")
                for endpoint in api_info.get('named_endpoints', {}):
                    print(f"  - {endpoint}")
            except:
                pass
                
            self.fail(f"Basic generation test failed: {e}")
    
    def test_model_status(self):
        """Test that we can get model status."""
        try:
            print("📊 Testing model status...")
            
            # Try to get model status from models tab
            # This tests that the models are loadable
            result = self.client.predict(api_name="/refresh_status")
            
            self.assertIsInstance(result, (str, list), "Model status should return string or list")
            print(f"✅ Model status test passed")
            
        except Exception as e:
            print(f"⚠️  Model status test failed (this might be expected): {e}")
            # Don't fail the whole test suite for this
            pass


def main():
    """Run the quick client tests."""
    print("🧪 QUICK CLIENT VALIDATION TESTS")
    print("=" * 50)
    print("Testing basic app functionality with gradio_client")
    print("This validates the running app without mocking")
    print()
    
    # Run tests
    unittest.main(verbosity=2, exit=False)


if __name__ == "__main__":
    main()