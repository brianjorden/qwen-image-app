"""
Tests for inpainting pipeline functionality.
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import torch
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock diffusers submodules more specifically
mock_inpaint_pipeline = Mock()
mock_diffusers = Mock()
mock_diffusers.pipelines = Mock()
mock_diffusers.pipelines.qwenimage = Mock()
mock_diffusers.pipelines.qwenimage.QwenImageInpaintPipeline = mock_inpaint_pipeline

# Mock external dependencies before importing our modules
with patch.dict('sys.modules', {
    'transformers': Mock(),
    'diffusers': mock_diffusers,
    'diffusers.pipelines': mock_diffusers.pipelines,
    'diffusers.pipelines.qwenimage': mock_diffusers.pipelines.qwenimage,
    'peft': Mock(),
    'accelerate': Mock(),
    'qwen_vl_utils': Mock()
}):
    from src.models import get_pipeline, get_inpaint_pipe
    from src.inpaint import (
        validate_mask_image, 
        validate_input_image,
        preprocess_for_inpainting,
        get_optimal_inpaint_strength,
        validate_inpaint_parameters,
        validate_mask_coverage,
        create_mask_from_bbox
    )


class TestInpaintPipeline(unittest.TestCase):
    """Test inpainting pipeline loading and functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock model manager
        self.model_manager_patcher = patch('src.models.get_model_manager')
        self.mock_model_manager = self.model_manager_patcher.start()
        
        # Mock pipeline components
        mock_manager = Mock()
        mock_manager.build_inpaint_pipeline.return_value = Mock()
        mock_manager.build_img2img_pipeline.return_value = Mock()
        mock_manager.build_pipeline.return_value = Mock()
        # Mock the load methods to avoid actual model loading
        mock_manager.load_text_encoder = Mock()
        mock_manager.load_tokenizer = Mock()
        mock_manager.load_transformer = Mock()
        mock_manager.load_vae = Mock()
        mock_manager.load_scheduler = Mock()
        # Mock component status
        mock_manager.text_encoder = True
        mock_manager.tokenizer = True
        mock_manager.transformer = True
        mock_manager.vae = True
        mock_manager.scheduler = True
        mock_manager.pipe_inpaint = None
        mock_manager.pipe_img2img = None
        mock_manager.pipe = None
        self.mock_model_manager.return_value = mock_manager
        
        # Mock config for inpaint module
        self.config_patcher = patch('src.inpaint.get_config')
        self.mock_config = self.config_patcher.start()
        mock_config_obj = Mock()
        mock_config_obj.resolution_multiple = 16
        self.mock_config.return_value = mock_config_obj
        
        # Mock config for models module
        self.models_config_patcher = patch('src.models.get_config')
        self.mock_models_config = self.models_config_patcher.start()
        self.mock_models_config.return_value = mock_config_obj

    def tearDown(self):
        """Clean up test fixtures."""
        self.model_manager_patcher.stop()
        self.config_patcher.stop()
        self.models_config_patcher.stop()

    @patch('src.models.ModelManager.build_inpaint_pipeline')
    def test_get_inpaint_pipeline(self, mock_build):
        """Test that inpainting pipeline can be retrieved."""
        mock_build.return_value = Mock()
        pipeline = get_inpaint_pipe()
        self.assertIsNotNone(pipeline)
        mock_build.assert_called_once()

    @patch('src.models.ModelManager.build_inpaint_pipeline')
    @patch('src.models.ModelManager.build_img2img_pipeline')
    @patch('src.models.ModelManager.build_pipeline')
    def test_get_pipeline_mode_selection(self, mock_txt2img, mock_img2img, mock_inpaint):
        """Test pipeline selection based on mode."""
        mock_inpaint.return_value = Mock()
        mock_img2img.return_value = Mock()
        mock_txt2img.return_value = Mock()
        
        # Test inpainting mode
        pipeline = get_pipeline("inpaint")
        mock_inpaint.assert_called()
        
        # Test img2img mode  
        pipeline = get_pipeline("img2img")
        mock_img2img.assert_called()
        
        # Test txt2img mode
        pipeline = get_pipeline("txt2img")
        mock_txt2img.assert_called()

    def test_validate_mask_image_pil(self):
        """Test mask image validation with PIL Image."""
        # Create a test mask image (grayscale)
        mask = Image.new('L', (512, 512), 128)
        
        result = validate_mask_image(mask)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, 'L')

    def test_validate_mask_image_rgb_conversion(self):
        """Test mask image validation with RGB to grayscale conversion."""
        # Create a test mask image (RGB)
        mask = Image.new('RGB', (512, 512), (128, 128, 128))
        
        result = validate_mask_image(mask)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, 'L')

    def test_validate_input_image_pil(self):
        """Test input image validation with PIL Image."""
        # Create a test input image
        image = Image.new('RGB', (512, 512), (255, 0, 0))
        
        result = validate_input_image(image)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, 'RGB')

    def test_get_optimal_inpaint_strength(self):
        """Test inpainting strength presets."""
        self.assertEqual(get_optimal_inpaint_strength("minimal"), 0.2)
        self.assertEqual(get_optimal_inpaint_strength("low"), 0.4)
        self.assertEqual(get_optimal_inpaint_strength("medium"), 0.6)
        self.assertEqual(get_optimal_inpaint_strength("high"), 0.8)
        self.assertEqual(get_optimal_inpaint_strength("maximum"), 0.95)
        self.assertEqual(get_optimal_inpaint_strength("unknown"), 0.6)  # Default

    def test_validate_inpaint_parameters(self):
        """Test inpainting parameter validation."""
        image = Image.new('RGB', (512, 512))
        mask = Image.new('L', (512, 512))
        
        # Valid parameters
        is_valid, error = validate_inpaint_parameters(image, mask, 0.6, 512, 512)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
        
        # Missing image
        is_valid, error = validate_inpaint_parameters(None, mask, 0.6, 512, 512)
        self.assertFalse(is_valid)
        self.assertIn("No input image", error)
        
        # Missing mask
        is_valid, error = validate_inpaint_parameters(image, None, 0.6, 512, 512)
        self.assertFalse(is_valid)
        self.assertIn("No mask image", error)
        
        # Invalid strength
        is_valid, error = validate_inpaint_parameters(image, mask, 1.5, 512, 512)
        self.assertFalse(is_valid)
        self.assertIn("Strength must be between", error)

    @patch('src.process.validate_dimensions')
    def test_preprocess_for_inpainting(self, mock_validate_dimensions):
        """Test inpainting preprocessing."""
        mock_validate_dimensions.return_value = (512, 512)
        
        # Create test images
        image = Image.new('RGB', (512, 512))
        mask = Image.new('L', (512, 512))
        
        processed_image, processed_mask, status = preprocess_for_inpainting(
            image, mask, 512, 512, 0.6
        )
        
        self.assertIsInstance(processed_image, Image.Image)
        self.assertIsInstance(processed_mask, Image.Image)
        self.assertIn("ready for inpainting", status)

    def test_validate_mask_coverage(self):
        """Test mask coverage validation."""
        # Create mask with 50% white coverage
        mask_array = np.zeros((100, 100), dtype=np.uint8)
        mask_array[:50, :] = 255  # Top half white
        mask = Image.fromarray(mask_array, mode='L')
        
        is_valid, message, coverage = validate_mask_coverage(mask)
        self.assertTrue(is_valid)
        self.assertAlmostEqual(coverage, 0.5, places=2)
        self.assertIn("50", message)

    def test_create_mask_from_bbox(self):
        """Test creating mask from bounding box."""
        mask = create_mask_from_bbox(100, 100, 25, 25, 50, 50)
        
        self.assertIsInstance(mask, Image.Image)
        self.assertEqual(mask.mode, 'L')
        self.assertEqual(mask.size, (100, 100))
        
        # Check that the center area is white (255) and corners are black (0)
        mask_array = np.array(mask)
        self.assertEqual(mask_array[50, 50], 255)  # Center should be white
        self.assertEqual(mask_array[0, 0], 0)      # Corner should be black


if __name__ == '__main__':
    unittest.main()