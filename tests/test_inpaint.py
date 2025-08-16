"""
Tests for inpainting preprocessing and utilities.
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mock external dependencies before importing our modules
with patch.dict('sys.modules', {
    'transformers': Mock(),
    'diffusers': Mock(),
    'peft': Mock(),
    'accelerate': Mock(),
    'qwen_vl_utils': Mock()
}):
    from src.inpaint import (
        resize_images_for_inpainting,
        prepare_inpaint_inputs,
        validate_mask_coverage,
        create_mask_from_bbox
    )


class TestInpaintPreprocessing(unittest.TestCase):
    """Test inpainting preprocessing functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock config
        self.config_patcher = patch('src.inpaint.get_config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.return_value.resolution_multiple = 16
        
        # Mock validate_dimensions function
        self.validate_dimensions_patcher = patch('src.inpaint.validate_dimensions')
        self.mock_validate_dimensions = self.validate_dimensions_patcher.start()
        self.mock_validate_dimensions.return_value = (512, 512)

    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
        self.validate_dimensions_patcher.stop()

    def test_resize_images_for_inpainting_no_resize_needed(self):
        """Test resizing when images are already correct size."""
        image = Image.new('RGB', (512, 512))
        mask = Image.new('L', (512, 512))
        
        resized_image, resized_mask = resize_images_for_inpainting(image, mask, 512, 512)
        
        self.assertEqual(resized_image.size, (512, 512))
        self.assertEqual(resized_mask.size, (512, 512))
        self.assertEqual(resized_image.mode, 'RGB')
        self.assertEqual(resized_mask.mode, 'L')

    def test_resize_images_for_inpainting_resize_needed(self):
        """Test resizing when images need to be resized."""
        image = Image.new('RGB', (256, 256))
        mask = Image.new('L', (256, 256))
        
        resized_image, resized_mask = resize_images_for_inpainting(image, mask, 512, 512)
        
        self.assertEqual(resized_image.size, (512, 512))
        self.assertEqual(resized_mask.size, (512, 512))

    def test_validate_mask_coverage_good_coverage(self):
        """Test mask coverage validation with good coverage."""
        # Create mask with 25% white coverage
        mask_array = np.zeros((100, 100), dtype=np.uint8)
        mask_array[:50, :50] = 255  # Quarter white
        mask = Image.fromarray(mask_array, mode='L')
        
        is_valid, message, coverage = validate_mask_coverage(mask, min_coverage=0.1, max_coverage=0.8)
        
        self.assertTrue(is_valid)
        self.assertAlmostEqual(coverage, 0.25, places=2)
        self.assertIn("25", message)

    def test_validate_mask_coverage_too_small(self):
        """Test mask coverage validation with too small coverage."""
        # Create mask with very small coverage
        mask_array = np.zeros((100, 100), dtype=np.uint8)
        mask_array[0, 0] = 255  # Just one pixel
        mask = Image.fromarray(mask_array, mode='L')
        
        is_valid, message, coverage = validate_mask_coverage(mask, min_coverage=0.05)
        
        self.assertFalse(is_valid)
        self.assertIn("too small", message)

    def test_validate_mask_coverage_too_large(self):
        """Test mask coverage validation with too large coverage."""
        # Create mask with very large coverage
        mask_array = np.ones((100, 100), dtype=np.uint8) * 255  # All white
        mask = Image.fromarray(mask_array, mode='L')
        
        is_valid, message, coverage = validate_mask_coverage(mask, max_coverage=0.5)
        
        self.assertFalse(is_valid)
        self.assertIn("too large", message)

    def test_create_mask_from_bbox_basic(self):
        """Test creating mask from bounding box."""
        mask = create_mask_from_bbox(200, 200, 50, 50, 100, 100)
        
        self.assertEqual(mask.size, (200, 200))
        self.assertEqual(mask.mode, 'L')
        
        # Convert to array for testing
        mask_array = np.array(mask)
        
        # Check that the rectangle area is white
        self.assertEqual(mask_array[75, 75], 255)  # Inside rectangle
        self.assertEqual(mask_array[100, 100], 255)  # Bottom-right corner of rectangle
        
        # Check that outside area is black
        self.assertEqual(mask_array[25, 25], 0)  # Outside rectangle
        self.assertEqual(mask_array[175, 175], 0)  # Outside rectangle

    def test_create_mask_from_bbox_edge_cases(self):
        """Test creating mask from bounding box with edge cases."""
        # Test full image mask
        mask = create_mask_from_bbox(100, 100, 0, 0, 100, 100)
        mask_array = np.array(mask)
        self.assertEqual(mask_array.sum(), 100 * 100 * 255)  # All pixels should be white
        
        # Test single pixel mask
        mask = create_mask_from_bbox(100, 100, 50, 50, 1, 1)
        mask_array = np.array(mask)
        self.assertEqual(mask_array[50, 50], 255)
        self.assertEqual(mask_array[51, 51], 0)

    @patch('src.inpaint.validate_mask_coverage')
    def test_prepare_inpaint_inputs(self, mock_validate_coverage):
        """Test preparing inpaint inputs."""
        mock_validate_coverage.return_value = (True, "Mask coverage: 25%", 0.25)
        
        image = Image.new('RGB', (512, 512))
        mask = Image.new('L', (512, 512))
        
        final_image, final_mask, status = prepare_inpaint_inputs(image, mask, 0.6, 512, 512)
        
        self.assertIsInstance(final_image, Image.Image)
        self.assertIsInstance(final_mask, Image.Image)
        self.assertIn("Inpainting ready", status)
        self.assertIn("25%", status)
        self.assertIn("strength: 0.6", status)

    @patch('src.inpaint.validate_mask_coverage')
    def test_prepare_inpaint_inputs_bad_coverage(self, mock_validate_coverage):
        """Test preparing inpaint inputs with bad mask coverage."""
        mock_validate_coverage.return_value = (False, "Mask coverage too small (1%)", 0.01)
        
        image = Image.new('RGB', (512, 512))
        mask = Image.new('L', (512, 512))
        
        with patch('builtins.print') as mock_print:
            final_image, final_mask, status = prepare_inpaint_inputs(image, mask, 0.6, 512, 512)
            
            # Should still work but print a warning
            mock_print.assert_called()
            warning_call = mock_print.call_args[0][0]
            self.assertIn("Warning", warning_call)


class TestMaskGeneration(unittest.TestCase):
    """Test mask generation utilities."""

    def test_create_simple_masks(self):
        """Test creating simple geometric masks."""
        # Square mask in center
        mask = create_mask_from_bbox(100, 100, 25, 25, 50, 50)
        mask_array = np.array(mask)
        
        # Check center is white
        self.assertEqual(mask_array[50, 50], 255)
        # Check corners are black  
        self.assertEqual(mask_array[0, 0], 0)
        self.assertEqual(mask_array[99, 99], 0)

    def test_create_edge_masks(self):
        """Test creating masks at image edges."""
        # Top-left corner mask
        mask = create_mask_from_bbox(100, 100, 0, 0, 25, 25)
        mask_array = np.array(mask)
        
        self.assertEqual(mask_array[10, 10], 255)  # Inside mask
        self.assertEqual(mask_array[50, 50], 0)    # Outside mask

    def test_create_thin_masks(self):
        """Test creating thin rectangular masks."""
        # Horizontal line mask
        mask = create_mask_from_bbox(100, 100, 0, 45, 100, 10)
        mask_array = np.array(mask)
        
        self.assertEqual(mask_array[50, 50], 255)  # Inside horizontal line
        self.assertEqual(mask_array[30, 50], 0)    # Above line
        self.assertEqual(mask_array[70, 50], 0)    # Below line


if __name__ == '__main__':
    unittest.main()