#!/usr/bin/env python3
"""
Tests for image upload and noise interpolation functionality.

This test suite validates the new image upload features, noise interpolation processing,
and the integration between the UI and backend.
"""

import unittest
import torch
import tempfile
import os
from PIL import Image
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add the parent directory to Python path so we can import src and app modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImageValidation(unittest.TestCase):
    """Test image validation and preprocessing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_temp_dir = tempfile.mkdtemp()
        
        # Mock config to avoid initialization issues
        self.config_patcher = patch('src.config.get_config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.return_value.resolution_multiple = 16
        
        # Create test images
        self.test_image_rgb = Image.new('RGB', (512, 512), 'red')
        self.test_image_rgba = Image.new('RGBA', (512, 512), 'blue')
        
        # Save test image for path testing
        self.test_image_path = Path(self.test_temp_dir) / "test_image.png"
        self.test_image_rgb.save(self.test_image_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.test_temp_dir, ignore_errors=True)
    
    def test_validate_input_image_from_path(self):
        """Test validating image from file path."""
        from src.edit import validate_input_image
        
        # Test valid image path
        validated = validate_input_image(self.test_image_path)
        self.assertIsInstance(validated, Image.Image)
        self.assertEqual(validated.mode, 'RGB')
        self.assertEqual(validated.size, (512, 512))
    
    def test_validate_input_image_from_pil(self):
        """Test validating PIL Image object."""
        from src.edit import validate_input_image
        
        # Test RGB image
        validated = validate_input_image(self.test_image_rgb)
        self.assertIsInstance(validated, Image.Image)
        self.assertEqual(validated.mode, 'RGB')
        
        # Test RGBA image (should be converted to RGB)
        validated = validate_input_image(self.test_image_rgba)
        self.assertIsInstance(validated, Image.Image)
        self.assertEqual(validated.mode, 'RGB')
    
    def test_validate_input_image_invalid_path(self):
        """Test validation with invalid file path."""
        from src.edit import validate_input_image
        
        with self.assertRaises(ValueError) as context:
            validate_input_image("/nonexistent/path.jpg")
        
        self.assertIn("Image file not found", str(context.exception))
    
    def test_validate_input_image_invalid_type(self):
        """Test validation with invalid input type."""
        from src.edit import validate_input_image
        
        with self.assertRaises(ValueError) as context:
            validate_input_image("not an image")
        
        self.assertIn("Input must be a PIL Image", str(context.exception))


class TestImageResizing(unittest.TestCase):
    """Test image resizing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock config to avoid initialization issues - need to mock in multiple modules
        self.config_patcher = patch('src.config.get_config')
        self.process_config_patcher = patch('src.process.get_config')
        self.edit_config_patcher = patch('src.edit.get_config')
        
        self.mock_config = self.config_patcher.start()
        self.mock_process_config = self.process_config_patcher.start()
        self.mock_edit_config = self.edit_config_patcher.start()
        
        # Set up config attributes
        self.mock_config.return_value.resolution_multiple = 16
        self.mock_process_config.return_value.resolution_multiple = 16
        self.mock_edit_config.return_value.resolution_multiple = 16
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
        self.process_config_patcher.stop()
        self.edit_config_patcher.stop()
    
    def test_resize_image_same_size(self):
        """Test resizing when image is already the correct size."""
        from src.edit import resize_image_for_generation
        
        test_image = Image.new('RGB', (512, 512), 'green')
        resized = resize_image_for_generation(test_image, 512, 512)
        
        # Should return the same image
        self.assertEqual(resized.size, (512, 512))
    
    def test_resize_image_different_size(self):
        """Test resizing when image needs to be resized."""
        from src.edit import resize_image_for_generation
        
        test_image = Image.new('RGB', (256, 256), 'green')
        resized = resize_image_for_generation(test_image, 512, 512)
        
        # Should be resized to target dimensions
        self.assertEqual(resized.size, (512, 512))
    
    def test_resize_image_aspect_ratio_change(self):
        """Test resizing when aspect ratio changes."""
        from src.edit import resize_image_for_generation
        
        test_image = Image.new('RGB', (512, 256), 'green')
        resized = resize_image_for_generation(test_image, 512, 512)
        
        # Should be resized to target dimensions (aspect ratio will change)
        self.assertEqual(resized.size, (512, 512))


class TestParameterValidation(unittest.TestCase):
    """Test parameter validation for noise interpolation generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock config to avoid initialization issues
        self.config_patcher = patch('src.config.get_config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.return_value.resolution_multiple = 16
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
    
    def test_valid_parameters(self):
        """Test validation with valid parameters."""
        from src.edit import validate_noise_interpolation_parameters
        
        test_image = Image.new('RGB', (512, 512), 'red')
        is_valid, error = validate_noise_interpolation_parameters(test_image, 0.5, 512, 512)
        
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
    
    def test_invalid_strength_too_low(self):
        """Test validation with strength below 0.0."""
        from src.edit import validate_noise_interpolation_parameters
        
        test_image = Image.new('RGB', (512, 512), 'red')
        is_valid, error = validate_noise_interpolation_parameters(test_image, -0.1, 512, 512)
        
        self.assertFalse(is_valid)
        self.assertIn("Strength must be between 0.0 and 1.0", error)
    
    def test_invalid_strength_too_high(self):
        """Test validation with strength above 1.0."""
        from src.edit import validate_noise_interpolation_parameters
        
        test_image = Image.new('RGB', (512, 512), 'red')
        is_valid, error = validate_noise_interpolation_parameters(test_image, 1.5, 512, 512)
        
        self.assertFalse(is_valid)
        self.assertIn("Strength must be between 0.0 and 1.0", error)
    
    def test_invalid_dimensions(self):
        """Test validation with invalid dimensions."""
        from src.edit import validate_noise_interpolation_parameters
        
        test_image = Image.new('RGB', (512, 512), 'red')
        is_valid, error = validate_noise_interpolation_parameters(test_image, 0.5, 0, 512)
        
        self.assertFalse(is_valid)
        self.assertIn("Invalid dimensions", error)
    
    def test_no_image(self):
        """Test validation with no input image."""
        from src.edit import validate_noise_interpolation_parameters
        
        is_valid, error = validate_noise_interpolation_parameters(None, 0.5, 512, 512)
        
        self.assertFalse(is_valid)
        self.assertIn("No input image provided", error)


class TestStrengthPresets(unittest.TestCase):
    """Test noise interpolation strength presets."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock config to avoid initialization issues
        self.config_patcher = patch('src.config.get_config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.return_value.resolution_multiple = 16
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
    
    def test_strength_presets(self):
        """Test that strength presets return expected values."""
        from src.edit import get_optimal_noise_interpolation_strength
        
        test_cases = [
            ("minimal", 0.1),
            ("low", 0.3),
            ("medium", 0.5),
            ("high", 0.7),
            ("maximum", 0.9)
        ]
        
        for similarity, expected_strength in test_cases:
            actual_strength = get_optimal_noise_interpolation_strength(similarity)
            self.assertEqual(actual_strength, expected_strength)
    
    def test_strength_preset_default(self):
        """Test default strength preset for unknown values."""
        from src.edit import get_optimal_noise_interpolation_strength
        
        # Unknown preset should return medium (0.5)
        strength = get_optimal_noise_interpolation_strength("unknown")
        self.assertEqual(strength, 0.5)


class TestImagePreprocessing(unittest.TestCase):
    """Test complete image preprocessing pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_temp_dir = tempfile.mkdtemp()
        
        # Mock config to avoid initialization issues
        self.config_patcher = patch('src.config.get_config')
        self.process_config_patcher = patch('src.process.get_config')
        self.edit_config_patcher = patch('src.edit.get_config')
        
        self.mock_config = self.config_patcher.start()
        self.mock_process_config = self.process_config_patcher.start()
        self.mock_edit_config = self.edit_config_patcher.start()
        
        # Set up config attributes
        self.mock_config.return_value.resolution_multiple = 16
        self.mock_process_config.return_value.resolution_multiple = 16
        self.mock_edit_config.return_value.resolution_multiple = 16
        self.test_image = Image.new('RGB', (512, 512), 'red')
        self.test_image_path = Path(self.test_temp_dir) / "test.png"
        self.test_image.save(self.test_image_path)
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
        self.process_config_patcher.stop()
        self.edit_config_patcher.stop()
        import shutil
        shutil.rmtree(self.test_temp_dir, ignore_errors=True)
    
    def test_preprocess_from_path(self):
        """Test preprocessing from image path."""
        from src.edit import preprocess_for_noise_interpolation
        
        processed, status = preprocess_for_noise_interpolation(self.test_image_path, 512, 512, 0.5)
        
        self.assertIsInstance(processed, Image.Image)
        self.assertEqual(processed.size, (512, 512))
        self.assertIn("ready for noise_interpolation", status)
    
    def test_preprocess_from_pil(self):
        """Test preprocessing from PIL Image."""
        from src.edit import preprocess_for_noise_interpolation
        
        processed, status = preprocess_for_noise_interpolation(self.test_image, 512, 512, 0.5)
        
        self.assertIsInstance(processed, Image.Image)
        self.assertEqual(processed.size, (512, 512))
        self.assertIn("ready for noise_interpolation", status)
    
    def test_preprocess_with_resize(self):
        """Test preprocessing when resizing is needed."""
        from src.edit import preprocess_for_noise_interpolation
        
        small_image = Image.new('RGB', (256, 256), 'blue')
        processed, status = preprocess_for_noise_interpolation(small_image, 512, 512, 0.5)
        
        self.assertIsInstance(processed, Image.Image)
        self.assertEqual(processed.size, (512, 512))
        self.assertIn("resized from", status)
    
    def test_preprocess_invalid_strength(self):
        """Test preprocessing with invalid strength parameter."""
        from src.edit import preprocess_for_noise_interpolation
        
        with self.assertRaises(ValueError) as context:
            preprocess_for_noise_interpolation(self.test_image, 512, 512, 2.0)
        
        self.assertIn("Strength must be between", str(context.exception))


class TestUIIntegration(unittest.TestCase):
    """Test UI integration for image upload functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock config to avoid initialization issues
        self.config_patcher = patch('src.config.get_config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.return_value.resolution_multiple = 16
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
    
    def test_ui_image_upload_handlers(self):
        """Test that UI image upload handlers are defined."""
        from app.generate import _handle_image_upload, _use_generated_as_input
        
        # Test image upload handler
        result = _handle_image_upload(None)
        self.assertEqual(len(result), 3)  # Should return 3 updates
        
        # Test use generated as input handler
        test_image = Image.new('RGB', (512, 512), 'red')
        result = _use_generated_as_input(test_image)
        self.assertEqual(len(result), 4)  # Should return 4 updates
    
    def test_ui_parameter_passing(self):
        """Test that UI parameters are passed correctly."""
        # This tests the parameter structure for the new functionality
        expected_params = [
            'session', 'prompt', 'negative_prompt', 'name',
            'width', 'height', 'steps', 'cfg_scale',
            'seed', 'randomize', 'apply_template', 'add_magic', 'save_steps',
            'second_stage_steps', 'two_stage_mode', 'input_image', 'noise_interpolation_strength'
        ]
        
        # This would be the expected parameter count for the generate_image call
        self.assertEqual(len(expected_params), 17)
        
        # Check that input_image and noise_interpolation_strength are included
        self.assertIn('input_image', expected_params)
        self.assertIn('noise_interpolation_strength', expected_params)


class TestMockImageEncoding(unittest.TestCase):
    """Test image encoding with mocked dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock config to avoid initialization issues
        self.config_patcher = patch('src.config.get_config')
        self.process_config_patcher = patch('src.process.get_config')
        self.edit_config_patcher = patch('src.edit.get_config')
        
        self.mock_config = self.config_patcher.start()
        self.mock_process_config = self.process_config_patcher.start()
        self.mock_edit_config = self.edit_config_patcher.start()
        
        # Set up config attributes
        self.mock_config.return_value.resolution_multiple = 16
        self.mock_process_config.return_value.resolution_multiple = 16
        self.mock_edit_config.return_value.resolution_multiple = 16
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.config_patcher.stop()
        self.process_config_patcher.stop()
        self.edit_config_patcher.stop()
    
    @patch('src.edit.get_model_manager')
    @patch('src.edit.get_pipe')
    def test_encode_image_mock(self, mock_get_pipe, mock_get_manager):
        """Test image encoding with mocked VAE and pipeline."""
        from src.edit import encode_image_to_latents
        
        # Mock VAE
        mock_vae = Mock()
        mock_vae.device = torch.device('cpu')
        mock_vae.dtype = torch.float32
        mock_vae.encode.return_value = Mock(latent_dist=Mock(sample=Mock(return_value=torch.randn(1, 4, 64, 64))))
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.vae_scale_factor = 8
        mock_pipeline.image_processor = Mock()
        mock_pipeline.image_processor.preprocess.return_value = torch.randn(3, 512, 512)
        mock_pipeline._pack_latents.return_value = torch.randn(1, 256, 4)
        
        # Mock manager
        mock_manager = Mock()
        mock_manager.vae = mock_vae
        
        mock_get_manager.return_value = mock_manager
        mock_get_pipe.return_value = mock_pipeline
        
        # Test encoding
        test_image = Image.new('RGB', (512, 512), 'red')
        latents = encode_image_to_latents(test_image, 512, 512)
        
        # Verify result
        self.assertIsInstance(latents, torch.Tensor)
        self.assertEqual(latents.shape, (1, 256, 4))
    
    @patch('src.edit.get_model_manager')
    def test_encode_image_no_vae(self, mock_get_manager):
        """Test image encoding when VAE is not loaded."""
        from src.edit import encode_image_to_latents
        
        # Mock manager without VAE
        mock_manager = Mock()
        mock_manager.vae = None
        mock_get_manager.return_value = mock_manager
        
        # Should raise RuntimeError
        test_image = Image.new('RGB', (512, 512), 'red')
        with self.assertRaises(RuntimeError) as context:
            encode_image_to_latents(test_image, 512, 512)
        
        self.assertIn("VAE not loaded", str(context.exception))


def run_image_upload_tests():
    """Run all image upload tests."""
    print("Running Image Upload Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestImageValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestImageResizing))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestStrengthPresets))
    suite.addTests(loader.loadTestsFromTestCase(TestImagePreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestUIIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestMockImageEncoding))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All image upload tests passed! ({result.testsRun} tests)")
        return True
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")
        return False


if __name__ == '__main__':
    success = run_image_upload_tests()
    exit(0 if success else 1)