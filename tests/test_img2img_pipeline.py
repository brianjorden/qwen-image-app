"""
Tests for the new img2img pipeline functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
from PIL import Image
import numpy as np

# Mock imports before importing modules that depend on them
with patch.dict('sys.modules', {
    'diffusers': Mock(),
    'transformers': Mock(), 
    'accelerate': Mock(),
    'peft': Mock(),
    'qwen_vl_utils': Mock()
}):
    from src.models import get_pipeline, get_img2img_pipe
    from src.edit import (
        preprocess_for_img2img, 
        get_optimal_img2img_strength,
        validate_img2img_parameters,
        compare_img2img_modes
    )
    from src.process import generate_image


class TestImg2ImgPipeline(unittest.TestCase):
    """Test the new img2img pipeline functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock image
        self.test_image = Image.new('RGB', (512, 512), color='red')
        
        # Mock config
        self.config_patcher = patch('src.config.get_config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.return_value.resolution_multiple = 64
        self.mock_config.return_value.default_noise_interpolation_strength = 0.5
        self.mock_config.return_value.default_img2img_strength = 0.6
        
        # Mock model manager
        self.model_manager_patcher = patch('src.models.get_model_manager')
        self.mock_model_manager = self.model_manager_patcher.start()
        
        # Mock pipeline components
        mock_manager = Mock()
        mock_manager.build_img2img_pipeline.return_value = Mock()
        mock_manager.build_pipeline.return_value = Mock()
        self.mock_model_manager.return_value = mock_manager

    def tearDown(self):
        """Clean up after tests."""
        self.config_patcher.stop()
        self.model_manager_patcher.stop()

    def test_get_img2img_pipeline(self):
        """Test that img2img pipeline can be retrieved."""
        with patch('src.models.QwenImageImg2ImgPipeline') as mock_pipeline:
            pipeline = get_img2img_pipe()
            self.assertIsNotNone(pipeline)
            self.mock_model_manager.return_value.build_img2img_pipeline.assert_called_once()

    def test_get_pipeline_mode_selection(self):
        """Test pipeline selection based on mode."""
        with patch('src.models.QwenImageImg2ImgPipeline') as mock_img2img:
            with patch('src.models.QwenImagePipeline') as mock_txt2img:
                # Test img2img mode
                pipeline = get_pipeline("img2img")
                self.mock_model_manager.return_value.build_img2img_pipeline.assert_called()
                
                # Test txt2img mode
                pipeline = get_pipeline("txt2img")
                self.mock_model_manager.return_value.build_pipeline.assert_called()

    def test_optimal_img2img_strength_presets(self):
        """Test that optimal strength presets work correctly."""
        self.assertEqual(get_optimal_img2img_strength("minimal"), 0.2)
        self.assertEqual(get_optimal_img2img_strength("low"), 0.4)
        self.assertEqual(get_optimal_img2img_strength("medium"), 0.6)
        self.assertEqual(get_optimal_img2img_strength("high"), 0.8)
        self.assertEqual(get_optimal_img2img_strength("maximum"), 0.95)
        self.assertEqual(get_optimal_img2img_strength("unknown"), 0.6)  # Default

    def test_validate_img2img_parameters(self):
        """Test img2img parameter validation."""
        # Valid parameters
        is_valid, error = validate_img2img_parameters(self.test_image, 0.6, 512, 512)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
        
        # Invalid strength
        is_valid, error = validate_img2img_parameters(self.test_image, 1.5, 512, 512)
        self.assertFalse(is_valid)
        self.assertIn("Strength must be between 0.0 and 1.0", error)
        
        # No image
        is_valid, error = validate_img2img_parameters(None, 0.6, 512, 512)
        self.assertFalse(is_valid)
        self.assertIn("No input image provided", error)
        
        # Invalid dimensions
        is_valid, error = validate_img2img_parameters(self.test_image, 0.6, -1, 512)
        self.assertFalse(is_valid)
        self.assertIn("Invalid dimensions", error)

    def test_preprocess_for_img2img(self):
        """Test img2img preprocessing."""
        with patch('src.edit.validate_input_image') as mock_validate:
            with patch('src.edit.resize_image_for_generation') as mock_resize:
                mock_validate.return_value = self.test_image
                mock_resize.return_value = self.test_image
                
                processed_image, status = preprocess_for_img2img(self.test_image, 512, 512, 0.6)
                
                self.assertEqual(processed_image, self.test_image)
                self.assertIn("img2img generation", status)
                mock_validate.assert_called_once_with(self.test_image)
                mock_resize.assert_called_once()

    def test_compare_img2img_modes(self):
        """Test the comparison function between img2img modes."""
        comparison = compare_img2img_modes(self.test_image, 0.6)
        
        self.assertIn("Noise Interpolation", comparison)
        self.assertIn("True Img2img", comparison)
        self.assertIn("strength=0.6", comparison)
        self.assertIn("Creative interpretation", comparison)
        self.assertIn("diffusion img2img process", comparison)

    @patch('src.process.get_pipeline')
    @patch('src.process.get_session_manager')
    @patch('src.process.get_model_manager')
    def test_generate_image_with_true_img2img(self, mock_manager, mock_session, mock_pipeline):
        """Test image generation with true img2img mode."""
        # Setup mocks
        mock_pipe = Mock()
        mock_pipe.device = 'cuda:0'
        mock_pipe_result = Mock()
        mock_pipe_result.images = [self.test_image]
        mock_pipe.return_value = mock_pipe_result
        mock_pipeline.return_value = mock_pipe
        
        mock_session_mgr = Mock()
        mock_session_mgr.current_session = "test_session"
        mock_session_mgr.get_default_session.return_value = "test_session"
        mock_session_mgr.set_session.return_value = Mock()
        mock_session.return_value = mock_session_mgr
        
        mock_mgr = Mock()
        mock_mgr.model_info = {}
        mock_manager.return_value = mock_mgr
        
        with patch('src.edit.preprocess_for_img2img') as mock_preprocess:
            with patch('src.metadata.save_image_with_metadata') as mock_save:
                with patch('src.config.get_config') as mock_config:
                    # Setup config
                    config = Mock()
                    config.enable_metadata_embed = True
                    config.output_format = 'png'
                    config.default_cfg = 4.0
                    config.default_steps = 20
                    config.default_height = 512
                    config.default_width = 512
                    config.default_seed = 42
                    mock_config.return_value = config
                    
                    mock_preprocess.return_value = (self.test_image, "Test status")
                    mock_save.return_value = "/test/path.png"
                    
                    # Test true img2img generation
                    result_image, result_path = generate_image(
                        prompt="test prompt",
                        input_image=self.test_image,
                        img2img_mode="true_img2img",
                        img2img_strength=0.7
                    )
                    
                    # Verify the pipeline was called with correct parameters
                    mock_pipeline.assert_called_with("img2img")
                    mock_preprocess.assert_called_once()
                    
                    # Verify the result
                    self.assertIsNotNone(result_image)
                    self.assertIsNotNone(result_path)

    @patch('src.process.get_pipeline')
    @patch('src.process.get_session_manager')
    @patch('src.process.get_model_manager')
    def test_generate_image_with_noise_interpolation(self, mock_manager, mock_session, mock_pipeline):
        """Test image generation with noise interpolation mode."""
        # Setup mocks
        mock_pipe = Mock()
        mock_pipe.device = 'cuda:0'
        mock_pipe_result = Mock()
        mock_pipe_result.images = [self.test_image]
        mock_pipe.return_value = mock_pipe_result
        mock_pipeline.return_value = mock_pipe
        
        mock_session_mgr = Mock()
        mock_session_mgr.current_session = "test_session"
        mock_session_mgr.get_default_session.return_value = "test_session"
        mock_session_mgr.set_session.return_value = Mock()
        mock_session.return_value = mock_session_mgr
        
        mock_mgr = Mock()
        mock_mgr.model_info = {}
        mock_manager.return_value = mock_mgr
        
        with patch('src.edit.preprocess_for_noise_interpolation') as mock_preprocess:
            with patch('src.edit.create_noise_interpolation_latents') as mock_create_latents:
                with patch('src.metadata.save_image_with_metadata') as mock_save:
                    with patch('src.config.get_config') as mock_config:
                        # Setup config
                        config = Mock()
                        config.enable_metadata_embed = True
                        config.output_format = 'png'
                        config.default_cfg = 4.0
                        config.default_steps = 20
                        config.default_height = 512
                        config.default_width = 512
                        config.default_seed = 42
                        config.default_noise_interpolation_strength = 0.5
                        config.default_img2img_strength = 0.6
                        mock_config.return_value = config
                        
                        mock_preprocess.return_value = (self.test_image, "Test status")
                        mock_create_latents.return_value = torch.randn(1, 16, 32, 32)
                        mock_save.return_value = "/test/path.png"
                        
                        # Test noise interpolation generation
                        result_image, result_path = generate_image(
                            prompt="test prompt",
                            input_image=self.test_image,
                            img2img_mode="noise_interpolation",
                            noise_interpolation_strength=0.5
                        )
                        
                        # Verify the pipeline was called with txt2img mode
                        mock_pipeline.assert_called_with("txt2img")
                        mock_preprocess.assert_called_once()
                        mock_create_latents.assert_called_once()
                        
                        # Verify the result
                        self.assertIsNotNone(result_image)
                        self.assertIsNotNone(result_path)

    def test_metadata_includes_img2img_mode(self):
        """Test that metadata includes img2img mode information."""
        with patch('src.process.get_pipeline') as mock_pipeline:
            with patch('src.process.get_session_manager') as mock_session:
                with patch('src.process.get_model_manager') as mock_manager:
                    with patch('src.edit.preprocess_for_img2img') as mock_preprocess:
                        with patch('src.metadata.save_image_with_metadata') as mock_save:
                            with patch('src.config.get_config') as mock_config:
                                # Setup mocks
                                config = Mock()
                                config.enable_metadata_embed = True
                                config.output_format = 'png'
                                config.default_cfg = 4.0
                                config.default_steps = 20
                                config.default_height = 512
                                config.default_width = 512
                                config.default_seed = 42
                                mock_config.return_value = config
                                
                                mock_pipe = Mock()
                                mock_pipe.device = 'cuda:0'
                                mock_pipe_result = Mock()
                                mock_pipe_result.images = [self.test_image]
                                mock_pipe.return_value = mock_pipe_result
                                mock_pipeline.return_value = mock_pipe
                                
                                mock_session_mgr = Mock()
                                mock_session_mgr.current_session = "test_session"
                                mock_session_mgr.get_default_session.return_value = "test_session"
                                mock_session_mgr.set_session.return_value = Mock()
                                mock_session.return_value = mock_session_mgr
                                
                                mock_mgr = Mock()
                                mock_mgr.model_info = {}
                                mock_manager.return_value = mock_mgr
                                
                                mock_preprocess.return_value = (self.test_image, "Test status")
                                mock_save.return_value = "/test/path.png"
                                
                                # Generate with true img2img
                                generate_image(
                                    prompt="test prompt",
                                    input_image=self.test_image,
                                    img2img_mode="true_img2img",
                                    img2img_strength=0.7
                                )
                                
                                # Check that save_image_with_metadata was called
                                mock_save.assert_called_once()
                                
                                # Check the metadata passed to save function
                                call_args = mock_save.call_args
                                metadata = call_args[0][2]  # Third argument is metadata
                                
                                self.assertIn('img2img_mode', metadata)
                                self.assertEqual(metadata['img2img_mode'], 'true_img2img')
                                self.assertIn('is_true_img2img', metadata)
                                self.assertTrue(metadata['is_true_img2img'])
                                self.assertIn('img2img_strength', metadata)
                                self.assertEqual(metadata['img2img_strength'], 0.7)


if __name__ == '__main__':
    unittest.main()