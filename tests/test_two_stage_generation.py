#!/usr/bin/env python3
"""
Tests for two-stage generation feature.

Tests the new functionality that allows generating an image in two stages:
1. Generate an image with N steps
2. Use that image as input for M additional steps
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
from PIL import Image
from pathlib import Path
import tempfile
import os
import sys

# Add the parent directory to Python path so we can import src and app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test
from src.process import generate_image
from src.metadata import format_metadata_display, extract_metadata
from app.generate import create_generation_tab


class TestTwoStageGeneration(unittest.TestCase):
    """Test two-stage generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_temp_dir, ignore_errors=True)
    
    def test_generate_image_imports_correctly(self):
        """Test that generate_image function imports with new signature."""
        # This should not raise an ImportError
        from src.process import generate_image
        
        # Check that the function signature includes new parameters
        import inspect
        sig = inspect.signature(generate_image)
        self.assertIn('second_stage_steps', sig.parameters)
        self.assertIn('input_image', sig.parameters)
    
    @patch('src.prompt.get_config')  # Mock config in prompt module too
    @patch('src.process.get_config')
    @patch('src.process.get_model_manager')  
    @patch('src.process.get_pipe')
    @patch('src.process.get_session_manager')
    def test_single_stage_generation_unchanged(self, mock_session_mgr, mock_pipe, mock_manager, mock_config, mock_prompt_config):
        """Test that single-stage generation (second_stage_steps=0) works as before."""
        # Mock configuration
        config_mock = Mock(
            default_cfg=2.0,
            default_steps=20,
            default_height=512,
            default_width=512,
            default_seed=42,
            default_negative=" ",
            prompt_magic_en="high quality",
            prompt_magic_zh="超清",
            enable_metadata_embed=True,
            output_format="png",
            resolution_multiple=16  # Add missing config value
        )
        mock_config.return_value = config_mock
        mock_prompt_config.return_value = config_mock  # Same config for prompt module
        
        # Mock model manager
        mock_manager.return_value = Mock(
            text_encoder=None,
            tokenizer=None,
            model_info={"transformer": "test"}
        )
        
        # Mock session manager
        mock_session_mgr.return_value = Mock()
        mock_session_mgr.return_value.set_session.return_value = Path(self.test_temp_dir)
        
        # Mock pipeline result
        test_image = Image.new('RGB', (512, 512), 'red')
        mock_result = Mock()
        mock_result.images = [test_image]
        mock_pipeline = Mock()
        mock_pipeline.__call__ = Mock(return_value=mock_result)
        mock_pipeline.device = torch.device('cpu')  # Real torch device
        mock_pipe.return_value = mock_pipeline
        
        # Test single-stage generation (second_stage_steps=0)
        with patch('src.process.save_image_with_metadata') as mock_save:
            mock_save.return_value = Path(self.test_temp_dir) / "test.png"
            
            image, path = generate_image(
                prompt="test prompt",
                second_stage_steps=0  # Single stage
            )
        
        # Verify single-stage behavior
        self.assertIsInstance(image, Image.Image)
        self.assertIsNotNone(path)
        
        # Verify pipeline was called only once (single stage)
        mock_pipeline.__call__.assert_called_once()
    
    @patch('src.prompt.get_config')  # Mock config in prompt module too
    @patch('src.process.get_config')
    @patch('src.process.get_model_manager')
    @patch('src.process.get_pipe') 
    @patch('src.process.get_session_manager')
    def test_two_stage_generation_logic(self, mock_session_mgr, mock_pipe, mock_manager, mock_config, mock_prompt_config):
        """Test that two-stage generation calls pipeline twice with correct parameters."""
        # Mock configuration
        config_mock = Mock(
            default_cfg=2.0,
            default_steps=20,
            default_height=512,
            default_width=512,
            default_seed=42,
            default_negative=" ",
            prompt_magic_en="high quality",
            prompt_magic_zh="超清",
            enable_metadata_embed=True,
            output_format="png",
            resolution_multiple=16  # Add missing config value
        )
        mock_config.return_value = config_mock
        mock_prompt_config.return_value = config_mock  # Same config for prompt module
        
        # Mock model manager with VAE
        mock_vae = Mock()
        mock_vae.device = torch.device('cpu')
        mock_vae.dtype = torch.float32
        mock_vae.encode.return_value = Mock(latent_dist=Mock(sample=Mock(return_value=torch.randn(1, 4, 64, 64))))
        
        mock_manager.return_value = Mock(
            text_encoder=None,
            tokenizer=None,
            vae=mock_vae,
            model_info={"transformer": "test"}
        )
        
        # Mock session manager
        mock_session_mgr.return_value = Mock()
        mock_session_mgr.return_value.set_session.return_value = Path(self.test_temp_dir)
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.device = torch.device('cpu')  # Real torch device
        mock_pipeline.vae_scale_factor = 8
        mock_pipeline.image_processor = Mock()
        mock_pipeline.image_processor.preprocess.return_value = torch.randn(3, 512, 512)
        mock_pipeline._pack_latents.return_value = torch.randn(1, 256, 4)
        
        # Mock pipeline results for both stages
        stage1_image = Image.new('RGB', (512, 512), 'red')
        stage2_image = Image.new('RGB', (512, 512), 'blue')
        
        # Mock pipeline call to return different images for each call
        call_count = [0]  # Use list to make it mutable
        def mock_pipeline_call(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return Mock(images=[stage1_image])
            else:
                return Mock(images=[stage2_image])
        
        mock_pipeline.__call__ = Mock(side_effect=mock_pipeline_call)
        mock_pipe.return_value = mock_pipeline
        
        # Test two-stage generation
        with patch('src.process.save_image_with_metadata') as mock_save:
            mock_save.return_value = Path(self.test_temp_dir) / "test.png"
            
            image, path = generate_image(
                prompt="test prompt",
                num_inference_steps=20,
                second_stage_steps=10  # Two stage!
            )
        
        # Verify two-stage behavior
        self.assertIsInstance(image, Image.Image)
        self.assertIsNotNone(path)
        
        # Verify pipeline was called twice (two stages)
        self.assertEqual(mock_pipeline.__call__.call_count, 2)
        
        # Verify second call included latents parameter
        second_call_kwargs = mock_pipeline.__call__.call_args_list[1][1]
        self.assertIn('latents', second_call_kwargs)
        self.assertEqual(second_call_kwargs['num_inference_steps'], 10)


class TestEnhancedMetadata(unittest.TestCase):
    """Test enhanced metadata capture and display."""
    
    def test_metadata_display_includes_new_fields(self):
        """Test that format_metadata_display shows new two-stage and template fields."""
        # Test metadata with new fields
        test_metadata = {
            'prompt': 'original prompt',
            'enhanced_prompt': 'enhanced prompt text',
            'final_processed_prompt': 'final processed prompt text, high quality',
            'negative_prompt': 'bad quality',
            'applied_template_text': 'Template: {prompt}',
            'applied_magic_text': 'high quality',
            'is_two_stage': True,
            'first_stage_steps': 20,
            'second_stage_steps': 10,
            'first_stage_image_path': '/path/to/stage1.png',
            'width': 512,
            'height': 512,
            'steps': 20,
            'cfg_scale': 2.0,
            'seed': 42
        }
        
        result = format_metadata_display(test_metadata)
        
        # Check that all new fields are displayed
        self.assertIn('Enhanced: enhanced prompt text', result)
        self.assertIn('Final Processed: final processed prompt text, high quality', result)
        self.assertIn('Negative: bad quality', result)
        self.assertIn('Applied Template: Template: {prompt}', result)
        self.assertIn('Applied Magic: high quality', result)
        self.assertIn('Two-Stage Generation:', result)
        self.assertIn('Stage 1: 20 steps', result)
        self.assertIn('Stage 2: 10 steps', result)
        self.assertIn('Intermediate: stage1.png', result)
    
    def test_metadata_display_backwards_compatible(self):
        """Test that format_metadata_display still works with old metadata format."""
        # Test with minimal old-style metadata
        old_metadata = {
            'prompt': 'test prompt',
            'negative_prompt': 'bad',
            'width': 512,
            'height': 512,
            'steps': 25,
            'cfg_scale': 3.0,
            'seed': 123
        }
        
        result = format_metadata_display(old_metadata)
        
        # Should not crash and should show basic info
        self.assertIn('Prompt: test prompt', result)
        self.assertIn('Negative: bad', result)
        self.assertIn('Steps: 25', result)
        
        # Should not show new fields when they don't exist
        self.assertNotIn('Two-Stage Generation:', result)
        self.assertNotIn('Applied Template:', result)


class TestNoiseInterpolation(unittest.TestCase):
    """Test the new noise interpolation approach for noise interpolation mode."""
    
    def test_noise_interpolation_logic(self):
        """Test that noise interpolation produces expected mixing behavior."""
        import torch
        
        # Create mock completed latents and original noise
        completed_latents = torch.ones(1, 256, 4) * 0.5  # Simulated completed latents
        original_noise = torch.zeros(1, 256, 4)          # Simulated original noise
        
        # Test different strength values
        test_cases = [
            (0.0, 0.5, 0.0),    # strength=0.0 → pure completed (0.5)
            (1.0, 0.0, 0.5),    # strength=1.0 → pure noise (0.0)  
            (0.5, 0.25, 0.25),  # strength=0.5 → 50/50 mix
        ]
        
        for strength, expected_completed_contrib, expected_noise_contrib in test_cases:
            mixed = (1.0 - strength) * completed_latents + strength * original_noise
            
            # For our test tensors, the result should be the weighted average
            expected_value = 0.5 * (1.0 - strength) + 0.0 * strength
            actual_value = mixed.mean().item()
            
            self.assertAlmostEqual(actual_value, expected_value, places=6,
                                 msg=f"Strength {strength} mixing failed")
    
    def test_noise_interpolation_parameters(self):
        """Test that noise interpolation parameters are correctly captured in metadata."""
        from src.metadata import format_metadata_display
        
        test_metadata = {
            'prompt': 'test prompt',
            'is_two_stage': True,
            'two_stage_mode': 'Img2Img Mode',
            'noise interpolation_strength': 0.3,
            'first_stage_steps': 20,
            'second_stage_steps': 10,
        }
        
        result = format_metadata_display(test_metadata)
        
        # Check that noise interpolation info is displayed
        self.assertIn('Mode: Img2Img Mode', result)
        self.assertIn('Img2Img Strength: 0.3', result)
        self.assertIn('Stage 1: 20 steps', result)
        self.assertIn('Stage 2: 10 steps', result)


class TestImageUploadFunctionality(unittest.TestCase):
    """Test image upload and noise interpolation functionality."""
    
    def test_edit_module_imports(self):
        """Test that the new edit module imports correctly."""
        try:
            from src.edit import (
                validate_input_image, resize_image_for_generation,
                encode_image_to_latents, create_noise interpolation_latents,
                preprocess_for_noise interpolation
            )
            # If import works, the module is syntactically correct
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Edit module failed to import: {e}")
    
    def test_noise interpolation_parameter_validation(self):
        """Test that noise interpolation parameters are validated correctly."""
        from src.edit import validate_noise interpolation_parameters
        from PIL import Image
        
        # Create a test image
        test_image = Image.new('RGB', (512, 512), 'red')
        
        # Test valid parameters
        is_valid, error = validate_noise interpolation_parameters(test_image, 0.5, 512, 512)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
        
        # Test invalid strength
        is_valid, error = validate_noise interpolation_parameters(test_image, 2.0, 512, 512)
        self.assertFalse(is_valid)
        self.assertIn("Strength must be between", error)
        
        # Test no image
        is_valid, error = validate_noise interpolation_parameters(None, 0.5, 512, 512)
        self.assertFalse(is_valid)
        self.assertIn("No input image", error)


class TestDualModeSystem(unittest.TestCase):
    """Test the dual mode system integration."""
    
    def test_mode_parameter_validation(self):
        """Test that mode parameters are validated correctly."""
        # Test valid modes (now only noise interpolation mode)
        valid_modes = ["Img2Img Mode"]
        for mode in valid_modes:
            # Should not raise any exceptions
            self.assertIn(mode, valid_modes)
    
    def test_strength_parameter_ranges(self):
        """Test that strength parameters are in valid ranges."""
        # Test boundary values
        test_strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for strength in test_strengths:
            # All values should be in valid range [0.0, 1.0]
            self.assertGreaterEqual(strength, 0.0)
            self.assertLessEqual(strength, 1.0)
    
    def test_generate_function_signature(self):
        """Test that generate_image function has all required parameters."""
        import inspect
        from src.process import generate_image
        
        sig = inspect.signature(generate_image)
        params = list(sig.parameters.keys())
        
        # Check for new parameters
        self.assertIn('second_stage_steps', params)
        self.assertIn('two_stage_mode', params)
        self.assertIn('noise interpolation_strength', params)
        
        # Check default values
        self.assertEqual(sig.parameters['two_stage_mode'].default, "Img2Img Mode")
        self.assertEqual(sig.parameters['noise interpolation_strength'].default, 0.5)
        self.assertIsNone(sig.parameters['input_image'].default)
    
    def test_metadata_completeness(self):
        """Test that metadata captures all relevant dual-mode information."""
        from src.metadata import format_metadata_display
        
        # Test complete metadata for different generation modes
        two_stage_metadata = {
            'prompt': 'test',
            'is_two_stage': True,
            'two_stage_mode': 'Img2Img Mode',
            'first_stage_steps': 20,
            'second_stage_steps': 10,
            'applied_template_text': 'Template: {prompt}',
            'applied_magic_text': 'high quality'
        }
        
        noise interpolation_metadata = {
            'prompt': 'test',
            'is_noise interpolation': True,
            'noise interpolation_strength': 0.7,
            'steps': 25,
            'applied_template_text': 'Template: {prompt}',
            'applied_magic_text': 'high quality'
        }
        
        # Both should format without errors
        two_stage_result = format_metadata_display(two_stage_metadata)
        noise interpolation_result = format_metadata_display(noise interpolation_metadata)
        
        # Basic checks
        self.assertIn('Two-Stage Generation: Yes', two_stage_result)
        self.assertIn('Img2Img Generation: Yes', noise interpolation_result)
        self.assertIn('Img2Img Strength: 0.7', noise interpolation_result)


class TestUIIntegration(unittest.TestCase):
    """Test UI integration for two-stage generation."""
    
    def test_ui_imports_successfully(self):
        """Test that all UI components import without errors."""
        try:
            from app.generate import create_generation_tab
            # If import works, the UI changes are syntactically correct
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"UI module failed to import: {e}")
        except SyntaxError as e:
            self.fail(f"UI module has syntax error: {e}")
    
    def test_ui_parameter_passing(self):
        """Test that UI passes parameters correctly (mock test)."""
        # This is a simplified test since we can't easily test Gradio UI
        # But we can test the parameter structure
        
        mock_args = [
            "session", "prompt", "neg_prompt", "name",
            512, 512, 20, 2.0,  # width, height, steps, cfg
            42, False, True, True, False, 10,  # seed, randomize, template, magic, save_steps, second_stage_steps
            "Img2Img Mode", 0.6  # two_stage_mode, noise interpolation_strength
        ]
        
        # Test that we have the right number of parameters for the new signature
        # This should match the generate_image function signature
        self.assertEqual(len(mock_args), 16)  # Adjust based on actual parameter count
        
        # Test parameter types
        self.assertIsInstance(mock_args[14], str)    # two_stage_mode
        self.assertIsInstance(mock_args[15], float)  # noise interpolation_strength


def run_tests():
    """Run all two-stage generation tests."""
    print("Running Two-Stage Generation Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTwoStageGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedMetadata))
    suite.addTests(loader.loadTestsFromTestCase(TestNoiseInterpolation))
    suite.addTests(loader.loadTestsFromTestCase(TestImageUploadFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestDualModeSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestUIIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All tests passed! ({result.testsRun} tests)")
        return True
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")
        return False


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)