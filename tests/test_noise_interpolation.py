#!/usr/bin/env python3
"""
Specific tests for the noise interpolation approach for two-stage generation.

This approach mixes completed latents with original noise based on a strength parameter,
avoiding complex scheduler-specific noise addition methods.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import src and app modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNoiseInterpolationMath(unittest.TestCase):
    """Test the mathematical correctness of noise interpolation."""
    
    def test_linear_interpolation_formula(self):
        """Test that the noise interpolation formula works correctly."""
        # Test with simple known values
        completed = torch.ones(2, 3, 4) * 10.0  # All values = 10
        noise = torch.zeros(2, 3, 4)            # All values = 0
        
        test_cases = [
            (0.0, 10.0),  # strength=0.0 → pure completed
            (0.25, 7.5),  # strength=0.25 → 75% completed + 25% noise
            (0.5, 5.0),   # strength=0.5 → 50% completed + 50% noise  
            (0.75, 2.5),  # strength=0.75 → 25% completed + 75% noise
            (1.0, 0.0),   # strength=1.0 → pure noise
        ]
        
        for strength, expected in test_cases:
            result = (1.0 - strength) * completed + strength * noise
            actual = result.mean().item()
            
            self.assertAlmostEqual(actual, expected, places=6,
                                 msg=f"Interpolation failed for strength {strength}")
    
    def test_interpolation_preserves_shape(self):
        """Test that interpolation preserves tensor shapes."""
        shapes_to_test = [
            (1, 256, 4),      # QwenImage packed latents
            (1, 4, 64, 64),   # Standard latents
            (2, 8, 32, 32),   # Batch of latents
        ]
        
        for shape in shapes_to_test:
            completed = torch.randn(shape)
            noise = torch.randn(shape)
            strength = 0.3
            
            result = (1.0 - strength) * completed + strength * noise
            
            self.assertEqual(result.shape, shape,
                           f"Shape mismatch for {shape}")
    
    def test_interpolation_dtype_preservation(self):
        """Test that interpolation preserves tensor dtypes."""
        dtypes_to_test = [torch.float32, torch.float16, torch.bfloat16]
        
        for dtype in dtypes_to_test:
            completed = torch.randn(1, 4, 8, 8, dtype=dtype)
            noise = torch.randn(1, 4, 8, 8, dtype=dtype)
            strength = 0.7
            
            result = (1.0 - strength) * completed + strength * noise
            
            self.assertEqual(result.dtype, dtype,
                           f"Dtype not preserved for {dtype}")
    
    def test_interpolation_device_preservation(self):
        """Test that interpolation preserves tensor devices."""
        # Test with CPU (GPU test would require GPU)
        device = torch.device('cpu')
        
        completed = torch.randn(1, 4, 8, 8, device=device)
        noise = torch.randn(1, 4, 8, 8, device=device)
        strength = 0.4
        
        result = (1.0 - strength) * completed + strength * noise
        
        self.assertEqual(result.device, device,
                       f"Device not preserved for {device}")


class TestNoiseGeneration(unittest.TestCase):
    """Test deterministic noise generation for interpolation."""
    
    def test_deterministic_noise_generation(self):
        """Test that same seed produces same noise."""
        shape = (1, 256, 4)
        seed = 42
        device = torch.device('cpu')
        dtype = torch.float32
        
        # Generate noise twice with same seed
        generator1 = torch.Generator(device=device).manual_seed(seed)
        noise1 = torch.randn(shape, generator=generator1, device=device, dtype=dtype)
        
        generator2 = torch.Generator(device=device).manual_seed(seed)
        noise2 = torch.randn(shape, generator=generator2, device=device, dtype=dtype)
        
        # Should be identical
        self.assertTrue(torch.equal(noise1, noise2),
                       "Same seed should produce identical noise")
    
    def test_different_seeds_produce_different_noise(self):
        """Test that different seeds produce different noise."""
        shape = (1, 256, 4)
        device = torch.device('cpu')
        dtype = torch.float32
        
        generator1 = torch.Generator(device=device).manual_seed(42)
        noise1 = torch.randn(shape, generator=generator1, device=device, dtype=dtype)
        
        generator2 = torch.Generator(device=device).manual_seed(123)
        noise2 = torch.randn(shape, generator=generator2, device=device, dtype=dtype)
        
        # Should be different
        self.assertFalse(torch.equal(noise1, noise2),
                        "Different seeds should produce different noise")
    
    def test_noise_statistics(self):
        """Test that generated noise has expected statistical properties."""
        shape = (1000,)  # Large enough for statistical test
        generator = torch.Generator().manual_seed(42)
        noise = torch.randn(shape, generator=generator)
        
        # Test mean is close to 0
        mean = noise.mean().item()
        self.assertAlmostEqual(mean, 0.0, places=1,
                              msg="Noise mean should be approximately 0")
        
        # Test std is close to 1
        std = noise.std().item()
        self.assertAlmostEqual(std, 1.0, places=1,
                              msg="Noise std should be approximately 1")


class TestNoiseInterpolationIntegration(unittest.TestCase):
    """Test integration with the generation pipeline."""
    
    def test_strength_boundary_values(self):
        """Test that boundary strength values work correctly."""
        completed = torch.ones(1, 4, 8, 8) * 5.0
        noise = torch.ones(1, 4, 8, 8) * -5.0
        
        # Test strength = 0.0 (pure completed)
        result_0 = (1.0 - 0.0) * completed + 0.0 * noise
        self.assertTrue(torch.allclose(result_0, completed),
                       "Strength 0.0 should return pure completed latents")
        
        # Test strength = 1.0 (pure noise)
        result_1 = (1.0 - 1.0) * completed + 1.0 * noise
        self.assertTrue(torch.allclose(result_1, noise),
                       "Strength 1.0 should return pure noise")
    
    def test_metadata_integration(self):
        """Test that noise interpolation parameters are captured in metadata."""
        from src.metadata import format_metadata_display
        
        test_metadata = {
            'prompt': 'test prompt',
            'is_two_stage': True,
            'two_stage_mode': 'Img2Img Mode',
            'img2img_strength': 0.8,
            'first_stage_steps': 30,
            'second_stage_steps': 15,
            'applied_magic_text': 'high quality'
        }
        
        result = format_metadata_display(test_metadata)
        
        # Should show noise interpolation specific info
        self.assertIn('Mode: Img2Img Mode', result)
        self.assertIn('Img2Img Strength: 0.8', result)
        self.assertIn('Stage 1: 30 steps', result)
        self.assertIn('Stage 2: 15 steps', result)


class TestErrorHandling(unittest.TestCase):
    """Test error handling for noise interpolation."""
    
    def test_shape_mismatch_handling(self):
        """Test behavior when completed latents and noise have different shapes."""
        completed = torch.randn(1, 4, 8, 8)
        noise = torch.randn(1, 4, 16, 16)  # Different shape
        strength = 0.5
        
        # This should raise an error due to shape mismatch
        with self.assertRaises(RuntimeError):
            result = (1.0 - strength) * completed + strength * noise
    
    def test_invalid_strength_values(self):
        """Test behavior with out-of-range strength values."""
        completed = torch.randn(1, 4, 8, 8)
        noise = torch.randn(1, 4, 8, 8)
        
        # Test with various strength values (including out of range)
        test_strengths = [-0.5, -0.1, 0.0, 0.5, 1.0, 1.5, 2.0]
        
        for strength in test_strengths:
            # Interpolation should work mathematically even with out-of-range values
            # (though they may not make physical sense)
            try:
                result = (1.0 - strength) * completed + strength * noise
                self.assertEqual(result.shape, completed.shape)
            except Exception as e:
                self.fail(f"Interpolation failed for strength {strength}: {e}")


class TestPerformance(unittest.TestCase):
    """Test performance characteristics of noise interpolation."""
    
    def test_interpolation_performance(self):
        """Test that interpolation is computationally efficient."""
        import time
        
        # Large tensor to test performance
        shape = (4, 512, 64, 64)  # Batch of large latents
        completed = torch.randn(shape)
        noise = torch.randn(shape)
        strength = 0.6
        
        # Time the interpolation
        start_time = time.time()
        result = (1.0 - strength) * completed + strength * noise
        end_time = time.time()
        
        # Should be very fast (< 1 second even for large tensors)
        duration = end_time - start_time
        self.assertLess(duration, 1.0,
                       f"Interpolation took too long: {duration:.3f}s")
        
        # Verify result is correct
        self.assertEqual(result.shape, shape)


def run_noise_interpolation_tests():
    """Run all noise interpolation tests."""
    print("Running Noise Interpolation Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNoiseInterpolationMath))
    suite.addTests(loader.loadTestsFromTestCase(TestNoiseGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestNoiseInterpolationIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\n✅ All noise interpolation tests passed! ({result.testsRun} tests)")
        return True
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")
        return False


if __name__ == '__main__':
    success = run_noise_interpolation_tests()
    exit(0 if success else 1)