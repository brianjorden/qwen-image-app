"""
Test runner for the qwen-image-app tests.
"""

import sys
import subprocess
from pathlib import Path

def run_test(test_file):
    """Run a single test file and return success status."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {test_file}")
    print('='*60)
    
    try:
        result = subprocess.run([
            sys.executable, test_file
        ], cwd=Path(__file__).parent, capture_output=False)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Failed to run {test_file}: {e}")
        return False


def main():
    """Run all tests in the tests directory."""
    test_dir = Path(__file__).parent
    test_files = [
        test_dir / "test_structure.py",
        test_dir / "test_imports.py",
        test_dir / "test_two_stage_generation.py",
        test_dir / "test_noise_interpolation.py",
        test_dir / "test_img2img_pipeline.py",
        test_dir / "test_image_upload.py",
        test_dir / "test_quick_client.py",
    ]
    
    print("🚀 STARTING qwen-image-app TEST SUITE")
    
    all_passed = True
    results = {}
    
    for test_file in test_files:
        if test_file.exists():
            success = run_test(test_file)
            results[test_file.name] = success
            all_passed &= success
        else:
            print(f"❌ Test file not found: {test_file}")
            all_passed = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED" 
        print(f"{test_name:<30} {status}")
    
    print("-"*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Application functionality verified.")
    else:
        print("❌ SOME TESTS FAILED! Check the output above.")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())