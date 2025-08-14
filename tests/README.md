# Qwen-Image Application Tests

This directory contains tests for the Qwen-Image application, especially focused on validating the GUI/business logic separation refactoring.

## Test Files

### `test_structure.py`
Tests that the refactored project structure is correct:
- Verifies all expected files and directories exist
- Checks clean separation between UI (`app/`) and business logic (`src/`)
- Validates that entry points are updated correctly

### `test_imports.py` 
Tests that all modules import correctly after refactoring:
- Tests all `src/` module imports
- Tests all `app/` module imports  
- Tests that the main app can be created without errors

### `test_two_stage_generation.py`
Tests the two-stage generation feature and enhanced metadata:
- Tests single-stage generation (backwards compatibility)
- Tests two-stage generation logic and pipeline calls
- Tests enhanced metadata capture (template text, magic prompts, two-stage info)
- Tests metadata display formatting for new fields
- Tests UI integration (syntax and import validation)
- Tests latent preservation callback mechanism
- Tests dual-mode system integration

### `test_noise_interpolation.py`
Tests the noise interpolation approach for img2img mode:
- Tests mathematical correctness of linear interpolation formula
- Tests deterministic noise generation with seeds
- Tests tensor shape, dtype, and device preservation
- Tests boundary conditions and error handling
- Tests performance characteristics
- Tests integration with metadata system

### `test_image_upload.py`
Tests the image upload and img2img functionality:
- Tests image validation and preprocessing from various sources
- Tests image resizing and format conversion
- Tests parameter validation for img2img generation
- Tests strength presets and optimal values
- Tests complete preprocessing pipeline
- Tests UI integration for image upload controls
- Tests mocked image encoding functionality

### `run_tests.py`
Main test runner that executes all tests and provides a summary.

## Running Tests

### Run All Tests
```bash
python tests/run_tests.py
```

### Run Individual Tests
```bash
python tests/test_structure.py
python tests/test_imports.py
python tests/test_two_stage_generation.py
python tests/test_noise_interpolation.py
```

### Quick Validation
After making changes to the refactored structure, run:
```bash
python tests/run_tests.py
```

This will validate that:
1. The directory structure is correct
2. All imports work properly
3. The app can be created successfully
4. UI and business logic are properly separated

## Adding New Tests

When adding new functionality or making changes:

1. Create new test files following the `test_*.py` naming convention
2. Add them to the `run_tests.py` file
3. Focus on testing the separation of concerns between `app/` and `src/`