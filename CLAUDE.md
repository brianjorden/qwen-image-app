# CLAUDE.md

**⚠️ EXPERIMENTAL CODEBASE ⚠️**

This file provides guidance to Claude Code when working with this qwen-image-app repository. This is a hacky experimental platform - lots of stuff is broken, everything will change, don't get attached to any APIs or patterns!

## Development Setup

### Environment Requirements
- **Python Project** using virtual environments with `uv` package manager
- **Installation**: Run `./scripts/install.sh` to set up complete environment
- **Activation**: **ALWAYS** run `source .venv/bin/activate` before any Python commands
- **Configuration**: Copy and edit `config.yaml` with your model paths

**Critical for Claude**: Never execute Python commands without first activating the virtual environment.

### Launch Commands
```bash
# GUI application (recommended)
./go.sh

# Manual launch
python -m app.app

# Testing
./scripts/test.sh
```

## 🏗️ Architecture Overview

Professional text-to-image generation system with multimodal chat capabilities, built around Qwen-Image and Qwen2.5-VL models.

### Core Business Logic (`src/`)

**Foundation Modules**:
- `config.py` - YAML configuration management with validation
- `models.py` - Multi-GPU model management with component control
- `process.py` - Advanced generation pipeline (txt2img, noise interpolation, two-stage)
- `metadata.py` - Comprehensive PNG metadata embedding/extraction

**Advanced Features**:
- `chat.py` - Multimodal VL model chat interface with message interactions
- `prompt.py` - External template system with enhancement capabilities
- `gallery.py` - Session-based organization and image management
- `analysis.py` - Text encoder analysis and comparison tools
- `edit.py` - Img2img utilities with noise interpolation
- `step.py` - Per-step callbacks for debugging and visualization

### Gradio UI Layer (`app/`)

**Interface Components**:
- `app.py` - Main application with tab coordination
- `generate.py` - Primary generation interface with advanced controls
- `chat.py` - Professional chat interface with edit/retry/undo
- `gallery.py` - Session-based image browser with metadata display
- `models.py` - Component loading controls with real-time status
- `analysis.py` - Encoder analysis tools and token debugging
- `config.py` - Live YAML configuration editor
- `shared.py` - Common utilities, CSS, and UI helpers

### External Systems

**Templates** (`templates/`):
- All prompts externalized for easy customization
- Hot-reloadable without application restart
- Language-aware selection (English/Chinese)
- Professional templates for optimal results

## 📝 Template System Architecture

### External Template Files
All prompts are stored in `templates/` for easy editing:

| Template | Purpose | Features |
|----------|---------|----------|
| `system.txt` | Chat system prompt | VL model personality |
| `chat.txt` | Chat formatting | Jinja2 conversation template |
| `image.txt` | Generation wrapper | Qwen-Image training template |
| `enhancement.txt` | Prompt improvement | English enhancement instructions |
| `enhancement_zh.txt` | Prompt improvement | Chinese enhancement instructions |
| `describe.txt` | Image analysis | Reverse-engineering template |

### Template Capabilities
- **Hot-Reloadable** - Changes apply without restart
- **Language-Aware** - Automatic Chinese/English detection  
- **Professional** - Expert-crafted for optimal results
- **Fallback Handling** - Built-in defaults if files missing
- **Dynamic Loading** - Runtime template loading with caching

## 🔧 Key Architecture Patterns

### Multi-GPU Distribution
**Intelligent Component Allocation**:
```yaml
# Example 5-GPU setup in config.yaml
gpu_text_encoder: "3:8GiB,4:16GiB"        # VL model spans GPUs 3,4
gpu_transformer: "0:13GiB,1:13GiB,2:15.5GiB"  # Diffusion on GPUs 0,1,2
gpu_vae: "cuda:3"                          # VAE on GPU 3
```

**Component Lifecycle**:
- Individual loading/unloading of text encoder, transformer, VAE, scheduler
- Real-time memory monitoring per GPU
- Auto-loading pipeline configuration

### Session Management System
**Organization Patterns**:
- **Date-based Sessions**: Automatic `YYYY-MM-DD` organization
- **URL Routing**: `http://localhost:7860/?session=project-name`
- **State Persistence**: Sessions survive application restarts
- **Chat History**: Conversation history preserved per session

### Generation Pipeline Architecture
**Advanced Generation Modes**:
- **Text-to-Image**: Standard generation with template application
- **Noise Interpolation**: Image transformation using noise mixing
- **Two-Stage**: Progressive refinement with intermediate saving
- **LoRA Integration**: Multiple adapters with individual strength control

**Template Processing**:
- External file-based templates with hot-reload
- Language-aware enhancement (Chinese/English detection)
- Magic prompt injection for quality improvement
- Token counting with 1024 limit enforcement

### Metadata System
**Comprehensive Parameter Preservation**:
- Complete generation settings embedded in PNG files
- Template application tracking (which templates used)
- Model component information and versions
- LoRA configurations and strengths
- Generation mode details (two-stage, noise interpolation)
- Extractable for reproduction and analysis

## ⚙️ Configuration System

### YAML-Based Configuration
**Centralized Settings** in `config.yaml`:
- **Model Paths**: All component locations (diffusion, VL, VAE, tokenizer, scheduler)
- **GPU Allocation**: Per-component memory limits and device mapping
- **Generation Defaults**: Steps, CFG, dimensions, quality settings
- **Feature Flags**: LoRA, metadata embedding, auto-loading, tiling
- **Template References**: External file paths for all prompt types
- **Chat Settings**: Context length (32K), response limits, model behavior

### Advanced Features

**Component Management**:
- Individual loading/unloading of model components
- Alternative text encoder support for experimentation
- Real-time status monitoring with memory usage
- Auto-loading pipeline on startup

**Generation Capabilities**:
- Text-to-image with template wrapping
- Img2img with noise interpolation
- Two-stage generation with intermediate results
- LoRA management with strength control
- Per-step image saving for debugging

**Analysis Tools**:
- Text encoder comparison and analysis
- Token counting with template effects
- Argmax roundtrip analysis for understanding model behavior
- Cross-encoder analysis for comparing different encoders

**Professional Chat Interface**:
- Multimodal VL model integration
- Message interactions (edit, retry, undo, copy)
- Image upload and inline display
- Context preservation across conversations
- Image-to-prompt reverse engineering

## 💬 Chat Interface Capabilities

### Professional Message Interactions
**Advanced UI Features**:
- **Edit Messages**: Click any message (user or assistant) for inline editing
- **Retry Generation**: Regenerate responses from any conversation point
- **Undo Actions**: Remove messages with content recovery to input field
- **Copy Messages**: One-click copying of individual messages
- **Image Integration**: Upload images that appear inline with full context

### Image Analysis Features
**Reverse Engineering**:
- **Describe for Generation**: Specialized template for creating generation-ready prompts
- **Professional Analysis**: Expert template analyzes composition, lighting, style, technique
- **Perfect Prompts**: Creates prompts capable of recreating uploaded images
- **Context Awareness**: Images preserved in conversation history for reference

### Technical Implementation
**VL Model Integration**:
- Qwen2.5-VL model with 32K context window
- Image and text processing with proper multimodal handling
- Temporary file management for image uploads
- Error handling with graceful degradation

## 🧪 Testing Guidelines

### Critical Testing Philosophy
**MANDATORY**: For any new functionality or bug fixes, **ALWAYS** create proper tests in `./tests/` rather than one-off manual commands.

**Why This Matters**:
1. **Reproducible Verification**: Others can validate your changes
2. **Regression Prevention**: Future changes won't break existing functionality
3. **Living Documentation**: Tests demonstrate correct usage patterns  
4. **CI/CD Readiness**: Automated testing pipeline integration
5. **Code Quality**: Forces consideration of edge cases and error conditions

### Test Architecture

**Test Categories**:
- **Unit Tests** (`test_*.py`): Individual functions and classes in isolation
- **Integration Tests**: Component interaction validation
- **UI Tests**: Interface module import and basic functionality
- **Feature Tests**: Complete workflow validation (txt2img, noise interpolation, two-stage)
- **Analysis Tests**: Text encoder and template functionality

### Test Implementation Process

**Step-by-Step**:
1. **Create Test File**: Follow `test_*.py` naming in `./tests/`
2. **Use unittest Framework**: Mock external dependencies (models, file I/O)
3. **Update Test Runner**: Add to `tests/run_tests.py`
4. **Document Tests**: Update `tests/README.md` with test descriptions
5. **Verify Execution**: Run `./scripts/test.sh` to ensure all tests pass

### Testing Best Practices

**Essential Patterns**:
- **Mock Heavy Dependencies**: Use `unittest.mock` for model loading, GPU operations
- **Test Error Conditions**: Validate failure modes, not just success paths
- **Backwards Compatibility**: Ensure changes don't break existing workflows
- **Descriptive Names**: Test method names should explain the scenario
- **Proper Cleanup**: Use `setUp()`/`tearDown()` for test isolation

### Example Implementation

```python
# ❌ AVOID: One-off testing
python -c "from src.process import generate_image; print('works')"

# ✅ PROPER: Structured test
class TestImageGeneration(unittest.TestCase):
    @patch('src.models.get_model_manager')
    def test_basic_generation_workflow(self, mock_manager):
        # Setup mocks
        mock_manager.return_value.text_encoder = Mock()
        
        # Test the actual functionality
        result = generate_image("test prompt", steps=5)
        
        # Validate results
        self.assertIsNotNone(result)
        mock_manager.assert_called_once()
```

**Current Test Coverage**:
- `test_imports.py` - Module import validation
- `test_structure.py` - Codebase organization checks
- `test_noise_interpolation.py` - Img2img functionality
- `test_two_stage_generation.py` - Two-stage workflow
- `test_image_upload.py` - UI image handling

## 🚀 Development Patterns

### Clean Architecture Principles
**Separation of Concerns**:
- **Business Logic** (`src/`): Core functionality independent of UI framework
- **UI Layer** (`app/`): Gradio-specific interface components
- **External Systems** (`templates/`, `config.yaml`): Configurable resources
- **Test Suite** (`tests/`): Comprehensive validation of all components

### Development Workflow
**Adding New Features**:
1. **Plan the Feature**: Understand requirements and integration points
2. **Implement Core Logic**: Add functionality to appropriate `src/` module
3. **Create UI Interface**: Add controls in relevant `app/` module
4. **Write Tests**: Create comprehensive test coverage
5. **Update Documentation**: Modify templates or docs as needed
6. **Test Integration**: Verify with existing features

### Code Organization Patterns
**Module Responsibilities**:
- **State Management**: Global managers for models, sessions, gallery
- **Component Isolation**: Each feature as independent, testable module
- **Shared Utilities**: Common functions in `app/shared.py`
- **Error Handling**: Graceful degradation with user feedback
- **Resource Management**: Automatic cleanup and memory management

### Best Practices for Claude
**When Working on This Codebase**:
1. **Always activate venv**: `source .venv/bin/activate` before any Python command
2. **Follow existing patterns**: Study similar implementations before adding new features
3. **Create proper tests**: Use `./tests/` directory for all validation
4. **Respect architecture**: Keep business logic in `src/`, UI in `app/`
5. **Update documentation**: Modify this file and templates as needed
6. **Test thoroughly**: Run `./scripts/test.sh` before completing work

### Component Interaction Patterns
**Key Integration Points**:
- **Model Manager**: Central state for all loaded components
- **Session Manager**: Coordinates file organization and URL routing  
- **Gallery Manager**: Links generation with browsing and metadata
- **Chat Manager**: Handles VL model interaction and conversation state
- **Template System**: Provides dynamic, hot-reloadable prompt management

### Template Development
**When Modifying Templates**:
- **Test Immediately**: Templates are hot-reloaded, test changes in real-time
- **Language Awareness**: Consider both English and Chinese variants
- **Professional Quality**: Maintain expert-level prompt engineering
- **Documentation**: Update template descriptions in this file

---

**Remember**: This codebase prioritizes clean architecture, comprehensive testing, and professional user experience. All changes should maintain these standards.