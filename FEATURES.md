# Features - qwen-image-app

**⚠️ EXPERIMENTAL FEATURES LIST ⚠️**

This is a hacky experimentation platform for exploring Qwen-Image model capabilities. Many features are broken, half-implemented, or will change completely. Don't rely on anything staying the same!

## 🎨 Core Generation

### Text-to-Image Generation
- **Qwen-Image Model Support** - Native integration with Qwen-Image transformer and VAE
- **Multi-GPU Distribution** - Intelligent model component distribution across available GPUs
- **Flexible Resolution** - Support for multiple aspect ratios with preset dimensions
- **Advanced Sampling** - Flow matching scheduler with configurable steps and CFG scales
- **Seed Control** - Deterministic generation with manual or random seed selection

### Weird Image Transformation (Not Img2Img!)
- **Noise Interpolation** - Bizarre image transformation using noise mixing (defaults to 0.99 strength)
- **Strength Control** - How weird you want things to get (0.0-1.0, but go wild with 0.99)
- **Image Upload** - Drag and drop images to transform them weirdly
- **Dimension Matching** - Resizes your input image to match generation size

### Two-Stage Generation
- **Progressive Refinement** - Two-stage generation with intermediate result saving
- **Noise Interpolation Mode** - Second stage uses img2img for enhanced detail
- **Flexible Step Control** - Independent step count for each stage
- **Intermediate Results** - Automatic saving of stage 1 images with metadata

### LoRA Support
- **Multiple LoRAs** - Load up to 4 LoRA adapters simultaneously
- **Individual Strength** - Per-LoRA strength control (0.0-2.0)
- **Hot Swapping** - Dynamic loading/unloading without model restart
- **Metadata Preservation** - LoRA configurations saved in generated image metadata

## 🤖 Multimodal Chat Interface

### VL Model Integration
- **Qwen2.5-VL Chat** - Full conversation interface with vision-language model
- **32K Context Window** - Support for long conversations with image context
- **Image Understanding** - Upload and discuss images inline with full context preservation
- **Response Control** - Configurable max tokens (50-4000) for response length

### Professional Message Interactions
- **✏️ Edit Messages** - Click any message (yours or assistant's) to edit inline
- **🔄 Retry Generation** - Regenerate responses from any point in conversation
- **↩️ Undo Messages** - Remove messages from conversation history with content recovery
- **📋 Copy Messages** - One-click copying of individual messages
- **🖼️ Image Display** - Images appear inline in conversation history

### Image Analysis Features
- **Image-to-Prompt** - Reverse-engineer perfect text-to-image prompts from uploaded images
- **Professional Templates** - Expert-crafted analysis templates for optimal prompt generation
- **Describe for Generation** - Specialized mode for creating generation-ready prompts
- **Context Preservation** - Full conversation history maintained with images

## 📝 Template & Enhancement System

### External Template System
- **File-Based Templates** - All prompts and templates stored in editable `./templates/` files
- **Dynamic Loading** - Templates load at runtime with fallback handling
- **Hot-Reloadable** - Update templates without application restart
- **Language-Aware** - Automatic Chinese/English template selection

### Template Types
- **System Prompts** (`system.txt`) - Chat interface system prompts
- **Chat Templates** (`chat.txt`) - Jinja2-based conversation formatting
- **Image Generation** (`image.txt`) - Text-to-image prompt wrapping
- **Enhancement** (`enhancement.txt`/`enhancement_zh.txt`) - Prompt improvement instructions
- **Image Description** (`describe.txt`) - Reverse-engineering analysis templates

### Prompt Enhancement
- **Local VL Enhancement** - Use loaded VL model for prompt improvement
- **Language Detection** - Automatic Chinese/English detection for appropriate enhancement
- **Magic Prompts** - Quality enhancement suffixes with language-aware selection
- **Preview & Edit** - Enhanced prompts shown in popup for review before application
- **Token Counting** - Real-time token count with 1024 token limit tracking

## 🖼️ Gallery & Session Management

### Session Organization
- **Date-Based Sessions** - Automatic organization by date (YYYY-MM-DD format)
- **Custom Sessions** - Create named sessions for specific projects
- **URL Routing** - Access sessions via `http://localhost:7860/?session=session-name`
- **Automatic Creation** - Sessions created automatically when needed

### Gallery Browser
- **Grid View** - 4-column responsive gallery layout with configurable rows
- **Metadata Display** - Click any image to view complete generation parameters
- **Sorting** - Images sorted by modification time (newest first)
- **File Management** - Delete images with automatic metadata cleanup

### Metadata System
- **PNG Embedding** - Complete generation parameters embedded in PNG files
- **Comprehensive Data** - Prompts, settings, model info, LoRA configs, timestamps
- **Template Tracking** - Applied templates and magic prompts preserved
- **Generation Mode Info** - Two-stage and img2img details captured
- **Extractable** - Metadata can be extracted and applied to new generations

## 🔧 Model Management

### Component-Level Control
- **Individual Loading** - Load/unload text encoder, transformer, VAE, scheduler separately
- **Status Monitoring** - Real-time component status with memory usage display
- **Alternative Encoders** - Support for secondary text encoders with independent GPU allocation
- **Pipeline Assembly** - Build complete pipeline from loaded components

### GPU Memory Management
- **Multi-GPU Distribution** - Intelligent allocation across available GPUs
- **Memory Tracking** - Real-time VRAM usage monitoring per GPU
- **Flexible Allocation** - Configure memory limits per component
- **Auto-Loading** - Optional automatic pipeline loading on startup

### Model Components
- **Text Encoder** - Qwen2.5-VL model for text understanding and chat
- **Transformer** - Qwen-Image diffusion transformer (~40GB)
- **VAE** - QwenImageVAE for encoding/decoding (~250MB)
- **Tokenizer** - Qwen tokenizer for text processing
- **Scheduler** - Flow matching scheduler for denoising
- **Processor** - Vision-language processor for multimodal inputs

## 🔍 Analysis & Debugging

### Text Encoder Analysis
- **Token Counting** - Real-time token analysis with template application
- **Argmax Roundtrip** - Encode then decode analysis for understanding model behavior
- **Cross-Encoder Analysis** - Compare different text encoders with cross-decoding
- **Greedy Continuation** - Generate text continuations from prompts
- **Template Effects** - Analyze impact of template wrapping on token processing

### Generation Debugging
- **Per-Step Saving** - Save intermediate images at each denoising step
- **Step Callbacks** - Customizable per-step processing with VAE decoding
- **Metadata Tracking** - Complete parameter history for reproducibility
- **Error Handling** - Comprehensive error reporting with stack traces

## ⚡ Performance & Optimization

### GPU Optimization
- **Multi-GPU Support** - Distribute model components across multiple GPUs
- **Memory Allocation** - Configurable VRAM limits per component
- **VAE Tiling** - Enable tiling for large image generation
- **Attention Slicing** - Optional attention optimization for memory efficiency

### Processing Features
- **Batch Generation** - JSONL-based batch processing with progress tracking
- **Queue System** - Persistent generation queue with auto-recovery
- **Session Targeting** - Queue items can target specific sessions
- **Background Processing** - Non-blocking generation with progress updates

## 🛠️ Configuration & Setup

### Configuration System
- **YAML-Based Config** - Centralized configuration in `config.yaml`
- **Live Editing** - In-app configuration editor with validation
- **Path Expansion** - Automatic home directory expansion for model paths
- **Validation** - Required field checking with helpful error messages

### Installation & Setup
- **Automated Installation** - `./scripts/install.sh` handles complete setup
- **Virtual Environment** - Isolated Python environment with `uv` package manager
- **Dependency Management** - Automatic CUDA PyTorch installation
- **Verification** - Post-install verification with system information

### Launch Options
- **GUI Mode** - Full Gradio web interface (`./go.sh`)
- **CLI Mode** - Command-line interface for scripting
- **Development Mode** - Debug mode with enhanced logging
- **Custom Ports** - Configurable host and port settings

## 🧪 Testing & Development

### Test Suite
- **Import Tests** - Verify all modules import correctly
- **Structure Tests** - Validate codebase organization
- **Feature Tests** - Test core functionality like noise interpolation
- **UI Tests** - Verify interface components load without errors
- **Integration Tests** - End-to-end workflow testing

### Development Tools
- **Clean Architecture** - Separated business logic (`src/`) and UI (`app/`)
- **Modular Design** - Each feature implemented as independent modules
- **Error Handling** - Comprehensive exception handling with user feedback
- **Logging** - Detailed logging for debugging and monitoring

## 🔐 Security & Reliability

### Data Handling
- **Local Processing** - All generation happens locally
- **Metadata Privacy** - No external data transmission
- **Session Isolation** - Clear separation between different sessions
- **File Security** - Safe file handling with path validation

### Error Recovery
- **Graceful Degradation** - Features continue working if components fail
- **Auto-Recovery** - Queue system recovers from interruptions
- **Fallback Handling** - Default values when templates or configs missing
- **Memory Management** - Automatic CUDA cache clearing and garbage collection

---

## Quick Feature Summary

| Category | Features |
|----------|----------|
| **Generation** | Text-to-image, img2img, two-stage, LoRA support |
| **UI** | Gradio web interface, chat, gallery, analysis tools |
| **Models** | Component control, multi-GPU, alternative encoders |
| **Templates** | External files, hot-reload, language-aware |
| **Enhancement** | VL model enhancement, magic prompts, preview |
| **Organization** | Sessions, metadata, queue system |
| **Analysis** | Token counting, encoder comparison, step debugging |
| **Performance** | Multi-GPU, batching, optimization options |
| **Config** | YAML-based, live editing, validation |
| **Development** | Test suite, modular design, error handling |

This feature set makes qwen-image-app a comprehensive solution for professional text-to-image generation with the Qwen-Image model family.