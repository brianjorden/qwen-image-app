# qwen-image-app

**⚠️ EXPERIMENTAL** - Hacky text-to-image generation playground built around Qwen-Image model. Lots of stuff is broken, everything will change multiple times, don't get attached to anything!

## 🚧 What This Is

🎨 **Text-to-Image Generation** - Basic generation with weird image transformation features
🤖 **Multimodal Chat** - VL model conversation that sometimes works
📝 **External Templates** - File-based prompts because why not
🖼️ **Gallery Thing** - Browse generated images if they don't crash
⚡ **Multi-GPU Stuff** - Tries to use multiple GPUs, results may vary
🔧 **Component Loading** - Load/unload model pieces individually

**This is a personal experimentation platform for exploring Qwen-Image capabilities. Features are half-implemented, APIs will change, stuff will break. You've been warned!**

[📋 **See Full Feature List →**](FEATURES.md)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Clone and setup
git clone <repository-url>
cd qwen-image-app

# Make go.sh executable and run (handles installation automatically)
chmod +x go.sh
./go.sh
```

### 2. Configure Models
```bash
# Create symlinks for models and LoRAs (recommended)
ln -s ~/models ./models
ln -s ~/loras ./loras

# Edit config.yaml with your model paths
nano config.yaml
```

### 3. Launch Application
```bash
# Start GUI
./go.sh
```

Access at: http://localhost:7860

## 🏗️ Architecture

### Core Components
```
src/                    # Business Logic
├── config.py          # YAML configuration management
├── models.py          # Multi-GPU model management
├── process.py         # Generation pipeline
├── chat.py            # VL model chat interface
├── gallery.py         # Session and image management
├── prompt.py          # Template processing & enhancement
├── metadata.py        # PNG metadata embedding
├── analysis.py        # Text encoder analysis tools
└── edit.py            # Img2img utilities

app/                    # Gradio UI
├── app.py             # Main application entry
├── generate.py        # Image generation interface
├── chat.py            # Chat interface with interactions
├── gallery.py         # Gallery browser
├── models.py          # Model management controls
└── config.py          # Configuration editor

templates/              # External Templates
├── system.txt         # Chat system prompt
├── image.txt          # Generation template
├── enhancement.txt    # Prompt enhancement (EN)
├── enhancement_zh.txt # Prompt enhancement (ZH)
└── describe.txt       # Image-to-prompt template
```

### Template System
All prompts are externalized for easy customization:
- **Hot-reloadable** - No restart required
- **Language-aware** - Automatic Chinese/English detection
- **Professional** - Expert-crafted for optimal results
- **File-based** - Easy editing and version control

## ⚙️ Configuration

### Essential Settings
Edit `config.yaml` with your setup:

```yaml
# Model Locations
model_diffusion: "~/models/Qwen-Image"
model_text_encoder: "~/models/Qwen2.5-VL-7B-Instruct-abliterated"
model_vae: "~/models/Qwen-Image/vae"

# GPU Distribution (example for 5-GPU setup)
gpu_text_encoder: "3:8GiB,4:16GiB"        # VL model on GPUs 3,4
gpu_transformer: "0:13GiB,1:13GiB,2:15.5GiB"  # Diffusion on GPUs 0,1,2
gpu_vae: "cuda:3"                          # VAE on GPU 3

# Features
auto_load_pipeline: true     # Load models on startup
enable_lora: true           # Enable LoRA adapters
enable_metadata_embed: true # Save parameters in images
```

[📖 **Full Configuration Guide →**](docs/configuration.md)

## 🎯 Usage Examples

### Interface Tabs
- **🎨 Generate** - Main image creation with two-stage and img2img options
- **🖼️ Gallery** - Browse images by session with metadata extraction
- **💬 Chat** - VL model conversation with image upload and message interactions
- **🔬 Analysis** - Text encoder comparison and token analysis tools
- **🔧 Models** - Component loading controls with real-time status
- **⚙️ Config** - Live YAML configuration editor

### Multimodal Chat Features
**Message Controls:**
- **Edit** - Click any message to modify (yours or assistant's)
- **Retry** - Regenerate responses from any point
- **Undo** - Remove messages with content recovery
- **Copy** - One-click message copying

**Image Analysis:**
- **Upload & Discuss** - Images appear inline with context preservation
- **Describe for Generation** - Get professional prompts from uploaded images
- **Reverse Engineering** - Analyze composition, lighting, and style

### Generation Modes
```bash
# Basic text-to-image
prompt: "mountain landscape"
→ Maybe enable "Add Quality Enhancement" if you're feeling lucky
→ Click "Enhance" for AI-improved prompts (when it works)

# Noise Interpolation (image transformation)
→ Upload input image
→ Crank strength to 0.99 for maximum transformation
→ Generate with different prompt for creative results

# Two-stage generation (experimental)
→ Set "Second Stage Steps" > 0
→ Stage 1 makes base image
→ Stage 2 applies noise interpolation transformation
```

### Session Workflow
```bash
# Organize by project
http://localhost:7860/?session=character-designs

# Or use date-based (automatic)
http://localhost:7860/?session=2024-01-15

# Gallery shows all session images with metadata
```

## 💾 Model Requirements

### Qwen-Image Model Family
Download required models and update paths in `config.yaml`:

| Component | Size | Purpose |
|-----------|------|---------|
| **Qwen-Image** | ~40GB | Main diffusion transformer |
| **Qwen2.5-VL-7B** | ~15GB | Vision-language model for chat |
| **QwenImageVAE** | ~250MB | Image encoder/decoder |
| **Tokenizer** | ~2MB | Text tokenization |
| **Scheduler** | ~1KB | Flow matching scheduler |

### GPU Memory Requirements
**Minimum:** 24GB total VRAM (single GPU)
**Recommended:** 48GB+ VRAM (multi-GPU)

**Example Distributions:**
```bash
# 5-GPU Setup (RTX 4090s)
GPU 0,1,2: Transformer (13GB each)
GPU 3: Text Encoder + VAE (8GB)
GPU 4: Text Encoder (16GB)

# 2-GPU Setup (RTX 4090s)
GPU 0: Transformer + VL model (24GB)
GPU 1: Transformer overflow (20GB)
```

## 🔧 Performance Tips

### GPU Optimization
- **Multi-GPU Distribution** - Spread models across available GPUs
- **Memory Management** - Configure per-component VRAM limits
- **VAE Tiling** - Enable for large image generation
- **Auto-Loading** - Pre-load models on startup

### Generation Speed
- **Component Loading** - Load only needed components
- **Batch Processing** - Use queue system for multiple images
- **Step Optimization** - Adjust steps based on quality needs
- **Template Caching** - Templates cached after first load

## 🛠️ Installation

### System Requirements
- **Python:** 3.10+
- **CUDA:** 12.1+ with compatible drivers
- **GPU:** 24GB+ VRAM total (multi-GPU recommended)
- **RAM:** 16GB+ system memory
- **Storage:** 100GB+ for models

### Quick Install
```bash
# Clone repository
git clone <repository-url>
cd qwen-image-app

# Install with automated script
chmod +x scripts/install.sh
./scripts/install.sh

# Activate environment
source .venv/bin/activate
```

### Verification
```bash
# Test installation
./scripts/test.sh

# Check GPU setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **CUDA Out of Memory** | Adjust GPU allocation in config, enable VAE tiling |
| **Models Not Loading** | Check paths in config.yaml, verify model files exist |
| **Slow Generation** | Balance GPU allocation, disable unused features |
| **Template Errors** | Verify template files exist in `./templates/` directory |
| **Chat Not Working** | Ensure VL model loaded, check processor status |
| **Queue Stuck** | Check Models tab for component status |

### Performance Issues
```bash
# Check GPU memory usage
nvidia-smi

# Monitor component loading
# → Models tab → "Refresh Status"

# Verify template loading
ls -la templates/
```

### Getting Help
1. 📋 Check [FEATURES.md](FEATURES.md) for complete capabilities
2. 🔍 Review configuration examples above
3. 🐛 Test with `./scripts/test.sh`
4. 💬 Open GitHub issue with logs and system info

## 🚀 Development

### Project Structure
- **Clean Architecture** - Business logic (`src/`) separated from UI (`app/`)
- **Modular Design** - Each feature as independent module
- **External Templates** - All prompts in editable files
- **Test Coverage** - Comprehensive test suite in `./tests/`

### Contributing
```bash
# Run tests before submitting
./scripts/test.sh

# Follow existing code patterns
# Add tests for new features
# Update documentation
```

## 📄 License

[License TBD]

## 🙏 Credits

- **Qwen-Image** - Alibaba Cloud's text-to-image model
- **Qwen2.5-VL** - Multimodal vision-language model
- **Diffusers** - HuggingFace's diffusion model library
- **Gradio** - Web interface framework

---

**Star ⭐ this repo if you find it useful!**
