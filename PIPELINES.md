# Pipeline Parallel Qwen-Image

## Overview

This implementation enables true pipeline parallelism for the Qwen-Image diffusion model, allowing multiple images to be generated simultaneously across multiple GPUs. Instead of processing images sequentially (one after another), the system processes multiple images in parallel by distributing different stages of the transformer across GPUs.

## The Problem

Standard diffusion model inference processes images sequentially:
- Image 1 uses all GPUs → completes → Image 2 uses all GPUs → completes → etc.
- With 4 images at 20 seconds each, total time = 80 seconds
- GPUs spend significant time idle waiting for their stage of the pipeline

## The Solution: Pipeline Parallelism

The transformer's 60 layers are automatically distributed across available GPUs:
- GPU 0: Layers 0-14 (15 layers)
- GPU 1: Layers 15-30 (16 layers)
- GPU 2: Layers 31-46 (16 layers)
- GPU 3: Layers 47-59 (13 layers)

Multiple images flow through this pipeline simultaneously, with different images at different stages.

## How It Works

### 1. Automatic Layer Detection
The system automatically detects how the model was distributed across GPUs by `accelerate`'s `dispatch_model`, eliminating manual configuration.

### 2. Traffic Cop Coordination
A "traffic cop" system prevents GPU conflicts:
- Each GPU stage can only process one image at a time
- When an image needs a stage, it waits if that stage is busy
- Stages are released immediately after processing, allowing the next waiting image to proceed

### 3. Thread-Based Parallelism
Each image generation runs in its own thread:
- Threads are identified by name (e.g., "image_0", "image_1")
- The transformer blocks use thread identity to coordinate stage access
- Independent scheduler instances prevent state corruption between threads

### 4. Implementation Details

**Key Components:**
- `BlockWrapper`: Wraps each transformer block to implement traffic cop logic
- `STAGE_LOCKS`: Threading locks that ensure exclusive GPU stage access
- `GPU_STAGE_BUSY`: Boolean flags tracking stage availability
- `VAE_LOCK`: Prevents concurrent VAE decoding (VAE isn't pipelined yet)

**The Flow:**
1. Multiple threads start image generation with slight stagger (0.25s)
2. Each thread independently processes through encoding → denoising → decoding
3. During denoising (20 steps), threads coordinate GPU usage via traffic cop
4. Images complete in non-deterministic order based on GPU availability
5. VAE decoding happens immediately after denoising (no backlog)

## Performance Characteristics

**Observed Results:**
- 4 images: ~21 seconds total (3.8x speedup)
- 12 images: ~61 seconds total (3.9x speedup)
- Average: ~5.1 seconds per image regardless of batch size

**Why Not 4x Speedup?**
- Pipeline fill/drain time at start and end
- VAE decoding is still sequential (protected by lock)
- Small coordination overhead from traffic cop system
- Memory bandwidth limitations

**Scaling Behavior:**
- Performance remains consistent as more images are added
- Completion order is non-deterministic (based on stage availability)
- System automatically balances load across GPUs

## Limitations

1. **VAE Not Pipelined**: The VAE decode step is protected by a lock and processes sequentially
2. **Fixed Stage Distribution**: Stages are determined by how `accelerate` distributes the model
3. **Memory Constraints**: Each GPU must have enough memory for its layers plus one image's activations

## Potential Improvements

1. **Pipeline the VAE**: Distribute VAE layers across GPUs for parallel decoding
2. **Dynamic Batching**: Process multiple images per stage when GPU memory allows
3. **Adaptive Scheduling**: Prioritize images closer to completion
4. **CPU Offloading**: Move completed latents to CPU while waiting for VAE

## Technical Requirements

- Multiple GPUs with sufficient memory
- Model distributed across GPUs using `accelerate`
- Thread-safe execution environment
- CUDA streams for true parallel execution

## Conclusion

This pipeline parallel implementation achieves ~3.9x speedup on 4+ GPUs by keeping all GPUs busy processing different images simultaneously. The traffic cop pattern ensures clean coordination without complex message passing, making it relatively simple to implement as a wrapper around existing diffusion pipelines.
