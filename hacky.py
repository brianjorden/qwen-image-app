#!/usr/bin/env python3
"""
Pipeline Parallel Qwen-Image
Processes multiple images simultaneously across GPU stages using traffic cop coordination
"""

import os
from pathlib import Path
import torch
import json
from datetime import datetime
import threading
import time
from typing import List, Dict, Optional
import signal
import sys

from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from diffusers import DiffusionPipeline

# === Configuration ===
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

MODEL_DIR = os.path.expanduser("~/models/Qwen-Image")
ENC_PATH = os.path.expanduser("~/models/Qwen-Image/text_encoder")
OUTDIR = Path("./output")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Memory allocation per GPU
enc_mem = {0: "0GiB", 1: "0GiB", 2: "0GiB", 3: "1.5GiB", 4: "15.5GiB"}
xfmr_mem = {0: "10.5GiB", 1: "10.5GiB", 2: "10.5GiB", 3: "14GiB", 4: "0GiB"}

# VAE needs to be on same device as render_device for now
vae_device = "cuda:3"  # Can be changed to CPU or other device
render_device = "cuda:0"

# === Global State ===
GPU_STAGE_BUSY = []
STAGE_LOCKS = []
EXIT_EARLY = False
STAGE_TIMEOUT = 30.0
VAE_LOCK = threading.Lock()

# Performance tracking (optional, can be disabled)
TRACK_PERFORMANCE = False
STAGE_WAIT_TIMES = {}
STAGE_PROCESS_TIMES = {}

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global EXIT_EARLY
    print("\n[INFO] Shutting down pipeline...")
    EXIT_EARLY = True
    time.sleep(0.5)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def detect_layer_distribution(transformer):
    """Detect which layers are on which GPUs"""
    layer_to_device = {}

    if hasattr(transformer, 'transformer_blocks'):
        blocks = transformer.transformer_blocks
    else:
        raise ValueError("Can't find transformer blocks in model")

    for i, block in enumerate(blocks):
        for param in block.parameters():
            device = param.device
            layer_to_device[i] = device.index if device.type == 'cuda' else -1
            break

    # Group consecutive layers on same device
    stage_ranges = []
    current_device = layer_to_device[0]
    start_layer = 0

    for i in range(1, len(blocks)):
        if layer_to_device[i] != current_device:
            stage_ranges.append((start_layer, i, current_device))
            start_layer = i
            current_device = layer_to_device[i]

    stage_ranges.append((start_layer, len(blocks), current_device))

    print(f"\n[INFO] GPU Stage Distribution:")
    for idx, (start, end, device) in enumerate(stage_ranges):
        print(f"  Stage {idx} (GPU {device}): layers {start}-{end-1} ({end-start} layers)")

    return stage_ranges


class PipelinedTransformer:
    """Wrapper that adds pipeline parallelism to transformer"""

    def __init__(self, transformer):
        self.transformer = transformer
        self.transformer_blocks = transformer.transformer_blocks
        self.stage_ranges = detect_layer_distribution(transformer)
        self.num_stages = len(self.stage_ranges)

        # Initialize global state
        global GPU_STAGE_BUSY, STAGE_LOCKS, STAGE_WAIT_TIMES, STAGE_PROCESS_TIMES
        GPU_STAGE_BUSY = [False] * self.num_stages
        STAGE_LOCKS = [threading.Lock() for _ in range(self.num_stages)]

        if TRACK_PERFORMANCE:
            STAGE_WAIT_TIMES = {i: [] for i in range(self.num_stages)}
            STAGE_PROCESS_TIMES = {i: [] for i in range(self.num_stages)}

        # Create CUDA streams for parallelism
        unique_gpus = set(device for _, _, device in self.stage_ranges)
        self.cuda_streams = {
            gpu_id: torch.cuda.Stream(device=f'cuda:{gpu_id}')
            for gpu_id in unique_gpus
        }

        # Wrap transformer blocks
        self._wrap_blocks()

    def _wrap_blocks(self):
        """Replace transformer blocks with traffic-cop-aware versions"""

        class BlockWrapper(torch.nn.Module):
            def __init__(self, layer_idx, stage_idx, parent, original_block):
                super().__init__()
                self.layer_idx = layer_idx
                self.stage_idx = stage_idx
                self.parent = parent
                self.original_block = original_block

            def forward(self, *args, **kwargs):
                # Identify which image this is
                thread_name = threading.current_thread().name
                image_id = int(thread_name.split('_')[-1]) if '_' in thread_name else 0

                # First layer of stage: acquire lock
                start_layer, _, _ = self.parent.stage_ranges[self.stage_idx]
                if self.layer_idx == start_layer:
                    if not self.parent.wait_for_stage(self.stage_idx, image_id):
                        return args[0]
                    STAGE_LOCKS[self.stage_idx].acquire()
                    GPU_STAGE_BUSY[self.stage_idx] = True

                # Process through original block
                result = self.original_block(*args, **kwargs)

                # Last layer of stage: release lock
                _, end_layer, _ = self.parent.stage_ranges[self.stage_idx]
                if self.layer_idx == end_layer - 1:
                    GPU_STAGE_BUSY[self.stage_idx] = False
                    STAGE_LOCKS[self.stage_idx].release()

                return result

        # Replace all blocks
        for i in range(len(self.transformer_blocks)):
            stage_idx = next(idx for idx, (start, end, _) in enumerate(self.stage_ranges)
                           if start <= i < end)
            original_block = self.transformer_blocks[i]
            self.transformer_blocks[i] = BlockWrapper(i, stage_idx, self, original_block)

    def wait_for_stage(self, stage_idx: int, image_id: int) -> bool:
        """Wait for GPU stage to become available"""
        wait_start = time.time()

        while GPU_STAGE_BUSY[stage_idx]:
            if EXIT_EARLY:
                return False

            elapsed = time.time() - wait_start
            if elapsed > STAGE_TIMEOUT:
                print(f"[WARNING] Image {image_id} timeout at stage {stage_idx}")
                return False

            time.sleep(0.001)

        if TRACK_PERFORMANCE:
            wait_time = time.time() - wait_start
            if wait_time > 0.01:
                STAGE_WAIT_TIMES[stage_idx].append(wait_time)

        return True


def process_single_image(pipe, job_id: int, prompt: str, height: int, width: int,
                        num_steps: int, seed: int):
    """Process a single image through the pipeline"""
    threading.current_thread().name = f"image_{job_id}"

    print(f"[Image {job_id}] Starting (seed={seed})")
    start_time = time.time()

    try:
        generator = torch.Generator(device=render_device).manual_seed(seed)
        scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)

        with torch.inference_mode():
            # === Encoding Phase ===
            prompt_embeds, prompt_mask = pipe.encode_prompt(
                prompt=prompt,
                device=render_device,
                num_images_per_prompt=1,
            )

            # === Latent Preparation ===
            num_channels = pipe.transformer.config.in_channels // 4
            latents, _ = pipe.prepare_latents(
                batch_size=1,
                num_channels_latents=num_channels,
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=render_device,
                generator=generator,
                latents=None,
            )

            # === Denoising Loop ===
            image_seq_len = latents.shape[1]
            mu = calculate_shift(image_seq_len)
            scheduler.set_timesteps(num_steps, device=render_device, mu=mu)
            timesteps = scheduler.timesteps

            for i, t in enumerate(timesteps):
                if EXIT_EARLY:
                    return None

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                txt_seq_lens = prompt_mask.sum(dim=1).tolist() if prompt_mask is not None else None

                # Forward through pipelined transformer
                noise_pred = pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                    img_shapes=[(1, height // pipe.vae_scale_factor // 2,
                                width // pipe.vae_scale_factor // 2)],
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]

                if i < len(timesteps) - 1:
                    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            print(f"[Image {job_id}] Denoising complete, decoding...")

            # === VAE Decode (immediately after denoising) ===
            with VAE_LOCK:
                # Move latents to VAE device if different
                if vae_device != render_device:
                    latents = latents.to(vae_device)

                with torch.cuda.device(int(vae_device.split(':')[1]) if 'cuda' in vae_device else None):
                    latents = unpack_latents(latents, height, width, pipe.vae_scale_factor)
                    latents = latents.to(pipe.vae.dtype)
                    latents = denormalize_latents(latents, pipe.vae)
                    image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
                    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

            # === Save Image ===
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"img_{timestamp}_job{job_id}_s{seed}.png"
            out_path = OUTDIR / filename
            image.save(str(out_path))

            elapsed = time.time() - start_time
            print(f"[Image {job_id}] Complete in {elapsed:.1f}s → {filename}")
            return out_path

    except Exception as e:
        print(f"[Image {job_id}] Error: {e}")
        return None


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                   base_shift=0.5, max_shift=1.15):
    """Calculate mu for Qwen scheduler"""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def unpack_latents(latents, height, width, vae_scale_factor):
    """Unpack Qwen-Image latents"""
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    batch_size, num_patches, channels = latents.shape
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, 1, height, width)
    return latents


def denormalize_latents(latents, vae):
    """Denormalize latents for VAE"""
    device, dtype = latents.device, latents.dtype
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    return latents / latents_std + latents_mean


def load_pipeline():
    """Load and distribute the pipeline across GPUs"""
    print("[INFO] Loading model components...")

    # Load text encoder
    enc_cfg = Qwen2_5_VLConfig.from_pretrained(ENC_PATH, local_files_only=True)
    with init_empty_weights():
        empty_enc = Qwen2_5_VLForConditionalGeneration(enc_cfg)

    layer_container = getattr(empty_enc.model, "language_model", empty_enc.model).layers
    enc_layer_cls = type(layer_container[0]).__name__

    enc_device_map = infer_auto_device_map(
        empty_enc,
        max_memory=enc_mem,
        dtype=torch.bfloat16,
        no_split_module_classes=[enc_layer_cls],
    )

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ENC_PATH,
        torch_dtype=torch.bfloat16,
        device_map=enc_device_map,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    # Load main pipeline
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        text_encoder=text_encoder,
        local_files_only=True,
    )
    pipe.set_progress_bar_config(disable=True)

    # Distribute transformer
    xfmr = pipe.transformer
    block_cls = type(xfmr.transformer_blocks[0]).__name__

    xfmr_map = infer_auto_device_map(
        xfmr,
        max_memory=xfmr_mem,
        dtype=torch.bfloat16,
        no_split_module_classes=[block_cls],
    )

    pipe.transformer = dispatch_model(xfmr, device_map=xfmr_map)

    # Setup VAE
    if hasattr(pipe, "vae"):
        pipe.vae.enable_tiling()
        # Keep VAE on specified device
        vae_map = {"": vae_device}
        pipe.vae = dispatch_model(pipe.vae, device_map=vae_map)

    return pipe


def main():
    """Main execution"""
    print("="*50)
    print("PIPELINE PARALLEL QWEN-IMAGE")
    print("="*50)

    # Load model
    pipe = load_pipeline()

    # Apply pipeline parallelism
    print("[INFO] Enabling pipeline parallelism...")
    pipelined_transformer = PipelinedTransformer(pipe.transformer)

    # Load jobs
    jobs = [
        {"prompt": "A serene mountain landscape at sunset", "seed": 4211111111},
        {"prompt": "A futuristic cyberpunk city", "seed": 4311111111},
        {"prompt": "An enchanted forest with glowing mushrooms", "seed": 4411111111},
        {"prompt": "A cozy cabin in a snowy winter scene", "seed": 4511111111},
        {"prompt": "A serene mountain landscape at sunset", "seed": 4222222222},
        {"prompt": "A futuristic cyberpunk city", "seed": 4322222222},
        {"prompt": "An enchanted forest with glowing mushrooms", "seed": 4422222222},
        {"prompt": "A cozy cabin in a snowy winter scene", "seed": 4522222222},
        {"prompt": "A serene mountain landscape at sunset", "seed": 4233333333},
        {"prompt": "A futuristic cyberpunk city", "seed": 4333333333},
        {"prompt": "An enchanted forest with glowing mushrooms", "seed": 4433333333},
        {"prompt": "A cozy cabin in a snowy winter scene", "seed": 4533333333},
    ]

    print(f"\n[INFO] Generating {len(jobs)} images in parallel...")
    overall_start = time.time()

    # Launch parallel threads
    threads = []
    for i, job in enumerate(jobs):
        thread = threading.Thread(
            target=process_single_image,
            args=(pipe, i, job["prompt"], 512, 512, 20, job["seed"]),
            name=f"image_{i}"
        )
        thread.start()
        threads.append(thread)
        time.sleep(0.25)  # Small stagger to fill pipeline

    # Wait for completion
    for thread in threads:
        thread.join()

    # Summary
    overall_time = time.time() - overall_start
    print(f"\n{'='*50}")
    print(f"Completed {len(jobs)} images in {overall_time:.1f}s")
    print(f"Average: {overall_time/len(jobs):.1f}s per image")
    print(f"Speedup: ~{(len(jobs)*20)/overall_time:.1f}x vs sequential")
    print("="*50)


if __name__ == "__main__":
    main()
