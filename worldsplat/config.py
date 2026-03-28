"""Central configuration for WorldSplat pipeline."""

from pathlib import Path

# --- Model defaults ---

# Video generation: Wan2.2-TI2V-5B — supports both text→video AND image→video
# in a single model. Fits on RTX 4090 (24GB). 720p @ 24fps.
VIDEO_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Fallback: lighter model for low-VRAM setups (8GB)
VIDEO_MODEL_ID_SMALL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Number of frames to generate
VIDEO_NUM_FRAMES = 33  # ~1.4 seconds at 24fps (fits 24GB VRAM)

# Video resolution — 480p default to fit 24GB GPUs.
# Use --height 720 --width 1280 on 48GB+ GPUs.
VIDEO_HEIGHT = 480
VIDEO_WIDTH = 832

# Inference steps (more = better quality, slower)
VIDEO_NUM_INFERENCE_STEPS = 50

# Guidance scale (how closely to follow the prompt)
VIDEO_GUIDANCE_SCALE = 5.0

# Flow shift — controls sampling schedule. 5.0 for 720p.
VIDEO_FLOW_SHIFT = 5.0

# --- Pose estimation ---
# MASt3R model (set up separately via git clone)
POSE_MODEL_ID = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

# --- Output paths ---
DEFAULT_OUTPUT_DIR = Path("output")

# --- Device ---
# "cuda" if available, falls back to "cpu" (won't actually work for generation)
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
