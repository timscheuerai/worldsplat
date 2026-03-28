"""Step 1: Text → Video frames using Wan 2.2 TI2V.

This module downloads Wan2.2-TI2V-5B (first run only) and generates a short
video from a text prompt (and optionally an input image). The video is saved
as individual PNG frames for 3D reconstruction.

Why Wan 2.2 TI2V-5B?
- Best VBench score among open-source video models
- Supports BOTH text→video AND image→video in one model
- Runs on RTX 4090 (24GB VRAM)
- 720p @ 24fps output
- Native HuggingFace Diffusers support
- Image→video mode is critical for later pipeline stages
  (iterative inpainting, filling holes in 3D scenes)

Usage:
    # Text only
    python -m worldsplat.video_gen "a narrow Tokyo alley at night"

    # Text + reference image
    python -m worldsplat.video_gen "camera moving forward" --image scene.png
"""

import logging
from pathlib import Path

import torch
from PIL import Image

from worldsplat.config import (
    DEFAULT_OUTPUT_DIR,
    DEVICE,
    DTYPE,
    VIDEO_FLOW_SHIFT,
    VIDEO_GUIDANCE_SCALE,
    VIDEO_HEIGHT,
    VIDEO_MODEL_ID,
    VIDEO_NUM_FRAMES,
    VIDEO_NUM_INFERENCE_STEPS,
    VIDEO_WIDTH,
)

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    """Turn a prompt into a filesystem-safe directory name."""
    slug = text.lower().strip()
    slug = "".join(c if c.isalnum() or c == " " else "" for c in slug)
    slug = "_".join(slug.split()[:6])  # first 6 words
    return slug or "scene"


def load_pipeline(model_id: str = VIDEO_MODEL_ID):
    """Load and return the Wan 2.2 video generation pipeline.

    Downloads the model on first run. Subsequent runs use the HuggingFace cache.
    """
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import (
        UniPCMultistepScheduler,
    )

    logger.info(f"Loading video model: {model_id}")
    logger.info("(First run downloads the model weights)")

    # Load VAE in float32 (required for quality)
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    )

    # Load the full pipeline
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=DTYPE)

    # Use UniPC scheduler with flow shift
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=VIDEO_FLOW_SHIFT
    )

    # Enable memory optimization — offloads to CPU when not in use
    pipe.enable_model_cpu_offload()

    logger.info("Model loaded successfully.")
    return pipe


def generate_video(
    prompt: str,
    image: Image.Image | Path | None = None,
    output_dir: Path | None = None,
    model_id: str = VIDEO_MODEL_ID,
    num_frames: int = VIDEO_NUM_FRAMES,
    height: int = VIDEO_HEIGHT,
    width: int = VIDEO_WIDTH,
    num_inference_steps: int = VIDEO_NUM_INFERENCE_STEPS,
    guidance_scale: float = VIDEO_GUIDANCE_SCALE,
    seed: int | None = None,
) -> Path:
    """Generate video frames from a text prompt (and optional image).

    Args:
        prompt: Text description of the scene to generate.
        image: Optional input image for image-to-video mode. Can be a PIL Image
               or a path to an image file.
        output_dir: Where to save frames. Defaults to output/<slug>/frames/
        model_id: HuggingFace model ID. Defaults to Wan2.2-TI2V-5B.
        num_frames: Number of video frames to generate.
        height: Video height in pixels.
        width: Video width in pixels.
        num_inference_steps: Diffusion steps (more = better, slower).
        guidance_scale: How closely to follow the prompt.
        seed: Random seed for reproducibility. None = random.

    Returns:
        Path to the directory containing the saved PNG frames.
    """
    # Set up output directory
    slug = _slugify(prompt)
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / slug
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Load input image if provided as path
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    # Resize image to match video dimensions if provided
    if image is not None:
        image = image.resize((width, height))

    # Load model
    pipe = load_pipeline(model_id=model_id)

    # Set seed if provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    # Generate video
    mode = "image+text → video" if image is not None else "text → video"
    logger.info(f'[{mode}] Generating {num_frames} frames for: "{prompt}"')
    logger.info(f"Resolution: {width}x{height}, steps: {num_inference_steps}")

    # Build generation kwargs
    gen_kwargs = dict(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    # Add image if provided (image-to-video mode)
    if image is not None:
        gen_kwargs["image"] = image

    result = pipe(**gen_kwargs)

    # Extract frames — Wan returns frames as a list of PIL images
    frames = result.frames[0]
    logger.info(f"Generated {len(frames)} frames.")

    # Save each frame as PNG
    import numpy as np

    for i, frame in enumerate(frames):
        if not isinstance(frame, Image.Image):
            # Convert float32 arrays to uint8
            if hasattr(frame, "numpy"):
                frame = frame.numpy()
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            # Squeeze any extra dimensions (e.g. batch dim)
            frame = np.squeeze(frame)
            frame = Image.fromarray(frame)
        frame_path = frames_dir / f"frame_{i:04d}.png"
        frame.save(frame_path)

    logger.info(f"Saved {len(frames)} frames to {frames_dir}")

    # Also export as mp4 for easy preview
    try:
        from diffusers.utils import export_to_video

        video_path = output_dir / "preview.mp4"
        export_to_video(frames, str(video_path))
        logger.info(f"Preview video saved to {video_path}")
    except Exception as e:
        logger.warning(f"Could not export preview video: {e}")

    # Save the prompt for reference
    (output_dir / "prompt.txt").write_text(prompt)

    return frames_dir


def main():
    """CLI entry point for standalone video generation."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate video from text prompt")
    parser.add_argument("prompt", help="Text description of the scene")
    parser.add_argument("--image", type=Path, default=None, help="Input image for I2V")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--model", default=VIDEO_MODEL_ID, help="Model ID")
    parser.add_argument("--frames", type=int, default=VIDEO_NUM_FRAMES)
    parser.add_argument("--height", type=int, default=VIDEO_HEIGHT)
    parser.add_argument("--width", type=int, default=VIDEO_WIDTH)
    parser.add_argument("--steps", type=int, default=VIDEO_NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance", type=float, default=VIDEO_GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    frames_dir = generate_video(
        prompt=args.prompt,
        image=args.image,
        output_dir=args.output,
        model_id=args.model,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )
    print(f"\nDone! Frames saved to: {frames_dir}")


if __name__ == "__main__":
    main()
