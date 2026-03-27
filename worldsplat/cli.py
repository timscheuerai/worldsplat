"""WorldSplat CLI — one command to generate navigable 3D worlds.

Usage:
    worldsplat video "a cabin in the mountains"     # Step 1 only: text → video frames
    worldsplat generate "a cabin in the mountains"   # Full pipeline: text → 3D scene
    worldsplat generate --fast "a cabin"             # Fast path via DiffSplat
"""

import argparse
import logging
import sys
from pathlib import Path

from worldsplat.config import (
    VIDEO_GUIDANCE_SCALE,
    VIDEO_HEIGHT,
    VIDEO_NUM_FRAMES,
    VIDEO_NUM_INFERENCE_STEPS,
    VIDEO_WIDTH,
)


def cmd_video(args):
    """Generate video frames from a text prompt (Step 1 only)."""
    from worldsplat.video_gen import generate_video

    kwargs = dict(
        prompt=args.prompt,
        image=args.image,
        output_dir=args.output,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )
    if args.model:
        kwargs["model_id"] = args.model
    frames_dir = generate_video(**kwargs)
    print(f"\nDone! Frames saved to: {frames_dir}")


def cmd_pose(args):
    """Estimate camera poses from existing video frames (Step 2 only)."""
    from worldsplat.pose_estimation import estimate_poses

    output_dir = args.output or args.frames_dir.parent
    colmap_dir = estimate_poses(
        frames_dir=args.frames_dir,
        output_dir=output_dir,
        subsample=args.subsample,
        scene_graph=args.scene_graph,
    )
    print(f"\nDone! COLMAP poses saved to: {colmap_dir}")


def cmd_generate(args):
    """Full pipeline: text → navigable 3D Gaussian Splat scene."""
    from worldsplat.pipeline import run_pipeline

    run_pipeline(
        prompt=args.prompt,
        output_dir=args.output,
        fast=args.fast,
        seed=args.seed,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="worldsplat",
        description="Open-source text-to-3D world generation.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- worldsplat video ---
    video_parser = subparsers.add_parser(
        "video", help="Generate video frames from a text prompt (Step 1 only)"
    )
    video_parser.add_argument("prompt", help="Text description of the scene")
    video_parser.add_argument(
        "--model", default=None, help="HuggingFace model ID (default: Wan2.2-TI2V-5B)"
    )
    video_parser.add_argument(
        "--image", type=Path, default=None, help="Input image for image-to-video mode"
    )
    video_parser.add_argument(
        "--output", type=Path, default=None, help="Output directory"
    )
    video_parser.add_argument(
        "--frames", type=int, default=VIDEO_NUM_FRAMES, help="Number of frames"
    )
    video_parser.add_argument(
        "--height", type=int, default=VIDEO_HEIGHT, help="Video height"
    )
    video_parser.add_argument(
        "--width", type=int, default=VIDEO_WIDTH, help="Video width"
    )
    video_parser.add_argument(
        "--steps", type=int, default=VIDEO_NUM_INFERENCE_STEPS, help="Inference steps"
    )
    video_parser.add_argument(
        "--guidance", type=float, default=VIDEO_GUIDANCE_SCALE, help="Guidance scale"
    )
    video_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    video_parser.set_defaults(func=cmd_video)

    # --- worldsplat pose ---
    pose_parser = subparsers.add_parser(
        "pose", help="Estimate camera poses from video frames (Step 2 only)"
    )
    pose_parser.add_argument(
        "frames_dir", type=Path, help="Directory containing frame_0000.png, frame_0001.png, etc."
    )
    pose_parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (defaults to parent of frames_dir)",
    )
    pose_parser.add_argument(
        "--subsample", type=int, default=2,
        help="Take every Nth frame (default 2)",
    )
    pose_parser.add_argument(
        "--scene-graph", default="swin-5",
        help="Pair matching strategy (default: swin-5)",
    )
    pose_parser.set_defaults(func=cmd_pose)

    # --- worldsplat generate ---
    gen_parser = subparsers.add_parser(
        "generate", help="Full pipeline: text → navigable 3D scene"
    )
    gen_parser.add_argument("prompt", help="Text description of the scene")
    gen_parser.add_argument(
        "--output", type=Path, default=None, help="Output directory"
    )
    gen_parser.add_argument(
        "--fast",
        action="store_true",
        help="Use DiffSplat fast path (lower quality, ~2 seconds)",
    )
    gen_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    gen_parser.set_defaults(func=cmd_generate)

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
