"""Pipeline orchestrator: chains all steps together.

Text → Video → Poses → Gaussian Splats → Viewer

Currently Step 1 (video) is implemented. Steps 2-4 are stubs that will be
filled in during subsequent development sessions.
"""

import logging
from pathlib import Path

from worldsplat.config import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)


def run_pipeline(
    prompt: str,
    output_dir: Path | None = None,
    fast: bool = False,
    seed: int | None = None,
) -> Path:
    """Run the full text-to-3D pipeline.

    Args:
        prompt: Text description of the scene.
        output_dir: Where to save all outputs.
        fast: If True, use DiffSplat fast path (skips video + pose steps).
        seed: Random seed for reproducibility.

    Returns:
        Path to the output directory containing the final .ply file.
    """
    from worldsplat.video_gen import _slugify

    slug = _slugify(prompt)
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    if fast:
        logger.info("Fast mode: using DiffSplat (not yet implemented)")
        # TODO: Implement DiffSplat fast path
        raise NotImplementedError(
            "DiffSplat fast path is not yet implemented. "
            "Run without --fast to use the standard pipeline."
        )

    # --- Step 1: Text → Video frames ---
    logger.info("=" * 60)
    logger.info("STEP 1/4: Generating video frames...")
    logger.info("=" * 60)
    from worldsplat.video_gen import generate_video

    frames_dir = generate_video(prompt=prompt, output_dir=output_dir, seed=seed)
    logger.info(f"Video frames saved to: {frames_dir}")

    # --- Step 2: Video frames → Camera poses ---
    logger.info("=" * 60)
    logger.info("STEP 2/4: Estimating camera poses...")
    logger.info("=" * 60)
    poses_dir = estimate_poses(frames_dir, output_dir)

    # --- Step 3: Posed frames → Gaussian Splats ---
    logger.info("=" * 60)
    logger.info("STEP 3/4: Building Gaussian Splats...")
    logger.info("=" * 60)
    ply_path = build_splats(frames_dir, poses_dir, output_dir)

    # --- Step 4: Launch viewer ---
    logger.info("=" * 60)
    logger.info("STEP 4/4: Launching viewer...")
    logger.info("=" * 60)
    launch_viewer(ply_path)

    return output_dir


def estimate_poses(frames_dir: Path, output_dir: Path) -> Path:
    """Step 2: Estimate camera poses from video frames using MASt3R."""
    from worldsplat.pose_estimation import estimate_poses as _estimate_poses

    return _estimate_poses(frames_dir, output_dir)


def build_splats(frames_dir: Path, poses_dir: Path, output_dir: Path) -> Path:
    """Step 3: Build Gaussian Splats from posed frames using gsplat.

    TODO: Implement in splat_builder.py
    """
    raise NotImplementedError(
        "Gaussian Splat reconstruction (gsplat) is not yet implemented.\n"
        "This will be added in a future development session."
    )


def launch_viewer(ply_path: Path):
    """Step 4: Launch web viewer for the generated Gaussian Splats.

    TODO: Implement in viewer.py
    """
    raise NotImplementedError(
        "Web viewer is not yet implemented.\n"
        "For now, open the .ply file in SuperSplat: https://superspl.at/editor"
    )
