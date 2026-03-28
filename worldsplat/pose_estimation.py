"""Step 2: Video frames → Camera poses using MASt3R.

MASt3R estimates where the camera was in 3D space for each frame.
It outputs COLMAP-format poses that gsplat can consume directly.

Setup requires:
    git clone --recursive https://github.com/naver/mast3r /tmp/mast3r
    pip install -e /tmp/mast3r
    pip install -e /tmp/mast3r/dust3r

License note: MASt3R is CC BY-NC-SA 4.0 (non-commercial use only).
"""

import logging
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from worldsplat.config import DEVICE, POSE_MODEL_ID

logger = logging.getLogger(__name__)


def _check_mast3r_installed():
    """Check that MASt3R and DUSt3R are installed, with a helpful error if not."""
    try:
        import mast3r  # noqa: F401
        import dust3r  # noqa: F401
    except ImportError:
        raise ImportError(
            "MASt3R is not installed. Set it up with:\n\n"
            "  git clone --recursive https://github.com/naver/mast3r /tmp/mast3r\n"
            "  pip install -e /tmp/mast3r\n"
            "  pip install -e /tmp/mast3r/dust3r\n"
        )


def _load_model(model_id=POSE_MODEL_ID, device=DEVICE):
    """Load the MASt3R model.

    Prefers a local .pth checkpoint (avoids HuggingFace from_pretrained bugs).
    Falls back to HuggingFace if no local file is found.
    """
    from mast3r.model import AsymmetricMASt3R

    # Check for local checkpoint first (avoids HF config compatibility issues)
    local_paths = [
        Path("/workspace/hf_cache/checkpoints") / f"{model_id.split('/')[-1]}.pth",
        Path("checkpoints") / f"{model_id.split('/')[-1]}.pth",
    ]

    for local_path in local_paths:
        if local_path.is_file():
            logger.info(f"Loading MASt3R from local checkpoint: {local_path}")
            from mast3r.model import load_model as mast3r_load_model
            model = mast3r_load_model(str(local_path), device=device)
            logger.info("MASt3R model loaded.")
            return model

    # Fall back to HuggingFace
    logger.info(f"Loading MASt3R model from HuggingFace: {model_id}")
    model = AsymmetricMASt3R.from_pretrained(model_id).to(device)
    logger.info("MASt3R model loaded.")
    return model


def _load_frames(frames_dir, size=512, subsample=1):
    """Load video frames as MASt3R-compatible image list.

    Args:
        frames_dir: Directory containing frame_0000.png, frame_0001.png, etc.
        size: Max image dimension for MASt3R (512 is the default/trained resolution).
        subsample: Take every Nth frame (1 = all frames).

    Returns:
        Tuple of (images list, frame_paths list).
    """
    from dust3r.utils.image import load_images

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if len(frame_paths) < 2:
        raise ValueError(f"Need at least 2 frames in {frames_dir}, found {len(frame_paths)}")

    # Subsample if requested (useful for 81-frame videos — every 2nd or 3rd frame)
    if subsample > 1:
        frame_paths = frame_paths[::subsample]
        logger.info(f"Subsampled to {len(frame_paths)} frames (every {subsample}th)")

    logger.info(f"Loading {len(frame_paths)} frames at size={size}")
    images = load_images([str(p) for p in frame_paths], size=size)
    return images, frame_paths


def _run_inference(model, images, scene_graph="swin-5", device=DEVICE, batch_size=1):
    """Run MASt3R inference on image pairs and perform global alignment.

    Args:
        model: Loaded MASt3R model.
        images: List of images from load_images().
        scene_graph: Pair selection strategy. "swin-N" = sliding window of N
                     (good for video). "complete" = all pairs (expensive).
        device: torch device.
        batch_size: Inference batch size.

    Returns:
        Optimized Scene object with camera poses and 3D points.
    """
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference

    # Create image pairs using sliding window (efficient for sequential video frames)
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    logger.info(f"Created {len(pairs)} image pairs (scene_graph={scene_graph})")

    # Run MASt3R inference on all pairs
    logger.info("Running MASt3R inference...")
    output = inference(pairs, model, device, batch_size=batch_size, verbose=True)

    # Global alignment: jointly optimize all camera poses and 3D structure
    mode = GlobalAlignerMode.PointCloudOptimizer if len(images) > 2 else GlobalAlignerMode.PairViewer
    logger.info(f"Running global alignment (mode={mode.name})...")

    scene = global_aligner(output, device=device, mode=mode)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init="mst",
            niter=300,
            schedule="cosine",
            lr=0.01,
        )
        logger.info(f"Global alignment converged (final loss={loss:.4f})")

    return scene


def _scene_to_colmap(scene, frame_paths, output_dir, original_size=None):
    """Extract camera parameters from MASt3R scene and write COLMAP format.

    MASt3R's get_im_poses() returns camera-to-world (cam2world) matrices.
    COLMAP uses world-to-camera, so we invert them.

    MASt3R intrinsics are for the downscaled images (default 512px max dim).
    We scale them back to the original resolution if original_size is provided.

    Args:
        scene: Optimized MASt3R Scene object.
        frame_paths: List of original frame file paths.
        output_dir: Where to write COLMAP files.
        original_size: (width, height) of original frames, for intrinsic scaling.

    Returns:
        Path to the COLMAP output directory (containing cameras.txt, images.txt, points3D.txt).
    """
    colmap_dir = Path(output_dir) / "sparse"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    # Extract from scene
    poses_c2w = scene.get_im_poses().detach().cpu().numpy()  # (N, 4, 4) cam-to-world
    focals = scene.get_focals().detach().cpu().numpy().squeeze()  # (N,) focal lengths
    imgs = scene.imgs  # list of numpy arrays (H, W, 3)

    # Get the resolution MASt3R used internally
    internal_h, internal_w = imgs[0].shape[:2]

    # Compute scale factor if we know the original resolution
    if original_size is not None:
        orig_w, orig_h = original_size
        scale = max(orig_w, orig_h) / max(internal_w, internal_h)
    else:
        # Try to read original size from the first frame
        from PIL import Image
        orig_img = Image.open(frame_paths[0])
        orig_w, orig_h = orig_img.size
        scale = max(orig_w, orig_h) / max(internal_w, internal_h)

    logger.info(
        f"Internal resolution: {internal_w}x{internal_h}, "
        f"original: {orig_w}x{orig_h}, scale factor: {scale:.2f}"
    )

    # --- cameras.txt ---
    # Use SIMPLE_PINHOLE model: params = [f, cx, cy]
    cameras_path = colmap_dir / "cameras.txt"
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i in range(len(frame_paths)):
            focal = float(focals[i]) * scale
            cx = (orig_w / 2.0)
            cy = (orig_h / 2.0)
            f.write(f"{i + 1} SIMPLE_PINHOLE {orig_w} {orig_h} {focal:.6f} {cx:.6f} {cy:.6f}\n")

    logger.info(f"Wrote {len(frame_paths)} cameras to {cameras_path}")

    # --- images.txt ---
    # Convert cam2world → world2cam, then rotation matrix → quaternion
    images_path = colmap_dir / "images.txt"
    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(len(frame_paths)):
            # Invert: cam2world → world2cam
            c2w = poses_c2w[i]
            w2c = np.linalg.inv(c2w)

            R_mat = w2c[:3, :3]
            t_vec = w2c[:3, 3]

            # scipy returns [x, y, z, w], COLMAP wants [w, x, y, z]
            quat_xyzw = Rotation.from_matrix(R_mat).as_quat()
            qw, qx, qy, qz = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]

            name = frame_paths[i].name
            f.write(
                f"{i + 1} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                f"{t_vec[0]:.10f} {t_vec[1]:.10f} {t_vec[2]:.10f} "
                f"{i + 1} {name}\n"
            )
            f.write("\n")  # empty line for 2D points (not needed for gsplat)

    logger.info(f"Wrote {len(frame_paths)} image poses to {images_path}")

    # --- points3D.txt ---
    # Extract 3D points and confidence masks from the scene
    pts3d_list = scene.get_pts3d()
    masks = scene.get_masks()

    points3d_path = colmap_dir / "points3D.txt"
    with open(points3d_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")

        point_id = 1
        # Sample points from each view (take confident points only)
        for view_idx in range(len(pts3d_list)):
            pts = pts3d_list[view_idx].detach().cpu().numpy()  # (H, W, 3)
            mask = masks[view_idx].detach().cpu().numpy()  # (H, W)
            img = (imgs[view_idx] * 255).astype(np.uint8)  # (H, W, 3)

            # Flatten and filter by confidence
            pts_flat = pts[mask]
            rgb_flat = img[mask]

            # Subsample points to keep file size reasonable (max ~5000 per view)
            n_pts = len(pts_flat)
            if n_pts > 5000:
                indices = np.random.choice(n_pts, 5000, replace=False)
                pts_flat = pts_flat[indices]
                rgb_flat = rgb_flat[indices]

            for j in range(len(pts_flat)):
                x, y, z = pts_flat[j]
                r, g, b = rgb_flat[j]
                f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0\n")
                point_id += 1

    logger.info(f"Wrote {point_id - 1} 3D points to {points3d_path}")

    return colmap_dir


def estimate_poses(frames_dir, output_dir, subsample=2, scene_graph="swin-5"):
    """Estimate camera poses from a directory of video frames.

    This is the main entry point. It:
    1. Loads video frames
    2. Runs MASt3R pairwise inference
    3. Globally aligns all cameras
    4. Writes COLMAP-format output for gsplat

    Args:
        frames_dir: Path to directory containing frame_0000.png, frame_0001.png, etc.
        output_dir: Where to write COLMAP-format output.
        subsample: Take every Nth frame (default 2 — 81 frames → 41 frames).
                   Reduces computation while keeping good coverage.
        scene_graph: Pair matching strategy. "swin-5" = sliding window of 5
                     (good for video). Increase for more overlap.

    Returns:
        Path to the COLMAP output directory (sparse/).
    """
    _check_mast3r_installed()

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)

    # Load frames
    images, frame_paths = _load_frames(frames_dir, subsample=subsample)
    logger.info(f"Loaded {len(images)} frames for pose estimation")

    # Load model and run inference + global alignment
    model = _load_model()
    scene = _run_inference(model, images, scene_graph=scene_graph)

    # Extract poses and write COLMAP format
    colmap_dir = _scene_to_colmap(scene, frame_paths, output_dir)

    logger.info(f"Pose estimation complete. COLMAP output at: {colmap_dir}")
    return colmap_dir
