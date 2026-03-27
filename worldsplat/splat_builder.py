"""Step 3: Posed frames → 3D Gaussian Splats using gsplat.

Takes COLMAP-format camera poses + the original frames and trains
a 3D Gaussian Splat representation. Outputs a .ply file.

Status: STUB — will be implemented in a future session.

Setup:
    pip install gsplat
"""


def build_splats(frames_dir, poses_dir, output_dir):
    """Build Gaussian Splats from posed video frames.

    Args:
        frames_dir: Path to directory containing the PNG frames.
        poses_dir: Path to COLMAP-format poses directory.
        output_dir: Where to write the output .ply file.

    Returns:
        Path to the output .ply file.
    """
    raise NotImplementedError("gsplat reconstruction — coming in a future session.")
