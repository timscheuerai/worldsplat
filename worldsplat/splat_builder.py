"""Step 3: Posed frames → 3D Gaussian Splats using gsplat.

Takes COLMAP-format camera poses + the original frames and trains
a 3D Gaussian Splat representation. Outputs a .ply file.

Setup:
    pip install gsplat
"""

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from worldsplat.config import DEVICE

logger = logging.getLogger(__name__)

# SH coefficient for converting RGB to zeroth-order spherical harmonics
C0 = 0.28209479177387814


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB [0, 1] to zeroth-order SH coefficient."""
    return (rgb - 0.5) / C0


def knn(points: torch.Tensor, k: int) -> torch.Tensor:
    """Find k nearest neighbor distances for each point (batched for large N)."""
    N = points.shape[0]
    # Use CPU to avoid GPU OOM on large point clouds
    points_cpu = points.cpu()
    batch_size = 2048
    all_dists = []
    for i in range(0, N, batch_size):
        batch = points_cpu[i : i + batch_size]
        d = torch.cdist(batch, points_cpu)  # [batch, N]
        topk, _ = d.topk(k, dim=-1, largest=False)
        all_dists.append(topk)
    result = torch.cat(all_dists, dim=0)
    return result.to(points.device)


def _parse_colmap_cameras(cameras_path: Path) -> dict:
    """Parse cameras.txt → dict of camera_id → (model, width, height, params)."""
    cameras = {}
    with open(cameras_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = (model, width, height, params)
    return cameras


def _parse_colmap_images(images_path: Path) -> list:
    """Parse images.txt → list of (image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name)."""
    images = []
    with open(images_path) as f:
        lines = [l.strip() for l in f if not l.startswith("#") and l.strip()]
    # Every other line is image data (odd lines are 2D point data, which we skip)
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]
        images.append((image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name))
    return images


def _parse_colmap_points3d(points3d_path: Path) -> tuple:
    """Parse points3D.txt → (positions [N,3], colors [N,3])."""
    positions = []
    colors = []
    with open(points3d_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            positions.append([x, y, z])
            colors.append([r, g, b])
    return np.array(positions, dtype=np.float32), np.array(colors, dtype=np.uint8)


def load_colmap_data(colmap_dir: Path, frames_dir: Path, max_init_points: int = 30000) -> dict:
    """Load COLMAP text-format data and images into tensors for training.

    Returns dict with: camtoworlds, Ks, images, points, points_rgb, image_names
    """
    cameras = _parse_colmap_cameras(colmap_dir / "cameras.txt")
    images_data = _parse_colmap_images(colmap_dir / "images.txt")
    points, points_rgb = _parse_colmap_points3d(colmap_dir / "points3D.txt")

    # Subsample points if too many (KNN on 200K+ points is very slow)
    if len(points) > max_init_points:
        indices = np.random.choice(len(points), max_init_points, replace=False)
        points = points[indices]
        points_rgb = points_rgb[indices]
        logger.info(f"Subsampled initial points: {len(points)}")

    # Sort by image name for consistency
    images_data.sort(key=lambda x: x[9])

    camtoworlds = []
    Ks = []
    image_list = []
    image_names = []
    widths = []
    heights = []

    for img in images_data:
        image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = img

        # Build world-to-camera matrix from quaternion + translation
        R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        t_w2c = np.array([tx, ty, tz])
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        c2w = np.linalg.inv(w2c)
        camtoworlds.append(c2w)

        # Camera intrinsics
        model, width, height, params = cameras[camera_id]
        if model == "SIMPLE_PINHOLE":
            f, cx, cy = params[0], params[1], params[2]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
        elif model == "PINHOLE":
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported camera model: {model}")
        Ks.append(K)
        widths.append(width)
        heights.append(height)

        # Load image
        img_path = frames_dir / name
        if img_path.exists():
            pil_img = Image.open(img_path).convert("RGB")
            image_list.append(np.array(pil_img))
            image_names.append(name)
        else:
            logger.warning(f"Image not found: {img_path}")

    camtoworlds = np.stack(camtoworlds)  # [N, 4, 4]
    Ks = np.stack(Ks)  # [N, 3, 3]

    # Normalize the scene: center at camera centroid, scale so cameras fit in unit sphere
    camera_positions = camtoworlds[:, :3, 3]
    scene_center = camera_positions.mean(axis=0)
    camtoworlds[:, :3, 3] -= scene_center
    points -= scene_center

    dists = np.linalg.norm(camtoworlds[:, :3, 3], axis=1)
    scene_scale = dists.max()
    camtoworlds[:, :3, 3] /= scene_scale
    points /= scene_scale

    logger.info(f"Scene scale: {scene_scale:.4f}, centered at {scene_center}")

    return {
        "camtoworlds": torch.from_numpy(camtoworlds).float(),
        "Ks": torch.from_numpy(Ks).float(),
        "images": [torch.from_numpy(img).float() / 255.0 for img in image_list],
        "widths": widths,
        "heights": heights,
        "points": torch.from_numpy(points).float(),
        "points_rgb": torch.from_numpy(points_rgb).float() / 255.0,
        "image_names": image_names,
        "scene_scale": scene_scale,
    }


def init_gaussians(
    points: torch.Tensor,
    points_rgb: torch.Tensor,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    sh_degree: int = 3,
    device: str = "cuda",
) -> tuple:
    """Initialize Gaussian parameters from a point cloud.

    Returns (splats dict, optimizers dict).
    """
    N = points.shape[0]

    # Initialize scales from KNN distances
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N]

    # Spherical harmonics: zeroth order from RGB, rest zeros
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(points_rgb)

    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(points.clone()),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(colors[:, :1, :].clone()),
            "shN": torch.nn.Parameter(colors[:, 1:, :].clone()),
        }
    ).to(device)

    # Learning rates (from gsplat defaults)
    lr_config = {
        "means": 1.6e-4,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }

    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr, "name": name}],
            eps=1e-15,
            betas=(0.9, 0.999),
        )
        for name, lr in lr_config.items()
    }

    return splats, optimizers


def train_splats(
    data: dict,
    max_steps: int = 2000,
    sh_degree: int = 3,
    ssim_lambda: float = 0.2,
    device: str = "cuda",
) -> torch.nn.ParameterDict:
    """Train 3D Gaussian Splats from COLMAP data.

    This is a simplified training loop focused on getting good results
    from video-generated scenes (forward-facing, moderate motion).
    """
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy

    camtoworlds = data["camtoworlds"].to(device)
    Ks = data["Ks"].to(device)
    images = [img.to(device) for img in data["images"]]
    width = data["widths"][0]
    height = data["heights"][0]

    # Initialize from point cloud
    logger.info(f"Initializing {len(data['points'])} Gaussians from point cloud...")
    splats, optimizers = init_gaussians(
        data["points"], data["points_rgb"],
        sh_degree=sh_degree, device=device,
    )
    logger.info(f"Initial Gaussians: {len(splats['means'])}")

    # Densification strategy
    strategy = DefaultStrategy()
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=1.0)

    # LR scheduler for means
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
    )

    num_images = len(images)
    pbar = tqdm(range(max_steps), desc="Training splats")

    for step in pbar:
        # Random image
        idx = step % num_images
        c2w = camtoworlds[idx : idx + 1]  # [1, 4, 4]
        K = Ks[idx : idx + 1]  # [1, 3, 3]
        pixels = images[idx].unsqueeze(0)  # [1, H, W, 3]

        # SH degree schedule
        sh_degree_to_use = min(step // 1000, sh_degree)

        # Render
        means = splats["means"]
        quats = splats["quats"]
        scales = torch.exp(splats["scales"])
        opacities = torch.sigmoid(splats["opacities"])
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)  # [N, K, 3]

        renders, alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(c2w),
            Ks=K,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
        )

        # Pre-backward densification
        strategy.step_pre_backward(
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
        )

        # Loss: L1 + SSIM
        l1loss = F.l1_loss(renders, pixels)

        # Simple SSIM (permute to NCHW for F.conv2d-based SSIM)
        renders_nchw = renders.permute(0, 3, 1, 2)
        pixels_nchw = pixels.permute(0, 3, 1, 2)
        ssimloss = 1.0 - _ssim(renders_nchw, pixels_nchw)

        loss = (1.0 - ssim_lambda) * l1loss + ssim_lambda * ssimloss

        # Regularization
        loss = loss + 0.01 * torch.sigmoid(splats["opacities"]).mean()
        loss = loss + 0.01 * torch.exp(splats["scales"]).mean()

        loss.backward()

        pbar.set_description(
            f"loss={loss.item():.4f} l1={l1loss.item():.4f} "
            f"GS={len(splats['means'])}"
        )

        # Optimizer step
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        scheduler.step()

        # Post-backward densification
        strategy.step_post_backward(
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
            packed=False,
        )

    logger.info(f"Training complete. Final Gaussians: {len(splats['means'])}")
    return splats


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute SSIM between two images in NCHW format."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    channel = img1.shape[1]

    # Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    window = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    window = window.repeat(channel, 1, 1, 1)
    window = window / window.sum()

    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def export_ply(splats: torch.nn.ParameterDict, output_path: Path):
    """Export trained Gaussians to PLY format compatible with viewers."""
    try:
        from gsplat import export_splats
        export_splats(
            means=splats["means"],
            scales=splats["scales"],
            quats=splats["quats"],
            opacities=splats["opacities"],
            sh0=splats["sh0"],
            shN=splats["shN"],
            format="ply",
            save_to=str(output_path),
        )
    except ImportError:
        # Fallback: manual PLY export
        _export_ply_manual(splats, output_path)

    logger.info(f"Exported {len(splats['means'])} Gaussians to {output_path}")


def _export_ply_manual(splats: torch.nn.ParameterDict, output_path: Path):
    """Manual PLY export (fallback if gsplat.export_splats is unavailable)."""
    means = splats["means"].detach().cpu().numpy()
    scales = splats["scales"].detach().cpu().numpy()
    quats = splats["quats"].detach().cpu().numpy()
    opacities = splats["opacities"].detach().cpu().numpy()
    sh0 = splats["sh0"].detach().cpu().numpy()  # [N, 1, 3]
    shN = splats["shN"].detach().cpu().numpy()  # [N, K, 3]

    N = means.shape[0]

    # Normalize quaternions
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)

    # Build PLY header
    sh_coeffs = np.concatenate([sh0.reshape(N, -1), shN.reshape(N, -1)], axis=-1)
    n_sh = sh_coeffs.shape[1]

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {N}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    for i in range(3):
        header += f"property float scale_{i}\n"
    header += "property float opacity\n"
    header += "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
    for i in range(n_sh):
        header += f"property float f_rest_{i}\n" if i >= 3 else f"property float f_dc_{i}\n"
    header += "end_header\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header.encode())
        for i in range(N):
            # position
            f.write(means[i].astype(np.float32).tobytes())
            # scale (log space)
            f.write(scales[i].astype(np.float32).tobytes())
            # opacity (logit space)
            f.write(np.array([opacities[i]], dtype=np.float32).tobytes())
            # rotation quaternion [w, x, y, z]
            q = quats[i]
            f.write(np.array([q[0], q[1], q[2], q[3]], dtype=np.float32).tobytes())
            # SH coefficients
            f.write(sh_coeffs[i].astype(np.float32).tobytes())


def build_splats(
    frames_dir: Path,
    poses_dir: Path,
    output_dir: Path,
    max_steps: int = 2000,
    sh_degree: int = 3,
    device: str = DEVICE,
) -> Path:
    """Build Gaussian Splats from posed video frames.

    Args:
        frames_dir: Path to directory containing the PNG frames.
        poses_dir: Path to COLMAP-format poses directory (sparse/).
        output_dir: Where to write the output .ply file.
        max_steps: Number of training iterations.
        sh_degree: Spherical harmonics degree.
        device: Compute device.

    Returns:
        Path to the output .ply file.
    """
    frames_dir = Path(frames_dir)
    poses_dir = Path(poses_dir)
    output_dir = Path(output_dir)

    # Load COLMAP data
    logger.info("Loading COLMAP data and images...")
    data = load_colmap_data(poses_dir, frames_dir)
    logger.info(
        f"Loaded {len(data['images'])} images, "
        f"{len(data['points'])} initial points"
    )

    # Train
    splats = train_splats(
        data,
        max_steps=max_steps,
        sh_degree=sh_degree,
        device=device,
    )

    # Export PLY
    ply_path = output_dir / "model.ply"
    export_ply(splats, ply_path)

    return ply_path
