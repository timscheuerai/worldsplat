# WorldSplat

Open-source text-to-3D world generation. One command, navigable Gaussian Splat scenes.

```bash
worldsplat generate "a cozy cabin in snowy mountains"
# → Opens a navigable 3D scene in your browser
```

## What is this?

WorldSplat takes a text prompt and produces a navigable 3D Gaussian Splat scene.
It stitches together video generation, camera pose estimation, and 3D reconstruction
into a single pipeline.

```
Text prompt → AI video → Camera poses → 3D Gaussian Splats → Browser viewer
```

Think of it as an open-source alternative to [World Labs Marble](https://www.worldlabs.ai/).

## Quick start

```bash
# Install
pip install -e .

# Generate video frames from a text prompt (needs GPU)
worldsplat video "a narrow Tokyo alley at night with neon signs"

# Full pipeline: text → navigable 3D (coming soon)
worldsplat generate "a forest clearing at sunrise"
```

## Requirements

- Python 3.10+
- NVIDIA GPU with 12+ GB VRAM (24 GB recommended)
- CUDA 12.1+

Tested on RunPod with A100 and RTX 4090.

## Pipeline

| Step | What it does | Tool |
|------|-------------|------|
| 1. Video generation | Text → multi-angle video | CogVideoX-2B |
| 2. Pose estimation | Video frames → camera positions | MASt3R |
| 3. 3D reconstruction | Posed frames → Gaussian Splats | gsplat |
| 4. Viewer | Navigate the 3D scene in a browser | antimatter15/splat |

## Status

- [x] Project scaffold
- [x] Video generation (CogVideoX-2B)
- [ ] Pose estimation (MASt3R)
- [ ] 3D reconstruction (gsplat)
- [ ] Web viewer
- [ ] End-to-end pipeline
- [ ] DiffSplat fast path

## License

Apache 2.0
