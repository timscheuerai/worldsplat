# WorldSplat

**Open-source text-to-3D world generation.** One command turns a text prompt into a navigable Gaussian Splat scene you can explore in your browser.

An open-source alternative to [World Labs Marble](https://www.worldlabs.ai/).

```bash
worldsplat generate "a cozy cabin in snowy mountains"
# → Generates video → estimates poses → builds 3D splats → opens in browser
```

## How It Works

WorldSplat chains together four AI steps into a single pipeline:

```
Text prompt → Video generation → Camera pose estimation → 3D reconstruction → Browser viewer
```

| Step | What happens | Model / Tool |
|------|-------------|--------------|
| **1. Video Generation** | Your text prompt becomes a short AI-generated video showing the scene from a moving camera | [Wan 2.2 TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) |
| **2. Pose Estimation** | Each video frame gets a precise camera position and orientation estimated from the visual content | [MASt3R](https://github.com/naver/mast3r) |
| **3. 3D Reconstruction** | The posed frames are used to train a Gaussian Splat — a point-cloud-like 3D representation | [gsplat](https://github.com/nerfstudio-project/gsplat) |
| **4. Web Viewer** | The resulting `.ply` file is served locally and rendered in your browser with free camera controls | [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) |

## Installation

### Requirements

- Python 3.10+
- NVIDIA GPU with 12+ GB VRAM (24 GB recommended)
- CUDA 12.1+

Tested on RunPod with RTX 4090 and A100 GPUs.

### Install WorldSplat

```bash
git clone https://github.com/timscheuerai/worldsplat.git
cd worldsplat
pip install -e ".[all]"
```

The `[all]` extra installs dependencies for pose estimation and splat building. You can also install subsets:

```bash
pip install -e "."          # Core only (video generation)
pip install -e ".[pose]"    # + pose estimation (MASt3R)
pip install -e ".[splat]"   # + 3D reconstruction (gsplat)
pip install -e ".[all]"     # Everything
```

### Install MASt3R (pose estimation)

MASt3R must be installed separately from source:

```bash
git clone --recursive https://github.com/naver/mast3r.git
pip install -e mast3r
pip install -e mast3r/dust3r
```

## Usage

### Full Pipeline (text to 3D)

```bash
# Generate a full navigable 3D scene from a text prompt
worldsplat generate "a narrow Tokyo alley at night with neon signs"
```

This runs all four steps automatically and opens the result in your browser.

### Step-by-Step

You can also run each step individually, which is useful for debugging, re-running specific steps, or using your own input data:

```bash
# Step 1: Generate video frames from a text prompt
worldsplat video "a forest clearing at sunrise"
# Output: output/a-forest-clearing-at-sunrise/frames/

# Step 2: Estimate camera poses from frames
worldsplat pose output/a-forest-clearing-at-sunrise/frames/
# Output: output/a-forest-clearing-at-sunrise/sparse/ (COLMAP format)

# Step 3: Build Gaussian Splats from posed frames
worldsplat splat output/a-forest-clearing-at-sunrise/
# Output: output/a-forest-clearing-at-sunrise/model.ply

# Step 4: View the result in your browser
worldsplat view output/a-forest-clearing-at-sunrise/model.ply
# Opens http://localhost:8080/viewer.html
```

### Video Generation Options

```bash
# Use a reference image for image-to-video generation
worldsplat video "a cabin in the mountains" --image reference.png

# Control video parameters
worldsplat video "a city skyline" \
  --frames 33 \
  --height 480 --width 832 \
  --steps 50 \
  --guidance 5.0 \
  --seed 42

# Use a different model
worldsplat video "a garden" --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

### Pose Estimation Options

```bash
# Subsample frames (take every Nth frame)
worldsplat pose frames/ --subsample 2

# Change pair matching strategy
worldsplat pose frames/ --scene-graph swin-5
```

### Splat Building Options

```bash
# Control training iterations (more = higher quality, slower)
worldsplat splat scene_dir/ --steps 7000
```

### Viewer Options

```bash
# Serve on a different port
worldsplat view model.ply --port 9090
```

You can also view `.ply` files with external tools:
- Drag and drop into [antimatter15/splat](https://antimatter15.com/splat/)
- Open in [SuperSplat editor](https://superspl.at/editor)

## Project Structure

```
worldsplat/
├── cli.py              # CLI with subcommands: video, pose, splat, view, generate
├── config.py           # Central configuration (model IDs, defaults, device setup)
├── pipeline.py         # Orchestrator chaining all four steps
├── video_gen.py        # Step 1: Text/image → video frames via Wan 2.2
├── pose_estimation.py  # Step 2: Frames → COLMAP camera poses via MASt3R
├── splat_builder.py    # Step 3: Posed frames → Gaussian Splats via gsplat
└── viewer.py           # Step 4: Local web viewer (GaussianSplats3D)
```

## Output Structure

Each generation creates a self-contained output directory:

```
output/a-cozy-cabin-in-snowy-mountains/
├── frames/             # Video frames (frame_0000.png, frame_0001.png, ...)
├── sparse/             # COLMAP-format camera poses
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
├── model.ply           # Final Gaussian Splat model
└── viewer.html         # Self-contained viewer page
```

## Current Status

- [x] Video generation (Wan 2.2 TI2V-5B, text-to-video and image-to-video)
- [x] Camera pose estimation (MASt3R → COLMAP format)
- [x] 3D Gaussian Splat reconstruction (gsplat)
- [x] Local web viewer (GaussianSplats3D)
- [x] End-to-end pipeline orchestration
- [ ] DiffSplat fast path (`--fast` flag, ~2 second generation)
- [ ] WorldGen integration (scene-level, 10-24GB)
- [ ] LucidDreamer quality path (`--quality` flag)

## Known Limitations

- The current video-based pipeline produces **flat "diorama" style** scenes from forward-facing video. This is an inherent limitation of reconstructing 3D from a single forward-moving camera trajectory. The planned v2 architecture (WorldGen, DiffSplat, LucidDreamer) will address this with native 3D generation.
- Requires a GPU with at least 12 GB VRAM. 24 GB recommended for default settings.
- MASt3R must be installed separately from source.

## Roadmap (v2 Architecture)

The next major version will add multiple generation backends:

1. **WorldGen** (default) — Apache 2.0, scene-level generation, runs in seconds on 10-24GB VRAM
2. **DiffSplat** (`--fast`) — MIT, feed-forward object-level generation in 1-2 seconds
3. **LucidDreamer** (`--quality`) — Scene-level iterative generation, highest quality (~35 min)
4. **Current pipeline** (`--video`) — Video-based reconstruction as fallback

## License

[Apache 2.0](LICENSE)
