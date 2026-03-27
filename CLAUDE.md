# WorldSplat

Open-source text-to-3D world generation pipeline. The "open-source World Labs Marble."

## Pipeline

```
Text prompt → Wan 2.2 video → MASt3R poses → gsplat reconstruction → web viewer
```

## Architecture

- `worldsplat/video_gen.py` — Step 1: Text/image → video frames via Wan2.2-TI2V-5B (HuggingFace Diffusers)
- `worldsplat/pose_estimation.py` — Step 2: Frames → COLMAP poses via MASt3R
- `worldsplat/splat_builder.py` — Step 3: STUB. Posed frames → Gaussian Splats via gsplat
- `worldsplat/viewer.py` — Step 4: STUB. Web viewer via antimatter15/splat
- `worldsplat/pipeline.py` — Orchestrator chaining steps 1-4
- `worldsplat/cli.py` — CLI: `worldsplat video "prompt"`, `worldsplat pose <frames_dir>`, and `worldsplat generate "prompt"`
- `worldsplat/config.py` — Model IDs, defaults, device settings

## Key decisions

- **Video model:** Wan2.2-TI2V-5B (not CogVideoX). Best VBench score (84.7%), supports both text→video AND image→video, fits RTX 4090 24GB.
- **Pose estimation:** MASt3R (CC BY-NC-SA 4.0, non-commercial). Outputs COLMAP format. 10-50x faster than traditional SfM.
- **3D reconstruction:** gsplat (MIT). Standard Gaussian Splatting library.
- **Viewer:** antimatter15/splat (pure WebGL, zero deps).

## Current status

- Phase 1 scaffold: COMPLETE (all files created)
- Video generation module: COMPLETE (untested — needs GPU)
- Pose estimation: COMPLETE (untested — needs GPU + MASt3R install)
- 3D reconstruction: STUB
- Web viewer: STUB
- End-to-end pipeline: wired but steps 2-4 raise NotImplementedError

## RunPod setup

- Pod `jq16nivc6tcx31` exists (RTX 3090, $0.22/hr) — may need to be recreated
- RunPod MCP server configured in `.mcp.json`
- Use RunPod MCP tools to manage pods directly

## Development workflow

1. Code locally, push to git
2. Clone/pull on RunPod pod
3. `pip install -e .` on the pod
4. Test with `worldsplat video "prompt"`

## Plan file

Full implementation plan at `.claude/plans/soft-jumping-lemur.md`

## Next steps (in order)

1. Test video generation on RunPod (run `worldsplat video "a narrow Tokyo alley"`)
2. Test pose estimation on RunPod (run `worldsplat pose output/<scene>/frames/`)
3. Implement Gaussian Splat reconstruction (gsplat) in `splat_builder.py`
4. Implement web viewer in `viewer.py`
5. Wire end-to-end pipeline
6. Add DiffSplat fast path (`--fast` flag)
