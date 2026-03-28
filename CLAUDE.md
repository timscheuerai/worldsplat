# WorldSplat

Open-source text-to-3D world generation pipeline. The "open-source World Labs Marble."

## Pipeline

```
Text prompt → Wan 2.2 video → MASt3R poses → gsplat reconstruction → web viewer
```

## Architecture

- `worldsplat/video_gen.py` — Step 1: Text/image → video frames via Wan2.2-TI2V-5B
- `worldsplat/pose_estimation.py` — Step 2: Frames → COLMAP poses via MASt3R
- `worldsplat/splat_builder.py` — Step 3: Posed frames → Gaussian Splats via gsplat
- `worldsplat/viewer.py` — Step 4: Web viewer (GaussianSplats3D, local)
- `worldsplat/pipeline.py` — Orchestrator chaining steps 1-4
- `worldsplat/cli.py` — CLI with subcommands: video, pose, splat, view, generate

## Current status (2026-03-28)

- Phase 1 (video → poses → gsplat → viewer): COMPLETE, tested end-to-end
- Produces flat "diorama" from forward-facing video (expected limitation)
- See `learnings.md` for detailed technical learnings and architecture research

## RunPod setup

- Pods `e8s74v1ad6r64h` and `nbfrny2qemg4mk` stopped (RTX 4090, $0.59/hr)
- RunPod MCP server configured in `.mcp.json`
- Requires torch >= 2.5.1, diffusers 0.37+, transformers < 5

## Development workflow

1. Code locally, push to git
2. Clone/pull on RunPod pod
3. `pip install -e .` on the pod
4. Test with `worldsplat video "prompt"`

## Next steps: Architecture v2 (see learnings.md for full research)

SV3D is object-only (not for scenes). Revised plan:

1. **Integrate WorldGen as default** (Apache 2.0, scene-level, seconds, 10-24GB)
2. **Add DiffSplat `--fast`** (MIT, object-level, 1-2s feed-forward)
3. **Add LucidDreamer `--quality`** (CC-BY-NC-SA, scene-level, ~35min iterative)
4. **Keep current pipeline as `--video`** fallback
5. **Fix web viewer** — self-contained local renderer
