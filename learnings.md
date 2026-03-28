# WorldSplat — Learnings & Progress

## Session 1: MVP Pipeline (2026-03-28)

### What we built
End-to-end text-to-3D pipeline: Text prompt → Wan 2.2 video → MASt3R poses → gsplat training → PLY export → web viewer.

Successfully generated a 3D Gaussian Splat of a Finnish lakehouse scene.

### What worked
- **Wan 2.2 TI2V-5B** generates high-quality video from text (33 frames, 480p, ~1.5 min on RTX 4090)
- **MASt3R** produces dense point clouds + camera poses from video frames (85K points, 17 cameras, ~1 min)
- **gsplat 1.5** trains Gaussian Splats from COLMAP data (232K Gaussians after 7000 steps, ~2 min)
- **Standard 3DGS PLY format** (with normals: nx/ny/nz) renders correctly in SuperSplat

### What didn't work / lessons learned

#### 1. Video-to-3D is fundamentally limited for full 3D scenes
**Problem:** A forward-moving video gives only ~17 views from similar angles. The resulting 3D scene is a flat "diorama" — looks okay from the original camera path but falls apart from other viewpoints.

**Root cause:** Video models don't understand 3D. They generate plausible 2D frames, and we recover 3D structure *after the fact* from parallax. With limited viewpoint diversity, depth is poorly constrained.

**Takeaway:** This approach is good for forward-facing scenes (like a streetview) but cannot create navigable 3D worlds. Need to pivot to 3D-native or multi-view approaches.

#### 2. PLY export format matters
**Problem:** First PLY export (via gsplat's `export_splats`) was missing normal properties (nx, ny, nz). The file wouldn't render in SuperSplat or other viewers.

**Fix:** Write PLY manually in standard INRIA 3DGS format: xyz, normals(zeros), f_dc, f_rest, opacity(logit), scale(log), rot(quaternion wxyz).

**Takeaway:** Always use the standard property ordering. Include nx/ny/nz even if zero. Don't rely on library export functions — write PLY manually for guaranteed compatibility.

#### 3. Training hyperparameters need scene-aware scaling
**Problem:** Means learning rate (1.6e-4) is tuned for unit-scale scenes, but our scene scale was only 0.38. Gaussians drifted far from the scene (positions in [-45, 24] range). Regularization (0.01 weight) was too aggressive — pushed opacities to near-zero.

**Fix:**
- Scale means LR by scene_scale: `1.6e-4 * scene_scale`
- Reduce regularization from 0.01 to 1e-3
- Filter outlier points before training (keep only within 10x camera sphere)

**Takeaway:** Gaussian Splatting training is sensitive to scene normalization. The means LR must scale with scene extent. Keep regularization light.

#### 4. gsplat 1.5 uses packed rasterization by default
**Problem:** `DefaultStrategy.step_post_backward()` crashed with `IndexError: tuple index out of range` because `info["radii"]` has shape `[nnz, 2]` (packed) not `[C, N]` (dense).

**Fix:** Pass `packed=True` to `strategy.step_post_backward()`.

**Takeaway:** gsplat 1.5's `rasterization()` returns packed tensors by default. Always pass `packed=True` to the strategy.

#### 5. KNN on large point clouds kills GPU
**Problem:** `torch.cdist` on 200K+ points creates a 200K×200K distance matrix. OOM on 24GB GPU.

**Fix:** Batched KNN (2048 batch, CPU). Also subsample initial points to 30K max.

**Takeaway:** Always batch KNN. Run on CPU for large point clouds. Subsample aggressively.

#### 6. MASt3R setup is fragile
**Problems encountered:**
- `img_size` passed as int not tuple → patch model.py
- HuggingFace `from_pretrained` uses wrong `head_type` → use local .pth checkpoint with `load_model()`
- Correct model is `catmlpdpt_metric`, not `catml`
- Need `git submodule update --init --recursive` for croco dependency

**Takeaway:** Always download .pth checkpoint directly. Always init submodules. Patch img_size bug.

#### 7. Diffusers version compatibility
**Problem:** diffusers 0.37.1 uses `torch._library.custom_ops._custom_op` with PEP 604 annotations (`float | None`) which torch 2.4.0 can't parse. diffusers 0.33.1 is too old for Wan 2.2 model weights.

**Fix:** Use torch 2.5.1+ with diffusers 0.37.1. Pin transformers < 5.0.

**Takeaway:** diffusers 0.37+ requires torch >= 2.5.

---

## Architecture Decision: Pivot from Video-to-3D to 3D-Native

### Why our current approach is a dead end for full 3D worlds
Our pipeline (text → video → SfM → 3DGS) recovers 3D from 2D, which fundamentally limits quality:
- Forward video = flat diorama
- No way to see behind objects
- Artifacts at scene boundaries
- This is "photogrammetry on synthetic video" — a hack, not true 3D generation

### How Marble (World Labs) likely works
- 3D-native generation: the model directly outputs 3D representations
- Not video → reconstruction, but direct 3D synthesis
- Trained on 3D data or uses architectures that enforce 3D consistency
- Exports: Gaussian Splats, meshes, depth maps, collider meshes
- Input: text, images, video, coarse 3D structures

### Better open-source approaches (ranked by fit for WorldSplat)

#### Tier 1: Best for "open Marble" (scene-level 3D)
| Method | Speed | License | Scene-level? | Key advantage |
|--------|-------|---------|-------------|---------------|
| **LucidDreamer** | Minutes | Proprietary (SNU) | Yes | Iterative inpainting + depth, expandable scenes |
| **RealmDreamer** | Minutes | Open-source | Yes | 88-95% user preference, multi-object scenes |
| **WonderWorld** | <10s | Academic | Yes | Fast scene extrapolation from single image |

#### Tier 2: Fast feed-forward (object-level, compose into scenes)
| Method | Speed | License | Key advantage |
|--------|-------|---------|---------------|
| **DiffSplat** | 1-2s | TBD | ICLR 2025 Oral, text+image → 3DGS |
| **LGM** | ~5s | MIT | High-res 512x512, multi-view input |
| **DreamGaussian** | 2min | GS license | Outputs mesh + Gaussians |

#### Tier 3: Multi-view generation (pair with reconstruction)
| Method | Speed | License | Key advantage |
|--------|-------|---------|---------------|
| **SV3D** | ~2s | Stability Community | 21 orbital views, high quality |
| **Zero123++** | Fast | Open | 4 consistent views, 5GB VRAM |
| **CAT3D** | ~1min | TBD | Flexible novel views, camera control |

#### Tier 4: SDS optimization (slow but high quality)
| Method | Speed | License | Key advantage |
|--------|-------|---------|---------------|
| **GaussianDreamer** | 15min | TBD | 21-40x faster than DreamFusion |

### Architecture v2 — Research-Backed Plan

#### Critical finding: SV3D is object-only, NOT for scenes

SV3D (Stability AI) was trained on Objaverse (synthetic objects on white backgrounds). It generates 21 orbital views of a single object at 576x576. It **cannot** generate scenes, landscapes, buildings, or multi-object compositions. This eliminates it as the default scene path.

Similarly, DiffSplat, LGM, and Splatter Image are all object-level only. None can generate environments.

#### What actually works for scene-level 3D (researched & verified)

| Approach | License | VRAM | Time | Scene? | Output | Status |
|----------|---------|------|------|--------|--------|--------|
| **WorldGen** | Apache 2.0 | 10-24 GB | Seconds | Yes | .ply GS | Active (Mar 2026) |
| **LucidDreamer** | CC-BY-NC-SA-4.0 | 15 GB | ~35 min | Yes | .ply GS | Stable |
| **DreamScene360** | Unstated | High | Minutes | Yes (360) | GS | Released |
| **RealmDreamer** | Unstated | 24 GB | ~10 hrs | Yes | GS | Too slow |
| **WonderWorld** | Unstated | 48 GB | <10s | Yes | FLAGS | Needs A100 |
| Our current pipeline | Mixed | 24 GB | ~5 min | Yes (flat) | .ply GS | Working |

#### Revised architecture

```
worldsplat generate "a Finnish lakehouse by a lake"
         │
         ├── --fast           DiffSplat (1-2s, object only, MIT)
         │                    Text → 3D Gaussians directly
         │
         ├── (default)        WorldGen (seconds, scene-level, Apache 2.0)
         │                    Text → panorama → depth → Gaussian Splats
         │
         ├── --quality        LucidDreamer (~35min, scene-level, CC-BY-NC-SA)
         │                    Text → iterative inpaint+depth → expanding GS scene
         │
         └── --video          Current pipeline (keep as fallback)
                              Text → Wan 2.2 video → MASt3R → gsplat
```

#### Key decisions (backed by research)

**Default path: WorldGen** (https://github.com/ZiYang-xie/WorldGen)
- Apache 2.0 license (most permissive)
- Text/image → panoramic image → depth → Gaussian Splat
- 10 GB low-VRAM mode, 24 GB standard mode
- Generates in seconds, not minutes
- Actively maintained (March 2026)
- Outputs standard `.ply` Gaussian Splats
- Uses FLUX.1-dev, Depth Anything, OneFormer

**Fast path: DiffSplat SD1.5** (https://github.com/chenguolin/DiffSplat)
- MIT license
- 1-2 seconds per object
- Text → 3D Gaussians (feed-forward, no optimization)
- Fits RTX 4090 with `--half_precision`
- Object-level only — clearly labeled as "object mode"
- Needs custom CUDA compilation (RaDe-GS rasterizer)

**Quality path: LucidDreamer** (https://github.com/luciddreamer-cvlab/LucidDreamer)
- CC-BY-NC-SA-4.0 (same as MASt3R)
- Iterative: generate view → estimate depth → inpaint unseen → expand scene
- ~35 minutes on A100, longer on RTX 4090
- 15 GB VRAM
- Outputs `.ply` Gaussian Splats directly
- Best for high-quality, expandable scenes

**Depth estimation: Depth Anything V2** (replaces ZoeDepth)
- Small model: Apache 2.0
- 10x faster than Marigold, higher accuracy
- Available via `transformers` pipeline (pip installable)
- Used by WorldGen and should replace ZoeDepth in LucidDreamer

**SV3D: object-only mode** (not default)
- Stability Community License (non-commercial)
- 21 orbital views at 576x576
- Camera poses are KNOWN (skip MASt3R entirely)
- Useful for `worldsplat generate --object "a vase"` but not scenes
- Fits RTX 4090 with `decoding_t=6`

### Implementation plan (priority order)

#### Phase 1: WorldGen integration (default path) — highest impact
1. Fork/vendor WorldGen into worldsplat
2. Wire up: text prompt → WorldGen → PLY output
3. Replace current default pipeline
4. Test on RunPod

#### Phase 2: DiffSplat fast path — instant gratification
1. Add DiffSplat SD1.5 as submodule/dependency
2. Implement `worldsplat generate "prompt" --fast`
3. Download weights on first run
4. Object-only, clearly labeled

#### Phase 3: LucidDreamer quality path — best scenes
1. Integrate LucidDreamer pipeline
2. Replace ZoeDepth with Depth Anything V2
3. Implement `worldsplat generate "prompt" --quality`
4. Output expanding Gaussian Splat scene

#### Phase 4: Keep current pipeline as --video fallback
1. Current Wan 2.2 → MASt3R → gsplat pipeline stays as `--video`
2. Useful for custom video input or when other paths unavailable

#### Phase 5: Web viewer fix
1. Self-contained local GS renderer (no external iframe)
2. Fix SharedArrayBuffer / CORS headers
3. Or switch to SuperSplat-based local viewer

---

## RunPod operational notes
- RTX 4090 SECURE cloud: $0.59/hr, US-NC, supports public IP
- Community machines in Spain don't support public IPs (avoid)
- Container disk doesn't persist pip installs across pod restarts
- Always `pip install scipy` after restart
- SSH via public IP works; RunPod proxy lacks PTY support
- Pod `e8s74v1ad6r64h` and `nbfrny2qemg4mk` stopped (delete when no longer needed)
