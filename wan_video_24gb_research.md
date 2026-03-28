# Running Wan 2.1/2.2 Video Generation on 24GB GPUs

Research compiled 2026-03-27. Focused on practical, tested solutions for RTX 3090/4090.

---

## 1. Model Sizes That Work on 24GB

| Model | Params | VRAM (no optimization) | VRAM (optimized) | 24GB Feasible? |
|-------|--------|----------------------|-------------------|----------------|
| Wan 2.1 T2V 1.3B | 1.3B | ~8.2 GB | ~6 GB | Yes, easily |
| Wan 2.2 TI2V 5B | 5B | ~24 GB | ~14-18 GB | Yes, with offloading |
| Wan 2.1/2.2 14B | 14B | 65-80 GB | ~13 GB (group offload) | Yes, with aggressive offloading (slow) |

**Bottom line**: The 1.3B model fits trivially. The 5B model fits with `--offload_model True --convert_model_dtype --t5_cpu`. The 14B model fits with group offloading but is slow on a single 24GB card.

---

## 2. Memory Optimization Techniques (Ranked by Effectiveness)

### A. Group Offloading (Best balance of memory/speed for diffusers)
The official diffusers docs show the 14B T2V model running in ~13GB VRAM using group offloading. This is the recommended approach.

```python
# pip install ftfy
import torch
from diffusers import AutoModel, WanPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video
from transformers import UMT5EncoderModel

text_encoder = UMT5EncoderModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16
)
vae = AutoModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae", torch_dtype=torch.float32
)
transformer = AutoModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16
)

# Group offloading: moves layer groups to CPU when not needed
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

apply_group_offloading(
    text_encoder,
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="block_level",
    num_blocks_per_group=4
)
transformer.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True  # overlaps data transfer and computation
)

pipeline = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    vae=vae,
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

prompt = "A cat walks on the grass, realistic"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=81,
    guidance_scale=5.0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
```

### B. Model CPU Offloading (Simple, moderate savings)
Moves entire models to GPU only when needed. Faster than sequential offloading.

```python
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
# Do NOT call pipe.to("cuda") -- offloading handles device placement
```

### C. Sequential CPU Offloading (Maximum savings, very slow)
Moves individual submodules. Saves the most VRAM but extremely slow.

```python
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
# Do NOT call pipe.to("cuda") first
```

### D. Layerwise Casting / FP8 (Halves transformer memory)
Stores weights in fp8, computes in bf16. ~50% memory reduction on transformer.

```python
from diffusers import AutoModel

transformer = AutoModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn,
    compute_dtype=torch.bfloat16
)
```

### E. Torchao Quantization (int8/int4 weight-only)
Requires PyTorch 2.5+ and torchao. Can use `PipelineQuantizationConfig` for pipeline-level quantization.

```python
from diffusers import DiffusionPipeline, PipelineQuantizationConfig, TorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": TorchAoConfig(Int8WeightOnlyConfig(group_size=128))}
)
pipeline = DiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

FP8 quantization is best on RTX 4090 (compute capability 8.9+). RTX 3090 does NOT support fp8 natively (compute capability 8.6) -- use int8 instead.

### F. Wan2GP (deepbeepmeep) -- All-in-one optimized runner
Wan2GP is a community project that wraps Wan models with aggressive VRAM optimizations. Supports GGUF, INT8, FP8, NV FP4, Nunchaku quantization. Claims to run with as little as 6GB VRAM.

- Repo: https://github.com/deepbeepmeep/Wan2GP
- Supports Wan 2.1/2.2, Hunyuan Video, LTX Video, Flux
- Includes Flash Attention 2 support
- GGUF provides ~15% speed gain on video diffusion models

### G. Original Wan repo flags (for non-diffusers usage)
```bash
python generate.py \
  --task t2v-1.3B \
  --size 832*480 \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --offload_model True \
  --t5_cpu \
  --sample_shift 8 \
  --sample_guide_scale 6 \
  --prompt "Your prompt"
```

For the 5B model:
```bash
python generate.py \
  --task ti2v-5B \
  --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --prompt "Your prompt"
```

---

## 3. Diffusers and PyTorch Version Compatibility

| Component | Recommended Version | Notes |
|-----------|-------------------|-------|
| PyTorch | 2.5+ (2.6+ for AutoModel) | Required for torchao quantization |
| Diffusers | Latest from main branch | `pip install git+https://github.com/huggingface/diffusers` |
| torchao | Latest | `pip install -U torchao` |
| ftfy | Any | Required dependency for Wan pipeline |
| accelerate | Latest | Required for device_map and offloading |

**Install command:**
```bash
pip install git+https://github.com/huggingface/diffusers
pip install -U torch torchao accelerate ftfy transformers
```

---

## 4. Working Code Examples

### Simplest: 1.3B T2V on 24GB (fits easily, no optimization needed)

```python
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A cat walks on the grass, realistic"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0
).frames[0]
export_to_video(output, "output.mp4", fps=15)
```

VRAM: ~8.2 GB. Generation time: ~4 min on RTX 4090.

### 5B TI2V on 24GB (with offloading)

```python
import torch
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="Two anthropomorphic cats in boxing gear fight on a spotlighted stage.",
    negative_prompt="low quality, blurred, static, deformed",
    height=704,
    width=1280,
    num_frames=121,
    guidance_scale=5.0,
    num_inference_steps=50,
).frames[0]
export_to_video(output, "output.mp4", fps=24)
```

VRAM: ~24 GB with offloading. Generation time: ~9 min on RTX 4090.

### 14B T2V on 24GB (with group offloading + fp8)

See Section 2A above for the full group offloading example. Combines:
- Group offloading for text encoder (block_level, 4 blocks per group)
- Group offloading for transformer (leaf_level with CUDA streams)
- VAE kept in float32

Result: ~13 GB VRAM for the 14B model.

---

## 5. 1.3B Quality for 3D Reconstruction

### Pros
- Only needs 8.2 GB VRAM -- trivial to run
- Fast iteration (~4 min per video)
- Decent text-prompt adherence
- Good for rapid prototyping of camera orbits

### Cons
- Motion quality is noticeably worse than 5B/14B
  - More warping artifacts
  - Less physically plausible camera movements
  - Inconsistent object geometry between frames
- 480p resolution is the sweet spot (720p trained insufficiently)
- For 3D reconstruction/Gaussian splatting, frame-to-frame consistency is critical, and the 1.3B struggles here

### Recommendation for 3D reconstruction
- **Use the 5B model minimum** for multi-view consistency
- Wan 2.2 5B fixed many of the motion artifacts from 2.1 (smoother camera movements, better physics)
- The 14B model with offloading gives the best consistency but is slow
- For WorldSplat-style pipelines, consider using the 5B with image-to-video (I2V) to anchor the first frame, ensuring geometric consistency
- **Alternative approach**: Use 1.3B for rapid prompt exploration, then re-generate final videos with 5B/14B

### Key quality research findings
- The 1.3B model's main limitation for 3D is inter-frame geometric drift
- Papers like CompSplat (2602.09816) specifically address frame-quality disparities in 3DGS from video
- Quality-guided density control and gap-aware masking can compensate for lower-quality input video

---

## 6. Resolution and Frame Count on 24GB

### Wan 2.1 T2V 1.3B
| Resolution | Frames | VRAM | Time (4090) |
|-----------|--------|------|-------------|
| 832x480 (480p) | 81 | ~8.2 GB | ~4 min |
| 1280x720 (720p) | 81 | ~12-14 GB | ~8 min |

480p is recommended for 1.3B (more stable, better trained).

### Wan 2.2 TI2V 5B
| Resolution | Frames | VRAM (w/ offload) | Time (4090) |
|-----------|--------|-------------------|-------------|
| 1280x704 (720p) | 121 | ~24 GB | ~9 min |
| 832x480 (480p) | 81 | ~14-16 GB | ~4 min |

### Wan 2.1/2.2 14B (with group offloading)
| Resolution | Frames | VRAM | Time (4090) |
|-----------|--------|------|-------------|
| 832x480 (480p) | 81 | ~13 GB | ~15-20 min |
| 1280x720 (720p) | 81 | ~18-22 GB | ~30+ min |

### Key constraints
- Frame count formula: `(num_frames - 1) % 4 == 0` (must be 5, 9, 13, 17, 21, 25... 81, 121, etc.)
- FPS: 15 fps for Wan 2.1, 24 fps for Wan 2.2 5B
- 81 frames = ~5 sec at 16fps
- 121 frames = ~5 sec at 24fps
- VAE slicing and VAE tiling are NOT supported for AutoencoderKLWan (noted in diffusers docs)

---

## 7. RunPod Setup

### Option A: One-click ComfyUI Template (Easiest)
- **Template URL**: https://get.runpod.io/wan-template
- Includes Wan 2.1 and 2.2 with pre-configured ComfyUI workflows
- Supports T2V, I2V, ControlNet, VACE, LoRA
- **Requirements**: Select a GPU pod with CUDA 12.8
- Set environment variables to `True` to trigger model downloads
- Ready in ~5 minutes after deploy
- Source: https://civitai.com/articles/11960

### Option B: Wan2GP Template
- Repo: https://github.com/ArpitKhurana-ai/wan2gp-template
- Optimized for A40 and consumer GPUs
- One-click RunPod template for Wan2GP
- CUDA 12.8 based

### Option C: Serverless Deployment
- Repo: https://github.com/lucidprogrammer/wan-video
- Production-grade serverless on RunPod
- Infrastructure-as-code setup

### Option D: ComfyUI Studio Docker
- Repo: https://github.com/diego-devita/comfyui-studio
- Docker image for RunPod with WAN 2.2, HunyuanVideo, CogVideoX support
- Pre-configured for multiple video generation models

### Option E: Manual Diffusers Setup on RunPod
```bash
# On a RunPod pod with 24GB GPU (RTX 3090/4090/A5000)
pip install git+https://github.com/huggingface/diffusers
pip install -U torch torchao accelerate ftfy transformers

# Download model
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir ./wan-1.3b

# Or for 5B
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir ./wan-5b
```

### RunPod GPU Recommendations
| GPU | VRAM | Best For |
|-----|------|----------|
| RTX 3090 | 24GB | 1.3B (easy), 5B (with offloading, slower) |
| RTX 4090 | 24GB | 1.3B (easy), 5B (with offloading), fp8 quantization |
| A5000 | 24GB | Similar to 4090 but better for serverless |
| RTX A6000 | 48GB | 14B without aggressive offloading |
| A100 80GB | 80GB | 14B at full speed, no offloading needed |

---

## Summary: Best Approach for 24GB + 3D Reconstruction

1. **Start with Wan 2.1 T2V 1.3B** for prompt exploration (8 GB, fast)
2. **Use Wan 2.2 TI2V 5B** for final video generation (24 GB with offloading)
3. Use `enable_model_cpu_offload()` for 5B model on 24GB
4. For 14B on 24GB, use group offloading with leaf_level + CUDA streams (~13 GB but slow)
5. On RTX 4090: use fp8 quantization for additional savings
6. On RTX 3090: use int8 quantization (no native fp8 support)
7. For best 3D reconstruction quality: 5B I2V at 720p with a reference first frame
8. RunPod: use the one-click template at https://get.runpod.io/wan-template

---

## Sources

- [Wan2.1 T2V 1.3B Diffusers Model Card](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
- [Wan2.2 TI2V 5B Diffusers Model Card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
- [Diffusers Memory Optimization Guide](https://huggingface.co/docs/diffusers/en/optimization/memory)
- [Diffusers Wan Pipeline Docs](https://huggingface.co/docs/diffusers/api/pipelines/wan)
- [Diffusers torchao Quantization Docs](https://huggingface.co/docs/diffusers/en/quantization/torchao)
- [Wan2GP - GPU Poor Video Generator](https://github.com/deepbeepmeep/Wan2GP)
- [Wan 2.2 VRAM Guide (Novita)](https://blogs.novita.ai/wan-2-2-vram-find-the-best-gpu-setup-for-deployment/)
- [RunPod Wan Template (Civitai)](https://civitai.com/articles/11960/wan2221-in-1-click-with-workflows-included-runpod-template)
- [ComfyUI WAN 2.2 RunPod Guide (DEV)](https://dev.to/shahtab_mohtasin_a5e8388d/complete-guide-comfyui-wan-22-on-runpod-5ge7)
- [RunPod Serverless Wan Deployment](https://github.com/lucidprogrammer/wan-video)
- [Wan2GP RunPod Template](https://github.com/ArpitKhurana-ai/wan2gp-template)
- [FP8 Wan Models (wangkanai)](https://huggingface.co/wangkanai/wan22-fp8-i2v)
- [Wan 2.2 Official Repo](https://github.com/Wan-Video/Wan2.2)
- [Wan 2.1 Official Repo](https://github.com/Wan-Video/Wan2.1)
- [CompSplat: Compression-aware 3DGS](https://arxiv.org/abs/2602.09816)
- [RTX 3090 vs 4090 vs 5090 Wan Benchmark](https://medium.com/@ttio2tech_28094/rtx-3090-vs-4090-vs-5090-gpu-speed-test-running-wan2-2-animate-34f4d4b9e3dd)
