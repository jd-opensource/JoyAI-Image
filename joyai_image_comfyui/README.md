# JoyAI-Image-Edit (Diffusers) for ComfyUI

**Important** 

We are working on the native integration of JoyAI-Image-Edit (rather than custom nodes). You can use the native nodes by installing ComfyUI from our PR: https://github.com/feice-huang/ComfyUI/tree/joyimage-edit-pr.
Workflow: https://github.com/user-attachments/files/28871922/workflow_joyimage_edit.json
The weights on Hugging Face have also been updated: https://huggingface.co/jdopensource/JoyAI-Image-Edit-ComfyUI


### Introduction

This is a ComfyUI integration of JoyAI-Image-Edit that uses HuggingFace Diffusers as the backend. It follows the qwen-image-edit file-loading convention: each model component is picked from a single-file checkpoint inside the standard ComfyUI model folders (`diffusion_models/`, `text_encoders/`, `vae/`), with built-in manual CPU offload for low-VRAM environments.

**Features**:
- ✨ Image Editing powered by `JoyImageEditPipeline` from Diffusers
- ✨ Standard ComfyUI single-file checkpoint loading (`diffusion_models` / `text_encoders` / `vae`)
- ✨ Manual phase-by-phase CPU offload for minimal VRAM usage
- ✨ Plug-and-play workflow

### Quick Start

#### 1. Requirements

- **diffusers**: >=0.39.0.dev0 (with `JoyImageEditPipeline` support; not yet released as of 0.38.0 — install from source: `pip install git+https://github.com/huggingface/diffusers.git`)
- **transformers**: >=4.57.0

#### 2. Installation Steps

**Step 1: Copy the Node Package**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jd-opensource/JoyAI-Image.git
cp -r JoyAI-Image/joyai_image_comfyui ./
rm -rf JoyAI-Image
```

**Step 2: Download Model Weights**

Download the single-file checkpoints from [Hugging Face](https://huggingface.co/jdopensource/JoyAI-Image-Edit-ComfyUI) and place each component into its corresponding ComfyUI standard folder:

```
ComfyUI/models/
├── diffusion_models/
│   └── joy_image_edit_bf16.safetensors        # transformer
├── text_encoders/
│   └── qwen3vl_joyimage_bf16.safetensors      # text encoder
└── vae/
    └── joy_image_edit_vae.safetensors         # VAE
```

Each loader node automatically lists the `.safetensors` files in its corresponding folder, so you can pick the component you want from a dropdown.

**Step 3: Restart ComfyUI**

ComfyUI will automatically discover the new nodes under `loaders/joyai` and `image/joyai`.

### Node Reference

This package provides 4 nodes:

| Node | Display Name | Category | Description |
|------|-------------|----------|-------------|
| `JoyAIImageEditUNETLoader` | Load JoyAI Diffusion Model | loaders/joyai | Loads the `JoyImageEditTransformer3DModel` from a `.safetensors` file in `ComfyUI/models/diffusion_models/` |
| `JoyAIImageEditCLIPLoader` | Load JoyAI CLIP | loaders/joyai | Loads the Qwen3VL text encoder from `ComfyUI/models/text_encoders/` and bundles it with the tokenizer + processor |
| `JoyAIImageEditVAELoader` | Load JoyAI VAE | loaders/joyai | Loads `AutoencoderKLWan` from a `.safetensors` file in `ComfyUI/models/vae/` |
| `JoyAIImageEditPipeline` | JoyAI Image Edit Pipeline | image/joyai | Assembles the pipeline and runs inference |

The loader nodes emit custom socket types `JOY_MODEL`, `JOY_CLIP`, `JOY_VAE` (kept separate from ComfyUI core `MODEL/CLIP/VAE` because the underlying classes — `JoyImageEditTransformer3DModel`, `Qwen3VLForConditionalGeneration`, `AutoencoderKLWan` — do not match ComfyUI's core model APIs).

#### Pipeline Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `prompt` | STRING | `""` | — | Text instruction describing the desired edit |
| `height` | INT | `0` | 0–2048 | Output height in pixels (0 = auto from input image, snaps to nearest bucket) |
| `width` | INT | `0` | 0–2048 | Output width in pixels (0 = auto from input image, snaps to nearest bucket) |
| `steps` | INT | `40` | 1–200 | Number of denoising steps |
| `guidance_scale` | FLOAT | `4.0` | 0.0–30.0 | Classifier-free guidance scale (>1.0 enables CFG) |
| `num_images_per_prompt` | INT | `1` | 1–8 | Number of images to generate per prompt |
| `seed` | INT | `0` | 0–2⁶⁴ | Random seed (control widget supports fixed/increment/decrement/randomize) |
| `cpu_offload` | BOOLEAN | `True` | — | Enable manual phase-by-phase CPU offload |

#### CPU Offload Strategy

When `cpu_offload=True`, the pipeline manually manages GPU memory in 4 phases:

1. **Text encoding** — text_encoder on GPU, encode prompt + negative prompt, then offload to CPU
2. **VAE encode** — VAE on GPU, encode reference image to latents, then offload to CPU
3. **Denoising** — transformer on GPU, run the full denoising loop, then offload to CPU
4. **VAE decode** — VAE on GPU, decode final latents to image, then offload to CPU

At any point, only one large model is on GPU.
