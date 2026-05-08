# JoyAI-Image-Edit (Diffusers) for ComfyUI

### Introduction

This is an alternative ComfyUI integration of JoyAI-Image-Edit that uses HuggingFace Diffusers as the backend. It supports the official Diffusers-format weights with built-in manual CPU offload for low-VRAM environments.

**Features**:
- ✨ Image Editing powered by `JoyImageEditPipeline` from Diffusers
- ✨ Manual phase-by-phase CPU offload for minimal VRAM usage
- ✨ No dependency on JoyAI-Image source code — only `diffusers` + `transformers`
- ✨ Plug-and-play Workflow

### Quick Start

#### 1. Requirements

- **diffusers**: >=0.39.0.dev0 (with JoyImageEditPipeline support; not yet released as of 0.38.0 — install from source: `pip install git+https://github.com/huggingface/diffusers.git`)
- **transformers**: >=4.57.0

#### 2. Installation Steps

**Step 1: Copy the Node Package**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jd-opensource/JoyAI-Image.git
cp -r JoyAI-Image/joyai_image_diffusers_comfyui ./
rm -rf JoyAI-Image
```

**Step 2: Download Model Weights**

Download the Diffusers-format model weights from [Hugging Face](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Diffusers) and place them in the `ComfyUI/models/diffusers/` directory:

```
ComfyUI/models/diffusers/
└── JoyAI-Image-Edit-Diffusers/
    ├── model_index.json
    ├── transformer/
    ├── text_encoder/
    ├── tokenizer/
    ├── processor/
    ├── vae/
    └── scheduler/
```

**Step 3: Restart ComfyUI**

ComfyUI will automatically load the new nodes.

#### 3. Using the Workflow

1. Open ComfyUI
2. Drag `joyai-image-edit-workflow.json` into the ComfyUI canvas, or use Menu → Load to import it
3. Select your model path in each loader node
4. Connect an input image and set your prompt
5. Click **Run** 

### Node Reference

This package provides 4 nodes:

| Node | Display Name | Category | Description |
|------|-------------|----------|-------------|
| `JoyAIImageEditDiffusersTransformerLoader` | Load JoyAI Transformer (Diffusers) | loaders/joyai | Loads the JoyImageEditTransformer3DModel |
| `JoyAIImageEditDiffusersTextEncoderLoader` | Load JoyAI Text Encoder (Diffusers) | loaders/joyai | Loads Qwen3VL text encoder, tokenizer, and processor |
| `JoyAIImageEditDiffusersVAELoader` | Load JoyAI VAE (Diffusers) | loaders/joyai | Loads AutoencoderKLWan |
| `JoyAIImageEditDiffusersPipeline` | JoyAI Image Edit Pipeline (Diffusers) | image/joyai | Assembles the pipeline and runs inference |

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
