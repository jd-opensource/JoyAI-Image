import os

import folder_paths
import numpy as np
import torch

from PIL import Image

import comfy.model_management


def _list_diffusers_models():
    models = []
    for p in folder_paths.get_folder_paths("diffusers"):
        if not os.path.isdir(p):
            continue
        for name in sorted(os.listdir(p)):
            if os.path.isdir(os.path.join(p, name)):
                models.append(name)
    return models


def _resolve_model(name):
    for p in folder_paths.get_folder_paths("diffusers"):
        candidate = os.path.join(p, name)
        if os.path.isdir(candidate):
            return candidate
    return name


class JoyAIImageEditTransformerLoader:
    """Load JoyAI Image Edit Transformer, auto-detecting bf16 or NF4."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (_list_diffusers_models(),),
            }
        }

    RETURN_TYPES = ("JOYEDIT_TRANSFORMER",)
    RETURN_NAMES = ("transformer",)
    FUNCTION = "load_transformer"
    CATEGORY = "loaders/joyai"

    def load_transformer(self, model_path):
        from diffusers import JoyImageEditTransformer3DModel

        transformer = JoyImageEditTransformer3DModel.from_pretrained(
            _resolve_model(model_path),
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        return (transformer,)


class JoyAIImageEditTextEncoderLoader:
    """Load JoyAI Text Encoder (Qwen3VL) with tokenizer and processor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (_list_diffusers_models(),),
            }
        }

    RETURN_TYPES = ("JOYEDIT_TEXT_ENCODER", "JOYEDIT_TOKENIZER", "JOYEDIT_PROCESSOR")
    RETURN_NAMES = ("text_encoder", "tokenizer", "processor")
    FUNCTION = "load_text_encoder"
    CATEGORY = "loaders/joyai"

    def load_text_encoder(self, model_path):
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        resolved = _resolve_model(model_path)
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            resolved, subfolder="text_encoder", torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            resolved, subfolder="tokenizer",
        )
        processor = Qwen3VLProcessor.from_pretrained(
            resolved, subfolder="processor",
        )
        return (text_encoder, tokenizer, processor)


class JoyAIImageEditVAELoader:
    """Load JoyAI VAE (AutoencoderKLWan)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (_list_diffusers_models(),),
            }
        }

    RETURN_TYPES = ("JOYEDIT_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders/joyai"

    def load_vae(self, model_path):
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(
            _resolve_model(model_path),
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        return (vae,)


class JoyAIImageEditPipeline:
    """Assemble JoyAI Image Edit pipeline and run inference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer": ("JOYEDIT_TRANSFORMER",),
                "text_encoder": ("JOYEDIT_TEXT_ENCODER",),
                "tokenizer": ("JOYEDIT_TOKENIZER",),
                "processor": ("JOYEDIT_PROCESSOR",),
                "vae": ("JOYEDIT_VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64, "tooltip": "Output height in pixels (0 = auto from input image, or 1024 for t2i)"}),
                "width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64, "tooltip": "Output width in pixels (0 = auto from input image, or 1024 for t2i)"}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 200}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1},
                ),
                "num_images_per_prompt": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True},
                ),
                "cpu_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "image/joyai"

    def run(self, transformer, text_encoder, tokenizer, processor, vae,
            prompt, height, width, steps, guidance_scale, num_images_per_prompt, seed, cpu_offload, image=None):
        from diffusers import FlowMatchEulerDiscreteScheduler, JoyImageEditPipeline
        from diffusers.pipelines.joyimage.pipeline_joyimage_edit import retrieve_timesteps

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=1.5,
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=4096,
            time_shift_type="exponential",
        )

        pipe = JoyImageEditPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            scheduler=scheduler,
            vae=vae,
        )

        gpu = comfy.model_management.get_torch_device()
        cpu = torch.device("cpu")

        pil_image = None
        if image is not None:
            img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np).convert("RGB")

        target_h = height if height > 0 else None
        target_w = width if width > 0 else None
        height, width = pipe.vae_image_processor.get_default_height_width(pil_image, target_h, target_w)
        processed_image = None
        if pil_image is not None:
            processed_image = pipe.vae_image_processor.resize_center_crop(pil_image, (height, width))

        do_cfg = guidance_scale > 1.0

        if not cpu_offload:
            pipe.to(gpu)
            with torch.inference_mode():
                output = pipe(
                    image=pil_image,
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=torch.manual_seed(seed),
                )
            results = []
            for img in output.images:
                result_np = np.array(img).astype(np.float32) / 255.0
                results.append(torch.from_numpy(result_np))
            return (torch.stack(results),)

        with torch.inference_mode():
            # Phase 1: Text encoding (whole-module offload)
            pipe.text_encoder.to(gpu)
            if processed_image is not None:
                prompt_embeds, _ = pipe.encode_prompt_multiple_images(
                    prompt=prompt,
                    images=processed_image,
                    device=gpu,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=4096,
                )
                if do_cfg:
                    negative_prompt_embeds, _ = pipe.encode_prompt_multiple_images(
                        prompt="",
                        images=processed_image,
                        device=gpu,
                        num_images_per_prompt=num_images_per_prompt,
                        max_sequence_length=4096,
                    )
            else:
                prompt_embeds, _ = pipe.encode_prompt(
                    prompt=prompt,
                    device=gpu,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=4096,
                )
                if do_cfg:
                    negative_prompt_embeds, _ = pipe.encode_prompt(
                        prompt="",
                        device=gpu,
                        num_images_per_prompt=num_images_per_prompt,
                        max_sequence_length=4096,
                    )
            prompt_embeds = prompt_embeds.to(cpu)
            if do_cfg:
                negative_prompt_embeds = negative_prompt_embeds.to(cpu)
            pipe.text_encoder.to(cpu)
            torch.cuda.empty_cache()

            # Phase 2: Prepare latents (VAE encode reference if i2i)
            num_channels_latents = pipe.transformer.config.in_channels
            generator = torch.Generator(device=cpu).manual_seed(int(seed))
            if processed_image is not None:
                pipe.vae.to(gpu)
            noise_latents, image_latents = pipe.prepare_latents(
                batch_size=num_images_per_prompt,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                video_length=1,
                dtype=prompt_embeds.dtype,
                device=gpu,
                generator=generator,
                image=[processed_image] if processed_image is not None else None,
            )
            if image_latents is not None:
                latents = torch.cat([image_latents, noise_latents], dim=1)
                image_latents = image_latents.to(cpu)
            else:
                latents = noise_latents
            latents = latents.to(cpu)
            if processed_image is not None:
                pipe.vae.to(cpu)
                torch.cuda.empty_cache()

            # Phase 3: Denoising (whole-module offload)
            pipe.transformer.to(gpu)
            latents = latents.to(gpu)
            prompt_embeds = prompt_embeds.to(gpu)
            if image_latents is not None:
                image_latents = image_latents.to(gpu)
            if do_cfg:
                negative_prompt_embeds = negative_prompt_embeds.to(gpu)

            timesteps_list, _ = retrieve_timesteps(pipe.scheduler, steps, gpu)

            for t in timesteps_list:
                if image_latents is not None:
                    latents[:, :1] = image_latents
                t_expand = t.repeat(latents.shape[0])

                noise_pred = pipe.transformer(
                    hidden_states=latents,
                    timestep=t_expand,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

                if do_cfg:
                    noise_pred_uncond = pipe.transformer(
                        hidden_states=latents,
                        timestep=t_expand,
                        encoder_hidden_states=negative_prompt_embeds,
                        return_dict=False,
                    )[0]
                    comb = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                    cond_norm = torch.norm(noise_pred, dim=2, keepdim=True)
                    noise_norm = torch.norm(comb, dim=2, keepdim=True)
                    noise_pred = comb * (cond_norm / noise_norm.clamp_min(1e-6))

                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            latents = latents.to(cpu)
            del prompt_embeds
            if image_latents is not None:
                del image_latents
            if do_cfg:
                del negative_prompt_embeds
            pipe.transformer.to(cpu)
            torch.cuda.empty_cache()

            # Phase 4: VAE decode (whole-module offload)
            pipe.vae.to(gpu)
            latents_flat = latents.to(gpu).flatten(0, 1)
            latents_flat = pipe.denormalize_latents(latents_flat)
            decoded = pipe.vae.decode(latents_flat, return_dict=False)[0]
            decoded = decoded.float().unflatten(0, (num_images_per_prompt, -1))
            decoded = decoded.permute(0, 1, 3, 2, 4, 5)[:, -1].squeeze(1)
            decoded = pipe.image_processor.postprocess(decoded, output_type="pil")
            pipe.vae.to(cpu)
            torch.cuda.empty_cache()

        results = []
        for img in decoded:
            result_np = np.array(img).astype(np.float32) / 255.0
            results.append(torch.from_numpy(result_np))
        return (torch.stack(results),)


NODE_CLASS_MAPPINGS = {
    "JoyAIImageEditDiffusersTransformerLoader": JoyAIImageEditTransformerLoader,
    "JoyAIImageEditDiffusersTextEncoderLoader": JoyAIImageEditTextEncoderLoader,
    "JoyAIImageEditDiffusersVAELoader": JoyAIImageEditVAELoader,
    "JoyAIImageEditDiffusersPipeline": JoyAIImageEditPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyAIImageEditDiffusersTransformerLoader": "Load JoyAI Transformer (Diffusers)",
    "JoyAIImageEditDiffusersTextEncoderLoader": "Load JoyAI Text Encoder (Diffusers)",
    "JoyAIImageEditDiffusersVAELoader": "Load JoyAI VAE (Diffusers)",
    "JoyAIImageEditDiffusersPipeline": "JoyAI Image Edit Pipeline (Diffusers)",
}
