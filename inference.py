"""Local inference entrypoint for the clean JoyAI-Image release."""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import torch
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

warnings.filterwarnings('ignore')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run local inference without FastAPI.')
    parser.add_argument('--ckpt-root', required=True, help='Checkpoint root.')
    parser.add_argument('--prompt', required=True, help='Edit prompt or T2I prompt.')
    parser.add_argument('--image', help='Optional input image path for image editing.')
    parser.add_argument('--output', default='example.png', help='Output image path.')
    parser.add_argument('--height', type=int, default=1024, help='Only used for text-to-image inference.')
    parser.add_argument('--width', type=int, default=1024, help='Only used for text-to-image inference.')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--neg-prompt', default='')
    parser.add_argument('--basesize', type=int, default=1024, help='Resize bucket base size for image editing inputs.')
    parser.add_argument('--rewrite-prompt', action='store_true')
    parser.add_argument('--config', help='Optional config path. Defaults to <ckpt-root>/infer_config.py.')
    parser.add_argument(
        '--rewrite-model',
        default='gpt-5',
        help=(
            'Model for prompt rewriting (default: gpt-5). '
            'Supports any OpenAI-compatible model, e.g. MiniMax-M2.7. '
            'Set MINIMAX_API_KEY to use MiniMax models automatically.'
        ),
    )
    parser.add_argument('--hsdp-shard-dim', type=int, help='Override config hsdp_shard_dim for multi-GPU FSDP inference.')
    return parser.parse_args()


def load_input_image(image_path: str | None) -> Image.Image | None:
    if not image_path:
        return None
    return Image.open(image_path).convert('RGB')


def is_rank0() -> bool:
    return int(os.environ.get('RANK', '0')) == 0


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device('cpu')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    return torch.device(f'cuda:{local_rank}')


def main() -> None:
    args = parse_args()

    from infer_runtime.model import InferenceParams, build_model
    from infer_runtime.settings import load_settings
    from modules.utils import maybe_init_distributed, clean_dist_env
    from modules.models.attention import describe_attention_backend

    dist_initialized = False
    try:
        settings = load_settings(
            ckpt_root=args.ckpt_root,
            config_path=args.config,
            rewrite_model=args.rewrite_model,
            default_seed=args.seed,
        )
        device = resolve_device()
        dist_initialized = maybe_init_distributed()

        if is_rank0():
            print(f'Chosen device: {device}')
            print(f'Attention backend: {describe_attention_backend()}')
            print(f'Config path: {settings.config_path}')
            print(f'Checkpoint path: {settings.ckpt_path}')
            if args.hsdp_shard_dim is not None:
                print(f'Override hsdp_shard_dim: {args.hsdp_shard_dim}')

        model = build_model(
            settings,
            device=device,
            hsdp_shard_dim_override=args.hsdp_shard_dim,
        )
        input_image = load_input_image(args.image)
        effective_prompt = model.maybe_rewrite_prompt(args.prompt, input_image, args.rewrite_prompt)

        start_time = time.time()
        output_image = model.infer(
            InferenceParams(
                prompt=effective_prompt,
                image=input_image,
                height=args.height,
                width=args.width,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                neg_prompt=args.neg_prompt,
                basesize=args.basesize,
            )
        )
        elapsed = time.time() - start_time

        if is_rank0():
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_image.save(output_path)
            print(f'Prompt used: {effective_prompt}')
            print(f'Saved output: {output_path}')
            print(f'Time taken: {elapsed:.2f} seconds')
    finally:
        if dist_initialized:
            clean_dist_env()


if __name__ == '__main__':
    main()
