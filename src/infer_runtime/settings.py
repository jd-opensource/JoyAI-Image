from __future__ import annotations

import os
from dataclasses import dataclass

from infer_runtime.checkpoints import resolve_checkpoint_layout

MINIMAX_BASE_URL = "https://api.minimax.io/v1"


def _is_minimax_model(model: str) -> bool:
    return model.lower().startswith("minimax")


@dataclass
class InferSettings:
    config_path: str
    ckpt_path: str
    rewrite_model: str
    openai_api_key: str | None
    openai_base_url: str | None
    default_seed: int


def load_settings(
    *,
    ckpt_root: str,
    config_path: str | None = None,
    rewrite_model: str | None = None,
    default_seed: int = 42,
) -> InferSettings:
    layout = resolve_checkpoint_layout(ckpt_root)
    default_config = layout.root / 'infer_config.py'
    if config_path is None and not default_config.exists():
        raise FileNotFoundError(
            f"Missing inference config: {default_config}. Pass --config explicitly to choose a config file."
        )

    rewrite_model = rewrite_model or 'gpt-5'
    api_key = os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('OPENAI_BASE_URL')

    # Auto-configure MiniMax when a MiniMax model is requested.
    # MINIMAX_API_KEY takes priority; base URL defaults to the MiniMax OpenAI-compatible endpoint.
    if _is_minimax_model(rewrite_model):
        minimax_key = os.environ.get('MINIMAX_API_KEY')
        if minimax_key:
            api_key = minimax_key
        base_url = base_url or MINIMAX_BASE_URL

    return InferSettings(
        config_path=config_path or str(default_config),
        ckpt_path=str(layout.transformer_ckpt),
        rewrite_model=rewrite_model,
        openai_api_key=api_key,
        openai_base_url=base_url,
        default_seed=default_seed,
    )
