# Adapted from https://github.com/hao-ai-lab/FastVideo/tree/main/fastvideo/attention

import torch
from einops import rearrange

from modules.models.flash_attn_utils import load_flash_attn

# Flash Attention is loaded through the `kernels` library (pre-built FA2
# binaries, no local compilation), with a transparent fallback to a source
# build / local installation of `flash-attn`. See `flash_attn_utils.py`.
_FLASH_ATTN_IMPORT_ERROR = None
try:
    _flash_attn_symbols = load_flash_attn()
    flash_attn_varlen_func = _flash_attn_symbols.get("flash_attn_varlen_func")
    if flash_attn_varlen_func is None:
        raise ImportError("flash_attn is not available via `kernels` or a source build.")
except Exception as exc:  # noqa: BLE001
    flash_attn_varlen_func = None
    _FLASH_ATTN_IMPORT_ERROR = exc



def is_flash_attn_available() -> bool:
    return flash_attn_varlen_func is not None


def get_preferred_attention_backend() -> str:
    return "flash_attn" if is_flash_attn_available() else "torch_spda"


def describe_attention_backend() -> str:
    backend = get_preferred_attention_backend()
    if backend == "flash_attn":
        return "flash_attn"
    if _FLASH_ATTN_IMPORT_ERROR is None:
        return "torch_spda"
    return f"torch_spda (flash_attn unavailable: {_FLASH_ATTN_IMPORT_ERROR})"


def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1],
                             dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: str = "flash_attn",
    *,
    causal: bool = False,
    softmax_scale: float = None,
    attn_kwargs: dict = None,
):
    """
    Args:
        q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
        v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads
    """
    assert backend in ["flash_attn"], f"Unsupported attention backend: {backend}"
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Input tensors must be 4D"
    batch_size = q.shape[0]

    cu_seqlens_q = attn_kwargs['cu_seqlens_q']
    cu_seqlens_kv = attn_kwargs['cu_seqlens_kv']
    max_seqlen_q = attn_kwargs['max_seqlen_q']
    max_seqlen_kv = attn_kwargs['max_seqlen_kv']
    x = flash_attn_varlen_func(
        q.view(q.shape[0] * q.shape[1], *q.shape[2:]),
        k.view(k.shape[0] * k.shape[1], *k.shape[2:]),
        v.view(v.shape[0] * v.shape[1], *v.shape[2:]),
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
    )
    output = x.view(
        batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
    )

    return output
