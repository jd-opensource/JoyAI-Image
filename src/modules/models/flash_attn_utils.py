"""Loader for the Flash Attention symbols used by the attention backend.

Building Flash Attention from source is slow. To avoid that, we first try to
load pre-built Flash Attention 3 binaries served through the Hugging Face
``kernels`` library. When pre-built binaries are not available for the current
platform (unsupported GPU/arch, no network, ...), we transparently fall back to
a source build / local installation of Flash Attention (the ``FLASH_ATTN3_PATH``
override or the installed ``flash-attn`` package), so the dependency stays
optional.

Inspired by https://github.com/OpenDriveLab/AgiBot-World/pull/159
"""

import logging
import os
import sys
from functools import lru_cache

logger = logging.getLogger(__name__)

# Kernel served by the `kernels` library that mirrors the Flash Attention 3 API.
_FLASH_ATTN_KERNEL = "kernels-community/flash-attn3"
_FLASH_ATTN_KERNEL_VERSION = 1


@lru_cache(maxsize=None)
def load_flash_attn():
    """Return a dict of Flash Attention symbols, or an empty dict if unavailable.

    The returned dict exposes the callable the attention backend relies on:
    ``flash_attn_varlen_func``.

    Resolution order:
      1. Pre-built FA3 binaries via the ``kernels`` library (no compilation).
      2. A source build / local installation of Flash Attention
         (the original behaviour, including the ``FLASH_ATTN3_PATH`` override).
    """
    symbols = _load_from_kernels()
    if symbols is not None:
        return symbols

    symbols = _load_from_source()
    if symbols is not None:
        return symbols

    return {}


def _load_from_kernels():
    try:
        from kernels import get_kernel

        module = get_kernel(_FLASH_ATTN_KERNEL, version=_FLASH_ATTN_KERNEL_VERSION)
        # The flash-attn3 kernel exposes its callables at the top level of the
        # module (unlike flash-attn2, which nests them under
        # `flash_attention_interface`).
        logger.info(
            "Loaded pre-built Flash Attention 3 binaries via `kernels` (%s).",
            _FLASH_ATTN_KERNEL,
        )
        return {
            "flash_attn_varlen_func": module.flash_attn_varlen_func,
        }
    except Exception as exc:  # noqa: BLE001 - any failure should trigger the source fallback
        logger.info(
            "Pre-built Flash Attention via `kernels` unavailable (%s); "
            "falling back to a source build.",
            exc,
        )
        return None


def _load_from_source():
    try:
        # Check for a Flash Attention 3 installation path first.
        flash_attn3_path = os.getenv("FLASH_ATTN3_PATH")
        if flash_attn3_path:
            logger.info("Using Flash Attention 3 from: %s", flash_attn3_path)
            sys.path.insert(0, flash_attn3_path)
            from flash_attn_interface import flash_attn_varlen_func
        else:
            from flash_attn.flash_attn_interface import flash_attn_varlen_func

        return {
            "flash_attn_varlen_func": flash_attn_varlen_func,
        }
    except ImportError:
        return None
