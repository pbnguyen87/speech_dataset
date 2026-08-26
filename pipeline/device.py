"""Chọn thiết bị chạy: auto | cpu | cuda | mps."""

import functools


@functools.lru_cache(maxsize=None)
def resolve(device: str = "auto") -> str:
    if device in ("cpu", "cuda", "mps"):
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def for_ctranslate2(device: str) -> str:
    """faster-whisper (ctranslate2) chỉ hỗ trợ cpu/cuda — mps rơi về cpu."""
    return device if device == "cuda" else "cpu"
