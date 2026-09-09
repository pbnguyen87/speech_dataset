"""Khóa GPU liên tiến trình (chế độ stream): s1/s4/s5 chạy song song nhưng chỉ một
stage dùng GPU tại một thời điểm, tránh cộng dồn VRAM -> OOM trên một card.

Dùng fcntl.flock trên <workdir>/.gpu.lock; tiến trình chết thì hệ điều hành tự nhả khóa.
"""

import contextlib
import fcntl
import os


@contextlib.contextmanager
def gpu_lock(workdir: str, enabled: bool = True):
    if not enabled:
        yield
        return
    path = os.path.join(workdir, ".gpu.lock")
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def wants_lock(cfg: dict) -> bool:
    """Chỉ khóa khi chạy stream trên cuda và config cho phép."""
    from . import device as device_mod

    return (bool(cfg.get("stream", {}).get("_active"))
            and cfg.get("stream", {}).get("gpu_lock", True)
            and device_mod.resolve(cfg["device"]) == "cuda")
