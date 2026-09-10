"""Tiện ích audio dùng chung: ffmpeg, hash, đọc/ghi wav."""

import hashlib
import subprocess


def sha1_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ffmpeg_to_wav(src: str, dst: str, sr: int, mono: bool = True) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src,
           "-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-f", "wav", dst]
    subprocess.run(cmd, check=True, capture_output=True)


def ffmpeg_cut(src: str, dst: str, start_s: float, dur_s: float) -> None:
    """Cắt [start, start+dur) giây của src -> dst (wav, giữ nguyên sr/kênh)."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start_s:.3f}",
           "-t", f"{dur_s:.3f}", "-i", src, "-f", "wav", dst]
    subprocess.run(cmd, check=True, capture_output=True)


def ffmpeg_concat(parts: list[str], dst: str) -> None:
    """Nối các wav cùng định dạng theo thứ tự -> dst (concat demuxer, không re-encode)."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in parts:
            f.write("file '" + os.path.abspath(p).replace("'", "'\\''") + "'\n")
        lst = f.name
    try:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
               "-i", lst, "-c", "copy", dst]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        os.remove(lst)


def ffmpeg_loudnorm(src: str, dst: str, target_lufs: float, sr: int) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src,
           "-af", f"loudnorm=I={target_lufs}:TP=-2:LRA=11",
           "-ar", str(sr), "-ac", "1", dst]
    subprocess.run(cmd, check=True, capture_output=True)


def load_wav(path: str, sr: int | None = None):
    """Đọc wav thành mảng float32 mono. Trả (samples, sample_rate)."""
    import numpy as np
    import soundfile as sf

    x, file_sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr is not None and file_sr != sr:
        x = resample(x, file_sr, sr)
        file_sr = sr
    return x, file_sr


def resample(x, src_sr: int, dst_sr: int):
    import numpy as np

    if src_sr == dst_sr:
        return x
    n_out = int(round(len(x) * dst_sr / src_sr))
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x).astype("float32")


def save_wav(path: str, x, sr: int) -> None:
    import soundfile as sf

    sf.write(path, x, sr)


def duration_seconds(path: str) -> float:
    import soundfile as sf

    info = sf.info(path)
    return info.frames / info.samplerate
