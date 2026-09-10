"""s1: tách vocal khỏi nhạc nền bằng demucs.

Chế độ (config separate.enabled):
  false : pass-through, manifest trỏ thẳng audio của s0
  true  : chạy demucs trên mọi file
  auto  : chạy demucs trên mẫu 30s giữa file; nếu tỷ lệ RMS nhạc nền/tổng
          vượt ngưỡng thì mới chạy full — tiết kiệm GPU cho file vốn sạch
"""

import os
import subprocess
import sys
import tempfile

from .. import audio_utils, device as device_mod, manifest
from ..gpulock import gpu_lock, wants_lock

STAGE = "s1_separate"
PREV = "s0_ingest"


def _run_demucs(src: str, out_dir: str, model: str, dev: str) -> str:
    """Chạy demucs two-stems, trả về đường dẫn file vocals.wav."""
    cmd = [sys.executable, "-m", "demucs.separate", "--two-stems", "vocals",
           "-n", model, "-d", dev if dev == "cuda" else "cpu", "-o", out_dir, src]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        # In stderr của demucs, nếu không CalledProcessError chỉ báo mã lỗi -> không debug được
        raise RuntimeError(f"demucs lỗi (mã {p.returncode}). Lệnh: {' '.join(cmd)}\n"
                           f"--- stderr (cuối) ---\n{p.stderr.strip()[-2000:]}")
    base = os.path.splitext(os.path.basename(src))[0]
    out = os.path.join(out_dir, model, base, "vocals.wav")
    if not os.path.exists(out):
        raise RuntimeError(f"demucs chạy xong nhưng không có {out}")
    return out


def separate_vocals(src: str, dst: str, model: str, dev: str, sr: int, chunk_s: float,
                    lock_ctx=None) -> None:
    """Tách vocal của cả file -> dst (wav mono sr). File dài hơn chunk_s được cắt thành
    từng khúc chunk_s giây, demucs chạy từng khúc rồi nối lại: demucs nạp NGUYÊN file vào
    RAM (10 giờ 44.1k stereo float32 = 12.7 GB + các stem) nên file dài bị OOM killer (mã -9).
    lock_ctx: callable trả context manager (khóa GPU) bọc mỗi lần chạy demucs."""
    import contextlib

    lock_ctx = lock_ctx or contextlib.nullcontext
    dur = audio_utils.duration_seconds(src)
    with tempfile.TemporaryDirectory() as td:
        if dur <= chunk_s:
            with lock_ctx():
                vocals = _run_demucs(src, td, model, dev)
            audio_utils.ffmpeg_to_wav(vocals, dst, sr)  # demucs xuất 44.1k stereo -> chuẩn pipeline
            return
        parts = []
        n = int(dur // chunk_s) + (1 if dur % chunk_s > 0.5 else 0)
        for i in range(n):
            piece = os.path.join(td, f"p{i:04d}.wav")
            audio_utils.ffmpeg_cut(src, piece, i * chunk_s, chunk_s)
            with lock_ctx():
                vocals = _run_demucs(piece, os.path.join(td, f"d{i:04d}"), model, dev)
            conv = os.path.join(td, f"v{i:04d}.wav")
            audio_utils.ffmpeg_to_wav(vocals, conv, sr)
            parts.append(conv)
            os.remove(piece)
            os.remove(vocals)
        audio_utils.ffmpeg_concat(parts, dst)


def _music_ratio(src: str, sample_s: int, model: str, dev: str, sr: int) -> float:
    """Cắt mẫu giữa file, tách thử, đo RMS(no_vocals)/RMS(tổng)."""
    import numpy as np

    x, _ = audio_utils.load_wav(src, sr)
    dur = len(x) / sr
    start = max(0, int((dur / 2 - sample_s / 2) * sr))
    sample = x[start:start + sample_s * sr]

    with tempfile.TemporaryDirectory() as td:
        spath = os.path.join(td, "sample.wav")
        audio_utils.save_wav(spath, sample, sr)
        _run_demucs(spath, td, model, dev)
        base = "sample"
        acc_path = os.path.join(td, model, base, "no_vocals.wav")
        acc, _ = audio_utils.load_wav(acc_path)

    total_rms = float(np.sqrt(np.mean(sample**2)) + 1e-9)
    acc_rms = float(np.sqrt(np.mean(acc**2)))
    return acc_rms / total_rms


class Worker:
    batch_n = 1
    flush_s = 0

    def __init__(self, cfg: dict, workdir: str):
        self.scfg = cfg["separate"]
        self.enabled = self.scfg["enabled"]
        self.sr = cfg["ingest"]["target_sr"]
        self.chunk_s = float(self.scfg.get("chunk_seconds", 7200))
        self.workdir = workdir
        self.lock = wants_lock(cfg)
        self.w = manifest.ManifestWriter(manifest.manifest_path(workdir, STAGE))
        if self.enabled is not False:
            self.dev = device_mod.resolve(cfg["device"])
            self.model = self.scfg["model"]
            self.out_audio = manifest.audio_dir(workdir, STAGE)
            print(f"[{STAGE}] mode={self.enabled} model={self.model} device={self.dev}")
        else:
            print(f"[{STAGE}] tắt — pass-through")

    def is_done(self, rec: dict) -> bool:
        return self.w.is_done(rec["id"])

    def ready(self, pending: list, upstream_done: bool):
        return pending, []

    def process(self, records: list[dict]) -> None:
        for i, rec in enumerate(records):
            if self.w.is_done(rec["id"]):
                continue
            if self.enabled is False:
                self.w.write({**rec, "separated": False})
                continue
            src = rec["audio_path"]
            need_sep, ratio = True, None
            if self.enabled == "auto":
                try:
                    with gpu_lock(self.workdir, self.lock):
                        ratio = _music_ratio(src, self.scfg["auto_sample_seconds"], self.model, self.dev, self.sr)
                    need_sep = ratio >= self.scfg["auto_music_ratio"]
                except Exception as e:
                    print(f"  auto-detect lỗi ({rec['id']}): {e} — chạy demucs full")

            if not need_sep:
                self.w.write({**rec, "separated": False, "music_ratio": ratio})
                print(f"  [{i+1}/{len(records)}] {rec['id']}: sạch (ratio={ratio:.3f}), bỏ qua demucs")
                continue

            dst = os.path.join(self.out_audio, f"{rec['id']}.wav")
            separate_vocals(src, dst, self.model, self.dev, self.sr, self.chunk_s,
                            lock_ctx=lambda: gpu_lock(self.workdir, self.lock))
            self.w.write({**rec, "audio_path": dst, "separated": True, "music_ratio": ratio})
            print(f"  [{i+1}/{len(records)}] {rec['id']}: đã tách vocal")

    def close(self) -> None:
        self.w.close()


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    records = manifest.read_records(manifest.manifest_path(workdir, PREV), limit)
    worker = Worker(cfg, workdir)
    print(f"[{STAGE}] {len(records)} file")
    worker.process(records)
    worker.close()
    print(f"[{STAGE}] xong")
    return manifest.manifest_path(workdir, STAGE)
