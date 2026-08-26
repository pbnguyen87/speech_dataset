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

STAGE = "s1_separate"
PREV = "s0_ingest"


def _run_demucs(src: str, out_dir: str, model: str, dev: str) -> str:
    """Chạy demucs two-stems, trả về đường dẫn file vocals.wav."""
    subprocess.run(
        [sys.executable, "-m", "demucs.separate", "--two-stems", "vocals",
         "-n", model, "-d", dev if dev == "cuda" else "cpu",
         "-o", out_dir, src],
        check=True, capture_output=True,
    )
    base = os.path.splitext(os.path.basename(src))[0]
    return os.path.join(out_dir, model, base, "vocals.wav")


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


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    scfg = cfg["separate"]
    enabled = scfg["enabled"]
    sr = cfg["ingest"]["target_sr"]
    records = manifest.read_records(manifest.manifest_path(workdir, PREV), limit)
    mpath = manifest.manifest_path(workdir, STAGE)

    if enabled is False:
        with manifest.ManifestWriter(mpath) as w:
            for rec in records:
                if not w.is_done(rec["id"]):
                    w.write({**rec, "separated": False})
        print(f"[{STAGE}] tắt — pass-through {len(records)} file")
        return mpath

    dev = device_mod.resolve(cfg["device"])
    model = scfg["model"]
    out_audio = manifest.audio_dir(workdir, STAGE)
    print(f"[{STAGE}] mode={enabled} model={model} device={dev}, {len(records)} file")

    with manifest.ManifestWriter(mpath) as w:
        for i, rec in enumerate(records):
            if w.is_done(rec["id"]):
                continue
            src = rec["audio_path"]

            need_sep = True
            ratio = None
            if enabled == "auto":
                try:
                    ratio = _music_ratio(src, scfg["auto_sample_seconds"], model, dev, sr)
                    need_sep = ratio >= scfg["auto_music_ratio"]
                except Exception as e:
                    print(f"  auto-detect lỗi ({rec['id']}): {e} — chạy demucs full")

            if not need_sep:
                w.write({**rec, "separated": False, "music_ratio": ratio})
                print(f"  [{i+1}/{len(records)}] {rec['id']}: sạch (ratio={ratio:.3f}), bỏ qua demucs")
                continue

            dst = os.path.join(out_audio, f"{rec['id']}.wav")
            with tempfile.TemporaryDirectory() as td:
                vocals = _run_demucs(src, td, model, dev)
                # demucs xuất 44.1k stereo -> đưa về chuẩn pipeline
                audio_utils.ffmpeg_to_wav(vocals, dst, sr)
            w.write({**rec, "audio_path": dst, "separated": True, "music_ratio": ratio})
            print(f"  [{i+1}/{len(records)}] {rec['id']}: đã tách vocal")

    print(f"[{STAGE}] xong")
    return mpath
