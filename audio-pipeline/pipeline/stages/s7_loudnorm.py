"""s7: chuẩn hóa âm lượng — lufs (ffmpeg loudnorm) | peak | off."""

import os

from .. import audio_utils, manifest

STAGE = "s7_loudnorm"
PREV = "s6_textnorm"


def peak_normalize(x, peak_dbfs: float):
    import numpy as np

    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak <= 0:
        return x
    target = 10 ** (peak_dbfs / 20)
    return (x * (target / peak)).astype("float32")


class Worker:
    batch_n = 1
    flush_s = 0

    def __init__(self, cfg: dict, workdir: str):
        self.lcfg = cfg["loudnorm"]
        self.mode = self.lcfg["mode"]
        self.sr = cfg["ingest"]["target_sr"]
        self.cleanup = bool(cfg.get("stream", {}).get("cleanup"))
        self.w = manifest.ManifestWriter(manifest.manifest_path(workdir, STAGE))
        self.out_audio = manifest.audio_dir(workdir, STAGE) if self.mode != "off" else None
        print(f"[{STAGE}] mode={self.mode}")

    def is_done(self, rec: dict) -> bool:
        return self.w.is_done(rec["id"])

    def ready(self, pending: list, upstream_done: bool):
        return pending, []

    def process(self, records: list[dict]) -> None:
        for i, rec in enumerate(records):
            if self.w.is_done(rec["id"]):
                continue
            if self.mode == "off":
                self.w.write({**rec, "loudnorm": "off"})
                continue
            src = rec["audio_path"]
            dst = os.path.join(self.out_audio, os.path.basename(src))
            if self.mode == "lufs":
                audio_utils.ffmpeg_loudnorm(src, dst, self.lcfg["target_lufs"], self.sr)
            else:  # peak
                x, file_sr = audio_utils.load_wav(src)
                audio_utils.save_wav(dst, peak_normalize(x, self.lcfg["peak_dbfs"]), file_sr)
            self.w.write({**rec, "audio_path": dst, "loudnorm": self.mode})
            if self.cleanup and os.path.exists(src):  # s7 là stage cuối đọc wav s2
                os.remove(src)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(records)}")

    def close(self) -> None:
        self.w.close()


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    records = manifest.read_records(manifest.manifest_path(workdir, PREV))
    worker = Worker(cfg, workdir)
    print(f"[{STAGE}] {len(records)} segment")
    worker.process(records)
    worker.close()
    print(f"[{STAGE}] xong")
    return manifest.manifest_path(workdir, STAGE)
