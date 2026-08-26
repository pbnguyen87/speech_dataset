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


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    lcfg = cfg["loudnorm"]
    mode = lcfg["mode"]
    sr = cfg["ingest"]["target_sr"]
    records = manifest.read_records(manifest.manifest_path(workdir, PREV))
    mpath = manifest.manifest_path(workdir, STAGE)

    if mode == "off":
        with manifest.ManifestWriter(mpath) as w:
            for rec in records:
                if not w.is_done(rec["id"]):
                    w.write({**rec, "loudnorm": "off"})
        print(f"[{STAGE}] off — pass-through")
        return mpath

    out_audio = manifest.audio_dir(workdir, STAGE)
    print(f"[{STAGE}] mode={mode}, {len(records)} segment")

    with manifest.ManifestWriter(mpath) as w:
        for i, rec in enumerate(records):
            if w.is_done(rec["id"]):
                continue
            dst = os.path.join(out_audio, os.path.basename(rec["audio_path"]))
            if mode == "lufs":
                audio_utils.ffmpeg_loudnorm(rec["audio_path"], dst,
                                            lcfg["target_lufs"], sr)
            else:  # peak
                x, file_sr = audio_utils.load_wav(rec["audio_path"])
                audio_utils.save_wav(dst, peak_normalize(x, lcfg["peak_dbfs"]), file_sr)
            w.write({**rec, "audio_path": dst, "loudnorm": mode})
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(records)}")

    print(f"[{STAGE}] xong")
    return mpath
