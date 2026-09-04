"""s0: quét thư mục raw, dedup theo hash, chuẩn hóa về WAV mono target_sr."""

import glob
import json
import os

from .. import audio_utils, manifest

STAGE = "s0_ingest"


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    raw_dir = cfg["raw_dir"]
    exts = {e.lower() for e in cfg["ingest"]["extensions"]}
    sr = cfg["ingest"]["target_sr"]

    files = sorted(
        f for f in glob.glob(os.path.join(raw_dir, "**", "*"), recursive=True)
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in exts
    )
    if limit is not None:
        files = files[:limit]
    print(f"[{STAGE}] {len(files)} file trong {raw_dir}")

    out_audio = manifest.audio_dir(workdir, STAGE)
    mpath = manifest.manifest_path(workdir, STAGE)
    seen_hashes = {r.get("hash") for r in manifest.iter_records(mpath)}

    with manifest.ManifestWriter(mpath) as w:
        for i, src in enumerate(files):
            h = audio_utils.sha1_file(src)
            file_id = h[:12]
            if w.is_done(file_id):
                continue
            if h in seen_hashes:
                print(f"  dedup: {os.path.basename(src)} trùng hash, bỏ qua")
                continue

            dst = os.path.join(out_audio, f"{file_id}.wav")
            try:
                audio_utils.ffmpeg_to_wav(src, dst, sr)
            except Exception as e:
                print(f"  LỖI convert {src}: {e}")
                continue

            # metadata từ crawler nếu có file sidecar cùng tên .json
            sidecar = os.path.splitext(src)[0] + ".json"
            source_meta = None
            if os.path.exists(sidecar):
                try:
                    with open(sidecar, encoding="utf-8") as f:
                        source_meta = json.load(f)
                except Exception:
                    pass

            rec = {
                "id": file_id,
                "hash": h,
                "source_path": os.path.relpath(src, raw_dir),
                "audio_path": dst,
                "sr": sr,
                "duration": audio_utils.duration_seconds(dst),
                "source_meta": source_meta,
            }
            seen_hashes.add(h)
            w.write(rec)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(files)}")

    n = len(manifest.read_records(mpath))
    print(f"[{STAGE}] xong: {n} file trong manifest")
    return mpath
