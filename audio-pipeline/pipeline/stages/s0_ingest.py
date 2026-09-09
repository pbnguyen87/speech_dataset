"""s0: quét thư mục raw, dedup theo hash, chuẩn hóa về WAV mono target_sr.

Worker: dùng chung cho chế độ lô (run) và stream (pipeline.stream). Ở chế độ stream
thư mục raw được quét lặp lại; file đủ điều kiện khi có sidecar .json cùng tên (crawler
ghi sidecar ngay sau khi tải xong) hoặc mtime đã cũ hơn settle_seconds.
"""

import glob
import json
import os
import time

from .. import audio_utils, manifest

STAGE = "s0_ingest"


class Worker:
    batch_n = 1
    flush_s = 0

    def __init__(self, cfg: dict, workdir: str):
        self.raw_dir = cfg["raw_dir"]
        self.exts = {e.lower() for e in cfg["ingest"]["extensions"]}
        self.sr = cfg["ingest"]["target_sr"]
        self.cleanup = bool(cfg.get("stream", {}).get("cleanup"))
        self.settle_s = float(cfg.get("stream", {}).get("settle_seconds", 60))
        self.streaming = bool(cfg.get("stream", {}).get("_active"))
        self.out_audio = manifest.audio_dir(workdir, STAGE)
        mpath = manifest.manifest_path(workdir, STAGE)
        self.w = manifest.ManifestWriter(mpath)
        self.seen_hashes = {r.get("hash") for r in manifest.iter_records(mpath)}
        # đường dẫn đã xử lý (kể cả trùng hash / lỗi convert) -> không quét lại
        self.skipped_path = os.path.join(workdir, STAGE, "skipped.jsonl")
        self.seen_paths = {r["source_path"] for r in manifest.iter_records(mpath)}
        self.seen_paths |= {r["source_path"] for r in manifest.iter_records(self.skipped_path)}

    # ---- nguồn item: đường dẫn file
    def scan(self, include_done: bool = False) -> list[str]:
        now = time.time()
        files = []
        for f in glob.glob(os.path.join(self.raw_dir, "**", "*"), recursive=True):
            if not os.path.isfile(f) or os.path.splitext(f)[1].lower() not in self.exts:
                continue
            if not include_done and self.is_done(f):
                continue
            if self.streaming and not self._settled(f, now):
                continue
            files.append(f)
        return sorted(files)

    def _settled(self, path: str, now: float) -> bool:
        if os.path.exists(os.path.splitext(path)[0] + ".json"):
            return True
        try:
            return now - os.path.getmtime(path) > self.settle_s
        except OSError:
            return False

    def is_done(self, path: str) -> bool:
        return os.path.relpath(path, self.raw_dir) in self.seen_paths

    def ready(self, pending: list, upstream_done: bool):
        return pending, []

    def _skip(self, src: str, reason: str) -> None:
        rel = os.path.relpath(src, self.raw_dir)
        self.seen_paths.add(rel)
        with open(self.skipped_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"source_path": rel, "reason": reason}, ensure_ascii=False) + "\n")

    def process(self, files: list[str]) -> None:
        for i, src in enumerate(files):
            if self.is_done(src):
                continue
            h = audio_utils.sha1_file(src)
            file_id = h[:12]
            if self.w.is_done(file_id) or h in self.seen_hashes:
                print(f"  dedup: {os.path.basename(src)} trùng hash, bỏ qua")
                self._skip(src, "dup")
                continue

            dst = os.path.join(self.out_audio, f"{file_id}.wav")
            try:
                audio_utils.ffmpeg_to_wav(src, dst, self.sr)
            except Exception as e:
                print(f"  LỖI convert {src}: {e}")
                self._skip(src, f"convert: {e}")
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

            rel = os.path.relpath(src, self.raw_dir)
            rec = {
                "id": file_id,
                "hash": h,
                "source_path": rel,
                "audio_path": dst,
                "sr": self.sr,
                "duration": audio_utils.duration_seconds(dst),
                "source_meta": source_meta,
            }
            self.seen_hashes.add(h)
            self.seen_paths.add(rel)
            self.w.write(rec)  # ghi manifest xong mới xóa raw -> kill giữa chừng chỉ rò file, không mất
            if self.cleanup:
                try:
                    os.remove(src)  # giữ sidecar
                except OSError:
                    pass
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(files)}")

    def close(self) -> None:
        self.w.close()


def run(cfg: dict, workdir: str, limit: int | None = None) -> str:
    worker = Worker(cfg, workdir)
    files = worker.scan(include_done=True)
    if limit is not None:
        files = files[:limit]
    print(f"[{STAGE}] {len(files)} file trong {worker.raw_dir}")
    worker.process(files)
    worker.close()
    mpath = manifest.manifest_path(workdir, STAGE)
    print(f"[{STAGE}] xong: {len(manifest.read_records(mpath))} file trong manifest")
    return mpath
